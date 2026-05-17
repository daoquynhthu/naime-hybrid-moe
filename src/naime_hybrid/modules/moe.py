import torch
import torch.nn.functional as F
from torch import nn


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SemanticMoERouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        semantic_dim: int,
        n_experts: int,
        top_k: int,
        use_semantic_router: bool = True,
        semantic_router_mode: str = "concat",
        semantic_router_prior_scale: float = 1.0,
        semantic_router_prior_clip: float = 0.0,
        router_jitter: float = 0.0,
        use_prior_gate: bool = False,
    ):
        super().__init__()
        if top_k > n_experts:
            raise ValueError("top_k cannot exceed n_experts")

        self.n_experts = n_experts
        self.top_k = top_k
        self.use_semantic_router = use_semantic_router
        self.semantic_router_mode = semantic_router_mode
        self.semantic_router_prior_scale = semantic_router_prior_scale
        self.semantic_router_prior_clip = semantic_router_prior_clip
        self.router_jitter = router_jitter
        self.use_prior_gate = use_prior_gate
        if semantic_router_mode not in {"concat", "prior", "hybrid"}:
            raise ValueError("semantic_router_mode must be one of: concat, prior, hybrid")

        if not use_semantic_router:
            self.proj = nn.Linear(d_model, n_experts, bias=False)
            self.semantic_prior = None
            self.prior_gate_proj = None
        elif semantic_router_mode == "concat":
            self.proj = nn.Linear(d_model + semantic_dim, n_experts, bias=False)
            self.semantic_prior = None
            self.prior_gate_proj = None
        elif semantic_router_mode == "prior":
            self.proj = nn.Linear(d_model, n_experts, bias=False)
            self.semantic_prior = nn.Linear(semantic_dim, n_experts, bias=False)
            self.prior_gate_proj = nn.Linear(semantic_dim, 1) if use_prior_gate else None
        else:
            self.proj = nn.Linear(d_model + semantic_dim, n_experts, bias=False)
            self.semantic_prior = nn.Linear(semantic_dim, n_experts, bias=False)
            self.prior_gate_proj = nn.Linear(semantic_dim, 1) if use_prior_gate else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        semantic_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.use_semantic_router:
            if semantic_states is None:
                raise ValueError("semantic_states are required when use_semantic_router=True")
            if self.semantic_router_mode in {"concat", "hybrid"}:
                router_input = torch.cat([hidden_states, semantic_states], dim=-1)
            else:
                router_input = hidden_states
        else:
            router_input = hidden_states

        if self.training and self.router_jitter > 0:
            noise = torch.empty_like(router_input).uniform_(1.0 - self.router_jitter, 1.0 + self.router_jitter)
            router_input = router_input * noise

        logits = self.proj(router_input)
        semantic_bias = None
        if self.use_semantic_router and self.semantic_prior is not None:
            semantic_bias = self.semantic_prior(semantic_states)
            if self.semantic_router_prior_clip > 0:
                semantic_bias = semantic_bias.clamp(
                    min=-self.semantic_router_prior_clip,
                    max=self.semantic_router_prior_clip,
                )
            if self.use_prior_gate and self.prior_gate_proj is not None:
                prior_gate = torch.sigmoid(self.prior_gate_proj(semantic_states))
                logits = logits + prior_gate * self.semantic_router_prior_scale * semantic_bias
            else:
                logits = logits + self.semantic_router_prior_scale * semantic_bias
        topk_logits, topk_indices = torch.topk(logits.float(), self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1).type_as(hidden_states)
        probs = torch.softmax(logits.float(), dim=-1).type_as(hidden_states)

        output = {
            "logits": logits,
            "probs": probs,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }
        if semantic_bias is not None:
            output["semantic_bias"] = semantic_bias
            output["semantic_prior_probs"] = torch.softmax(semantic_bias.float(), dim=-1).type_as(hidden_states)
        return output


class TopKMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        semantic_dim: int,
        n_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        use_semantic_router: bool = True,
        semantic_router_mode: str = "concat",
        semantic_router_prior_scale: float = 1.0,
        semantic_router_prior_clip: float = 0.0,
        router_jitter: float = 0.0,
        dispatch_mode: str = "sparse",
        use_prior_gate: bool = False,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.dispatch_mode = dispatch_mode
        if dispatch_mode not in {"auto", "dense", "sparse"}:
            raise ValueError("dispatch_mode must be auto, dense, or sparse")
        self.router = SemanticMoERouter(
            d_model=d_model,
            semantic_dim=semantic_dim,
            n_experts=n_experts,
            top_k=top_k,
            use_semantic_router=use_semantic_router,
            semantic_router_mode=semantic_router_mode,
            semantic_router_prior_scale=semantic_router_prior_scale,
            semantic_router_prior_clip=semantic_router_prior_clip,
            router_jitter=router_jitter,
            use_prior_gate=use_prior_gate,
        )
        self.shared_expert = SwiGLUExpert(d_model, expert_hidden_dim)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, expert_hidden_dim) for _ in range(n_experts)])

    def _resolve_dispatch_mode(self, hidden_states: torch.Tensor) -> str:
        if self.dispatch_mode != "auto":
            return self.dispatch_mode
        num_tokens = hidden_states.size(0) * hidden_states.size(1)
        if self.n_experts <= 2:
            return "dense"
        if self.n_experts <= 4 and self.top_k * 2 >= self.n_experts and num_tokens >= 128:
            return "dense"
        if hidden_states.is_cuda and self.n_experts <= 8 and self.top_k * 2 >= self.n_experts and num_tokens >= 256:
            return "dense"
        return "sparse"

    def _dense_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts], dim=-2)
        gather_idx = topk_indices.unsqueeze(-1).expand(*topk_indices.shape, hidden_states.size(-1))
        selected = torch.gather(expert_outputs, dim=-2, index=gather_idx)
        return torch.sum(selected * topk_weights.unsqueeze(-1), dim=-2)

    def _sparse_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, d_model = hidden_states.shape
        num_tokens = batch * seq_len
        flat_states = hidden_states.reshape(num_tokens, d_model)
        flat_expert_ids = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1, 1).type_as(flat_states)
        token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(self.top_k)

        # Group by expert id to avoid scanning all routed edges with a boolean
        # mask per expert. This keeps sparse routing semantics but reduces
        # Python/tensor overhead in the dispatch path.
        perm = torch.argsort(flat_expert_ids, stable=True)
        expert_ids_sorted = flat_expert_ids.index_select(0, perm)
        token_ids_sorted = token_ids.index_select(0, perm)
        weights_sorted = flat_weights.index_select(0, perm)

        expert_input = flat_states.index_select(0, token_ids_sorted)
        routed = torch.zeros_like(flat_states)

        counts = torch.bincount(expert_ids_sorted, minlength=self.n_experts)
        for expert_idx in range(self.n_experts):
            count = int(counts[expert_idx])
            if count == 0:
                continue
            start = int(counts[:expert_idx].sum())
            end = start + count
            out = self.experts[expert_idx](expert_input[start:end]) * weights_sorted[start:end]
            routed.index_add_(0, token_ids_sorted[start:end], out)

        return routed.view(batch, seq_len, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        semantic_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        router_out = self.router(hidden_states, semantic_states)
        topk_indices = router_out["topk_indices"]
        topk_weights = router_out["topk_weights"]
        dispatch_mode = self._resolve_dispatch_mode(hidden_states)
        if dispatch_mode == "dense":
            routed = self._dense_dispatch(hidden_states, topk_indices, topk_weights)
        else:
            routed = self._sparse_dispatch(hidden_states, topk_indices, topk_weights)
        output = self.shared_expert(hidden_states) + routed

        router_probs = router_out["probs"]
        token_load = F.one_hot(topk_indices, num_classes=self.n_experts).float().mean(dim=(0, 1, 2))
        prob_load = router_probs.float().mean(dim=(0, 1))
        load_balance = self.n_experts * torch.sum(token_load * prob_load)
        entropy = -(router_probs.float() * torch.log(router_probs.float().clamp_min(1e-9))).sum(dim=-1).mean()
        semantic_prior_entropy = torch.tensor(0.0, device=hidden_states.device)
        if "semantic_prior_probs" in router_out:
            prior_probs = router_out["semantic_prior_probs"].float()
            semantic_prior_entropy = -(prior_probs * torch.log(prior_probs.clamp_min(1e-9))).sum(dim=-1).mean()

        aux = {
            **router_out,
            "token_load": token_load,
            "prob_load": prob_load,
            "load_balance": load_balance,
            "router_entropy": entropy,
            "semantic_prior_entropy": semantic_prior_entropy,
            "dispatch_dense": torch.tensor(
                1.0 if dispatch_mode == "dense" else 0.0,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
        }
        return output, aux
