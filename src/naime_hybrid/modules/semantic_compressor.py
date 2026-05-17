import torch
import torch.nn.functional as F
from torch import nn

from .gate import GumbelBlockGate
from .norm import RMSNorm


class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(x)
        scores = self.score(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=1).type_as(x)
        return torch.sum(weights * x, dim=1)


class SemanticCompressor(nn.Module):
    """NAIME-style read-window / write-stride semantic compressor."""

    def __init__(
        self,
        d_model: int,
        z_dim: int,
        stride: int,
        window: int,
        target_sparsity: float,
        gate_eval_mode: str = "prob",
        logvar_clip: float = 10.0,
        semantic_scales: str = "local",
        mid_stride: int = 32,
        mid_window: int = 64,
        use_global_semantic: bool = False,
        semantic_fusion: str = "local",
        semantic_pred_horizon: int = 0,
        downstream_deterministic: bool = False,
        downstream_detach_latent: bool = False,
    ):
        super().__init__()
        if window < stride:
            raise ValueError("window should be >= stride for read-window/write-stride compression")

        self.d_model = d_model
        self.z_dim = z_dim
        self.stride = stride
        self.window = window
        self.gate_eval_mode = gate_eval_mode
        self.logvar_clip = logvar_clip
        self.semantic_scales = semantic_scales
        self.mid_stride = mid_stride
        self.mid_window = mid_window
        self.use_global_semantic = use_global_semantic
        self.semantic_fusion = semantic_fusion
        self.semantic_pred_horizon = semantic_pred_horizon
        self.downstream_deterministic = downstream_deterministic
        self.downstream_detach_latent = downstream_detach_latent

        if semantic_scales not in {"local", "local_mid", "local_mid_global"}:
            raise ValueError("semantic_scales must be local, local_mid, or local_mid_global")
        if semantic_fusion not in {"local", "gated_sum", "concat"}:
            raise ValueError("semantic_fusion must be local, gated_sum, or concat")
        if mid_window < mid_stride:
            raise ValueError("mid_window should be >= mid_stride")
        if semantic_scales in {"local_mid", "local_mid_global"}:
            if mid_stride % stride != 0 or mid_window % stride != 0:
                raise ValueError("mid_stride and mid_window must be multiples of stride when using mid semantic")

        self.pool = AttentionPool(d_model)
        self.gate = GumbelBlockGate(d_model, target_sparsity)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, z_dim * 2),
        )
        self.state_to_model = nn.Linear(z_dim, d_model)
        self.mid_pool = AttentionPool(d_model)
        self.mid_to_model = nn.Linear(d_model, d_model)
        self.global_pool = AttentionPool(d_model)
        self.global_to_model = nn.Linear(d_model, d_model)
        self.fusion_gate = nn.Linear(d_model * 3, 3)
        self.fusion_concat = nn.Linear(d_model * 3, d_model)
        self.semantic_predictor = nn.Linear(z_dim, d_model)

    def _window_pool(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        stride: int,
        window: int,
        pool: AttentionPool,
    ) -> torch.Tensor:
        batch, seq_len, dim = hidden_states.shape
        block_count = (seq_len + stride - 1) // stride
        padded_seq_len = (block_count - 1) * stride + window
        pad = padded_seq_len - seq_len
        if pad > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)

        # Materialize all sliding windows at once so pooling stays inside a
        # single batched path instead of launching one Python loop per block.
        windows = hidden_states.unfold(1, window, stride).permute(0, 1, 3, 2).contiguous()
        window_mask = mask.unfold(1, window, stride).contiguous()
        pooled = pool(
            windows.reshape(batch * block_count, window, dim),
            window_mask.reshape(batch * block_count, window),
        )
        return pooled.view(batch, block_count, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        batch, seq_len, dim = hidden_states.shape
        if dim != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {dim}")

        if attention_mask is None:
            mask = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden_states.device)
        else:
            mask = attention_mask.to(torch.bool)

        block_summary = self._window_pool(hidden_states, mask, self.stride, self.window, self.pool)
        block_count = block_summary.size(1)
        flat_summary = block_summary.reshape(-1, dim)

        params = self.encoder(flat_summary)
        mu, logvar = params.chunk(2, dim=-1)
        logvar = logvar.clamp(-self.logvar_clip, self.logvar_clip)
        std = torch.exp(0.5 * logvar.float()).type_as(mu)
        if self.training:
            z_flat = mu + std * torch.randn_like(std)
        else:
            z_flat = mu
        z = z_flat.view(batch, block_count, self.z_dim)

        alpha, logits, prob, clean_prob = self.gate(flat_summary, tau=tau, eval_mode=self.gate_eval_mode)
        alpha = alpha.view(batch, block_count)
        gate_logits = logits.view(batch, block_count)
        gate_prob = prob.view(batch, block_count)
        gate_clean_prob = clean_prob.view(batch, block_count)

        mu_view = mu.view(batch, block_count, self.z_dim)
        z_downstream = z.detach() if self.downstream_detach_latent else z
        mu_downstream = mu_view.detach() if self.downstream_detach_latent else mu_view
        local_semantic = self.state_to_model(z_downstream)
        local_semantic_mu = self.state_to_model(mu_downstream)
        semantic_model = local_semantic
        semantic_model_mu = local_semantic_mu
        mid_semantic = torch.zeros_like(local_semantic)
        global_semantic = torch.zeros_like(local_semantic)
        fusion_weights = torch.zeros(batch, block_count, 3, device=hidden_states.device, dtype=hidden_states.dtype)

        if self.semantic_scales in {"local_mid", "local_mid_global"}:
            mid_stride_blocks = self.mid_stride // self.stride
            mid_window_blocks = self.mid_window // self.stride
            block_mask = torch.ones(batch, block_count, dtype=torch.bool, device=hidden_states.device)
            mid_summary = self._window_pool(
                block_summary, block_mask, mid_stride_blocks, mid_window_blocks, self.mid_pool
            )
            mid_positions = (torch.arange(block_count, device=hidden_states.device) + 0.5) / mid_stride_blocks
            mid_idx_lo = mid_positions.floor().long().clamp(0, mid_summary.size(1) - 1)
            mid_idx_hi = (mid_idx_lo + 1).clamp(0, mid_summary.size(1) - 1)
            t = (mid_positions - mid_idx_lo.float()).unsqueeze(-1).clamp(0, 1)
            mid_semantic = (1 - t) * self.mid_to_model(mid_summary[:, mid_idx_lo, :]) + t * self.mid_to_model(
                mid_summary[:, mid_idx_hi, :]
            )

        if self.semantic_scales == "local_mid_global" or self.use_global_semantic:
            global_summary = self.global_pool(hidden_states, mask)
            global_semantic = self.global_to_model(global_summary).unsqueeze(1).expand(-1, block_count, -1)

        if self.semantic_fusion == "gated_sum" and self.semantic_scales != "local":
            fusion_input = torch.cat([local_semantic, mid_semantic, global_semantic], dim=-1)
            fusion_weights = torch.softmax(self.fusion_gate(fusion_input).float(), dim=-1).type_as(local_semantic)
            semantic_model = (
                fusion_weights[..., 0:1] * local_semantic
                + fusion_weights[..., 1:2] * mid_semantic
                + fusion_weights[..., 2:3] * global_semantic
            )
            semantic_model_mu = (
                fusion_weights[..., 0:1] * local_semantic_mu
                + fusion_weights[..., 1:2] * mid_semantic
                + fusion_weights[..., 2:3] * global_semantic
            )
        elif self.semantic_fusion == "concat" and self.semantic_scales != "local":
            semantic_model = self.fusion_concat(torch.cat([local_semantic, mid_semantic, global_semantic], dim=-1))
            semantic_model_mu = self.fusion_concat(
                torch.cat([local_semantic_mu, mid_semantic, global_semantic], dim=-1)
            )

        token_block_idx = torch.arange(seq_len, device=hidden_states.device) // self.stride
        token_semantic = semantic_model[:, token_block_idx, :]
        token_semantic_mu = semantic_model_mu[:, token_block_idx, :]
        token_semantic_downstream = token_semantic_mu if self.downstream_deterministic else token_semantic
        token_alpha = alpha[:, token_block_idx].unsqueeze(-1).type_as(token_semantic)
        token_gate_prob = gate_prob[:, token_block_idx].unsqueeze(-1).type_as(token_semantic)
        token_gate_clean_prob = gate_clean_prob[:, token_block_idx].unsqueeze(-1).type_as(token_semantic)

        kl = -0.5 * (1.0 + logvar - mu.pow(2) - torch.exp(logvar.float()).type_as(logvar))
        kl = kl.sum(dim=-1).view(batch, block_count)
        semantic_pred_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.semantic_pred_horizon > 0 and block_count > self.semantic_pred_horizon:
            pred_source = mu_view if self.downstream_deterministic else z
            pred = self.semantic_predictor(pred_source[:, : -self.semantic_pred_horizon, :]).float()
            target = block_summary[:, self.semantic_pred_horizon :, :].detach().float()
            semantic_pred_loss = F.mse_loss(pred, target)

        return {
            "z": z,
            "mu": mu_view,
            "logvar": logvar.view(batch, block_count, self.z_dim),
            "kl": kl,
            "block_summary": block_summary,
            "alpha": alpha,
            "gate_logits": gate_logits,
            "gate_prob": gate_prob,
            "gate_clean_prob": gate_clean_prob,
            "token_semantic": token_semantic,
            "token_semantic_mu": token_semantic_mu,
            "token_semantic_downstream": token_semantic_downstream,
            "token_alpha": token_alpha,
            "token_gate_prob": token_gate_prob,
            "token_gate_clean_prob": token_gate_clean_prob,
            "local_semantic": local_semantic,
            "local_semantic_mu": local_semantic_mu,
            "mid_semantic": mid_semantic,
            "global_semantic": global_semantic,
            "fusion_weights": fusion_weights,
            "semantic_pred_loss": semantic_pred_loss,
        }
