import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm


@dataclass(frozen=True)
class AttentionKVCache:
    """Committed key/value memory for autoregressive attention.

    The cache stores unrepeated grouped-query keys and values with shape
    ``[batch, n_kv_heads, seq_len, head_dim]``. Callers should treat cached
    entries as immutable committed memory: future tokens may read them, but
    later hidden-state refinement should not rewrite already committed entries.
    """

    key: torch.Tensor
    value: torch.Tensor

    @property
    def seq_len(self) -> int:
        return int(self.key.size(2))

    def validate(self, *, batch_size: int, n_kv_heads: int, head_dim: int) -> None:
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise ValueError("attention KV cache tensors must be 4D [batch, n_kv_heads, seq_len, head_dim]")
        if self.key.shape != self.value.shape:
            raise ValueError(f"attention key/value cache shape mismatch: {self.key.shape} vs {self.value.shape}")
        if self.key.size(0) != batch_size:
            raise ValueError(f"attention KV cache batch mismatch: expected {batch_size}, got {self.key.size(0)}")
        if self.key.size(1) != n_kv_heads:
            raise ValueError(f"attention KV cache head mismatch: expected {n_kv_heads}, got {self.key.size(1)}")
        if self.key.size(3) != head_dim:
            raise ValueError(f"attention KV cache head_dim mismatch: expected {head_dim}, got {self.key.size(3)}")

    def detach(self) -> "AttentionKVCache":
        return AttentionKVCache(key=self.key.detach(), value=self.value.detach())

    def to(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "AttentionKVCache":
        kwargs: dict[str, torch.device | str | torch.dtype] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        if not kwargs:
            return self
        return AttentionKVCache(key=self.key.to(**kwargs), value=self.value.to(**kwargs))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embedding")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end = position_offset + seq_len
        if end > self.cos.size(0):
            raise ValueError(f"rotary position range {end} exceeds max_seq_len={self.cos.size(0)}")
        cos = self.cos[position_offset:end].view(1, 1, seq_len, -1).to(dtype=q.dtype, device=q.device)
        sin = self.sin[position_offset:end].view(1, 1, seq_len, -1).to(dtype=q.dtype, device=q.device)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _effective_position_offset(past_key_value: AttentionKVCache | None, position_offset: int) -> int:
    if past_key_value is None:
        return int(position_offset)
    if position_offset == 0:
        return past_key_value.seq_len
    return int(position_offset)


def _build_cached_attention_mask(
    *,
    batch_size: int,
    query_len: int,
    total_key_len: int,
    past_len: int,
    attention_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    if attention_mask is None and past_len == 0:
        return None

    query_positions = torch.arange(past_len, past_len + query_len, device=device).view(query_len, 1)
    key_positions = torch.arange(total_key_len, device=device).view(1, total_key_len)
    attn_mask = key_positions <= query_positions
    attn_mask = attn_mask.view(1, 1, query_len, total_key_len)

    if attention_mask is not None:
        key_mask = attention_mask.to(device=device, dtype=torch.bool)
        if key_mask.ndim != 2 or key_mask.size(0) != batch_size:
            raise ValueError("attention_mask must have shape [batch, seq_len] for cached attention")
        if key_mask.size(1) == query_len and past_len > 0:
            past_mask = torch.ones(batch_size, past_len, dtype=torch.bool, device=device)
            key_mask = torch.cat([past_mask, key_mask], dim=1)
        elif key_mask.size(1) != total_key_len:
            raise ValueError(
                f"attention_mask length must be current query length {query_len} or total key length {total_key_len}; "
                f"got {key_mask.size(1)}"
            )
        attn_mask = attn_mask & key_mask.view(batch_size, 1, 1, total_key_len)

    return attn_mask


class GQAAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_repeat = n_heads // n_kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_value: AttentionKVCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionKVCache]:
        batch, seq_len, _ = hidden_states.shape
        past_len = 0
        if past_key_value is not None:
            past_key_value.validate(batch_size=batch, n_kv_heads=self.n_kv_heads, head_dim=self.head_dim)
            past_len = past_key_value.seq_len
        effective_offset = _effective_position_offset(past_key_value, position_offset)

        q = self.q_proj(hidden_states).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, seq_len=seq_len, position_offset=effective_offset)

        if past_key_value is not None:
            k_unrepeated = torch.cat([past_key_value.key.to(device=k.device, dtype=k.dtype), k], dim=2)
            v_unrepeated = torch.cat([past_key_value.value.to(device=v.device, dtype=v.dtype), v], dim=2)
        else:
            k_unrepeated = k
            v_unrepeated = v
        total_key_len = k_unrepeated.size(2)

        k_full = k_unrepeated.repeat_interleave(self.kv_repeat, dim=1)
        v_full = v_unrepeated.repeat_interleave(self.kv_repeat, dim=1)

        attn_mask = _build_cached_attention_mask(
            batch_size=batch,
            query_len=seq_len,
            total_key_len=total_key_len,
            past_len=past_len,
            attention_mask=attention_mask,
            device=hidden_states.device,
        )
        if attn_mask is None:
            attn_output = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.o_proj(attn_output)
        if not use_cache:
            return output
        return output, AttentionKVCache(key=k_unrepeated, value=v_unrepeated)


class MLAAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        d_latent: int = 128,
        d_rope_per_head: int = 32,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_repeat = n_heads // n_kv_heads
        self.dropout = dropout
        self.d_latent = d_latent
        self.d_rope_per_head = d_rope_per_head
        self.d_no_rope_per_head = self.head_dim - d_rope_per_head

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.kv_compress = nn.Linear(d_model, d_latent, bias=False)
        self.k_rope_proj = nn.Linear(d_model, n_kv_heads * d_rope_per_head, bias=False)
        self.k_no_rope_proj = nn.Linear(d_latent, n_kv_heads * self.d_no_rope_per_head, bias=False)
        self.v_proj = nn.Linear(d_latent, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(d_rope_per_head, max_seq_len=max_seq_len, theta=rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_value: AttentionKVCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionKVCache]:
        batch, seq_len, _ = hidden_states.shape
        past_len = 0
        if past_key_value is not None:
            past_key_value.validate(batch_size=batch, n_kv_heads=self.n_kv_heads, head_dim=self.head_dim)
            past_len = past_key_value.seq_len
        effective_offset = _effective_position_offset(past_key_value, position_offset)

        q = self.q_proj(hidden_states).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q_no_rope = q[..., : self.d_no_rope_per_head]
        q_rope = q[..., self.d_no_rope_per_head :]
        q_rope, _ = self.rope(
            q_rope,
            torch.zeros_like(q_rope),
            seq_len=seq_len,
            position_offset=effective_offset,
        )
        q = torch.cat([q_no_rope, q_rope], dim=-1)
        q = self.q_norm(q)

        c_kv = self.kv_compress(hidden_states)
        k_no_rope = (
            self.k_no_rope_proj(c_kv).view(batch, seq_len, self.n_kv_heads, self.d_no_rope_per_head).transpose(1, 2)
        )
        k_rope = (
            self.k_rope_proj(hidden_states).view(batch, seq_len, self.n_kv_heads, self.d_rope_per_head).transpose(1, 2)
        )
        _, k_rope = self.rope(
            torch.zeros_like(k_rope),
            k_rope,
            seq_len=seq_len,
            position_offset=effective_offset,
        )
        k = torch.cat([k_no_rope, k_rope], dim=-1)
        k = self.k_norm(k)

        v = self.v_proj(c_kv).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            k_unrepeated = torch.cat([past_key_value.key.to(device=k.device, dtype=k.dtype), k], dim=2)
            v_unrepeated = torch.cat([past_key_value.value.to(device=v.device, dtype=v.dtype), v], dim=2)
        else:
            k_unrepeated = k
            v_unrepeated = v
        total_key_len = k_unrepeated.size(2)

        k_full = k_unrepeated.repeat_interleave(self.kv_repeat, dim=1)
        v_full = v_unrepeated.repeat_interleave(self.kv_repeat, dim=1)

        attn_mask = _build_cached_attention_mask(
            batch_size=batch,
            query_len=seq_len,
            total_key_len=total_key_len,
            past_len=past_len,
            attention_mask=attention_mask,
            device=hidden_states.device,
        )
        if attn_mask is None:
            attn_output = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.o_proj(attn_output)
        if not use_cache:
            return output
        return output, AttentionKVCache(key=k_unrepeated, value=v_unrepeated)
