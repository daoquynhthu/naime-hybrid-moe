import math

import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm


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

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[:seq_len].view(1, 1, seq_len, -1).to(dtype=q.dtype, device=q.device)
        sin = self.sin[:seq_len].view(1, 1, seq_len, -1).to(dtype=q.dtype, device=q.device)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


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

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k, seq_len=seq_len)

        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)

        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device).tril()
        attn_mask = causal_mask.view(1, 1, seq_len, seq_len)
        if attention_mask is not None:
            key_mask = attention_mask.to(torch.bool).view(batch, 1, 1, seq_len)
            attn_mask = attn_mask & key_mask

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.o_proj(attn_output)


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

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q_no_rope = q[..., : self.d_no_rope_per_head]
        q_rope = q[..., self.d_no_rope_per_head :]
        q_rope, _ = self.rope(q_rope, torch.zeros_like(q_rope), seq_len=seq_len)
        q = torch.cat([q_no_rope, q_rope], dim=-1)
        q = self.q_norm(q)

        c_kv = self.kv_compress(hidden_states)
        k_no_rope = (
            self.k_no_rope_proj(c_kv).view(batch, seq_len, self.n_kv_heads, self.d_no_rope_per_head).transpose(1, 2)
        )
        k_rope = (
            self.k_rope_proj(hidden_states).view(batch, seq_len, self.n_kv_heads, self.d_rope_per_head).transpose(1, 2)
        )
        _, k_rope = self.rope(torch.zeros_like(k_rope), k_rope, seq_len=seq_len)
        k = torch.cat([k_no_rope, k_rope], dim=-1)
        k = self.k_norm(k)

        v = self.v_proj(c_kv).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)

        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device).tril()
        attn_mask = causal_mask.view(1, 1, seq_len, seq_len)
        if attention_mask is not None:
            key_mask = attention_mask.to(torch.bool).view(batch, 1, 1, seq_len)
            attn_mask = attn_mask & key_mask

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.o_proj(attn_output)
