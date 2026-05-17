# Phase 1 Safe Parameters

Hardware checked on 2026-05-09:

```text
CPU: Intel Core i9-14900HX, 24 cores / 32 logical processors
RAM: ~16 GB
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
VRAM: 8151 MiB
CUDA: available, compute capability 12.0
torch: 2.9.1+cu130
```

## Recommended Faster Config

Use this first:

```text
seq_len: 256
batch_size: 16
d_model: 256
n_layers: 6
n_dense_layers:
  dense: 6
  token_moe / naime_state_moe: 2
n_heads: 4
n_kv_heads: 2
d_ff: 1024
n_experts: 4
top_k: 2
expert_hidden_dim: 512
stride: 8
window: 12
z_dim: 64
```

Measured NAIME-State MoE rough training peak:

```text
d_model=256, layers=6, seq_len=256, batch=16 -> ~3.93 GB allocated peak
```

This uses the GPU much better while leaving room on an 8 GB GPU for fragmentation, dataloader, checkpoints, and Windows display usage.

## Turbo Config

Use this if the machine is otherwise idle:

```text
seq_len: 256
batch_size: 24
d_model: 256
n_layers: 6
n_dense_layers: 2
n_heads: 4
n_kv_heads: 2
d_ff: 1024
n_experts: 4
top_k: 2
expert_hidden_dim: 512
stride: 8
window: 12
z_dim: 64
```

Measured NAIME-State MoE rough training peak:

```text
d_model=256, layers=6, seq_len=256, batch=24 -> ~5.72 GB allocated peak
```

Turbo is faster but has less safety margin.

## Conservative Alternative

Use if the machine is busy or CUDA memory is fragmented:

```text
seq_len: 256
batch_size: 4
d_model: 192
n_layers: 4
n_dense_layers:
  dense: 4
  token_moe / naime_state_moe: 1
n_heads: 4
n_kv_heads: 2
d_ff: 768
n_experts: 4
top_k: 2
expert_hidden_dim: 384
stride: 8
window: 12
z_dim: 48
```

Measured NAIME-State MoE rough forward/backward peak:

```text
d_model=192, layers=4, seq_len=256, batch=4 -> ~1.02 GB allocated peak
```

## Stretch But Still Reasonable

Use after the first three runs are stable:

```text
seq_len: 512
batch_size: 4
d_model: 192
n_layers: 4
n_dense_layers: 1
n_heads: 4
n_kv_heads: 2
d_ff: 768
n_experts: 4
top_k: 2
expert_hidden_dim: 384
stride: 8
window: 12
z_dim: 48
```

Measured NAIME-State MoE rough forward/backward peak:

```text
d_model=192, layers=4, seq_len=512, batch=4 -> ~1.97 GB allocated peak
```

## Recommendation

Start Phase 1 with the faster config (`batch_size=16`). Use turbo only if the machine is idle and the first run is stable.

Rationale:

- enough capacity to compare dense / token_moe / naime_state_moe;
- low enough VRAM pressure for stable checkpointing and long runs;
- small enough for quick iteration on 16 GB system RAM;
- avoids over-optimizing before the mechanism is proven.
