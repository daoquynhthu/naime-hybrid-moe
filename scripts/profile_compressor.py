import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from naime_hybrid.modules.semantic_compressor import SemanticCompressor


def profile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    d_model = 384
    seq_len = 512
    batch = 9
    z_dim = 96
    stride = 16
    window = 24
    mid_stride = 32
    mid_window = 64

    compressor = (
        SemanticCompressor(
            d_model=d_model,
            z_dim=z_dim,
            stride=stride,
            window=window,
            target_sparsity=0.55,
            semantic_scales="local_mid_global",
            mid_stride=mid_stride,
            mid_window=mid_window,
            use_global_semantic=True,
            semantic_fusion="concat",
            semantic_pred_horizon=1,
        )
        .to(device)
        .train()
    )

    x = torch.randn(batch, seq_len, d_model, device=device)
    mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)

    # warmup
    for _ in range(10):
        _ = compressor(x, mask)

    torch.cuda.synchronize()

    n_runs = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_runs):
        _ = compressor(x, mask)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    per_run_ms = total_ms / n_runs
    print(f"total {n_runs} runs: {total_ms:.2f} ms  per run: {per_run_ms:.2f} ms")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        _ = compressor(x, mask)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    profile()
