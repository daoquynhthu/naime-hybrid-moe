# Kernel Optimization Pipeline

This workflow keeps CUDA-kernel iteration repeatable and auditable. It is meant
for high-impact paths only: language-model loss, state softmax/matmul/update,
and other profiler-proven hotspots. Marginal kernels should stay in PyTorch or
TorchInductor until the profiler says otherwise.

## Default Command

```powershell
.\scripts\run_kernel_pipeline.ps1
```

The pipeline performs:

1. Local style/syntax checks for the kernel pipeline paths.
2. Code sync to the remote training workspace.
3. Remote CUDA extension build through the kernel unit test.
4. Baseline V7 profiler run.
5. Candidate V7 profiler run.
6. Short real-training smoke test.
7. Summary of throughput and peak-memory deltas.
8. Cleanup of the temporary smoke checkpoint run.

Profiler tables are written to the profile directory by default; the console
prints only the high-signal throughput and memory summary.

## Important Options

```powershell
.\scripts\run_kernel_pipeline.ps1 `
  -Template v7_remote_64m_probe `
  -BatchSize 29 `
  -ProfileSteps 5 `
  -SmokeSteps 50 `
  -BaselineBackend torch `
  -CandidateBackend cuda_ext_ce `
  -UseFusedStateAttention $true
```

Use `-SkipSync` only when the remote code is already known to match the local
workspace. Use `-KeepSmokeRun` only when the smoke checkpoint itself is needed
for debugging; otherwise the script removes it to protect disk space.

The default local checks intentionally cover only the active kernel/profiler
paths. Full-repository style cleanup should be done separately so historical
tooling scripts do not block fast kernel iteration.

## Decision Rule

A kernel candidate is worth keeping only if it improves at least one primary
metric without hurting stability:

- Higher profiled tokens/sec.
- Lower peak memory.
- Passing CUDA kernel unit tests.
- Passing a real training smoke run with no NaN or silent exit.

For V7, `cuda_ext_ce` is now the preferred explicit backend for remote kernel
experiments. The next major target is a true fused `lm_head + cross entropy`
backward path that avoids materializing the full logits gradient where possible.

## MoE Dispatch Sweep

Use this when changing expert count, top-k, or dispatch internals:

```powershell
.\scripts\remote.ps1 cmd "cd <remote_repo>; <remote_python> scripts/benchmark_moe_dispatch.py --experts 4 8 16 32 --top-k 2 --modes dense sparse auto"
```

Current 4090-class findings for V7 64M shapes:

- 4 experts, top-2: optimized sparse is already slightly faster than dense in the isolated MoE benchmark.
- 8/16/32 experts: sparse becomes materially faster and uses far less memory.
- Full-model profiler can overstate sparse overhead during compile warmup; confirm with a real training smoke before changing defaults.
