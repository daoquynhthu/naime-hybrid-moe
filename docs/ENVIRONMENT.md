# Environment

Date: 2026-05-17

## Current Environment

```text
python 3.12.10 (venv at .venv312)
torch 2.11.0+cu128
CUDA 12.8
triton-windows 3.6.0
```

FlashAttention is available through PyTorch's built-in `scaled_dot_product_attention`.
`torch.compile` works with Triton JIT (verified: 2.4x throughput on 67M V5 model).

## Workspace Layout

Machine-specific paths live in `configs/workspace.local.json`, which is ignored
by git. Copy `configs/workspace.example.json` and fill in your own local data
root, run root, HuggingFace cache, and optional remote paths.

## GPU

```
NVIDIA GeForce RTX 5060 Laptop GPU (8 GiB VRAM)
auto-batch typical: 7-9 for 67M model with ctx512
```

## Migrating to Another Machine

When copying this project directory to a new machine, run the setup script once:

```powershell
.\scripts\setup_env.ps1
```

This script:
1. Creates `.venv312` with the system Python 3.12
2. Installs all dependencies from `requirements.txt`
3. Copies Python C headers and import libs into the venv for Triton JIT
4. Creates `Include/` and `libs/` NTFS junctions at project root (git-ignored)
5. Verifies `torch` import

## External Environment (Optional)

If you want to reuse an external interpreter, set `NAIME_EXTERNAL_PYTHON` and
pass `-UseVoice`:

```powershell
$env:NAIME_EXTERNAL_PYTHON = "<PATH_TO_EXTERNAL_PYTHON>"
.\scripts\run_tests.ps1 -UseVoice
```

## Commands

```powershell
.\scripts\run_smoke.ps1          # Smoke check
.\scripts\run_tests.ps1          # Run tests
.\scripts\train_model.ps1 ...    # Train
.\scripts\infer.ps1 ...          # Inference
```

Add `-CompileModel` for `torch.compile` acceleration (~2.4x throughput).

## Data Preparation

```powershell
# 1B-token FineWeb-Edu corpus (ctx1024, GPT-2 tokenized, HF disk format)
.\scripts\prepare_fineweb_edu_1b.ps1 -Output <LOCAL_FINEWEB_1B>
```

Requires network access to HuggingFace for the first run. Subsequent runs use
the cached raw dataset configured by `HF_HOME` or your workspace config.
