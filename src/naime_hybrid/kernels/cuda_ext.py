from __future__ import annotations

import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils import cpp_extension


def _candidate_cuda_homes() -> list[Path]:
    names = ["NAIME_CUDA_HOME", "CUDA_HOME", "CUDA_PATH", "CUDA_PATH_V13_0", "CUDA_PATH_V12_4"]
    candidates = [Path(os.environ[name]) for name in names if os.environ.get(name)]
    if os.name == "nt":
        root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if root.exists():
            candidates.extend(sorted(root.glob("v*"), reverse=True))
    return candidates


def _nvcc_name() -> str:
    return "nvcc.exe" if os.name == "nt" else "nvcc"


def _prepend_path(path: Path) -> None:
    if not path.exists():
        return
    current = os.environ.get("PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    path_str = str(path)
    if path_str not in parts:
        os.environ["PATH"] = path_str + (os.pathsep + current if current else "")


def _ensure_venv_tools_on_path() -> None:
    # torch.utils.cpp_extension shells out to the ``ninja`` executable. When a
    # remote launcher invokes the venv Python by absolute path, the venv Scripts
    # directory is not guaranteed to be on PATH, so the extension can silently
    # fall back even though ninja is installed in the active environment.
    _prepend_path(Path(sys.executable).resolve().parent)


def _find_cuda_home() -> Path | None:
    for path in _candidate_cuda_homes():
        if (path / "bin" / _nvcc_name()).exists():
            return path
    return None


def _configure_windows_msvc_env() -> None:
    """Make nvcc able to find MSVC without requiring a Developer Prompt.

    PyTorch's extension builder asks distutils for the Visual Studio build
    environment when compiling the C++ translation unit, but nvcc later invokes
    ``cl.exe`` directly for the CUDA translation unit. On Windows, inheriting the
    MSVC PATH/INCLUDE/LIB values here makes the native backend self-contained
    from a normal PowerShell session.
    """

    if sys.platform != "win32":
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vc_env = cpp_extension._get_vc_env("x64")  # type: ignore[attr-defined]
    for key, value in vc_env.items():
        if not value:
            continue
        upper_key = key.upper()
        if upper_key in {"PATH", "INCLUDE", "LIB", "LIBPATH"}:
            os.environ[upper_key] = value
    os.environ.setdefault("VSCMD_ARG_TGT_ARCH", "x64")
    os.environ.setdefault("VSCMD_ARG_HOST_ARCH", "x64")


@lru_cache(maxsize=1)
def load_cuda_extension():
    """Build and load NAIME native CUDA kernels.

    The loader is intentionally runtime-JIT based so the same repository can be
    moved between Windows and Linux without shipping platform-specific binaries.
    Set ``NAIME_CUDA_HOME`` to force a specific CUDA toolkit.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA extension requested but torch.cuda is unavailable")
    _ensure_venv_tools_on_path()
    cuda_home = _find_cuda_home()
    if cuda_home is not None:
        os.environ["CUDA_HOME"] = str(cuda_home)
        cpp_extension.CUDA_HOME = str(cuda_home)
    _configure_windows_msvc_env()
    if sys.platform == "win32":
        # Avoid PyTorch's optional compiler-version probe failing on localized
        # MSVC output before ninja reaches the real build.
        os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")

    source_dir = Path(__file__).resolve().parent / "native"
    sources = [
        str(source_dir / "fused_lm_ce.cpp"),
        str(source_dir / "fused_lm_ce_cuda.cu"),
    ]
    build_root = Path(os.environ.get("NAIME_EXT_BUILD_DIR", Path.home() / ".cache" / "naime_hybrid" / "torch_extensions"))
    build_root.mkdir(parents=True, exist_ok=True)

    cxx_flags = ["/O2"] if sys.platform == "win32" else ["-O3"]
    cuda_flags = ["-O3"]
    if sys.platform != "win32":
        cuda_flags.append("--use_fast_math")

    return cpp_extension.load(
        name="naime_cuda_kernels",
        sources=sources,
        build_directory=str(build_root),
        extra_cflags=cxx_flags,
        extra_cuda_cflags=cuda_flags,
        with_cuda=True,
        verbose=bool(int(os.environ.get("NAIME_EXT_VERBOSE", "0"))),
    )
