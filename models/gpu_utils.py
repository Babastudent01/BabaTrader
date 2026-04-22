"""
models/gpu_utils.py
GPU device detection for PyTorch and XGBoost/LightGBM.

Priority order
--------------
1. NVIDIA CUDA      — fastest; works via the stock PyTorch wheel
2. AMD DirectML     — Windows-only; requires ``pip install torch-directml``
                      (ROCm is Linux-only; on Windows AMD = DirectML)
3. Apple MPS        — macOS M-series chips
4. CPU              — universal fallback

XGBoost / LightGBM note
------------------------
XGBoost and LightGBM GPU acceleration only support NVIDIA CUDA.
AMD GPUs are NOT supported by those libraries.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Cache the detected device so detection only runs once ────────────────────
_cached_torch_device = None
_cached_xgb_device:  str | None = None
_cached_lgbm_device: str | None = None


def get_torch_device():
    """
    Return the best available PyTorch compute device.

    Returns a ``torch.device`` for CUDA / MPS / CPU, or the
    ``torch_directml`` device object for AMD on Windows.
    The result is cached after the first call.
    """
    global _cached_torch_device
    if _cached_torch_device is not None:
        return _cached_torch_device

    try:
        import torch

        # ── 1. NVIDIA CUDA ────────────────────────────────────────────────
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(
                "GPU [NVIDIA CUDA]: %s (%.1f GB VRAM) — using CUDA.", name, vram
            )
            _cached_torch_device = torch.device("cuda")
            return _cached_torch_device

        # ── 2. AMD / Intel / any DirectML (Windows) ───────────────────────
        try:
            import torch_directml  # type: ignore[import]
            dev = torch_directml.device(0)
            name = torch_directml.device_name(0)
            # Quick smoke-test — DirectML has limited op support
            _ = torch.ones(2, 2).to(dev) @ torch.ones(2, 2).to(dev)
            logger.info(
                "GPU [AMD/DirectML]: %s — using DirectML. "
                "(Install: pip install torch-directml)", name
            )
            _cached_torch_device = dev
            return _cached_torch_device
        except ImportError:
            logger.debug(
                "torch-directml not installed. "
                "AMD GPU users: pip install torch-directml"
            )
        except Exception as exc:
            logger.warning("DirectML device test failed (%s) — falling back to CPU.", exc)

        # ── 3. Apple MPS ──────────────────────────────────────────────────
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            logger.info("GPU [Apple MPS]: Metal Performance Shaders — using MPS.")
            _cached_torch_device = torch.device("mps")
            return _cached_torch_device

        # ── 4. CPU fallback ───────────────────────────────────────────────
        logger.info(
            "No GPU detected for PyTorch. "
            "  NVIDIA users: pip install torch (default wheel has CUDA)\n"
            "  AMD users (Windows): pip install torch-directml\n"
            "  AMD users (Linux): pip install torch --index-url "
            "https://download.pytorch.org/whl/rocm6.2\n"
            "Using CPU."
        )
        _cached_torch_device = torch.device("cpu")
        return _cached_torch_device

    except ImportError:
        logger.warning("PyTorch not installed — cannot detect GPU.")
        return None


def get_xgb_device() -> str:
    """
    Return the XGBoost ``device`` string: ``'cuda'`` if an NVIDIA GPU is
    available, otherwise ``'cpu'``.

    XGBoost does NOT support AMD GPUs; only NVIDIA CUDA is accelerated.
    """
    global _cached_xgb_device
    if _cached_xgb_device is not None:
        return _cached_xgb_device

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("XGBoost/LightGBM GPU: NVIDIA CUDA (%s).", name)
            _cached_xgb_device = "cuda"
        else:
            _cached_xgb_device = "cpu"
    except ImportError:
        # torch not installed — try xgboost's own CUDA check
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            _cached_xgb_device = "cuda" if result.returncode == 0 else "cpu"
        except Exception:
            _cached_xgb_device = "cpu"

    if _cached_xgb_device == "cpu":
        logger.debug("XGBoost/LightGBM: no NVIDIA GPU detected — using CPU.")
    return _cached_xgb_device


def get_lgbm_device() -> str:
    """Return ``'gpu'`` if NVIDIA CUDA is available, else ``'cpu'``."""
    xgb_dev = get_xgb_device()
    return "gpu" if xgb_dev == "cuda" else "cpu"


def device_summary() -> str:
    """Return a one-line summary of detected GPU support."""
    lines = []
    try:
        import torch
        if torch.cuda.is_available():
            lines.append(f"PyTorch/XGBoost: NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
        else:
            try:
                import torch_directml
                lines.append(f"PyTorch: AMD DirectML ({torch_directml.device_name(0)})")
                lines.append("XGBoost/LightGBM: CPU (AMD GPU not supported by XGBoost)")
            except ImportError:
                lines.append("PyTorch/XGBoost: CPU (no GPU detected)")
    except ImportError:
        lines.append("PyTorch not installed")
    return " | ".join(lines) if lines else "CPU only"
