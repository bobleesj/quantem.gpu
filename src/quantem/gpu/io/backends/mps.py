"""Temporary phase-1 MPS IO shim.

The Metal decompressor still lives in the legacy widget kernel tree during
phase 1. Keep this shim explicit so CUDA/CPU IO can move to ``quantem.gpu`` now
without pretending that the MPS implementation has already been folded in.
"""
from __future__ import annotations


def __getattr__(name: str):
    try:
        import quantem.widget.kernels.io.mps as legacy_mps
    except ImportError as exc:
        raise RuntimeError(
            "The MPS IO backend is not folded into quantem.gpu in phase 1. "
            "Install quantem.widget for the temporary legacy MPS shim, or use "
            "backend='cuda'/'cpu'."
        ) from exc
    return getattr(legacy_mps, name)
