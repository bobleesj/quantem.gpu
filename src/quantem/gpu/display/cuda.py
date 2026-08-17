"""CUDA implementation of the shared scientific display arithmetic."""

from typing import Literal

import cupy as cp

DisplayScale = Literal["linear", "log"]

_normalize = cp.ElementwiseKernel(
    "float32 value, float32 low, float32 high, int32 log_scale",
    "float32 normalized",
    r"""
    float x = value;
    float lo = low;
    float hi = fmaxf(low, high);
    if (isnan(x)) {
        normalized = 0.0f;
        return;
    }
    if (isinf(x)) {
        normalized = x > 0.0f ? 1.0f : 0.0f;
        return;
    }
    if (log_scale != 0) {
        x = copysignf(log1pf(fabsf(x)), x);
        lo = copysignf(log1pf(fabsf(lo)), lo);
        hi = copysignf(log1pf(fabsf(hi)), hi);
    }
    if (!(hi > lo)) {
        normalized = 0.5f;
    } else {
        float span = hi - lo;
        float t;
        if (isinf(span)) {
            float negative_magnitude = -lo;
            float center;
            if (negative_magnitude <= hi) {
                float ratio = negative_magnitude / hi;
                center = ratio / (1.0f + ratio);
            } else {
                float ratio = hi / negative_magnitude;
                center = 1.0f / (1.0f + ratio);
            }
            t = x >= 0.0f
                ? center + (1.0f - center) * (x / hi)
                : center * (1.0f - x / lo);
        } else {
            t = (x - lo) / span;
        }
        normalized = fminf(1.0f, fmaxf(0.0f, t));
    }
    """,
    "quantem_display_normalize_f32",
)


def normalize(
    values: cp.ndarray,
    low: float,
    high: float,
    scale: DisplayScale = "linear",
) -> cp.ndarray:
    """Normalize a CUDA-resident image with shared float32 semantics."""
    if scale not in {"linear", "log"}:
        raise ValueError("scale must be 'linear' or 'log'")
    return _normalize(
        cp.asarray(values, dtype=cp.float32),
        cp.float32(low),
        cp.float32(high),
        cp.int32(scale == "log"),
    )


def histogram(
    values: cp.ndarray,
    low: float,
    high: float,
    scale: DisplayScale = "linear",
) -> cp.ndarray:
    """Return raw uint32 counts in 256 bins without leaving CUDA."""
    values = cp.asarray(values, dtype=cp.float32)
    finite = values[cp.isfinite(values)]
    indices = cp.minimum(
        cp.floor(normalize(finite, low, high, scale) * cp.float32(256)),
        cp.float32(255),
    ).astype(cp.int32)
    return cp.bincount(indices.reshape(-1), minlength=256).astype(cp.uint32)


def colorize(
    values: cp.ndarray,
    lut: cp.ndarray,
    low: float,
    high: float,
    scale: DisplayScale = "linear",
) -> cp.ndarray:
    """Return exact uint8 RGBA pixels from a CUDA-resident image and LUT."""
    colors = cp.asarray(lut)
    if colors.shape not in {(256, 3), (256, 4)}:
        raise ValueError("lut must have shape (256, 3) or (256, 4)")
    if colors.dtype.kind == "f":
        colors = cp.floor(colors * cp.float32(255) + cp.float32(0.5)).astype(
            cp.uint8
        )
    else:
        colors = colors.astype(cp.uint8, copy=False)
    indices = cp.minimum(
        cp.floor(normalize(values, low, high, scale) * cp.float32(255)),
        cp.float32(255),
    ).astype(cp.int32)
    rgba = cp.empty((*indices.shape, 4), dtype=cp.uint8)
    rgba[..., :3] = colors[indices, :3]
    rgba[..., 3] = colors[indices, 3] if colors.shape[1] == 4 else 255
    return rgba


__all__ = ["colorize", "histogram", "normalize"]
