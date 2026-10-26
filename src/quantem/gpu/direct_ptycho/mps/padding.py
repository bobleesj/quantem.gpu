"""Retain rectangular scan positions on the fixed-size Metal FFT grid."""

import numpy as np
import torch

from quantem.gpu.direct_ptycho.mps.kernels import MPS_FFT_CONFIGS


def pad_scan(data: object) -> object:
    """Center dense scans in a supported grid using the mean detector pattern.

    Measured values retain their dtype and are copied unchanged. Integer means
    are truncated to the input dtype, matching the CUDA SSB padding convention.
    Only the small detector sum crosses to the host for an exact division.

    Parameters
    ----------
    data : object
        Dense 4D NumPy array or Torch MPS tensor. Other supported frame sources
        pass through unchanged.

    Returns
    -------
    object
        The original supported square source, or an array with identical dtype
        and measured values centered in the next supported square scan.
    """
    if not isinstance(data, (np.ndarray, torch.Tensor)) or data.ndim != 4:
        return data
    rows, cols = data.shape[:2]
    # Square scans already work through the generic MLX FFT path, including
    # small inputs. Preserve their measured extent and exact BF exports.
    if rows == cols and rows <= max(MPS_FFT_CONFIGS):
        return data
    target = next((n for n in sorted(MPS_FFT_CONFIGS) if n >= max(rows, cols)), None)
    if target is None:
        raise ValueError(
            f"MPS SSB supports scans up to 1024x1024; got {rows}x{cols}. "
            "Select a region with at most 1024 positions per side."
        )
    top, left = (target - rows) // 2, (target - cols) // 2
    shape = (target, target, *data.shape[2:])
    if isinstance(data, torch.Tensor):
        if data.is_floating_point():
            from quantem.gpu.detector import mean

            pattern = torch.as_tensor(mean(data), device=data.device, dtype=data.dtype)
        else:
            total = torch.zeros(data.shape[2:], dtype=torch.int64, device=data.device)
            for start in range(0, rows, 16):
                total += data[start:start + 16].to(torch.int64).sum(dim=(0, 1))
            pattern = torch.as_tensor(
                total.cpu().numpy().astype(np.float64) / (rows * cols),
                dtype=data.dtype, device=data.device,
            )
        padded = pattern.expand(shape).clone()
    else:
        pattern = data.mean(axis=(0, 1), dtype=np.float64).astype(data.dtype)
        padded = np.broadcast_to(pattern, shape).copy()
    padded[top:top + rows, left:left + cols] = data
    print(
        f"MPS SSB: padded the {rows}x{cols} scan to {target}x{target} with its "
        "mean pattern for the Metal FFT. Every measured position is retained; "
        "the phase covers the padded grid."
    )
    return padded
