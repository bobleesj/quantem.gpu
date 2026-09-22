"""Resample scan positions while retaining detector coordinates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from quantem.gpu.io._read import _read
from quantem.gpu.io.models import FourDSTEMData


def resample_scan(
    data: FourDSTEMData,
    positions: torch.Tensor,
    *,
    mode: str = "bilinear",
    output_dtype: torch.dtype | np.dtype | str | None = None,
    output_device: str | torch.device | None = None,
    output: np.ndarray | None = None,
    verbose: bool = False,
) -> torch.Tensor | np.ndarray:
    """Resample loaded diffraction data at specified scan coordinates.

    Parameters
    ----------
    data : FourDSTEMData
        Acquisition returned by ``quantem.gpu.io.load``. Its representation
        stays resident and caller-owned. Decoding batches are automatic.
    positions : torch.Tensor
        Source ``(row, column)`` coordinates in pixel units, with shape
        ``(output_rows, output_columns, 2)`` on the compute GPU. Coordinates
        outside the scan use the nearest border value.
    mode : str, optional
        Interpolation: ``"bilinear"``, ``"bicubic"``, or ``"nearest"``.
    output_dtype : torch.dtype, numpy.dtype, str, optional
        Default float32. ``"same"`` retains the working input dtype. Integer
        outputs are rounded and clipped to their representable range.
    output_device : str or torch.device, optional
        Destination device. Defaults to the coordinate tensor's device.
    output : numpy.ndarray, optional
        Preallocated output, including a memory-mapped array.
    verbose : bool, optional
        Show resampling progress.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Corrected data in scan-row, scan-column, detector-row, detector-column
        order, or the supplied output array.

    Examples
    --------
    >>> corrected = resample_scan(loaded, source_positions)
    """
    import torch
    from torch.nn import functional as F
    from tqdm.auto import tqdm

    with torch.inference_mode():
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape (output_rows, output_columns, 2).")
        if positions.device.type not in {"cuda", "mps"}:
            raise ValueError("Place positions on the compute GPU before resampling.")
        if mode not in {"bilinear", "bicubic", "nearest"}:
            raise ValueError("mode must be 'bilinear', 'bicubic', or 'nearest'.")
        scan_rows, scan_columns, *detector_shape = data.shape
        output_shape = (*positions.shape[:2], *detector_shape)
        if output is not None and tuple(output.shape) != output_shape:
            raise ValueError(f"output must have shape {output_shape}; got {output.shape}.")
        dtype = torch.float32
        if output_dtype == "same":
            dtype = torch.from_numpy(np.empty(0, dtype=data.dtype)).dtype
        elif output_dtype is not None:
            dtype = (output_dtype if isinstance(output_dtype, torch.dtype)
                     else torch.from_numpy(np.empty(0, dtype=output_dtype)).dtype)
        device = positions.device
        target = torch.device(output_device) if output_device is not None else device
        positions = positions.to(torch.float32)
        grid = torch.stack((
            2 * positions[..., 1] / max(scan_columns - 1, 1) - 1,
            2 * positions[..., 0] / max(scan_rows - 1, 1) - 1,
        ), dim=-1)[None]
        result = (output if output is not None else
                  torch.empty(output_shape, dtype=dtype, device=target))
        pixels = math.prod(detector_shape)
        result_flat = result.reshape(*positions.shape[:2], pixels)
        # Bound decoded input, interpolation scratch, and corrected working tensors.
        area = max(scan_rows * scan_columns, math.prod(positions.shape[:2]))
        batch = min(pixels, max(1, (32 << 20) // (area * 4 * 4)))
        for start in tqdm(range(0, pixels, batch), disable=not verbose,
                          desc="Resampling scan"):
            stop = min(start + batch, pixels)
            values = _read(data, pixel_range=(start, stop))
            values = values.permute(2, 0, 1).contiguous().to(device, torch.float32)
            corrected = F.grid_sample(
                values[None], grid, mode=mode, padding_mode="border", align_corners=True,
            )[0].permute(1, 2, 0)
            if not dtype.is_floating_point:
                bounds = torch.iinfo(dtype)
                corrected = corrected.round().clamp_(bounds.min, bounds.max)
            if output is not None:
                result_flat[..., start:stop] = corrected.cpu().numpy().astype(output.dtype)
            else:
                result_flat[..., start:stop] = corrected.to(device=target, dtype=dtype)
            del values, corrected
        return result
