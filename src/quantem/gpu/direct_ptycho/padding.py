"""Mean-pattern padding of a scan onto the square grid the direct ptychography Fourier transforms take.

Direct ptychography transforms every bright-field image over the scan, and the hand-written CUDA and Metal FFT kernels take square
scans of ``SCAN_SIDES`` positions. CUDA runs only those sides; Metal also runs any other square scan of at most 1024
positions through MLX. A scan neither backend runs as measured is centred on the next supported square and the
synthetic positions take the scan's mean diffraction pattern, which keeps the detector statistics (probe disk, dark
field) of a real position. Every measured position is kept and the padding is printed, since the phase then covers
positions that were never measured. A scan larger than 1024 is refused: fitting it would mean cropping measured
positions.
"""

import numpy as np
import torch

# The square scan sides the CUDA and Metal FFT kernels are written for.
SCAN_SIDES = (128, 256, 512, 1024)


def padded_side(scan_shape: tuple[int, int], *, any_square: bool) -> int | None:
    """The square side a ``(rows, cols)`` scan is padded to, or None when the backend runs it as measured.

    ``any_square`` is True for Metal, whose MLX path runs any square scan up to 1024, and False for CUDA, whose
    kernels take only ``SCAN_SIDES``. The side is the smallest of ``SCAN_SIDES`` that holds both scan dimensions.
    """
    rows, cols = (int(size) for size in scan_shape)
    if max(rows, cols) > SCAN_SIDES[-1]:
        raise ValueError(
            f"Direct ptychography reconstructs scans of at most 1024 x 1024 positions; got {rows} x {cols}. Select a region with at "
            "most 1024 positions per side: direct ptychography never crops a scan."
        )
    if rows == cols and (any_square or rows in SCAN_SIDES):
        return None
    return next(side for side in SCAN_SIDES if side >= max(rows, cols))


def announce_padding(scan_shape: tuple[int, int], side: int) -> None:
    """Print the one line that says a scan was padded, by how much, and how to avoid it.

    The padded fraction is ``1 - rows * cols / side**2``: the share of the reconstructed positions that hold the
    mean pattern instead of a measurement.
    """
    rows, cols = scan_shape
    fraction = 1.0 - rows * cols / (side * side)
    sides = ", ".join(str(size) for size in SCAN_SIDES)
    print(
        f"Direct ptycho: {rows} x {cols} scan padded to {side} x {side} with its mean pattern ({fraction:.0%} of positions are "
        f"padding); crop to a supported side ({sides}) to fit measured positions only."
    )


def pad_scan(data, *, any_square: bool):
    """Centre a dense 4D scan on its supported square grid, filling the new positions with the mean pattern.

    ``data`` is a ``(scan_row, scan_col, det_row, det_col)`` Torch tensor or NumPy array in its native dtype; the
    measured values are copied unchanged into the centre of the padded grid. The detector total accumulates in int64
    for integer counts (exact) and in float64 for floating intensities, 16 scan rows at a time so the transient stays
    a 16-row slab. Metal has no float64, so floating intensities on MPS take ``detector.mean``'s pattern instead. Data that
    needs no padding, and frame sources other than a dense 4D array, are returned unchanged.
    """
    if not isinstance(data, (np.ndarray, torch.Tensor)) or data.ndim != 4:
        return data
    rows, cols = (int(size) for size in data.shape[:2])
    side = padded_side((rows, cols), any_square=any_square)
    if side is None:
        return data
    counts = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
    if counts.is_floating_point() and counts.device.type == "mps":
        from quantem.gpu.detector import mean

        pattern = torch.as_tensor(mean(counts), dtype=counts.dtype, device=counts.device)
    else:
        accumulator = torch.float64 if counts.is_floating_point() else torch.int64
        total = torch.zeros(counts.shape[2:], dtype=accumulator, device=counts.device)
        for start in range(0, rows, 16):
            total += counts[start:start + 16].to(accumulator).sum(dim=(0, 1))
        # the exact total divided in float64 on the host (Metal has no float64; only this pattern crosses), then
        # cast to the counts' dtype, truncating integer means so a padded position holds counts as a measured one
        pattern = torch.from_numpy(total.cpu().numpy() / (rows * cols)).to(counts.dtype).to(counts.device)
    padded = pattern.expand(side, side, *counts.shape[2:]).clone()
    top, left = (side - rows) // 2, (side - cols) // 2
    padded[top:top + rows, left:left + cols] = counts
    announce_padding((rows, cols), side)
    return padded.numpy() if isinstance(data, np.ndarray) else padded
