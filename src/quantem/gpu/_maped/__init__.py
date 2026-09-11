"""Experimental bounded encoded MAPED merge; not a public workflow API."""
import math
from pathlib import Path

import cupy as cp
import torch
import torch.nn.functional as functional

from quantem.gpu._compact.streamed import StreamedCounts


def _weights(shape, real_shifts, diffraction_shifts):
    """Reproduce existing MAPED one-pixel real blending, without padding."""
    rows, cols, height, width = shape
    device = real_shifts.device
    row = torch.arange(rows, dtype=torch.float32, device=device)
    col = torch.arange(cols, dtype=torch.float32, device=device)
    # The one-pixel Tukey window is exactly zero at either endpoint and one inside.
    window = ((row > 0) & (row < rows - 1))[:, None] * ((col > 0) & (col < cols - 1))[None]
    window = window.to(torch.float32)[None, None]
    real_weights = []
    detector_weights = []
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                            torch.linspace(-1, 1, width, device=device), indexing='ij')
    for real, diffraction in zip(real_shifts, diffraction_shifts):
        r = row[:, None].expand(-1, cols) - real[0]
        c = col[None].expand(rows, -1) - real[1]
        grid = torch.stack((2 * c / (cols - 1) - 1, 2 * r / (rows - 1) - 1), -1)[None]
        real_weights.append(functional.grid_sample(window, grid, align_corners=True)[0, 0])
        grid = torch.stack((xx - 2 * diffraction[1] / width,
                            yy - 2 * diffraction[0] / height), -1)[None]
        detector_weights.append(functional.grid_sample(
            torch.ones((1, 1, height, width), device=device), grid, align_corners=True)[0, 0].clamp(0, 1))
    return torch.stack(real_weights), torch.stack(detector_weights)


def _merge_regions(sources, real_shifts, diffraction_shifts, scans_per_region=1024):
    """Yield GPU float32 regions for fixed bilinear, unpadded MAPED settings.

    All packed owners remain resident. Caller writes each region before advancing.
    The supported scientific settings match the retained default merge benchmark:
    real_space_edge_blend=1, diffraction_edge_blend=0, no padding or scaling.
    """
    shape = sources[0].shape
    if any(source.shape != shape for source in sources):
        raise ValueError('All packed tilts must have matching complete geometry.')
    rows, cols, height, width = shape
    real = torch.from_dlpack(real_shifts)
    diffraction = torch.from_dlpack(diffraction_shifts)
    wi, wd = _weights(shape, real, diffraction)
    edge = 1 - wd.sum(0).clamp(0, 1)
    module = cp.RawModule(
        code=Path(__file__).with_name("regions.cu").read_text(),
        options=("--fmad=false",),
    )
    sample_kernel = module.get_function("sample")
    sample_dense = module.get_function("sample_dense")
    accumulate = module.get_function("accumulate")
    masks = [
        cp.ones((height, width), cp.uint8)
        if source.metadata.get("pixel_mask") is None
        else (cp.asarray(source.metadata["pixel_mask"]) == 0).astype(cp.uint8)
        for source in sources
    ]
    torch.cuda.synchronize()
    real_cp = cp.from_dlpack(real)
    diffraction_cp = cp.from_dlpack(diffraction)
    wi_cp = cp.from_dlpack(wi).reshape(len(sources), -1)
    real_row_shifts = cp.asnumpy(real_cp[:, 0])
    decode_errors = cp.zeros(1, cp.uint32)
    for first in range(0, rows * cols, scans_per_region):
        stop = min(first + scans_per_region, rows * cols)
        count = stop - first
        numerator = cp.zeros((count, height, width), cp.float32)
        sampled = cp.empty_like(numerator)
        launch = (((numerator.size + 255) // 256,), (256,))
        decode_errors.fill(0)
        for i, source in enumerate(sources):
            resident = source.data
            decoded = None
            if isinstance(resident, StreamedCounts):
                first_row = first // cols
                stop_row = (stop - 1) // cols
                shift_row = math.floor(-float(real_row_shifts[i]))
                decoded_first_row = max(0, first_row + shift_row)
                decoded_stop_row = min(rows, stop_row + shift_row + 2)
                if decoded_first_row < decoded_stop_row:
                    decoded = resident.decode_scan_range_device(
                        decoded_first_row * cols,
                        decoded_stop_row * cols,
                        errors=decode_errors,
                    )
                    sample_dense(
                        *launch,
                        (
                            decoded,
                            cp.int32(decoded.dtype.itemsize),
                            masks[i],
                            real_cp[i],
                            sampled,
                            cp.int32(first),
                            cp.int32(count),
                            cp.int32(rows),
                            cp.int32(cols),
                            cp.int32(height * width),
                            cp.int32(decoded_first_row),
                            cp.int32(decoded_stop_row - decoded_first_row),
                        ),
                    )
                else:
                    sampled.fill(0)
            else:
                sample_kernel(
                    *launch,
                    (
                        *resident._arrays,
                        masks[i],
                        real_cp[i],
                        sampled,
                        cp.int32(first),
                        cp.int32(count),
                        cp.int32(rows),
                        cp.int32(cols),
                        cp.int32(height * width),
                        cp.int32(resident.block_frames),
                    ),
                )
            accumulate(
                *launch,
                (
                    sampled,
                    diffraction_cp[i],
                    wi_cp[i, first:stop],
                    numerator,
                    cp.int32(count),
                    cp.int32(height),
                    cp.int32(width),
                ),
            )
            del decoded
        cp.cuda.get_current_stream().synchronize()
        if int(decode_errors.get()[0]):
            raise ValueError(
                "An ANS count stream failed reconstruction during MAPED merging."
            )
        denominator = torch.einsum('ns,nhw->shw', wi.reshape(len(sources), -1)[:, first:stop], wd)
        denominator += edge[None]
        torch.cuda.synchronize()
        denominator_cp = cp.from_dlpack(denominator)
        cp.divide(numerator, denominator_cp, out=numerator)
        numerator[denominator_cp == 0] = 0
        cp.cuda.get_current_stream().synchronize()
        yield first, numerator
        del numerator, sampled, denominator_cp, denominator
