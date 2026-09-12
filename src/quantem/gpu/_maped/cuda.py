"""CUDA implementation for bounded MAPED merging from encoded residents."""
import json
import math
from pathlib import Path
import time

import cupy as cp
import torch
import torch.nn.functional as functional

from quantem.gpu._compact.streamed import StreamedCounts
from quantem.gpu._maped._provenance import summary_record
from quantem.gpu.io._hot_pixels import correction_is_applied


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
        or correction_is_applied(source.metadata)
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


def _automatic_region_frames(shape) -> int:
    """Choose a bounded internal scan region from currently free CUDA memory."""
    free, _ = cp.cuda.runtime.memGetInfo()
    detector_pixels = math.prod(shape[2:])
    # Numerator, sampled values, denominator, and encoded output coexist at the
    # write boundary. Leave a fixed reserve for CUDA libraries and alignment data.
    bytes_per_frame = detector_pixels * (4 + 4 + 4 + 2)
    available = max(0, int(free) - 512 * 1024**2)
    frames = max(1, min(1024, available // max(1, bytes_per_frame * 4)))
    scan_columns = int(shape[1])
    if frames >= scan_columns:
        frames = max(scan_columns, frames // scan_columns * scan_columns)
    return int(frames)


def merge_to_scaled_h5(
    sources,
    real_shifts,
    diffraction_shifts,
    output_path,
    *,
    release_sources_before_reopen: bool = False,
    verbose: bool = False,
):
    """Merge aligned residents into a packed globally scaled uint16 archive.

    The sources remain caller-owned. Both MAPED passes reuse the same resident
    buffers; only bounded float32 regions and one encoded output region coexist.

    Parameters
    ----------
    sources
        Exact CUDA ANS or packed 4D-STEM acquisitions with identical shapes.
    real_shifts, diffraction_shifts
        CUDA Torch arrays shaped ``(n_sources, 2)`` in row/column order.
    output_path
        Destination master HDF5 path. External data files are placed beside it.
    release_sources_before_reopen
        Release the encoded inputs after the saved merge is complete and before
        reopening its packed result. Use only when the caller owns the sources
        and no longer needs them. This prevents the packed loader's temporary
        buffers from overlapping the input residents.
    verbose
        Print one completion line with measured precision.

    Returns
    -------
    FourDSTEMData
        Reopened device-resident packed result. ``metadata["precision"]`` holds
        the complete scaled-uint16 error report.
    """
    import h5py
    import numpy as np

    from quantem.gpu.io import load
    from quantem.gpu.io.backends.cuda.precision import (
        encode_measure_scaled_uint16,
    )
    from quantem.gpu.io.save import H5Writer

    sources = list(sources)
    if not sources:
        raise ValueError("Provide at least one aligned resident acquisition.")
    shape = tuple(sources[0].shape)
    if len(shape) != 4 or any(tuple(source.shape) != shape for source in sources):
        raise ValueError("Aligned resident acquisitions must share one 4D shape.")
    for name, shifts in (
        ("real_shifts", real_shifts),
        ("diffraction_shifts", diffraction_shifts),
    ):
        if (
            not torch.is_tensor(shifts)
            or shifts.device.type != "cuda"
            or shifts.device.index is None
            or shifts.dtype != torch.float32
            or tuple(shifts.shape) != (len(sources), 2)
        ):
            raise ValueError(
                f"{name} must be a CUDA float32 tensor shaped "
                f"({len(sources)}, 2) in row/column order."
            )
    requested_device = real_shifts.device.index
    for source in sources:
        resident = source.data
        source_device = getattr(
            resident, "device", getattr(resident, "_device_id", None)
        )
        if source_device != requested_device:
            raise ValueError(
                "Every encoded source and both shift arrays must share one CUDA device."
            )
    # Backend code owns CuPy device state. MAPED callers provide only the Torch
    # device and encoded source contract.
    cp.cuda.Device(requested_device).use()
    output_path = Path(output_path)
    region_frames = _automatic_region_frames(shape)
    started = time.perf_counter()

    low = cp.asarray(cp.inf, dtype=cp.float32)
    high = cp.asarray(-cp.inf, dtype=cp.float32)
    for _, region in _merge_regions(
        sources, real_shifts, diffraction_shifts, region_frames
    ):
        cp.minimum(low, cp.min(region), out=low)
        cp.maximum(high, cp.max(region), out=high)
        del region
    cp.cuda.get_current_stream().synchronize()
    intensity_min, intensity_max = float(low.get()), float(high.get())
    if (
        not np.isfinite(intensity_min)
        or not np.isfinite(intensity_max)
        or intensity_max <= intensity_min
    ):
        raise ValueError("The merged output has no finite intensity range.")
    range_seconds = time.perf_counter() - started
    scale = (intensity_max - intensity_min) / 65535.0
    values = math.prod(shape)
    summaries = summary_record(shape, sources)
    report = {
        "version": 1,
        "storage": "scaled_uint16",
        "source_dtype": "float32",
        "source_shape": list(shape),
        "intensity_min": intensity_min,
        "intensity_max": intensity_max,
        "scale": scale,
        "offset": intensity_min,
        "values": values,
        "scope": "all merged values",
        "range_scope": "complete merged output",
        "measurement": "GPU comparison against merged float32 regions",
        "selection": {"scan_region": None, "detector_region": None},
    }
    metadata = {
        "quantem_precision_v1": json.dumps({**report, "complete": False}),
        "source_dtype": "float32",
        "storage_dtype": "uint16",
        "working_dtype": "float32",
        "working_shape": np.asarray(shape, dtype=np.int64),
        "source_shape": np.asarray(shape, dtype=np.int64),
        "scan_shape": np.asarray(shape[:2], dtype=np.int64),
        "detector_shape": np.asarray(shape[2:], dtype=np.int64),
        "representation": "packed",
        "residency": "device",
        "lossless_exact": False,
        "quantem_maped_summary_v1": json.dumps(summaries),
    }
    writer = H5Writer(
        output_path,
        shape[0] * shape[1],
        shape[2:],
        scan_shape=shape[:2],
        metadata=metadata,
        dtype=np.uint16,
        frames_per_file=32768,
        compression="lz4",
    )
    stats = [
        cp.zeros((), dtype=cp.float64),
        cp.zeros((), dtype=cp.float64),
        cp.zeros((), dtype=cp.uint64),
        cp.zeros((), dtype=cp.uint64),
        cp.zeros((), dtype=cp.uint64),
    ]
    encode_events = []
    write_started = time.perf_counter()
    try:
        for _, region in _merge_regions(
            sources, real_shifts, diffraction_shifts, region_frames
        ):
            encode_begin, encode_end = cp.cuda.Event(), cp.cuda.Event()
            encode_begin.record()
            encoded = encode_measure_scaled_uint16(
                region,
                {"scale": scale, "offset": intensity_min},
                stats,
            )
            encode_end.record()
            encode_events.append((encode_begin, encode_end))
            writer.write(encoded)
            del region, encoded
        writer.close(wait=True)
    except BaseException:
        writer.abort()
        raise
    cp.cuda.get_current_stream().synchronize()
    write_seconds = time.perf_counter() - write_started
    encode_seconds = sum(
        cp.cuda.get_elapsed_time(begin, end) for begin, end in encode_events
    ) / 1000.0
    report.update(
        rmse=float(cp.sqrt(stats[0] / values).get()),
        max_abs_error=float(stats[1].get()),
        positive_to_zero=int(stats[2].get()),
        changed=int(stats[3].get()),
        overflow=int(stats[4].get()),
        clipped=int(stats[4].get()),
        complete=True,
    )
    source_representations = {
        str(source.metadata.get("representation", "unknown")) for source in sources
    }
    merge_record = {
        "version": 1,
        "backend": "cuda",
        "source_representation": (
            source_representations.pop()
            if len(source_representations) == 1
            else "mixed"
        ),
        "region_frames": region_frames,
        "range_seconds": range_seconds,
        "gpu_encode_seconds": encode_seconds,
        "merge_encode_write_seconds": write_seconds,
        "released_sources_before_reopen": bool(release_sources_before_reopen),
        "real_space_shifts_row_column": real_shifts.detach().cpu().tolist(),
        "diffraction_shifts_row_column": diffraction_shifts.detach().cpu().tolist(),
    }
    with h5py.File(output_path, "r+") as handle:
        handle.attrs["quantem_precision_v1"] = json.dumps(report)
        handle.attrs["quantem_maped_merge_v1"] = json.dumps(merge_record)
        handle.attrs["quantem_maped_summary_v1"] = json.dumps(summaries)
    if verbose:
        print(
            f"Merged {len(sources)} resident tilts to {output_path} "
            f"as scaled uint16 (RMSE {report['rmse']:.4g}, "
            f"max error {report['max_abs_error']:.4g})."
        )
    if release_sources_before_reopen:
        for source in sources:
            source.close()
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
    reopen_started = time.perf_counter()
    result = load(
        output_path,
        backend="cuda",
        representation="packed",
        verbose=False,
    )
    cp.cuda.get_current_stream().synchronize()
    release_unused = getattr(result.data, "release_unused_blocks", None)
    if callable(release_unused):
        release_unused()
    cp.get_default_memory_pool().free_all_blocks()
    merge_record["reopen_seconds"] = time.perf_counter() - reopen_started
    merge_record["total_seconds"] = time.perf_counter() - started
    with h5py.File(output_path, "r+") as handle:
        handle.attrs["quantem_maped_merge_v1"] = json.dumps(merge_record)
    result.metadata["maped_merge"] = merge_record
    result.metadata["maped_summary"] = summaries
    return result
