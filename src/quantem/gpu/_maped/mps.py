"""Bounded Metal MAPED merge for exact ANS residents."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from quantem.gpu._maped._provenance import summary_record
from quantem.gpu.io.backends.mps._streamed import MPSStreamedCounts
from quantem.gpu.io.backends.mps.packed import (
    _allocate_shared,
    _buffer_view,
    _complete,
    _metal_module,
    _release,
)
from quantem.gpu.io.backends.mps.precision import (
    MetalArray,
    _dispatch as precision_dispatch,
    _parameters as precision_parameters,
    encode_measure,
)


@lru_cache(maxsize=1)
def _runtime():
    metal = _metal_module()
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("MPS MAPED needs an available Metal GPU.")
    options = metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(False)
    source = Path(__file__).with_name("regions.msl").read_text()
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None:
        raise RuntimeError(f"MPS MAPED Metal compilation failed: {error}")
    pipelines = {}
    for name in (
        "real_weights",
        "detector_weights",
        "detector_edge",
        "sample_dense",
        "accumulate",
        "normalize",
    ):
        function = library.newFunctionWithName_(f"maped_{name}")
        pipeline, error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if pipeline is None:
            raise RuntimeError(f"MPS MAPED {name} pipeline failed: {error}")
        pipelines[name] = pipeline
    return device, metal, device.newCommandQueue(), pipelines


def _upload(device, metal, values, label):
    values = np.ascontiguousarray(values)
    buffer = _allocate_shared(device, metal, max(1, values.nbytes), label)
    if values.nbytes:
        _buffer_view(buffer, values.nbytes)[:] = memoryview(values).cast("B")
    return buffer


def _dispatch(
    queue,
    metal,
    pipeline,
    buffers,
    byte_values,
    count,
    label,
    *,
    command=None,
):
    owned = command is None
    command = queue.commandBuffer() if owned else command
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for index, value in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(value, 0, index)
    for index, value in byte_values:
        encoder.setBytes_length_atIndex_(value, len(value), index)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(int(count), 1, 1), metal.MTLSizeMake(256, 1, 1)
    )
    encoder.endEncoding()
    if owned:
        _complete(command, label)


def _range(region):
    parameters, floats = precision_parameters(region)
    parameters[14] = parameters[15] = min(region.size, 8192)
    stats = MetalArray((parameters[14], 4), np.float32)
    try:
        precision_dispatch("range", [region, stats], parameters, floats)
        rows = stats.get()
        if np.any(rows[:, 2]):
            raise ValueError("The merged output contains non-finite intensities.")
        if np.any(rows[:, 3]):
            raise ValueError(
                "The merged output contains float32 subnormal intensities."
            )
        return float(rows[:, 0].min()), float(rows[:, 1].max())
    finally:
        stats.release()


def _merge_weights(shape, real_np, diffraction_np):
    rows, cols, height, width = shape
    scans = rows * cols
    pixels = height * width
    source_count = len(real_np)
    device, metal, queue, pipelines = _runtime()
    real_buffer = _upload(
        device, metal, real_np.astype(np.float32), "MAPED real shifts"
    )
    diffraction_buffer = _upload(
        device, metal, diffraction_np.astype(np.float32), "MAPED diffraction shifts"
    )
    real_weights = MetalArray((source_count, scans), np.float32)
    real_sampling = MetalArray((source_count, scans, 4), np.float32)
    detector_weights = MetalArray((source_count, pixels), np.float32)
    detector_indices = MetalArray((source_count, pixels, 4), np.int32)
    detector_sampling = MetalArray((source_count, pixels, 4), np.float32)
    detector_edge = MetalArray((pixels,), np.float32)
    try:
        command = queue.commandBuffer()
        parameters = np.asarray(
            [source_count, scans, rows, cols], np.uint64
        ).tobytes()
        _dispatch(
            queue,
            metal,
            pipelines["real_weights"],
            [real_buffer, real_weights._mtl, real_sampling._mtl],
            [(3, parameters)],
            source_count * scans,
            "MPS MAPED real weights",
            command=command,
        )
        parameters = np.asarray(
            [source_count, pixels, height, width], np.uint64
        ).tobytes()
        _dispatch(
            queue,
            metal,
            pipelines["detector_weights"],
            [
                diffraction_buffer,
                detector_weights._mtl,
                detector_indices._mtl,
                detector_sampling._mtl,
            ],
            [(4, parameters)],
            source_count * pixels,
            "MPS MAPED detector weights",
            command=command,
        )
        _dispatch(
            queue,
            metal,
            pipelines["detector_edge"],
            [detector_weights._mtl, detector_edge._mtl],
            [(2, parameters)],
            pixels,
            "MPS MAPED detector edge",
            command=command,
        )
        _complete(command, "MPS MAPED weights")
        return (
            real_weights,
            real_sampling,
            detector_weights,
            detector_indices,
            detector_sampling,
            detector_edge,
        )
    except BaseException:
        real_weights.release()
        real_sampling.release()
        detector_weights.release()
        detector_indices.release()
        detector_sampling.release()
        detector_edge.release()
        raise
    finally:
        _release(real_buffer)
        _release(diffraction_buffer)


def _merge_regions(sources, real_np, diffraction_np, scans_per_region):
    shape = tuple(sources[0].shape)
    rows, cols, height, width = shape
    pixels = height * width
    device, metal, queue, pipelines = _runtime()
    weights = _merge_weights(shape, real_np, diffraction_np)
    (
        real_weights,
        real_sampling,
        detector_weights,
        detector_indices,
        detector_sampling,
        detector_edge,
    ) = weights
    try:
        for first in range(0, rows * cols, scans_per_region):
            stop = min(first + scans_per_region, rows * cols)
            count = stop - first
            numerator = MetalArray((count, height, width), np.float32)
            sampled = MetalArray((count, height, width), np.float32)
            decoded_ranges = []
            max_decoded = 1
            for real_shift in real_np:
                first_row = first // cols
                stop_row = (stop - 1) // cols
                shift_row = math.floor(-float(real_shift[0]))
                decoded_first_row = max(0, first_row + shift_row)
                decoded_stop_row = min(rows, stop_row + shift_row + 2)
                decoded_ranges.append((decoded_first_row, decoded_stop_row))
                max_decoded = max(
                    max_decoded,
                    max(0, decoded_stop_row - decoded_first_row) * cols,
                )
            decoded = MetalArray((max_decoded, height, width), np.uint16)
            try:
                command = queue.commandBuffer()
                for source_index, source in enumerate(sources):
                    decoded_first_row, decoded_stop_row = decoded_ranges[source_index]
                    if decoded_first_row < decoded_stop_row:
                        source._encode_scan_range_into(
                            command,
                            decoded_first_row * cols,
                            decoded_stop_row * cols,
                            decoded._mtl,
                        )
                    parameters = np.asarray(
                        [
                            first,
                            count,
                            rows,
                            cols,
                            pixels,
                            pixels,
                            source.dtype.itemsize,
                            decoded_first_row,
                            decoded_stop_row - decoded_first_row,
                            source_index,
                            rows * cols,
                        ],
                        np.uint64,
                    ).tobytes()
                    _dispatch(
                        queue,
                        metal,
                        pipelines["sample_dense"],
                        [
                            decoded._mtl,
                            source._valid,
                            real_sampling._mtl,
                            sampled._mtl,
                        ],
                        [(4, parameters)],
                        count * ((pixels + 3) // 4),
                        "MPS MAPED real-space sample",
                        command=command,
                    )
                    parameters = np.asarray(
                        [
                            first,
                            count,
                            rows,
                            cols,
                            height,
                            width,
                            source_index,
                            rows * cols,
                            pixels,
                        ],
                        np.uint64,
                    ).tobytes()
                    _dispatch(
                        queue,
                        metal,
                        pipelines["accumulate"],
                        [
                            sampled._mtl,
                            numerator._mtl,
                            real_weights._mtl,
                            detector_indices._mtl,
                            detector_sampling._mtl,
                        ],
                        [(5, parameters)],
                        count * ((pixels + 3) // 4),
                        "MPS MAPED diffraction sample",
                        command=command,
                    )
                parameters = np.asarray(
                    [
                        first,
                        count,
                        rows,
                        cols,
                        height,
                        width,
                        len(sources),
                        rows * cols,
                        pixels,
                    ],
                    np.uint64,
                ).tobytes()
                _dispatch(
                    queue,
                    metal,
                    pipelines["normalize"],
                    [
                        numerator._mtl,
                        real_weights._mtl,
                        detector_weights._mtl,
                        detector_edge._mtl,
                    ],
                    [(4, parameters)],
                    count * ((pixels + 3) // 4),
                    "MPS MAPED normalize",
                    command=command,
                )
                _complete(command, "MPS MAPED region")
                for source in sources:
                    source._check_errors()
                yield first, numerator
                numerator = None
            finally:
                decoded.release()
                sampled.release()
                if numerator is not None:
                    numerator.release()
    finally:
        for value in weights:
            value.release()


def _automatic_region_frames(shape):
    """Use large row-aligned regions while bounding MAPED temporaries."""
    detector_pixels = math.prod(shape[2:])
    working_bytes_per_frame = detector_pixels * 16
    frames = max(1, min(4096, (5 * 1024**3 // 2) // working_bytes_per_frame))
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
    """Merge exact MPS ANS sources with a bounded Metal working set."""
    from quantem.gpu.io import load
    from quantem.gpu.io._precision import _finish_report
    from quantem.gpu.io.save import H5Writer

    sources = list(sources)
    if not sources:
        raise ValueError("Provide at least one aligned resident acquisition.")
    shape = tuple(sources[0].shape)
    if len(shape) != 4 or any(tuple(source.shape) != shape for source in sources):
        raise ValueError("Aligned resident acquisitions must share one 4D shape.")
    for source in sources:
        if not isinstance(source.data, MPSStreamedCounts):
            raise ValueError(
                "MPS bounded MAPED currently requires exact ANS residents."
            )
    for name, shifts in (
        ("real_shifts", real_shifts),
        ("diffraction_shifts", diffraction_shifts),
    ):
        if (
            not torch.is_tensor(shifts)
            or shifts.device.type != "mps"
            or shifts.dtype != torch.float32
            or tuple(shifts.shape) != (len(sources), 2)
        ):
            raise ValueError(
                f"{name} must be an MPS float32 tensor shaped "
                f"({len(sources)}, 2) in row/column order."
            )
    # Shift vectors are 28 float32 values for seven tilts. They cross only as
    # launch metadata; all detector and scan arithmetic remains on Metal.
    torch.mps.synchronize()
    real_np = real_shifts.detach().cpu().numpy()
    diffraction_np = diffraction_shifts.detach().cpu().numpy()
    output_path = Path(output_path)
    region_frames = _automatic_region_frames(shape)
    started = time.perf_counter()
    low, high = math.inf, -math.inf
    for _, region in _merge_regions(
        [source.data for source in sources], real_np, diffraction_np, region_frames
    ):
        region_low, region_high = _range(region)
        low, high = min(low, region_low), max(high, region_high)
        region.release()
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("The merged output has no finite intensity range.")
    range_seconds = time.perf_counter() - started
    report = {
        "version": 1,
        "storage": "scaled_uint16",
        "source_dtype": "float32",
        "source_shape": list(shape),
        "intensity_min": low,
        "intensity_max": high,
        "scale": (high - low) / 65535.0,
        "offset": low,
        "values": 0,
        "squared_error": 0.0,
        "max_abs_error": 0.0,
        "positive_to_zero": 0,
        "changed": 0,
        "overflow": 0,
        "clipped": 0,
        "scope": "all merged values",
        "range_scope": "complete merged output",
        "measurement": "GPU comparison against merged float32 regions",
        "selection": {"scan_region": None, "detector_region": None},
    }
    summaries = summary_record(shape, len(sources))
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
    write_started = time.perf_counter()
    encode_seconds = 0.0
    try:
        for _, region in _merge_regions(
            [source.data for source in sources], real_np, diffraction_np, region_frames
        ):
            encoded = None
            try:
                encode_started = time.perf_counter()
                encoded = encode_measure(region, report)
                torch.mps.synchronize()
                encode_seconds += time.perf_counter() - encode_started
                writer.write(encoded)
            finally:
                if encoded is not None:
                    encoded.release()
                region.release()
        writer.close(wait=True)
    except BaseException:
        writer.abort()
        raise
    report["clipped"] = report["overflow"]
    _finish_report(report)
    report["complete"] = True
    write_seconds = time.perf_counter() - write_started
    source_representations = {
        str(source.metadata.get("representation", "unknown")) for source in sources
    }
    merge_record = {
        "version": 1,
        "backend": "mps",
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
        "real_space_shifts_row_column": real_np.tolist(),
        "diffraction_shifts_row_column": diffraction_np.tolist(),
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
        torch.mps.empty_cache()
    reopen_started = time.perf_counter()
    result = load(
        output_path, backend="mps", representation="packed", verbose=False
    )
    torch.mps.synchronize()
    release_unused = getattr(result.data, "release_unused_blocks", None)
    if callable(release_unused):
        release_unused()
    torch.mps.empty_cache()
    merge_record["reopen_seconds"] = time.perf_counter() - reopen_started
    merge_record["total_seconds"] = time.perf_counter() - started
    with h5py.File(output_path, "r+") as handle:
        handle.attrs["quantem_maped_merge_v1"] = json.dumps(merge_record)
    result.metadata["maped_merge"] = merge_record
    result.metadata["maped_summary"] = summaries
    return result
