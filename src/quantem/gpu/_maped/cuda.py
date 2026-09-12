"""CUDA implementation for bounded MAPED merging from encoded residents."""
import json
import math
from functools import cache
from pathlib import Path
import time

import cupy as cp
import numpy as np
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


@cache
def _region_kernels(device: int):
    """Compile and retain MAPED region kernels once per CUDA device."""
    with cp.cuda.Device(device):
        module = cp.RawModule(
            code=Path(__file__).with_name("regions.cu").read_text(),
            options=("--fmad=false",),
        )
        return {
            name: module.get_function(name)
            for name in ("sample", "sample_dense", "accumulate", "normalize")
        }


def _merge_plan(sources, real_shifts, diffraction_shifts):
    """Build interpolation state shared by both bounded merge passes."""
    shape = sources[0].shape
    _, _, height, width = shape
    real = torch.from_dlpack(real_shifts)
    diffraction = torch.from_dlpack(diffraction_shifts)
    real_weights, detector_weights = _weights(shape, real, diffraction)
    detector_edge = 1 - detector_weights.sum(0).clamp(0, 1)
    torch.cuda.synchronize()
    return {
        "real": real,
        "diffraction": diffraction,
        "real_weights": real_weights,
        "detector_weights": detector_weights,
        "detector_edge": detector_edge,
        "real_cp": cp.from_dlpack(real),
        "diffraction_cp": cp.from_dlpack(diffraction),
        "real_weights_cp": cp.from_dlpack(real_weights).reshape(len(sources), -1),
        "detector_weights_cp": cp.from_dlpack(detector_weights),
        "detector_edge_cp": cp.from_dlpack(detector_edge),
        "real_row_shifts": cp.asnumpy(cp.from_dlpack(real)[:, 0]),
        "masks": [
            cp.ones((height, width), cp.uint8)
            if source.metadata.get("pixel_mask") is None
            or correction_is_applied(source.metadata)
            else (cp.asarray(source.metadata["pixel_mask"]) == 0).astype(cp.uint8)
            for source in sources
        ],
    }


def _merge_regions(
    sources,
    real_shifts,
    diffraction_shifts,
    scans_per_region=1024,
    *,
    plan=None,
):
    """Yield GPU float32 regions for fixed bilinear, unpadded MAPED settings.

    All packed owners remain resident. Caller writes each region before advancing.
    The supported scientific settings match the retained default merge benchmark:
    real_space_edge_blend=1, diffraction_edge_blend=0, no padding or scaling.
    """
    shape = sources[0].shape
    if any(source.shape != shape for source in sources):
        raise ValueError('All packed tilts must have matching complete geometry.')
    rows, cols, height, width = shape
    plan = _merge_plan(sources, real_shifts, diffraction_shifts) if plan is None else plan
    kernels = _region_kernels(cp.cuda.Device().id)
    sample_kernel = kernels["sample"]
    sample_dense = kernels["sample_dense"]
    accumulate = kernels["accumulate"]
    normalize = kernels["normalize"]
    masks = plan["masks"]
    real_cp = plan["real_cp"]
    diffraction_cp = plan["diffraction_cp"]
    wi_cp = plan["real_weights_cp"]
    wd_cp = plan["detector_weights_cp"]
    edge_cp = plan["detector_edge_cp"]
    real_row_shifts = plan["real_row_shifts"]
    decode_errors = cp.zeros(1, cp.uint32)
    for first in range(0, rows * cols, scans_per_region):
        stop = min(first + scans_per_region, rows * cols)
        count = stop - first
        numerator = cp.empty((count, height, width), cp.float32)
        sampled = cp.empty_like(numerator)
        vectors = count * ((height * width + 3) // 4)
        launch = (((vectors + 255) // 256,), (256,))
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
                    cp.int32(i == 0),
                ),
            )
            del decoded
        normalize(
            *launch,
            (
                numerator,
                wi_cp,
                wd_cp,
                edge_cp,
                cp.int32(first),
                cp.int32(count),
                cp.int32(rows * cols),
                cp.int32(height * width),
                cp.int32(len(sources)),
            ),
        )
        if int(decode_errors.get()[0]):
            raise ValueError(
                "An ANS count stream failed reconstruction during MAPED merging."
            )
        yield first, numerator
        del numerator, sampled


def _automatic_region_frames(shape) -> int:
    """Choose a bounded internal scan region from currently free CUDA memory."""
    free, _ = cp.cuda.runtime.memGetInfo()
    detector_pixels = math.prod(shape[2:])
    # Numerator, sampled values, denominator, and encoded output coexist at the
    # write boundary. Leave a fixed reserve for CUDA libraries and alignment data.
    bytes_per_frame = detector_pixels * (4 + 4 + 4 + 2)
    available = max(0, int(free) - 512 * 1024**2)
    frames = max(1, min(4096, available // max(1, bytes_per_frame * 4)))
    scan_columns = int(shape[1])
    if frames >= scan_columns:
        frames = max(scan_columns, frames // scan_columns * scan_columns)
    return int(frames)


class _ResidentMerge:
    """Re-readable bounded MAPED output consumed by :func:`quantem.gpu.io.save`.

    This is an internal data source, not a second MAPED API.  It exposes the
    same shape/dtype/blocks contract used by generic bounded precision saving,
    while retaining the fused CUDA implementation and its measured timings.
    """

    dtype = np.dtype("float32")
    report_context = {
        "scope": "all merged values",
        "range_scope": "complete merged output",
        "measurement": "GPU comparison against merged float32 regions",
    }

    def __init__(self, sources, real_shifts, diffraction_shifts):
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
        device_id = int(real_shifts.device.index)
        for source in sources:
            resident = source.data
            source_device = getattr(
                resident, "device", getattr(resident, "_device_id", None)
            )
            if source_device != device_id:
                raise ValueError(
                    "Every encoded source and both shift arrays must share one CUDA device."
                )
        cp.cuda.Device(device_id).use()
        self.sources = sources
        self.shape = shape
        self.real_shifts = real_shifts
        self.diffraction_shifts = diffraction_shifts
        self._device_id = device_id
        self.region_frames = _automatic_region_frames(shape)
        self.plan = _merge_plan(sources, real_shifts, diffraction_shifts)
        self.started = time.perf_counter()
        self.range_seconds = 0.0
        self.encode_seconds = 0.0
        self.write_pass_seconds = 0.0
        self.release_sources_before_reopen = False
        self.save_metadata = {
            "quantem_maped_summary_v1": json.dumps(summary_record(shape, sources))
        }

    def blocks(self):
        """Yield one bounded float32 output pass."""
        for _, region in _merge_regions(
            self.sources,
            self.real_shifts,
            self.diffraction_shifts,
            self.region_frames,
            plan=self.plan,
        ):
            yield region

    def range(self):
        """Measure the global output range without materializing the result."""
        started = time.perf_counter()
        low = cp.asarray(cp.inf, dtype=cp.float32)
        high = cp.asarray(-cp.inf, dtype=cp.float32)
        for region in self.blocks():
            cp.minimum(low, cp.min(region), out=low)
            cp.maximum(high, cp.max(region), out=high)
        cp.cuda.get_current_stream().synchronize()
        result = float(low.get()), float(high.get())
        if not all(np.isfinite(value) for value in result):
            raise ValueError("The merged output contains non-finite intensities.")
        self.range_seconds = time.perf_counter() - started
        return result

    def encode_blocks(self, report):
        """Yield scaled uint16 blocks while accumulating error on CUDA."""
        if report["storage"] != "scaled_uint16":
            raise ValueError(
                "Resident MAPED saving currently supports dtype='scaled_uint16'."
            )
        from quantem.gpu.io.backends.cuda.precision import (
            encode_measure_scaled_uint16,
        )

        stats = [
            cp.zeros((), dtype=cp.float64),
            cp.zeros((), dtype=cp.float64),
            cp.zeros((), dtype=cp.uint64),
            cp.zeros((), dtype=cp.uint64),
            cp.zeros((), dtype=cp.uint64),
        ]
        events = []
        started = time.perf_counter()
        for region in self.blocks():
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record()
            encoded = encode_measure_scaled_uint16(region, report, stats)
            end.record()
            events.append((begin, end))
            yield encoded
        cp.cuda.get_current_stream().synchronize()
        self.write_pass_seconds = time.perf_counter() - started
        self.encode_seconds = sum(
            cp.cuda.get_elapsed_time(begin, end) for begin, end in events
        ) / 1000.0
        report["values"] = math.prod(self.shape)
        report["squared_error"] = float(stats[0].get())
        report["max_abs_error"] = float(stats[1].get())
        report["positive_to_zero"] = int(stats[2].get())
        report["changed"] = int(stats[3].get())
        report["overflow"] = int(stats[4].get())
        report["clipped"] = report["overflow"]
        representations = {
            str(source.metadata.get("representation", "unknown"))
            for source in self.sources
        }
        record = {
            "version": 1,
            "backend": "cuda",
            "source_representation": (
                representations.pop() if len(representations) == 1 else "mixed"
            ),
            "region_frames": self.region_frames,
            "range_seconds": self.range_seconds,
            "gpu_encode_seconds": self.encode_seconds,
            "merge_encode_write_seconds": self.write_pass_seconds,
            "released_sources_before_reopen": bool(
                self.release_sources_before_reopen
            ),
            "real_space_shifts_row_column": self.real_shifts.detach().cpu().tolist(),
            "diffraction_shifts_row_column": (
                self.diffraction_shifts.detach().cpu().tolist()
            ),
        }
        self.save_metadata["quantem_maped_merge_v1"] = json.dumps(record)

    def close(self):
        """Release merge planning buffers while retaining caller-owned inputs."""
        self.plan = None


def resident_merge(sources, real_shifts, diffraction_shifts):
    """Return an internal bounded source for generic GPU saving."""
    return _ResidentMerge(sources, real_shifts, diffraction_shifts)
