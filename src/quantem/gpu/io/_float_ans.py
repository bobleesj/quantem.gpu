"""Portable float32 bit-lane residents with bounded accelerator decoding."""

import base64
import copy
from contextlib import nullcontext
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Any

import numpy as np

from . import _qem_metadata

PROFILE = "float32-bit-lanes-rans-v1"
MAX_DECODE_BYTES = 32 << 20
_BLOCK = 64 << 20


def _device_scoped(operation):
    """Keep queries on the resident's GPU even if a client switches devices."""

    @wraps(operation)
    def run(self, *args, **kwargs):
        with self.device_context():
            return operation(self, *args, **kwargs)

    return run


class FloatANSResident:
    """Own encoded IEEE bits; materialize at most a 32 MiB scan window.

    File measurements are never cast to counts. Raw reads preserve every bit;
    scientific products apply the recorded mean-dark subtraction exactly once.
    Use ``io.load(path, backend='mps')`` or ``backend='cuda'`` to construct one.
    """

    dtype = np.dtype("float32")
    interval = 512

    def __init__(
        self, header: dict[str, Any], backend: str, device: int | str | None = None
    ):
        self.header = copy.deepcopy(header)
        self.shape = tuple(header["shape"])
        self.backend = backend
        self.is_released = False
        self.valid_pixels = np.ones(self.shape[2:], bool)
        self._background = None
        self.peak_decode_bytes = 0
        self._mean = None
        lane_shape = (*self.shape[:2], 128, 256)
        if backend == "cuda":
            import cupy as cp
            from .backends.cuda.float_ans import CUDAFloatLanes

            selected = (
                cp.cuda.Device().id
                if device is None
                else int(str(device).removeprefix("cuda:"))
            )
            with cp.cuda.Device(selected):
                self._lanes = CUDAFloatLanes(lane_shape, np.uint16)
            self.device = selected
        else:
            from .backends.mps.float_ans import MPSFloatLanes

            self._lanes = MPSFloatLanes(lane_shape, np.uint16)
            self.device = self._lanes.device

    @property
    def nbytes(self) -> int:
        if self.is_released:
            return 0
        cached = (self._background, self._mean)
        return self._lanes.nbytes + sum(
            value.nbytes
            if self.backend == "cuda"
            else value.numel() * value.element_size()
            for value in cached
            if value is not None
        )

    @property
    def logical_nbytes(self) -> int:
        return math.prod(self.shape) * 4

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            "Float ANS stays encoded; request a DP, product or bounded scan window."
        )

    def device_context(self):
        if self.backend == "cuda":
            import cupy as cp

            return cp.cuda.Device(self.device)
        return nullcontext()

    def _check(self, first=None, stop=None):
        if self.is_released:
            raise RuntimeError("The float ANS resident was released; load it again.")
        if first is not None:
            if (
                type(first) is not int
                or type(stop) is not int
                or not 0 <= first < stop <= math.prod(self.shape[:2])
            ):
                raise ValueError("Choose a nonempty scan range within the acquisition.")
            size = (stop - first) * 128 * 128 * 4
            if size > MAX_DECODE_BYTES:
                raise ValueError(
                    "Decode requests are limited to 32 MiB; process scan windows of at most 512 frames."
                )
            self.peak_decode_bytes = max(self.peak_decode_bytes, size)

    def decode_scan_range_device(self, first: int, stop: int) -> Any:
        """Return original float32 measurement bits in one bounded device buffer."""
        self._check(first, stop)
        output = self._lanes.decode_scan_range_device(first, stop)
        if self.backend == "cuda":
            return output.view(np.float32).reshape(stop - first, 128, 128)
        output.dtype = self.dtype
        output.shape = (stop - first, 128, 128)
        return output

    def extract_diffraction_device(self, scan_row: int, scan_column: int) -> Any:
        """Return one original DP without applying a display correction."""
        if not 0 <= scan_row < self.shape[0] or not 0 <= scan_column < self.shape[1]:
            raise IndexError("Choose a scan row and column within the acquisition.")
        first = scan_row * self.shape[1] + scan_column
        output = self.decode_scan_range_device(first, first + 1)
        if self.backend == "cuda":
            return output.reshape(128, 128)
        output.shape = (128, 128)
        return output

    @_device_scoped
    def _tensor(self, first, stop, *, corrected=True):
        self._check(first, stop)
        if self.backend == "cuda":
            output = self.decode_scan_range_device(first, stop)
        else:
            import torch

            output = self._lanes._decode_scan_range_torch(first, stop)
            output = output.view(torch.float32).reshape(stop - first, 128, 128)
        if corrected and self._background is not None:
            output -= self._background
        return output

    def _array(self, values):
        if self.backend == "cuda":
            import cupy as cp

            with cp.cuda.Device(self.device):
                return cp.asarray(values)
        import torch

        return torch.as_tensor(values, device="mps")

    def _empty(self, shape, *, zero=False):
        if self.backend == "cuda":
            import cupy as cp

            with cp.cuda.Device(self.device):
                return (cp.zeros if zero else cp.empty)(shape, cp.float32)
        import torch

        return (torch.zeros if zero else torch.empty)(
            shape, dtype=torch.float32, device="mps"
        )

    def synchronize(self) -> None:
        if self.backend == "cuda":
            import cupy as cp

            with cp.cuda.Device(self.device):
                cp.cuda.get_current_stream().synchronize()
        else:
            import torch

            torch.mps.synchronize()

    def _sum(self, values, axes):
        return values.sum(axis=axes) if self.backend == "cuda" else values.sum(dim=axes)

    def _where(self, condition, values, other=0):
        if self.backend == "cuda":
            import cupy as cp

            return cp.where(condition, values, other)
        import torch

        return torch.where(condition, values, other)

    @_device_scoped
    def products_device(
        self, mask: np.ndarray | None = None, *, moments: bool = False
    ) -> tuple[Any, ...]:
        """Reduce bounded windows; scale weights before computing CoM moments."""
        self._check()
        mask = np.ones((128, 128), bool) if mask is None else np.asarray(mask)
        if mask.shape != (128, 128) or not np.all((mask == 0) | (mask == 1)):
            raise ValueError(
                "Float detector masks must be binary with shape (128,128)."
            )
        selected = self._array(mask.astype(bool))
        total = self._empty((math.prod(self.shape[:2]),))
        products = [total]
        if moments:
            products += [self._empty(total.shape), self._empty(total.shape)]
            row = self._array(np.arange(128, dtype=np.float32)[:, None])
            column = self._array(np.arange(128, dtype=np.float32)[None, :])
        for chunk in self._lanes.chunks:
            first, stop = chunk.first, chunk.first + chunk.scans
            values = self._tensor(first, stop)
            values = self._where(selected, values)
            if moments:
                magnitude = (
                    abs(values).max(axis=(1, 2), keepdims=True)
                    if self.backend == "cuda"
                    else values.abs().amax(dim=(1, 2), keepdim=True)
                )
                # Preserve ordinary dyadic measurements exactly; rescale only
                # extreme exponents which can overflow weighted moments or
                # underflow. Unconditional division perturbs zero signed sums.
                extreme = (magnitude > 2.0**100) | (
                    (magnitude > 0) & (magnitude < 2.0**-100)
                )
                values /= self._where(extreme, magnitude, 1)
            total[first:stop] = self._sum(values, (1, 2))
            if moments:
                products[1][first:stop] = self._sum(values * row, (1, 2))
                products[2][first:stop] = self._sum(values * column, (1, 2))
            del values
        return tuple(value.reshape(self.shape[:2]) for value in products)

    def detector_sum_device(self, mask: np.ndarray) -> Any:
        self._check()
        values = np.asarray(mask)
        if values.shape != (128, 128) or not np.all((values == 0) | (values == 1)):
            raise ValueError(
                "Float detector masks must be binary with shape (128,128)."
            )
        size = max(chunk.scans for chunk in self._lanes.chunks) * 65536
        self.peak_decode_bytes = max(self.peak_decode_bytes, size)
        return self._lanes.float_detector(values, self._background)

    @_device_scoped
    def mean_dp_device(self) -> Any:
        """Compute and cache the full-scan mean on the accelerator."""
        self._check()
        if self._mean is None:
            total = self._empty((128, 128), zero=True)
            for chunk in self._lanes.chunks:
                values = self._tensor(chunk.first, chunk.first + chunk.scans)
                total += self._sum(values, 0)
                del values
            self._mean = total / math.prod(self.shape[:2])
        return self._mean.copy() if self.backend == "cuda" else self._mean.clone()

    @_device_scoped
    def reduce_frames_device(self, indices, mode="mean"):
        """Reduce selected scan positions with bounded GPU working storage."""
        self._check()
        indices = np.asarray(list(indices))
        if (
            indices.ndim != 1
            or not len(indices)
            or indices.dtype.kind not in "iu"
            or indices.min() < 0
            or indices.max() >= math.prod(self.shape[:2])
        ):
            raise ValueError("Select integer scan positions within the acquisition.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Select each scan position once for a region reduction.")
        if mode not in ("sum", "mean", "max"):
            raise ValueError("Use mean, sum or max for the selected DPs.")
        total = None
        for chunk in self._lanes.chunks:
            local = (
                indices[
                    (indices >= chunk.first) & (indices < chunk.first + chunk.scans)
                ]
                - chunk.first
            )
            if not len(local):
                continue
            first, stop = int(local.min()), int(local.max()) + 1
            values = self._tensor(chunk.first + first, chunk.first + stop)
            selected = values[self._array((local - first).astype(np.int64))]
            if mode == "max":
                reduced = (
                    selected.max(axis=0)
                    if self.backend == "cuda"
                    else selected.amax(dim=0)
                )
                if total is None:
                    total = reduced
                else:
                    if self.backend == "cuda":
                        import cupy as cp

                        total = cp.maximum(total, reduced)
                    else:
                        import torch

                        total = torch.maximum(total, reduced)
            else:
                reduced = self._sum(selected, 0)
                total = reduced if total is None else total + reduced
            del selected, values
        return total / len(indices) if mode == "mean" else total

    def release(self) -> None:
        if not self.is_released:
            self.synchronize()
            self._lanes.release()
            self._lanes = None
            self._background = self._mean = None
            self.is_released = True


def load_float_ans(
    path: str | Path,
    header: dict[str, Any],
    start: int,
    *,
    backend: str,
    device: int | str | None = None,
):
    """Authenticate and upload encoded chunks; never expand a complete source."""
    from .models import FourDSTEMData

    started = time.perf_counter()
    source = FloatANSResident(header, backend, device)
    digest, block_bytes, block_index = hashlib.sha256(), 0, 0
    arrays = []
    try:
        with open(path, "rb", buffering=0) as handle:
            handle.seek(start)
            for chunk in header["chunks"]:
                arrays = []
                for name, dtype in (
                    ("payload", "u1"),
                    ("offset", "<u4"),
                    ("model", "u1"),
                ):
                    raw = handle.read(chunk[name + "_bytes"])
                    if len(raw) != chunk[name + "_bytes"]:
                        raise ValueError("QEM ended during loading; recopy the file.")
                    view = memoryview(raw)
                    while view:
                        part = min(len(view), _BLOCK - block_bytes)
                        digest.update(view[:part])
                        block_bytes += part
                        view = view[part:]
                        if block_bytes == _BLOCK:
                            if digest.hexdigest() != header["sha256"][block_index]:
                                raise ValueError(
                                    "QEM body checksum mismatch; recopy the file."
                                )
                            digest, block_bytes, block_index = (
                                hashlib.sha256(),
                                0,
                                block_index + 1,
                            )
                    values = np.frombuffer(raw, dtype)
                    if backend == "cuda":
                        arrays.append(source._array(values))
                    else:
                        from .backends.mps._streamed import _upload

                        arrays.append(
                            _upload(
                                source._lanes._device,
                                source._lanes._metal,
                                values,
                                "Float ANS encoded bytes",
                            )
                        )
                if backend == "cuda":
                    from quantem.gpu._compact.streamed import Chunk

                    source._lanes.chunks.append(
                        Chunk(chunk["first"], chunk["scans"], tuple(arrays))
                    )
                else:
                    from .backends.mps._streamed import _Chunk

                    source._lanes.chunks.append(
                        _Chunk(chunk["first"], chunk["scans"], tuple(arrays))
                    )
                arrays = []  # The resident now owns these buffers.
            if block_bytes and digest.hexdigest() != header["sha256"][block_index]:
                raise ValueError("QEM body checksum mismatch; recopy the file.")
        source._lanes.ready_scans = math.prod(source.shape[:2])
        background = header["empad"].get("background")
        if background is not None:
            raw = base64.b64decode(background["values_float32_le"], validate=True)
            if len(raw) != 128 * 128 * 4:
                raise ValueError(
                    "QEM mean-dark plane must contain 128x128 float32 values."
                )
            source._background = source._array(
                np.frombuffer(raw, "<f4").copy().reshape(128, 128)
            )
        source.synchronize()
        metadata = _qem_metadata.effective_metadata(
            header.get("metadata", {}), header["scientific_metadata"]
        )
        metadata.update(
            scientific_metadata=header["scientific_metadata"],
            qem_empad=header["empad"],
            container=header["container"],
            container_version=header["container_version"],
            source_path=str(Path(path).resolve()),
            source_kind="resident",
            device=f"cuda:{source.device}" if backend == "cuda" else "mps",
            n_frames=math.prod(source.shape[:2]),
            source_shape=source.shape,
            shape=source.shape,
            working_shape=source.shape,
            scan_shape=source.shape[:2],
            detector_shape=source.shape[2:],
            dtype="float32",
            working_dtype="float32",
            source_dtype="float32",
            backend=backend,
            representation="encoded",
            residency="device",
            resident_profile=PROFILE,
            physical_resident_bytes=source.nbytes,
            source_logical_tensor_bytes=source.logical_nbytes,
            working_logical_tensor_bytes=source.logical_nbytes,
            file_counts_exact=True,
            lossless_exact=True,
            scan_bin=1,
            detector_bin=1,
            crop=None,
            background_applied=False,
            background_applied_by_reader=background is not None,
            load_timings={
                "resident_ready_seconds": time.perf_counter() - started,
                "encode_seconds": 0.0,
                "verified_encoded_bytes": header["bytes"],
            },
        )
        return FourDSTEMData(source, metadata)
    except BaseException:
        if backend == "mps":
            from .backends.mps.packed import _release

            for buffer in arrays:
                _release(buffer)
        source.release()
        raise


def save_float_ans(
    path: str | Path, source: FloatANSResident, metadata: dict[str, Any] | None = None
) -> None:
    """Save encoded resident bytes without a decoded cube or original source file."""
    source._check()
    path = Path(path)
    if path.suffix.lower() != ".qem" or path.exists():
        raise ValueError("Choose a non-existing .qem destination.")
    source.synchronize()
    header = copy.deepcopy(source.header)
    if metadata is not None:
        header["scientific_metadata"] = _qem_metadata.acquisition_metadata(
            source.shape, metadata
        )
    blob = json.dumps(
        header, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()
    if len(blob) > 16 << 20:
        raise ValueError("QEM metadata exceeds the 16 MiB limit.")
    descriptor, temporary = tempfile.mkstemp(prefix=".qem-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(
                _qem_metadata.MAGIC + struct.pack("<QQ", len(blob), 56 + len(blob))
            )
            output.write(hashlib.sha256(blob).digest() + blob)
            for chunk in source._lanes.chunks:
                if source.backend == "cuda":
                    arrays = (array.get() for array in chunk.arrays)
                else:
                    from .backends.mps.packed import _buffer_view

                    arrays = (_buffer_view(buffer) for buffer in chunk.buffers)
                for array in arrays:
                    output.write(memoryview(array).cast("B"))
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
