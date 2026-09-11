"""Private bounded exact count-rANS primitives, independent of disk framing.

This module consumes authenticated format arrays supplied by the IO owner. It
does not interpret research folders or create a second public load function.
The recurrence matches the retained detector-conditioned rANS experiments;
geometry, model selection, allocation bounds, and lifecycle are explicit here.
"""

from functools import cache
from pathlib import Path

import numpy as np

from quantem.gpu.io._ans_contract import _validate_arrays


def _flat_scan_indices(
    scan_positions: np.ndarray, scan_shape: tuple[int, int]
) -> np.ndarray:
    """Flatten exact row/column indices without signed/unsigned float promotion."""
    positions = np.asarray(scan_positions)
    if (
        positions.ndim != 2
        or positions.shape[1] != 2
        or positions.dtype.kind not in "iu"
    ):
        raise ValueError(
            "scan_positions must be an integer (N, 2) array in row, column order."
        )
    if (
        np.any(positions < 0)
        or np.any(positions[:, 0] >= scan_shape[0])
        or np.any(positions[:, 1] >= scan_shape[1])
    ):
        raise IndexError(f"Scan positions must be inside {scan_shape}.")
    positions = positions.astype(np.uint64, copy=False)
    return positions[:, 0] * np.uint64(scan_shape[1]) + positions[:, 1]


@cache
def _kernels(device_id: int):
    """Compile once per device without creating accelerator work at import time."""
    import cupy as cp

    with cp.cuda.Device(device_id):
        source = Path(__file__).with_suffix(".cu").read_text()
        module = cp.RawModule(code=source, options=("--std=c++11",))
        return {
            name: module.get_function(name)
            for name in (
                "dense_measure_packed",
                "dense_write_packed",
                "ans_validate",
                "ans_decode_block",
                "ans_diffraction",
                "ans_detector_sum",
                "ans_measure_packed",
                "ans_write_packed",
                "packed_diffraction",
                "packed_decode_block",
                "packed_detector_sum",
            )
        }


class CudaANSResidentCounts:
    """Private native uint8/uint16 source with bounded per-detector rANS streams.

    Stream number is ``block * detector_count + detector_pixel``. Each stream
    uses one model selector. Payload/table arrays are copied into owned device
    buffers and every stream is validated before the constructor returns.
    No full dense tensor is allocated. Admission does traverse all streams;
    subsequent selected diffraction accesses decode only a requested block's
    prefixes. A literal column needs just the requested two bytes.

    Returned device arrays are caller-owned, not reused by later requests.
    Each operation synchronizes its error flag before publication. This is an
    exact correctness primitive, not a claim of interactive timing qualification.
    """

    def __init__(
        self,
        *,
        shape,
        block_frames,
        scale,
        payload,
        offsets,
        model_ids,
        context_offsets,
        symbols,
        cumulative,
        frequencies,
        literal,
        dtype="uint16",
    ):
        self._dtype = np.dtype(dtype)
        if self._dtype not in (np.dtype("uint8"), np.dtype("uint16")):
            raise ValueError("Native count dtype must be uint8 or uint16.")
        shape, arrays = _validate_arrays(
            shape=shape,
            block_frames=block_frames,
            scale=scale,
            payload=payload,
            offsets=offsets,
            model_ids=model_ids,
            context_offsets=context_offsets,
            symbols=symbols,
            cumulative=cumulative,
            frequencies=frequencies,
            literal=literal,
        )
        import cupy as cp

        self.shape = shape
        self.block_frames = int(block_frames)
        self.scale = int(scale)
        block_frames = self.block_frames
        scale = self.scale
        self._scan_count = shape[0] * shape[1]
        self._detector_count = shape[2] * shape[3]
        self._block_count = (self._scan_count + block_frames - 1) // block_frames
        self._stream_count = self._block_count * self._detector_count
        self._device_id = cp.cuda.Device().id
        self._kernels = _kernels(self._device_id)
        self._arrays = ()
        self.is_released = False
        try:
            self._arrays = tuple(cp.array(value, copy=True) for value in arrays)
            errors = cp.zeros(1, dtype=cp.uint32)
            self._launch(
                "ans_validate",
                self._stream_count,
                (
                    *self._arrays,
                    errors,
                    np.uint64(self._scan_count),
                    np.uint32(self._detector_count),
                    np.uint32(block_frames),
                    np.uint32(scale),
                    np.uint64(self._stream_count),
                    np.uint32(np.iinfo(self._dtype).max),
                ),
            )
            self._check(errors)
        except BaseException:
            self.release()
            raise

    @property
    def dtype(self) -> np.dtype:
        """Logical source dtype, never the dtype of compressed bytes."""
        return self._dtype

    @property
    def resident_bytes(self) -> int:
        """Owned payload, index and probability-table bytes, excluding outputs."""
        return sum(array.nbytes for array in self._arrays)

    @property
    def nbytes(self) -> int:
        """Physical payload, index and table bytes, excluding returned arrays."""
        return self.resident_bytes

    @property
    def logical_nbytes(self) -> int:
        """Bytes of the equivalent dense native-count tensor."""
        return self._scan_count * self._detector_count * self.dtype.itemsize

    def _require_resident(self):
        import cupy as cp

        if self.is_released:
            raise RuntimeError(
                "The ANS source was released; load a new source before reading it."
            )
        if cp.cuda.Device().id != self._device_id:
            raise RuntimeError(
                "Select this source's original CUDA device before using its buffers."
            )

    def _launch(self, name, count, arguments):
        self._kernels[name](((count + 127) // 128,), (128,), arguments)

    @staticmethod
    def _check(errors):
        status = int(errors.get()[0])
        if status & 2:
            raise ValueError(
                "Decoded counts exceed the declared native dtype; preserve uint16."
            )
        if status:
            raise ValueError(
                "Malformed rANS stream: normalization or exact terminal state failed."
            )

    def decode_block_device(self, block_index: int):
        """Decode one block to caller-owned ``(scan, row, column)`` native counts."""
        self._require_resident()
        if (
            not isinstance(block_index, (int, np.integer))
            or not 0 <= block_index < self._block_count
        ):
            raise IndexError(
                f"block_index must be in [0, {self._block_count}); got {block_index}."
            )
        import cupy as cp

        count = min(
            self.block_frames, self._scan_count - block_index * self.block_frames
        )
        output = cp.empty((count, *self.shape[2:]), dtype=cp.uint16)
        errors = cp.zeros(1, dtype=cp.uint32)
        self._launch(
            "ans_decode_block",
            self._detector_count,
            (
                *self._arrays,
                output,
                errors,
                np.uint64(block_index),
                np.uint32(self._detector_count),
                np.uint32(count),
                np.uint32(self.scale),
            ),
        )
        self._check(errors)
        return output.astype(self.dtype, copy=False)

    def gather_diffraction_device(self, scan_positions: np.ndarray):
        """Read exact requested patterns, retaining request order and duplicates."""
        self._require_resident()
        flat = _flat_scan_indices(scan_positions, self.shape[:2])
        import cupy as cp

        output = cp.empty((len(flat), *self.shape[2:]), dtype=cp.uint16)
        if not len(flat):
            return output.astype(self.dtype, copy=False)
        requested = cp.asarray(flat, dtype=cp.uint64)
        errors = cp.zeros(1, dtype=cp.uint32)
        self._launch(
            "ans_diffraction",
            output.size,
            (
                *self._arrays,
                requested,
                output,
                errors,
                np.uint64(len(flat)),
                np.uint32(self._detector_count),
                np.uint32(self.block_frames),
                np.uint32(self.scale),
            ),
        )
        self._check(errors)
        return output.astype(self.dtype, copy=False)

    def extract_diffraction_device(self, scan_row: int, scan_column: int):
        """Read one exact requested pattern with native uint16 values."""
        return self.gather_diffraction_device(np.asarray([[scan_row, scan_column]]))[0]

    def detector_sum_device(self, mask: np.ndarray):
        """Sum a binary detector mask to exact scan-shaped uint64 counts."""
        self._require_resident()
        values = np.asarray(mask)
        if (
            values.shape != self.shape[2:]
            or values.dtype.kind not in "buif"
            or np.any((values != 0) & (values != 1))
        ):
            raise ValueError(
                f"mask must have detector shape {self.shape[2:]} and contain only zero or one."
            )
        import cupy as cp

        selected = cp.asarray(values.reshape(-1), dtype=cp.uint8)
        output = cp.zeros(self.shape[:2], dtype=cp.uint64)
        errors = cp.zeros(1, dtype=cp.uint32)
        self._launch(
            "ans_detector_sum",
            self._stream_count,
            (
                *self._arrays,
                selected,
                output,
                errors,
                np.uint64(self._scan_count),
                np.uint32(self._detector_count),
                np.uint32(self.block_frames),
                np.uint32(self.scale),
                np.uint64(self._stream_count),
            ),
        )
        self._check(errors)
        return output

    def release(self):
        """Release only buffers owned by this source, never global allocator state."""
        self._arrays = ()
        self.is_released = True

    def to_packed(self):
        """Transcode directly to exact 0..16-bit streams without a dense tensor.

        The original source remains valid. The caller explicitly releases it
        after accepting the new source. Both sources coexist during conversion;
        ``conversion_owned_buffer_peak_bytes`` includes both plus index scratch,
        but is not a measurement of process memory or allocator reserve.
        """
        self._require_resident()
        import cupy as cp

        widths = cp.empty(self._stream_count, dtype=cp.uint8)
        lengths = cp.empty(self._stream_count, dtype=cp.uint64)
        errors = cp.zeros(1, dtype=cp.uint32)
        arguments = (
            np.uint64(self._scan_count),
            np.uint32(self._detector_count),
            np.uint32(self.block_frames),
            np.uint32(self.scale),
            np.uint64(self._stream_count),
        )
        self._launch(
            "ans_measure_packed",
            self._stream_count,
            (
                *self._arrays,
                widths,
                lengths,
                errors,
                *arguments,
            ),
        )
        self._check(errors)
        offsets = cp.zeros(self._stream_count + 1, dtype=cp.uint64)
        cp.cumsum(lengths, dtype=cp.uint64, out=offsets[1:])
        word_count = int(offsets[-1].get())
        if word_count > np.iinfo(np.intp).max // 4:
            raise MemoryError("Packed output exceeds the addressable array size.")
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        if word_count * 4 > free_bytes:
            raise MemoryError(
                "The original ANS source and packed output do not fit together; release other sources first."
            )
        words = cp.empty(word_count, dtype=cp.uint32)
        self._launch(
            "ans_write_packed",
            self._stream_count,
            (
                *self._arrays,
                widths,
                offsets,
                words,
                errors,
                *arguments,
            ),
        )
        self._check(errors)
        result = CudaPackedResidentCounts(
            self.shape,
            self.block_frames,
            self.dtype,
            words,
            offsets,
            widths,
            self._device_id,
            self._kernels,
        )
        result.conversion_owned_buffer_peak_bytes = (
            self.resident_bytes + result.resident_bytes + lengths.nbytes + errors.nbytes
        )
        return result


class CudaPackedResidentCounts:
    """Private word-aligned 0..16-bit source produced by exact GPU conversion.

    This is an accelerator layout, not a new disk format. Construction is only
    for already-validated device arrays produced by ``to_packed``. No ANS table
    or source lifetime is retained. Requested counts have direct bit access.
    """

    def __init__(
        self, shape, block_frames, dtype, words, offsets, widths, device_id, kernels
    ):
        self.shape = shape
        self.block_frames = block_frames
        self.dtype = dtype
        self._arrays = (words, offsets, widths)
        self._device_id = device_id
        self._kernels = kernels
        self._scan_count = shape[0] * shape[1]
        self._detector_count = shape[2] * shape[3]
        self._block_count = (self._scan_count + block_frames - 1) // block_frames
        self._stream_count = self._block_count * self._detector_count
        self.is_released = False
        self.conversion_owned_buffer_peak_bytes = None

    @classmethod
    def from_array(cls, values, shape):
        """Pack complete native counts using the existing exact block layout."""
        import cupy as cp

        if not isinstance(values, cp.ndarray) or values.dtype not in (
            np.dtype("uint8"), np.dtype("uint16")
        ):
            raise TypeError("Load native uint8/uint16 counts on CUDA before packing.")
        if len(shape) != 4 or values.size != int(np.prod(shape)):
            raise ValueError("Provide the complete four-dimensional scan/detector shape.")
        if not values.flags.c_contiguous:
            raise ValueError("Use contiguous native counts before packing.")
        block_frames = 128
        scan_count = shape[0] * shape[1]
        detector_count = shape[2] * shape[3]
        stream_count = ((scan_count + block_frames - 1) // block_frames) * detector_count
        device_id = values.device.id
        with cp.cuda.Device(device_id):
            kernels = _kernels(device_id)
            widths = cp.empty(stream_count, cp.uint8)
            lengths = cp.empty(stream_count, cp.uint64)
            dimensions = (
                np.uint64(scan_count), np.uint32(detector_count),
                np.uint32(block_frames), np.uint64(stream_count),
            )
            launch = (((stream_count + 127) // 128,), (128,))
            kernels["dense_measure_packed"](
                *launch, (values, np.uint32(values.dtype.itemsize),
                          widths, lengths, *dimensions)
            )
            offsets = cp.zeros(stream_count + 1, cp.uint64)
            cp.cumsum(lengths, dtype=cp.uint64, out=offsets[1:])
            words = cp.empty(int(offsets[-1].get()), cp.uint32)
            kernels["dense_write_packed"](
                *launch, (values, np.uint32(values.dtype.itemsize),
                          widths, offsets, words, *dimensions)
            )
            cp.cuda.get_current_stream().synchronize()
            result = cls(tuple(shape), block_frames, values.dtype, words,
                         offsets, widths, device_id, kernels)
            result.conversion_owned_buffer_peak_bytes = (
                values.nbytes + result.resident_bytes + lengths.nbytes
            )
            return result

    @property
    def resident_bytes(self) -> int:
        """Owned bit payload, word offsets and widths, excluding returned arrays."""
        return sum(array.nbytes for array in self._arrays)

    @property
    def nbytes(self) -> int:
        """Physical resident array bytes, not equivalent dense tensor bytes."""
        return self.resident_bytes

    @property
    def logical_nbytes(self) -> int:
        """Bytes of an equivalent dense source at its original count dtype."""
        return self._scan_count * self._detector_count * self.dtype.itemsize

    def _require_resident(self):
        import cupy as cp

        if self.is_released:
            raise RuntimeError(
                "The packed source was released; load a new source before reading it."
            )
        if cp.cuda.Device().id != self._device_id:
            raise RuntimeError(
                "Select this source's original CUDA device before using its buffers."
            )

    def _launch(self, name, count, arguments):
        self._kernels[name](((count + 127) // 128,), (128,), arguments)

    def decode_block_device(self, block_index: int):
        """Decode one bounded block to caller-owned native counts."""
        self._require_resident()
        if (
            not isinstance(block_index, (int, np.integer))
            or not 0 <= block_index < self._block_count
        ):
            raise IndexError(
                f"block_index must be in [0, {self._block_count}); got {block_index}."
            )
        import cupy as cp

        count = min(
            self.block_frames, self._scan_count - block_index * self.block_frames
        )
        output = cp.empty((count, *self.shape[2:]), dtype=cp.uint16)
        self._launch(
            "packed_decode_block",
            output.size,
            (
                *self._arrays,
                output,
                np.uint64(block_index),
                np.uint32(count),
                np.uint32(self._detector_count),
            ),
        )
        cp.cuda.get_current_stream().synchronize()
        return output.astype(self.dtype, copy=False)

    def gather_diffraction_device(self, scan_positions: np.ndarray):
        """Read native patterns with direct bit access, retaining order and duplicates."""
        self._require_resident()
        flat = _flat_scan_indices(scan_positions, self.shape[:2])
        import cupy as cp

        output = cp.empty((len(flat), *self.shape[2:]), dtype=cp.uint16)
        if len(flat):
            requested = cp.asarray(flat, dtype=cp.uint64)
            self._launch(
                "packed_diffraction",
                output.size,
                (
                    *self._arrays,
                    requested,
                    output,
                    np.uint64(len(flat)),
                    np.uint32(self._detector_count),
                    np.uint32(self.block_frames),
                ),
            )
            cp.cuda.get_current_stream().synchronize()
        return output.astype(self.dtype, copy=False)

    def extract_diffraction_device(self, scan_row: int, scan_column: int):
        """Read one exact requested pattern with its original native dtype."""
        return self.gather_diffraction_device(np.asarray([[scan_row, scan_column]]))[0]

    def detector_sum_device(self, mask: np.ndarray):
        """Fuse bit extraction and exact uint64 binary-mask accumulation."""
        self._require_resident()
        values = np.asarray(mask)
        if (
            values.shape != self.shape[2:]
            or values.dtype.kind not in "buif"
            or np.any((values != 0) & (values != 1))
        ):
            raise ValueError(
                f"mask must have detector shape {self.shape[2:]} and contain only zero or one."
            )
        import cupy as cp

        selected = cp.asarray(values.reshape(-1), dtype=cp.uint8)
        output = cp.zeros(self.shape[:2], dtype=cp.uint64)
        self._launch(
            "packed_detector_sum",
            self._stream_count,
            (
                *self._arrays,
                selected,
                output,
                np.uint64(self._scan_count),
                np.uint32(self._detector_count),
                np.uint32(self.block_frames),
                np.uint64(self._stream_count),
            ),
        )
        cp.cuda.get_current_stream().synchronize()
        return output

    def release(self):
        """Release this source without affecting prior outputs or other sources."""
        self._arrays = ()
        self.is_released = True
