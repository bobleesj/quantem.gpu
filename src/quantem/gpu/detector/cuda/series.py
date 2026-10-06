"""Joint native queries over equally shaped dense or encoded acquisitions."""

import threading
import time
from functools import cache
from pathlib import Path

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.resident.cuda import counts
from quantem.gpu.resident.hot_pixels import correction_is_applied
from quantem.gpu.resident.queries import DetectorQueries


@cache
def _kernels(device):
    """Compile once per device; the series kernels include the streamed-count reader they dispatch to."""
    source = (
        Path(counts.__file__).with_name("kernels").joinpath("streamed.cu").read_text()
        + Path(__file__).with_name("kernels").joinpath("series.cu").read_text()
    )
    with cp.cuda.Device(device):
        module = cp.RawModule(code=source, options=("--std=c++17",))
        return {
            name: module.get_function(name)
            for name in (
                "series_sum_u32",
                "series_sum_u64",
                "series_frame_u8",
                "series_frame_u16",
            )
        }


class CudaSeriesCompute(DetectorQueries):
    """Borrow complete sources and calculate all acquisitions in one launch.

    Source buffers are retained without copying or dense expansion. The initial
    joint contract accepts equally shaped uint8/uint16 acquisitions. The
    arithmetic bound selects uint32 or uint64 sums without narrowing counts.
    Ordinary NumPy session outputs retain their existing conversion policy.
    """

    def __init__(self, acquisitions):
        if not acquisitions:
            raise ValueError("Select at least one complete acquisition.")
        self.owners = tuple(acquisitions)
        self.keepalive = []
        rows, valid_masks, source_dtypes = [], [], []
        common_shape = None
        self.device = None
        self.sum_blocks = 0
        for loaded in acquisitions:
            # A loaded dataset records its pixel mask; a raw source carries no metadata.
            if isinstance(loaded, Dataset4dstemGPU):
                source, metadata = loaded.data, loaded.metadata
            else:
                source, metadata = loaded, {}
            shape = tuple(source.shape)
            if len(shape) != 4 or any(size < 1 for size in shape):
                raise ValueError(
                    f"Each acquisition must have a complete 4D shape; got {shape}."
                )
            if common_shape is not None and shape != common_shape:
                raise ValueError(
                    f"Linked acquisitions must have the same shape; got {shape} and "
                    f"{common_shape}. Open differently shaped acquisitions separately."
                )
            common_shape = shape
            dtype = np.dtype(source.dtype)
            if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
                raise TypeError(
                    f"Joint native counts require uint8 or uint16; got {dtype}."
                )
            source_dtypes.append(dtype)
            descriptor = np.zeros(20, np.uint64)
            descriptor[1] = dtype.itemsize
            if isinstance(source, cp.ndarray):
                if not source.flags.c_contiguous:
                    raise ValueError(
                        "Native acquisition storage must be contiguous; load it with io.load."
                    )
                device = source.device.id
                arrays = [source]
            elif isinstance(source, counts.StreamedCounts):
                if source.is_released or source.ready_scans != int(np.prod(shape[:2])):
                    raise ValueError("The streamed acquisition must be completely loaded.")
                device = source.device
                descriptor[0] = 4
                descriptor[2] = source.interval
                chunk_rows, first_stream = [], 0
                for chunk in source.chunks:
                    payload, offsets, models = chunk.arrays[:3]
                    chunk_rows.append([chunk.first, chunk.scans, first_stream,
                                       payload.data.ptr, offsets.data.ptr, models.data.ptr])
                    first_stream += ((chunk.scans + source.interval - 1) // source.interval) * int(np.prod(shape[2:]))
                    self.keepalive.extend(chunk.arrays)
                with cp.cuda.Device(device):
                    pointers = cp.asarray(chunk_rows, dtype=cp.uint64)
                arrays = [pointers, source.decoding]
                descriptor[11] = len(chunk_rows)
                descriptor[12] = first_stream
            else:
                raise TypeError(
                    "Joint native queries require CUDA sources returned by io.load."
                )
            if self.device is not None and device != self.device:
                raise ValueError(
                    "All linked acquisitions must be resident on the same CUDA device."
                )
            self.device = device
            self.keepalive.extend(arrays)
            descriptor[9 : 9 + len(arrays)] = [array.data.ptr for array in arrays]
            valid = np.ones(shape[2:], np.bool_)
            pixel_mask = metadata.get("pixel_mask")
            if pixel_mask is not None and not correction_is_applied(metadata):
                if np.shape(pixel_mask) != shape[2:]:
                    raise ValueError(
                        "Stored pixel mask does not match the complete detector shape."
                    )
                valid &= np.asarray(pixel_mask) == 0
            excluded = metadata.get("excluded_detector_pixels", ())
            if len(excluded):
                valid.reshape(-1)[np.asarray(excluded, dtype=np.intp)] = False
            if isinstance(source, counts.StreamedCounts):
                valid &= source.valid_pixels
            valid_masks.append(valid)
            # Dense sources sum one scan per block; streamed sources decode 256 streams per block.
            blocks = int(np.prod(shape[:2]))
            if descriptor[0] == 4:
                blocks = (int(descriptor[12]) + 255) // 256
            self.sum_blocks = max(self.sum_blocks, blocks)
            rows.append(descriptor)
        self.series_shape = (len(acquisitions),)
        self.scan_shape, self.det_shape = common_shape[:2], common_shape[2:]
        self.n_frames = int(np.prod(self.scan_shape))
        self.pixels = int(np.prod(self.det_shape))
        self.frame_dtype = np.result_type(*source_dtypes)
        maximum = self.pixels * int(np.iinfo(self.frame_dtype).max)
        if maximum * len(acquisitions) > np.iinfo(np.uint64).max:
            raise OverflowError(
                "This series exceeds the exact uint64 mean-numerator bound."
            )
        self.sum_dtype = np.dtype(
            "uint32" if maximum <= np.iinfo(np.uint32).max else "uint64"
        )
        self.valid_pixels = np.asarray(valid_masks)
        with cp.cuda.Device(self.device):
            self.valid = cp.asarray(self.valid_pixels, dtype=cp.uint8)
            for index, descriptor in enumerate(rows):
                descriptor[18] = self.valid.data.ptr + index * self.pixels
            self.descriptors = cp.asarray(np.asarray(rows))
            self.mask = cp.empty(self.det_shape, cp.uint8)
            self.errors = cp.zeros(1, cp.uint32)
            self.begin, self.end = cp.cuda.Event(), cp.cuda.Event()
            self.kernels = _kernels(self.device)
            cp.cuda.get_current_stream().synchronize()
        self.keepalive.extend((self.valid, self.descriptors, self.mask, self.errors))
        regions = {(a.data.ptr, a.data.ptr + a.nbytes) for a in self.keepalive}
        self.regions = np.asarray(sorted(regions), dtype=np.uint64)
        self.backend_metadata = {
            "backend": "cuda",
            "device": f"cuda:{self.device}",
            "query_abi": "native-count-series-v1",
            "frame_dtype": self.frame_dtype.name,
            "sum_dtype": self.sum_dtype.name,
            "series_shape": self.series_shape,
            "resident_bytes": sum(end - start for start, end in regions),
        }
        self.lock = threading.Lock()
        self.last = {}

    def frame(self, index):
        """Host copy of the scan position's native counts for every acquisition."""
        return self.frame_native(index).get()

    def masked_sum(self, mask):
        """Exact counts of a binary mask; the session converts them to float32 for display."""
        return self.masked_sum_native(mask)

    def masked_sum_exact(self, mask):
        """Exact counts of a binary mask, every acquisition in one launch."""
        return self.masked_sum_native(mask)

    def masked_sum_exact_native(self, mask, *, out=None):
        """Exact counts in the series' sum dtype (uint32 or uint64), left on the device."""
        return self.masked_sum_native(mask, out=out)

    def mean_dp(self):
        raise NotImplementedError(
            "Joint series currently supports frame and binary masked_sum; select an acquisition for mean_dp."
        )

    def reduce_frames(self, indices, reduce="mean"):
        raise NotImplementedError(
            "Joint scan ROI reductions are not implemented; select an acquisition for reduce_frames."
        )

    def center_of_mass(self, mask):
        raise NotImplementedError(
            "Joint center_of_mass is not implemented; select an acquisition for this calculation."
        )

    def _output(self, out, shape, dtype):
        """Validate a caller buffer: a mismatched one would be overrun, and an overlapping one would corrupt the sources."""
        if out is None:
            return cp.empty(shape, dtype)
        if (
            not isinstance(out, cp.ndarray)
            or out.shape != shape
            or out.dtype != dtype
            or out.device.id != self.device
            or not out.flags.c_contiguous
        ):
            raise ValueError(
                f"out must be a contiguous {dtype}{shape} array on cuda:{self.device}."
            )
        first, stop = out.data.ptr, out.data.ptr + out.nbytes
        if np.any((first < self.regions[:, 1]) & (stop > self.regions[:, 0])):
            raise ValueError("out must not overlap any resident source or query index.")
        return out

    def _complete(self, started):
        """Wait for the query and refuse its result if any stream failed exact decoding."""
        self.end.record()
        self.end.synchronize()
        if int(self.errors.get()[0]):
            raise ValueError(
                "An encoded stream failed exact decoding; this result is not ready."
            )
        self.last = {
            "gpu_ms": float(cp.cuda.get_elapsed_time(self.begin, self.end)),
            "wall_ms": (time.perf_counter() - started) * 1000,
            "acquisitions": len(self.owners),
            "query_launches": 1,
        }

    def masked_sum_native(self, mask, *, out=None):
        """Return exact counts for a binary mask, every acquisition in one launch."""
        values = np.asarray(mask)
        if values.shape != self.det_shape or not np.all((values == 0) | (values == 1)):
            raise ValueError(
                f"Provide a binary detector mask with shape {self.det_shape}."
            )
        with self.lock, cp.cuda.Device(self.device):
            started = time.perf_counter()
            result = self._output(
                out, (*self.series_shape, *self.scan_shape), self.sum_dtype
            )
            self.begin.record()
            self.mask.set(values.astype(np.uint8))
            self.errors.fill(0)
            result.fill(0)
            self.kernels[f"series_sum_u{self.sum_dtype.itemsize * 8}"](
                (self.sum_blocks, self.series_shape[0]),
                (256,),
                (
                    self.descriptors,
                    self.mask,
                    result,
                    self.errors,
                    np.uint64(self.n_frames),
                    np.uint32(self.pixels),
                ),
            )
            self._complete(started)
            return result

    def frame_native(self, index, *, out=None):
        """Return the scan position's native counts for every acquisition in one launch."""
        index = int(index)
        if not 0 <= index < self.n_frames:
            raise IndexError(f"Choose a scan index inside {self.scan_shape}.")
        with self.lock, cp.cuda.Device(self.device):
            started = time.perf_counter()
            result = self._output(
                out, (*self.series_shape, *self.det_shape), self.frame_dtype
            )
            self.begin.record()
            self.errors.fill(0)
            self.kernels[f"series_frame_u{self.frame_dtype.itemsize * 8}"](
                ((self.pixels + 255) // 256, self.series_shape[0]),
                (256,),
                (
                    self.descriptors,
                    result,
                    self.errors,
                    np.uint64(index),
                    np.uint32(self.pixels),
                ),
            )
            self._complete(started)
            return result
