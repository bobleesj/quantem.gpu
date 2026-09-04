"""Exact compact-HDF5 residency and detector interactions on CUDA.

The implementation dispatches the shared QuantEM v1 raw-LZ4 and v3 direct
bit-packed formats without conflating their layouts. It never materializes the
logical ``(scan_row, scan_column, detector_row, detector_column)`` tensor. The
authenticated bit-packed payload remains resident for selected-diffraction and
virtual-detector requests.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quantem.gpu.io._compact_h5 import (
    CompactH5Index,
    CompactH5ReferenceDecoder,
    CompactH5Shard,
)

try:
    import cupy as cp
except ImportError:  # pragma: no cover - exercised on non-CUDA hosts
    cp = None


_CUDA_COMPACT_SOURCE = r"""
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

struct CompactDetectorEntry {
    uint32_t pixel;
    int coefficient;
};

__device__ __forceinline__ uint32_t compact_width(
    const uint32_t* headers,
    uint32_t header_base,
    uint32_t checkpoint_words,
    uint32_t tile
) {
    const uint32_t packed = headers[
        header_base + checkpoint_words + tile / 8u
    ];
    return (packed >> ((tile % 8u) * 4u)) & 15u;
}

__device__ __forceinline__ uint32_t sum_width_nibbles(
    uint32_t packed,
    uint32_t count
) {
    const uint32_t mask = count >= 8u
        ? 0xffffffffu
        : (count == 0u ? 0u : (1u << (count * 4u)) - 1u);
    packed &= mask;
    const uint32_t bytes =
        (packed & 0x0f0f0f0fu) + ((packed >> 4u) & 0x0f0f0f0fu);
    return (bytes & 0xffu)
        + ((bytes >> 8u) & 0xffu)
        + ((bytes >> 16u) & 0xffu)
        + ((bytes >> 24u) & 0xffu);
}

__device__ __forceinline__ uint32_t compact_descriptor(
    const uint32_t* headers,
    uint32_t tile_count,
    uint32_t header_encoding,
    uint32_t header_words_per_pixel,
    uint32_t pixel,
    uint32_t tile
) {
    if (header_encoding == 0u) {
        return headers[pixel * tile_count + tile];
    }
    const uint32_t checkpoint_words = (tile_count + 31u) / 32u;
    const uint32_t header_base = pixel * header_words_per_pixel;
    const uint32_t checkpoint = tile / 32u;
    uint32_t offset = headers[header_base];
    if (checkpoint != 0u) {
        offset += headers[header_base + checkpoint];
    }
    const uint32_t first_width_word = checkpoint * 4u;
    const uint32_t tile_width_word = tile / 8u;
    for (uint32_t word = first_width_word; word < tile_width_word; ++word) {
        offset += sum_width_nibbles(
            headers[header_base + checkpoint_words + word], 8u
        );
    }
    const uint32_t packed = headers[
        header_base + checkpoint_words + tile_width_word
    ];
    offset += sum_width_nibbles(packed, tile % 8u);
    const uint32_t width = (packed >> ((tile % 8u) * 4u)) & 15u;
    return (offset << 5u) | width;
}

__device__ __forceinline__ uint32_t compact_sample(
    const uint32_t* payload,
    const uint32_t* headers,
    uint32_t tile_count,
    uint32_t header_encoding,
    uint32_t header_words_per_pixel,
    uint32_t scan_tile,
    uint32_t pixel,
    uint32_t scan
) {
    const uint32_t descriptor = compact_descriptor(
        headers,
        tile_count,
        header_encoding,
        header_words_per_pixel,
        pixel,
        scan / scan_tile
    );
    const uint32_t width = descriptor & 31u;
    if (width == 0u) return 0u;
    const uint32_t bit = (scan % scan_tile) * width;
    const uint32_t index = (descriptor >> 5u) + bit / 32u;
    const uint32_t shift = bit & 31u;
    uint32_t value = payload[index] >> shift;
    if (shift + width > 32u) {
        value |= payload[index + 1u] << (32u - shift);
    }
    return value & ((1u << width) - 1u);
}

extern "C" __global__ void compact_h5_lz4_decode(
    const uint8_t* compressed,
    const uint32_t* input_offsets,
    uint8_t* decoded,
    uint32_t decoded_bytes,
    uint32_t chunk_bytes,
    uint32_t chunk_count,
    uint32_t* status
) {
    const uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count) return;
    uint32_t input = input_offsets[chunk];
    const uint32_t input_end = input_offsets[chunk + 1u];
    const uint32_t output_start = chunk * chunk_bytes;
    const uint32_t expected = decoded_bytes - output_start < chunk_bytes
        ? decoded_bytes - output_start
        : chunk_bytes;
    uint32_t output = 0u;
    uint32_t error = input >= input_end || expected == 0u ? 1u : 0u;
    uint32_t token_count = 0u;
    while (error == 0u && input < input_end && output < expected) {
        if (++token_count > expected) {
            error = 2u;
            break;
        }
        const uint32_t token = compressed[input++];
        uint32_t literal_count = token >> 4u;
        if (literal_count == 15u) {
            uint32_t extension = 255u;
            while (extension == 255u) {
                if (input >= input_end) {
                    error = 3u;
                    break;
                }
                extension = compressed[input++];
                if (literal_count > 0xffffffffu - extension) {
                    error = 3u;
                    break;
                }
                literal_count += extension;
            }
        }
        if (error != 0u) break;
        if (literal_count > input_end - input || literal_count > expected - output) {
            error = 4u;
            break;
        }
        for (uint32_t i = 0u; i < literal_count; ++i) {
            decoded[output_start + output + i] = compressed[input + i];
        }
        input += literal_count;
        output += literal_count;
        if (output == expected || input == input_end) break;
        if (input_end - input < 2u) {
            error = 5u;
            break;
        }
        const uint32_t match_offset = (uint32_t)compressed[input]
            | ((uint32_t)compressed[input + 1u] << 8u);
        input += 2u;
        if (match_offset == 0u || match_offset > output) {
            error = 6u;
            break;
        }
        uint32_t match_count = token & 15u;
        if (match_count == 15u) {
            uint32_t extension = 255u;
            while (extension == 255u) {
                if (input >= input_end) {
                    error = 7u;
                    break;
                }
                extension = compressed[input++];
                if (match_count > 0xfffffffbu - extension) {
                    error = 7u;
                    break;
                }
                match_count += extension;
            }
        }
        if (error != 0u) break;
        match_count += 4u;
        if (match_count > expected - output) {
            error = 8u;
            break;
        }
        for (uint32_t i = 0u; i < match_count; ++i) {
            decoded[output_start + output] =
                decoded[output_start + output - match_offset];
            ++output;
        }
    }
    status[chunk] = error != 0u
        ? error
        : (input == input_end && output == expected ? 0u : 9u);
}

extern "C" __global__ void compact_h5_validate_descriptors(
    const uint32_t* descriptors,
    uint32_t descriptor_count,
    uint32_t payload_words,
    uint32_t* status
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= descriptor_count) return;
    const uint32_t descriptor = descriptors[index];
    const uint32_t width = descriptor & 31u;
    const uint32_t offset = descriptor >> 5u;
    uint32_t error = 0u;
    if (width > 16u) error |= 1u;
    if (offset >= (1u << 27u)) error |= 2u;
    const uint32_t expected_next = offset + width * 4u;
    if (index + 1u < descriptor_count) {
        if ((descriptors[index + 1u] >> 5u) != expected_next) error |= 4u;
    } else if (expected_next != payload_words) {
        error |= 8u;
    }
    if (error != 0u) atomicOr(status, error);
}

extern "C" __global__ void compact_h5_validate_compact_headers(
    const uint32_t* headers,
    const uint32_t* excluded,
    uint32_t pixel_count,
    uint32_t tile_count,
    uint32_t header_words_per_pixel,
    uint32_t payload_words,
    uint32_t* status
) {
    const uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const uint32_t checkpoint_words = (tile_count + 31u) / 32u;
    const uint32_t width_words = (tile_count + 7u) / 8u;
    const uint32_t expected_words = checkpoint_words + width_words;
    if (header_words_per_pixel != expected_words) {
        atomicOr(status, 1u);
        return;
    }
    const uint32_t base = pixel * expected_words;
    const uint32_t pixel_base = headers[base];
    uint32_t relative_word = 0u;
    uint32_t errors = pixel == 0u && pixel_base != 0u ? 2u : 0u;
    for (uint32_t checkpoint = 0u; checkpoint < checkpoint_words; ++checkpoint) {
        if (checkpoint != 0u && headers[base + checkpoint] != relative_word) {
            errors |= 4u;
        }
        const uint32_t first_width_word = checkpoint * 4u;
        const uint32_t words_in_checkpoint = width_words - first_width_word < 4u
            ? width_words - first_width_word
            : 4u;
        for (uint32_t word = 0u; word < words_in_checkpoint; ++word) {
            const uint32_t packed = headers[
                base + checkpoint_words + first_width_word + word
            ];
            const uint32_t first_tile = (first_width_word + word) * 8u;
            const uint32_t remaining_tiles = tile_count - first_tile;
            const uint32_t width_count = remaining_tiles < 8u
                ? remaining_tiles
                : 8u;
            for (uint32_t nibble = 0u; nibble < width_count; ++nibble) {
                const uint32_t width = (packed >> (nibble * 4u)) & 15u;
                if (width > 8u) errors |= 8u;
                if (excluded[pixel] != 0u && width != 0u) errors |= 16u;
                relative_word += width;
            }
            if (width_count < 8u && (packed >> (width_count * 4u)) != 0u) {
                errors |= 32u;
            }
        }
    }
    bool coverage_invalid = pixel_base > payload_words
        || relative_word > payload_words - (
            pixel_base < payload_words ? pixel_base : payload_words
        );
    if (!coverage_invalid) {
        const uint32_t expected_next = pixel_base + relative_word;
        if (pixel + 1u < pixel_count) {
            const uint32_t next_base = headers[(pixel + 1u) * expected_words];
            coverage_invalid = next_base != expected_next;
        } else {
            coverage_invalid = expected_next != payload_words;
        }
    }
    if (coverage_invalid) errors |= 64u;
    if (errors != 0u) atomicOr(status, errors);
}

extern "C" __global__ void compact_h5_selected_diffraction(
    const uint32_t* payload,
    const uint32_t* headers,
    const uint32_t* excluded,
    uint32_t* diffraction,
    uint32_t scan,
    uint32_t tile_count,
    uint32_t header_encoding,
    uint32_t header_words_per_pixel,
    uint32_t scan_tile,
    uint32_t pixel_count
) {
    const uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    diffraction[pixel] = excluded[pixel] != 0u
        ? 0u
        : compact_sample(
            payload,
            headers,
            tile_count,
            header_encoding,
            header_words_per_pixel,
            scan_tile,
            pixel,
            scan
        );
}

extern "C" __global__ void compact_h5_detector_update(
    const uint32_t* payload,
    const uint32_t* headers,
    const CompactDetectorEntry* entries,
    const uint32_t* previous,
    uint32_t* next,
    uint32_t scan_count,
    uint32_t tile_count,
    uint32_t header_encoding,
    uint32_t header_words_per_pixel,
    uint32_t scan_tile,
    uint32_t entry_count,
    uint32_t output_offset,
    uint32_t rebase,
    uint32_t* status
) {
    const uint32_t scan_lane = threadIdx.x & 31u;
    const uint32_t entry_lane = threadIdx.x >> 5u;
    const uint32_t scan = blockIdx.x * 32u + scan_lane;
    int64_t partial = 0;
    if (scan < scan_count) {
        for (uint32_t index = entry_lane; index < entry_count; index += 4u) {
            const CompactDetectorEntry entry = entries[index];
            const uint32_t value = compact_sample(
                payload,
                headers,
                tile_count,
                header_encoding,
                header_words_per_pixel,
                scan_tile,
                entry.pixel,
                scan
            );
            partial += (int64_t)entry.coefficient * (int64_t)value;
        }
    }
    __shared__ int64_t partials[128];
    partials[threadIdx.x] = partial;
    __syncthreads();
    if (entry_lane == 0u && scan < scan_count) {
        int64_t output = rebase != 0u ? 0 : (int64_t)previous[output_offset + scan];
        output += partials[scan_lane];
        output += partials[scan_lane + 32u];
        output += partials[scan_lane + 64u];
        output += partials[scan_lane + 96u];
        if (output < 0 || output > 0xffffffffll) {
            atomicOr(status, 1u);
        } else {
            next[output_offset + scan] = (uint32_t)output;
        }
    }
}
"""


_CUDA_FUNCTION_NAMES = (
    "compact_h5_lz4_decode",
    "compact_h5_validate_descriptors",
    "compact_h5_validate_compact_headers",
    "compact_h5_selected_diffraction",
    "compact_h5_detector_update",
)
_CUDA_MODULES: dict[int, dict[str, Any]] = {}


@dataclass(frozen=True)
class CudaCompactH5LoadMetrics:
    """Measured phases for one successful compact CUDA-resident load."""

    metadata_ms: float
    whole_file_integrity_ms: float
    nvrtc_compile_ms: float
    source_read_ms: float
    host_header_validation_ms: float
    host_payload_integrity_ms: float
    descriptor_preparation_ms: float
    gpu_upload_ms: float
    gpu_header_validation_ms: float
    gpu_decode_ms: float
    decoded_integrity_ms: float
    total_ms: float
    resident_bytes: int
    maximum_transient_bytes: int
    device_free_bytes_before: int
    device_free_bytes_after: int
    memory_pool_used_bytes: int
    decoded_shard_sha256_checks: int
    direct_payload_sha256_checks: int


@dataclass(frozen=True)
class CudaCompactH5DetectorMetrics:
    """One exact resident virtual-detector update."""

    mode: str
    changed_detector_pixels: int
    wall_ms: float
    gpu_ms: float
    fft_dispatch_count: int = 0


@dataclass
class _CudaCompactShard:
    payload: Any
    headers: Any


class CudaCompactH5ResidentSource:
    """Exact detector operations over compact CUDA-resident shard buffers."""

    def __init__(
        self,
        *,
        index: CompactH5Index,
        load_metrics: CudaCompactH5LoadMetrics,
        shards: list[_CudaCompactShard],
        excluded: Any,
        maximum_widths: np.ndarray,
        detector_outputs: list[Any],
        diffraction_output: Any,
        memory_pool: Any,
        kernels: dict[str, Any],
    ) -> None:
        self.metadata = index
        self.load_metrics = load_metrics
        self._shards = shards
        self._excluded = excluded
        self._maximum_widths = maximum_widths
        self._detector_outputs = detector_outputs
        self._diffraction_output = diffraction_output
        self._memory_pool = memory_pool
        self._kernels = kernels
        self._active_detector_output = 0
        self._detector_mask = np.zeros(index.shape[2] * index.shape[3], np.uint8)
        self._has_detector = False
        self.is_released = False

    def extract_diffraction_device(self, scan_row: int, scan_column: int):
        """Return a device-resident exact u32 diffraction pattern.

        The returned array is a reusable output owned by this source.  Consume
        or copy it before the next selected-diffraction request.
        """
        self._require_resident()
        scan_rows, scan_columns, detector_rows, detector_columns = self.metadata.shape
        if not 0 <= scan_row < scan_rows or not 0 <= scan_column < scan_columns:
            raise IndexError(
                f"Scan (row: {scan_row}, column: {scan_column}) is outside "
                f"shape ({scan_rows}, {scan_columns})."
            )
        global_scan = scan_row * scan_columns + scan_column
        shard_index, local_scan = divmod(global_scan, self.metadata.scans_per_shard)
        pixel_count = detector_rows * detector_columns
        tile_count = (
            self.metadata.scans_per_shard + self.metadata.scan_tile - 1
        ) // self.metadata.scan_tile
        header_words_per_pixel = _header_words_per_pixel(self.metadata, tile_count)
        kernel = self._kernels["compact_h5_selected_diffraction"]
        kernel(
            ((pixel_count + 255) // 256,),
            (256,),
            (
                self._shards[shard_index].payload,
                self._shards[shard_index].headers,
                self._excluded,
                self._diffraction_output,
                np.uint32(local_scan),
                np.uint32(tile_count),
                np.uint32(self.metadata.header_encoding),
                np.uint32(header_words_per_pixel),
                np.uint32(self.metadata.scan_tile),
                np.uint32(pixel_count),
            ),
        )
        return self._diffraction_output

    def extract_diffraction(self, scan_row: int, scan_column: int) -> np.ndarray:
        """Return one complete exact row-major diffraction pattern on the host."""
        return cp.asnumpy(
            self.extract_diffraction_device(scan_row, scan_column)
        ).reshape(self.metadata.shape[2:])

    def update_virtual_detector(self, mask: np.ndarray) -> CudaCompactH5DetectorMetrics:
        """Apply a row-major zero-or-one detector mask without dense expansion."""
        self._require_resident()
        pixel_count = self.metadata.shape[2] * self.metadata.shape[3]
        values = np.asarray(mask)
        if values.size != pixel_count:
            raise ValueError(
                "A compact virtual-detector mask requires exactly "
                f"{pixel_count} row-major values; got {values.size}."
            )
        values = values.reshape(-1)
        if not np.issubdtype(values.dtype, np.bool_) and not np.issubdtype(
            values.dtype, np.integer
        ):
            raise TypeError("A compact virtual-detector mask must contain integers.")
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(
                "A compact virtual-detector mask must contain only zero or one."
            )
        normalized = values.astype(np.uint8, copy=True)
        if self.metadata.excluded_detector_pixels:
            normalized[list(self.metadata.excluded_detector_pixels)] = 0
        selected_widths = self._maximum_widths[normalized != 0].astype(np.uint64)
        maximum_sum = int(np.sum((np.uint64(1) << selected_widths) - 1))
        if maximum_sum > np.iinfo(np.uint32).max:
            raise OverflowError(
                f"This detector can sum to {maximum_sum}, beyond exact u32 output. "
                "Use a narrower detector or a future u64 reduction path."
            )

        is_rebase = not self._has_detector
        if is_rebase:
            changed = np.flatnonzero(normalized)
            coefficients = np.ones(changed.size, dtype=np.int32)
        else:
            changed = np.flatnonzero(normalized != self._detector_mask)
            coefficients = np.where(normalized[changed] != 0, 1, -1).astype(np.int32)
        if changed.size == 0:
            self._detector_mask = normalized
            self._has_detector = True
            return CudaCompactH5DetectorMetrics(
                mode="rebase" if is_rebase else "delta",
                changed_detector_pixels=0,
                wall_ms=0.0,
                gpu_ms=0.0,
            )

        packed_entries = np.empty((changed.size, 2), dtype=np.uint32)
        packed_entries[:, 0] = changed.astype(np.uint32, copy=False)
        packed_entries[:, 1] = coefficients.view(np.uint32)
        next_output = 1 - self._active_detector_output
        tile_count = (
            self.metadata.scans_per_shard + self.metadata.scan_tile - 1
        ) // self.metadata.scan_tile
        header_words_per_pixel = _header_words_per_pixel(self.metadata, tile_count)
        started = time.perf_counter()
        with cp.cuda.using_allocator(self._memory_pool.malloc):
            entries = cp.asarray(packed_entries)
            status = cp.zeros(1, dtype=cp.uint32)
            begin = cp.cuda.Event()
            end = cp.cuda.Event()
            begin.record()
            for shard_index, shard in enumerate(self._shards):
                self._kernels["compact_h5_detector_update"](
                    ((self.metadata.scans_per_shard + 31) // 32,),
                    (128,),
                    (
                        shard.payload,
                        shard.headers,
                        entries,
                        self._detector_outputs[self._active_detector_output],
                        self._detector_outputs[next_output],
                        np.uint32(self.metadata.scans_per_shard),
                        np.uint32(tile_count),
                        np.uint32(self.metadata.header_encoding),
                        np.uint32(header_words_per_pixel),
                        np.uint32(self.metadata.scan_tile),
                        np.uint32(changed.size),
                        np.uint32(shard_index * self.metadata.scans_per_shard),
                        np.uint32(1 if is_rebase else 0),
                        status,
                    ),
                )
            end.record()
            end.synchronize()
            gpu_ms = float(cp.cuda.get_elapsed_time(begin, end))
            error = int(status.get()[0])
        if error:
            raise RuntimeError(
                "CUDA compact virtual-detector update exceeded its exact u32 "
                "publication contract; the previous result remains active."
            )
        self._active_detector_output = next_output
        self._detector_mask = normalized
        self._has_detector = True
        return CudaCompactH5DetectorMetrics(
            mode="rebase" if is_rebase else "delta",
            changed_detector_pixels=int(changed.size),
            wall_ms=(time.perf_counter() - started) * 1_000.0,
            gpu_ms=gpu_ms,
        )

    def virtual_detector_values_device(self):
        """Return the last completely published device-resident u32 scan map."""
        self._require_resident()
        if not self._has_detector:
            raise RuntimeError("Run update_virtual_detector before reading a result.")
        return self._detector_outputs[self._active_detector_output]

    def virtual_detector_values(self) -> np.ndarray:
        """Copy the last complete detector map to a host row-column array."""
        return cp.asnumpy(self.virtual_detector_values_device()).reshape(
            self.metadata.shape[:2]
        )

    @property
    def memory_pool_used_bytes(self) -> int:
        """Return bytes currently owned by this source's isolated CUDA pool."""
        return int(self._memory_pool.used_bytes())

    def release_resident_storage(self) -> None:
        """Release this source's private CUDA pool and invalidate interactions."""
        if self.is_released:
            return
        self._shards.clear()
        self._detector_outputs.clear()
        self._excluded = None
        self._diffraction_output = None
        self._maximum_widths = np.empty(0, dtype=np.uint8)
        self._detector_mask = np.empty(0, dtype=np.uint8)
        self.is_released = True
        self._memory_pool.free_all_blocks()

    def _require_resident(self) -> None:
        if self.is_released:
            raise RuntimeError(
                "The compact CUDA source has been released. Load it again before "
                "requesting scientific output."
            )


def load_compact_h5_cuda(
    path: str | Path,
    *,
    expected_whole_file_sha256: str | None = None,
    should_cancel: Callable[[], bool] = lambda: False,
) -> CudaCompactH5ResidentSource:
    """Decode, authenticate, and publish a compact source in CUDA buffers.

    Parameters
    ----------
    path
        QuantEM compact v1 or portable v3 HDF5 file.
    expected_whole_file_sha256
        Required for QGIX v3 because embedded records authenticate direct
        payloads but not compact headers. Optional as an additional v1 identity
        check.
    should_cancel
        Callback checked before each shard.  A cancelled load publishes no
        partial source.

    Returns
    -------
    CudaCompactH5ResidentSource
        Exact bit-packed resident source and detector operations.
    """
    if cp is None:
        raise RuntimeError(
            "Compact CUDA loading requires CuPy and an NVIDIA CUDA device. "
            "Install a matching cupy-cuda package or use the Metal/WebGPU adapter."
        )
    total_start = time.perf_counter()
    metadata_start = time.perf_counter()
    index = CompactH5Index.from_file(path)
    metadata_ms = (time.perf_counter() - metadata_start) * 1_000.0
    if index.schema_version not in {1, 3}:
        raise ValueError(
            f"CUDA compact loading does not support QGIX v{index.schema_version}."
        )
    whole_file_integrity_ms = 0.0
    if index.schema_version == 3:
        index.require_raw_reconstruction()
        if (
            not isinstance(expected_whole_file_sha256, str)
            or len(expected_whole_file_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_whole_file_sha256
            )
        ):
            raise ValueError(
                "QGIX v3 CUDA loading requires the externally sealed lowercase "
                "expected_whole_file_sha256."
            )
    if expected_whole_file_sha256 is not None:
        integrity_started = time.perf_counter()
        observed_whole_file_sha256 = _sha256_file(index.path)
        whole_file_integrity_ms = (time.perf_counter() - integrity_started) * 1_000.0
        if observed_whole_file_sha256 != expected_whole_file_sha256:
            raise ValueError(
                f"Compact HDF5 whole-file SHA-256 is {observed_whole_file_sha256}, "
                f"expected {expected_whole_file_sha256}."
            )
    if should_cancel():
        raise RuntimeError("Compact CUDA load was cancelled before allocation.")
    if index.schema_version == 3:
        return _load_compact_h5_cuda_v3(
            index,
            total_start=total_start,
            metadata_ms=metadata_ms,
            whole_file_integrity_ms=whole_file_integrity_ms,
            should_cancel=should_cancel,
        )

    device = cp.cuda.Device()
    free_before, _ = cp.cuda.runtime.memGetInfo()
    compile_start = time.perf_counter()
    kernels, compiled_now = _cuda_kernels(device.id)
    nvrtc_compile_ms = (
        (time.perf_counter() - compile_start) * 1_000.0 if compiled_now else 0.0
    )
    pool = cp.cuda.MemoryPool()
    resident_shards: list[_CudaCompactShard] = []
    detector_pixel_count = index.shape[2] * index.shape[3]
    maximum_widths = np.zeros(detector_pixel_count, dtype=np.uint8)
    source_read_ms = 0.0
    descriptor_ms = 0.0
    gpu_decode_ms = 0.0
    integrity_ms = 0.0
    maximum_transient_bytes = 0
    descriptor_threads = 256
    file_descriptor = os.open(index.path, os.O_RDONLY)
    try:
        with (
            cp.cuda.using_allocator(pool.malloc),
            ThreadPoolExecutor(max_workers=1) as read_executor,
        ):
            pending_read = read_executor.submit(
                _read_shard_envelope,
                file_descriptor,
                index.shards[0],
                0,
            )
            for shard_index, shard in enumerate(index.shards):
                if should_cancel():
                    raise RuntimeError(
                        f"Compact CUDA load was cancelled before shard {shard_index}."
                    )
                envelope, range_start, read_ms = pending_read.result()
                if shard_index + 1 < len(index.shards):
                    pending_read = read_executor.submit(
                        _read_shard_envelope,
                        file_descriptor,
                        index.shards[shard_index + 1],
                        shard_index + 1,
                    )
                envelope_view = memoryview(envelope)
                payload_start = shard.payload_offset - range_start
                length_start = shard.lengths_offset - range_start
                width_start = shard.widths_offset - range_start
                payload_bytes = envelope_view[
                    payload_start : payload_start + shard.payload_bytes
                ]
                length_bytes = envelope_view[
                    length_start : length_start + shard.lengths_bytes
                ]
                width_bytes = envelope_view[
                    width_start : width_start + shard.widths_bytes
                ]
                source_read_ms += read_ms

                preparation_start = time.perf_counter()
                widths = np.frombuffer(width_bytes, dtype=np.uint8)
                lengths = np.frombuffer(length_bytes, dtype=np.uint8)
                payload_words = _validate_widths(
                    index, shard, shard_index, widths, maximum_widths
                )
                compressed_offsets = np.empty(lengths.size + 1, dtype=np.uint32)
                compressed_offsets[0] = 0
                np.cumsum(lengths.astype(np.uint32) + 1, out=compressed_offsets[1:])
                if int(compressed_offsets[-1]) != shard.payload_bytes:
                    raise ValueError(
                        f"Compact shard {shard_index} chunk lengths cover "
                        f"{int(compressed_offsets[-1])} bytes, expected "
                        f"{shard.payload_bytes}."
                    )
                compressed = cp.asarray(np.frombuffer(payload_bytes, dtype=np.uint8))
                input_offsets = cp.asarray(compressed_offsets)
                widths_device = cp.asarray(widths)
                word_offsets = cp.empty(widths.size, dtype=cp.uint32)
                word_offsets[0] = 0
                if widths.size > 1:
                    word_offsets[1:] = cp.cumsum(
                        widths_device[:-1], dtype=cp.uint32
                    ) * np.uint32(4)
                descriptors = (word_offsets << np.uint32(5)) | widths_device.astype(
                    cp.uint32
                )
                decoded = cp.empty(shard.decoded_bytes, dtype=cp.uint8)
                decode_status = cp.empty(shard.chunk_count, dtype=cp.uint32)
                descriptor_status = cp.zeros(1, dtype=cp.uint32)
                descriptor_ms += (time.perf_counter() - preparation_start) * 1_000.0
                maximum_transient_bytes = max(
                    maximum_transient_bytes,
                    len(payload_bytes)
                    + len(length_bytes)
                    + len(width_bytes)
                    + int(shard.decoded_bytes)
                    + int(shard.descriptor_count) * 4
                    + int(shard.chunk_count + 1) * 4,
                )

                begin = cp.cuda.Event()
                end = cp.cuda.Event()
                begin.record()
                kernels["compact_h5_lz4_decode"](
                    ((shard.chunk_count + 255) // 256,),
                    (256,),
                    (
                        compressed,
                        input_offsets,
                        decoded,
                        np.uint32(shard.decoded_bytes),
                        np.uint32(index.payload_chunk_bytes),
                        np.uint32(shard.chunk_count),
                        decode_status,
                    ),
                )
                kernels["compact_h5_validate_descriptors"](
                    (
                        (shard.descriptor_count + descriptor_threads - 1)
                        // descriptor_threads,
                    ),
                    (descriptor_threads,),
                    (
                        descriptors,
                        np.uint32(shard.descriptor_count),
                        np.uint32(payload_words),
                        descriptor_status,
                    ),
                )
                end.record()
                end.synchronize()
                gpu_decode_ms += float(cp.cuda.get_elapsed_time(begin, end))
                failed_chunks = np.flatnonzero(cp.asnumpy(decode_status))
                if failed_chunks.size:
                    chunk = int(failed_chunks[0])
                    code = int(decode_status[chunk].get())
                    raise ValueError(
                        f"Compact shard {shard_index} raw LZ4 chunk {chunk} "
                        f"failed with decoder status {code}."
                    )
                descriptor_error = int(descriptor_status.get()[0])
                if descriptor_error:
                    raise ValueError(
                        f"Compact shard {shard_index} failed CUDA descriptor "
                        f"coverage validation with status {descriptor_error}."
                    )

                integrity_start = time.perf_counter()
                decoded_host = cp.asnumpy(decoded)
                digest = hashlib.sha256(decoded_host).hexdigest()
                if digest != shard.decoded_sha256:
                    raise ValueError(
                        f"Compact shard {shard_index} decoded SHA-256 is {digest}, "
                        f"expected {shard.decoded_sha256}."
                    )
                integrity_ms += (time.perf_counter() - integrity_start) * 1_000.0
                resident_shards.append(
                    _CudaCompactShard(
                        payload=decoded.view(cp.uint32),
                        headers=descriptors,
                    )
                )
                del (
                    compressed,
                    input_offsets,
                    widths_device,
                    word_offsets,
                    decode_status,
                    descriptor_status,
                    decoded_host,
                )

            excluded = cp.zeros(detector_pixel_count, dtype=cp.uint32)
            if index.excluded_detector_pixels:
                excluded[list(index.excluded_detector_pixels)] = 1
            scan_count = index.shape[0] * index.shape[1]
            detector_outputs = [
                cp.zeros(scan_count, dtype=cp.uint32),
                cp.zeros(scan_count, dtype=cp.uint32),
            ]
            diffraction_output = cp.empty(detector_pixel_count, dtype=cp.uint32)
            cp.cuda.Stream.null.synchronize()
        free_after, _ = cp.cuda.runtime.memGetInfo()
        metrics = CudaCompactH5LoadMetrics(
            metadata_ms=metadata_ms,
            whole_file_integrity_ms=whole_file_integrity_ms,
            nvrtc_compile_ms=nvrtc_compile_ms,
            source_read_ms=source_read_ms,
            host_header_validation_ms=0.0,
            host_payload_integrity_ms=0.0,
            descriptor_preparation_ms=descriptor_ms,
            gpu_upload_ms=0.0,
            gpu_header_validation_ms=0.0,
            gpu_decode_ms=gpu_decode_ms,
            decoded_integrity_ms=integrity_ms,
            total_ms=(time.perf_counter() - total_start) * 1_000.0,
            resident_bytes=index.resident_bytes,
            maximum_transient_bytes=maximum_transient_bytes,
            device_free_bytes_before=int(free_before),
            device_free_bytes_after=int(free_after),
            memory_pool_used_bytes=int(pool.used_bytes()),
            decoded_shard_sha256_checks=len(index.shards),
            direct_payload_sha256_checks=0,
        )
        return CudaCompactH5ResidentSource(
            index=index,
            load_metrics=metrics,
            shards=resident_shards,
            excluded=excluded,
            maximum_widths=maximum_widths,
            detector_outputs=detector_outputs,
            diffraction_output=diffraction_output,
            memory_pool=pool,
            kernels=kernels,
        )
    except Exception:
        resident_shards.clear()
        pool.free_all_blocks()
        raise
    finally:
        os.close(file_descriptor)


def _load_compact_h5_cuda_v3(
    index: CompactH5Index,
    *,
    total_start: float,
    metadata_ms: float,
    whole_file_integrity_ms: float,
    should_cancel: Callable[[], bool],
) -> CudaCompactH5ResidentSource:
    """Authenticate and upload direct QGIX v3 payloads and compact headers."""
    device = cp.cuda.Device()
    free_before, _ = cp.cuda.runtime.memGetInfo()
    compile_start = time.perf_counter()
    kernels, compiled_now = _cuda_kernels(device.id)
    nvrtc_compile_ms = (
        (time.perf_counter() - compile_start) * 1_000.0 if compiled_now else 0.0
    )
    pool = cp.cuda.MemoryPool()
    resident_shards: list[_CudaCompactShard] = []
    detector_pixel_count = index.shape[2] * index.shape[3]
    tile_count = index.scans_per_shard // index.scan_tile
    header_words_per_pixel = _header_words_per_pixel(index, tile_count)
    maximum_widths = np.zeros(detector_pixel_count, dtype=np.uint8)
    source_read_ms = 0.0
    host_header_validation_ms = 0.0
    host_payload_integrity_ms = 0.0
    gpu_upload_ms = 0.0
    gpu_header_validation_ms = 0.0
    maximum_transient_bytes = 0
    payload_sha256_checks = 0
    file_descriptor = os.open(index.path, os.O_RDONLY)
    try:
        with (
            cp.cuda.using_allocator(pool.malloc),
            ThreadPoolExecutor(max_workers=1) as read_executor,
        ):
            excluded = cp.zeros(detector_pixel_count, dtype=cp.uint32)
            if index.excluded_detector_pixels:
                excluded[list(index.excluded_detector_pixels)] = 1
            pending_read = read_executor.submit(
                _read_shard_envelope,
                file_descriptor,
                index.shards[0],
                0,
            )
            for shard_index, shard in enumerate(index.shards):
                if should_cancel():
                    raise RuntimeError(
                        f"Compact CUDA load was cancelled before shard {shard_index}."
                    )
                envelope, range_start, read_ms = pending_read.result()
                if shard_index + 1 < len(index.shards):
                    pending_read = read_executor.submit(
                        _read_shard_envelope,
                        file_descriptor,
                        index.shards[shard_index + 1],
                        shard_index + 1,
                    )
                envelope_view = memoryview(envelope)
                payload_start = shard.payload_offset - range_start
                header_start = shard.widths_offset - range_start
                payload_bytes = envelope_view[
                    payload_start : payload_start + shard.payload_bytes
                ]
                header_bytes = envelope_view[
                    header_start : header_start + shard.widths_bytes
                ]
                source_read_ms += read_ms
                maximum_transient_bytes = max(maximum_transient_bytes, len(envelope))

                header_validation_started = time.perf_counter()
                CompactH5ReferenceDecoder(index).validate_shard_metadata(shard_index)
                _update_v3_maximum_widths(
                    header_bytes,
                    detector_pixel_count=detector_pixel_count,
                    tile_count=tile_count,
                    header_words_per_pixel=header_words_per_pixel,
                    maximum_widths=maximum_widths,
                )
                host_header_validation_ms += (
                    time.perf_counter() - header_validation_started
                ) * 1_000.0

                payload_integrity_started = time.perf_counter()
                digest = hashlib.sha256(payload_bytes).hexdigest()
                if digest != shard.decoded_sha256:
                    raise ValueError(
                        f"Compact v3 shard {shard_index} direct payload SHA-256 "
                        f"is {digest}, expected {shard.decoded_sha256}."
                    )
                host_payload_integrity_ms += (
                    time.perf_counter() - payload_integrity_started
                ) * 1_000.0
                payload_sha256_checks += 1

                upload_started = time.perf_counter()
                payload_device = cp.asarray(
                    np.frombuffer(payload_bytes, dtype="<u4"), dtype=cp.uint32
                )
                headers_device = cp.asarray(
                    np.frombuffer(header_bytes, dtype="<u4"), dtype=cp.uint32
                )
                cp.cuda.Stream.null.synchronize()
                gpu_upload_ms += (time.perf_counter() - upload_started) * 1_000.0

                validation_status = cp.zeros(1, dtype=cp.uint32)
                validation_begin = cp.cuda.Event()
                validation_end = cp.cuda.Event()
                validation_begin.record()
                kernels["compact_h5_validate_compact_headers"](
                    ((detector_pixel_count + 255) // 256,),
                    (256,),
                    (
                        headers_device,
                        excluded,
                        np.uint32(detector_pixel_count),
                        np.uint32(tile_count),
                        np.uint32(header_words_per_pixel),
                        np.uint32(shard.decoded_bytes // 4),
                        validation_status,
                    ),
                )
                validation_end.record()
                validation_end.synchronize()
                gpu_header_validation_ms += float(
                    cp.cuda.get_elapsed_time(validation_begin, validation_end)
                )
                validation_error = int(validation_status.get()[0])
                if validation_error:
                    raise ValueError(
                        f"Compact v3 shard {shard_index} failed CUDA header "
                        f"validation with status {validation_error}."
                    )
                resident_shards.append(
                    _CudaCompactShard(
                        payload=payload_device,
                        headers=headers_device,
                    )
                )
                del validation_status

            scan_count = index.shape[0] * index.shape[1]
            detector_outputs = [
                cp.zeros(scan_count, dtype=cp.uint32),
                cp.zeros(scan_count, dtype=cp.uint32),
            ]
            diffraction_output = cp.empty(detector_pixel_count, dtype=cp.uint32)
            cp.cuda.Stream.null.synchronize()
        free_after, _ = cp.cuda.runtime.memGetInfo()
        metrics = CudaCompactH5LoadMetrics(
            metadata_ms=metadata_ms,
            whole_file_integrity_ms=whole_file_integrity_ms,
            nvrtc_compile_ms=nvrtc_compile_ms,
            source_read_ms=source_read_ms,
            host_header_validation_ms=host_header_validation_ms,
            host_payload_integrity_ms=host_payload_integrity_ms,
            descriptor_preparation_ms=0.0,
            gpu_upload_ms=gpu_upload_ms,
            gpu_header_validation_ms=gpu_header_validation_ms,
            gpu_decode_ms=0.0,
            decoded_integrity_ms=0.0,
            total_ms=(time.perf_counter() - total_start) * 1_000.0,
            resident_bytes=index.resident_bytes,
            maximum_transient_bytes=maximum_transient_bytes,
            device_free_bytes_before=int(free_before),
            device_free_bytes_after=int(free_after),
            memory_pool_used_bytes=int(pool.used_bytes()),
            decoded_shard_sha256_checks=0,
            direct_payload_sha256_checks=payload_sha256_checks,
        )
        return CudaCompactH5ResidentSource(
            index=index,
            load_metrics=metrics,
            shards=resident_shards,
            excluded=excluded,
            maximum_widths=maximum_widths,
            detector_outputs=detector_outputs,
            diffraction_output=diffraction_output,
            memory_pool=pool,
            kernels=kernels,
        )
    except Exception:
        resident_shards.clear()
        pool.free_all_blocks()
        raise
    finally:
        os.close(file_descriptor)


def _cuda_kernels(device_id: int) -> tuple[dict[str, Any], bool]:
    if device_id in _CUDA_MODULES:
        return _CUDA_MODULES[device_id], False
    module = cp.RawModule(
        code=_CUDA_COMPACT_SOURCE,
        options=("--std=c++11",),
        backend="nvrtc",
        name_expressions=_CUDA_FUNCTION_NAMES,
    )
    kernels = {name: module.get_function(name) for name in _CUDA_FUNCTION_NAMES}
    _CUDA_MODULES[device_id] = kernels
    return kernels, True


def _pread_exact(
    descriptor: int,
    offset: int,
    byte_count: int,
    label: str,
) -> bytearray:
    result = bytearray(byte_count)
    view = memoryview(result)
    position = 0
    while position < byte_count:
        chunk = os.pread(descriptor, byte_count - position, offset + position)
        if not chunk:
            raise ValueError(
                f"Compact HDF5 {label} ended after {position} of {byte_count} bytes."
            )
        view[position : position + len(chunk)] = chunk
        position += len(chunk)
    return result


def _read_shard_envelope(
    descriptor: int,
    shard: CompactH5Shard,
    shard_index: int,
) -> tuple[bytearray, int, float]:
    ranges = [
        (offset, offset + byte_count)
        for offset, byte_count in (
            (shard.payload_offset, shard.payload_bytes),
            (shard.lengths_offset, shard.lengths_bytes),
            (shard.widths_offset, shard.widths_bytes),
        )
        if byte_count
    ]
    range_start = min(start for start, _ in ranges)
    range_end = max(stop for _, stop in ranges)
    started = time.perf_counter()
    result = _pread_exact(
        descriptor,
        range_start,
        range_end - range_start,
        f"shard {shard_index} payload envelope",
    )
    return result, range_start, (time.perf_counter() - started) * 1_000.0


def _header_words_per_pixel(index: CompactH5Index, tile_count: int) -> int:
    if index.header_encoding == 0:
        return tile_count
    checkpoint_words = (tile_count + 31) // 32
    width_words = (tile_count + 7) // 8
    return checkpoint_words + width_words


def _update_v3_maximum_widths(
    header_bytes: memoryview,
    *,
    detector_pixel_count: int,
    tile_count: int,
    header_words_per_pixel: int,
    maximum_widths: np.ndarray,
) -> None:
    headers = np.frombuffer(header_bytes, dtype="<u4")
    expected_words = detector_pixel_count * header_words_per_pixel
    if headers.size != expected_words:
        raise ValueError(
            f"Compact v3 headers contain {headers.size} words, expected "
            f"{expected_words}."
        )
    headers = headers.reshape(detector_pixel_count, header_words_per_pixel)
    checkpoint_words = (tile_count + 31) // 32
    packed_widths = headers[:, checkpoint_words:]
    for tile in range(tile_count):
        widths = (packed_widths[:, tile // 8] >> np.uint32((tile % 8) * 4)) & np.uint32(
            15
        )
        np.maximum(maximum_widths, widths, out=maximum_widths, casting="unsafe")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _validate_widths(
    index: CompactH5Index,
    shard: CompactH5Shard,
    shard_index: int,
    widths: np.ndarray,
    maximum_widths: np.ndarray,
) -> int:
    if widths.size != shard.descriptor_count:
        raise ValueError(f"Compact shard {shard_index} descriptor count changed.")
    maximum = int(widths.max(initial=0))
    if maximum > 16:
        raise ValueError(
            f"Compact shard {shard_index} requires {maximum} bits per sample, "
            "beyond exact uint16."
        )
    if index.manifest.get("working_dtype") == "uint8" and maximum > 8:
        wide_pixels = set(
            (np.flatnonzero(widths > 8) // ((index.scans_per_shard + 127) // 128))
            .astype(int)
            .tolist()
        )
        unexpected = wide_pixels.difference(index.excluded_detector_pixels)
        if unexpected:
            raise ValueError(
                f"Compact shard {shard_index} requires more than eight bits for "
                f"nonexcluded detector pixels {sorted(unexpected)[:8]}, but its "
                "legacy manifest declares uint8 working values."
            )
    reshaped = widths.reshape(maximum_widths.size, -1)
    np.maximum(maximum_widths, reshaped.max(axis=1), out=maximum_widths)
    payload_words = int(widths.sum(dtype=np.uint64)) * 4
    if payload_words > 1 << 27:
        raise ValueError(
            f"Compact shard {shard_index} exceeds the 27-bit descriptor offset."
        )
    if payload_words * 4 != shard.decoded_bytes:
        raise ValueError(
            f"Compact shard {shard_index} widths cover {payload_words * 4} "
            f"decoded bytes, expected {shard.decoded_bytes}."
        )
    return payload_words
