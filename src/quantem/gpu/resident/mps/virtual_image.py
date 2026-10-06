"""Raw-Metal virtual-detector reductions over chunked unified-memory frames.

Dragging a virtual detector over a 4D-STEM acquisition sums a detector mask at
every scan position. On an Apple GPU ``MetalVirtualImage`` runs those sums, the
detector total, the mean pattern of selected positions, the centre of mass and
detector binning directly on the resident uint8/uint16/uint32 Metal buffers, so
an interaction never copies or casts the 4D data.
"""

import time
import weakref
from pathlib import Path
from typing import Self

import numpy as np

from quantem.gpu.device.metal_runtime import (
    SharedArray,
    allocate_shared,
    metal_device,
    metal_module,
    metal_pipelines,
    metal_queue,
    numpy_view,
    release_buffer,
)

_REDUCTIONS_MSL = Path(__file__).with_name("kernels").joinpath("reductions.msl").read_text()

# Chunk buffers bound per command buffer. Binding all 27 (19.3 GB) in one command
# buffer errors with status=5 (working-set limit). Measured sweet spot is 13:
# CPB=8 -> 3 fps, CPB=10/13 -> 8 fps, CPB=27 -> status=5 error. Fewer, fuller
# command buffers mean fewer waitUntilCompleted stalls (wait-last, not wait-each,
# cut BF from 475 ms to 194 ms). 13 keeps each command buffer's working set under
# the limit while halving the wait count vs 8.
_CHUNKS_PER_COMMAND_BUFFER = 13
_COMMAND_BUFFER_BYTES = 9_500_000_000
_KERNEL_SUFFIX = {np.dtype(np.uint8): "u8", np.dtype(np.uint16): "u16", np.dtype(np.uint32): "u32"}


class MetalVirtualImage:
    """Raw-Metal BF/DF/ADF over a list of unified-memory Metal-buffer chunks.

    Each chunk is a ``SharedArray`` (NumPy view over a Metal buffer) of shape
    ``(frames, detector_row, detector_col)``. The kernels read the underlying
    ``_mtl`` buffer directly: no Torch, no copy and no dtype cast while the user
    drags BF/DF/ADF masks. The chunks are borrowed; the output, parameter and
    scratch buffers belong to this object and are released with it. Every
    returned array belongs to the caller.
    """

    def __init__(self, chunks: list):
        self._metal = metal_module()
        self.chunks = chunks
        # PyObjC never frees a Metal buffer when its wrapper is collected, so every
        # buffer this object allocates is listed here and released with the object.
        self._buffers = []
        weakref.finalize(self, _release_buffers, self._buffers)
        self._dtype = np.dtype(chunks[0].dtype)
        if self._dtype not in _KERNEL_SUFFIX:
            raise TypeError(
                f"MetalVirtualImage supports uint8, uint16, and uint32 chunk-backed data, got {self._dtype}."
            )
        suffix = _KERNEL_SUFFIX[self._dtype]
        self.det = tuple(int(size) for size in chunks[0].shape[1:])
        self.ndet = self.det[0] * self.det[1]
        frame_counts = [int(chunk.shape[0]) for chunk in chunks]
        self.n = sum(frame_counts)
        self._offsets = [0]
        for count in frame_counts:
            self._offsets.append(self._offsets[-1] + count)
        # A binned uint32 pixel can exceed the uint16 sidecar, so uint32 frames have none.
        binnable = self._dtype != np.dtype(np.uint32)
        # uint8 frames sum four pixels per thread in 1024-frame blocks, then merge the blocks.
        blocked_u8 = self._dtype == np.dtype(np.uint8) and self.ndet % 4 == 0
        names = (
            f"masked_sum_{suffix}", f"rowspan_sum_{suffix}", f"detector_sum_exact_{suffix}",
            f"mean_dp_sum_{suffix}", f"gather_columns_f32_tiled_{suffix}", f"com_{suffix}",
            *((f"bin_detector_{suffix}",) if binnable else ()),
            *(("detector_sum_u8_block_partial", "detector_sum_u8_block_merge") if blocked_u8 else ()),
        )
        pipelines = metal_pipelines(_REDUCTIONS_MSL, names)
        self._masked_sum_pipeline = pipelines[f"masked_sum_{suffix}"]
        self._span_sum_pipeline = pipelines[f"rowspan_sum_{suffix}"]
        self._detector_sum_exact_pipeline = pipelines[f"detector_sum_exact_{suffix}"]
        self._mean_pipeline = pipelines[f"mean_dp_sum_{suffix}"]
        self._gather_pipeline = pipelines[f"gather_columns_f32_tiled_{suffix}"]
        self._com_pipeline = pipelines[f"com_{suffix}"]
        self._bin_pipeline = pipelines.get(f"bin_detector_{suffix}")
        self._u8_partial_pipeline = pipelines.get("detector_sum_u8_block_partial")
        self._u8_merge_pipeline = pipelines.get("detector_sum_u8_block_merge")
        self._u8_partial_buffer = None
        if blocked_u8:
            blocks = (max(frame_counts) + 1023) // 1024
            self._u8_partial_buffer = self._allocate(blocks * self.ndet * np.dtype(np.int32).itemsize)
        # Every output and parameter buffer is allocated once and reused by every query.
        self._sum_buffers = [self._allocate(count * 8) for count in frame_counts]
        self._sum_views = [
            numpy_view(buffer, np.uint64, count)
            for buffer, count in zip(self._sum_buffers, frame_counts)
        ]
        # Centre of mass writes three exact moments (total, row, column) per frame.
        self._com_buffers = [self._allocate(count * 3 * 8) for count in frame_counts]
        self._com_views = [
            numpy_view(buffer, np.uint64, count * 3)
            for buffer, count in zip(self._com_buffers, frame_counts)
        ]
        self._ndet_buffer = self._uint32_buffer(self.ndet)
        self._detector_cols_buffer = self._uint32_buffer(self.det[1])
        self._frame_count_buffers = [self._uint32_buffer(count) for count in frame_counts]
        self._exact_buffers = [self._allocate(self.ndet * np.dtype(np.uint64).itemsize) for _ in chunks]
        self._exact_views = [
            numpy_view(buffer, np.uint64, self.ndet) for buffer in self._exact_buffers
        ]
        self._mask_buffer = self._allocate(self.ndet)
        self._mask_view = numpy_view(self._mask_buffer, np.uint8, self.ndet)
        self._span_count_buffer = self._allocate(4)
        self._span_count_view = numpy_view(self._span_count_buffer, np.uint32, 1)
        self._span_capacity = 0
        self._span_start_buffer = self._span_end_buffer = None
        self._span_start_view = self._span_end_view = None
        self._selected_capacity = [0] * len(chunks)
        self._selected_buffers = [None] * len(chunks)
        self._selected_views = [None] * len(chunks)
        self._selected_count_buffers = [self._allocate(4) for _ in chunks]
        self._selected_count_views = [
            numpy_view(buffer, np.uint32, 1) for buffer in self._selected_count_buffers
        ]
        self._selected_sum_buffers = [self._allocate(self.ndet * 8) for _ in chunks]
        self._selected_sum_views = [
            numpy_view(buffer, np.uint64, self.ndet) for buffer in self._selected_sum_buffers
        ]

    def masked_sum(self, mask: np.ndarray) -> np.ndarray:
        """Return the exact detector-masked sum of every scan position as ``(N,)`` uint64.

        A mask of at most 512 contiguous row spans is summed span by span, which
        reads only the selected pixels; a larger mask is read through a dense
        0/1 detector mask.
        """
        mask = np.asarray(mask, dtype=bool).reshape(self.det)
        starts, ends = [], []
        for row in range(mask.shape[0]):
            cols = np.flatnonzero(mask[row])
            if cols.size == 0:
                continue
            breaks = np.flatnonzero(cols[1:] != cols[:-1] + 1) + 1
            for span in np.split(cols, breaks):
                starts.append(row * self.det[1] + int(span[0]))
                ends.append(row * self.det[1] + int(span[-1]))
        span_count = len(starts)
        if span_count == 0:
            return np.zeros(self.n, dtype=np.uint64)
        if span_count <= 512:
            if span_count > self._span_capacity:
                self._span_capacity = max(64, span_count)
                self._span_start_buffer = self._replace(self._span_start_buffer, self._span_capacity * 4)
                self._span_end_buffer = self._replace(self._span_end_buffer, self._span_capacity * 4)
                self._span_start_view = numpy_view(
                    self._span_start_buffer, np.uint32, self._span_capacity
                )
                self._span_end_view = numpy_view(
                    self._span_end_buffer, np.uint32, self._span_capacity
                )
            self._span_start_view[:span_count] = np.asarray(starts, dtype=np.uint32)
            self._span_end_view[:span_count] = np.asarray(ends, dtype=np.uint32)
            self._span_count_view[0] = span_count

            def encode(encoder, index):
                for slot, buffer in enumerate(
                    (
                        self.chunks[index]._mtl,
                        self._span_start_buffer,
                        self._span_end_buffer,
                        self._sum_buffers[index],
                        self._span_count_buffer,
                        self._ndet_buffer,
                        self._frame_count_buffers[index],
                    )
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
                self._dispatch(encoder, self.chunks[index].shape[0])

            self._run(self._span_sum_pipeline, encode)
        else:
            self._mask_view[:] = mask.reshape(-1).astype(np.uint8)

            def encode(encoder, index):
                for slot, buffer in enumerate(
                    (
                        self.chunks[index]._mtl,
                        self._mask_buffer,
                        self._sum_buffers[index],
                        self._ndet_buffer,
                        self._frame_count_buffers[index],
                    )
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
                self._dispatch(encoder, self.chunks[index].shape[0])

            self._run(self._masked_sum_pipeline, encode)
        image = np.empty(self.n, dtype=np.uint64)
        for index, view in enumerate(self._sum_views):
            image[self._offsets[index] : self._offsets[index + 1]] = view
        return image

    def center_of_mass(self, mask: np.ndarray | None = None):
        """Per-scan-position centre of mass over the masked detector.

        Returns ``(com_col, com_row)``, each ``(N,)`` float32 in absolute
        detector pixels (col = sum col*I / sum I, row = sum row*I / sum I): the
        DPC vector field before mean subtraction and rotation, and 0 where a
        pattern holds no counts. ``mask`` None means the full detector. The
        kernel reads the frames in place into exact 64-bit integer moments;
        Metal has no double, so the moments are divided here in float64 and
        rounded once to float32, the same values as the CUDA kernels.
        """
        if mask is None:
            self._mask_view[:] = 1
        else:
            self._mask_view[:] = np.asarray(mask, dtype=bool).reshape(-1).astype(np.uint8)

        def encode(encoder, index):
            for slot, buffer in enumerate(
                (
                    self.chunks[index]._mtl,
                    self._mask_buffer,
                    self._com_buffers[index],
                    self._ndet_buffer,
                    self._detector_cols_buffer,
                    self._frame_count_buffers[index],
                )
            ):
                encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
            self._dispatch(encoder, self.chunks[index].shape[0])

        self._run(self._com_pipeline, encode)
        moments = np.empty((self.n, 3), dtype=np.uint64)
        for index, view in enumerate(self._com_views):
            moments[self._offsets[index] : self._offsets[index + 1]] = view.reshape(-1, 3)
        total = moments[:, 0].astype(np.float64)
        centers = np.zeros((2, self.n), dtype=np.float64)
        np.divide(moments[:, 1:].T.astype(np.float64), total, out=centers, where=total > 0)
        com_row, com_col = centers.astype(np.float32)
        return com_col, com_row

    def sum_frames(self, frame_indices: np.ndarray) -> np.ndarray:
        """Exact uint64 sum of selected scan positions as one ``(det_row, det_col)`` pattern.

        Each chunk sums its selected frames on the GPU in 64 bits; the chunk
        sums are added on the host. Repeated indices count each time.
        """
        indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return np.zeros(self.det, dtype=np.uint64)
        if int(indices.min()) < 0 or int(indices.max()) >= self.n:
            raise IndexError("frame index out of bounds")
        indices = np.sort(indices.astype(np.uint32, copy=False))
        active = []
        for index in range(len(self.chunks)):
            start, stop = self._offsets[index], self._offsets[index + 1]
            low = int(np.searchsorted(indices, start, side="left"))
            high = int(np.searchsorted(indices, stop, side="left"))
            count = high - low
            if count == 0:
                continue
            if count > self._selected_capacity[index]:
                capacity = max(256, count, self._selected_capacity[index] * 2)
                self._selected_capacity[index] = capacity
                self._selected_buffers[index] = self._replace(self._selected_buffers[index], capacity * 4)
                self._selected_views[index] = numpy_view(
                    self._selected_buffers[index], np.uint32, capacity
                )
            self._selected_views[index][:count] = indices[low:high] - start
            self._selected_count_views[index][0] = count
            active.append(index)

        def encode(encoder, index):
            for slot, buffer in enumerate(
                (
                    self.chunks[index]._mtl,
                    self._selected_buffers[index],
                    self._selected_sum_buffers[index],
                    self._ndet_buffer,
                    self._selected_count_buffers[index],
                )
            ):
                encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
            self._dispatch(encoder, self.ndet)

        self._run(self._mean_pipeline, encode, selected=set(active))
        total = np.zeros(self.ndet, dtype=np.uint64)
        for index in active:
            total += self._selected_sum_views[index]
        return total.reshape(self.det)

    def detector_sum_exact(self) -> np.ndarray:
        """Return the exact uint64 detector sum over all resident chunks.

        Each chunk writes its own uint64 detector plane, so no sum wraps and no
        atomic order varies; the small planes are added in a fixed host order.
        uint8 frames sum four pixels per thread over 1024-frame blocks first.
        """
        metal = self._metal
        tiled_u8 = self._u8_partial_pipeline is not None

        def encode(encoder, index):
            if tiled_u8:
                blocks = (int(self.chunks[index].shape[0]) + 1023) // 1024
                encoder.setComputePipelineState_(self._u8_partial_pipeline)
                for slot, buffer in enumerate(
                    (
                        self.chunks[index]._mtl,
                        self._u8_partial_buffer,
                        self._ndet_buffer,
                        self._frame_count_buffers[index],
                    )
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
                self._dispatch(encoder, self.ndet // 4 * blocks)
                encoder.memoryBarrierWithScope_(metal.MTLBarrierScopeBuffers)
                encoder.setComputePipelineState_(self._u8_merge_pipeline)
                encoder.setBuffer_offset_atIndex_(self._u8_partial_buffer, 0, 0)
                encoder.setBuffer_offset_atIndex_(self._exact_buffers[index], 0, 1)
                encoder.setBuffer_offset_atIndex_(self._ndet_buffer, 0, 2)
                encoder.setBytes_length_atIndex_(np.asarray([blocks], dtype=np.uint32).tobytes(), 4, 3)
            else:
                for slot, buffer in enumerate(
                    (
                        self.chunks[index]._mtl,
                        self._exact_buffers[index],
                        self._ndet_buffer,
                        self._frame_count_buffers[index],
                    )
                ):
                    encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
            self._dispatch(encoder, self.ndet)

        self._run(None if tiled_u8 else self._detector_sum_exact_pipeline, encode)
        total = np.zeros(self.ndet, dtype=np.uint64)
        for chunk_sum in self._exact_views:
            np.add(total, chunk_sum, out=total)
        return total.reshape(self.det)

    def gather_columns_float32(
        self, rows: np.ndarray, cols: np.ndarray, *, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Gather detector pixels over all scan positions into ``(pixel, frame)`` float32 storage.

        SSB fits and reconstructs from bright-field detector pixels, one scan
        image per pixel. A provided ``out`` allocation is wrapped as a no-copy
        Metal buffer, so SSB gathers directly into MLX-owned unified memory.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        if rows.shape != cols.shape:
            raise ValueError("rows and cols must have matching shapes.")
        if rows.size == 0:
            return np.empty((0, self.n), dtype=np.float32)
        if (
            int(rows.min()) < 0
            or int(rows.max()) >= self.det[0]
            or int(cols.min()) < 0
            or int(cols.max()) >= self.det[1]
        ):
            raise IndexError("detector column index out of bounds")
        indices = (rows * self.det[1] + cols).astype(np.uint32, copy=False)
        output_shape = (int(indices.size), self.n)
        output_nbytes = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize
        owns_output = out is None
        if owns_output:
            output_buffer = allocate_shared(output_nbytes, "gathered detector columns")
            output = numpy_view(output_buffer, np.float32, int(indices.size) * self.n)
            output = output.reshape(*output_shape).view(SharedArray)
            output._mtl = output_buffer
        else:
            output = np.asarray(out)
            if output.dtype != np.float32 or output.shape != output_shape:
                raise ValueError(
                    "MPS column gather output must be a C-contiguous float32 "
                    f"array with shape {output_shape}; got {output.dtype} "
                    f"with shape {output.shape}."
                )
            if not output.flags.c_contiguous or not output.flags.writeable:
                raise ValueError("MPS column gather output must be writable and C-contiguous.")
            output_buffer = metal_device().newBufferWithBytesNoCopy_length_options_deallocator_(
                output, output_nbytes, self._metal.MTLResourceStorageModeShared, None
            )
            if output_buffer is None:
                raise RuntimeError(
                    "Metal could not wrap the provided unified-memory output. "
                    "Use an MLX-allocated float32 destination."
                )
        indices_buffer = allocate_shared(int(indices.nbytes), "gathered pixel indices")
        numpy_view(indices_buffer, np.uint32, int(indices.size))[:] = indices
        pixel_count = np.asarray([indices.size], dtype=np.uint32).tobytes()
        total_frames = np.asarray([self.n], dtype=np.uint32).tobytes()

        def encode(encoder, index):
            for slot, buffer in enumerate(
                (
                    self.chunks[index]._mtl,
                    indices_buffer,
                    output_buffer,
                    self._ndet_buffer,
                    self._frame_count_buffers[index],
                )
            ):
                encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
            encoder.setBytes_length_atIndex_(total_frames, 4, 5)
            encoder.setBytes_length_atIndex_(
                np.asarray([self._offsets[index]], dtype=np.uint32).tobytes(), 4, 6
            )
            encoder.setBytes_length_atIndex_(pixel_count, 4, 7)
            # 16x16 tiles transpose frame-major reads into pixel-major writes.
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                self._metal.MTLSizeMake(
                    (int(indices.size) + 15) // 16, (int(self.chunks[index].shape[0]) + 15) // 16, 1
                ),
                self._metal.MTLSizeMake(16, 16, 1),
            )

        self._run(self._gather_pipeline, encode)
        release_buffer(indices_buffer)
        if not owns_output:
            release_buffer(output_buffer)
        return output

    def binned(self, factor: int = 2, *, verbose: bool = True) -> Self:
        """Build a detector-binned uint16 copy of the frames for fast live interaction.

        ``factor * factor`` raw pixels sum into one binned pixel, computed on
        the GPU from the resident full-resolution chunks (no disk re-decode, no
        decompress scratch). bin4 keeps the copy at 1.2 GB so the whole no-bin
        viewer fits a 24 GB Mac (~20.5 GB); bin2 (4.8 GB) is for boxes with
        memory to spare. The returned image owns the binned chunks and releases
        them with itself.
        """
        factor = int(factor)
        if self.det[0] % factor or self.det[1] % factor:
            raise ValueError(f"bin{factor} fast interaction requires detector dims divisible by {factor}")
        if verbose:
            print(f"Building detector-bin{factor} virtual-image cache")
        started = time.perf_counter()
        if self._dtype == np.dtype(np.uint32):
            raise ValueError(
                "fast detector-bin sidecars for native uint32 MPS chunks are "
                "not enabled because a binned pixel can exceed uint32. Load "
                "with output_dtype=np.uint16 after a count-range check, or use "
                "native no-bin uint32 products."
            )
        binned_shape = (self.det[0] // factor, self.det[1] // factor)
        binned_pixels = binned_shape[0] * binned_shape[1]
        binned_chunks = []
        for chunk in self.chunks:
            frames = int(chunk.shape[0])
            buffer = self._allocate(frames * binned_pixels * 2)
            binned = numpy_view(buffer, np.uint16, frames * binned_pixels)
            binned = binned.reshape((frames, *binned_shape)).view(SharedArray)
            binned._mtl = buffer
            binned_chunks.append(binned)
        parameters = [self._uint32_buffer(value) for value in (binned_shape[1], binned_pixels, factor)]
        binned_cols_buffer, binned_pixels_buffer, factor_buffer = parameters

        def encode(encoder, index):
            for slot, buffer in enumerate(
                (
                    self.chunks[index]._mtl,
                    binned_chunks[index]._mtl,
                    self._detector_cols_buffer,
                    binned_cols_buffer,
                    binned_pixels_buffer,
                    self._frame_count_buffers[index],
                    factor_buffer,
                )
            ):
                encoder.setBuffer_offset_atIndex_(buffer, 0, slot)
            self._dispatch(encoder, int(self.chunks[index].shape[0]) * binned_pixels)

        self._run(self._bin_pipeline, encode)
        for buffer in parameters:
            self._release(buffer)
        sidecar = MetalVirtualImage(binned_chunks)
        # The binned chunks now belong to the sidecar, which releases them with itself.
        for chunk in binned_chunks:
            self._buffers.remove(chunk._mtl)
            sidecar._buffers.append(chunk._mtl)
        if verbose:
            elapsed = time.perf_counter() - started
            print(f"Detector-bin{factor} virtual-image cache ready in {elapsed:.2f}s")
        return sidecar

    # ---

    def _run(self, pipeline, encode, *, selected=None):
        """Encode one dispatch per chunk in command buffers that stay under the working-set limit.

        ``encode(encoder, chunk_index)`` binds one chunk's buffers and
        dispatches; ``pipeline`` None leaves pipeline selection to ``encode``.
        ``selected`` limits the chunks. Command buffers on one queue run in
        order, so waiting on the last waits for all.
        """
        commands = []
        start, size, count = 0, 0, 0
        groups = []
        for index, chunk in enumerate(self.chunks):
            nbytes = int(np.asarray(chunk).nbytes)
            if count and (count >= _CHUNKS_PER_COMMAND_BUFFER or size + nbytes > _COMMAND_BUFFER_BYTES):
                groups.append(range(start, index))
                start, size, count = index, 0, 0
            size += nbytes
            count += 1
        groups.append(range(start, len(self.chunks)))
        for group in groups:
            indices = [index for index in group if selected is None or index in selected]
            if not indices:
                continue
            command = metal_queue().commandBuffer()
            encoder = command.computeCommandEncoder()
            if pipeline is not None:
                encoder.setComputePipelineState_(pipeline)
            for index in indices:
                encode(encoder, index)
            encoder.endEncoding()
            command.commit()
            commands.append(command)
        commands[-1].waitUntilCompleted()

    def _dispatch(self, encoder, threads):
        """Launch ``threads`` threads in 256-thread groups; kernels return past the end."""
        size = self._metal.MTLSizeMake
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            size((int(threads) + 255) // 256, 1, 1), size(256, 1, 1)
        )

    def _uint32_buffer(self, value):
        """A 4-byte Metal buffer holding one uint32 kernel parameter."""
        buffer = self._allocate(4)
        numpy_view(buffer, np.uint32, 1)[0] = value
        return buffer

    def _allocate(self, nbytes: int):
        """A shared Metal buffer that this object owns and releases with itself."""
        buffer = allocate_shared(nbytes, "virtual image")
        self._buffers.append(buffer)
        return buffer

    def _replace(self, old, nbytes: int):
        """Grow scratch: release the old buffer (if any) and own a new one of ``nbytes``."""
        if old is not None:
            self._release(old)
        return self._allocate(nbytes)

    def _release(self, buffer) -> None:
        """Release one owned buffer before the object goes away."""
        self._buffers.remove(buffer)
        release_buffer(buffer)


def _release_buffers(buffers: list) -> None:
    """Finalizer of a ``MetalVirtualImage``: release every buffer it still owns, exactly once."""
    for buffer in buffers:
        release_buffer(buffer)
    buffers.clear()
