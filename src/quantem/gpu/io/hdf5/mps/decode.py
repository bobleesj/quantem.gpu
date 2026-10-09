"""MPS (Apple Metal GPU) bitshuffle+LZ4 decompression for Arina 4D-STEM.

Bounded encoded and precision loads read a block of compressed HDF5 chunks on
the host (``formats.hdf5.reads.FrameReader.prepare``) and decode it here, with
Metal compute shaders writing straight into a unified-memory buffer that the
ANS encoder or the precision converter then consumes. Decoded values keep
their stored dtype; ``exact_uint16`` stores uint32 counts that fit 16 bits as
uint16 for the encoded loader. Kernels compile on first use, so this module
imports on Linux.
"""

from functools import cache
from pathlib import Path

import numpy as np

from quantem.gpu.device.metal_runtime import (
    SharedArray,
    allocate_shared,
    complete_command,
    metal_module,
    metal_pipelines,
    metal_queue,
    numpy_view,
    release_buffer,
    shared_array,
    upload_shared,
)

__all__ = ["MPSDecompressor", "exact_uint16", "load_prepared_frames"]

_KERNELS = Path(__file__).parent / "kernels"
# LZ4 occupancy: compressed 8 KB bitshuffle blocks packed per threadgroup (more
# SIMD groups give better latency hiding on the M5); 8 is the most stable
# setting. The threadgroup input cache is sized to exactly this many blocks
# by compile-time token substitution, because an oversized fixed buffer hurts
# occupancy.
_LZ4_BLOCKS_PER_GROUP = 8
_SOURCE = (_KERNELS / "bslz4.msl").read_text().replace(
    "LZ4_BLOCKS_PER_TG", str(_LZ4_BLOCKS_PER_GROUP)
)
_NAMES = (
    "h5lz4dc_batched",
    "shuf_8192_32_batched",
    "shuf_8192_16_batched",
    "shuf_tail_16_batched",
    "shuf_tail_32_batched",
    "shuf_8_batched",
    "narrow_u32_u16",
)


def load_prepared_frames(prepared: dict) -> SharedArray:
    """Decode prepared HDF5 frames into a fresh Metal-backed array.

    ``prepared`` is the frame plan from ``FrameReader.prepare``. The returned
    ``SharedArray`` ``(n_frames, det_row, det_col)`` owns its Metal buffer
    through ``_mtl``; the caller releases it with ``release_buffer``.
    """
    frame_bytes = int(prepared["frame_bytes"])
    _bitshuffle_tail_elements(frame_bytes, np.dtype(prepared["dtype"]).itemsize)
    total_frames = int(prepared["total_frames"])
    decoder = MPSDecompressor(
        max_compressed_bytes=max(1, len(prepared["read_buffer"])),
        max_frames=total_frames,
        frame_bytes=frame_bytes,
        n_blocks_per_frame=(frame_bytes + 8191) // 8192,
    )
    try:
        return decoder.load_prepared_frames(prepared)
    finally:
        decoder.free()


def exact_uint16(frames: SharedArray, first: int, valid: np.ndarray) -> tuple[SharedArray, int]:
    """Copy decoded uint32 counts into a new uint16 array without changing any valid count.

    Arina writes uint32 files whose counts fit 16 bits, and the encoded
    resident stores uint16. The kernel copies every value and flags a valid
    count above 65535 instead of clipping it, so such a batch is refused and
    the original file must be kept. A value above 65535 at a flagged pixel
    (``valid`` False) is the detector's uint32 marker; it is stored as 0 and
    counted, and the pixel mask keeps the position. ``first`` is the batch's
    first scan index, named in the error. Returns the uint16 array and the
    number of markers stored as 0; the caller releases ``frames`` and the
    returned array.
    """
    count = int(frames.size)
    flag = allocate_shared(8, "count range flags")
    mask = upload_shared(np.asarray(valid, np.uint8).reshape(-1), "valid detector pixels")
    output = allocate_shared(count * 2, "uint16 counts")
    try:
        numpy_view(flag, np.uint32, 2)[:] = 0
        command = metal_queue().commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(metal_pipelines(_SOURCE, _NAMES)["narrow_u32_u16"])
        encoder.setBuffer_offset_atIndex_(frames._mtl, 0, 0)
        encoder.setBuffer_offset_atIndex_(output, 0, 1)
        encoder.setBuffer_offset_atIndex_(flag, 0, 2)
        _set_u32(encoder, 3, count)
        encoder.setBuffer_offset_atIndex_(mask, 0, 4)
        _set_u32(encoder, 5, int(valid.size))
        encoder.dispatchThreads_threadsPerThreadgroup_(
            metal_module().MTLSizeMake(count, 1, 1), metal_module().MTLSizeMake(256, 1, 1)
        )
        encoder.endEncoding()
        complete_command(command, "uint32 count narrowing")
        above, zeroed = (int(value) for value in numpy_view(flag, np.uint32, 2))
    except BaseException:
        release_buffer(output)
        raise
    finally:
        release_buffer(flag)
        release_buffer(mask)
    if above:
        release_buffer(output)
        raise ValueError(
            f"Frames {first} to {first + len(frames) - 1} hold counts above 65535 at unflagged "
            "pixels, so the uint32 acquisition cannot be encoded as exact uint16 counts. Keep "
            "the original file; this encoded writer does not support these uint32 values."
        )
    return shared_array(output, np.dtype("uint16"), frames.shape), zeroed


class MPSDecompressor:
    """MPS-accelerated decompressor for bitshuffle+LZ4 HDF5 chunks.

    The compressed input and its block metadata live in reusable unified-memory
    Metal buffers sized for ``max_frames`` frames of ``frame_bytes`` bytes;
    the LZ4 scratch is allocated only when a frame layout needs it, because
    the fused uint16 kernel decodes straight into the output.

    Parameters
    ----------
    max_compressed_bytes : int
        Largest compressed batch one call may hold.
    max_frames : int
        Largest number of frames one call may decode.
    frame_bytes : int
        Decompressed bytes per frame.
    n_blocks_per_frame : int
        8 KiB LZ4 blocks per frame.
    """

    def __init__(
        self,
        *,
        max_compressed_bytes: int,
        max_frames: int,
        frame_bytes: int,
        n_blocks_per_frame: int,
    ):
        self.max_compressed_bytes = max_compressed_bytes
        self.max_frames = max_frames
        self.frame_bytes = frame_bytes
        self.n_blocks_per_frame = n_blocks_per_frame
        self._compressed_mtl = allocate_shared(max_compressed_bytes, "compressed HDF5 chunks")
        self._compressed_np = numpy_view(self._compressed_mtl, np.uint8, max_compressed_bytes)
        self._lz4_mtl = None
        self._chunk_offsets_mtl = allocate_shared(max_frames * 4, "chunk offsets")
        self._chunk_offsets_np = numpy_view(self._chunk_offsets_mtl, np.uint32, max_frames)
        max_blocks = max_frames * n_blocks_per_frame
        self._block_starts_mtl = allocate_shared(max_blocks * 4, "LZ4 block starts")
        self._block_starts_np = numpy_view(self._block_starts_mtl, np.uint32, max_blocks)
        self._block_counts_mtl = allocate_shared(max_frames * 4, "LZ4 block counts")
        self._block_counts_np = numpy_view(self._block_counts_mtl, np.uint32, max_frames)
        self._block_offsets_mtl = allocate_shared((max_frames + 1) * 4, "LZ4 block offsets")
        self._block_offsets_np = numpy_view(self._block_offsets_mtl, np.uint32, max_frames + 1)

    def free(self) -> None:
        """Release every Metal buffer this decompressor still owns.

        PyObjC never frees these buffers on its own, so a decompressor that
        goes out of use would otherwise keep its scratch for the life of the
        process.
        """
        for name in (
            "_compressed_mtl", "_lz4_mtl", "_chunk_offsets_mtl",
            "_block_starts_mtl", "_block_counts_mtl", "_block_offsets_mtl",
        ):
            release_buffer(getattr(self, name))
            setattr(self, name, None)
        # The NumPy views point at freed memory now, so drop them together.
        self._compressed_np = self._chunk_offsets_np = None
        self._block_starts_np = self._block_counts_np = self._block_offsets_np = None

    def __del__(self):
        # A decoder kept by a caller and dropped without free() still returns
        # its scratch. At interpreter shutdown the module globals free() needs
        # may be gone; the OS reclaims the buffers then anyway.
        try:
            self.free()
        except (TypeError, AttributeError, NameError):
            pass

    def load_prepared_frames(self, prepared: dict) -> SharedArray:
        """Decode one prepared frame plan into a fresh Metal-backed array.

        The compressed bytes and block metadata of ``prepared`` are copied into
        this decoder's unified-memory buffers, then decoded in the stored
        dtype into a new ``SharedArray`` that owns its buffer.
        """
        read_buffer = prepared["read_buffer"]
        total_frames = int(prepared["total_frames"])
        frame_shape = tuple(int(size) for size in prepared["frame_shape"])
        dtype = np.dtype(prepared["dtype"])
        frame_bytes = int(prepared["frame_bytes"])
        elem_size = int(dtype.itemsize)
        _bitshuffle_tail_elements(frame_bytes, elem_size)
        if total_frames > self.max_frames:
            raise ValueError(
                f"Prepared crop has {total_frames} frames but this MPS decoder "
                f"was allocated for {self.max_frames}."
            )
        if len(read_buffer) > self.max_compressed_bytes:
            raise ValueError(
                f"Prepared compressed crop is {len(read_buffer)} bytes but this "
                f"MPS decoder was allocated for {self.max_compressed_bytes} bytes."
            )
        chunk_offsets = np.asarray(prepared["chunk_offsets"], dtype=np.uint64)
        if int(chunk_offsets.max(initial=0)) > np.iinfo(np.uint32).max:
            raise ValueError(
                "MPS sparse crop compressed buffer exceeds 32-bit chunk offsets; "
                "load a smaller scan_region."
            )
        block_starts = np.asarray(prepared["block_starts"], dtype=np.uint32)
        block_counts = np.asarray(prepared["block_counts"], dtype=np.uint32)
        block_offsets = np.asarray(prepared["block_offsets"], dtype=np.uint32)
        n_blocks = int(block_starts.size)
        if n_blocks > self._block_starts_np.size:
            raise ValueError(
                f"Prepared crop has {n_blocks} LZ4 blocks but this MPS decoder "
                f"was allocated for {self._block_starts_np.size}."
            )

        self._compressed_np[: len(read_buffer)] = read_buffer
        self._chunk_offsets_np[:total_frames] = chunk_offsets.astype(np.uint32, copy=False)
        self._block_starts_np[:n_blocks] = block_starts
        self._block_counts_np[:total_frames] = block_counts[:total_frames]
        self._block_offsets_np[: total_frames + 1] = block_offsets[: total_frames + 1]
        max_blocks = int(block_counts[:total_frames].max(initial=1))

        out_mtl = allocate_shared(total_frames * frame_bytes, "decoded frames")
        try:
            self._submit_gpu(
                total_frames, frame_bytes, elem_size, out_mtl, max_blocks
            ).waitUntilCompleted()
        except BaseException:
            release_buffer(out_mtl)
            raise
        return shared_array(out_mtl, dtype, (total_frames, *frame_shape))

    def _submit_gpu(self, n_frames, frame_bytes, elem_size, out_mtl, max_blocks):
        """Encode LZ4 decode and bit unshuffle of ``n_frames`` and commit them.

        Whole-block uint16 frames use one fused kernel that unshuffles straight
        into ``out_mtl``. Other layouts decode LZ4 into a scratch buffer first,
        then unshuffle every complete 8 KiB block and the partial final block
        exactly as the CUDA decoder does. Returns the committed command buffer.
        """
        Metal = metal_module()
        pipelines = metal_pipelines(_SOURCE, _NAMES)
        tail_elements = _bitshuffle_tail_elements(frame_bytes, elem_size)
        command = metal_queue().commandBuffer()
        encoder = command.computeCommandEncoder()
        if elem_size == 2 and tail_elements == 0:
            encoder.setComputePipelineState_(_packed_u16_pipeline())
            encoder.setBuffer_offset_atIndex_(self._compressed_mtl, 0, 0)
            encoder.setBuffer_offset_atIndex_(self._chunk_offsets_mtl, 0, 1)
            encoder.setBuffer_offset_atIndex_(self._block_starts_mtl, 0, 2)
            encoder.setBuffer_offset_atIndex_(self._block_counts_mtl, 0, 3)
            encoder.setBuffer_offset_atIndex_(self._block_offsets_mtl, 0, 4)
            _set_u32(encoder, 5, frame_bytes // 2)
            encoder.setBuffer_offset_atIndex_(out_mtl, 0, 6)
            # The shader's unshuffle loop advances by four SIMD groups.  It
            # therefore requires exactly 4 x 32 = 128 threads: 64 leaves half
            # the bit-plane groups unwritten, while 256 overlaps group writes.
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, 1, max_blocks),
                Metal.MTLSizeMake(128, 1, 1),
            )
            encoder.endEncoding()
            command.commit()
            return command
        if self._lz4_mtl is None:
            self._lz4_mtl = allocate_shared(self.max_frames * self.frame_bytes, "LZ4 scratch")
        lz4_mtl = self._lz4_mtl

        # LZ4 decode: frames along X, each frame's 8 KiB blocks along Z,
        # _LZ4_BLOCKS_PER_GROUP blocks per threadgroup. The barrier lets the
        # unshuffle read complete blocks.
        encoder.setComputePipelineState_(pipelines["h5lz4dc_batched"])
        encoder.setBuffer_offset_atIndex_(self._compressed_mtl, 0, 0)
        encoder.setBuffer_offset_atIndex_(self._chunk_offsets_mtl, 0, 1)
        encoder.setBuffer_offset_atIndex_(self._block_starts_mtl, 0, 2)
        encoder.setBuffer_offset_atIndex_(self._block_counts_mtl, 0, 3)
        encoder.setBuffer_offset_atIndex_(self._block_offsets_mtl, 0, 4)
        _set_u32(encoder, 5, 8192)
        _set_u32(encoder, 6, frame_bytes)
        encoder.setBuffer_offset_atIndex_(lz4_mtl, 0, 7)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(
                n_frames, 1,
                (max_blocks + _LZ4_BLOCKS_PER_GROUP - 1) // _LZ4_BLOCKS_PER_GROUP,
            ),
            Metal.MTLSizeMake(32, _LZ4_BLOCKS_PER_GROUP, 1),
        )
        encoder.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)

        # uint8 frames: one kernel unshuffles complete and partial blocks.
        if elem_size == 1:
            encoder.setComputePipelineState_(pipelines["shuf_8_batched"])
            encoder.setBuffer_offset_atIndex_(lz4_mtl, 0, 0)
            encoder.setBuffer_offset_atIndex_(out_mtl, 0, 1)
            _set_u32(encoder, 2, frame_bytes)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, 1, (frame_bytes + 255) // 256),
                Metal.MTLSizeMake(256, 1, 1),
            )
            encoder.endEncoding()
            command.commit()
            return command

        # Unshuffle every complete 8 KiB block: 32 SIMD groups per
        # threadgroup, frames along X.
        if elem_size == 2:
            groups_per_block = 8192 // (elem_size * 32)
            frame_elems = frame_bytes // 2
            unshuffle_pipeline = pipelines["shuf_8192_16_batched"]
        else:
            groups_per_block = 2048 // 32
            frame_elems = frame_bytes // 4
            unshuffle_pipeline = pipelines["shuf_8192_32_batched"]
        groups_per_frame = (frame_bytes // 8192) * groups_per_block
        encoder.setComputePipelineState_(unshuffle_pipeline)
        encoder.setBuffer_offset_atIndex_(lz4_mtl, 0, 0)
        encoder.setBuffer_offset_atIndex_(out_mtl, 0, 1)
        _set_u32(encoder, 2, frame_elems)
        _set_u32(encoder, 3, groups_per_block)
        _set_u32(encoder, 4, groups_per_frame)
        if groups_per_frame:
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, 1, (groups_per_frame + 31) // 32),
                Metal.MTLSizeMake(32, 32, 1),
            )

        # The partial final block follows the canonical CUDA tail unshuffle.
        if tail_elements:
            encoder.setComputePipelineState_(
                pipelines["shuf_tail_16_batched" if elem_size == 2 else "shuf_tail_32_batched"]
            )
            encoder.setBuffer_offset_atIndex_(lz4_mtl, 0, 0)
            encoder.setBuffer_offset_atIndex_(out_mtl, 0, 1)
            _set_u32(encoder, 2, frame_bytes)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, 1, (tail_elements + 255) // 256),
                Metal.MTLSizeMake(256, 1, 1),
            )
        encoder.endEncoding()
        command.commit()
        return command


@cache
def _packed_u16_pipeline():
    """Compile the scratch-free uint16 decode kernel on first use.

    The fused full-uint16 path decodes LZ4 and unshuffles bit planes in one
    dispatch. Its kernel lives in ``qh5idx.metal``, which native Swift also
    compiles, so it is not part of the bslz4 library.
    """
    name = "h5lz4dc_unshuffle_u16_single_block_packed_h5"
    return metal_pipelines((_KERNELS / "qh5idx.metal").read_text(), (name,))[name]


def _set_u32(encoder, index: int, value: int) -> None:
    """Bind one uint32 scalar kernel argument at buffer ``index``."""
    encoder.setBytes_length_atIndex_(
        np.array([value], dtype=np.uint32).tobytes(), 4, index
    )


def _bitshuffle_tail_elements(frame_bytes: int, elem_size: int) -> int:
    """Return an admissible final partial-block element count."""
    if int(elem_size) not in (1, 2, 4):
        raise ValueError(
            "MPS bitshuffle/LZ4 GPU decode supports 1-byte uint8, 2-byte "
            "uint16 and 4-byte uint32 source elements. Use the CPU backend or "
            "repack this source into a supported integer dtype."
        )
    tail_bytes = int(frame_bytes) % 8192
    if tail_bytes == 0:
        return 0
    if tail_bytes % int(elem_size):
        raise ValueError(
            "MPS bitshuffle/LZ4 load cannot exactly decode a partial final "
            f"block of {tail_bytes} bytes for {elem_size}-byte elements. Use "
            "the CPU backend or repack this source."
        )
    tail_elements = tail_bytes // int(elem_size)
    if tail_elements % 8:
        raise ValueError(
            "MPS bitshuffle/LZ4 load supports a partial final block only "
            "when it contains a multiple of 8 elements; got "
            f"{tail_elements}. Use the CPU backend or repack this source."
        )
    return tail_elements
