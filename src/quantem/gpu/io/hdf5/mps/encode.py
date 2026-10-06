"""Metal bitshuffle+LZ4 compression of detector frame batches into HDF5 chunk bytes.

Two encoders exist because they read different inputs. The native PyObjC
pipeline compresses 16-bit codes already held in a shared Metal buffer
(precision exports) without a Torch round trip; its kernels shuffle 16-bit
values only. The MLX kernels compress Torch MPS tensors of every save dtype
(uint8, uint16, float16 and float32 output), which ``io.save`` needs for Metal
arrays. Both pack the compressed blocks into HDF5 chunks on the host.
"""

import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit, prange

from quantem.gpu.device.metal_runtime import (
    allocate_shared,
    metal_module,
    metal_pipelines,
    metal_queue,
    numpy_view,
    release_buffer,
)
from quantem.gpu.formats.hdf5.frames import BLOCK_SIZE

try:
    import torch
except ImportError:  # pragma: no cover - only for minimal IO-only installs
    torch = None

_MPS_LZ4_HASH_SIZE = 2048
_MPS_LZ4_HASH_SHIFT = 21
_MPS_LZ4_MAX_OUT = BLOCK_SIZE + 1024
_SAVE_SOURCE = (Path(__file__).parent / "kernels" / "save_uint16.msl").read_text()


class NativeU16Compressor:
    """Reusable native Metal compressor for 16-bit frame batches held in shared Metal buffers."""

    def __init__(self, max_frames: int, frame_bytes: int, n_8kb: int):
        self.max_frames = int(max_frames)
        self.frame_bytes = int(frame_bytes)
        self.n_8kb = int(n_8kb)
        self.max_out = int(_MPS_LZ4_MAX_OUT)
        self.max_blocks = self.max_frames * self.n_8kb
        self.shuffled = allocate_shared(self.max_frames * self.frame_bytes, "shuffled frames")
        self.comp = allocate_shared(self.max_blocks * self.max_out, "compressed blocks")
        self.sizes = allocate_shared(self.max_blocks * 4, "compressed block sizes")
        self.comp_np = numpy_view(self.comp, np.uint8, self.max_blocks * self.max_out)
        self.sizes_np = numpy_view(self.sizes, np.uint32, self.max_blocks)
        self.offsets = np.arange(self.max_blocks, dtype=np.int64) * self.max_out
        self._start_buf, self._start_np = self._u32_buffer()
        self._n_buf, self._n_np = self._u32_buffer()
        self._frame_bytes_buf, self._frame_bytes_np = self._u32_buffer(self.frame_bytes)
        self._n_chunks_buf, self._n_chunks_np = self._u32_buffer()
        self._n_8kb_buf, self._n_8kb_np = self._u32_buffer(self.n_8kb)
        self._max_out_buf, self._max_out_np = self._u32_buffer(self.max_out)
        self._buffers = [
            self.shuffled,
            self.comp,
            self.sizes,
            self._start_buf,
            self._n_buf,
            self._frame_bytes_buf,
            self._n_chunks_buf,
            self._n_8kb_buf,
            self._max_out_buf,
        ]

    def _u32_buffer(self, value: int = 0):
        """One uint32 kernel argument in its own shared buffer, with a host view to update it."""
        buf = allocate_shared(4, "kernel argument")
        view = numpy_view(buf, np.uint32, 1)
        view[0] = np.uint32(value)
        return buf, view

    def close(self) -> None:
        for buf in self._buffers:
            release_buffer(buf)
        self._buffers = []

    def __del__(self):
        try:
            self.close()
        except (AttributeError, NameError, TypeError):
            pass

    def compress(self, chunk, frame_offset: int, n_frames: int):
        """Compress frames ``frame_offset`` to ``frame_offset + n_frames`` of ``chunk``'s buffer."""
        n_frames = int(n_frames)
        if n_frames > self.max_frames:
            raise ValueError(
                f"native MPS compressor was allocated for {self.max_frames} "
                f"frames, got {n_frames}."
            )
        metal = metal_module()
        pipelines = metal_pipelines(_SAVE_SOURCE, ("bshuf_u16_save", "lz4_rle_save"))
        n_blocks = n_frames * self.n_8kb
        self._start_np[0] = np.uint32(frame_offset)
        self._n_np[0] = np.uint32(n_frames)
        self._n_chunks_np[0] = np.uint32(n_blocks)

        cmd = metal_queue().commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pipelines["bshuf_u16_save"])
        enc.setBuffer_offset_atIndex_(chunk._mtl, 0, 0)
        enc.setBuffer_offset_atIndex_(self.shuffled, 0, 1)
        enc.setBuffer_offset_atIndex_(self._start_buf, 0, 2)
        enc.setBuffer_offset_atIndex_(self._n_buf, 0, 3)
        enc.setBuffer_offset_atIndex_(self._frame_bytes_buf, 0, 4)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            metal.MTLSizeMake(
                (n_frames * self.frame_bytes + 255) // 256, 1, 1
            ),
            metal.MTLSizeMake(256, 1, 1),
        )
        enc.setComputePipelineState_(pipelines["lz4_rle_save"])
        enc.setBuffer_offset_atIndex_(self.shuffled, 0, 0)
        enc.setBuffer_offset_atIndex_(self.comp, 0, 1)
        enc.setBuffer_offset_atIndex_(self.sizes, 0, 2)
        enc.setBuffer_offset_atIndex_(self._n_chunks_buf, 0, 3)
        enc.setBuffer_offset_atIndex_(self._frame_bytes_buf, 0, 4)
        enc.setBuffer_offset_atIndex_(self._n_8kb_buf, 0, 5)
        enc.setBuffer_offset_atIndex_(self._max_out_buf, 0, 6)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            metal.MTLSizeMake(n_blocks, 1, 1),
            metal.MTLSizeMake(32, 1, 1),
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return pack_chunks(
            self.comp_np,
            self.sizes_np[:n_blocks].copy(),
            self.offsets[:n_blocks],
            n_frames,
            self.n_8kb,
            self.frame_bytes,
        )


def compress_tensor_batch(data_mps, n_8kb, frame_bytes, output_dtype):
    """Compress a torch MPS 16-bit/32-bit batch to HDF5 bslz4 chunks."""
    import mlx.core as mx

    if not (torch.is_tensor(data_mps) and data_mps.device.type == "mps"):
        raise TypeError("compress_tensor_batch expects a torch tensor on MPS")
    output_dtype = np.dtype(output_dtype)
    if data_mps.dtype not in (torch.uint8, torch.uint16, torch.float16, torch.float32):
        raise TypeError(
            "MPS compressed save supports uint8/uint16/float32 input, "
            f"got {data_mps.dtype}"
        )
    if output_dtype not in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError(
            "MPS compressed save supports uint8/uint16/float32 output, "
            f"got {output_dtype}"
        )
    if output_dtype == np.dtype(np.float32) and data_mps.dtype != torch.float32:
        raise TypeError("MPS float32 compressed save requires float32 input data")
    if data_mps.dtype == torch.uint8 and output_dtype != np.dtype(np.uint8):
        raise TypeError("MPS uint8 input can only be saved as uint8")
    if data_mps.dtype == torch.float16:
        data_mps = data_mps.view(torch.uint16)
    itemsize = int(output_dtype.itemsize)
    tail_bytes = frame_bytes % BLOCK_SIZE
    if tail_bytes:
        tail_items = tail_bytes // itemsize
        if tail_bytes % itemsize or tail_items % 8:
            raise ValueError(
                "MPS bitshuffle/LZ4 save supports partial final blocks only "
                "when the partial detector frame contains a multiple of 8 "
                f"elements; got frame_bytes={frame_bytes}, output dtype={output_dtype}."
            )

    n = int(data_mps.shape[0])
    frame_elems = frame_bytes // itemsize
    total = n * frame_bytes
    data_mps = data_mps.reshape(n, -1).contiguous()
    # Torch and MLX use separate Metal command queues. The DLPack capsule
    # shares storage but does not make an asynchronous Torch cat/cast visible
    # to MLX; synchronize before the MLX bitshuffle kernel consumes it.
    torch.mps.synchronize()
    empty_u8 = mx.zeros((1,), dtype=mx.uint8)
    empty_u16 = mx.zeros((1,), dtype=mx.uint16)
    empty_f32 = mx.zeros((1,), dtype=mx.float32)
    src_u8 = mx.from_dlpack(data_mps) if data_mps.dtype == torch.uint8 else empty_u8
    src_u16 = mx.from_dlpack(data_mps) if data_mps.dtype == torch.uint16 else empty_u16
    src_f32 = mx.from_dlpack(data_mps) if data_mps.dtype == torch.float32 else empty_f32

    bitshuffle = _mps_bitshuffle_fwd_u16_kernel()
    shuffled = bitshuffle(
        inputs=[src_u8, src_u16, src_f32],
        template=[
            ("TOTAL", total),
            ("FRAME_BYTES", frame_bytes),
            ("FRAME_ELEMS", frame_elems),
            ("OUT_BITS", itemsize * 8),
            ("OUT_BYTES", itemsize),
            ("INPUT_U8", 1 if data_mps.dtype == torch.uint8 else 0),
            ("INPUT_FLOAT", 1 if data_mps.dtype == torch.float32 else 0),
            ("OUTPUT_FLOAT", 1 if output_dtype == np.dtype(np.float32) else 0),
        ],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(total,)],
        output_dtypes=[mx.uint8],
    )[0]
    mx.eval(shuffled)

    max_out = _MPS_LZ4_MAX_OUT
    n_blocks = n * n_8kb
    # Shuffled uint16 counts are long zero runs; the run-length encoder is
    # faster there, while the hash encoder also finds repeats in other dtypes.
    if output_dtype == np.dtype(np.uint16):
        lz4 = _mps_lz4_rle_compress_kernel()
        template = [
            ("N_CHUNKS", n_blocks),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("FRAME_BYTES", frame_bytes),
            ("BLOCKS_PER_FRAME", n_8kb),
            ("MAX_OUT", max_out),
            ("THREADS", 32),
        ]
    else:
        lz4 = _mps_lz4_compress_kernel()
        template = [
            ("N_CHUNKS", n_blocks),
            ("BLOCK_SIZE", BLOCK_SIZE),
            ("FRAME_BYTES", frame_bytes),
            ("BLOCKS_PER_FRAME", n_8kb),
            ("MAX_OUT", max_out),
            ("THREADS", 32),
            ("HASH_SIZE", _MPS_LZ4_HASH_SIZE),
            ("HASH_SHIFT", _MPS_LZ4_HASH_SHIFT),
        ]
    comp_buf, sizes = lz4(
        inputs=[shuffled],
        template=template,
        grid=(n_blocks * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n_blocks * max_out,), (n_blocks,)],
        output_dtypes=[mx.uint8, mx.uint32],
    )
    mx.eval(comp_buf, sizes)

    comp_np = np.array(comp_buf, copy=False)
    sizes_np = np.array(sizes, copy=False)
    offsets_np = np.arange(n_blocks, dtype=np.int64) * int(max_out)
    return pack_chunks(comp_np, sizes_np, offsets_np, n, n_8kb, frame_bytes)


@lru_cache(maxsize=1)
def _mps_bitshuffle_fwd_u16_kernel():
    """Compile, once per process, the MLX Metal forward bitshuffle kernel."""
    import mlx.core as mx

    source = (Path(__file__).parent / "kernels" / "save_bitshuffle.msl").read_text()
    return mx.fast.metal_kernel(
        name="quantem_bslz4_fwd_u16",
        input_names=["src_u8", "src_u16", "src_f32"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "fast"},
    )


@lru_cache(maxsize=1)
def _mps_lz4_compress_kernel():
    """Compile, once per process, the MLX Metal hash-match LZ4 encoder."""
    import mlx.core as mx

    source = (Path(__file__).parent / "kernels" / "save_lz4_hash.msl").read_text()
    return mx.fast.metal_kernel(
        name="quantem_bslz4_lz4_encode",
        input_names=["shuffled"],
        output_names=["out", "sizes"],
        source=source,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "fast"},
    )


@lru_cache(maxsize=1)
def _mps_lz4_rle_compress_kernel():
    """Compile, once per process, the MLX Metal run-length LZ4 encoder."""
    import mlx.core as mx

    source = (Path(__file__).parent / "kernels" / "save_lz4_rle.msl").read_text()
    return mx.fast.metal_kernel(
        name="quantem_bslz4_lz4_rle_encode",
        input_names=["shuffled"],
        output_names=["out", "sizes"],
        source=source,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "fast"},
    )


def pack_chunks(compact_np, sizes_np, offsets_np, n_frames, n_8kb,
                 frame_bytes):
    """Return packed HDF5 chunk bytes, per-frame starts, and sizes."""
    sizes_2d = sizes_np.reshape(n_frames, n_8kb)
    frame_comp_sizes = sizes_2d.sum(axis=1).astype(np.int64)
    header_overhead = 12 + n_8kb * 4
    chunk_sizes = header_overhead + frame_comp_sizes

    chunk_starts = np.zeros(n_frames + 1, dtype=np.int64)
    np.cumsum(chunk_sizes, out=chunk_starts[1:])
    packed = np.empty(int(chunk_starts[n_frames]), dtype=np.uint8)
    header_bytes = np.frombuffer(
        struct.pack(">QI", frame_bytes, BLOCK_SIZE), dtype=np.uint8
    ).copy()

    _pack_chunks_numba(
        packed, chunk_starts, compact_np,
        sizes_np.astype(np.int64), offsets_np.astype(np.int64),
        header_bytes, n_frames, n_8kb,
    )
    return packed, chunk_starts[:n_frames], chunk_sizes


@njit(cache=True, parallel=True)
def _pack_chunks_numba(packed, chunk_starts, compact, sizes, offsets,
                       header_bytes, n_frames, n_8kb):
    """Pack bitshuffle+LZ4 blocks into HDF5 chunk byte buffers."""
    for i in prange(n_frames):
        dst = chunk_starts[i]
        for h in range(12):
            packed[dst + h] = header_bytes[h]
        pos = dst + 12
        base = i * n_8kb
        for b in range(n_8kb):
            idx = base + b
            sz = sizes[idx]
            packed[pos] = (sz >> 24) & 0xFF
            packed[pos + 1] = (sz >> 16) & 0xFF
            packed[pos + 2] = (sz >> 8) & 0xFF
            packed[pos + 3] = sz & 0xFF
            pos += 4
            off = offsets[idx]
            for j in range(sz):
                packed[pos + j] = compact[off + j]
            pos += sz
