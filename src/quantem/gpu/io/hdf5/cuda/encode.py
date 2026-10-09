"""CUDA bitshuffle+LZ4 compression of detector frame batches into HDF5 chunk bytes.

Saving an acquisition compresses each frame on the GPU with the same 8 KiB
bitshuffle+LZ4 blocks hdf5plugin writes, then packs every frame's blocks
behind the chunk header, so the HDF5 writer only appends finished chunks with
``write_direct_chunk``. Other readers decode the files with the standard filter.
"""

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.formats.hdf5.frames import BLOCK_SIZE
from quantem.gpu.io.hdf5.cuda.decode import kernel


def compress_batch(data_gpu, n_8kb, frame_bytes):
    """Compress a contiguous 16-bit or 32-bit frame batch on GPU."""
    max_out = BLOCK_SIZE * 2
    cuda_max_z = 65535
    frame_count = int(data_gpu.shape[0])
    n_blocks = frame_count * n_8kb
    itemsize = int(data_gpu.dtype.itemsize)
    n_full_8kb = frame_bytes // BLOCK_SIZE
    tail_bytes = frame_bytes % BLOCK_SIZE
    if tail_bytes:
        tail_items = tail_bytes // itemsize
        if tail_bytes % itemsize or tail_items % 8:
            raise ValueError(
                "GPU bitshuffle/LZ4 save supports partial final blocks only "
                "when the partial detector frame contains a multiple of 8 "
                f"elements; got frame_bytes={frame_bytes}, dtype={data_gpu.dtype}."
            )

    shuffled = cp.empty(frame_count * frame_bytes, dtype=cp.uint8)
    if itemsize == 1:
        data_u8 = data_gpu.reshape(frame_count, -1).view(cp.uint8)
        total_bytes = np.uint64(frame_count * frame_bytes)
        kernel("bitshuffle_fwd_8")(
            ((int(total_bytes) + 255) // 256,), (256,),
            (
                data_u8,
                shuffled,
                np.uint32(frame_bytes),
                total_bytes,
            ),
        )
    elif itemsize == 2:
        data_u16 = data_gpu.reshape(frame_count, -1).view(cp.uint16)
        for start in range(0, frame_count, cuda_max_z):
            end = min(start + cuda_max_z, frame_count)
            batch_n = end - start
            out = shuffled[start * frame_bytes:end * frame_bytes]
            if n_full_8kb:
                kernel("bitshuffle_fwd_8192_16")(
                    (n_full_8kb, 16, batch_n), (256, 1, 1),
                    (
                        data_u16[start:end],
                        out,
                        np.uint32(frame_bytes),
                    ),
                )
            if tail_bytes:
                tail_bitplane_bytes = (tail_bytes // itemsize) // 8
                kernel("bitshuffle_fwd_tail_16")(
                    ((tail_bitplane_bytes + 127) // 128, 16, batch_n),
                    (128, 1, 1),
                    (
                        data_u16[start:end],
                        out,
                        np.uint32(frame_bytes),
                    ),
                )
    elif itemsize == 4:
        frame_u32s = frame_bytes // 4
        data_u32 = data_gpu.reshape(frame_count, -1).view(cp.uint32)
        for start in range(0, frame_count, cuda_max_z):
            end = min(start + cuda_max_z, frame_count)
            batch_n = end - start
            out = shuffled[start * frame_bytes:end * frame_bytes]
            if n_full_8kb:
                kernel("bitshuffle_fwd_8192_32")(
                    (n_full_8kb, 2, batch_n), (32, 32, 1),
                    (
                        data_u32[start:end],
                        out.view(cp.uint32),
                        np.uint32(frame_u32s),
                    ),
                )
            if tail_bytes:
                tail_bitplane_bytes = (tail_bytes // itemsize) // 8
                kernel("bitshuffle_fwd_tail_32")(
                    ((tail_bitplane_bytes + 127) // 128, 32, batch_n),
                    (128, 1, 1),
                    (
                        data_u32[start:end],
                        out,
                        np.uint32(frame_bytes),
                    ),
                )
    else:
        raise TypeError(f"Unsupported save dtype itemsize: {itemsize}")

    compressed = cp.empty(n_blocks * max_out, dtype=cp.uint8)
    block_sizes = cp.empty(n_blocks, dtype=cp.uint32)
    if tail_bytes:
        kernel("lz4_compress_var_kernel")(
            (n_blocks,), (32,),
            (
                shuffled,
                compressed,
                block_sizes,
                np.uint32(frame_bytes),
                np.uint32(BLOCK_SIZE),
                np.uint32(max_out),
                np.uint32(n_8kb),
                np.uint32(n_blocks),
            ),
        )
    else:
        kernel("lz4_compress_kernel")(
            (n_blocks,), (32,),
            (shuffled, compressed, block_sizes, np.uint32(BLOCK_SIZE), np.uint32(n_blocks)),
        )
    del shuffled

    return _pack_chunks(compressed, block_sizes, frame_count, n_8kb, frame_bytes, max_out)


def _pack_chunks(compressed, block_sizes, n_frames, n_8kb, frame_bytes, max_out):
    """Pack bitshuffle+LZ4 blocks into HDF5 chunk bytes on GPU."""
    sizes_2d = block_sizes.reshape(n_frames, n_8kb)
    frame_comp_sizes = sizes_2d.sum(axis=1, dtype=cp.uint64)
    header_overhead = np.uint64(12 + n_8kb * 4)
    chunk_sizes_device = frame_comp_sizes + header_overhead
    chunk_starts_device = cp.empty(n_frames + 1, dtype=cp.uint64)
    chunk_starts_device[0] = 0
    chunk_starts_device[1:] = cp.cumsum(chunk_sizes_device)
    packed_bytes = int(chunk_starts_device[-1].get())
    packed_device = cp.empty(packed_bytes, dtype=cp.uint8)
    kernel("pack_h5_chunks_kernel")(
        (n_frames,), (256,),
        (
            compressed,
            block_sizes,
            chunk_starts_device,
            packed_device,
            np.uint32(n_frames),
            np.uint32(n_8kb),
            np.uint32(max_out),
            np.uint64(frame_bytes),
            np.uint32(BLOCK_SIZE),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    packed = packed_device.get()
    # int64, the same chunk table the Metal compressor hands the writer.
    chunk_starts = chunk_starts_device[:-1].get().astype(np.int64, copy=False)
    chunk_sizes = chunk_sizes_device.get().astype(np.int64, copy=False)
    return packed, chunk_starts, chunk_sizes
