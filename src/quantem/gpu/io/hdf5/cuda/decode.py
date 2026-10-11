"""CUDA bitshuffle+LZ4 decoding of prepared HDF5 frames, and the shared CUDA codec kernels.

Bounded encoded and precision loads read a block of compressed HDF5 chunks on
the host (``formats.hdf5.reads.FrameReader.prepare``) and decode it here: LZ4
blocks expand in parallel, then bit planes unshuffle into the stored dtype.
The same kernel module also holds the forward kernels ``io.hdf5.cuda.encode``
uses to write bitshuffle+LZ4 HDF5. Kernels compile on first launch, so
importing this module needs no GPU.
"""

from pathlib import Path

import numpy as np

from quantem.gpu.device.cuda_runtime import _release_pinned, cp, cuda_module
from quantem.gpu.formats.hdf5.frames import BLOCK_SIZE

_SOURCE = (Path(__file__).parent / "kernels" / "bslz4.cu").read_text()
_NAMES = (
    "h5lz4dc_batched",
    "shuf_8192_32_batched",
    "shuf_8192_16_batched",
    "shuf_tail_16_batched",
    "shuf_tail_32_batched",
    "shuf_8_batched",
    "bitshuffle_fwd_8192_32",
    "bitshuffle_fwd_8",
    "bitshuffle_fwd_8192_16",
    "bitshuffle_fwd_tail_16",
    "bitshuffle_fwd_tail_32",
    "lz4_compress_var_kernel",
    "pack_h5_chunks_kernel",
)


def kernel(name: str):
    """Return codec kernel ``name`` compiled for the current device and context."""
    return cuda_module(_SOURCE, _NAMES, ("-std=c++11", "-w"))[name]


def decompress_prepared(prepared: dict, batch_bytes_target: int = 1 << 28):
    """Transfer prepared compressed frames, decode them on the GPU and return the staging buffer.

    Frames of 1-, 2- or 4-byte elements decode in batches of about
    ``batch_bytes_target`` bytes: LZ4 expands a batch into one reused scratch
    buffer and the unshuffle writes it straight into the final result, so peak
    memory is ``compressed + batch + final`` rather than two full
    uncompressed copies. Returns the stored-dtype CuPy array
    ``(n_frames, det_row, det_col)``; stored values are not altered. The
    page-locked staging buffer returns to its pool after the final
    synchronize; a decode that fails before then keeps it, because a queued
    copy may still read it.
    """
    read_buffer = prepared["read_buffer"]
    chunk_offsets = prepared["chunk_offsets"]
    block_starts = prepared["block_starts"]
    block_counts = prepared["block_counts"]
    block_offsets = prepared["block_offsets"]
    total_frames = prepared["total_frames"]
    frame_shape = prepared["frame_shape"]
    frame_bytes = prepared["frame_bytes"]
    source_dtype = prepared["dtype"]
    source_itemsize = int(np.dtype(source_dtype).itemsize)
    n_full_8kb = frame_bytes // BLOCK_SIZE
    tail_bytes = frame_bytes % BLOCK_SIZE
    # Bitshuffle leaves a final remainder of fewer than 8 elements
    # unshuffled, which these decoders do not read.
    if tail_bytes % source_itemsize or (tail_bytes // source_itemsize) % 8:
        _release_pinned(read_buffer)  # nothing was copied from it yet
        raise ValueError(
            "GPU bitshuffle/LZ4 load supports partial final blocks "
            "only when the partial detector frame contains a "
            f"multiple of 8 elements; got frame_shape={frame_shape}."
        )

    # Cap at max_batch=10000 to match the kernel's old launch characteristics
    # and stay well under any plausible single-batch scratch allocation.
    max_batch_from_target = max(1, int(batch_bytes_target // frame_bytes))
    max_batch = min(10000, max_batch_from_target, total_frames)
    n_batches = (total_frames + max_batch - 1) // max_batch

    # Stream + async-overlap the H2D with the kernels whenever there is more
    # than one batch (a single-batch file cannot overlap). Async upload of
    # batch N+1 hides behind batch N's LZ4/bitshuffle work. The streaming
    # slicer requires chunk offsets to be monotonic in output order, so
    # out-of-order selections use one full compressed upload.
    offsets_monotonic = bool(
        total_frames <= 1
        or np.all(chunk_offsets[1:] >= chunk_offsets[:-1])
    )
    streaming_upload = offsets_monotonic and n_batches > 1

    block_starts_device = cp.asarray(block_starts)
    block_counts_device = cp.asarray(block_counts)
    block_offsets_device = cp.asarray(block_offsets)
    if streaming_upload:
        # Precompute every batch's compressed slice + rebased chunk offsets
        # once. block_starts are chunk-relative so they need no rebasing.
        batch_slices = []
        all_rebased = np.empty(total_frames, dtype=np.uint64)
        max_batch_compressed = 0
        for frame_start in range(0, total_frames, max_batch):
            frame_stop = min(frame_start + max_batch, total_frames)
            byte_start = int(chunk_offsets[frame_start])
            byte_stop = (
                len(read_buffer)
                if frame_stop == total_frames
                else int(chunk_offsets[frame_stop])
            )
            all_rebased[frame_start:frame_stop] = (
                chunk_offsets[frame_start:frame_stop] - np.uint64(byte_start)
            )
            max_batch_compressed = max(max_batch_compressed, byte_stop - byte_start)
            batch_slices.append((byte_start, byte_stop))
        all_rebased_device = cp.asarray(all_rebased)
        # Double-buffered async H2D: upload batch N+1 into the spare buffer on
        # a copy stream while batch N's kernels run on the main stream. The
        # read_buffer is page-locked so .set(stream=) is a true async DMA;
        # events keep the copy stream from overwriting a buffer whose LZ4
        # kernel has not finished reading it.
        compressed_slots = [cp.empty(max_batch_compressed, dtype=cp.uint8) for _ in range(2)]
        copy_stream = cp.cuda.Stream(non_blocking=True)
        copy_done = [cp.cuda.Event() for _ in range(2)]
        kernel_done = [cp.cuda.Event() for _ in range(2)]
        main_stream = cp.cuda.get_current_stream()

        def upload_batch(batch_index, slot):
            byte_start, byte_stop = batch_slices[batch_index]
            compressed_slots[slot][: byte_stop - byte_start].set(
                read_buffer[byte_start:byte_stop], stream=copy_stream
            )
            copy_stream.record(copy_done[slot])

        upload_batch(0, 0)
        compressed_device = None
        chunk_offsets_device = None
    else:
        compressed_device = cp.empty(len(read_buffer), dtype=cp.uint8)
        compressed_device.set(read_buffer)
        chunk_offsets_device = cp.asarray(chunk_offsets)
    batch_scratch_bytes = max_batch * frame_bytes
    lz4_scratch = cp.empty(batch_scratch_bytes, dtype=cp.uint8)
    result = cp.empty((total_frames,) + frame_shape, dtype=source_dtype)
    errors = cp.zeros(1, dtype=cp.uint32)
    max_blocks = int(block_counts.max())


    for batch_index, start in enumerate(range(0, total_frames, max_batch)):
        end = min(start + max_batch, total_frames)
        batch_n = end - start
        # Streaming upload (double-buffered async): this batch's bytes are
        # already in flight on the copy stream; wait for them, then prefetch
        # the next batch into the spare buffer so its H2D overlaps this
        # batch's kernels. block_starts are chunk-relative (no rebasing);
        # chunk_offsets were rebased per-batch into all_rebased_device.
        if streaming_upload:
            slot = batch_index % 2
            main_stream.wait_event(copy_done[slot])
            if batch_index + 1 < n_batches:
                next_slot = (batch_index + 1) % 2
                if batch_index >= 1:
                    copy_stream.wait_event(kernel_done[next_slot])
                upload_batch(batch_index + 1, next_slot)
            batch_compressed = compressed_slots[slot]
            batch_chunk_offsets = all_rebased_device[start:]
        else:
            batch_compressed = compressed_device
            batch_chunk_offsets = chunk_offsets_device[start:]

        # 1. LZ4 decompress this batch into lz4_scratch (from offset 0).
        kernel("h5lz4dc_batched")(
            ((max_blocks + 1) // 2, 1, batch_n),
            (32, 2, 1),
            (
                batch_compressed,
                batch_chunk_offsets,
                block_starts_device,
                block_counts_device[start:],
                block_offsets_device[start:],
                np.uint32(BLOCK_SIZE),
                np.uint32(frame_bytes),
                lz4_scratch,
                errors,
            ),
        )
        # LZ4 is the only consumer of the compressed buffer; once it has run
        # the copy stream may refill this slot for batch_index+2.
        if streaming_upload:
            main_stream.record(kernel_done[slot])

        # 2. Unshuffle this batch straight into its frames of the result (the kernels
        #    index from the pointer they are given and batch_n bounds their work); a
        #    second scratch buffer and a copy moved every decoded count twice more.
        decoded = result[start:end]
        if source_itemsize == 1:
            # One kernel unshuffles complete blocks and the final partial block.
            batch_bytes = batch_n * frame_bytes
            kernel("shuf_8_batched")(
                ((batch_bytes + 255) // 256,),
                (256,),
                (lz4_scratch, decoded, np.uint32(frame_bytes), np.uint64(batch_bytes)),
            )
        elif source_itemsize == 2:
            if n_full_8kb:
                kernel("shuf_8192_16_batched")(
                    (n_full_8kb, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_scratch,
                        decoded,
                        np.uint32(frame_bytes),
                    ),
                )
            if tail_bytes:
                tail_elems = tail_bytes // source_itemsize
                kernel("shuf_tail_16_batched")(
                    ((tail_elems + 255) // 256, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_scratch,
                        decoded,
                        np.uint32(frame_bytes),
                    ),
                )
        else:
            frame_u32s = frame_bytes // 4
            if n_full_8kb:
                kernel("shuf_8192_32_batched")(
                    (n_full_8kb, 2, batch_n),
                    (32, 32, 1),
                    (
                        lz4_scratch.view(cp.uint32),
                        decoded,
                        np.uint32(frame_u32s),
                    ),
                )
            if tail_bytes:
                tail_elems = tail_bytes // source_itemsize
                kernel("shuf_tail_32_batched")(
                    ((tail_elems + 255) // 256, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_scratch,
                        decoded,
                        np.uint32(frame_bytes),
                    ),
                )

    cp.cuda.Device().synchronize()
    _release_pinned(read_buffer)
    if int(errors.get()[0]):
        raise ValueError(
            "A compressed HDF5 chunk is truncated or corrupt and did not decode to its "
            "frame size; check that the acquisition finished writing, or copy it again."
        )
    return result
