"""Exact integer spatial indexes shared by native Metal and Python MPS."""

import math

import numpy as np

from ._ans import MPSANSArray
from .packed import _allocate_shared, _buffer_view, _complete, _release


def _encode(resident, command, name, buffers, parameters, count, *, groups=False):
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(resident._pipelines[name])
    for index, buffer in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buffer, 0, index)
    data = np.asarray(parameters, np.uint32).tobytes()
    encoder.setBytes_length_atIndex_(data, len(data), len(buffers))
    size = resident._metal.MTLSizeMake
    if groups:
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            size(count, 1, 1), size(32, 1, 1)
        )
    else:
        resident._dispatch_threads(encoder, count)
    encoder.endEncoding()


def _allocate(resident, size, label):
    return _allocate_shared(resident._device, resident._metal, max(4, size), label)


def _geometry(resident):
    rows, cols = resident.shape[2:]
    leaves = math.ceil(rows / 8) * math.ceil(cols / 8)
    roots = math.ceil(rows / 32) * math.ceil(cols / 32)
    return rows, cols, leaves, roots


def build_index(resident, raw, scans):
    """Pack exact sums from one native-count staging window on Metal."""
    rows, cols, leaves, roots = _geometry(resident)
    fields = leaves + roots
    streams = math.ceil(scans / 512) * fields
    buffers = []
    try:
        for size in (scans * fields * 4, streams, streams * 4, (streams + 1) * 8):
            buffers.append(_allocate(resident, size, "camera index staging"))
        values, widths, lengths, starts = buffers
        command = resident._queue.commandBuffer()
        _encode(
            resident,
            command,
            "camera_fields",
            (raw, resident._valid, values),
            (scans, rows, cols, resident.dtype.itemsize),
            scans * fields,
            groups=True,
        )
        _encode(
            resident,
            command,
            "camera_field_widths",
            (values, widths, lengths),
            (scans, fields),
            streams,
        )
        _complete(command, "camera field widths")
        offsets = np.frombuffer(_buffer_view(starts, (streams + 1) * 8), np.uint64)
        offsets[0] = 0
        np.cumsum(
            np.frombuffer(_buffer_view(lengths, streams * 4), np.uint32),
            dtype=np.uint64,
            out=offsets[1:],
        )
        words = _allocate(resident, int(offsets[-1]) * 4, "camera packed fields")
        buffers.append(words)
        command = resident._queue.commandBuffer()
        _encode(
            resident,
            command,
            "camera_pack_fields",
            (values, widths, starts, words),
            (scans, fields),
            streams,
        )
        _complete(command, "camera field packing")
        buffers.remove(words)
        buffers.remove(starts)
        buffers.remove(widths)
        return words, starts, widths
    finally:
        for buffer in buffers:
            _release(buffer)


def detector_sum(resident, mask):
    """Return exact uint64 sums while decoding only boundary residual columns."""
    rows, cols, leaves, roots = _geometry(resident)
    fields, pixels = leaves + roots, rows * cols
    buffers = []
    output = MPSANSArray(
        resident._device, resident._metal, resident.shape[:2], np.uint64
    )
    try:
        for size in (
            pixels,
            leaves * 4,
            fields * 4,
            fields * 4,
            pixels * 4,
            pixels * 4,
            8,
        ):
            buffers.append(_allocate(resident, size, "camera mask plan"))
        mask_buffer, leaf_buffer, selected, signs, edge_ids, edge_signs, counts = (
            buffers
        )
        _buffer_view(mask_buffer, pixels)[:] = memoryview(
            np.ascontiguousarray(mask, np.uint8)
        ).cast("B")
        _buffer_view(counts, 8)[:] = b"\0" * 8
        command = resident._queue.commandBuffer()
        _encode(
            resident,
            command,
            "camera_mask_leaves",
            (mask_buffer, resident._valid, leaf_buffer, edge_ids, edge_signs, counts),
            (rows, cols),
            leaves,
            groups=True,
        )
        _encode(
            resident,
            command,
            "camera_mask_roots",
            (leaf_buffer, selected, signs, counts),
            (rows, cols),
            roots,
            groups=True,
        )
        _complete(command, "camera mask plan")
        field_count, edge_count = map(
            int, np.frombuffer(_buffer_view(counts, 8), np.uint32)
        )
        if field_count > fields or edge_count > pixels:
            raise ValueError("Invalid camera spatial mask plan.")
        resident._clear_errors()
        command = resident._queue.commandBuffer()
        encoder = command.computeCommandEncoderWithDispatchType_(
            resident._metal.MTLDispatchTypeConcurrent
        )
        encoder.setComputePipelineState_(
            resident._pipelines["camera_index_sum_u64_simd"]
        )
        for index, buffer in enumerate((selected, signs, output.buffer), 3):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        size = resident._metal.MTLSizeMake
        for chunk, spatial in zip(resident.chunks, resident.spatial_chunks):
            for index, buffer in enumerate(spatial):
                encoder.setBuffer_offset_atIndex_(buffer, 0, index)
            parameters = np.asarray(
                (chunk.scans, fields, field_count, chunk.first), np.uint32
            ).tobytes()
            encoder.setBytes_length_atIndex_(parameters, len(parameters), 6)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                size(chunk.scans, 1, 1), size(32, 1, 1)
            )
        encoder.endEncoding()
        if edge_count:
            _encode_delta(
                resident, command, edge_ids, edge_signs, output.buffer, edge_count
            )
        _complete(command, "camera indexed detector")
        resident._check_errors()
        return output
    except BaseException:
        output.release()
        raise
    finally:
        for buffer in buffers:
            _release(buffer)


def detector_delta(resident, mask, previous, output):
    """Decode changed columns only and add their signed counts on Metal."""
    masks = [np.asarray(value) for value in (mask, previous)]
    if any(
        value.shape != resident.shape[2:] or not np.all((value == 0) | (value == 1))
        for value in masks
    ):
        raise ValueError(
            "Current and previous masks must be binary and match the detector shape."
        )
    if (
        output.is_released
        or output.shape != resident.shape[:2]
        or output.dtype != np.uint64
    ):
        raise ValueError(
            "Pass the live uint64 output returned by detector_delta_device."
        )
    difference = (
        (masks[0].astype(np.int8) - masks[1].astype(np.int8)) * resident.valid_pixels
    ).ravel()
    selected = np.flatnonzero(difference).astype(np.uint32)
    if not len(selected):
        return output
    if len(selected) > 4096:
        fresh = resident.detector_sum_device(mask)
        try:
            command = resident._queue.commandBuffer()
            blit = command.blitCommandEncoder()
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
                fresh.buffer, 0, output.buffer, 0, output.nbytes
            )
            blit.endEncoding()
            _complete(command, "camera detector rebase")
        finally:
            fresh.release()
        return output
    from ._streamed import _upload

    buffers = []
    try:
        ids = _upload(
            resident._device, resident._metal, selected, "camera changed pixels"
        )
        buffers.append(ids)
        signs = _upload(
            resident._device,
            resident._metal,
            difference[selected].astype(np.int32),
            "camera changed signs",
        )
        buffers.append(signs)
        resident._clear_errors()
        command = resident._queue.commandBuffer()
        _encode_delta(resident, command, ids, signs, output.buffer, len(selected))
        _complete(command, "camera detector delta")
        resident._check_errors()
        return output
    finally:
        for buffer in buffers:
            _release(buffer)


def _encode_delta(resident, command, ids, signs, output, count):
    """Submit independent chunk-column jobs without per-chunk staging."""
    encoder = command.computeCommandEncoderWithDispatchType_(
        resident._metal.MTLDispatchTypeConcurrent
    )
    encoder.setComputePipelineState_(resident._pipelines["camera_delta_u64"])
    for index, buffer in enumerate(
        (resident._decoding, resident._errors, ids, signs, output), 3
    ):
        encoder.setBuffer_offset_atIndex_(buffer, 0, index)
    size = resident._metal.MTLSizeMake
    for chunk in resident.chunks:
        for index, buffer in enumerate(chunk.buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        parameters = np.asarray(
            (
                chunk.scans,
                math.prod(resident.shape[2:]),
                count,
                chunk.first,
            ),
            np.uint32,
        ).tobytes()
        encoder.setBytes_length_atIndex_(parameters, len(parameters), 8)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            size(math.ceil(chunk.scans / 512) * math.ceil(count / 32), 1, 1),
            size(32, 1, 1),
        )
    encoder.endEncoding()
