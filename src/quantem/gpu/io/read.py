"""Bounded reads of a loaded acquisition as Torch tensors on its own GPU.

A complete scan rarely fits as a dense array, so a region is decoded in scan
blocks of at most 256 MiB and only the requested detector pixels are kept.
"""

import math
from functools import partial

import numpy as np
import torch

from quantem.gpu.resident.cuda import precision as cuda_precision
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.resident.float_ans import MAX_DECODE_BYTES, FloatANSResident
from quantem.gpu.resident.mps import precision as metal_precision
from quantem.gpu.resident.mps.arrays import MetalArray
from quantem.gpu.resident.mps.counts import MPSStreamedCounts

_BLOCK_BYTES = 256 * 1024**2


def read(data, *, scan_region=None, detector_region=None):
    """Return a requested logical region as a Torch tensor on the source GPU."""
    shape = tuple(data.shape)
    if len(shape) != 4:
        raise ValueError(f"read() requires 4D-STEM shape; got {shape}.")
    row0, row1, column0, column1 = _region(scan_region, shape[:2], "scan_region")
    detector_row0, detector_row1, detector_column0, detector_column1 = _region(
        detector_region, shape[2:], "detector_region"
    )
    payload = data.data
    device = resident_device(payload)
    if device is None or device.type not in {"cuda", "mps"}:
        raise TypeError("read() requires a CUDA or MPS resident source.")
    output_shape = (
        row1 - row0,
        column1 - column0,
        detector_row1 - detector_row0,
        detector_column1 - detector_column0,
    )
    _check_allocation(output_shape, data.dtype, device)
    if torch.is_tensor(payload):
        return payload[
            row0:row1,
            column0:column1,
            detector_row0:detector_row1,
            detector_column0:detector_column1,
        ].contiguous()

    # Bound decoded scratch by scan blocks. Native CUDA streamed counts select
    # detector pixels while decoding; every other resident decodes whole frames.
    # Calibrated subclasses decode differently, so only the exact type crops.
    selected_region = None
    if type(payload) is StreamedCounts:
        selected_region = (
            detector_row0, detector_row1, detector_column0, detector_column1
        )
    frame_shape = output_shape[2:] if selected_region else shape[2:]
    frame_bytes = math.prod(frame_shape) * np.dtype(data.dtype).itemsize
    decode_bytes = (
        min(_BLOCK_BYTES, MAX_DECODE_BYTES)
        if isinstance(payload, FloatANSResident)
        else _BLOCK_BYTES
    )
    block_columns = min(column1 - column0, max(1, decode_bytes // frame_bytes))
    block_rows = max(1, decode_bytes // (frame_bytes * block_columns))
    tensor = None
    for block_row0 in range(row0, row1, block_rows):
        block_row1 = min(row1, block_row0 + block_rows)
        for block_column0 in range(column0, column1, block_columns):
            block_column1 = min(column1, block_column0 + block_columns)
            block = _decode_rows(
                payload, shape, device, block_row0, block_row1,
                block_column0, block_column1, detector_region=selected_region,
            )
            if selected_region is None:
                block = block[
                    :, detector_row0:detector_row1, detector_column0:detector_column1
                ]
            if tensor is None:
                tensor = torch.empty(
                    output_shape, dtype=block.dtype, device=block.device
                )
            tensor[
                block_row0 - row0 : block_row1 - row0,
                block_column0 - column0 : block_column1 - column0,
            ] = block.reshape(
                block_row1 - block_row0,
                block_column1 - block_column0,
                *output_shape[2:],
            )
            del block
    return tensor


def resident_device(payload):
    """Normalize the device a resident reports, without decoding it.

    Residents report their GPU as a ``torch.device``, a CUDA ordinal, a CuPy
    device, a ``_device_id`` attribute or a string such as ``"mps"``, and
    callers may wrap owners of their own, so every spelling is accepted here.
    ``None`` means the payload reports no device.
    """
    device = getattr(payload, "device", None)
    if not isinstance(device, torch.device):
        device_id = getattr(payload, "_device_id", None)
        if device_id is None:
            device_id = getattr(device, "id", None)
        if device_id is not None:
            device = torch.device("cuda", int(device_id))
        elif isinstance(device, (int, np.integer)):
            device = torch.device("cuda", int(device))
        else:
            device = torch.device(str(device)) if device is not None else None
    return device


def _decode_rows(
    payload, shape, device, row0, row1, column0, column1, *, detector_region=None
):
    """Decode scan rows [row0, row1) and columns [column0, column1).

    Each resident type has one bounded decoder: streamed counts and float ANS
    decode a scan range (exact CUDA streamed counts also crop
    ``detector_region``), precision and Metal streamed residents decode a
    range straight into a tensor, and plain device arrays are sliced. Returns a
    ``(positions, detector_row, detector_col)`` tensor in row-major scan order.
    """
    decoded_to_tensor = (
        (cuda_precision.PrecisionSource,) if device.type == "cuda"
        else (metal_precision.PrecisionSource, MPSStreamedCounts)
    )
    if isinstance(payload, (StreamedCounts, FloatANSResident)):
        decode = payload.decode_scan_range_device
        if detector_region is not None:
            decode = partial(decode, detector_region=detector_region)
    elif isinstance(payload, decoded_to_tensor):
        decode = payload._decode_scan_range_torch
    else:
        # A plain device array, such as a dense CuPy acquisition.
        try:
            return _torch_value(payload[row0:row1, column0:column1]).reshape(
                -1, *shape[2:]
            )
        except (TypeError, IndexError, KeyError, RuntimeError) as error:
            raise TypeError(
                "This loaded representation does not support bounded scan reads."
            ) from error
    # Range decoders take flat row-major scan indices, so a block narrower than
    # the scan decodes one range per scan row.
    if column0 == 0 and column1 == shape[1]:
        return _torch_value(decode(row0 * shape[1], row1 * shape[1]))
    parts = [
        _torch_value(decode(row * shape[1] + column0, row * shape[1] + column1))
        for row in range(row0, row1)
    ]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def _torch_value(value):
    """Wrap one decoded block as a Torch tensor on its GPU without a host copy.

    CUDA decoders return CuPy arrays, shared through DLPack. Metal decoders
    return shared buffers, copied into a Torch MPS tensor and then released.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type not in {"cuda", "mps"}:
            raise TypeError("read() requires accelerator-resident data.")
        return value
    if isinstance(value, MetalArray):
        try:
            tensor = value.to_torch()
            torch.mps.synchronize()
            return tensor
        finally:
            value.release()
    try:
        tensor = torch.from_dlpack(value)
    except (RuntimeError, TypeError, AttributeError, BufferError) as error:
        raise TypeError(
            "This loaded representation does not provide accelerator-native bounded reads."
        ) from error
    if tensor.device.type not in {"cuda", "mps"}:
        raise TypeError("read() never materializes detector values on CPU.")
    return tensor


def _region(value, shape, name):
    """Validate one ``(row_start, row_stop, column_start, column_stop)`` region.

    ``None`` selects the whole axis pair, so a caller can crop only the scan or
    only the detector.
    """
    if value is None:
        return (0, shape[0], 0, shape[1])
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 4
        or any(not isinstance(item, (int, np.integer)) for item in value)
    ):
        raise TypeError(
            f"{name} must be (row_start, row_stop, column_start, column_stop)."
        )
    row0, row1, column0, column1 = map(int, value)
    if not (0 <= row0 < row1 <= shape[0] and 0 <= column0 < column1 <= shape[1]):
        raise ValueError(f"{name}={tuple(value)} must lie within {shape}.")
    return row0, row1, column0, column1


def _check_allocation(shape, dtype, device):
    """Refuse a region whose dense tensor would not fit the GPU's free memory.

    The decoder needs scratch beside the output, so 256 MiB stays in reserve;
    without the check a large region fails midway with an allocator error.
    """
    requested = math.prod(shape) * np.dtype(dtype).itemsize
    reserve = 256 * 1024**2
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if requested + reserve > free:
            raise MemoryError(
                f"The requested region needs about {requested / 2**30:.2f} GiB "
                f"but only {free / 2**30:.2f} GiB is free on {device}. "
                "Request a smaller scan_region."
            )
    elif device.type == "mps":
        available = (
            torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory()
        )
        if requested + reserve > available:
            raise MemoryError(
                f"The requested region needs about {requested / 2**30:.2f} GiB "
                "beyond the available MPS working set. Request a smaller scan_region."
            )
