"""Bounded reads of original measurements as Torch or native GPU arrays.

A complete scan rarely fits as a dense array, so only the requested detector
pixels are kept: whole scan rows of exact integer counts decode straight into the
result, and every other region in scan blocks of at most 256 MiB.
"""

import math
from functools import partial

import numpy as np
import torch

from quantem.gpu.formats.float8 import is_float8, torch_dtype

from quantem.gpu.resident.cuda import precision as cuda_precision
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.resident.bit_ans import MAX_DECODE_BYTES, BitANSResident
from quantem.gpu.resident.mps import precision as metal_precision
from quantem.gpu.resident.mps.arrays import MetalArray
from quantem.gpu.resident.mps.counts import MPSStreamedCounts

_BLOCK_BYTES = 256 * 1024**2


def read(data, *, scan_region=None, detector_region=None, output="torch"):
    """Return a region as Torch or a native array, preserving its original bits."""
    if output not in ("torch", "native"):
        raise ValueError("Choose output='torch' or output='native'.")
    shape = tuple(data.shape)
    if len(shape) != 4:
        raise ValueError(f"read() requires 4D-STEM shape; got {shape}.")
    row0, row1, column0, column1 = _region(scan_region, shape[:2], "scan_region")
    detector_row0, detector_row1, detector_column0, detector_column1 = _region(
        detector_region, shape[2:], "detector_region"
    )
    payload = data.data
    if isinstance(payload, np.ndarray) or (
        isinstance(payload, torch.Tensor) and payload.device.type == "cpu"
    ):
        # CPU readers and scientific algorithms may own arrays or tensors.
        selected = payload[
            row0:row1,
            column0:column1,
            detector_row0:detector_row1,
            detector_column0:detector_column1,
        ]
        return np.array(selected, copy=True) if output == "native" else torch.as_tensor(selected).contiguous()
    device = resident_device(payload)
    if device is None or device.type not in {"cuda", "mps"}:
        raise TypeError("read() requires a CUDA or MPS resident source.")
    metal_double = device.type == "mps" and np.dtype(data.dtype) == np.dtype("float64")
    float8 = is_float8(data.dtype)
    if metal_double and output == "torch":
        raise NotImplementedError(
            "Torch MPS has no float64 tensor. Use read(output='native') for "
            "exact float64 Metal storage, or CUDA for float64 tensor calculations."
        )
    output_shape = (
        row1 - row0,
        column1 - column0,
        detector_row1 - detector_row0,
        detector_column1 - detector_column0,
    )
    _check_allocation(((2,) + output_shape) if output == "native" and device.type == "mps" else output_shape, data.dtype, device)
    if torch.is_tensor(payload):
        result = payload[
            row0:row1,
            column0:column1,
            detector_row0:detector_row1,
            detector_column0:detector_column1,
        ].contiguous()
        return _native_result(result, data.dtype) if output == "native" else result

    # Bound decoded scratch by scan blocks. Native integer residents select
    # detector pixels while decoding; every other resident decodes whole frames.
    # Calibrated subclasses decode differently, so only the exact type crops.
    selected_region = None
    if type(payload) in (StreamedCounts, MPSStreamedCounts):
        selected_region = (
            detector_row0, detector_row1, detector_column0, detector_column1
        )
        if column0 == 0 and column1 == shape[1]:
            # Whole scan rows are one consecutive scan range, which the decoder writes
            # cropped into its output: that output is the result. Block by block, every
            # count was decoded into scratch and copied again (an SSB bright-field crop
            # of a 512 x 512 scan spent 45% of its read on the copies).
            decode = (
                payload.decode_scan_range_torch if device.type == "mps"
                else payload.decode_scan_range_device
            )
            counts = decode(
                row0 * shape[1], row1 * shape[1], detector_region=selected_region
            )
            result = _as_dtype(torch_value(counts).reshape(output_shape), data.dtype)
            return _native_result(result, data.dtype) if output == "native" else result
    frame_shape = output_shape[2:] if selected_region else shape[2:]
    frame_bytes = math.prod(frame_shape) * np.dtype(data.dtype).itemsize
    if isinstance(payload, BitANSResident):
        frame_bytes = payload.decode_frame_bytes
    decode_bytes = (
        min(_BLOCK_BYTES, MAX_DECODE_BYTES)
        if isinstance(payload, BitANSResident)
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
                reinterpret_dtype=np.int64 if metal_double else (np.uint8 if float8 else None),
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
            # Free this block before the next decode, so one block of scratch is live.
            del block
    tensor = _as_dtype(tensor, data.dtype)
    if output == "native":
        return _native_result(tensor, data.dtype)
    return tensor.view(torch_dtype(data.dtype)) if float8 else tensor


def _as_dtype(tensor, dtype):
    """Cast counts held in a narrower unsigned type to the dtype the dataset reports; values are unchanged.

    A load with ``dtype="native"`` reports the source file's dtype for counts
    stored narrower. Torch casts through int32, which every device supports.
    """
    target = np.dtype(dtype)
    if target.kind != "u" or str(tensor.dtype).removeprefix("torch.") == target.name:
        return tensor
    wide = tensor.to(torch.int32)
    if target.itemsize == 4:
        return wide.view(torch.uint32)
    return wide.to(torch.int16).view(torch.uint16) if target.itemsize == 2 else wide.to(torch.uint8)


def _native_result(tensor, dtype):
    """Return an independently owned native array, without a host transfer."""
    if tensor.device.type == "cuda":
        from quantem.gpu.device.cuda_runtime import cp

        with cp.cuda.Device(tensor.device.index):
            if is_float8(dtype):
                from quantem.gpu.resident.cuda.float8 import Float8Array

                return Float8Array(cp.from_dlpack(tensor.view(torch.uint8)), dtype)
            return cp.from_dlpack(tensor)
    from quantem.gpu.device.metal_runtime import complete_command, metal_queue, tensor_buffer

    result = MetalArray(tensor.shape, dtype)
    try:
        torch.mps.synchronize()
        command = metal_queue().commandBuffer()
        encoder = command.blitCommandEncoder()
        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            tensor_buffer(tensor), tensor.storage_offset() * tensor.element_size(), result._mtl, 0, result.nbytes
        )
        encoder.endEncoding()
        complete_command(command, "Native read output")
        return result
    except BaseException:
        result.release()
        raise


def resident_device(payload):
    """Normalize the device a resident reports, without decoding it.

    Residents report their GPU as a ``torch.device``, a CUDA ordinal, a CuPy
    device or a string such as ``"mps"``, and callers may wrap owners of their
    own, so every spelling is accepted here. ``None`` means the payload
    reports no device.
    """
    # Read by attribute: CuPy is never imported here, and callers' own owners reach this.
    device = getattr(payload, "device", None)
    if not isinstance(device, torch.device):
        device_id = getattr(device, "id", None)
        if device_id is not None:
            device = torch.device("cuda", int(device_id))
        elif isinstance(device, (int, np.integer)):
            device = torch.device("cuda", int(device))
        else:
            device = torch.device(str(device)) if device is not None else None
    return device


def _decode_rows(
    payload, shape, device, row0, row1, column0, column1, *, detector_region=None,
    reinterpret_dtype=None,
):
    """Decode scan rows [row0, row1) and columns [column0, column1).

    Each resident type has one bounded decoder: streamed counts and float ANS
    decode a scan range (exact integer residents also crop
    ``detector_region``), precision and Metal streamed residents decode a
    range straight into a tensor, and plain device arrays are sliced. Returns a
    ``(positions, detector_row, detector_col)`` tensor in row-major scan order.
    """
    decoded_to_tensor = (
        (cuda_precision.PrecisionSource,) if device.type == "cuda"
        else (metal_precision.PrecisionSource, MPSStreamedCounts)
    )
    if isinstance(payload, (StreamedCounts, BitANSResident)):
        decode = payload.decode_scan_range_device
        if detector_region is not None:
            decode = partial(decode, detector_region=detector_region)
    elif isinstance(payload, decoded_to_tensor):
        decode = payload.decode_scan_range_torch
        if detector_region is not None:
            decode = partial(decode, detector_region=detector_region)
    else:
        # A plain device array, such as a dense CuPy acquisition.
        try:
            return torch_value(payload[row0:row1, column0:column1]).reshape(
                -1, *shape[2:]
            )
        except (TypeError, IndexError, KeyError, RuntimeError) as error:
            raise TypeError(
                "This loaded representation does not support bounded scan reads."
            ) from error
    # Range decoders take flat row-major scan indices, so a block narrower than
    # the scan decodes one range per scan row.
    if column0 == 0 and column1 == shape[1]:
        return torch_value(decode(row0 * shape[1], row1 * shape[1]), reinterpret_dtype=reinterpret_dtype)
    parts = [
        torch_value(decode(row * shape[1] + column0, row * shape[1] + column1), reinterpret_dtype=reinterpret_dtype)
        for row in range(row0, row1)
    ]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def torch_value(value, *, reinterpret_dtype=None):
    """Wrap one decoded block as a Torch tensor on its GPU without a host copy.

    CUDA decoders return CuPy arrays, shared through DLPack. Metal decoders
    return shared buffers, copied into a Torch MPS tensor and then released.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type not in {"cuda", "mps"}:
            raise TypeError("read() requires accelerator-resident data.")
        return value
    from quantem.gpu.resident.cuda.float8 import Float8Array

    if isinstance(value, Float8Array):
        try:
            if reinterpret_dtype is not None:
                return torch.from_dlpack(value.values)
            tensor = value.to_torch()
            return tensor
        finally:
            value.release()
    if isinstance(value, MetalArray):
        try:
            if reinterpret_dtype is not None:
                if np.dtype(reinterpret_dtype).itemsize != value.dtype.itemsize:
                    raise ValueError("A raw read view must retain the element size.")
                value.dtype = np.dtype(reinterpret_dtype)
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
    """Validate one ``(row_start, row_stop, col_start, col_stop)`` region.

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
            f"{name} must be (row_start, row_stop, col_start, col_stop)."
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
