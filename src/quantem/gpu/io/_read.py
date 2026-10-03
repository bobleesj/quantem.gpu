"""Backend-neutral bounded reads from loaded accelerator residents."""

from __future__ import annotations

import math

import numpy as np

_BLOCK_BYTES = 256 * 1024**2


def _region(value, shape, name):
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


def _torch_value(value):
    import torch

    if torch.is_tensor(value):
        if value.device.type not in {"cuda", "mps"}:
            raise TypeError("read() requires accelerator-resident data.")
        return value
    to_torch = getattr(value, "to_torch", None)
    if callable(to_torch):
        try:
            tensor = to_torch()
            if tensor.device.type == "mps":
                torch.mps.synchronize()
            return tensor
        finally:
            release = getattr(value, "release", None)
            if callable(release):
                release()
    try:
        tensor = torch.from_dlpack(value)
    except Exception as error:
        raise TypeError(
            "This loaded representation does not provide accelerator-native bounded reads."
        ) from error
    if tensor.device.type not in {"cuda", "mps"}:
        raise TypeError("read() never materializes detector values on CPU.")
    return tensor


def _check_allocation(shape, dtype, device):
    import torch

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
    elif device.type == "mps" and hasattr(torch.mps, "recommended_max_memory"):
        available = (
            torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory()
        )
        if requested + reserve > available:
            raise MemoryError(
                f"The requested region needs about {requested / 2**30:.2f} GiB "
                "beyond the available MPS working set. Request a smaller scan_region."
            )


def resident_device(payload):
    """Normalize backend resident device identifiers without decoding."""
    import torch

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


def read(data, *, scan_region=None, detector_region=None):
    """Return a requested logical region as a Torch tensor on the source GPU."""
    import torch

    shape = tuple(data.shape)
    if len(shape) != 4:
        raise ValueError(f"read() requires 4D-STEM shape; got {shape}.")
    row0, row1, column0, column1 = _region(
        scan_region, shape[:2], "scan_region"
    )
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

    # Bound decoded scratch by scan rows. Integer CUDA streams can select
    # detector pixels directly; other representations decode whole frames.
    from quantem.gpu._compact.streamed import StreamedCounts
    from ._float_ans import FloatANSResident, MAX_DECODE_BYTES

    # Calibrated subclasses have different decode semantics; keep their path.
    selected_region = None
    if type(payload) is StreamedCounts:
        selected_region = (detector_row0, detector_row1,
                           detector_column0, detector_column1)
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
                payload, shape, block_row0, block_row1,
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


def _decode_rows(
    payload, shape, row0, row1, column0, column1, *, detector_region=None
):
    """Decode scan rows [row0, row1) and columns [column0, column1).

    Returns a (positions, detector_rows, detector_columns) tensor in row-major
    scan order, optionally cropped by the CUDA streamed-count decoder.
    """
    import torch

    if detector_region is not None:
        if column0 == 0 and column1 == shape[1]:
            return _torch_value(payload.decode_scan_range_device(
                row0 * shape[1], row1 * shape[1], detector_region=detector_region
            ))
        parts = [_torch_value(payload.decode_scan_range_device(
            row * shape[1] + column0, row * shape[1] + column1,
            detector_region=detector_region,
        )) for row in range(row0, row1)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    gather = getattr(payload, "gather_diffraction_device", None)
    if callable(gather):
        rows, columns = np.meshgrid(
            np.arange(row0, row1, dtype=np.int64),
            np.arange(column0, column1, dtype=np.int64),
            indexing="ij",
        )
        positions = np.stack((rows.ravel(), columns.ravel()), axis=1)
        return _torch_value(gather(positions))
    decode = getattr(payload, "_decode_scan_range_torch", None)
    if not callable(decode):
        decode = getattr(payload, "decode_scan_range_device", None)
    frame_native = getattr(payload, "frame_native", None)
    if not callable(decode) and callable(frame_native):
        indices = [
            row * shape[1] + column
            for row in range(row0, row1)
            for column in range(column0, column1)
        ]
        parts = [_torch_value(frame_native(index)) for index in indices]
        return parts[0][None] if len(parts) == 1 else torch.stack(parts)
    if not callable(decode):
        try:
            return _torch_value(payload[
                row0:row1, column0:column1
            ]).reshape(-1, *shape[2:])
        except Exception as error:
            raise TypeError(
                "This loaded representation does not support bounded scan reads."
            ) from error
    if column0 == 0 and column1 == shape[1]:
        return _torch_value(decode(row0 * shape[1], row1 * shape[1]))
    parts = [
        _torch_value(decode(row * shape[1] + column0, row * shape[1] + column1))
        for row in range(row0, row1)
    ]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
