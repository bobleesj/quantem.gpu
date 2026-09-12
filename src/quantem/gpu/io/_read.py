"""Backend-neutral bounded reads from loaded accelerator residents."""

from __future__ import annotations

import math

import numpy as np


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
        tensor = to_torch()
        if tensor.device.type == "mps":
            torch.mps.synchronize()
        release = getattr(value, "release", None)
        if callable(release):
            release()
        return tensor
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

    gather = getattr(payload, "gather_diffraction_device", None)
    if callable(gather):
        rows, columns = np.meshgrid(
            np.arange(row0, row1, dtype=np.int64),
            np.arange(column0, column1, dtype=np.int64),
            indexing="ij",
        )
        positions = np.stack((rows.ravel(), columns.ravel()), axis=1)
        tensor = _torch_value(gather(positions))
    else:
        decode = getattr(payload, "decode_scan_range_device", None)
        frame_native = getattr(payload, "frame_native", None)
        if not callable(decode) and callable(frame_native):
            indices = [
                row * shape[1] + column
                for row in range(row0, row1)
                for column in range(column0, column1)
            ]
            parts = [_torch_value(frame_native(index)) for index in indices]
            tensor = parts[0][None] if len(parts) == 1 else torch.stack(parts)
        elif not callable(decode):
            try:
                tensor = _torch_value(payload[
                    row0:row1, column0:column1
                ]).reshape(-1, *shape[2:])
            except Exception as error:
                raise TypeError(
                    "This loaded representation does not support bounded scan reads."
                ) from error
        elif column0 == 0 and column1 == shape[1]:
            tensor = _torch_value(
                decode(row0 * shape[1], row1 * shape[1])
            )
        else:
            parts = [
                _torch_value(
                    decode(row * shape[1] + column0, row * shape[1] + column1)
                )
                for row in range(row0, row1)
            ]
            tensor = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    tensor = tensor[
        :,
        detector_row0:detector_row1,
        detector_column0:detector_column1,
    ]
    valid_pixels = getattr(payload, "valid_pixels", None)
    if valid_pixels is not None and not bool(np.asarray(valid_pixels).all()):
        selected_valid = np.asarray(valid_pixels)[
            detector_row0:detector_row1,
            detector_column0:detector_column1,
        ]
        for invalid_row, invalid_column in np.argwhere(~selected_valid):
            tensor[..., int(invalid_row), int(invalid_column)] = 0
    return tensor.reshape(output_shape).contiguous()
