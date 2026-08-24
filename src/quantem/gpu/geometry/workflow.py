"""Scan-coordinate geometry operations for 4D-STEM data."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

Interpolation = Literal["auto", "nearest", "bilinear"]
OutputShape = Literal["full", "same"]

_CUDA_NEAREST = None
_CUDA_BILINEAR = None


def _array_kind(data) -> str:
    """Return the resident array family without importing optional backends."""
    module = type(data).__module__.split(".", 1)[0]
    if module in {"cupy", "torch"}:
        return module
    if isinstance(data, np.ndarray):
        return "numpy"
    raise TypeError(
        "rotate_scan expects a 4D NumPy, CuPy, or Torch array, or the "
        f"LoadResult returned by quantem.gpu.io.load; got {type(data).__name__}."
    )


def _unwrap(data):
    """Return numeric data and a function that restores a load result."""
    fields = getattr(data, "_fields", ())
    if "data" not in fields or "metadata" not in fields:
        return data, None, None
    return data.data, data, dict(data.metadata)


def _restore_load_result(template, array, metadata):
    """Restore a public load result with transformed data and metadata."""
    if template is None:
        return array
    return template._replace(data=array, metadata=metadata)


def _quarter_turn(angle_degrees: float) -> int | None:
    """Return an exact counterclockwise quarter-turn count when available."""
    turns = round(angle_degrees / 90.0)
    if math.isclose(angle_degrees, turns * 90.0, abs_tol=1e-10):
        return turns % 4
    return None


def _full_scan_shape(
    scan_shape: tuple[int, int],
    angle_degrees: float,
) -> tuple[int, int]:
    """Return the smallest pixel-centered canvas containing a rotation."""
    scan_rows, scan_columns = scan_shape
    angle_radians = math.radians(angle_degrees)
    cosine = abs(math.cos(angle_radians))
    sine = abs(math.sin(angle_radians))
    output_rows = max(1, math.ceil(scan_rows * cosine + scan_columns * sine))
    output_columns = max(
        1,
        math.ceil(scan_rows * sine + scan_columns * cosine),
    )
    return output_rows, output_columns


def _scan_coordinates(
    source_shape: tuple[int, int],
    output_shape: tuple[int, int],
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inverse-mapped source coordinates and their validity mask."""
    source_rows, source_columns = source_shape
    output_rows, output_columns = output_shape
    output_row, output_column = np.indices(output_shape, dtype=np.float64)
    output_row -= (output_rows - 1) / 2.0
    output_column -= (output_columns - 1) / 2.0
    angle_radians = math.radians(angle_degrees)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    source_column = (
        cosine * output_column - sine * output_row + (source_columns - 1) / 2.0
    )
    source_row = sine * output_column + cosine * output_row + (source_rows - 1) / 2.0
    valid = (
        (source_row >= 0.0)
        & (source_row <= source_rows - 1)
        & (source_column >= 0.0)
        & (source_column <= source_columns - 1)
    )
    return source_row, source_column, valid


def _center_to_shape(data, output_shape: tuple[int, int], fill_value: float):
    """Center-crop or pad an exact rotation to one scan shape."""
    source_rows, source_columns = (int(value) for value in data.shape[:2])
    output_rows, output_columns = output_shape
    kind = _array_kind(data)
    if kind == "torch":
        import torch

        result = torch.full(
            (output_rows, output_columns, *data.shape[2:]),
            fill_value,
            dtype=data.dtype,
            device=data.device,
        )
    elif kind == "cupy":
        import cupy as cp

        result = cp.full(
            (output_rows, output_columns, *data.shape[2:]),
            fill_value,
            dtype=data.dtype,
        )
    else:
        result = np.full(
            (output_rows, output_columns, *data.shape[2:]),
            fill_value,
            dtype=data.dtype,
        )
    copy_rows = min(source_rows, output_rows)
    copy_columns = min(source_columns, output_columns)
    source_row = (source_rows - copy_rows) // 2
    source_column = (source_columns - copy_columns) // 2
    output_row = (output_rows - copy_rows) // 2
    output_column = (output_columns - copy_columns) // 2
    result[
        output_row : output_row + copy_rows,
        output_column : output_column + copy_columns,
    ] = data[
        source_row : source_row + copy_rows,
        source_column : source_column + copy_columns,
    ]
    valid = np.zeros(output_shape, dtype=bool)
    valid[
        output_row : output_row + copy_rows,
        output_column : output_column + copy_columns,
    ] = True
    return result, valid


def _exact_rotation(
    data,
    quarter_turns: int,
    output_shape: OutputShape,
    fill_value: float,
):
    """Apply one lossless quarter-turn rotation on the resident backend."""
    kind = _array_kind(data)
    if quarter_turns == 0:
        rotated = data
    elif kind == "torch":
        import torch

        rotated = torch.rot90(data, quarter_turns, dims=(0, 1)).contiguous()
    elif kind == "cupy":
        import cupy as cp

        rotated = cp.ascontiguousarray(cp.rot90(data, quarter_turns, axes=(0, 1)))
    else:
        rotated = np.ascontiguousarray(np.rot90(data, quarter_turns, axes=(0, 1)))
    if output_shape == "same" and rotated.shape[:2] != data.shape[:2]:
        return _center_to_shape(rotated, tuple(data.shape[:2]), fill_value)
    return rotated, np.ones(rotated.shape[:2], dtype=bool)


def _numpy_rotation(
    data: np.ndarray,
    source_row: np.ndarray,
    source_column: np.ndarray,
    interpolation: Literal["nearest", "bilinear"],
    fill_value: float,
) -> np.ndarray:
    """Apply the reference inverse-mapped rotation to a NumPy array."""
    output_shape = (*source_row.shape, *data.shape[2:])
    if interpolation == "nearest":
        if np.issubdtype(data.dtype, np.integer) and not float(fill_value).is_integer():
            raise ValueError(
                "fill_value must be an integer when nearest interpolation "
                f"preserves integer data; got {fill_value!r}."
            )
        output = np.full(output_shape, fill_value, dtype=data.dtype)
        nearest_row = np.rint(source_row).astype(np.intp)
        nearest_column = np.rint(source_column).astype(np.intp)
        inside = (
            (nearest_row >= 0)
            & (nearest_row < data.shape[0])
            & (nearest_column >= 0)
            & (nearest_column < data.shape[1])
        )
        output[inside] = data[nearest_row[inside], nearest_column[inside]]
        return np.ascontiguousarray(output)

    output = np.full(output_shape, fill_value, dtype=np.float32)
    row0 = np.floor(source_row).astype(np.intp)
    column0 = np.floor(source_column).astype(np.intp)
    row_fraction = source_row - row0
    column_fraction = source_column - column0
    for row_offset, column_offset, weight in (
        (0, 0, (1.0 - row_fraction) * (1.0 - column_fraction)),
        (0, 1, (1.0 - row_fraction) * column_fraction),
        (1, 0, row_fraction * (1.0 - column_fraction)),
        (1, 1, row_fraction * column_fraction),
    ):
        sample_row = row0 + row_offset
        sample_column = column0 + column_offset
        inside = (
            (sample_row >= 0)
            & (sample_row < data.shape[0])
            & (sample_column >= 0)
            & (sample_column < data.shape[1])
        )
        delta = data[sample_row[inside], sample_column[inside]].astype(
            np.float32,
            copy=False,
        ) - np.float32(fill_value)
        output[inside] += delta * weight[inside][(...,) + (None,) * (data.ndim - 2)]
    return np.ascontiguousarray(output)


def _cuda_kernels():
    """Create dtype-specialized CUDA kernels on their first use."""
    import cupy as cp

    global _CUDA_NEAREST, _CUDA_BILINEAR
    if _CUDA_NEAREST is None:
        _CUDA_NEAREST = cp.ElementwiseKernel(
            "raw T source, int64 source_rows, int64 source_columns, "
            "int64 output_columns, int64 detector_size, float64 cosine, "
            "float64 sine, float64 source_center_row, "
            "float64 source_center_column, float64 output_center_row, "
            "float64 output_center_column, T fill_value",
            "T rotated",
            """
            const long long detector_index = i % detector_size;
            const long long output_scan_index = i / detector_size;
            const long long output_row = output_scan_index / output_columns;
            const long long output_column = output_scan_index % output_columns;
            const double row = (double)output_row - output_center_row;
            const double column = (double)output_column - output_center_column;
            const double source_column = cosine * column - sine * row
                + source_center_column;
            const double source_row = sine * column + cosine * row
                + source_center_row;
            const long long nearest_row = llrint(source_row);
            const long long nearest_column = llrint(source_column);
            if (nearest_row >= 0 && nearest_row < source_rows
                    && nearest_column >= 0 && nearest_column < source_columns) {
                const long long source_index =
                    (nearest_row * source_columns + nearest_column) * detector_size
                    + detector_index;
                rotated = source[source_index];
            } else {
                rotated = fill_value;
            }
            """,
            "quantem_rotate_scan_nearest",
        )
    if _CUDA_BILINEAR is None:
        _CUDA_BILINEAR = cp.ElementwiseKernel(
            "raw T source, int64 source_rows, int64 source_columns, "
            "int64 output_columns, int64 detector_size, float64 cosine, "
            "float64 sine, float64 source_center_row, "
            "float64 source_center_column, float64 output_center_row, "
            "float64 output_center_column, float32 fill_value",
            "float32 rotated",
            """
            const long long detector_index = i % detector_size;
            const long long output_scan_index = i / detector_size;
            const long long output_row = output_scan_index / output_columns;
            const long long output_column = output_scan_index % output_columns;
            const double row = (double)output_row - output_center_row;
            const double column = (double)output_column - output_center_column;
            const double source_column = cosine * column - sine * row
                + source_center_column;
            const double source_row = sine * column + cosine * row
                + source_center_row;
            const long long row0 = (long long)floor(source_row);
            const long long column0 = (long long)floor(source_column);
            const float row_fraction = (float)(source_row - (double)row0);
            const float column_fraction = (float)(source_column - (double)column0);
            float value = fill_value;
            const long long sample_rows[4] = {row0, row0, row0 + 1, row0 + 1};
            const long long sample_columns[4] = {
                column0, column0 + 1, column0, column0 + 1
            };
            const float weights[4] = {
                (1.0f - row_fraction) * (1.0f - column_fraction),
                (1.0f - row_fraction) * column_fraction,
                row_fraction * (1.0f - column_fraction),
                row_fraction * column_fraction
            };
            for (int sample = 0; sample < 4; ++sample) {
                const long long sample_row = sample_rows[sample];
                const long long sample_column = sample_columns[sample];
                if (sample_row >= 0 && sample_row < source_rows
                        && sample_column >= 0 && sample_column < source_columns) {
                    const long long source_index =
                        (sample_row * source_columns + sample_column) * detector_size
                        + detector_index;
                    value += ((float)source[source_index] - fill_value)
                        * weights[sample];
                }
            }
            rotated = value;
            """,
            "quantem_rotate_scan_bilinear",
        )
    return _CUDA_NEAREST, _CUDA_BILINEAR


def _cupy_rotation(
    data,
    output_shape: tuple[int, int],
    angle_degrees: float,
    interpolation: Literal["nearest", "bilinear"],
    fill_value: float,
):
    """Apply one contiguous inverse-mapped CUDA rotation."""
    import cupy as cp

    source = cp.ascontiguousarray(data)
    source_rows, source_columns = (int(value) for value in source.shape[:2])
    output_rows, output_columns = output_shape
    detector_size = int(np.prod(source.shape[2:], dtype=np.int64))
    angle_radians = math.radians(angle_degrees)
    nearest_kernel, bilinear_kernel = _cuda_kernels()
    output_dtype = source.dtype if interpolation == "nearest" else cp.float32
    output = cp.empty((*output_shape, *source.shape[2:]), dtype=output_dtype)
    common = (
        source,
        np.int64(source_rows),
        np.int64(source_columns),
        np.int64(output_columns),
        np.int64(detector_size),
        np.float64(math.cos(angle_radians)),
        np.float64(math.sin(angle_radians)),
        np.float64((source_rows - 1) / 2.0),
        np.float64((source_columns - 1) / 2.0),
        np.float64((output_rows - 1) / 2.0),
        np.float64((output_columns - 1) / 2.0),
    )
    if interpolation == "nearest":
        if source.dtype.kind in "iu" and not float(fill_value).is_integer():
            raise ValueError(
                "fill_value must be an integer when nearest interpolation "
                f"preserves integer data; got {fill_value!r}."
            )
        nearest_kernel(*common, source.dtype.type(fill_value), output)
    else:
        bilinear_kernel(*common, np.float32(fill_value), output)
    return output


def _rotate_array(
    data,
    angle_degrees: float,
    output_shape: OutputShape,
    interpolation: Interpolation,
    fill_value: float,
):
    """Rotate one supported resident array and return its validity mask."""
    if data.ndim != 4:
        raise ValueError(
            "rotate_scan expects shape "
            "(scan_row, scan_col, detector_row, detector_col); "
            f"got {data.ndim}D shape {tuple(data.shape)}."
        )
    quarter_turns = _quarter_turn(angle_degrees)
    if quarter_turns is not None:
        return _exact_rotation(data, quarter_turns, output_shape, fill_value), "exact"

    resolved_interpolation = "bilinear" if interpolation == "auto" else interpolation
    source_shape = tuple(int(value) for value in data.shape[:2])
    target_shape = (
        source_shape
        if output_shape == "same"
        else _full_scan_shape(source_shape, angle_degrees)
    )
    source_row, source_column, valid = _scan_coordinates(
        source_shape,
        target_shape,
        angle_degrees,
    )
    kind = _array_kind(data)
    if kind == "cupy":
        rotated = _cupy_rotation(
            data,
            target_shape,
            angle_degrees,
            resolved_interpolation,
            fill_value,
        )
    elif kind == "torch":
        import torch

        if data.requires_grad:
            raise ValueError(
                "rotate_scan is a scientific data transform and does not retain "
                "Torch autograd history. Pass data.detach() before rotating."
            )
        if data.is_cuda:
            import cupy as cp

            cupy_data = cp.from_dlpack(data.detach())
            cupy_rotated = _cupy_rotation(
                cupy_data,
                target_shape,
                angle_degrees,
                resolved_interpolation,
                fill_value,
            )
            rotated = torch.from_dlpack(cupy_rotated)
        elif data.device.type == "cpu":
            rotated = torch.from_numpy(
                _numpy_rotation(
                    data.detach().numpy(),
                    source_row,
                    source_column,
                    resolved_interpolation,
                    fill_value,
                )
            )
        else:
            raise NotImplementedError(
                "Arbitrary-angle rotate_scan currently supports CUDA and CPU "
                f"arrays; got Torch device {data.device}. Use a 90-degree "
                "rotation on this device or run the arbitrary rotation on CUDA."
            )
    else:
        rotated = _numpy_rotation(
            data,
            source_row,
            source_column,
            resolved_interpolation,
            fill_value,
        )
    return (rotated, valid), resolved_interpolation


def _mask_on_backend(mask: np.ndarray, data):
    """Place the small scan-validity mask beside its transformed data."""
    kind = _array_kind(data)
    if kind == "cupy":
        import cupy as cp

        return cp.asarray(mask)
    if kind == "torch":
        import torch

        return torch.as_tensor(mask, device=data.device)
    return mask


def rotate_scan(
    data,
    angle_degrees: float,
    *,
    output_shape: OutputShape = "full",
    interpolation: Interpolation = "auto",
    fill_value: float = 0.0,
    return_valid_mask: bool = False,
):
    """Rotate the scan plane of a 4D-STEM acquisition.

    Diffraction patterns are never rotated. Positive angles rotate the displayed
    scan image counterclockwise. Exact multiples of 90 degrees use a lossless,
    dtype-preserving path. Other angles default to bilinear interpolation and
    therefore return float32 data.

    Parameters
    ----------
    data
        A 4D NumPy, CuPy, or Torch array ordered as ``(scan_row, scan_col,
        detector_row, detector_col)``, or a ``LoadResult`` returned by
        :func:`quantem.gpu.io.load`.
    angle_degrees
        Counterclockwise rotation in the displayed scan plane.
    output_shape
        ``"full"`` preserves the rotated field. ``"same"`` keeps the input
        scan shape and may crop or pad the field.
    interpolation
        ``"auto"`` uses exact quarter turns and bilinear interpolation
        otherwise. ``"nearest"`` preserves integer samples at arbitrary
        angles. Exact quarter turns remain lossless for every setting.
    fill_value
        Value outside the measured scan field.
    return_valid_mask
        Return a scan-plane mask identifying output positions whose mapped
        centers lie inside the source field.

    Returns
    -------
    array or LoadResult
        Rotated data in the same resident array family. A load result retains
        its metadata and records scan-rotation provenance.
    tuple, optional
        ``(rotated, valid_mask)`` when ``return_valid_mask=True``.

    Raises
    ------
    TypeError
        If ``data`` is not a supported resident array or load result.
    ValueError
        If the input is not scan-axis-leading 4D-STEM data or an option is
        incompatible with dtype-preserving interpolation.
    NotImplementedError
        If an arbitrary-angle rotation is requested on an unsupported device.

    Examples
    --------
    Rotate a 90-degree acquisition into the 0-degree scan frame without
    changing any diffraction-pattern counts.

    >>> from quantem.gpu import geometry, io
    >>> raw_90 = io.load("scan_90_master.h5")
    >>> oriented_90 = geometry.rotate_scan(raw_90, angle_degrees=-90)
    """
    if isinstance(angle_degrees, (bool, np.bool_)):
        raise TypeError("angle_degrees must be a finite number, not bool.")
    try:
        angle = float(angle_degrees)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"angle_degrees must be a finite number; got {angle_degrees!r}."
        ) from exc
    if not math.isfinite(angle):
        raise ValueError(
            f"angle_degrees must be a finite number; got {angle_degrees!r}."
        )
    if output_shape not in {"full", "same"}:
        raise ValueError(
            f"output_shape must be 'full' or 'same'; got {output_shape!r}."
        )
    if interpolation not in {"auto", "nearest", "bilinear"}:
        raise ValueError(
            "interpolation must be 'auto', 'nearest', or 'bilinear'; "
            f"got {interpolation!r}."
        )

    array, template, metadata = _unwrap(data)
    (rotated, valid), resolved_interpolation = _rotate_array(
        array,
        angle,
        output_shape,
        interpolation,
        float(fill_value),
    )
    if metadata is not None:
        history = list(metadata.get("scan_rotation_history", ()))
        history.append(
            {
                "angle_degrees": angle,
                "interpolation": resolved_interpolation,
                "output_shape": output_shape,
                "source_scan_shape": tuple(int(value) for value in array.shape[:2]),
                "result_scan_shape": tuple(int(value) for value in rotated.shape[:2]),
            }
        )
        metadata["scan_shape"] = tuple(int(value) for value in rotated.shape[:2])
        scan_sampling = metadata.get("scan_sampling")
        quarter_turns = _quarter_turn(angle)
        if (
            quarter_turns is not None
            and quarter_turns % 2
            and scan_sampling is not None
            and len(scan_sampling) == 2
        ):
            metadata["scan_sampling"] = (
                scan_sampling[1],
                scan_sampling[0],
            )
        metadata["scan_rotation_history"] = history
    result = _restore_load_result(template, rotated, metadata)
    if not return_valid_mask:
        return result
    return result, _mask_on_backend(valid, rotated)
