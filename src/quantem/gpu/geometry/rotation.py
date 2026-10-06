"""Scan-coordinate geometry operations for 4D-STEM data."""

import math
import sys
from copy import deepcopy
from typing import Literal

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.geometry import cuda
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.resident.cuda.counts import StreamedCounts

Interpolation = Literal["auto", "nearest", "bilinear"]
OutputShape = Literal["full", "same"]


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

    The encoded acquisition that :func:`quantem.gpu.io.load` returns on CUDA is
    rotated into a new encoded acquisition on the same GPU, band by band, so it
    is never decoded whole. Its counts are reordered, never interpolated: use a
    multiple of 90 degrees, or ``interpolation="nearest"`` at other angles.

    Parameters
    ----------
    data
        A 4D NumPy, CuPy, or Torch array ordered as ``(scan_row, scan_col,
        detector_row, detector_col)``, or a ``Dataset4dstemGPU`` returned by
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
        Value outside the measured scan field. For an encoded acquisition,
        a count that its dtype holds.
    return_valid_mask
        Return a scan-plane mask identifying output positions whose mapped
        centers lie inside the source field.

    Returns
    -------
    array or Dataset4dstemGPU
        Rotated data in the same resident array family. A load result retains
        its metadata and records scan-rotation provenance; an encoded one
        stays encoded, and the source acquisition is left unchanged.
    tuple, optional
        ``(rotated, valid_mask)`` when ``return_valid_mask=True``. The mask
        sits beside array data and is NumPy for an encoded acquisition.

    Raises
    ------
    TypeError
        If ``data`` is not a supported resident array or load result.
    ValueError
        If the input is not scan-axis-leading 4D-STEM data or an option is
        incompatible with dtype-preserving interpolation.
    NotImplementedError
        If an arbitrary-angle rotation is requested on an unsupported device,
        an encoded acquisition would need bilinear interpolation, or the
        encoded acquisition is on an Apple GPU (MPS) or is not uint8/uint16
        counts.

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

    loaded = isinstance(data, Dataset4dstemGPU)
    array = data.data if loaded else data
    metadata = deepcopy(data.metadata) if loaded else None
    if loaded and _quarter_turn(angle) is None:
        spacing = metadata.get("scan_sampling_A", metadata.get("scan_sampling"))
        if spacing is not None and not np.isscalar(spacing) and not math.isclose(
            float(spacing[0]), float(spacing[1]), rel_tol=1e-12
        ):
            raise NotImplementedError(
                "Arbitrary-angle calibrated rotation requires equal scan row/column "
                "sampling. Use exact 90-degree turns or resample first."
            )
    if _array_kind(array) is not None:
        (rotated, valid), resolved_interpolation = _rotate_array(
            array,
            angle,
            output_shape,
            interpolation,
            float(fill_value),
        )
    elif loaded and type(array) is StreamedCounts:
        (rotated, valid), resolved_interpolation = _rotate_encoded(
            data,
            angle,
            output_shape,
            interpolation,
            float(fill_value),
        )
    elif loaded and data.device is not None and data.device.type == "mps":
        raise NotImplementedError(
            "Scan rotation of an encoded acquisition has no Apple GPU (MPS) "
            "implementation; it runs on CUDA only. Load the acquisition with "
            "io.load(path, backend='cuda') on a CUDA machine, or rotate the scan "
            "images reconstructed from it."
        )
    elif loaded:
        raise NotImplementedError(
            "rotate_scan rotates encoded uint8/uint16 count acquisitions on CUDA; "
            f"this acquisition is a {type(array).__name__} on {data.device}."
        )
    else:
        raise TypeError(
            "rotate_scan expects a 4D NumPy, CuPy, or Torch array, or the "
            f"Dataset4dstemGPU returned by quantem.gpu.io.load; got {type(data).__name__}."
        )
    if loaded:
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
        metadata["working_shape"] = tuple(int(value) for value in rotated.shape)
        metadata["working_dtype"] = str(rotated.dtype).removeprefix("torch.")
        metadata["n_frames"] = int(rotated.shape[0] * rotated.shape[1])
        # Sizes of the source's storage; the rotated data reports its own.
        for key in ("working_logical_tensor_bytes", "physical_resident_bytes", "index_bytes"):
            metadata.pop(key, None)
        quarter_turns = _quarter_turn(angle)
        if quarter_turns is not None and quarter_turns % 2:
            for key in ("scan_sampling", "scan_sampling_A", "sampling", "units"):
                pair = metadata.get(key)
                if pair is not None and not np.isscalar(pair) and len(pair) >= 2:
                    metadata[key] = (pair[1], pair[0], *pair[2:])
        scientific = metadata.get("scientific_metadata")
        if scientific is not None:
            axes = scientific["axes"]
            if quarter_turns is not None and quarter_turns % 2:
                row_sampling = axes[0].pop("sampling", None)
                column_sampling = axes[1].pop("sampling", None)
                if column_sampling is not None:
                    axes[0]["sampling"] = column_sampling
                if row_sampling is not None:
                    axes[1]["sampling"] = row_sampling
                prefix = "scan_controller/regular_scan/pixel_size_"
                for section in ("electron_microscope", "calibration_overrides"):
                    quantities = scientific.get(section, {})
                    row = quantities.pop(prefix + "row", None)
                    column = quantities.pop(prefix + "column", None)
                    if column is not None:
                        quantities[prefix + "row"] = column
                    if row is not None:
                        quantities[prefix + "column"] = row
            for axis, size in zip(axes, rotated.shape):
                axis["size"] = int(size)
        metadata["scan_rotation_history"] = history
    result = Dataset4dstemGPU(rotated, metadata) if loaded else rotated
    if not return_valid_mask:
        return result
    return result, _mask_on_backend(valid, rotated)


def _rotate_encoded(
    dataset: Dataset4dstemGPU,
    angle_degrees: float,
    output_shape: OutputShape,
    interpolation: Interpolation,
    fill_value: float,
):
    """Rotate an encoded CUDA acquisition into a new encoded acquisition.

    io.load keeps the acquisition entropy-coded on the GPU, and its dense form
    often exceeds GPU memory, so it cannot go through the array path. A
    rotation that keeps counts exact only reorders scan positions (a quarter
    turn, or nearest neighbor at other angles), so the dense reference applied
    to the grid of scan indices names the source position of every output
    position, with the same conventions; the CUDA gather then copies those
    frames band by band. Returns ``((rotated, valid), interpolation)`` like
    :func:`_rotate_array`.
    """
    source = dataset.data
    if _quarter_turn(angle_degrees) is None and interpolation != "nearest":
        raise NotImplementedError(
            "Interpolating an encoded acquisition would turn its counts into float32 "
            "values. Rotate by a multiple of 90 degrees, pass interpolation='nearest', "
            "or rotate the scan images reconstructed from it."
        )
    if not fill_value.is_integer() or not 0 <= fill_value <= np.iinfo(source.dtype).max:
        raise ValueError(
            f"fill_value must be a {source.dtype} count for an encoded acquisition; "
            f"got {fill_value!r}."
        )
    scan_rows, scan_cols = dataset.shape[:2]
    scan_index = np.arange(scan_rows * scan_cols, dtype=np.int64).reshape(
        scan_rows, scan_cols, 1, 1
    )
    (index_map, valid), resolved_interpolation = _rotate_array(
        scan_index, angle_degrees, output_shape, interpolation, -1.0
    )
    rotated = cuda.gather_scan_positions(dataset, index_map, int(fill_value))
    return (rotated, valid), resolved_interpolation


# ---------------------------------------------------------------------------
# Dense arrays
# ---------------------------------------------------------------------------


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
        rotated = cuda.rotate_array(
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
            rotated = torch.from_dlpack(
                cuda.rotate_array(
                    cp.from_dlpack(data.detach()),
                    target_shape,
                    angle_degrees,
                    resolved_interpolation,
                    fill_value,
                )
            )
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
        rotated = cp.ascontiguousarray(cp.rot90(data, quarter_turns, axes=(0, 1)))
    else:
        rotated = np.ascontiguousarray(np.rot90(data, quarter_turns, axes=(0, 1)))
    if output_shape == "same" and rotated.shape[:2] != data.shape[:2]:
        return _center_to_shape(rotated, tuple(data.shape[:2]), fill_value)
    return rotated, np.ones(rotated.shape[:2], dtype=bool)


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


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _mask_on_backend(mask: np.ndarray, data):
    """Place the small scan-validity mask beside its transformed data.

    An encoded acquisition is not an array; its mask stays in NumPy, the form
    detector queries take.
    """
    kind = _array_kind(data)
    if kind == "cupy":
        return cp.asarray(mask)
    if kind == "torch":
        import torch

        return torch.as_tensor(mask, device=data.device)
    return mask


def _quarter_turn(angle_degrees: float) -> int | None:
    """Return an exact counterclockwise quarter-turn count when available."""
    turns = round(angle_degrees / 90.0)
    if math.isclose(angle_degrees, turns * 90.0, abs_tol=1e-10):
        return turns % 4
    return None


def _array_kind(data) -> str | None:
    """Return the dense array family, or None for anything else.

    An array can be a CuPy or Torch array only once that library is imported,
    so looking the library up among the imported modules never imports it on
    a machine that does not use it.
    """
    if isinstance(data, np.ndarray):
        return "numpy"
    cupy = sys.modules.get("cupy")
    if cupy is not None and isinstance(data, cupy.ndarray):
        return "cupy"
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(data, torch.Tensor):
        return "torch"
    return None
