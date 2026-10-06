"""Read prepared diffraction stacks and their bounded MATLAB companions."""

import math
from pathlib import Path

import h5py
import numpy as np


def prepared_stack_metadata(data: h5py.Dataset) -> dict:
    """Recognize a prepared /dp stack without guessing a square scan."""
    if not isinstance(data, h5py.Dataset) or data.name != "/dp" or data.ndim != 3:
        return {}
    path = Path(data.file.filename)
    companion = path.parent / "params_backup.mat"
    if not path.name.startswith("data_roi") or not companion.is_file():
        return {}
    if companion.stat().st_size > 1 << 20:
        raise ValueError(
            "params_backup.mat exceeds 1 MiB; supply a bounded calibration file."
        )
    from scipy.io import loadmat

    names = (
        "ADU",
        "Np_p",
        "alpha",
        "voltage",
        "dk",
        "dx",
        "df",
        "cs",
        "dose",
        "rbf",
        "sftx",
        "sfty",
        "sftx0",
        "sfty0",
        "crop_idx0",
    )
    values = loadmat(companion, variable_names=names)
    if not all(name in values for name in ("Np_p", "voltage", "alpha", "dk", "dx")):
        return {}
    params = {}
    for name in names:
        if name not in values:
            continue
        value = np.asarray(values[name])
        if (
            value.size > 100
            or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"Invalid prepared calibration {name}; retain the original and correct its companion."
            )
        params[name] = value.reshape(-1).tolist()
    detector_shape = params["Np_p"]
    # MATLAB HDF5 reverses the detector dimensions on disk.
    if len(detector_shape) != 2 or tuple(reversed(detector_shape)) != data.shape[1:]:
        raise ValueError(
            "Np_p disagrees with /dp detector dimensions; choose the matching params_backup.mat."
        )
    retained = {"prepared_stack/params_backup": params}
    result = {
        "source_metadata": retained,
        "source_metadata_coverage": "reader-retained",
    }
    for name, target in (("voltage", "voltage_kV"), ("alpha", "semiangle_mrad")):
        if len(params[name]) == 1 and params[name][0] > 0:
            result[target] = params[name][0]
    if len(params["dk"]) == 1 and params["dk"][0] > 0:
        result["detector_sampling"] = params["dk"] * 2
        result["detector_sampling_unit"] = "1/angstrom"
    positions_path = path.parent / "data_position.hdf5"
    if positions_path.is_file():
        with h5py.File(positions_path, "r") as source:
            positions = source.get("probe_positions_0")
            if (
                not isinstance(positions, h5py.Dataset)
                or positions.shape != (2, data.shape[0])
                or positions.dtype.kind not in "fiu"
                or positions.size * positions.dtype.itemsize > 8 << 20
            ):
                raise ValueError(
                    "Probe positions do not match /dp; restore the matching data_position.hdf5."
                )
            coordinates = positions[()]
        slow, fast = coordinates
        if not np.isfinite(coordinates).all():
            raise ValueError(
                "Probe coordinates must be finite; correct data_position.hdf5."
            )
        boundaries = np.flatnonzero(slow != slow[0])
        columns = int(boundaries[0]) if boundaries.size else len(slow)
        rows, remainder = divmod(len(slow), columns)
        if remainder:
            raise ValueError(
                "Probe positions are not a complete raster; provide explicit scan geometry."
            )
        slow_grid, fast_grid = slow.reshape(rows, columns), fast.reshape(rows, columns)
        if (
            not np.all(slow_grid == slow_grid[:, :1])
            or not np.all(fast_grid == fast_grid[:1])
            or len(np.unique(slow)) != rows
            or len(np.unique(fast)) != columns
        ):
            raise ValueError(
                "Probe positions are not a regular raster; do not reshape an irregular scan."
            )
        result["scan_shape"] = (rows, columns)
        retained["prepared_stack/probe_positions_0"] = coordinates.tolist()
        # Retain coordinates without assigning undocumented position units.
        retained["prepared_stack/position_units"] = "unspecified"
    elif "crop_idx0" in params:
        bounds = params["crop_idx0"]
        if len(bounds) == 4 and all(
            value >= 1 and value == int(value) for value in bounds
        ):
            shape = (int(bounds[3] - bounds[2] + 1), int(bounds[1] - bounds[0] + 1))
            if min(shape) > 0 and math.prod(shape) == data.shape[0]:
                result["scan_shape"] = shape
                retained["prepared_stack/scan_shape_source"] = (
                    "MATLAB inclusive crop_idx0, reversed storage axes"
                )
    return result
