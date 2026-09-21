"""Retain NCEM EMD calibration without guessing unknown units or axis spacing."""

import math

import h5py
import numpy as np


def _value(value):
    if isinstance(value, bytes):
        return value.decode("utf8")
    if isinstance(value, np.ndarray):
        return (
            [_value(item) for item in value.flat]
            if value.ndim
            else _value(value.item())
        )
    return value.item() if isinstance(value, np.generic) else value


def dataset_metadata(data: h5py.Dataset, metadata: dict) -> dict:
    """Merge bounded source metadata and regular EMD dimension calibration."""
    result = dict(metadata)
    retained = {
        key: _value(value)
        for key, value in metadata.items()
        if "/" in key and value is not None
    }
    dimensions = []
    for obj in (data.file, data.parent, data):
        for name, value in obj.attrs.items():
            if np.asarray(value).size <= 100:
                retained[obj.name + "@" + name] = _value(value)
    for index, size in enumerate(data.shape, 1):
        dimension = data.parent.get(f"dim{index}")
        if not isinstance(dimension, h5py.Dataset):
            dimensions.append(None)
            continue
        # EMD 1.0 permits either an explicit coordinate vector or two values
        # describing a regular axis. EMD 0.1 also writes column vectors.
        if dimension.size not in (2, size) or dimension.size > 4096:
            retained[dimension.name + "@retention"] = "coordinate vector not retained"
            dimensions.append(None)
            continue
        values = dimension[()].reshape(-1)
        retained[dimension.name] = _value(values)
        fields = {}
        for field in ("name", "units"):
            if field in dimension.attrs:
                value = dimension.attrs[field]
            else:
                sibling = data.parent.get(f"dim{index}_{field}")
                value = (
                    sibling[()]
                    if isinstance(sibling, h5py.Dataset) and sibling.size == 1
                    else ""
                )
            stored = _value(value)
            retained[dimension.name + "@" + field] = stored
            while isinstance(stored, list) and len(stored) == 1:
                stored = stored[0]
            fields[field] = stored if isinstance(stored, str) else ""
        spacing = None
        if values.dtype.kind in "fiu" and len(values) >= 2:
            difference = np.diff(values.astype(np.float64))
            if (
                np.all(np.isfinite(difference))
                and difference[0] > 0
                and np.allclose(difference, difference[0], rtol=1e-6, atol=0)
            ):
                spacing = float(difference[0])
        dimensions.append((spacing, fields["units"].strip("[] ")))
    result["source_metadata"] = retained
    result["source_metadata_coverage"] = "reader-retained"
    if len(data.shape) != 4:
        return result
    scan_units = {
        "m": 1e10,
        "mm": 1e7,
        "um": 1e4,
        "µm": 1e4,
        "nm": 10,
        "pm": 0.01,
        "A": 1,
        "Å": 1,
        "angstrom": 1,
    }
    detector_units = {
        "1/m": (1e-10, "1/angstrom"),
        "1/nm": (0.1, "1/angstrom"),
        "nm^-1": (0.1, "1/angstrom"),
        "nm⁻¹": (0.1, "1/angstrom"),
        "1/A": (1, "1/angstrom"),
        "1/Å": (1, "1/angstrom"),
        "1/angstrom": (1, "1/angstrom"),
        "Å^-1": (1, "1/angstrom"),
        "Å⁻¹": (1, "1/angstrom"),
        "rad": (1000, "mrad"),
        "mrad": (1, "mrad"),
    }
    scan, detector, units = [], [], []
    for index, dimension in enumerate(dimensions):
        if dimension is None or dimension[0] is None:
            continue
        spacing, unit = dimension
        if index < 2 and unit in scan_units:
            scan.append(spacing * scan_units[unit])
        elif index >= 2 and unit in detector_units:
            factor, normalized = detector_units[unit]
            detector.append(spacing * factor)
            units.append(normalized)
    if len(scan) == 2 and all(math.isfinite(value) for value in scan):
        result["scan_sampling_A"] = scan
    if len(detector) == 2 and units[0] == units[1]:
        result["detector_sampling"] = detector
        result["detector_sampling_unit"] = units[0]
    return result
