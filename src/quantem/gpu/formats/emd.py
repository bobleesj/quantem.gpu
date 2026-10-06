"""Read EMD files: NCEM dimension calibration and Velox scope metadata.

Neither reader guesses unknown units or axis spacing; missing fields stay
missing so callers can merge what was found.
"""

import json
import math
from pathlib import Path

import h5py
import numpy as np


def _value(value):
    """Convert an HDF5 attribute to plain Python values so metadata stays JSON-serializable."""
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


def read_emd_metadata(emd_path) -> dict:
    """Extract scope-side fields from a Velox EMD file.

    Reads the first image's ``Data/Image/<hash>/Metadata`` JSON (Velox
    stores metadata as a uint8 byte vector per frame). Returns a dict
    with whichever of the following keys were found; missing keys are
    omitted so callers can merge via ``dict.update`` without clobbering:

    - ``stem_magnification``    : float, e.g. 5_100_000 for 5.1 Mx
    - ``field_of_view_nm``      : float, FullScanFieldOfView.x in nm
    - ``voltage_kV``            : float, AccelerationVoltage / 1000
    - ``semiangle_mrad``        : float, probe semiangle when exposed

    Returns ``{}`` on any parse failure so callers can always `.update()`
    the result into an existing config dict without guarding. The EMD
    format version varies across microscope builds, so missing-field
    handling is the common path, not the edge case.
    """
    path = Path(emd_path)
    if not path.is_file():
        return {}
    try:
        with h5py.File(path, "r") as handle:
            if "Data/Image" not in handle:
                return {}
            image_group = handle["Data/Image"]
            first_hash = next(iter(image_group.keys()), None)
            if first_hash is None:
                return {}
            meta_ds = image_group[first_hash].get("Metadata")
            if meta_ds is None:
                return {}
            # Velox stores metadata as a (nbytes, nframes) uint8 JSON buffer.
            # Frame 0 is sufficient; per-frame blobs are near-identical.
            raw = meta_ds[:, 0] if meta_ds.ndim == 2 else meta_ds[()]
            raw_bytes = bytes(np.asarray(raw).tolist()).rstrip(b"\x00")
            document = json.loads(raw_bytes)
    except (OSError, ValueError, KeyError):
        return {}

    result: dict = {}
    optics = document.get("Optics") or {}
    custom = document.get("CustomProperties") or {}

    # Velox wraps most scalars as {"type": "double", "value": "5100000"};
    # AccelerationVoltage is historically a bare string. Accept both.
    def as_float(value):
        if isinstance(value, dict):
            value = value.get("value")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    magnification = as_float(custom.get("StemMagnification"))
    if magnification is not None:
        result["stem_magnification"] = magnification

    fov = optics.get("FullScanFieldOfView")
    if isinstance(fov, dict):
        fov_x = as_float(fov.get("x"))
        if fov_x is not None:
            # Velox reports FOV in metres; screener works in nm.
            result["field_of_view_nm"] = fov_x * 1e9

    voltage = as_float(optics.get("AccelerationVoltage"))
    if voltage is not None:
        result["voltage_kV"] = voltage / 1000.0

    semiangle = as_float(optics.get("ConvergenceSemiAngle") or optics.get("SemiConvergenceAngle"))
    if semiangle is not None:
        # Velox stores the convergence angle in radians.
        result["semiangle_mrad"] = semiangle * 1000.0
    return result


def find_emd_sibling(master_path) -> Path | None:
    """Locate a Velox EMD next to an Arina master file.

    Arina writes ``<stem>_master.h5`` alongside data chunk files; when
    the operator also exports the scan to Velox, the EMD usually lands
    in the same folder. Strategy:

    1. Prefer a file named ``<stem>.emd`` (strict match).
    2. Fall back to any ``*.emd`` in the same directory - Dectris
       operators often batch-rename after the fact.

    Returns ``None`` when no EMD sibling is found.
    """
    master = Path(master_path)
    folder = master.parent
    stem = master.stem
    stem = stem.removesuffix("_master")
    candidates = list(folder.glob(f"{stem}.emd")) + list(folder.glob(f"{stem}*.emd"))
    if candidates:
        return candidates[0]
    others = list(folder.glob("*.emd"))
    return others[0] if len(others) == 1 else None
