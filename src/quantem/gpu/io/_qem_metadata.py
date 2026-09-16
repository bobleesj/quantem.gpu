"""Scientific metadata shared by QEM readers, independent of payload codecs."""

import copy
import math

import numpy as np

MAGIC = b"QEMDATA1"
CONTAINER = "quantem.qem"
CONTAINER_VERSION = 1
LEGACY_SCHEMA = "quantem.scientific-metadata/1"
SCHEMA = "quantem.scientific-metadata/2"
AXIS_NAMES = ("scan_row", "scan_column", "detector_row", "detector_column")

# Conversion factors are source unit -> public scientific-metadata unit.
_LENGTH = ("angstrom", {"m": 1e10, "nm": 10, "angstrom": 1, "Å": 1})
_DETECTOR = {
    "rad": ("mrad", 1000), "mrad": ("mrad", 1),
    "1/nm": ("1/angstrom", 0.1), "1/Å": ("1/angstrom", 1),
    "1/angstrom": ("1/angstrom", 1),
}
_QUANTITIES = {
    "electron_source/accelerating_voltage": ("kV", {"V": 0.001, "kV": 1}),
    "electron_source/beam_energy": ("keV", {"eV": 0.001, "keV": 1}),
    "illumination_system/semi_convergence_angle": ("mrad", {"rad": 1000, "mrad": 1}),
    "scan_controller/regular_scan/dwell_time": ("us", {"s": 1e6, "ms": 1000, "us": 1}),
    "imaging_system/camera_length": ("mm", {"m": 1000, "cm": 10, "mm": 1}),
    "scan_controller/regular_scan/pixel_size_y": _LENGTH,
    "scan_controller/regular_scan/pixel_size_x": _LENGTH,
}


def _microscopy_quantity(quantity: dict, path: str) -> dict:
    """Convert a known quantity without altering its evidence or source record."""
    if not isinstance(quantity, dict) or not isinstance(quantity.get("unit"), str):
        raise ValueError(f"Invalid QEM quantity at {path}; supply a value and unit.")
    result = dict(quantity)
    unit = quantity.get("unit")
    if path == "detector" or path.startswith("imaging_system/reciprocal_pixel_size_"):
        conversion = _DETECTOR.get(unit)
    else:
        if path != "scan" and path not in _QUANTITIES:
            raise ValueError(f"Unknown QEM calibration quantity {path!r}; update the reader.")
        target, factors = _LENGTH if path == "scan" else _QUANTITIES[path]
        factor = factors.get(unit)
        conversion = None if factor is None else (target, factor)
    value = quantity.get("value")
    if (conversion is None or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value <= 0 or not math.isfinite(value * conversion[1])):
        raise ValueError(f"Invalid QEM quantity {path}: {quantity!r}; check value and unit.")
    result.update(value=value * conversion[1], unit=conversion[0])
    return result


def microscopy_metadata(scientific: dict) -> dict:
    """Convert saved metadata to schema 2 without modifying source fields.

    Parameters
    ----------
    scientific : dict
        Schema-1 or schema-2 scientific metadata from a validated header.

    Returns
    -------
    dict
        Independent metadata with microscopy units and unchanged source fields.

    Raises
    ------
    ValueError
        A known quantity has an unsupported unit or invalid physical value.

    Examples
    --------
    >>> microscopy_metadata({"schema": LEGACY_SCHEMA})["schema"] == SCHEMA
    True
    """
    result = copy.deepcopy(scientific)
    if result.get("schema") not in (LEGACY_SCHEMA, SCHEMA):
        raise ValueError("Unsupported QEM metadata schema; update the reader.")
    for axis in result.get("axes", []):
        if "sampling" in axis:
            kind = "scan" if axis["name"].startswith("scan_") else "detector"
            axis["sampling"] = _microscopy_quantity(axis["sampling"], kind)
    for section in ("electron_microscope", "calibration_overrides"):
        if not isinstance(result.get(section, {}), dict):
            raise ValueError(f"QEM {section} must contain named quantities.")
        for path, quantity in result.get(section, {}).items():
            if path in _QUANTITIES or path.startswith("imaging_system/reciprocal_pixel_size_"):
                result[section][path] = _microscopy_quantity(quantity, path)
    result["schema"] = SCHEMA
    return result


def _legacy_overrides(scientific: dict) -> dict:
    """Adapt public schema-2 units to the existing calculation boundary."""
    overrides = copy.deepcopy(scientific.get("calibration_overrides", {}))
    if scientific.get("schema") == SCHEMA:
        for path, quantity in overrides.items():
            unit = quantity.get("unit")
            target, factor = {
                "angstrom": ("m", 1e-10), "kV": ("V", 1000),
                "us": ("s", 1e-6), "mm": ("m", 0.001),
                "1/angstrom": ("1/Å", 1), "mrad": ("mrad", 1),
            }.get(unit, (unit, 1))
            canonical = _microscopy_quantity(quantity, path)
            if canonical["unit"] != unit:
                raise ValueError(f"Invalid schema-2 calibration override unit at {path}: {unit!r}.")
            quantity.update(value=quantity["value"] * factor, unit=target)
    _validate_overrides(overrides)
    return overrides


def json_metadata(value):
    """Preserve NumPy metadata without arbitrary-object serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"QEM metadata must contain JSON values, not {type(value).__name__}."
    )


def no_duplicate_keys(pairs):
    """Reject repeated manifest fields instead of silently keeping the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate QEM manifest field: {key}.")
        result[key] = value
    return result


def reject_constant(value):
    """Reject NaN and infinity so a manifest always round-trips exactly."""
    raise ValueError(f"QEM metadata cannot contain nonfinite number {value}.")


def acquisition_metadata(shape, metadata: dict) -> dict:
    """Normalize recorded calibration without replacing source fields."""
    if "scientific_metadata" in metadata:
        saved = copy.deepcopy(metadata["scientific_metadata"])
        validate_header(dict(container=CONTAINER, container_version=CONTAINER_VERSION,
                             codec="retained", profile="retained", shape=list(shape),
                             scientific_metadata=saved))
        return microscopy_metadata(saved)
    source = dict(metadata.get("source_metadata", {}))
    quantities = {}

    def quantity(path, factors, output_unit, source_path=None):
        source_path = source_path or "electron_microscope/" + path
        text = str(source.get(source_path, ""))
        parts = text.split()
        unit = source.get(source_path + "@units")
        if unit is None and len(parts) == 2:
            unit = parts[1]
        try:
            value = float(parts[0]) * factors[unit]
        except (KeyError, IndexError, ValueError, TypeError):
            return
        if math.isfinite(value) and value > 0:
            quantities[path] = dict(
                value=value, unit=output_unit, provenance="source_metadata"
            )

    quantity("electron_source/accelerating_voltage", {"V": 1, "kV": 1000}, "V")
    if "electron_source/accelerating_voltage" not in quantities:
        quantity(
            "electron_source/accelerating_voltage",
            {"eV": 1, "keV": 1000},
            "V",
            "entry/instrument/detector/incident_energy",
        )
    quantity(
        "illumination_system/semi_convergence_angle", {"rad": 1000, "mrad": 1}, "mrad"
    )
    quantity(
        "scan_controller/regular_scan/dwell_time",
        {"s": 1, "ms": 0.001, "us": 1e-6, "µs": 1e-6, "μs": 1e-6},
        "s",
    )
    quantity("imaging_system/camera_length", {"m": 1, "cm": 0.01, "mm": 0.001}, "m")
    for suffix in ("y", "x"):
        quantity(
            f"imaging_system/reciprocal_pixel_size_{suffix}",
            {"rad": 1000, "mrad": 1},
            "mrad",
        )
    voltage = metadata.get("voltage_kV")
    if "electron_source/accelerating_voltage" not in quantities and voltage is not None:
        value = float(voltage) * 1000
        if math.isfinite(value) and value > 0:
            quantities["electron_source/accelerating_voltage"] = dict(
                value=value, unit="V", provenance="source_metadata"
            )
    axes = [dict(name=name, size=int(size)) for name, size in zip(AXIS_NAMES, shape)]
    scan = metadata.get("scan_sampling_A")
    if scan is not None and len(scan) == 2:
        for axis, suffix, value in zip(axes, ("y", "x"), scan):
            value = float(value) * 1e-10
            if math.isfinite(value) and value > 0:
                sampling = dict(value=value, unit="m", provenance="source_metadata")
                axis["sampling"] = sampling
                quantities[f"scan_controller/regular_scan/pixel_size_{suffix}"] = dict(
                    sampling
                )
    detector = metadata.get(
        "detector_sampling", metadata.get("detector_sampling_inv_A")
    )
    unit = metadata.get("detector_sampling_unit", "1/angstrom")
    if detector is not None and len(detector) == 2:
        for axis, value in zip(axes[2:], detector):
            value = float(value)
            if math.isfinite(value) and value > 0:
                axis["sampling"] = dict(
                    value=value, unit=unit, provenance="source_metadata"
                )
    return microscopy_metadata(dict(
        schema=LEGACY_SCHEMA,
        axes=axes,
        electron_microscope=quantities,
        source_metadata=source,
        source_metadata_coverage="reader-retained",
        calibration_overrides={},
        processing=[dict(operation="lossless_storage", changes_measurements=False)],
        source_format=source.get(
            "sourceFormat", metadata.get("source_kind", "unknown")
        ),
    ))


def validate_header(header: dict) -> None:
    """Validate common metadata before interpreting an encoded acquisition."""
    if not isinstance(header, dict) or not isinstance(
        header.get("scientific_metadata"), dict
    ):
        raise ValueError(
            "Missing QEM scientific metadata; re-export the original acquisition."
        )
    scientific = header["scientific_metadata"]
    axes = scientific.get("axes", [])
    if not isinstance(axes, list) or not all(isinstance(axis, dict) for axis in axes):
        raise ValueError("Invalid QEM axes; re-export the original acquisition.")
    if (
        header.get("container") != CONTAINER
        or header.get("container_version") != CONTAINER_VERSION
        or header.get("codec") != header.get("profile")
        or scientific.get("schema") not in (LEGACY_SCHEMA, SCHEMA)
        or [axis.get("size") for axis in axes] != header.get("shape")
        or [axis.get("name") for axis in axes] != list(AXIS_NAMES)
    ):
        raise ValueError(
            "Unsupported or inconsistent QEM metadata; update the reader or re-export the original."
        )
    if scientific["schema"] == SCHEMA:
        canonical = microscopy_metadata(scientific)
        if canonical != scientific:
            raise ValueError("Schema-2 QEM quantities require canonical microscopy units.")
    _legacy_overrides(scientific)


def _validate_overrides(overrides: dict) -> None:
    units = {
        "electron_source/accelerating_voltage": {"V"},
        "illumination_system/semi_convergence_angle": {"mrad"},
        "scan_controller/regular_scan/dwell_time": {"s"},
        "imaging_system/camera_length": {"m"},
    }
    pairs = [
        ("scan_controller/regular_scan/pixel_size_", {"m"}),
        ("imaging_system/reciprocal_pixel_size_", {"mrad", "1/nm", "1/Å"}),
    ]
    for prefix, allowed in pairs:
        units.update({prefix + axis: allowed for axis in ("y", "x")})
    if not isinstance(overrides, dict):
        raise ValueError("QEM calibration overrides must be named quantities.")
    for path, quantity in overrides.items():
        if not isinstance(quantity, dict):
            raise ValueError(f"Invalid QEM calibration quantity at {path}.")
        value = quantity.get("value")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
                or quantity.get("unit") not in units.get(path, set())
                or quantity.get("provenance") != "user_override"
                or not isinstance(quantity.get("evidence"), str)
                or not quantity["evidence"]):
            raise ValueError(f"Invalid QEM calibration override at {path}; check value, units and evidence.")
        if path.startswith("scan_controller/regular_scan/pixel_size_") and not 1e-14 <= value <= 1e-6:
            raise ValueError("Scan sampling must be between 0.0001 and 10000 angstrom per pixel.")
    for prefix, _ in pairs:
        row, column = overrides.get(prefix + "y"), overrides.get(prefix + "x")
        if ((row is None) != (column is None)
                or row is not None and row["unit"] != column["unit"]):
            raise ValueError("QEM calibration requires both row and column in the same units.")


def effective_metadata(metadata: dict, scientific: dict) -> dict:
    """Restore explicit user calibration without overwriting its recorded source.

    Examples
    --------
    >>> effective_metadata({}, {"calibration_overrides": {}})
    {}
    """
    overrides = _legacy_overrides(scientific)
    result = dict(metadata)
    for prefix, field, factor in (
        ("scan_controller/regular_scan/pixel_size_", "scan_sampling_A", 1e10),
        ("imaging_system/reciprocal_pixel_size_", "detector_sampling", 1),
    ):
        if prefix + "y" in overrides:
            result[field] = [overrides[prefix + axis]["value"] * factor for axis in ("y", "x")]
            if field == "detector_sampling":
                result["detector_sampling_unit"] = overrides[prefix + "y"]["unit"]
                result.pop("detector_sampling_inv_A", None)
    voltage = overrides.get("electron_source/accelerating_voltage")
    if voltage is not None:
        result["voltage_kV"] = voltage["value"] / 1000
    return result
