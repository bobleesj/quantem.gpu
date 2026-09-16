"""Scientific metadata shared by QEM readers, independent of payload codecs."""

import copy
import math

MAGIC = b"QEMDATA1"
SCHEMA = "quantem.scientific-metadata/1"
AXIS_NAMES = ("scan_row", "scan_column", "detector_row", "detector_column")


def acquisition_metadata(shape, metadata: dict) -> dict:
    """Normalize recorded calibration without replacing source fields."""
    if "scientific_metadata" in metadata:
        saved = copy.deepcopy(metadata["scientific_metadata"])
        validate_header(dict(container="quantem.qem", container_version=1,
                             codec="retained", profile="retained", shape=list(shape),
                             scientific_metadata=saved))
        return saved
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
    return dict(
        schema=SCHEMA,
        axes=axes,
        electron_microscope=quantities,
        source_metadata=source,
        source_metadata_coverage="reader-retained",
        calibration_overrides={},
        processing=[dict(operation="lossless_storage", changes_measurements=False)],
        source_format=source.get(
            "sourceFormat", metadata.get("source_kind", "unknown")
        ),
    )


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
        header.get("container") != "quantem.qem"
        or header.get("container_version") != 1
        or header.get("codec") != header.get("profile")
        or scientific.get("schema") != SCHEMA
        or [axis.get("size") for axis in axes] != header.get("shape")
        or [axis.get("name") for axis in axes] != list(AXIS_NAMES)
    ):
        raise ValueError(
            "Unsupported or inconsistent QEM metadata; update the reader or re-export the original."
        )
    _validate_overrides(scientific.get("calibration_overrides", {}))


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
    overrides = scientific.get("calibration_overrides", {})
    _validate_overrides(overrides)
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
