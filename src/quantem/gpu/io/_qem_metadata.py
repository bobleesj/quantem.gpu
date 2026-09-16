"""Scientific metadata shared by QEM readers, independent of payload codecs."""

import math

MAGIC = b"QEMDATA1"
SCHEMA = "quantem.scientific-metadata/1"
AXIS_NAMES = ("scan_row", "scan_column", "detector_row", "detector_column")


def acquisition_metadata(shape, metadata: dict) -> dict:
    """Normalize recorded calibration without replacing source fields."""
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
