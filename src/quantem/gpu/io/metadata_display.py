"""Compact acquisition calibration tables, without reading detector values."""

from quantem.gpu.formats.qem.metadata import DETECTOR_PIXEL_SIZE, SCAN_PIXEL_SIZE


def calibration_rows(metadata: dict) -> dict:
    """Describe effective recorded values and their saved override provenance."""
    scientific = metadata.get("scientific_metadata", {})
    overrides = scientific.get("calibration_overrides", {})

    def row(value, unit="", paths=()):
        if value is None:
            return {"Value": "Not recorded", "Unit": unit, "Metadata": "Not recorded"}
        if isinstance(value, (list, tuple)):
            value = tuple(value)
        overridden = [path in overrides for path in paths]
        origin = "Recorded"
        if any(overridden):
            origin = (
                "Saved override" if all(overridden) else "Recorded + saved override"
            )
        return {"Value": value, "Unit": unit, "Metadata": origin}

    detector_key, detector_unit = (
        ("detector_sampling_inv_A", "1/angstrom per pixel")
        if metadata.get("detector_sampling_inv_A") is not None
        else ("detector_sampling_mrad", "mrad per pixel")
    )
    if metadata.get(detector_key) is None:
        detector_unit = ""
    rows = {
        "Accelerating voltage": row(
            metadata.get("voltage_kV"), "kV", ("electron_source/accelerating_voltage",)
        ),
        "Convergence semiangle": row(
            metadata.get("semiangle_mrad"),
            "mrad",
            ("illumination_system/semi_convergence_angle",),
        ),
        "Scan step (row, col)": row(
            metadata.get("scan_sampling_A"), "angstrom", SCAN_PIXEL_SIZE
        ),
        "Detector step (row, col)": row(
            metadata.get(detector_key), detector_unit, DETECTOR_PIXEL_SIZE
        ),
        "Scan-detector rotation": row(
            metadata.get("scan_detector_rotation_deg"), "deg"
        ),
        "Dwell time": row(
            metadata.get("dwell_time_us"),
            "us",
            ("scan_controller/regular_scan/dwell_time",),
        ),
    }
    if metadata.get("camera_length_mm") is not None:
        rows["Camera length"] = row(
            metadata["camera_length_mm"], "mm", ("imaging_system/camera_length",)
        )
    correction = metadata.get("hot_pixel_correction", {})
    rows["Pixel correction"] = row(
        f"{correction.get('method', 'unspecified')} ({correction['pixel_count']} pixels)"
        if correction.get("applied") and "pixel_count" in correction
        else "Not applied"
        if correction.get("applied") is False
        else None
    )
    source_fields = scientific.get(
        "source_metadata", metadata.get("source_metadata", {})
    )
    if source_fields:
        rows["Source metadata"] = row(
            f"{len(source_fields)} fields retained in data.metadata"
        )
    return rows
