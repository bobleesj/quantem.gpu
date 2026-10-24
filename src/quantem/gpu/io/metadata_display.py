"""Compact acquisition calibration tables, without reading detector values.

One label set. Every calibration table of quantem.gpu (``data.calibration.table()``, ``data.metadata_table()``,
``io.inspect``, ``SSB.calibration_table()``) and of applications built on it names its rows, units and sources
with the constants below, so the same quantity reads the same in every table.
"""

import math
import numbers
from typing import TYPE_CHECKING

from quantem.gpu.formats.qem.metadata import (
    BRIGHT_FIELD_DISK_FIT,
    BRIGHT_FIELD_DISK_KEY,
    DETECTOR_PIXEL_SIZE,
    SCAN_PIXEL_SIZE,
)

if TYPE_CHECKING:
    import pandas as pd

# Row label of each calibration, by its metadata key. detector_sampling_inv_A shares the detector row.
CALIBRATION_LABELS = {
    "voltage_kV": "Accelerating voltage",
    "semiangle_mrad": "Convergence semiangle",
    "scan_sampling_A": "Scan step (row, col)",
    "detector_sampling_mrad": "Detector step (row, col)",
    "detector_sampling_inv_A": "Detector step (row, col)",
    "scan_detector_rotation_deg": "Scan-detector rotation",
    "scan_detector_rotation_flipped": "Rotation flipped",
    "bright_field_center_px": "Bright-field centre (row, col)",
    "bright_field_edge_radius_px": "Bright-field edge radius",
}
# Unit of each calibration as tables print it.
CALIBRATION_UNITS = {
    "voltage_kV": "kV",
    "semiangle_mrad": "mrad",
    "scan_sampling_A": "angstrom",
    "detector_sampling_mrad": "mrad per pixel",
    "detector_sampling_inv_A": "1/angstrom per pixel",
    "scan_detector_rotation_deg": "deg",
    "scan_detector_rotation_flipped": "",
    "bright_field_center_px": "px",
    "bright_field_edge_radius_px": "px",
}
# Where a value comes from. The file's own states:
RECORDED = "Recorded"
OVERRIDDEN = "Overridden"  # set on data.calibration and saved beside the file's record
PARTLY_OVERRIDDEN = "Recorded + overridden"
NOT_RECORDED = "Not recorded"
FITTED_BRIGHT_FIELD_DISK = "Fitted (bright-field disk)"  # data.calibration.fit_detector_sampling()
# and the states only a session's launch table states, for the value it uses:
EXPLICIT_OVERRIDE = "Explicit override"  # a keyword given to DirectPtycho(...) or DirectPtycho.open(...)
ESTIMATED = "Estimated"
ESTIMATED_ROTATION = "Estimated (centre-of-mass curl)"
DEFAULT = "Default"
MODIFIED_IN_SESSION = "Modified in session"
PENDING_PROBE_FIT = "Pending probe fit"


def calibration_rows(metadata: dict) -> dict:
    """Describe effective recorded values and the values the user overrode.

    Returns ``{label: {"Value", "Unit", "Metadata"}}`` with the labels, units and source states of this module.
    """
    scientific = metadata.get("scientific_metadata", {})
    overrides = scientific.get("calibration_overrides", {})
    recorded_quantities = scientific.get("electron_microscope", {})

    def overridden_path(path) -> bool:
        # a value set equal to the file's record confirms it: the source stays Recorded
        override, recorded = overrides.get(path), recorded_quantities.get(path)
        if override is None:
            return False
        if not isinstance(override, dict) or not isinstance(recorded, dict):
            return True
        try:
            same = (override.get("unit") == recorded.get("unit")
                    and math.isclose(float(override["value"]), float(recorded["value"]), rel_tol=1e-12, abs_tol=0.0))
        except (KeyError, TypeError, ValueError):
            same = False
        return not same

    def row(value, unit="", paths=(), *, positive=False):
        # a physical calibration of zero or less (a placeholder some detectors write) is no calibration
        if value is None or (positive and not all(
                isinstance(item, numbers.Real) and math.isfinite(item) and item > 0
                for item in (value if isinstance(value, (list, tuple)) else (value,)))):
            return {"Value": NOT_RECORDED, "Unit": unit, "Metadata": NOT_RECORDED}
        if isinstance(value, (list, tuple)):
            value = tuple(value)
        overridden = [overridden_path(path) for path in paths]
        origin = RECORDED
        if any(overridden):
            origin = OVERRIDDEN if all(overridden) else PARTLY_OVERRIDDEN
        if paths and all(isinstance(overrides.get(path), dict)
                         and overrides[path].get("provenance") == BRIGHT_FIELD_DISK_FIT for path in paths):
            origin = FITTED_BRIGHT_FIELD_DISK
        return {"Value": value, "Unit": unit, "Metadata": origin}

    detector_key = (
        "detector_sampling_inv_A" if metadata.get("detector_sampling_inv_A") is not None else "detector_sampling_mrad"
    )
    detector_unit = CALIBRATION_UNITS[detector_key] if metadata.get(detector_key) is not None else ""
    rows = {
        CALIBRATION_LABELS["voltage_kV"]: row(
            metadata.get("voltage_kV"), CALIBRATION_UNITS["voltage_kV"], ("electron_source/accelerating_voltage",), positive=True
        ),
        CALIBRATION_LABELS["semiangle_mrad"]: row(
            metadata.get("semiangle_mrad"),
            CALIBRATION_UNITS["semiangle_mrad"],
            ("illumination_system/semi_convergence_angle",), positive=True
        ),
        CALIBRATION_LABELS["scan_sampling_A"]: row(
            metadata.get("scan_sampling_A"), CALIBRATION_UNITS["scan_sampling_A"], SCAN_PIXEL_SIZE, positive=True
        ),
        CALIBRATION_LABELS[detector_key]: row(
            metadata.get(detector_key), detector_unit, DETECTOR_PIXEL_SIZE, positive=True
        ),
        CALIBRATION_LABELS["scan_detector_rotation_deg"]: row(
            metadata.get("scan_detector_rotation_deg"), CALIBRATION_UNITS["scan_detector_rotation_deg"]
        ),
        CALIBRATION_LABELS["scan_detector_rotation_flipped"]: row(
            None if metadata.get("scan_detector_rotation_flipped") is None
            else ("yes" if metadata["scan_detector_rotation_flipped"] else "no")
        ),
        "Dwell time": row(
            metadata.get("dwell_time_us"),
            "us",
            ("scan_controller/regular_scan/dwell_time",),
        ),
    }
    disk = metadata.get(BRIGHT_FIELD_DISK_KEY)
    if isinstance(disk, dict):
        for name in ("bright_field_center_px", "bright_field_edge_radius_px"):
            if disk.get(name) is not None:
                value = disk[name]
                rows[CALIBRATION_LABELS[name]] = {"Value": tuple(value) if isinstance(value, list) else value,
                                                  "Unit": CALIBRATION_UNITS[name], "Metadata": FITTED_BRIGHT_FIELD_DISK}
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


def display_value(value, digits: int = 6):
    """A calibration value as a table shows it: floats to ``digits`` significant figures, pairs element by element.

    Tables round for reading; sessions and records keep the full value.
    """
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, tuple, list)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(display_value(item, digits) for item in value)
    if isinstance(value, numbers.Integral):
        return value
    return float(f"{float(value):.{digits}g}")


def calibration_table(metadata: dict) -> "pd.DataFrame":
    """The microscope calibration rows of :func:`calibration_rows` as ``Value``, ``Unit`` and ``Source`` columns."""
    import pandas as pd

    rows = calibration_rows(metadata)
    labels = dict.fromkeys(CALIBRATION_LABELS[name] for name in
                           ("voltage_kV", "semiangle_mrad", "scan_sampling_A", "detector_sampling_mrad",
                            "bright_field_center_px", "bright_field_edge_radius_px")
                           if CALIBRATION_LABELS[name] in rows)
    table = {label: {"Value": display_value(rows[label]["Value"]), "Unit": rows[label]["Unit"], "Source": rows[label]["Metadata"]}
             for label in labels}
    return pd.DataFrame.from_dict(table, orient="index", columns=["Value", "Unit", "Source"]).rename_axis("Calibration")
