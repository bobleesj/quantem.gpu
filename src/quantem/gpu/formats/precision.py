"""The saved record of a scaled-precision export: storage, calibration and measured error.

``io.save(..., dtype="scaled_uint16")`` stores intensities as uint16 codes and
records, in one JSON attribute (``.qem``: the ``intensity_calibration``
header), how to restore them: one scale and offset per frame region, plus the
GPU-measured conversion error. Readers must refuse a record that does not
cover every frame exactly once, because a missing or overlapping region would
silently restore the wrong intensities.
"""

import json
import math
from pathlib import Path

import h5py

PRECISION_ATTRIBUTE = "quantem_precision_v1"


def saved_precision(path):
    """Read only persisted precision metadata, never detector values."""
    if Path(path).suffix.lower() not in {".h5", ".hdf5"} or not h5py.is_hdf5(path):
        return None
    try:
        with h5py.File(path, "r") as handle:
            value = handle.attrs.get(PRECISION_ATTRIBUTE)
    except OSError:
        return None
    if value is None:
        return None
    report = json.loads(value)
    if report.get("complete") is False:
        raise ValueError(
            "This precision export did not complete; repeat the export from its source."
        )
    if report.get("version") not in {1, 2} or report.get("storage") not in {
        "float16",
        "scaled_uint16",
    }:
        raise ValueError(
            "Unsupported saved precision metadata; use a compatible QuantEM version."
        )
    if report["version"] == 2:
        validate_regions(report)
    return report


def validate_regions(report):
    """Reject incomplete or ambiguous per-frame calibration metadata."""
    regions = report.get("regions", [])
    first = 0
    for region in regions:
        if (
            region.get("first_frame") != first
            or not isinstance(region.get("stop_frame"), int)
            or region["stop_frame"] <= first
            or not math.isfinite(region.get("scale", math.nan))
            or region["scale"] <= 0
            or not math.isfinite(region.get("offset", math.nan))
        ):
            raise ValueError(
                "Invalid regional intensity calibration; repeat the export from its source."
            )
        first = region["stop_frame"]
    if not regions or first != math.prod(report["source_shape"][:2]):
        raise ValueError(
            "Regional calibration does not cover the saved scan; repeat the export."
        )


def part_reports(report, ends):
    """Associate each packed part with its authoritative calibration."""
    if report.get("version") != 2:
        return [report] * len(ends)
    validate_regions(report)
    result = []
    index, first = 0, 0
    for stop in ends:
        while first >= report["regions"][index]["stop_frame"]:
            index += 1
        region = report["regions"][index]
        if stop > region["stop_frame"]:
            raise ValueError(
                "A packed part crosses calibration boundaries; reload using a compatible reader."
            )
        result.append(region)
        first = stop
    return result
