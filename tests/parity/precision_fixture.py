"""Independent NumPy oracle for packed display-precision backends."""

from __future__ import annotations

import numpy as np


def make_precision_fixture() -> np.ndarray:
    """Return deterministic signed intensities with scan and detector structure."""

    scan_row, scan_col, det_row, det_col = np.indices((5, 6, 7, 9))
    values = (
        17.0 * scan_row
        - 9.0 * scan_col
        + 2.25 * det_row
        + 0.375 * det_col
        + np.sin((scan_row * 11 + scan_col * 7 + det_row * 5 + det_col) / 3.0)
        * 31.0
    ).astype(np.float32)
    values[0, 0, 0, 0] = 0.0
    values[4, 5, 6, 8] = 1716.4449462890625
    return values


def encode_precision_reference(values: np.ndarray, report: dict) -> np.ndarray:
    """Encode one archive using only the documented NumPy operations."""

    if report["storage"] == "float16":
        return values.astype(np.float16)
    return np.rint(
        (values.astype(np.float64) - report["offset"]) / report["scale"]
    ).clip(0, 65535).astype(np.uint16)


def restore_precision_reference(values: np.ndarray, report: dict) -> np.ndarray:
    """Apply the documented archive conversion with NumPy only."""

    codes = encode_precision_reference(values, report)
    if report["storage"] == "float16":
        return codes.astype(np.float32)
    return (
        codes.astype(np.float64) * report["scale"] + report["offset"]
    ).astype(np.float32)


def precision_error_reference(
    values: np.ndarray, restored: np.ndarray
) -> dict[str, float | int]:
    """Measure persisted report fields independently in restored units."""

    difference = restored.astype(np.float64) - values.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "positive_to_zero": int(np.count_nonzero((values > 0) & (restored == 0))),
        "changed": int(np.count_nonzero(values != restored)),
        "overflow": int(np.count_nonzero(~np.isfinite(restored))),
    }
