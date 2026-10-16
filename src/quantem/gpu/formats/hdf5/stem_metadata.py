"""Read the paired ARINA microscope file without reading virtual images."""

import math
import sys
from pathlib import Path

import h5py
import numpy as np


def _companion_path(master: Path) -> Path | None:
    if not master.name.endswith("_master.h5"):
        return None
    companion = master.with_name(master.name.removesuffix("_master.h5") + ".h5")
    return companion if companion.is_file() else None


def _scalar(value: object) -> object:
    values = np.asarray(value)
    if values.size == 1:
        result = values.reshape(-1)[0].item()
        return result.decode() if isinstance(result, bytes) else result
    return values.tolist()


def _positive(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= sys.float_info.min else None


def _read_stem_metadata(master: Path, metadata: dict, recorded_shape=None) -> dict:
    """Normalize an exactly paired STEM Metadata group with a matching raster.

    This legacy ARINA schema stores Pixel Size in nm, High Tension in V,
    and Camera Length in m. The source calibration flag is authoritative:
    a nominal pixel size is retained but never promoted to calibrated sampling.
    Scan Rotation is not the scan-to-detector rotation.
    """
    companion = _companion_path(master)
    if companion is None:
        return {}
    try:
        with h5py.File(companion, "r") as handle:
            if "STEM Metadata" not in handle:
                return {}
            raw = {
                key: _scalar(value)
                for key, value in handle["STEM Metadata"].attrs.items()
                if np.asarray(value).size <= 100
            }
    except (OSError, TypeError, ValueError, UnicodeDecodeError):
        return {"microscope_metadata_warning": "Paired microscope metadata is unreadable."}

    result = {"STEM Metadata@" + key: value for key, value in raw.items()}
    result["microscope_metadata_source"] = companion.name
    rows, columns = (_positive(raw.get(key)) for key in ("NumPixelsY", "NumPixelsX"))
    if (rows is None or columns is None or not rows.is_integer() or not columns.is_integer()
            or rows * columns != metadata.get("n_frames")):
        result["microscope_metadata_warning"] = (
            "Paired microscope metadata does not match the detector frame count; "
            "its calibration was not applied."
        )
        return result
    shape = (int(rows), int(columns))
    if recorded_shape is not None and tuple(recorded_shape) != shape:
        result["microscope_metadata_warning"] = (
            "Paired microscope metadata has mismatched scan dimensions; "
            "its calibration was not applied."
        )
        return result
    result.update(scan_shape=shape, sourceFormat="ARINA HDF5 + STEM metadata")
    notes = []
    pixel_nm = _positive(raw.get("Pixel Size"))
    calibrated = raw.get("IsPixelSizeCalibrated") == 1
    result["scan_sampling_calibrated"] = calibrated
    if pixel_nm is not None and math.isfinite(pixel_nm * 10):
        result["scan_sampling_nominal_A"] = [pixel_nm * 10, pixel_nm * 10]
        if calibrated:
            result["scan_sampling_A"] = result["scan_sampling_nominal_A"].copy()
        else:
            notes.append(
                "The recorded scan step is nominal (IsPixelSizeCalibrated is not 1). "
                "Confirm scan calibration before quantitative analysis."
            )
    else:
        notes.append("The recorded scan step is invalid; supply a verified scan calibration.")
    for key, field, factor, path, unit in (
        ("High Tension", "voltage_kV", 0.001, "electron_source/accelerating_voltage", "V"),
        ("Camera Length", "camera_length_mm", 1000, "imaging_system/camera_length", "m"),
    ):
        value = _positive(raw.get(key))
        if value is not None and math.isfinite(value * factor):
            result[field] = value * factor
            result["electron_microscope/" + path] = value
            result["electron_microscope/" + path + "@units"] = unit
    detector_energy = _positive(metadata.get(
        "entry/instrument/detector/detectorSpecific/photon_energy"))
    if (detector_energy is not None and "voltage_kV" in result
            and not math.isclose(detector_energy / 1000, result["voltage_kV"], rel_tol=1e-6)):
        notes.append(
            "Detector photon_energy disagrees with microscope High Tension; "
            "the microscope voltage is used. Both recorded values are retained."
        )
    if notes:
        result["microscope_metadata_warning"] = " ".join(notes)
    return result
