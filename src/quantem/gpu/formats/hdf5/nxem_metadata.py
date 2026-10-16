"""Read acquisition-matched NXem calibration without reading detector data."""

import math
from pathlib import Path

import h5py

from quantem.gpu.formats.hdf5.stem_metadata import _positive, _scalar


def _nxem_companion(master: Path) -> Path | None:
    if not master.name.endswith("_master.h5"):
        return None
    name = master.name.removesuffix("_master.h5") + "_em_metadata.h5"
    companion = master.with_name(name)
    return companion if companion.is_file() else None


def _read_nxem_metadata(master: Path, metadata: dict, recorded_shape=None) -> dict:
    """Use explicitly unit-labelled quantities only after matching the raster."""
    companion = _nxem_companion(master)
    if companion is None:
        return {}
    result = {"microscope_metadata_source": companion.name}
    raw = {}
    try:
        with h5py.File(companion, "r") as handle:
            def visit(name, item):
                if not isinstance(item, h5py.Dataset) or item.size != 1:
                    return
                key = "electron_microscope/" + name
                raw[key] = _scalar(item[()])
                if "units" in item.attrs:
                    raw[key + "@units"] = _scalar(item.attrs["units"])

            handle["electron_microscope"].visititems(visit)
    except (OSError, KeyError, TypeError, ValueError, UnicodeDecodeError):
        result["microscope_metadata_warning"] = (
            "Paired NXem metadata is unreadable; confirm the companion file."
        )
        return result

    scan = "electron_microscope/scan_controller/regular_scan/"
    rows, columns = (_positive(raw.get(scan + "n_pixels_" + axis)) for axis in ("y", "x"))
    if (rows is None or columns is None or not rows.is_integer() or not columns.is_integer()
            or rows * columns != metadata.get("n_frames")
            or raw.get(scan + "n_frames", 1) != 1
            or raw.get("electron_microscope/scan_controller/scan_type", "regular") != "regular"
            or (recorded_shape is not None and tuple(recorded_shape) != (rows, columns))):
        result["microscope_metadata_warning"] = (
            "Paired NXem raster does not match the acquisition; "
            "its calibration was not applied. Confirm the companion file."
        )
        return result
    result.update(raw)
    result.update(scan_shape=(int(rows), int(columns)), sourceFormat="ARINA HDF5 + NXem metadata")

    def quantity(path: str, factors: dict[str, float]) -> float | None:
        key = "electron_microscope/" + path
        value = _positive(raw.get(key))
        factor = factors.get(raw.get(key + "@units"))
        if value is None or factor is None:
            return None
        converted = value * factor
        return converted if math.isfinite(converted) else None

    for field, path, factors in (
        ("scan_sampling_A", "scan_controller/regular_scan/pixel_size_",
         {"m": 1e10, "um": 1e4, "µm": 1e4, "nm": 10, "angstrom": 1, "Å": 1}),
        ("detector_sampling_mrad", "imaging_system/reciprocal_pixel_size_",
         {"rad": 1000, "mrad": 1}),
    ):
        pair = [quantity(path + axis, factors) for axis in ("y", "x")]
        if all(value is not None for value in pair):
            result[field] = pair
    for field, path, factors in (
        ("voltage_kV", "electron_source/accelerating_voltage", {"V": .001, "kV": 1}),
        ("camera_length_mm", "imaging_system/camera_length", {"m": 1000, "mm": 1, "cm": 10}),
        ("semiangle_mrad", "illumination_system/semi_convergence_angle", {"rad": 1000, "mrad": 1}),
    ):
        value = quantity(path, factors)
        if value is not None:
            result[field] = value
    # Scan-controller rotation is not a calibrated scan-to-detector rotation.
    return result
