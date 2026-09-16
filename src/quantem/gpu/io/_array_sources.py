"""Explicit NumPy and EMPAD float-export readers for portable QEM conversion."""

from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .models import FourDSTEMData


def _pair(text: str) -> tuple[int, int]:
    values = tuple(int(part.strip()) for part in text.strip("()[] ").split(","))
    if len(values) != 2 or min(values) <= 0:
        raise ValueError(
            "EMPAD scan shape must give positive (row, column) dimensions."
        )
    return values


def _empad(path: Path, scan_shape: tuple[int, int] | None) -> FourDSTEMData:
    xml = path if path.suffix.lower() == ".xml" else path.with_suffix(".xml")
    fields, original, shape, record_rows = {}, {}, scan_shape, 130
    raw = path
    if xml.exists():
        with xml.open("rb") as handle:
            text = handle.read(4 * 1024 * 1024 + 1)
        if (
            len(text) > 4 * 1024 * 1024
            or b"<!DOCTYPE" in text.upper()
            or b"<!ENTITY" in text.upper()
        ):
            raise ValueError("EMPAD XML must be at most 4 MiB with no DTD or entities.")
        root = ET.fromstring(text)

        def visit(node, prefix=""):
            for child in node:
                component = (
                    child.attrib.get("mode", "acquire")
                    if child.tag == "scan_parameters"
                    else child.tag
                )
                key = prefix + component
                if len(child):
                    visit(child, key + "/")
                else:
                    value = (child.text or "").strip()
                    if key in fields and fields[key] != value:
                        raise ValueError(f"Conflicting EMPAD XML field {key}.")
                    fields[key] = value

        visit(root)
        original = {
            "empad_xml": text.decode("utf8"),
            **{"empad/" + k: v for k, v in fields.items()},
        }
        modern = fields.get("sensor/type") == "EMPAD2"
        if "sensor/type" in fields or "rawfile/filename" in fields:
            if (
                not modern
                or fields.get("rawfile/datatype") != "float32"
                or fields.get("scan/type") != "scan"
                or _pair(fields.get("sensor/shape", "0,0")) != (128, 128)
            ):
                raise ValueError(
                    "Open an EMPAD raster float32 export, not encoded detector words."
                )
            if any(
                key in fields
                for key in (
                    "grabber/avg_scan_even_offset",
                    "grabber/avg_scan_odd_offset",
                )
            ):
                raise NotImplementedError(
                    "EMPAD2 raw acquisition calibration is not implemented in the Python reference reader; use a verified float32 export."
                )
            shape, record_rows = _pair(fields["scan/shape"]), 128
            filename = fields["rawfile/filename"]
        else:
            candidates = []
            for row_key, col_key in (
                ("pix_y", "pix_x"),
                ("acquire/scan_resolution_y", "acquire/scan_resolution_x"),
            ):
                if row_key in fields or col_key in fields:
                    candidates.append(
                        _pair(fields.get(row_key, "0") + "," + fields.get(col_key, "0"))
                    )
            if not candidates or any(
                candidate != candidates[0] for candidate in candidates
            ):
                raise ValueError(
                    "EMPAD XML needs consistent row/column scan dimensions."
                )
            shape = candidates[0]
            element = root.find("raw_file")
            filename = "" if element is None else element.attrib.get("filename", "")
            if fields.get("type", "scan") != "scan":
                raise ValueError("Only EMPAD raster scans are supported.")
        basename = filename.replace("\\", "/").split("/")[-1]
        if not basename or Path(basename).suffix.lower() != ".raw":
            raise ValueError("EMPAD XML must name a sibling .raw file.")
        raw = xml.parent / basename
        if path.suffix.lower() == ".raw" and raw.resolve() != path.resolve():
            raise ValueError(
                "EMPAD XML names a different RAW file; open the matching XML."
            )
        if scan_shape is not None and tuple(scan_shape) != shape:
            raise ValueError("scan_shape conflicts with EMPAD XML; omit the override.")
    elif path.suffix.lower() == ".xml":
        raise FileNotFoundError(path)
    if (
        shape is None
        or len(shape) != 2
        or any(type(n) is not int or n <= 0 for n in shape)
    ):
        raise ValueError(
            "Headerless EMPAD-G1 RAW requires scan_shape=(rows, columns); the detector record is 130x128 float32."
        )
    if raw.stat().st_size != math.prod(shape) * record_rows * 128 * 4:
        raise ValueError(
            "EMPAD RAW length disagrees with the declared layout; restore the matching XML/RAW files."
        )
    data = np.memmap(raw, dtype="<f4", mode="r", shape=(*shape, record_rows, 128))[
        :, :, :128, :
    ]
    metadata = dict(
        source_kind="empad-float-export",
        source_format="EMPAD float32",
        source_metadata=original,
        backend="cpu",
        representation="dense",
        source_path=str(path),
        background_applied=False,
    )
    modern = record_rows == 128
    for field, target, unit in (
        (
            "iom_measurements/ColumnSourceHighVoltage"
            if modern
            else "iom_measurements/high_voltage",
            "electron_source/accelerating_voltage",
            "V",
        ),
        (
            "iom_measurements/ColumnOpticsGetCameraLengthNominalCameraLength"
            if modern
            else "iom_measurements/nominal_camera_length",
            "imaging_system/camera_length",
            "m",
        ),
        (
            "scan/exposure_time" if modern else "exposure_time",
            "scan_controller/regular_scan/dwell_time",
            "s" if modern else "ms",
        ),
    ):
        if field in fields:
            original["electron_microscope/" + target] = fields[field] + " " + unit

    def positive(key: str) -> float | None:
        try:
            value = float(fields[key])
        except (KeyError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None

    fov_root = "iom_measurements/full_scan_field_of_view/"
    x, y, factor = (positive(fov_root + key) for key in ("x", "y", "scale_factor"))
    fov = fields.get("iom_measurements/optics.get_full_scan_field_of_view")
    if x is not None and x == y and factor is not None:
        # EMPAD 1.2 records maximum-axis FOV including the instrument scale.
        # Sampling stays isotropic for rectangular scans, matching the native reader.
        metadata["scan_sampling_A"] = [x / factor / max(shape) * 1e10] * 2
    elif fov:
        values = json.loads(fov)
        if len(values) == 2 and all(
            isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values
        ):
            metadata["scan_sampling_A"] = [v / n * 1e10 for v, n in zip(values, shape)]
    angle = (
        positive("iom_measurements/calibrated_diffraction_angle") if modern else None
    )
    reciprocal = (
        positive("iom_measurements/calibrated_pixelsize") if not modern else None
    )
    if angle is not None:
        metadata["detector_sampling"] = [angle * 1000] * 2
        metadata["detector_sampling_unit"] = "mrad"
        for axis in ("y", "x"):
            original[
                "electron_microscope/imaging_system/reciprocal_pixel_size_" + axis
            ] = f"{angle} rad"
    elif reciprocal is not None:
        # EMPAD's legacy field uses the native contract: value * 1e9 in 1/nm,
        # then *0.1 to reach the QEM microscopy unit 1/angstrom.
        metadata["detector_sampling"] = [reciprocal * 1e8] * 2
        metadata["detector_sampling_unit"] = "1/angstrom"
    metadata["qem_empad"] = dict(
        format_identifier="empad-float-export/v1",
        format_name="EMPAD float32",
        microscope_metadata=original,
    )
    return FourDSTEMData(data, metadata)


def load_array_source(
    path: str | Path, scan_shape: tuple[int, int] | None = None
) -> FourDSTEMData:
    """Map explicit array sources without silently changing measurement precision."""
    path = Path(path)
    if path.suffix.lower() != ".npy":
        return _empad(path, scan_shape)
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 4 or min(data.shape) <= 0:
        raise ValueError(
            "NumPy acquisition must have four axes: scan row, scan column, detector row, detector column."
        )
    if scan_shape is not None and tuple(scan_shape) != data.shape[:2]:
        raise ValueError("scan_shape disagrees with the NumPy array.")
    return FourDSTEMData(
        data,
        dict(
            backend="cpu",
            representation="dense",
            source_kind="numpy",
            source_format="NumPy",
            source_metadata={},
            source_path=str(path),
        ),
    )
