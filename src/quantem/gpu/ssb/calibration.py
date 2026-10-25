"""Export a fitted SSB model in ShowPtycho's calibration format."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from quantem.gpu.geometry import sample_to_propagator_tilt
from quantem.gpu.optics.physics import ssb_upsampling_factor

CALIBRATION_SCHEMA = "quantem.ssb-calibration/6"


def capture_calibration(session, result):
    """Snapshot small acquisition metadata, without hashing files or retaining GPU state."""
    if result.calibration_context is not None or session.source_kind == "array" or session.source_path is None:
        return
    shape = session.input_scan_shape
    prepared_shape = tuple(int(n) // result.upsampling_factor for n in result.object_wave.shape[-2:])
    result.calibration_context = {
        "scan_shape": list(shape),
        "scan_region": list(session.scan_region or (0, shape[0], 0, shape[1])),
        "session_identity": {
            "data": {"kind": session.source_kind, "dtype": session.source_dtype,
                     "bytes": session.source_bytes, "scan_shape": list(prepared_shape),
                     "scan_region": None if session.scan_region is None else list(session.scan_region),
                     "detector_bin": session.source_detector_bin},
            "geometry": {
                "voltage_kV": session.voltage_kV, "semiangle_mrad": session.semiangle_mrad,
                "scan_sampling_A": list(session.scan_sampling_A),
                "detector_sampling_mrad": (None if session.detector_sampling_mrad is None
                                           else list(session.detector_sampling_mrad)),
                "scan_detector_rotation_deg": result.scan_detector_rotation_deg,
                "scan_detector_rotation_flipped": session.scan_detector_rotation_flipped,
                "bright_field_radius_px": session.bright_field_radius_px,
                "bright_field_intensity_threshold": session.bright_field_intensity_threshold,
                "num_bright_field_pixels": result.num_bright_field_pixels,
                "bright_field_center_px": (None if session.bright_field_center_px is None
                                          else list(session.bright_field_center_px)),
            },
        },
    }
    result._calibration_source = {
        "source_path": session.source_path, "scan_shape": shape,
        "hot_pixel_correction": deepcopy((session.source_provenance or {}).get("hot_pixel_correction")),
        "storage_path": session.source_storage_path, "calibration_path": session.calibration_path,
        "manifest_path": session.source_manifest_path,
    }


def save_calibration(result, path):
    """Write a calibration snapshot, with no reconstruction or objective evaluation."""
    from quantem.gpu.ssb.persistence import (
        _write_atomically,
        json_value,
        portable_source,
        source_identity,
    )

    path = Path(path).expanduser()
    if path.suffix.lower() != ".json":
        raise ValueError("Use a JSON filename, for example fit.save_calibration('calibration.json').")
    if result.calibration_context is None:
        raise ValueError("This fit has no persistent acquisition calibration. Fit a file opened with DirectPtycho.open() "
                         "before saving a calibration for ShowPtycho.")
    if result.fit_scan_region is not None:
        raise ValueError("This fit describes a subregion of its session. Select that region with session.select_region(...), "
                         "fit it, and save its calibration to preserve the region's acquisition identity.")
    method = result.reconstruction_method
    if method not in ("ssb", "phase", "complex"):
        raise ValueError(f"Unsupported direct reconstruction method {method!r}; save an SSB, phase or complex result.")
    context = deepcopy(result.calibration_context)
    identity = context["session_identity"]
    if "source" not in identity:
        if result._calibration_source is None:
            raise ValueError("This saved fit has no acquisition fingerprint. Reopen its acquisition and reload the fit.")
        identity["source"] = source_identity(**result._calibration_source)
    geometry = identity["geometry"]
    if any(geometry.get(key) is None for key in (
        "voltage_kV", "semiangle_mrad", "scan_sampling_A", "detector_sampling_mrad", "num_bright_field_pixels",
    )):
        raise ValueError("The fit's acquisition calibration is incomplete. Fit a calibrated DirectPtycho.open() session first.")
    sample = result.sample_tilt_mrad
    depth = result.depth_spread_nm
    # A zero-depth model applies no depth-dependent tilt correction.
    if depth == 0 and sample is not None:
        sample, depth = None, None
    if (sample is None) != (depth is None) or (depth is not None and depth < 0):
        raise ValueError("Sample tilt and positive depth spread must describe one complete model.")
    factor = (ssb_upsampling_factor(scan_sampling_A=geometry["scan_sampling_A"],
                                   semiangle_mrad=geometry["semiangle_mrad"], voltage_kV=geometry["voltage_kV"])
              if method == "ssb" else result.upsampling_factor)
    record = {
        "schema": CALIBRATION_SCHEMA, "id": uuid4().hex[:12], "generated_at_utc": datetime.now(UTC).isoformat(),
        "notes": "", "aberrations": dict(result.aberrations),
        "scan_detector_rotation_deg": result.scan_detector_rotation_deg,
        "scan_detector_rotation_flipped": result.scan_detector_rotation_flipped,
        **{key: geometry[key] for key in ("voltage_kV", "semiangle_mrad", "scan_sampling_A", "detector_sampling_mrad")},
        "source": portable_source(result.source_path, path.parent), "scan_region": context["scan_region"],
        "scan_shape": context["scan_shape"], "bright_field_radius_px": geometry["bright_field_radius_px"],
        "sample_tilt_mrad": sample, "depth_spread_nm": depth,
        "propagator_tilt_mrad": (None if sample is None else sample_to_propagator_tilt(
            sample, scan_detector_rotation_deg=result.scan_detector_rotation_deg,
            scan_detector_rotation_flipped=result.scan_detector_rotation_flipped)),
        "loss": result.loss,
        # ShowPtycho's conventional SSB view, the phase of the mean wave, whatever image the result holds (a result
        # saved with a retired estimator included); the regularized solver is recorded separately below.
        "phase_estimator": "phase_of_mean",
        "reconstruction_method": method,
        "reconstruction_options": deepcopy(result.reconstruction_options),
        "reconstruction_diagnostics": deepcopy(result.reconstruction_diagnostics),
        "upsampling_factor": factor, "upsampling_mode": "auto" if method == "ssb" else "explicit",
        "display_negate_phase": False,
        "display": {"tilt_correction": sample is not None, "max_columns": 3},
        "session_identity": identity, "local_tilt": None,
    }
    # Finite JSON only; do not silently alter coefficients or emit NaN/Infinity.
    encoded = json.dumps(json_value(record), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(path, "w", lambda file: file.write(encoded), encoding="utf-8")
    return path
