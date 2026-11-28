"""Export a fitted direct ptychography model in ShowPtycho's calibration format."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from quantem.gpu.geometry import sample_to_propagator_tilt
from quantem.gpu.optics.physics import direct_auto_upsample

# 10: the tilt along the detector axes is "tilt_mrad" (was "propagator_tilt_mrad"): one name with myptycho's
#    Propagator(tilt_mrad=); the scan-frame tilt stays "sample_tilt_mrad". No number changes.
# 9: the schema takes the direct-ptychography name, quantem.direct-calibration; the layout is that of 8. A record
#    also says what its fit searched, "max_order" and "higher_order_limit" (None when it was not a fifth-order fit;
#    absent in a record written before they were added): load_calibration rebuilds the fit's search bounds from them.
# 8: the output factor is "upsample" with "upsample_mode" (were "upsampling", "upsampling_mode"), the conventional
#    method is "phase_of_mean", the direct strengths are phase_damp and amplitude_damp (were
#    phase_damping and amplitude_damping), and "phase_estimator" is gone. The method and its options are one
#    "reconstruction" dict, {"method": ..., "phase_damp": ..., ...}, as ShowPtycho's Save writes it (a first /8
#    writer here put "reconstruction_method" beside "reconstruction_options"; scripts/rewrite_saved_records.py
#    folds those into "reconstruction").
# 7: the output factor is "upsampling" (was "upsampling_factor") and reconstruction_options names the direct
#    strengths phase_damping and amplitude_damping (were phase_regularization and amplitude_regularization).
# An earlier record is rewritten once with quantem.widget's scripts/rewrite_direct_calibration.py.
CALIBRATION_SCHEMA = "quantem.direct-calibration/10"
# Every calibration record a reader accepts, the current one first.
READABLE_CALIBRATION_SCHEMAS = (CALIBRATION_SCHEMA,)


def capture_calibration(session, result):
    """Snapshot small acquisition metadata, without hashing files or retaining GPU state.

    A scan held in GPU memory (no file yet, ``source_path`` None) records its calibration too: the record then names
    no source file (2026-11-16, a fit on ``spectra.arina.latest()`` could not save the calibration streaming reuses).
    """
    if result.calibration_context is not None or session.source_kind == "array":
        return
    shape = session.input_scan_shape
    prepared_shape = tuple(int(n) // result.upsample for n in result.object_wave.shape[-2:])
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
    result._calibration_source = None if session.source_path is None else {
        "source_path": session.source_path, "scan_shape": shape,
        "hot_pixel_correction": deepcopy((session.source_provenance or {}).get("hot_pixel_correction")),
        "storage_path": session.source_storage_path, "calibration_path": session.calibration_path,
        "manifest_path": session.source_manifest_path,
    }


def save_calibration(result, path):
    """Write a calibration snapshot, with no reconstruction or objective evaluation."""
    from quantem.gpu.direct_ptycho.persistence import (
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
    if method not in ("phase_of_mean", "phase", "complex"):
        raise ValueError(f"Unsupported direct reconstruction method {method!r}; save a direct ptychography, phase or complex result.")
    context = deepcopy(result.calibration_context)
    identity = context["session_identity"]
    if "source" not in identity:
        if result._calibration_source is None and result.source_path is not None:
            raise ValueError("This saved fit has no acquisition fingerprint. Reopen its acquisition and reload the fit.")
        # a scan in GPU memory has no file to fingerprint: the record names no source and loads onto any scan its shape
        identity["source"] = (None if result._calibration_source is None
                              else source_identity(**result._calibration_source))
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
    factor = (direct_auto_upsample(scan_sampling_A=geometry["scan_sampling_A"],
                                   semiangle_mrad=geometry["semiangle_mrad"], voltage_kV=geometry["voltage_kV"])
              if method == "phase_of_mean" else result.upsample)
    record = {
        "schema": CALIBRATION_SCHEMA, "id": uuid4().hex[:12], "generated_at_utc": datetime.now(UTC).isoformat(),
        "notes": "", "aberrations": dict(result.aberrations),
        "scan_detector_rotation_deg": result.scan_detector_rotation_deg,
        "scan_detector_rotation_flipped": result.scan_detector_rotation_flipped,
        **{key: geometry[key] for key in ("voltage_kV", "semiangle_mrad", "scan_sampling_A", "detector_sampling_mrad")},
        "source": portable_source(result.source_path, path.parent), "scan_region": context["scan_region"],
        "scan_shape": context["scan_shape"], "bright_field_radius_px": geometry["bright_field_radius_px"],
        "sample_tilt_mrad": sample, "depth_spread_nm": depth,
        "tilt_mrad": (None if sample is None else sample_to_propagator_tilt(
            sample, scan_detector_rotation_deg=result.scan_detector_rotation_deg,
            scan_detector_rotation_flipped=result.scan_detector_rotation_flipped)),
        "loss": result.loss,
        "max_order": result.optimizer.get("max_order"), "higher_order_limit": result.optimizer.get("higher_order_limit"),
        "reconstruction": {**deepcopy(result.reconstruction_options), "method": method},
        "reconstruction_diagnostics": deepcopy(result.reconstruction_diagnostics),
        "upsample": factor, "upsample_mode": "auto" if method == "phase_of_mean" else result.upsample_mode,
        "display_negate_phase": False,
        "display": {"tilt_correction": sample is not None, "max_columns": 3},
        "session_identity": identity, "local_tilt": None,
    }
    # Finite JSON only; do not silently alter coefficients or emit NaN/Infinity.
    encoded = json.dumps(json_value(record), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(path, "w", lambda file: file.write(encoded), encoding="utf-8")
    return path


def _highest_order(aberrations) -> int:
    """The highest aberration order with a non-zero term in ``aberrations`` (1: C10 and C12 only)."""
    from .higher_order import ORDERS

    return max([n for n, m in ORDERS if n >= 2 and float(aberrations.get(f"C{n}{m}_nm", 0.0)) != 0.0], default=1)


def load_calibration(path):
    """The fit a saved calibration describes, with no scan and no fitting.

    The record is the model: its aberrations, sample tilt, depth spread and rotation are taken as written, by
    ``save_calibration`` or by ShowPtycho after its sliders were moved. The fit coordinates a series refits from
    are computed from them (``fifth_order_parameters.coordinates``), so they cannot disagree with the aberrations,
    and the search bounds are those of a fit with the record's ``max_order`` and ``higher_order_limit`` (a record
    without them: the highest order it holds, and the default limit). The result has no image
    (``object_wave`` is None); ``series(fit=...)`` and ``run(fit)`` use it as they use the fit that was saved.
    """
    from quantem.gpu.device import resolve
    from quantem.gpu.direct_ptycho.fifth_order_fit import DEFAULT_HIGHER_ORDER_LIMIT, fifth_order_bounds
    from quantem.gpu.direct_ptycho.fifth_order_parameters import coordinates
    from quantem.gpu.direct_ptycho.results import DirectPtychoResult
    from quantem.gpu.direct_ptycho.units import validate_aberrations
    from quantem.gpu.optics.physics import wavelength_A_from_kV

    path = Path(path).expanduser()
    record = json.loads(path.read_text(encoding="utf-8"))
    found = record.get("schema") if isinstance(record, dict) else None
    if found not in READABLE_CALIBRATION_SCHEMAS:
        raise ValueError(f"{path} is a {found!r} record; a Direct ptycho calibration is {CALIBRATION_SCHEMA!r}: an "
                         "earlier layout is rewritten once with quantem.widget's scripts/rewrite_direct_calibration.py.")
    rows, columns = record["scan_shape"]
    if record["scan_region"] != [0, rows, 0, columns]:
        raise ValueError(f"{path} is the calibration of scan region {record['scan_region']}; a series needs the "
                         "calibration of a whole scan.")
    aberrations = validate_aberrations(record["aberrations"], higher_order=True)
    tilt, depth = record["sample_tilt_mrad"], record["depth_spread_nm"]
    if (tilt is None) != (depth is None):
        raise ValueError(f"{path}: sample tilt and depth spread must describe one complete model.")
    tilt = None if tilt is None else (float(tilt[0]), float(tilt[1]))
    max_order = record.get("max_order") or _highest_order(aberrations)
    limit = record.get("higher_order_limit") or DEFAULT_HIGHER_ORDER_LIMIT
    reconstruction = dict(record["reconstruction"])
    method = reconstruction.pop("method")
    identity = record.get("session_identity") or {"geometry": {
        key: record[key] for key in ("voltage_kV", "semiangle_mrad", "scan_sampling_A", "detector_sampling_mrad",
                                     "bright_field_radius_px")}}
    geometry = identity["geometry"]
    center = geometry.get("bright_field_center_px")
    fit = DirectPtychoResult(
        object_wave=None, backend=resolve("auto"), aberrations=aberrations, sample_tilt_mrad=tilt,
        depth_spread_nm=None if depth is None else float(depth),
        scan_detector_rotation_deg=record["scan_detector_rotation_deg"],
        scan_detector_rotation_flipped=bool(record["scan_detector_rotation_flipped"]), loss=record["loss"],
        optimizer={
            "model": "broad_fifth_v1", "max_order": int(max_order), "higher_order_limit": float(limit),
            "bounds": fifth_order_bounds(limit, tilt=depth is not None, max_order=int(max_order)).tolist(),
            "coordinates": coordinates(aberrations, tilt, depth, wavelength_A_from_kV(float(record["voltage_kV"])),
                                       float(record["semiangle_mrad"]) * 1e-3).tolist(),
            "stop": "loaded from a saved calibration",
        },
        num_bright_field_pixels=geometry.get("num_bright_field_pixels"), voltage_kV=record["voltage_kV"],
        semiangle_mrad=record["semiangle_mrad"], upsample=int(record["upsample"]),
        upsample_mode=record["upsample_mode"], reconstruction_method=method, reconstruction_options=reconstruction,
        reconstruction_diagnostics=deepcopy(record.get("reconstruction_diagnostics") or {}),
        bright_field_center_px=None if center is None else tuple(center),
        bright_field_radius_px=record["bright_field_radius_px"], saved_path=path,
        metadata={"calibration": {key: record.get(key) for key in ("schema", "id", "generated_at_utc", "notes")}},
        calibration_context={"scan_shape": [rows, columns], "scan_region": list(record["scan_region"]),
                             "session_identity": deepcopy(identity)},
    )
    if method == "phase":
        # as run(fit, method="phase", ...) leaves them on its fit: the phase settings a series of this fit starts from
        fit._phase_settings = {key: reconstruction[key] for key in (
            "phase_damp", "smoothness_inv_A", "detector_weighting") if reconstruction.get(key) is not None}
    return fit
