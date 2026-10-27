"""The three reconstructions of a measured acquisition on MPS against CUDA references of the same optics.

The fixture lives outside the repository; without it this test skips. It is the native
``SSBDirectMeasuredParityTests`` format. ``SSB_DIRECT_FIXTURE`` names a folder with ``case.json`` (calibration, scan
and detector shape, selected detector pixels, count type), ``counts.bin`` (selected BF counts, plane-major
``[pixel, scan row, scan column]``, after the loader's flagged-pixel median fill), ``detector_sum.u64``,
``optics.json`` (``optics_sets``: name, public ``aberrations``, ``sample_tilt_mrad``, ``depth_spread_nm``,
``rotation_deg``) and ``references.json`` (``references``: file, ``amplitude_file``, method, ``upsampling``,
``factor`` "1", "2" or "auto", setting, ``requested`` regularization, optics) with the CUDA ``.npy`` arrays.

The default presets run unless ``SSB_DIRECT_ALL=1``; ``SSB_DIRECT_OPTICS`` picks one optics set. Gates per
reconstruction: phase correlation >= 0.99999, rms difference <= 1e-4 rad and <= 1 % of the reference's phase std;
complex amplitude rms <= 1e-4 of its mean. ``SSB_DIRECT_GATES`` may name a JSON of per-image limits,
``{"<handle>/<optics>/<file stem>": {"rms_max": ..., "max_abs_max": ...}}`` (2x and 10x the CUDA reference's own
float32 error against a float64 evaluation), which the regularized methods must then also meet.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from .test_ssb_thick_sample_mps import _require_mps

pytestmark = pytest.mark.slow

# CUDA reference records name the strengths by their earlier keywords


def _session(root: Path, rotation_deg: float):
    from quantem.gpu import DirectPtycho
    from quantem.gpu.direct_ptycho.brightfield import BrightfieldDisk
    from quantem.gpu.direct_ptycho.mps.frames import MpsBfColumnFrames

    case = json.loads((root / "case.json").read_text())
    calibration = case["calibration"]
    rows, cols = int(case["detectorRows"]), int(case["detectorColumns"])
    pixels = np.asarray(case["pixels"], np.int64)
    sampling = np.broadcast_to(np.asarray(calibration["detector_sampling_mrad"], float), (2,))
    selection = BrightfieldDisk(
        rows=pixels // cols, cols=pixels % cols, bright_field_center_px=tuple(calibration["bright_field_center_px"]),
        bright_field_radius_px=float(calibration["bright_field_radius_px"]),
        bright_field_edge_radius_px=float(calibration["semiangle_mrad"]) / float(sampling[0]),
        detector_shape=(rows, cols),
    )
    frames = MpsBfColumnFrames(
        root / "counts.bin", selection=selection, scan_shape=(int(case["scanRows"]), int(case["scanColumns"])),
        dtype={1: np.uint8, 2: np.uint16}[int(case["countType"])],
        detector_sum=np.fromfile(root / "detector_sum.u64", np.uint64).reshape(rows, cols),
    )
    return DirectPtycho(frames, backend="mps", voltage_kV=float(calibration["voltage_kV"]),
                        semiangle_mrad=float(calibration["semiangle_mrad"]),
                        scan_sampling_A=tuple(np.broadcast_to(calibration["scan_sampling_A"], (2,)).tolist()),
                        detector_sampling_mrad=tuple(sampling.tolist()), scan_detector_rotation_deg=rotation_deg)


def _compare(actual, reference, wrap):
    a, b = np.asarray(actual, np.float64).ravel(), np.asarray(reference, np.float64).ravel()
    d = a - b
    if wrap:
        d -= 2 * np.pi * np.round(d / (2 * np.pi))
    return {"corr": float(np.corrcoef(a, b)[0, 1]), "rms": float(np.sqrt(np.mean(d * d))),
            "max": float(np.abs(d).max()), "std": float(b.std()), "mean": float(b.mean())}


def test_three_methods_match_cuda():
    root = os.environ.get("SSB_DIRECT_FIXTURE")
    if root is None:
        pytest.skip("Set SSB_DIRECT_FIXTURE to a measured fixture with CUDA ssb/phase/complex references.")
    _require_mps()
    root = Path(root)
    optics_sets = json.loads((root / "optics.json").read_text())["optics_sets"]
    record = json.loads((root / "references.json").read_text())
    gates = json.loads(Path(os.environ["SSB_DIRECT_GATES"]).read_text()) if os.environ.get("SSB_DIRECT_GATES") else {}
    handle = record.get("handle", root.name)
    every = os.environ.get("SSB_DIRECT_ALL") == "1"
    chosen = os.environ.get("SSB_DIRECT_OPTICS")
    failures, checked, ptycho = [], 0, None
    try:
        for optics in optics_sets:
            if chosen is not None and optics["name"] != chosen:
                continue
            if ptycho is None:
                ptycho = _session(root, optics["rotation_deg"])
            else:
                ptycho.set_rotation(optics["rotation_deg"])
            ptycho.sample_tilt_mrad = tuple(optics["sample_tilt_mrad"])
            ptycho.depth_spread_nm = optics["depth_spread_nm"]
            auto = ptycho._resolve_upsampling("auto")
            for entry in record["references"]:
                if entry["optics"] != optics["name"] or not (every or entry["setting"] in (None, "default")):
                    continue
                if entry["factor"] == "auto":
                    assert entry["upsampling"] == auto, "Auto factor differs from CUDA's"
                options = dict(entry.get("requested") or {})
                result = ptycho.reconstruct(None, method=entry["method"], aberrations=dict(optics["aberrations"]),
                                            upsampling=entry["upsampling"], compute_loss=False, **options)
                stem = Path(entry["file"]).name[: -len("-phase.npy")]
                label = f"{optics['name']} {stem}"
                p = _compare(result.phase, np.load(root / entry["file"]), wrap=True)
                problems = []
                if not (p["corr"] >= 0.99999 and p["rms"] <= 1e-4 and p["rms"] <= 0.01 * p["std"]):
                    problems.append(f"phase corr {p['corr']} rms {p['rms']:.3g} (std {p['std']:.3g})")
                gate = gates.get(f"{handle}/{optics['name']}/{stem}-phase")
                if gate and entry["method"] != "ssb" and not (p["rms"] <= gate["rms_max"]
                                                              and p["max"] <= gate["max_abs_max"]):
                    problems.append(f"phase rms {p['rms']:.3g} max {p['max']:.3g} beyond {gate}")
                if entry.get("amplitude_file"):
                    assert result.amplitude_estimated, label
                    q = _compare(result.amplitude, np.load(root / entry["amplitude_file"]), wrap=False)
                    if q["rms"] > 1e-4 * q["mean"]:
                        problems.append(f"amplitude rms {q['rms']:.3g} (mean {q['mean']:.6g})")
                    gate = gates.get(f"{handle}/{optics['name']}/{stem}-amplitude")
                    if gate and not (q["rms"] / q["mean"] <= gate["rms_max"] and q["max"] <= gate["max_abs_max"]):
                        problems.append(f"amplitude rms/mean {q['rms'] / q['mean']:.3g} max {q['max']:.3g} beyond {gate}")
                else:
                    assert not result.amplitude_estimated, label
                print(f"DIRECT_PARITY {label}: phase corr {p['corr']:.9f} rms {p['rms']:.3g} max {p['max']:.3g}"
                      + ("" if not problems else " FAIL " + "; ".join(problems)))
                failures += [f"{label}: {problem}" for problem in problems]
                checked += 1
    finally:
        if ptycho is not None:
            ptycho.close()
    assert checked > 0, "no reference matched"
    assert not failures, "\n".join(failures)

