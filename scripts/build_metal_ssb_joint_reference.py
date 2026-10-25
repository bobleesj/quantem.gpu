"""Generate independent synthetic fixtures for native joint SSB validation.

Run with the installed QuantEM MPS implementation. Writes the geometry, the
selected BF counts and the agreement objective of three cases, which
``MetalSSBJointTests`` reads. Outputs are disposable test data, never
acquisitions.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from quantem.gpu.ssb.brightfield import BrightfieldDisk
from quantem.gpu.ssb.mps.frames import ArrayFrames
from quantem.gpu.ssb.mps.prepared import prepare_selection
from quantem.gpu.ssb.mps.thick_sample import thick_fit_batch


def _floats(array: object) -> list[float]:
    return np.asarray(array, dtype=np.float32).ravel().tolist()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    n = 128
    random = np.random.default_rng(42)
    scan_row, scan_col = np.indices((n, n))
    detector_row, detector_col = np.indices((13, 13))
    rows, cols = np.nonzero((detector_row - 6) ** 2 + (detector_col - 6) ** 2 <= 36)
    counts = random.poisson(18, size=(n, n, 13, 13)).astype(np.uint16)
    for row, col in zip(rows, cols, strict=True):
        argument = 2 * np.pi * (scan_col * (row + 1) + scan_row * (col + 1)) / n
        counts[:, :, row, col] += np.round(
            8 + 6 * np.cos(argument + 0.2 * row)
        ).astype(np.uint16)
    selection = BrightfieldDisk(rows, cols, (6, 6), 6, 6, (13, 13))
    prepared = prepare_selection(
        ArrayFrames(counts), scan_shape=(n, n), selection=selection,
        voltage_kV=300, semiangle_mrad=30, scan_sampling=(0.4, 0.4),
        detector_sampling_mrad=(5, 5), scan_detector_rotation_deg=0, chunk_bf=32,
    )
    geometry = {
        "brightfieldKX": _floats(prepared.kx),
        "brightfieldKY": _floats(prepared.ky),
        "brightfieldAlphaSquared": _floats(prepared.alpha_k2_1d),
        "brightfieldAperture": _floats(prepared.aperture_k_1d),
        "brightfieldCos2Phi": _floats(prepared.cos2_k_1d),
        "brightfieldSin2Phi": _floats(prepared.sin2_k_1d),
        "qxByRow": _floats(prepared.q_row),
        "qyByColumn": _floats(prepared.q_col),
        "wavelengthAngstroms": prepared.wavelength,
        "phaseFactor": float(np.float32(np.pi / prepared.wavelength)),
        "semiangleRadians": prepared.semiangle_rad,
        "angularSamplingYRadians": prepared.ang_y_rad,
        "angularSamplingXRadians": prepared.ang_x_rad,
        "dcValue": [float(prepared.dc_value.real), float(prepared.dc_value.imag)],
        "referenceRotationDegrees": 0,
    }
    (output / "geometry.json").write_text(json.dumps(geometry))
    np.ascontiguousarray(counts[:, :, rows, cols].transpose(2, 0, 1)).tofile(
        output / "counts.u16"
    )
    cases = []
    for name, c10, tilt, depth in (
        ("thin", 42, (0, 0), 0),
        ("tilted", 42, (8, -3), 240),
        ("negative", -42, (-8, 3), 240),
    ):
        loss = -float(thick_fit_batch(prepared, [[c10, 12, 0.3, *tilt, depth]])[0])
        cases.append({
            "name": name,
            "parameters": {
                "aberrations": {
                    "c10Angstroms": c10, "c12Angstroms": 12, "phi12Radians": 0.3,
                },
                "depthAngstroms": depth,
                "tiltRowMrad": tilt[0], "tiltColumnMrad": tilt[1],
            },
            "loss": loss,
        })
    (output / "cases.json").write_text(json.dumps(cases, indent=2))
    print(f"Reference fixtures written to {output}")


if __name__ == "__main__":
    _main()
