"""Export the Python MPS ``complex_wave`` reference for the native Swift SSB parity test.

The scene is the synthetic 128 x 128 scan, 12 x 12 detector of
``SSBModelReconstructionTests.scene()`` (native/swift/Tests/MetalSSBKernelsTests), rebuilt here from the same
formula, so no acquisition is involved. The script writes the complex64 object wave as raw little-endian
interleaved float32 and its settings as JSON next to it.

    PYTHONPATH=src python scripts/export_ssb_complex_wave_fixture.py

Requires Apple silicon with MLX. Regenerate only when the Python ``complex_wave`` definition changes.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

from quantem.gpu.ssb import SSB

FIXTURES = Path(__file__).resolve().parents[1] / "native/swift/Tests/MetalSSBKernelsTests/Fixtures"
DETECTOR, SCAN = 12, 128
CENTER, RADIUS = (5.4, 5.7), 3.6
ROTATION_DEG = 17.25
ABERRATIONS = {"C10_nm": 6.1, "C12_nm": 1.4, "phi12_deg": 40.1}


def scene() -> np.ndarray:
    """Counts ``[scan row, scan col, detector row, detector col]``: a weak lattice with a slow gradient."""
    r = np.arange(SCAN, dtype=np.float64)[:, None]
    c = np.arange(SCAN, dtype=np.float64)[None, :]
    data = np.zeros((SCAN, SCAN, DETECTOR, DETECTOR), np.uint16)
    for pixel in range(DETECTOR * DETECTOR):
        dr = pixel // DETECTOR - CENTER[0]
        dc = pixel % DETECTOR - CENTER[1]
        x = 2 * np.pi * r / 6.4
        y = 2 * np.pi * c / 5.12
        value = (400 + 40 * np.cos(x + 0.3 * dr) * np.cos(y - 0.2 * dc) + 25 * np.sin(x + y + 0.1 * dr * dc)
                 + ((r.astype(int) * 31 + c.astype(int) * 17 + pixel * 7) % 13))
        # Swift's rounded() rounds half away from zero; every value here is positive.
        data[:, :, pixel // DETECTOR, pixel % DETECTOR] = np.floor(value + 0.5).astype(np.uint16)
    return data


def main() -> None:
    ssb = SSB(scene(), backend="mps", voltage_kV=300, semiangle_mrad=30, scan_sampling_A=0.4,
              detector_sampling_mrad=7.5, bright_field_center_px=CENTER, bright_field_radius_px=RADIUS,
              scan_detector_rotation_deg=ROTATION_DEG)
    result = ssb.reconstruct(aberrations=ABERRATIONS, phase_estimator="complex_wave", compute_loss=False)
    wave = np.ascontiguousarray(np.asarray(result.object_wave, dtype=np.complex64))
    FIXTURES.mkdir(parents=True, exist_ok=True)
    wave.view(np.float32).astype("<f4").tofile(FIXTURES / "ssb_complex_wave_mps.f32")
    revision = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    (FIXTURES / "ssb_complex_wave_mps.json").write_text(json.dumps({
        "schema": "quantem.gpu/ssb-complex-wave-fixture/1",
        "scene": "SSBModelReconstructionTests.scene()",
        "backend": result.backend,
        "phase_estimator": result.phase_estimator,
        "shape": list(wave.shape),
        "layout": "row-major complex64 as interleaved little-endian float32 (real, imag)",
        "bright_field_pixels": int(result.num_bright_field_pixels),
        "scan_detector_rotation_deg": ROTATION_DEG,
        "aberrations": ABERRATIONS,
        "python_revision": revision,
    }, indent=2) + "\n")
    print(f"{wave.shape} complex wave, {result.num_bright_field_pixels} BF pixels, "
          f"mean |w| {np.abs(wave).mean():.4f}, phase std {np.angle(wave).std():.3e}")


if __name__ == "__main__":
    main()
