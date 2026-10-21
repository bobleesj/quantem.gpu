"""A noise-free weak-phase 4D-STEM simulation with a known probe, for the higher-order SSB hardware tests.

One slice, a periodic random phase (0.05 rad rms) on a 64 x 64 scan at 0.25 A, a 64 x 64 detector: the regime where
the SSB model is exact, so a reconstruction with the probe's own coefficients must come closest to the object and a
fit must return them. The scan covers one period, so ``tile`` copies of it are the same specimen on a larger scan
(CUDA reconstructs 128, 256, 512 and 1024 square scans without padding).
"""

import math
import platform

import numpy as np
import pytest

ORDERS = {"C10": (1, 0), "C12": (1, 2), "C21": (2, 1), "C23": (2, 3), "C30": (3, 0)}
TRUTH = {"C10_nm": -5.0, "C12_nm": 2.0, "phi12_deg": 30.0, "C21_nm": 40.0, "phi21_deg": 60.0,
         "C23_nm": 30.0, "phi23_deg": -20.0, "C30_nm": 3000.0}
SIDE, STEP_A = 64, 0.25


def backend_or_skip() -> str:
    """The session backend of this machine (MPS on Apple silicon, else CUDA), or skip without one."""
    if platform.system() == "Darwin":
        if platform.machine() != "arm64":
            pytest.skip("MPS SSB requires Apple silicon")
        pytest.importorskip("mlx.core")
        return "mps"
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    return "cuda"


def chi(kr, kc, wavelength, aberrations):
    """chi(k) of the public aberration mapping (nm, degrees), polar Krivanek form."""
    alpha = wavelength * np.hypot(kr, kc)
    phi = np.arctan2(kc, kr)
    total = np.zeros_like(alpha)
    for name, (n, m) in ORDERS.items():
        magnitude = 10.0 * aberrations.get(f"{name}_nm", 0.0)
        angle = math.radians(aberrations.get(f"phi{name[1:]}_deg", 0.0))
        total += magnitude * alpha ** (n + 1) / (n + 1) * np.cos(m * (phi - angle))
    return 2 * np.pi / wavelength * total


def simulate(aberrations, tile: int = 1):
    """Intensities (scan row, scan col, detector row, detector col), the object phase and the detector sampling."""
    from quantem.gpu.optics.physics import wavelength_A_from_kV

    wavelength = float(wavelength_A_from_kV(300.0))
    k = np.fft.fftfreq(SIDE, STEP_A)
    kr, kc = np.meshgrid(k, k, indexing="ij")
    alpha = wavelength * np.hypot(kr, kc)
    aperture = np.clip((30e-3 - alpha) / (wavelength * k[1]) + 0.5, 0.0, 1.0)
    probe = aperture * np.exp(-1j * chi(kr, kc, wavelength, aberrations))
    rng = np.random.default_rng(5)
    smooth = np.exp(-0.5 * (kr ** 2 + kc ** 2) / 0.8 ** 2)
    phase = np.fft.ifft2(np.fft.fft2(rng.standard_normal((SIDE, SIDE))) * smooth).real
    phase *= 0.05 / phase.std()
    transmission = np.exp(1j * phase)
    rows, cols = np.meshgrid(np.arange(SIDE) * STEP_A, np.arange(SIDE) * STEP_A, indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    data = np.empty((SIDE * SIDE, SIDE, SIDE), dtype=np.float32)
    for start in range(0, rows.size, 512):
        r0, c0 = rows[start:start + 512, None, None], cols[start:start + 512, None, None]
        wave = np.fft.ifft2(probe[None] * np.exp(-2j * np.pi * (kr * r0 + kc * c0))) * transmission[None]
        intensity = np.abs(np.fft.fftshift(np.fft.fft2(wave), axes=(-2, -1))) ** 2
        data[start:start + 512] = intensity / intensity[0].sum() * 1e6
    data = data.reshape(SIDE, SIDE, SIDE, SIDE)
    if tile > 1:
        data, phase = np.tile(data, (tile, tile, 1, 1)), np.tile(phase, (tile, tile))
    return data, phase, wavelength * k[1] * 1e3


def session(data, detector_mrad, backend, **options):
    from quantem.gpu import SSB

    return SSB(data, backend=backend, voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=STEP_A,
               detector_sampling_mrad=detector_mrad, scan_detector_rotation_deg=0.0,
               bright_field_center_px=(SIDE / 2, SIDE / 2), bright_field_radius_px=30.0 / detector_mrad, **options)


def correlation(image, reference) -> float:
    a = np.asarray(image, np.float64) - np.mean(image)
    b = reference - reference.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
