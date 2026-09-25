"""Thick-sample SSB (sample tilt + thickness) on CUDA.

Top: the public workflow recovers a known sample tilt from a simulated tilted crystal and gives no tilt for the untilted
control. Middle: the thick path reduces exactly to standard SSB when the depth weighting is off. Bottom: CuPy's FFT still
loads after torch has loaded a different cuFFT (the import order of every notebook that imports quantem.widget first).
"""

import os
import subprocess
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _require_cuda():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")


def _simulated_crystal(tilt_mrad, thickness_A=152.0):
    """abTEM 4D-STEM of BaTiO3 [001] (a = 4 A) leaning by ``tilt_mrad`` (row, col): every atom at depth z is shifted by z x tilt.

    300 kV, 30 mrad, probe focused at mid-depth, noise-free, 64 x 64 scan at 0.25 A = 4 x 4 unit cells (periodic for the scan
    FFT), potential box 9 x 9 cells so the detector sampling is lambda / 36 A = 0.55 mrad.
    """
    abtem = pytest.importorskip("abtem")
    from ase import Atoms

    abtem.config.set({"device": "gpu"})
    a, cells = 4.0, 9
    layers, box = int(round(thickness_A / a)), cells * a
    basis = [("Ba", (0, 0, 0)), ("Ti", (0.5, 0.5, 0.5)), ("O", (0.5, 0.5, 0)), ("O", (0.5, 0, 0.5)), ("O", (0, 0.5, 0.5))]
    symbols, positions = [], []
    for i in range(cells):
        for j in range(cells):
            for k in range(layers):
                for symbol, (fr, fc, fz) in basis:
                    z = (k + fz) * a
                    symbols.append(symbol)
                    positions.append((((i + fr) * a + z * tilt_mrad[0] * 1e-3) % box, ((j + fc) * a + z * tilt_mrad[1] * 1e-3) % box, z + 0.5))
    atoms = Atoms(symbols, positions=positions, cell=[box, box, layers * a + 1.0], pbc=True)
    potential = abtem.Potential(atoms, sampling=0.08, slice_thickness=a / 2, projection="infinite", parametrization="lobato")
    probe = abtem.Probe(energy=300e3, semiangle_cutoff=30, defocus=layers * a / 2)
    scan = abtem.GridScan(start=(2 * a, 2 * a), end=(6 * a, 6 * a), gpts=(64, 64), endpoint=False)
    measurement = probe.scan(potential, scan=scan, detectors=abtem.PixelatedDetector(max_angle=45)).compute()
    return np.asarray(measurement.array, dtype=np.float32), float(measurement.angular_sampling[0])


def _session(data, det_mrad, scan_A=0.25):
    from quantem.gpu import SSB

    return SSB.from_array(data, backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=scan_A, det_sampling=det_mrad,
                          rotation_angle_deg=0.0)


@pytest.mark.parametrize("tilt_mrad", [(3.0, -4.0), (0.0, 0.0)])
def test_fit_tilt_recovers_known_tilt(tilt_mrad):
    _require_cuda()
    data, det_mrad = _simulated_crystal(tilt_mrad)
    ssb = _session(data, det_mrad)
    assert ssb.supports_tilt
    result = ssb.fit(tilt=True, verbose=False)
    # Validated 2026-09-24: (3, -4) -> (3.0, -4.1); (0, 0) -> (-0.3, -0.1). 1 mrad is the scan/detector sampling limit here.
    assert abs(result.tilt_mrad[0] - tilt_mrad[0]) < 1.0
    assert abs(result.tilt_mrad[1] - tilt_mrad[1]) < 1.0
    assert result.tilt_fit_gain > 1.2      # the thick model explains the thick data better than standard SSB


# ---


def test_zero_depth_weighting_is_standard_ssb():
    """Thickness below the sinc cutoff gives every weight exactly 1: the thick path must reproduce the fused standard SSB."""
    _require_cuda()
    rng = np.random.default_rng(7)
    counts = rng.poisson(40.0, size=(128, 128, 32, 32)).astype(np.uint16)
    rr, cc = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rr, cc) < 10] += 200      # bright-field disk
    ssb = _session(counts, det_mrad=3.0, scan_A=0.3)
    ssb.reconstruct({"C10": 0.0, "C12": 0.0, "phi12": 0.0})
    for aberrations in ({"C10": 30.0, "C12": 5.0, "phi12": 0.3}, {"C10": -40.0, "C12": 12.0, "phi12": -1.0}):
        standard, standard_loss = ssb.preview(aberrations)
        thick, thick_loss = ssb.preview(aberrations, tilt_mrad=(5.0, -3.0), depth_spread_nm=1e-9)
        np.testing.assert_allclose(thick, standard, atol=5e-6)
        assert abs(thick_loss - standard_loss) <= 1e-5 * abs(standard_loss)


def _synthetic_session():
    rng = np.random.default_rng(7)
    counts = rng.poisson(40.0, size=(128, 128, 32, 32)).astype(np.uint16)
    rr, cc = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rr, cc) < 10] += 200
    ssb = _session(counts, det_mrad=3.0, scan_A=0.3)
    ssb.reconstruct({"C10": 0.0, "C12": 0.0, "phi12": 0.0})
    return ssb, ssb._backend_protocol._accelerator


_ENGINE_CASES = [(300.0, 50.0, 0.3, (0.0, 0.0), 0.0), (-400.0, 120.0, -1.0, (5.0, -3.0), 150.0), (100.0, 0.0, 0.0, (-12.0, 8.0), 400.0)]


def test_torch_reference_matches_cuda_engine():
    """ssb/torch_ssb.py is the readable reference: phase, loss and tilt objective equal the CUDA engine's (engine units)."""
    _require_cuda()
    torch = pytest.importorskip("torch")
    from quantem.gpu.ssb.torch_ssb import TorchSSB

    ssb, engine = _synthetic_session()
    reference = TorchSSB.from_ssb(ssb)
    for C10, C12, phi12, tilt, thickness in _ENGINE_CASES:
        if thickness == 0.0:
            phase, loss = engine.reconstruct_with_loss(C10, C12, phi12)
        else:
            phase, loss = engine.reconstruct_thick(C10, C12, phi12, tilt, thickness)
        torch_phase, torch_loss = reference.reconstruct(C10, C12, phi12, tilt, thickness)
        np.testing.assert_allclose(torch_phase.cpu().numpy(), cp.asnumpy(phase), atol=2e-5)
        assert abs(torch_loss - float(loss)) <= 1e-4 * abs(float(loss))
        fit = engine.thick_fit(C10, C12, phi12, tilt, thickness)
        assert abs(reference.fit_power(C10, C12, phi12, tilt, thickness) - fit) <= 1e-5 * abs(fit)
    del torch


def test_batched_fit_kernel_matches_reference_objective():
    """The fused batch kernel (fast path of fit(tilt=True)) equals SSBEngine.thick_fit row by row."""
    _require_cuda()
    _, engine = _synthetic_session()
    rng = np.random.default_rng(3)
    params = np.column_stack([rng.uniform(-300, 300, 12), rng.uniform(0, 200, 12), rng.uniform(-1.5, 1.5, 12),
                              rng.uniform(-20, 20, 12), rng.uniform(-20, 20, 12), rng.choice([0.0, 150.0, 400.0], 12)])
    reference = np.array([engine.thick_fit(p[0], p[1], p[2], (p[3], p[4]), p[5]) for p in params])
    np.testing.assert_allclose(engine.thick_fit_batch(params), reference, rtol=1e-4)
    # the half-plane, band-only objective equals the full-plane sum it replaces
    for C10, C12, phi12, tilt, thickness in _ENGINE_CASES:
        full = engine._thick_fit_full_plane(C10, C12, phi12, tilt, thickness)
        assert abs(engine.thick_fit(C10, C12, phi12, tilt, thickness) - full) <= 1e-5 * abs(full)


def test_cupy_fft_loads_after_torch_cufft():
    """torch (conda CUDA 13) loading its cuFFT first must not stop CuPy (pip CUDA 12) from loading its own after quantem.gpu."""
    _require_cuda()
    pytest.importorskip("torch")
    code = (
        "import torch; torch.fft.fft(torch.ones(4, device='cuda'))\n"
        "import quantem.gpu\n"
        "import cupy as cp; print(complex(cp.fft.fft2(cp.ones((4, 4), cp.complex64)).sum()))\n"
    )
    environment = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("(16+0j)")
