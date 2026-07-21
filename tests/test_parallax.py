from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


def _shift_image_np(image: np.ndarray, shift_row: float, shift_col: float) -> np.ndarray:
    n_row, n_col = image.shape
    f_row = np.fft.fftfreq(n_row).reshape(-1, 1)
    f_col = np.fft.fftfreq(n_col).reshape(1, -1)
    phase = np.exp(-2j * np.pi * (f_row * shift_row + f_col * shift_col))
    return np.fft.ifft2(np.fft.fft2(image) * phase).real.astype(np.float32)


def _synthetic_parallax_cube(scan_shape=(12, 12), det_shape=(16, 16)):
    scan_r, scan_c = scan_shape
    det_r, det_c = det_shape
    rng = np.random.default_rng(11)
    base = rng.standard_normal(scan_shape).astype(np.float32)
    base += np.linspace(-1.0, 1.0, scan_c, dtype=np.float32)[None, :]
    base += np.linspace(-0.5, 0.5, scan_r, dtype=np.float32)[:, None]
    center = (det_r // 2, det_c // 2)
    radius = 2
    rows = np.arange(det_r)[:, None]
    cols = np.arange(det_c)[None, :]
    mask = (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius**2
    data = np.zeros((scan_r, scan_c, det_r, det_c), dtype=np.float32)
    for row, col in np.argwhere(mask):
        shift_row = 0.15 * (row - center[0])
        shift_col = -0.12 * (col - center[1])
        data[:, :, row, col] = _shift_image_np(base, shift_row, shift_col)
    data += 0.01 * rng.standard_normal(data.shape).astype(np.float32)
    return data, center, radius


def test_parallax_public_names_are_lazy() -> None:
    import quantem.gpu as qg

    assert "parallax" in qg.__all__
    assert "Parallax" in qg.__all__
    assert "ParallaxResult" in qg.__all__


def test_parallax_synthetic_end_to_end() -> None:
    cp = pytest.importorskip("cupy")
    from quantem.gpu import ParallaxResult, parallax

    data_np, center, radius = _synthetic_parallax_cube()
    data = cp.asarray(data_np)

    result = parallax(
        data,
        scan_shape=data_np.shape[:2],
        center=center,
        bf_radius=radius,
        sampling_radius=radius,
        upsampling_factor=1,
        verbose=False,
    )

    assert isinstance(result, ParallaxResult)
    assert tuple(int(v) for v in result.image.shape) == data_np.shape[:2]
    assert tuple(int(v) for v in result.density.shape) == data_np.shape[:2]
    assert len(result.shifts) == int(np.count_nonzero(
        (np.arange(data_np.shape[2])[:, None] - center[0]) ** 2
        + (np.arange(data_np.shape[3])[None, :] - center[1]) ** 2
        <= radius**2
    ))
    assert result.bf_images is None
    assert np.isfinite(cp.asnumpy(result.image)).all()
    assert float(cp.std(result.image)) > 0.0


def test_parallax_aberration_fit_recovers_known_coefficients() -> None:
    pytest.importorskip("cupy")
    from quantem.gpu.ssb.optics.aberration_fitting import fit_aberrations_svd_polar
    from quantem.gpu.ssb.optics.physics import wavelength_A_from_kV

    gpts = (16, 16)
    center = (8, 8)
    radius = 3
    rows = np.arange(gpts[0])[:, None]
    cols = np.arange(gpts[1])[None, :]
    bf_mask = (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius**2
    sampling = (0.25, 0.25)
    wavelength = wavelength_A_from_kV(300)

    c10 = 125.0
    c12 = 35.0
    phi12 = 0.37
    c12a = c12 * np.cos(2 * phi12)
    c12b = c12 * np.sin(2 * phi12)
    aberration_matrix = np.array(
        [[c10 + c12a, c12b], [c12b, c10 - c12a]],
        dtype=np.float64,
    )

    kxa = np.fft.fftfreq(gpts[0], sampling[0]).astype(np.float64)
    kya = np.fft.fftfreq(gpts[1], sampling[1]).astype(np.float64)
    kx = np.broadcast_to(kxa[:, None], gpts)[bf_mask]
    ky = np.broadcast_to(kya[None, :], gpts)[bf_mask]
    basis = np.stack([kx, ky], axis=1) * wavelength
    shifts_ang = basis @ aberration_matrix

    got = fit_aberrations_svd_polar(shifts_ang, bf_mask, wavelength, gpts, sampling)

    assert got["C10"] == pytest.approx(c10, abs=1e-8)
    assert got["C12"] == pytest.approx(c12, abs=1e-8)
    assert got["phi12"] == pytest.approx(phi12, abs=1e-8)
    assert got["rotation_angle"] == pytest.approx(0.0, abs=1e-8)


def test_parallax_aberration_fit_end_to_end() -> None:
    cp = pytest.importorskip("cupy")
    from quantem.gpu import parallax

    data_np, center, radius = _synthetic_parallax_cube(scan_shape=(10, 10))
    data = cp.asarray(data_np)

    result = parallax(
        data,
        scan_shape=data_np.shape[:2],
        center=center,
        bf_radius=radius,
        sampling_radius=radius,
        upsampling_factor=1,
        voltage_kV=300,
        scan_sampling=0.5,
        fit_aberrations=True,
        verbose=False,
    )

    assert result.aberrations is not None
    for key in ("C10", "C12", "phi12", "rotation_angle"):
        assert np.isfinite(result.aberrations[key])


def test_parallax_result_converts_aberrations_for_ssb() -> None:
    from quantem.gpu import ParallaxResult

    result = ParallaxResult(
        image=np.zeros((2, 2), dtype=np.float32),
        density=np.ones((2, 2), dtype=np.float32),
        shifts=[],
        aberrations={
            "C10": 125.0,
            "C12": 35.0,
            "phi12": 0.37,
            "rotation_angle": np.deg2rad(-12.0),
        },
    )

    assert result.to_ssb_aberrations() == {
        "C10": 12.5,
        "C12": 3.5,
        "phi12": 0.37,
    }
    assert result.rotation_angle_deg() == pytest.approx(-12.0)
    assert result.to_ssb_kwargs() == {
        "aberrations": {"C10": 12.5, "C12": 3.5, "phi12": 0.37},
        "rotation_angle_deg": pytest.approx(-12.0),
    }


def test_parallax_result_rejects_missing_ssb_conversion_inputs() -> None:
    from quantem.gpu import ParallaxResult

    result = ParallaxResult(
        image=np.zeros((2, 2), dtype=np.float32),
        density=np.ones((2, 2), dtype=np.float32),
        shifts=[],
        aberrations={"C10": 10.0, "phi12": 0.0},
    )

    with pytest.raises(ValueError, match="Missing"):
        result.to_ssb_aberrations()


def test_parallax_real_env_crop_recovers_aberrations_when_available() -> None:
    cp = pytest.importorskip("cupy")
    from quantem.gpu import parallax
    from quantem.gpu.detector import detect_bf_radius
    from quantem.gpu.io.hdf5 import load_scan_region

    master_env = "QUANTEM_GPU_PARALLAX_MASTER"
    master_raw = os.environ.get(master_env)
    if not master_raw:
        pytest.skip(f"{master_env} is not set.")
    master = Path(master_raw).expanduser()
    if not master.exists():
        pytest.skip(f"{master_env} does not point to an existing file.")

    scan_shape = (48, 48)
    loaded = load_scan_region(
        str(master),
        (232, 280, 232, 280),
        scan_shape=(512, 512),
        det_bin=1,
        verbose=False,
        output_dtype=np.float32,
    )
    data = loaded.data
    center, radius = detect_bf_radius(data.mean(axis=(0, 1)))
    center = tuple(int(v) for v in center)
    radius = int(radius)

    got = parallax(
        data,
        scan_shape=scan_shape,
        center=center,
        bf_radius=radius,
        sampling_radius=min(radius, 5),
        upsampling_factor=1,
        voltage_kV=300,
        scan_sampling=0.5,
        fit_aberrations=True,
        verbose=False,
    )

    assert got.aberrations is not None
    assert tuple(int(v) for v in got.image.shape) == scan_shape
    assert tuple(int(v) for v in got.density.shape) == scan_shape
    assert np.isfinite(cp.asnumpy(got.image)).all()
    assert np.isfinite(cp.asnumpy(got.density)).all()
    shifts = np.asarray(got.shifts, dtype=np.float64)
    assert shifts.shape[1] == 2
    assert np.isfinite(shifts).all()
    assert float(np.sqrt(np.max(np.sum(shifts**2, axis=1)))) < 100.0
    for key in ("C10", "C12", "phi12", "rotation_angle"):
        assert np.isfinite(got.aberrations[key])
