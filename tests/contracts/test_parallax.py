"""Parallax reconstruction on the encoded acquisition io.load returns."""

import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pytest


def test_parallax_public_names_are_lazy() -> None:
    import quantem.gpu as qg

    assert "parallax" in qg.__all__
    assert qg.parallax.__all__ == ["ParallaxResult", "run"]
    assert "Parallax" not in qg.__all__
    assert "ParallaxResult" not in qg.__all__


@pytest.mark.slow
def test_parallax_reconstructs_encoded_acquisition_from_its_exact_counts(
    tmp_path, monkeypatch
) -> None:
    """Disk detection and alignment receive exactly what the stored counts give."""
    cp = _cuda()
    from quantem.gpu import io, parallax
    from quantem.gpu.parallax.cuda import reconstruction

    counts, _, _ = _synthetic_parallax_counts()
    received = {}
    detect, align = reconstruction.detect_bf_radius, reconstruction.align_vbf_stack_multiscale_cp

    def record_detection(mean_dp):
        received["mean_dp"] = cp.asnumpy(mean_dp)
        received["disk"] = detect(mean_dp)
        return received["disk"]

    def record_alignment(**kwargs):
        received["stack"] = cp.asnumpy(kwargs["vbf_stack"])
        return align(**kwargs)

    monkeypatch.setattr(reconstruction, "detect_bf_radius", record_detection)
    monkeypatch.setattr(reconstruction, "align_vbf_stack_multiscale_cp", record_alignment)
    with io.load(
        _write_master(tmp_path, counts), backend="cuda",
        scan_shape=counts.shape[:2], verbose=False,
    ) as loaded:
        result = parallax.run(loaded, upsampling_factor=1)

    scans = counts.shape[0] * counts.shape[1]
    exact_mean = (counts.sum(axis=(0, 1), dtype=np.uint64) / scans).astype(np.float32)
    np.testing.assert_array_equal(received["mean_dp"], exact_mean)
    (center_row, center_col), radius = received["disk"]
    rows, cols = np.indices(counts.shape[2:])
    disk = (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius**2
    np.testing.assert_array_equal(
        received["stack"], counts[:, :, disk].transpose(2, 0, 1).astype(np.float32)
    )
    assert isinstance(result, parallax.ParallaxResult)
    assert len(result.shifts) == int(disk.sum())
    assert tuple(result.image.shape) == counts.shape[:2]
    assert np.isfinite(cp.asnumpy(result.image)).all()
    assert float(cp.std(result.image)) > 0.0


def test_parallax_shifts_each_image_once_and_upsamples_on_the_finer_grid(tmp_path) -> None:
    """The image sums the bright-field images, each shifted once by its returned shift.

    Upsampling tiles every image's spectrum and shifts it on the finer grid, as
    QuantEM's DirectPtychography.reconstruct does, so the samples of the
    differently tilted images interleave (tiling the summed spectrum instead
    left every odd row and column zero). The density is the same sum of images
    of ones. Reference: the same sums in NumPy float64.
    """
    cp = _cuda()
    from quantem.gpu import io, parallax

    counts, center, radius = _synthetic_parallax_counts()
    rows, cols = np.indices(counts.shape[2:])
    disk = (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius**2
    stack = counts[:, :, disk].transpose(2, 0, 1).astype(np.float64)
    with io.load(
        _write_master(tmp_path, counts), backend="cuda",
        scan_shape=counts.shape[:2], verbose=False,
    ) as loaded:
        results = {
            factor: parallax.run(loaded, center=center, bf_radius=radius, upsampling_factor=factor)
            for factor in (1, 2)
        }

    for factor, result in results.items():
        shifts = np.asarray(result.shifts)
        expected = _shifted_sum(stack, shifts, factor)
        np.testing.assert_allclose(
            cp.asnumpy(result.image), expected, rtol=0, atol=1e-5 * np.abs(expected).max()
        )
        np.testing.assert_allclose(
            cp.asnumpy(result.density), _shifted_sum(np.ones_like(stack), shifts, factor),
            rtol=0, atol=1e-5 * len(stack),
        )
    np.testing.assert_array_equal(cp.asnumpy(results[1].density), len(stack))
    assert np.abs(cp.asnumpy(results[2].image)[1::2, 1::2]).min() > 0.0


def test_parallax_repeats_bit_identically(tmp_path) -> None:
    """Two runs give identical shifts and images.

    The detector binning summed float32 values with atomic additions, whose
    order varied from run to run and changed the last bits of the shifts.
    """
    cp = _cuda()
    from quantem.gpu import io, parallax

    counts, center, radius = _synthetic_parallax_counts((24, 24), (32, 32), radius=6)
    with io.load(
        _write_master(tmp_path, counts), backend="cuda",
        scan_shape=counts.shape[:2], verbose=False,
    ) as loaded:
        first, second = (
            parallax.run(loaded, center=center, bf_radius=radius, upsampling_factor=2)
            for _ in range(2)
        )

    assert first.shifts == second.shifts
    np.testing.assert_array_equal(cp.asnumpy(first.image), cp.asnumpy(second.image))


def test_parallax_aberration_fit_end_to_end(tmp_path) -> None:
    _cuda()
    from quantem.gpu import io, parallax

    counts, center, radius = _synthetic_parallax_counts(scan_shape=(10, 10))
    with io.load(
        _write_master(tmp_path, counts), backend="cuda",
        scan_shape=counts.shape[:2], verbose=False,
    ) as loaded:
        result = parallax.run(
            loaded,
            center=center,
            bf_radius=radius,
            upsampling_factor=1,
            voltage_kV=300,
            scan_sampling=0.5,
            fit_aberrations=True,
        )

    assert result.aberrations is not None
    for key in ("C10", "C12", "phi12", "rotation_angle"):
        assert np.isfinite(result.aberrations[key])


def test_parallax_rejects_data_it_cannot_reconstruct() -> None:
    from quantem.gpu import parallax
    from quantem.gpu.io.dataset import Dataset4dstemGPU

    counts = np.zeros((2, 2, 8, 8), np.uint16)
    with pytest.raises(TypeError, match="io.load"):
        parallax.run(counts)
    with pytest.raises(ValueError, match="CUDA GPU"):
        parallax.run(Dataset4dstemGPU(counts, {}))


def test_parallax_reports_that_apple_gpus_are_not_implemented(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("An Apple GPU (MPS) is required.")
    from quantem.gpu import io, parallax

    counts, center, radius = _synthetic_parallax_counts()
    with (
        io.load(
            _write_master(tmp_path, counts), backend="mps",
            scan_shape=counts.shape[:2], verbose=False,
        ) as loaded,
        pytest.raises(NotImplementedError, match=r"Apple GPU \(MPS\)"),
    ):
        parallax.run(loaded, center=center, bf_radius=radius)


def test_parallax_real_acquisition_recovers_aberrations_when_available() -> None:
    cp = _cuda()
    from quantem.gpu import io, parallax

    master_env = "QUANTEM_GPU_PARALLAX_MASTER"
    master_raw = os.environ.get(master_env)
    if not master_raw:
        pytest.skip(f"{master_env} is not set.")
    master = Path(master_raw).expanduser()
    if not master.exists():
        pytest.skip(f"{master_env} does not point to an existing file.")

    with io.load(master, backend="cuda", verbose=False) as loaded:
        scan_shape = loaded.shape[:2]
        got = parallax.run(
            loaded,
            sampling_radius=5,
            upsampling_factor=1,
            voltage_kV=300,
            scan_sampling=0.5,
            fit_aberrations=True,
        )

    assert tuple(got.image.shape) == scan_shape
    assert tuple(got.density.shape) == scan_shape
    assert np.isfinite(cp.asnumpy(got.image)).all()
    shifts = np.asarray(got.shifts, dtype=np.float64)
    assert shifts.shape[1] == 2
    assert np.isfinite(shifts).all()
    assert float(np.sqrt(np.max(np.sum(shifts**2, axis=1)))) < 100.0
    for key in ("C10", "C12", "phi12", "rotation_angle"):
        assert np.isfinite(got.aberrations[key])


# ---------------------------------------------------------------------------
# Aberration fitting and result conversion
# ---------------------------------------------------------------------------


def test_parallax_aberration_fit_recovers_known_coefficients() -> None:
    pytest.importorskip("cupy")
    from quantem.gpu.optics.cuda import fit_aberrations_svd_polar
    from quantem.gpu.optics.physics import wavelength_A_from_kV

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


def test_parallax_result_converts_aberrations_for_ssb() -> None:
    from quantem.gpu.parallax import ParallaxResult

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


def test_parallax_result_rejects_missing_ssb_conversion_inputs() -> None:
    from quantem.gpu.parallax import ParallaxResult

    result = ParallaxResult(
        image=np.zeros((2, 2), dtype=np.float32),
        density=np.ones((2, 2), dtype=np.float32),
        shifts=[],
        aberrations={"C10": 10.0, "phi12": 0.0},
    )

    with pytest.raises(ValueError, match="Missing"):
        result.to_ssb_aberrations()


# ---------------------------------------------------------------------------
# Synthetic acquisitions
# ---------------------------------------------------------------------------


def _cuda():
    """Return CuPy, skipping where no CUDA device can run parallax."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("A CUDA device is required for parallax.")
    return cp


def _shifted_sum(stack, shifts, factor):
    """Float64 sum of the images, each tiled ``factor`` times in frequency and shifted once on that grid."""
    rows, cols = stack.shape[1:]
    spectrum = np.zeros((factor * rows, factor * cols), np.complex128)
    for image, (shift_row, shift_col) in zip(stack, shifts):
        ramp_row = np.exp(-2j * np.pi * shift_row * np.fft.fftfreq(factor * rows, d=1.0 / factor))
        ramp_col = np.exp(-2j * np.pi * shift_col * np.fft.fftfreq(factor * cols, d=1.0 / factor))
        spectrum += ramp_row[:, None] * np.tile(np.fft.fft2(image), (factor, factor)) * ramp_col[None, :]
    return np.fft.ifft2(spectrum).real


def _synthetic_parallax_counts(scan_shape=(12, 12), detector_shape=(16, 16), radius=2):
    """Counts whose bright-field images are one scene shifted in proportion to the tilt.

    Returns ``(counts, center, radius)`` of a uint16 acquisition with a
    bright disk of ``radius`` pixels on a dim background, as a parallax
    measurement of defocus records it.
    """
    rng = np.random.default_rng(11)
    scene = rng.standard_normal(scan_shape)
    scene += np.linspace(-1.0, 1.0, scan_shape[1])[None, :]
    scene += np.linspace(-0.5, 0.5, scan_shape[0])[:, None]
    center = (detector_shape[0] // 2, detector_shape[1] // 2)
    row_frequency = np.fft.fftfreq(scan_shape[0])[:, None]
    col_frequency = np.fft.fftfreq(scan_shape[1])[None, :]
    expected = np.full((*scan_shape, *detector_shape), 2.0)
    rows, cols = np.indices(detector_shape)
    disk = (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius**2
    for row, col in np.argwhere(disk):
        shift_row, shift_col = 0.15 * (row - center[0]), -0.12 * (col - center[1])
        phase = np.exp(-2j * np.pi * (row_frequency * shift_row + col_frequency * shift_col))
        expected[:, :, row, col] = 200.0 + 40.0 * np.fft.ifft2(np.fft.fft2(scene) * phase).real
    return rng.poisson(expected).astype(np.uint16), center, radius


def _write_master(folder: Path, counts: np.ndarray) -> Path:
    """Write counts as an Arina-style master: bitshuffle/LZ4 frames in scan order."""
    path = folder / "parallax_master.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=counts.reshape(-1, *counts.shape[2:]),
            chunks=(1, *counts.shape[2:]),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        handle["entry/instrument/detector/detectorSpecific/ntrigger"] = (
            counts.shape[0] * counts.shape[1]
        )
    return path
