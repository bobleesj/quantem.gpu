"""Parallax reconstruction on the encoded acquisition io.load returns."""

import math
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pytest


@pytest.mark.slow
def test_parallax_reconstructs_encoded_acquisition_from_its_exact_counts(
    tmp_path, monkeypatch
) -> None:
    """Disk detection and alignment receive exactly what the stored counts give."""
    cp = _cuda()
    from quantem.gpu import io, parallax
    from quantem.gpu.parallax import workflow

    counts, _, _ = _synthetic_parallax_counts()
    received = {}
    detect, align = workflow.detect_bf_radius, workflow.align_vbf_stack_multiscale_cp

    def record_detection(mean_dp):
        received["mean_dp"] = cp.asnumpy(mean_dp)
        received["disk"] = detect(mean_dp)
        return received["disk"]

    def record_alignment(**kwargs):
        received["stack"] = cp.asnumpy(kwargs["vbf_stack"])
        return align(**kwargs)

    monkeypatch.setattr(workflow, "detect_bf_radius", record_detection)
    monkeypatch.setattr(workflow, "align_vbf_stack_multiscale_cp", record_alignment)
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


@pytest.mark.slow
def test_parallax_recovers_known_aberrations_and_rotation(tmp_path) -> None:
    """A wave-optical acquisition with a known probe gives back its C10, C12, phi12 and rotation, in SSB's convention.

    The probe is SSB's model (``aperture * exp(-i chi)``) and the scan axes are turned 20 degrees from the
    detector axes, which SSB describes as ``rotation_angle_deg=-20`` with phi12 in the scan frame (0.4 - 20
    degrees). The fit used to read the detector frequencies off an unshifted FFT grid around a centred disk and
    to scale them with the scan step. Parallax is a geometric-optics estimate of the shifts, so the
    coefficients are held to 1-2 %, not to rounding; the exact algebra is the next test.
    """
    _cuda()
    from quantem.gpu import io, parallax

    counts, scan_sampling_A = _wave_optical_counts(
        c10_nm=20.0, c12_nm=3.0, phi12=0.4, rotation_deg=20.0, det_sampling_mrad=1.0
    )
    with io.load(
        _write_master(tmp_path, counts), backend="cuda",
        scan_shape=counts.shape[:2], verbose=False,
    ) as loaded:
        result = parallax.run(
            loaded, upsampling_factor=1, fit_aberrations=True,
            scan_sampling_A=scan_sampling_A, det_sampling=1.0,
        )

    assert result.aberrations["C10_nm"] == pytest.approx(20.0, rel=0.01)
    assert result.aberrations["C12_nm"] == pytest.approx(3.0, rel=0.02)
    assert math.radians(result.aberrations["phi12_deg"]) == pytest.approx(0.4 - np.radians(20.0), abs=0.01)
    assert result.rotation_angle_deg == pytest.approx(-20.0, abs=0.05)


def test_parallax_fit_recovers_exact_shifts_on_a_centred_disk() -> None:
    """Shifts made from known coefficients come back to rounding, for an off-centre disk on a full detector.

    The model is the one ``fit_aberrations_from_shifts`` documents: scan-frame shifts ``alpha @ A R`` for beam
    angles ``alpha`` measured from the disk centre (92.6, 99.3) of a 192 x 192 detector. SSB's rotation r turns the
    detector frame back by r, so phi12 comes back in the scan frame as ``phi12 + r``.
    """
    from quantem.gpu.parallax.workflow import fit_aberrations_from_shifts

    c10_nm, c12_nm, phi12, rotation_deg = -21.0, 4.0, 0.3, 37.0
    rows, cols = np.indices((192, 192))
    disk = (rows - 92.6) ** 2 + (cols - 99.3) ** 2 <= 30.0**2
    angles_rad = np.stack([rows[disk] - 92.6, cols[disk] - 99.3], axis=1) * 0.55e-3
    aberration_A = 10 * np.array([
        [c10_nm + c12_nm * np.cos(2 * phi12), c12_nm * np.sin(2 * phi12)],
        [c12_nm * np.sin(2 * phi12), c10_nm - c12_nm * np.cos(2 * phi12)],
    ])
    turn = np.radians(-rotation_deg)
    rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])

    aberrations, rotation_angle_deg = fit_aberrations_from_shifts(
        angles_rad @ aberration_A @ rotation, angles_rad
    )

    assert aberrations["C10_nm"] == pytest.approx(c10_nm, rel=1e-12)
    assert aberrations["C12_nm"] == pytest.approx(c12_nm, rel=1e-12)
    assert math.radians(aberrations["phi12_deg"]) == pytest.approx(phi12 + np.radians(rotation_deg), abs=1e-12)
    assert rotation_angle_deg == pytest.approx(rotation_deg, abs=1e-10)


def test_detector_bins_around_the_disk_centre_are_whole() -> None:
    """Every bin inside the disk holds bin_factor x bin_factor pixels on a 256-pixel detector binned by 3.

    The bins follow QuantEM's corner-centred mask: offsets from the disk centre, moved by half the detector. The
    detector's own indices put the wrap (and with 256 = 85 x 3 + 1, a short bin) through the disk centre.
    """
    cp = _cuda()
    from quantem.gpu.parallax.cuda.alignment import bin_mapping

    rows, cols = np.nonzero(np.hypot(*np.indices((256, 256)) - 128.0) <= 20)
    mapping = cp.asnumpy(bin_mapping(cp.asarray((rows - 128) % 256), cp.asarray((cols - 128) % 256), (256, 256), 3))

    inner = [np.hypot(rows[mapping == index] - 128.0, cols[mapping == index] - 128.0).max() < 14
             for index in range(mapping.max() + 1)]
    sizes = np.bincount(mapping)[inner]
    assert len(sizes) > 50 and (sizes == 9).all()


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
            scan_sampling_A=0.5,
            semiangle_mrad=30,
            fit_aberrations=True,
        )

    assert tuple(got.image.shape) == scan_shape
    assert tuple(got.density.shape) == scan_shape
    assert np.isfinite(cp.asnumpy(got.image)).all()
    shifts = np.asarray(got.shifts, dtype=np.float64)
    assert shifts.shape[1] == 2
    assert np.isfinite(shifts).all()
    assert float(np.sqrt(np.max(np.sum(shifts**2, axis=1)))) < 100.0
    assert np.isfinite([*got.aberrations.values(), got.rotation_angle_deg]).all()


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


def _wave_optical_counts(*, c10_nm, c12_nm, phi12, rotation_deg, det_sampling_mrad, scan=128, det=64, step=2):
    """Counts of a weak phase object scanned by SSB's model probe: ``aperture * exp(-i chi)`` at 300 kV, 25 mrad.

    The specimen is a smooth random phase (0.4 rad rms) on a grid of the detector's real-space pixels, four times
    the scanned field so a turned scan never wraps; the scan steps ``step`` pixels along axes turned by
    ``rotation_deg`` from the detector axes. Returns ``(counts, scan_sampling_A)``.
    """
    wavelength = 0.019687489006849
    pixel_A = wavelength / (det * det_sampling_mrad * 1e-3)
    rng = np.random.default_rng(3)
    size = 4 * scan * step
    frequency = np.fft.fftfreq(size, pixel_A)
    blur = np.exp(-(frequency[:, None] ** 2 + frequency[None, :] ** 2) / (2 * 0.35**2))
    phase = np.fft.ifft2(np.fft.fft2(rng.standard_normal((size, size))) * blur).real
    specimen = np.exp(1j * 0.4 * phase / phase.std())
    angle_row, angle_col = np.meshgrid(*2 * [np.fft.fftfreq(det, pixel_A) * wavelength], indexing="ij")
    azimuth = np.arctan2(angle_col, angle_row)
    chi = np.pi / wavelength * (angle_row**2 + angle_col**2) * 10 * (c10_nm + c12_nm * np.cos(2 * (azimuth - phi12)))
    probe = (np.hypot(angle_row, angle_col) < 25e-3) * np.exp(-1j * chi)
    cycles_row, cycles_col = np.meshgrid(np.fft.fftfreq(det), np.fft.fftfreq(det), indexing="ij")
    window = np.arange(det)
    turn = np.radians(rotation_deg)
    counts = np.empty((scan, scan, det, det), np.uint16)
    for scan_row in range(scan):
        for scan_col in range(scan):
            row = step * (np.cos(turn) * scan_row - np.sin(turn) * scan_col)
            col = step * (np.sin(turn) * scan_row + np.cos(turn) * scan_col)
            row0, col0 = int(np.floor(row)), int(np.floor(col))
            # the sub-pixel part of the probe position as a Fourier phase ramp
            wave = np.fft.fftshift(np.fft.ifft2(
                probe * np.exp(-2j * np.pi * (cycles_row * (row - row0) + cycles_col * (col - col0)))
            ))
            exit_wave = wave * specimen[np.ix_((row0 + window) % size, (col0 + window) % size)]
            # 2e5 electrons per probe position
            intensity = np.abs(np.fft.fftshift(np.fft.fft2(exit_wave))) ** 2 / (det * det * np.sum(np.abs(wave) ** 2))
            counts[scan_row, scan_col] = rng.poisson(2e5 * intensity)
    return counts, pixel_A * step


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
