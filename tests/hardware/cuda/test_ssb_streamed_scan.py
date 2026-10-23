"""Full-scan FFTs from exact encoded counts, with bounded detector batches."""

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.slow

from quantem.gpu import SSB
from quantem.gpu.io.dataset import Dataset4dstemGPU, resident_metadata
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.ssb.cuda.streamed import StreamedSSB
from quantem.gpu.ssb.units import aberrations_to_engine

SETTINGS = {
    "backend": "cuda", "voltage_kV": 300., "semiangle_mrad": 30.,
    "scan_sampling_A": (.3, .35), "detector_sampling_mrad": (6., 7.),
    "bright_field_center_px": (3.15, 3.6), "bright_field_radius_px": 3.1,
    "scan_detector_rotation_deg": 27.,
}
ABERRATIONS = {"C10_nm": 4.25, "C12_nm": 1.5, "phi12_deg": 17.}


def _counts(shape):
    """Nonperiodic, positive counts and an invalid high-count detector pixel."""
    values = cp.random.RandomState(9).poisson(15, size=shape).astype(cp.uint16)
    values[..., 2, 3] = 65535
    return values


def _encoded(values, chunk_scans=1777):
    valid = np.ones(values.shape[-2:], bool)
    valid[2, 3] = False
    source = StreamedCounts(values.shape, values.dtype, valid)
    frames = values.reshape(-1, *values.shape[-2:])
    for first in range(0, frames.shape[0], chunk_scans):
        source.append(cp.ascontiguousarray(frames[first:first + chunk_scans]))
    return Dataset4dstemGPU(source, resident_metadata(values.shape, values.dtype, source.nbytes, backend="cuda"))


@pytest.mark.parametrize("scan_shape, dtype", [((130, 258), cp.uint16), ((128, 128), cp.uint8)])
def test_encoded_fft_columns_preserve_counts(scan_shape, dtype):
    """Selected counts survive irregular chunks and the last detector batch."""
    values = _counts((*scan_shape, 8, 8)).astype(dtype)
    values[..., 0, 0] = np.iinfo(dtype).max
    with _encoded(values) as data:
        source = data.data
        descriptors = cp.asarray([
            [*(array.data.ptr for array in chunk.arrays[:3]), chunk.first, chunk.scans]
            for chunk in source.chunks
        ], dtype=cp.uint64)
        selected = cp.arange(61, dtype=cp.uint32)[::-1].copy()
        scans = math.prod(scan_shape)
        counts = cp.empty(scans * 32, cp.uint16)
        output = cp.empty((32, scan_shape[0], scan_shape[1] + 2), cp.float32)
        errors = cp.zeros(1, cp.uint32)
        for first in range(0, len(selected), 32):
            pixels = selected[first:first + 32]
            streams = math.ceil(max(chunk.scans for chunk in source.chunks) / source.interval) * len(pixels)
            source.kernels["columns_u16"](
                (math.ceil(streams / 128), len(source.chunks)), (128,),
                (descriptors, source.decoding, pixels, counts, errors,
                 *(np.uint32(value) for value in (len(pixels), 64, source.interval))),
            )
            source.kernels["columns_fft"](
                (math.ceil(scans / 32),), (32, 8),
                (counts, output, *(np.uint32(value) for value in (len(pixels), *scan_shape))),
            )
            expected = values.reshape(scans, 64)[:, pixels].T.reshape(len(pixels), *scan_shape)
            cp.testing.assert_array_equal(output[:len(pixels), :, :scan_shape[1]], expected)
        assert int(errors.get()[0]) == 0


@pytest.mark.parametrize("transposed, factor, depth", [(False, 1, 0.), (False, 3, 14.), (True, 2, 14.)])
def test_streamed_phase_matches_stored_spectra(transposed, factor, depth):
    values = _counts((128, 128, 8, 8))
    with _encoded(values) as data:
        streamed = StreamedSSB(
            data, mean_pattern=None, center=SETTINGS["bright_field_center_px"],
            radius=SETTINGS["bright_field_radius_px"], threshold=0., semiangle_mrad=30.,
            detector_sampling_mrad=SETTINGS["detector_sampling_mrad"], scan_detector_rotation_flipped=transposed,
        )
        values[..., 2, 3] = 0  # independent dense reference's scientific validity mask
        with SSB(values, **SETTINGS, scan_detector_rotation_flipped=transposed) as dense:
            expected, _ = dense.preview(ABERRATIONS, compute_loss=False,
                upsampling_factor=factor, sample_tilt_mrad=(1., -2.), depth_spread_nm=depth, output="native")
            phase, loss = streamed.phase(aberrations_to_engine(ABERRATIONS), voltage_kV=300., semiangle_mrad=30.,
                scan_sampling_A=SETTINGS["scan_sampling_A"], rotation_deg=27., upsampling_factor=factor,
                sample_tilt_mrad=(1., -2.), depth_spread_A=depth * 10)
            assert streamed.num_bright_field_pixels == dense.num_bright_field_pixels
            assert loss is None and phase.dtype == cp.float32
            # R2C and C2C forward transforms have different float32 rounding.
            cp.testing.assert_allclose(phase, expected, atol=1e-6, rtol=1e-5)
        assert data.read(scan_region=(0, 1, 0, 1))[0, 0, 2, 3].item() == 65535


def test_public_resident_fit_region_then_complete_1024_scan(tmp_path):
    values = _counts((1024, 1024, 8, 8))
    with _encoded(values, chunk_scans=65536) as data:
        with SSB(data, **SETTINGS, aberrations=ABERRATIONS, scan_detector_rotation_flipped=True) as session:
            assert session._data is data and session._compute is None
            region = (173, 301, 251, 379)
            fitted = session.find_aberrations(search="primary", tilt=False, scan_region=region, num_trials=0,
                refinement=None, check_rotation=False, verbose=False)
            assert fitted.phase.shape == (128, 128)
            assert fitted.fit_scan_region == region
            assert fitted.bright_field_center_px == SETTINGS["bright_field_center_px"]
            with SSB(data, **SETTINGS, aberrations=ABERRATIONS, scan_detector_rotation_flipped=True,
                     scan_region=region) as explicit:
                expected_fit = explicit.find_aberrations(search="primary", tilt=False, num_trials=0, refinement=None,
                    check_rotation=False, verbose=False)
                cp.testing.assert_array_equal(fitted.phase, expected_fit.phase)
            result = session.reconstruct(fitted, compute_loss=False)
            assert result.phase.shape == (1024, 1024) and result.loss is None
            assert result.fit_scan_region == region
            assert result.object_sampling_A == SETTINGS["scan_sampling_A"]
            assert session._compute is None
            # Existing 1024 callers retain diagnostic loss and other estimators.
            with_loss = session.reconstruct(fitted)
            assert math.isfinite(with_loss.loss)
            cp.testing.assert_allclose(result.phase, with_loss.phase, atol=1e-6, rtol=1e-5)
            # A widget export must still contain exactly the selected counts,
            # including when the session was first prepared without a dense crop.
            from quantem.gpu import io

            exported, _ = session.export_brightfield(tmp_path / "brightfield")
            backend = session._backend()
            original_rows = backend.bf_inds_col + session._crop_origin[0]
            original_cols = backend.bf_inds_row + session._crop_origin[1]
            expected_counts = values[..., original_rows, original_cols]
            with io.load(exported, backend="cuda") as reopened:
                actual_counts = cp.from_dlpack(reopened.read())
                cp.testing.assert_array_equal(actual_counts[..., 0, :], expected_counts)
            historical, _ = session.preview(ABERRATIONS, compute_loss=False, phase_estimator="mean_phase")
            assert historical.shape == (1024, 1024) and np.isfinite(historical).all()
            coherent = session.reconstruct(fitted, compute_loss=False, phase_estimator="complex_wave")
            assert coherent.amplitude_estimated and coherent.phase.shape == (1024, 1024)
            # Dense reference is limited to 1024; the public large path matches it.
            values[..., 2, 3] = 0
            with SSB(values, **SETTINGS, scan_detector_rotation_flipped=True) as dense:
                expected = dense.reconstruct(fitted, compute_loss=False)
                cp.testing.assert_allclose(result.phase, expected.phase, atol=1e-6, rtol=1e-5)
        assert not data.data.is_released
        assert data.read(scan_region=(0, 1, 0, 1))[0, 0, 2, 3].item() == 65535


@pytest.mark.parametrize("scan_shape", [(1024, 2048), (4096, 4096)])
def test_global_large_scan_matches_periodic_native_reference(scan_shape):
    """A repeated fixture is a shape/FFT check, not a new measured field of view."""
    tile = _counts((128, 128, 8, 8))
    valid = np.ones((8, 8), bool)
    valid[2, 3] = False
    shape = (*scan_shape, 8, 8)
    source = StreamedCounts(shape, np.uint16, valid)
    # Build independent encoded scan rows without holding the dense large cube.
    for row in range(0, scan_shape[0], 128):
        source.append(cp.tile(tile, (1, scan_shape[1] // 128, 1, 1)).reshape(-1, 8, 8))
    with Dataset4dstemGPU(source, resident_metadata(shape, np.uint16, source.nbytes, backend="cuda")) as data:
        tile[..., 2, 3] = 0
        with SSB(tile, **SETTINGS) as reference:
            expected, _ = reference.preview(ABERRATIONS, compute_loss=False, output="native",
                                             depth_spread_nm=14., sample_tilt_mrad=(1., -2.))
        with SSB(data, **SETTINGS) as session:
            actual, loss = session.preview(ABERRATIONS, compute_loss=False, output="native",
                                            depth_spread_nm=14., sample_tilt_mrad=(1., -2.))
            assert actual.shape == scan_shape and loss is None
            cp.testing.assert_allclose(actual, cp.tile(expected, (scan_shape[0] // 128, scan_shape[1] // 128)),
                                       atol=1e-6, rtol=1e-5)


def test_open_large_file_keeps_encoded_source_and_closes_only_its_owner(tmp_path):
    from quantem.gpu import io

    values = _counts((1024, 1024, 8, 8))
    with _encoded(values, 65536) as original:
        path = tmp_path / "scan.qem"
        io.save(path, original)
        with SSB.open(str(path), **SETTINGS) as session:
            owner = session._owned_dataset
            assert owner is session._data and not owner.data.is_released
            first = session.reconstruct(aberrations=ABERRATIONS, compute_loss=False,
                                        save_dir=tmp_path / "result")
            second = session.reconstruct(aberrations=ABERRATIONS, compute_loss=False,
                                         save_dir=tmp_path / "result")
            assert second.reused
            cp.testing.assert_array_equal(first.phase, second.phase)
        assert owner.data.is_released
        assert not original.data.is_released
