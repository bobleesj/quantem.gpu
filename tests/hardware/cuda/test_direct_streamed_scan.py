"""Full-scan FFTs from exact encoded counts, with bounded detector batches."""

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.slow

from quantem.gpu import DirectPtycho
from quantem.gpu.io.dataset import Dataset4dstem, resident_metadata
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.direct_ptycho.cuda.streamed import StreamedDirect
from quantem.gpu.direct_ptycho.units import ENGINE_PER_NM, aberrations_to_engine, higher_order_arrays, validate_aberrations

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
    return Dataset4dstem(source, resident_metadata(values.shape, values.dtype, source.nbytes, backend="cuda"))


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
        streamed = StreamedDirect(
            data, mean_pattern=None, center=SETTINGS["bright_field_center_px"],
            radius=SETTINGS["bright_field_radius_px"], threshold=0., semiangle_mrad=30.,
            detector_sampling_mrad=SETTINGS["detector_sampling_mrad"], scan_detector_rotation_flipped=transposed,
            full_scan_only=False,
        )
        values[..., 2, 3] = 0  # independent dense reference's scientific validity mask
        with DirectPtycho(values, **SETTINGS, scan_detector_rotation_flipped=transposed) as dense:
            expected, _ = dense.preview(ABERRATIONS, compute_loss=False,
                upsample=factor, sample_tilt_mrad=(1., -2.), depth_spread_nm=depth, output="native")
            phase, loss = streamed.phase(aberrations_to_engine(ABERRATIONS), voltage_kV=300., semiangle_mrad=30.,
                scan_sampling_A=SETTINGS["scan_sampling_A"], rotation_deg=27., upsample=factor,
                sample_tilt_mrad=(1., -2.), depth_spread_A=depth * 10)
            assert streamed.num_bright_field_pixels == dense.num_bright_field_pixels
            assert loss is None and phase.dtype == cp.float32
            # R2C and C2C forward transforms have different float32 rounding.
            cp.testing.assert_allclose(phase, expected, atol=1e-6, rtol=1e-5)
        assert data.read(scan_region=(0, 1, 0, 1))[0, 0, 2, 3].item() == 65535


@pytest.mark.parametrize("transposed, factor", [(False, 1), (True, 2)])
def test_streamed_fifth_order_phase_matches_stored_spectra(transposed, factor):
    """The batched detector columns evaluate the same 14-term chi as the dense spectrum: phases agree in float32."""
    fifth = validate_aberrations({
        **ABERRATIONS, "C21_nm": 40., "phi21_deg": -15., "C23_nm": 25., "phi23_deg": 30., "C30_nm": 1500.,
        "C32_nm": 900., "phi32_deg": 27., "C34_nm": 700., "phi34_deg": -44., "C41_nm": 2e4, "phi41_deg": -50.,
        "C43_nm": 1.5e4, "phi43_deg": 5., "C45_nm": 1e4, "phi45_deg": 35., "C50_nm": -4e5, "C52_nm": 3e5,
        "phi52_deg": 14., "C54_nm": 2e5, "phi54_deg": 1., "C56_nm": 1e5, "phi56_deg": -10.,
    }, higher_order=True)
    magnitudes, angles = higher_order_arrays(fifth)
    values = _counts((128, 128, 8, 8))
    with _encoded(values) as data:
        streamed = StreamedDirect(
            data, mean_pattern=None, center=SETTINGS["bright_field_center_px"],
            radius=SETTINGS["bright_field_radius_px"], threshold=0., semiangle_mrad=30.,
            detector_sampling_mrad=SETTINGS["detector_sampling_mrad"], scan_detector_rotation_flipped=transposed,
            full_scan_only=True,
        )
        values[..., 2, 3] = 0  # independent dense reference's scientific validity mask
        with DirectPtycho(values, **SETTINGS, scan_detector_rotation_flipped=transposed) as dense:
            expected, _ = dense.preview(fifth, compute_loss=False, upsample=factor, output="native")
            options = dict(voltage_kV=300., semiangle_mrad=30., scan_sampling_A=SETTINGS["scan_sampling_A"],
                           rotation_deg=27., upsample=factor, sample_tilt_mrad=(0., 0.), depth_spread_A=0.)
            phase, _ = streamed.phase(aberrations_to_engine(fifth), **options,
                higher_order_magnitudes=magnitudes * np.float32(ENGINE_PER_NM), higher_order_angles=angles)
            low_order, _ = streamed.phase(aberrations_to_engine(ABERRATIONS), **options)
            # the higher orders must change the image, or agreement below would not test them
            assert float(cp.abs(phase - low_order).max()) > 1e-3
            cp.testing.assert_allclose(phase, expected, atol=1e-6, rtol=1e-5)


def test_fit_region_parks_the_owned_acquisition_then_restores_it(tmp_path, monkeypatch):
    """A session on a device too small for the dense path fits a region with the encoded scan parked off the device."""
    from quantem.gpu import io
    from quantem.gpu.direct_ptycho import workflow

    monkeypatch.setattr(workflow, "_exceeds_device", lambda data: True)
    values = _counts((256, 256, 8, 8))
    path = tmp_path / "scan.qem"
    with _encoded(values) as original:
        io.save(path, original)
    options = dict(voltage_kV=300., semiangle_mrad=30., scan_sampling_A=SETTINGS["scan_sampling_A"], rotation_deg=27.,
                   upsample=1, sample_tilt_mrad=(0., 0.), depth_spread_A=0.)
    region = (64, 192, 64, 192)
    with DirectPtycho.open(str(path), **SETTINGS) as session:
        streamed = session._streamed
        assert streamed.full_scan_only and session._reclaimable_bytes() == session._data.data.nbytes - (
            session._data.data.valid.nbytes + session._data.data.detector_total_device().nbytes)
        before, _ = streamed.phase(aberrations_to_engine(ABERRATIONS), **options)
        with session.select_region(region) as regional:
            expected, _ = regional.preview(ABERRATIONS, compute_loss=False, output="native")
        with session._region_for_fit(region) as regional:
            assert streamed.source.chunks == [] and streamed.chunks == ()
            with pytest.raises(ValueError, match="parked"):
                session._data.read(scan_region=(0, 1, 0, 1))
            assert regional.scan_shape == (128, 128)
            phase, _ = regional.preview(ABERRATIONS, compute_loss=False, output="native")
            cp.testing.assert_array_equal(phase, expected)
        assert streamed.source._parked is None
        after, _ = streamed.phase(aberrations_to_engine(ABERRATIONS), **options)
        cp.testing.assert_array_equal(after, before)


# ---


def test_parked_counts_return_the_same_counts_and_refuse_reads_meanwhile():
    """An encoded acquisition parked in host memory decodes to the same counts once restored, in part or whole."""
    values = _counts((128, 128, 8, 8))
    frames = values.reshape(-1, 8, 8)
    with _encoded(values) as data:
        source = data.data
        device_bytes = source.nbytes
        source.park()
        assert source.chunks == []
        with pytest.raises(ValueError, match="parked"):
            source.decode_scan_range_device(0, 128)
        source.restore(4000, 6000)
        assert [chunk.first for chunk in source.chunks] == [3554, 5331]
        cp.testing.assert_array_equal(source.decode_scan_range_device(4000, 6000), frames[4000:6000])
        with pytest.raises(ValueError, match="parked"):
            source.decode_scan_range_device(3000, 4100)
        source.park()
        assert source.chunks == []
        # a device with no room refuses the return and loses nothing: the next restore succeeds
        from quantem.gpu.resident.cuda import counts

        def no_room(arrays):
            raise cp.cuda.memory.OutOfMemoryError(1, 0, 0)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(counts, "retain", no_room)
            with pytest.raises(MemoryError, match="intact in host memory"):
                source.restore()
        assert source.chunks == [] and source._parked is not None
        source.restore()
        assert source._parked is None and source.nbytes == device_bytes
        cp.testing.assert_array_equal(source.decode_scan_range_device(0, frames.shape[0]), frames)


# ---


def test_public_resident_fit_region_then_complete_1024_scan(tmp_path):
    values = _counts((1024, 1024, 8, 8))
    with _encoded(values, chunk_scans=65536) as data:
        with DirectPtycho(data, **SETTINGS, aberrations=ABERRATIONS, scan_detector_rotation_flipped=True) as session:
            assert session._data is data and session._compute is None
            region = (173, 301, 251, 379)
            fitted = session.aberrations(search="primary", tilt=False, scan_region=region, trials=0,
                refinement=None, check_rotation=False, verbose=False)
            assert fitted.phase.shape == (128, 128)
            assert fitted.fit_scan_region == region
            assert fitted.bright_field_center_px == SETTINGS["bright_field_center_px"]
            with DirectPtycho(data, **SETTINGS, aberrations=ABERRATIONS, scan_detector_rotation_flipped=True,
                     scan_region=region) as explicit:
                expected_fit = explicit.aberrations(search="primary", tilt=False, trials=0, refinement=None,
                    check_rotation=False, verbose=False)
                cp.testing.assert_array_equal(fitted.phase, expected_fit.phase)
            result = session.run(fitted, compute_loss=False)
            assert result.phase.shape == (1024, 1024) and result.loss is None
            assert result.fit_scan_region == region
            assert result.object_sampling_A == SETTINGS["scan_sampling_A"]
            assert session._compute is None
            # Existing 1024 callers retain diagnostic loss and other estimators.
            with_loss = session.run(fitted)
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
            with io.load(exported, device="cuda") as reopened:
                actual_counts = cp.from_dlpack(reopened.read())
                cp.testing.assert_array_equal(actual_counts[..., 0, :], expected_counts)
            # the phase-variance loss still reads the full Fourier stack at 1024
            phase, loss = session.preview(ABERRATIONS)
            assert phase.shape == (1024, 1024) and np.isfinite(phase).all() and np.isfinite(loss)
            # Dense reference is limited to 1024; the public large path matches it.
            values[..., 2, 3] = 0
            with DirectPtycho(values, **SETTINGS, scan_detector_rotation_flipped=True) as dense:
                expected = dense.run(fitted, compute_loss=False)
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
    with Dataset4dstem(source, resident_metadata(shape, np.uint16, source.nbytes, backend="cuda")) as data:
        tile[..., 2, 3] = 0
        with DirectPtycho(tile, **SETTINGS) as reference:
            expected, _ = reference.preview(ABERRATIONS, compute_loss=False, output="native",
                                             depth_spread_nm=14., sample_tilt_mrad=(1., -2.))
        with DirectPtycho(data, **SETTINGS) as session:
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
        with DirectPtycho.open(str(path), **SETTINGS) as session:
            owner = session._owned_dataset
            assert owner is session._data and not owner.data.is_released
            first = session.run(aberrations=ABERRATIONS, compute_loss=False,
                                        save_dir=tmp_path / "result")
            second = session.run(aberrations=ABERRATIONS, compute_loss=False,
                                         save_dir=tmp_path / "result")
            assert second.reused
            cp.testing.assert_array_equal(first.phase, second.phase)
        assert owner.data.is_released
        assert not original.data.is_released
