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


def test_whole_scan_fit_holds_the_band_spectrum_and_the_encoded_scan_in_turn(tmp_path, monkeypatch):
    """Beyond the dense path, aberrations() fits every scan position on G(q, k) at the fit band's frequencies.

    The encoded scan waits in host memory except while the fit reconstructs a phase, when the band spectrum leaves
    the device instead; afterwards the scan is back, the band spectrum gone and the stack limit as before.
    """
    from quantem.gpu import io
    from quantem.gpu.direct_ptycho import workflow

    monkeypatch.setattr(workflow, "_exceeds_device", lambda data: True)
    path = tmp_path / "scan.qem"
    with _encoded(_counts((256, 256, 8, 8))) as original:
        io.save(path, original)
    turns = []
    batched = StreamedDirect.phase

    def phase(self, *args, **kwargs):
        if session._band_fit is not None:
            turns.append((len(self.chunks), session._compute.G_qk is None))
        return batched(self, *args, **kwargs)

    monkeypatch.setattr(StreamedDirect, "phase", phase)
    stack_bytes = cp.cuda.runtime.deviceGetLimit(cp.cuda.runtime.cudaLimitStackSize)
    with DirectPtycho.open(str(path), **SETTINGS) as session:
        streamed = session._streamed
        chunks = len(streamed.chunks)
        fit = session.aberrations(show=False, verbose=False)
        assert fit.fit_scan_region is None and turns and all(turn == (chunks, True) for turn in turns)
        assert session._compute is None and session._band_fit is None
        assert streamed.source._parked is None and len(streamed.chunks) == chunks
        assert cp.cuda.runtime.deviceGetLimit(cp.cuda.runtime.cudaLimitStackSize) == stack_bytes
        cp.testing.assert_array_equal(fit.phase, session.run(fit, compute_loss=False).phase)


@pytest.mark.parametrize("transposed", [False, True])
def test_band_spectrum_scores_trials_as_the_whole_spectrum(transposed, monkeypatch):
    """G(q, k) gathered at the fit band from batched detector columns is the dense half plane there, and every fit
    objective (the joint agreement, the fifth-order agreement and its exact gradients) reads the same values."""
    from quantem.gpu.direct_ptycho import workflow
    from quantem.gpu.direct_ptycho.cuda.exact_derivatives import FifthOrderGradients

    monkeypatch.setattr(workflow, "_exceeds_device", lambda data: True)
    values = _counts((128, 128, 8, 8))
    band_inv_A = workflow.WHOLE_SCAN_FIT_BAND_INV_A
    rng = np.random.default_rng(3)
    points = np.column_stack([rng.uniform(-80, 80, 6), rng.uniform(0, 40, 6), rng.uniform(-1.5, 1.5, 6),
                              rng.uniform(-8, 8, (6, 2)), rng.uniform(0, 60, 6)])
    z = rng.uniform(-0.5, 0.5, (5, 28))
    z[:, 5] = rng.uniform(0.2, 2, 5)
    with _encoded(values) as data, DirectPtycho(data, **SETTINGS, scan_detector_rotation_flipped=transposed) as encoded:
        values[..., 2, 3] = 0  # independent dense reference's scientific validity mask
        with DirectPtycho(values, **SETTINGS, scan_detector_rotation_flipped=transposed) as dense, \
                encoded._whole_scan_for_fit(encoded._whole_scan_band(False), False):
            band, whole = encoded._backend(), dense._backend()
            flat = whole._get_accelerator().thick._prepare_fit_band(band_inv_A)[2]
            expected = whole.G_qk.reshape(whole.G_qk.shape[0], -1)[:, flat]
            # R2C and C2C forward transforms of the same counts round differently in float32
            # measured 2.3e-7 of the largest |G|, 1.3e-7 relative agreement, 6e-7 of the largest gradient
            cp.testing.assert_allclose(band.G_qk[:, 0], expected, rtol=0, atol=1e-6 * float(cp.abs(expected).max()))
            assert flat.size == band.G_qk.shape[2]
            np.testing.assert_allclose(band.sample_agreement(points, band_inv_A),
                                       whole.sample_agreement(points, band_inv_A), rtol=5e-7)
            np.testing.assert_allclose(band.band_power(band_inv_A), whole.band_power(band_inv_A), rtol=1e-7)
            band_fifth = FifthOrderGradients(band._get_accelerator())
            whole_fifth = FifthOrderGradients(whole._get_accelerator())
            np.testing.assert_allclose(band_fifth.values(z), whole_fifth.values(z), rtol=2e-7)
            gradients = whole_fifth.gradients(z)
            np.testing.assert_allclose(band_fifth.gradients(z), gradients, rtol=0,
                                       atol=2e-6 * float(np.abs(gradients).max()))
            with pytest.raises(NotImplementedError, match="fit band's frequencies only"):
                band.preview_upsampled(aberrations_to_engine(ABERRATIONS), upsample=1, compute_loss=False)


def test_band_spectrum_of_a_scan_region_scores_trials_as_the_region_alone(monkeypatch):
    """A scan region's band spectrum, gathered from the whole encoded scan, is the dense half plane of the region's
    own counts at the region's band, and the joint agreement and the fifth-order values read the same there."""
    from quantem.gpu.direct_ptycho import workflow
    from quantem.gpu.direct_ptycho.cuda.exact_derivatives import FifthOrderGradients

    monkeypatch.setattr(workflow, "_exceeds_device", lambda data: True)
    values = _counts((256, 256, 8, 8))
    region = (64, 192, 32, 160)  # a side the dense reference takes without padding
    band_inv_A = workflow.WHOLE_SCAN_FIT_BAND_INV_A
    rng = np.random.default_rng(5)
    points = np.column_stack([rng.uniform(-80, 80, 6), rng.uniform(0, 40, 6), rng.uniform(-1.5, 1.5, 6),
                              rng.uniform(-8, 8, (6, 2)), rng.uniform(0, 60, 6)])
    z = rng.uniform(-0.5, 0.5, (5, 28))
    z[:, 5] = rng.uniform(0.2, 2, 5)
    with _encoded(values) as data, DirectPtycho(data, **SETTINGS) as encoded:
        cropped = cp.ascontiguousarray(values[region[0]:region[1], region[2]:region[3]])
        cropped[..., 2, 3] = 0  # independent dense reference's scientific validity mask
        placement = encoded._whole_scan_band(False, region)
        with DirectPtycho(cropped, **SETTINGS) as dense, encoded._whole_scan_for_fit(placement, False):
            band, whole = encoded._backend(), dense._backend()
            flat = whole._get_accelerator().thick._prepare_fit_band(band_inv_A)[2]
            expected = whole.G_qk.reshape(whole.G_qk.shape[0], -1)[:, flat]
            assert flat.size == band.G_qk.shape[2] == placement[0][2].size
            # R2C and C2C forward transforms of the same counts round differently in float32
            cp.testing.assert_allclose(band.G_qk[:, 0], expected, rtol=0, atol=1e-6 * float(cp.abs(expected).max()))
            np.testing.assert_allclose(band.sample_agreement(points, band_inv_A),
                                       whole.sample_agreement(points, band_inv_A), rtol=5e-7)
            np.testing.assert_allclose(FifthOrderGradients(band._get_accelerator()).values(z),
                                       FifthOrderGradients(whole._get_accelerator()).values(z), rtol=2e-7)



@pytest.mark.parametrize("resident_slabs", [1, 0])
def test_streamed_band_slabs_sum_to_the_band_held_whole(resident_slabs, monkeypatch):
    """A band spectrum the device holds only in part (its first slab of pixels, or none) streams the rest slab by slab;
    every fit objective then reads what it reads from the spectrum held whole: the joint agreement and its Newton
    derivatives, the band power, the fifth-order agreement and its gradients, on every pixel and on a subsample whose
    stride does not divide the slab."""
    from quantem.gpu.direct_ptycho import workflow
    from quantem.gpu.direct_ptycho.cuda.exact_derivatives import FifthOrderGradients, LowOrderDerivatives
    from quantem.gpu.direct_ptycho.cuda.thick import BAND_SLAB_ROWS

    monkeypatch.setattr(workflow, "_exceeds_device", lambda data: True)
    # 48 x 48 detector: about a thousand bright-field pixels, so the band spectrum spans four slabs
    settings = {**SETTINGS, "detector_sampling_mrad": (1.6, 1.75), "bright_field_center_px": (23.3, 24.1),
                "bright_field_radius_px": 18.0}
    band_inv_A = workflow.WHOLE_SCAN_FIT_BAND_INV_A
    rng = np.random.default_rng(5)
    points = np.column_stack([rng.uniform(-80, 80, 4), rng.uniform(0, 40, 4), rng.uniform(-1.5, 1.5, 4),
                              rng.uniform(-8, 8, (4, 2)), rng.uniform(0, 60, 4)])
    newton = np.column_stack([rng.uniform(-0.5, 0.5, (3, 5)), rng.uniform(0.2, 2, 3)])
    z = rng.uniform(-0.5, 0.5, (3, 28))
    z[:, 5] = rng.uniform(0.2, 2, 3)

    def objectives(backend):
        engine = backend._get_accelerator()
        low = LowOrderDerivatives(engine, band_inv_A)
        fifth = FifthOrderGradients(engine)
        # the band power first: it is the first reader of freshly allocated slab buffers
        record = {"power": backend.band_power(band_inv_A)}
        record.update(agreement=backend.sample_agreement(points, band_inv_A), low=low.values(newton),
                      low_derivatives=low.derivatives([0, 2]))
        for stride in (1, 14):
            record[f"fifth {stride}"] = fifth.values(z, stride)
            record[f"gradients {stride}"] = fifth.gradients(z, stride)
        return record

    with _encoded(_counts((64, 64, 48, 48))) as data, DirectPtycho(data, **settings) as encoded:
        band, _, _ = encoded._whole_scan_band(False)
        row_bytes = int(band[2].size) * 8
        assert encoded.num_bright_field_pixels > 3 * 256 and BAND_SLAB_ROWS == 256
        with encoded._whole_scan_for_fit((band, None, None), False):
            whole = objectives(encoded._backend())
        with encoded._whole_scan_for_fit((band, (resident_slabs * 256 + 2 * 256) * row_bytes, None), False):
            backend = encoded._backend()
            assert backend._resident_rows == resident_slabs * 256 and backend.G_qk.shape[0] == resident_slabs * 256
            streamed = objectives(backend)
    for name in ("agreement", "power", "low", "fifth 1", "fifth 14"):
        np.testing.assert_allclose(streamed[name], whole[name], rtol=1e-6, err_msg=name)
    for name in ("gradients 1", "gradients 14"):
        np.testing.assert_allclose(streamed[name], whole[name], rtol=0, atol=1e-5 * np.abs(whole[name]).max(), err_msg=name)
    for (value, gradient, hessian), (value_whole, gradient_whole, hessian_whole) in zip(streamed["low_derivatives"],
                                                                                         whole["low_derivatives"]):
        np.testing.assert_allclose(value, value_whole, rtol=1e-6)
        np.testing.assert_allclose(gradient, gradient_whole, rtol=0, atol=1e-5 * np.abs(gradient_whole).max())
        np.testing.assert_allclose(hessian, hessian_whole, rtol=0, atol=1e-5 * np.abs(hessian_whole).max())

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
        # a device with no room that cannot read host memory either (no unified addressing) refuses the return and
        # loses nothing: the next restore succeeds
        from quantem.gpu.resident.cuda import counts

        def no_room(*args):
            raise cp.cuda.memory.OutOfMemoryError(1, 0, 0)

        def no_unified_addressing(*args):
            raise MemoryError("no unified addressing")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(counts, "retain", no_room)
            patch.setattr(counts, "host_mapped", no_unified_addressing)
            with pytest.raises(MemoryError, match="intact in host memory"):
                source.restore()
        assert source.chunks == [] and source._parked is not None
        source.restore()
        assert source._parked is None and source.nbytes == device_bytes
        cp.testing.assert_array_equal(source.decode_scan_range_device(0, frames.shape[0]), frames)



def test_shared_passes_give_every_refinement_what_it_gets_alone(monkeypatch):
    """Refinements that share passes over G (``SharedPasses``, ``_MergedEvaluations``) see the values and gradients
    they see alone, bit for bit: per-member band weights, mixed strides and searches of different lengths included."""
    from quantem.gpu.direct_ptycho import workflow
    from quantem.gpu.direct_ptycho._resident_search import _Device, _MergedEvaluations, _search
    from quantem.gpu.direct_ptycho.cuda.exact_derivatives import FifthOrderGradients, SharedPasses

    monkeypatch.setattr(workflow, "_exceeds_device", lambda data: True)
    rng = np.random.default_rng(7)
    z = rng.uniform(-0.4, 0.4, (3, 28))
    z[:, 5] = 1.0
    with _encoded(_counts((128, 128, 8, 8))) as data, DirectPtycho(data, **SETTINGS) as encoded:
        with encoded._whole_scan_for_fit(encoded._whole_scan_band(False), False):
            engine = encoded._backend()._get_accelerator()
            objective = FifthOrderGradients(engine)
            mask = np.arange(objective.band_frequencies()[0].size) % 2 == 0

            def alone(target, point, stride):
                values = target.values(point, stride)
                return values, target.gradients(point, stride)

            expected = [alone(objective, z[0], 1), alone(objective.with_band_weights(mask), z[1], 1),
                        alone(objective, z[2], 4)]
            shared = SharedPasses(objective)
            members = [shared.member(), shared.member().with_band_weights(mask), shared.member()]
            got = shared.run([lambda: alone(members[0], z[0], 1), lambda: alone(members[1], z[1], 1),
                              lambda: [alone(members[2], z[2], 4), alone(members[2], z[0], 4)][0]])
            for (values, gradients), (values_alone, gradients_alone) in zip(got, expected):
                np.testing.assert_array_equal(values, values_alone)
                np.testing.assert_array_equal(gradients, gradients_alone)
            device = _Device("cuda")
            evaluate = lambda trial: engine.thick.fit_resident(trial, (0.2, 0.9))
            bounds = device.pack([-300.0, 0.0, -1.5, -25.0, -25.0, 0.0, 300.0, 200.0, 1.5, 25.0, 25.0, 300.0])
            thin = device.pack([-300.0, 0.0, -1.5, 0.0, 0.0, 0.0, 300.0, 200.0, 1.5, 0.0, 0.0, 0.0])
            separate = [_search(device, evaluate, thin, 40, 3, thin=True), _search(device, evaluate, bounds, 24, 3)]
            merged = _MergedEvaluations(evaluate, cp).run([lambda shared: _search(device, shared, thin, 40, 3, thin=True),
                                                           lambda shared: _search(device, shared, bounds, 24, 3)])
            for (points, scores), (points_alone, scores_alone) in zip(merged, separate):
                cp.testing.assert_array_equal(points, points_alone)
                cp.testing.assert_array_equal(scores, scores_alone)


def test_retain_frees_idle_pool_blocks_when_the_exact_allocation_runs_out(monkeypatch):
    """An encoded chunk's own allocation that runs out frees the memory pool's idle blocks and succeeds on retry."""
    from quantem.gpu.resident.cuda import counts

    arrays = (cp.arange(1000, dtype=cp.uint16), cp.ones(77, cp.uint32))
    from types import SimpleNamespace

    calls, freed, empty = [], [], cp.empty
    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: SimpleNamespace(free_all_blocks=lambda: freed.append(True)))

    def runs_out_once(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise cp.cuda.runtime.CUDARuntimeError(2)  # cudaErrorMemoryAllocation
        return empty(*args, **kwargs)

    monkeypatch.setattr(cp, "empty", runs_out_once)
    kept = counts.retain(arrays)
    assert len(calls) == 2 and freed
    for array, view in zip(arrays, kept):
        cp.testing.assert_array_equal(view, array)


def test_encoded_chunks_share_blocks_and_a_restored_scan_takes_one_allocation():
    """Loaded chunks pack back to back into blocks sized to the scans still to come, not one allocation each (each
    rounded up to whole 2 MiB pages); a parked scan returns into one allocation; every scan decodes as appended."""
    values = _counts((40, 40, 16, 16))
    source = _encoded(values, chunk_scans=50).data
    page = 2 << 20
    encoded = sum(math.ceil(array.nbytes / 256) * 256 for chunk in source.chunks for array in chunk.arrays)

    def allocations():
        return {id(array.data.mem): array.data.mem.size for chunk in source.chunks for array in chunk.arrays}

    loaded = allocations()
    assert len(source.chunks) == 32 and source._blocks is None
    assert sum(loaded.values()) <= math.ceil(encoded / page) * page + len(loaded) * page < 32 * page
    source.park()
    source.restore()
    assert list(allocations().values()) == [math.ceil(encoded / page) * page]
    frames = values.reshape(-1, 16, 16)
    cp.testing.assert_array_equal(source.decode_scan_range_device(0, 1600).reshape(frames.shape), frames)


def test_encoded_scan_the_device_has_no_room_for_reads_the_same_from_host_memory(tmp_path):
    """Chunks the device has no room for stay in page-locked host memory, read there in place: every pattern, the
    detector total, a virtual image, the centre of mass and a whole-scan phase give the device-resident scan's bits.
    Parked, the scan returns to host memory while the device still has no room and to the device once it has; a
    `.qem` the device has no room for loads into host memory the same way."""
    from quantem.gpu import io
    from quantem.gpu.detector.cuda.streamed_series import StreamedSeriesCompute
    from quantem.gpu.io import qem
    from quantem.gpu.resident.cuda import counts

    values = _counts((40, 40, 16, 16))
    mask = np.zeros((16, 16), bool)
    mask[4:12, 4:12] = True
    options = dict(voltage_kV=300., semiangle_mrad=30., scan_sampling_A=SETTINGS["scan_sampling_A"], rotation_deg=27.,
                   upsample=1, sample_tilt_mrad=(1., -2.), depth_spread_A=0.)

    def reads(data):
        source = data.data
        source._detector_total = None  # summed again from the chunks, as for a scan read from a file
        queries = StreamedSeriesCompute([data])
        streamed = StreamedDirect(
            data, mean_pattern=None, center=SETTINGS["bright_field_center_px"],
            radius=SETTINGS["bright_field_radius_px"], threshold=0., semiangle_mrad=30.,
            detector_sampling_mrad=SETTINGS["detector_sampling_mrad"], scan_detector_rotation_flipped=False,
            full_scan_only=False,
        )
        return (source.decode_scan_range_device(0, 1600), source.detector_total_device(),
                queries.masked_sum_exact(mask), *queries.center_of_mass(),
                streamed.phase(aberrations_to_engine(ABERRATIONS), **options)[0])

    def assert_same(expected, got):
        for one, other in zip(expected, got, strict=True):
            cp.testing.assert_array_equal(cp.asarray(other), cp.asarray(one))

    device_data = _encoded(values, chunk_scans=50)
    on_device = reads(device_data)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(counts, "_allocate", lambda nbytes: None)
        data = _encoded(values, chunk_scans=50)
        source = data.data
        assert source.host_nbytes == sum(chunk.nbytes for chunk in source.chunks) > 0
        assert_same(on_device, reads(data))
        source.park()
        source.restore()
        assert source.host_nbytes == sum(chunk.nbytes for chunk in source.chunks)
        assert_same(on_device, reads(data))
    source.park()
    source.restore()
    assert source.host_nbytes == 0
    assert_same(on_device, reads(data))
    saved = tmp_path / "scan.qem"
    io.save(saved, device_data)

    def host_arena(nbytes, device):
        host = counts.pinned(nbytes)
        return counts.host_mapped(host, device), host

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(qem, "_arena", host_arena)
        with io.load(saved, device="cuda", verbose=False) as loaded:
            assert loaded.data.host_nbytes == sum(chunk.nbytes for chunk in loaded.data.chunks) > 0
            assert_same(on_device, reads(loaded))

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
