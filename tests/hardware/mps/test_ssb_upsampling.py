"""Fine-grid MPS reconstruction retains measured aliases and native loss."""

import math

import numpy as np
import pytest

from .test_ssb_thick_sample_mps import _require_mps


@pytest.mark.parametrize("angular_sampling", [(0.0007, 0.0006), (0.0003, 0.005), (0.08, 0.01)])
def test_alias_aperture_bounds_preserve_soft_edges(angular_sampling):
    """Check open, closed and anisotropic soft edges against the GPU model."""
    _require_mps()
    from quantem.gpu.ssb.mps.hardware import require_mlx
    from quantem.gpu.ssb.mps.upsampling import _ALIAS_GEOMETRY, _HEADER

    mx = require_mlx()
    wavelength, semiangle = 0.02, 0.03
    ay, ax = angular_sampling
    angles = np.linspace(-np.pi, np.pi, 513, dtype=np.float32)
    edge_width = np.hypot(np.cos(angles) * ay, np.sin(angles) * ax)
    offsets = np.linspace(-1.0, 1.0, 65, dtype=np.float32)
    radius = np.maximum(0, semiangle + offsets[:, None] * edge_width) / wavelength
    # Include both one-ULP neighbours of the soft-edge samples and the origin.
    radius = np.concatenate((radius, np.nextafter(radius, np.float32(0)),
                             np.nextafter(radius, np.float32(np.inf))))
    row = np.append((radius * np.cos(angles)).ravel(), np.float32(0))
    col = np.append((radius * np.sin(angles)).ravel(), np.float32(0))
    kernel = mx.fast.metal_kernel(
        name="test_alias_aperture_bounds",
        input_names=["row", "col", "params"],
        output_names=["original", "bounded"],
        header=_HEADER + _ALIAS_GEOMETRY,
        compile_options={"math_mode": "safe"},
        source="""
        uint i = thread_position_in_grid.x;
        if (i >= COUNT) return;
        float4 a = tf_geometry(row[i], col[i], params[0], params[1], params[2], params[3]);
        float4 b = alias_geometry(row[i], col[i], params[0], params[1], params[2], params[3]);
        for (uint j = 0; j < 4; ++j) {
            original[i * 4 + j] = a[j];
            bounded[i * 4 + j] = b[j];
        }
        """,
    )
    original, bounded = kernel(
        inputs=[mx.array(row), mx.array(col), mx.array([wavelength, semiangle, ay, ax])],
        template=[("COUNT", row.size)], grid=(row.size, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(row.size, 4)] * 2,
        output_dtypes=[mx.float32] * 2,
    )
    original, bounded = np.asarray(original), np.asarray(bounded)
    np.testing.assert_array_equal(bounded[:, 3].view(np.uint32), original[:, 3].view(np.uint32))
    active = original[:, 3] != 0
    np.testing.assert_array_equal(bounded[active].view(np.uint32), original[active].view(np.uint32))


def _disk_session(n=128, source_path=None, detector_sampling=(10.0, 10.0)):
    from quantem.gpu import SSB

    rng = np.random.default_rng(9)
    counts = rng.poisson(30.0, size=(n, n, 16, 16)).astype(np.uint16)
    rr, cc = np.meshgrid(np.arange(16) - 7.5, np.arange(16) - 7.5, indexing="ij")
    counts[:, :, np.hypot(rr, cc) < 3] += 150
    if source_path is not None:
        np.save(source_path, counts)
    return SSB(
        counts,
        backend="mps",
        voltage_kV=300,
        semiangle_mrad=30,
        scan_sampling_A=(0.3, 0.3),
        detector_sampling_mrad=detector_sampling,
        scan_detector_rotation_deg=0.0,  # the simulated scan has no rotation
        source_path=None if source_path is None else str(source_path),
    )


def _zero_inserted_reference(prepared, factor, coefs, tilt, depth):
    """Independent GPU scan construction, followed by the existing model graph: the phase of the mean wave."""
    from quantem.gpu.ssb.mps.prepared import expand_hermitian_mx
    from quantem.gpu.ssb.mps.thick_sample import _compiled_terms, _params

    mx = prepared.mx
    native = prepared.scan_shape[0]
    n = native * factor
    q = mx.array((np.fft.fftfreq(n) * n * float(prepared.q_row[1])).astype(np.float32))
    params = _params(prepared, *coefs, tilt, depth)
    total = mx.zeros((n, n), dtype=mx.complex64)
    for start in range(0, prepared.g_qk.shape[0], 7):
        stop = min(start + 7, prepared.g_qk.shape[0])
        source = expand_hermitian_mx(mx, prepared.g_qk[start:stop], native)
        images = mx.real(mx.fft.ifft2(source))
        sparse = mx.zeros((stop - start, n, n))
        sparse[:, ::factor, ::factor] = images
        aliases = mx.fft.fft2(sparse)
        real, imag, weight2 = _compiled_terms()(
            mx.real(aliases),
            mx.imag(aliases),
            q[None, :, None],
            q[None, None, :],
            prepared.kx[start:stop, None, None],
            prepared.ky[start:stop, None, None],
            params,
        )
        corrected = (real + 1j * imag) / mx.maximum(mx.sqrt(weight2), 1e-8)
        corrected[:, 0, 0] = prepared.dc_value
        total = total + mx.sum(corrected, axis=0)
        mx.eval(total)
    total[0, 0] = (
        total[0, 0]
        + (prepared.num_bright_field_pixels - prepared.g_qk.shape[0])
        * prepared.dc_value
    )
    wave = mx.fft.ifft2(total / prepared.num_bright_field_pixels)
    total = mx.arctan2(mx.imag(wave), mx.real(wave))
    mx.eval(total)
    return np.asarray(total)


@pytest.mark.parametrize("factor", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("depth_nm", [0.0, 18.0])
def test_alias_sampling_preserves_gpu_model_and_native_loss(factor, depth_nm):
    _require_mps()
    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": math.degrees(-0.7)}
    sample = {"sample_tilt_mrad": (-9.0, 5.0), "depth_spread_nm": depth_nm}
    with _disk_session() as ssb:
        native, native_loss = ssb.preview(coefs, **sample)
        prepared = ssb._backend()._prepared
        expected = _zero_inserted_reference(
            prepared, factor, (-80.0, 40.0, -0.7), (-9, 5), depth_nm * 10
        )
        phase, loss = ssb.preview(coefs, **sample, upsampling_factor=factor)
        assert phase.shape == (128 * factor, 128 * factor)
        np.testing.assert_allclose(phase, expected, rtol=0, atol=2e-5)
        assert loss == native_loss
        restored, restored_loss = ssb.preview(coefs, **sample)
        np.testing.assert_array_equal(restored, native)
        assert restored_loss == native_loss


# The fused kernel and the MLX reference differ by float32 reduction order. The largest pixel difference moves with
# the float32 rounding of k: 4.2e-8 to 6.8e-8 at 512 over seven roundings of the same k, so the bound is 1e-7.
# The unchanged 1024 kernel has a 1.4e-6 rad difference on this fixture;
# retain the existing thick-model 5e-6 float32 bound for that path.
@pytest.mark.parametrize(
    "n,atol", [(128, 1e-7), (256, 1e-7), (512, 1e-7), (1024, 5e-6)]
)
def test_fused_depth_model_at_all_native_sizes(n, atol):
    _require_mps()
    from quantem.gpu.ssb.mps.thick_sample import reconstruct_thick_reference

    with _disk_session(n) as ssb:
        coefs = {"C10_nm": 3.0, "C12_nm": 0.5, "phi12_deg": math.degrees(0.3)}
        _, preview_loss = ssb.preview(coefs, sample_tilt_mrad=(5.0, -3.0), depth_spread_nm=15.0)
        # the phase-variance objective's own mean BF phase, the fused model's product
        phase, loss = ssb._backend().sample_phase_variance(
            {"C10": 30.0, "C12": 5.0, "phi12": 0.3},
            {"tilt_row_mrad": 5.0, "tilt_col_mrad": -3.0, "depth_spread_A": 150.0},
        )
        assert preview_loss == loss
        expected, expected_loss = reconstruct_thick_reference(
            ssb._backend()._prepared,
            C10=30.0,
            C12=5.0,
            phi12=0.3,
            sample_tilt_mrad=(5.0, -3.0),
            thickness=150.0,
        )
        np.testing.assert_allclose(phase, expected, rtol=0, atol=atol)
        assert abs(loss - expected_loss) <= 1e-6 * abs(expected_loss)


def test_slider_changes_match_fresh_sessions():
    """Reuse a session while changing optics, rotation, and output sampling."""
    _require_mps()
    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}
    states = [
        (coefs, (-9., 5.), 18., 0., 3),
        (coefs, (-7., 5.), 18., 0., 3),
        (coefs, (-7., 5.), 24., 0., 3),
        (coefs | {"C10_nm": 6.0}, (-7., 5.), 24., 0., 3),
        (coefs | {"C21_nm": 20., "phi21_deg": 12.}, (-7., 5.), 24., 0., 3),
        (coefs, (-9., 5.), 18., 180., 3),
        (coefs, (-9., 5.), 18., 180., 2),
        (coefs, (0., 0.), 0., 0., 3),
        (coefs, (-9., 5.), 18., 0., 3),
    ]
    with _disk_session() as reused:
        for aberrations, tilt, depth, rotation, factor in states:
            options = dict(
                sample_tilt_mrad=tilt, depth_spread_nm=depth,
                upsampling_factor=factor, compute_loss=False,
            )
            reused.set_rotation(rotation)
            phase, _ = reused.preview(aberrations, **options)
            with _disk_session() as fresh:
                fresh.set_rotation(rotation)
                expected, _ = fresh.preview(aberrations, **options)
            np.testing.assert_array_equal(phase, expected)


@pytest.mark.parametrize("factor,higher", [(1, False), (3, False), (1, True)])
def test_cancelled_preview_returns_no_partial_result_and_session_recovers(factor, higher):
    """A renewed drag can abandon a diagnostic and reuse the same GPU session."""
    from concurrent.futures import CancelledError

    _require_mps()
    coefs = {"C10_nm": -8., "C12_nm": 4., "phi12_deg": -40.}
    if higher:
        coefs.update(C21_nm=20., phi21_deg=12.)
    settings = dict(sample_tilt_mrad=(-9., 5.), depth_spread_nm=18., upsampling_factor=factor)
    with _disk_session() as ssb:
        expected, expected_loss = ssb.preview(coefs, **settings)
        checks = []

        def cancel():
            checks.append(True)
            return len(checks) >= 3

        with pytest.raises(CancelledError):
            ssb.preview(coefs, **settings, cancel=cancel)
        assert len(checks) == 3
        phase, loss = ssb.preview(coefs, **settings, cancel=lambda: False)
        np.testing.assert_array_equal(phase, expected)
        assert loss == expected_loss


def test_coherent_preview_cancels_after_queued_batches_and_recovers(monkeypatch):
    """A superseded slider request discards completed partial accumulation."""
    from concurrent.futures import CancelledError

    _require_mps()
    from quantem.gpu.ssb.mps import upsampling

    from .test_ssb_thick_sample_mps import _poisson_disk_session

    # the per-term coherent kernel (higher orders; the two-term transfer has the separable kernel, depth or not)
    monkeypatch.setattr(upsampling, "_SEPARABLE", False)
    monkeypatch.setattr(upsampling, "_WORK_BYTES", (128 * 3) ** 2 * 8 * 3)
    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}
    settings = dict(sample_tilt_mrad=(-9.0, 5.0), depth_spread_nm=18.0,
                    upsampling_factor=3, compute_loss=False)
    with _poisson_disk_session() as ssb:
        expected, _ = ssb.preview(coefs, **settings)
        assert ssb._backend()._prepared.g_qk.shape[0] > 4 * 64
        kernel = upsampling._coherent_alias_kernel()
        dispatched = []

        def tracked_kernel(**kwargs):
            dispatched.append(True)
            return kernel(**kwargs)

        monkeypatch.setattr(
            upsampling, "_coherent_alias_kernel",
            lambda has_depth=True: tracked_kernel,
        )
        with pytest.raises(CancelledError):
            ssb.preview(coefs, **settings, cancel=lambda: len(dispatched) >= 4)
        assert len(dispatched) == 4
        actual, loss = ssb.preview(coefs, **settings, cancel=lambda: False)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert loss is None


@pytest.mark.parametrize("depth_nm", [0., 18.])
def test_interruptible_native_loss_keeps_exact_full_bf_sums(monkeypatch, depth_nm):
    """Persist accumulators across partial BF tiles, including a short last tile."""
    from quantem.gpu.ssb.mps import reconstruct

    _require_mps()
    coefs = {"C10_nm": -8., "C12_nm": 4., "phi12_deg": -40.}
    options = dict(sample_tilt_mrad=(-9., 5.), depth_spread_nm=depth_nm)
    with _disk_session(512) as ssb:
        expected, expected_loss = ssb.preview(coefs, **options)
        for tile in [8, 17]:
            monkeypatch.setattr(reconstruct, "_DIAGNOSTIC_BF_TILE", tile)
            phase, loss = ssb.preview(coefs, **options, cancel=lambda: False)
            np.testing.assert_array_equal(phase, expected)
            assert loss == expected_loss


def test_automatic_sampling_rotation_and_saved_result(tmp_path):
    """Replay fine sampling and both rotations without refitting."""
    _require_mps()
    from quantem.gpu.optics.physics import ssb_upsampling_factor

    coefs = {"C10_nm": 3.0, "C12_nm": 0.5, "phi12_deg": 15.0}
    with _disk_session(source_path=tmp_path / "source.npy") as ssb:
        fine = ssb.reconstruct(aberrations=coefs, upsampling_factor="auto")
        factor = ssb_upsampling_factor(
            voltage_kV=300,
            semiangle_mrad=30,
            scan_sampling_A=(0.3, 0.3),
        )
        assert fine.upsampling_factor == factor
        assert fine.phase.shape == (128 * factor, 128 * factor)
        np.testing.assert_allclose(fine.object_sampling_A, (0.3 / factor,) * 2)
        assert fine.amplitude_estimated is False
        original = fine.phase.copy()
        ssb.set_rotation(180.0)
        opposite = ssb.reconstruct(aberrations=coefs, upsampling_factor="auto")
        assert opposite.scan_detector_rotation_deg == 180.0
        assert not np.array_equal(original, opposite.phase)
        ssb.set_rotation(0.0)
        restored = ssb.reconstruct(
            aberrations=coefs, upsampling_factor="auto", save_dir=tmp_path / "fine"
        )
        np.testing.assert_array_equal(restored.phase, original)
        assert (tmp_path / "fine" / "ssb-reconstruct.npz").is_file()


@pytest.mark.parametrize("depth_nm", [0.0, 18.0])
def test_coherent_half_spectrum_retains_full_complex_result(monkeypatch, depth_nm):
    """Reduce paired frequencies only when the complete soft aperture fits."""
    _require_mps()
    from quantem.gpu.ssb.mps import upsampling

    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}
    sample = {"sample_tilt_mrad": (-9.0, 5.0), "depth_spread_nm": depth_nm}
    with _disk_session() as ssb:
        ssb.set_rotation(165.8)
        half, _ = ssb.preview(coefs, **sample, upsampling_factor=4, compute_loss=False)
        prepared = ssb._backend()._prepared
        assert upsampling._coherent_columns(prepared, 512) == 257
        # Native Nyquist frequencies intersect the aperture and are unpaired.
        assert upsampling._coherent_columns(prepared, 128) == 128
        monkeypatch.setattr(upsampling, "_coherent_columns", lambda prepared, n: n)
        full, _ = ssb.preview(coefs, **sample, upsampling_factor=4, compute_loss=False)
        np.testing.assert_allclose(half, full, rtol=0, atol=2e-5)


@pytest.mark.parametrize("depth_nm", [0.0, 18.0])
@pytest.mark.parametrize("higher_order", [False, True])
@pytest.mark.parametrize("small_work_budget", [False, True])
def test_coherent_reduction_keeps_partial_bf_groups(
    monkeypatch, depth_nm, higher_order, small_work_budget
):
    """Every BF contribution survives grouping, including an incomplete tail."""
    _require_mps()
    from quantem.gpu.ssb.mps import upsampling

    from .test_ssb_thick_sample_mps import _poisson_disk_session

    if small_work_budget:
        # Exercise accumulation across several launches and an incomplete tail.
        monkeypatch.setattr(upsampling, "_WORK_BYTES", 3 * (128 * 3) ** 2 * 8 * 3)
    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}
    if higher_order:
        coefs.update(C21_nm=300.0, phi21_deg=23.0)
    sample = {"sample_tilt_mrad": (-9.0, 5.0), "depth_spread_nm": depth_nm}
    with _poisson_disk_session() as ssb:
        ssb.set_rotation(165.8)
        actual, loss = ssb.preview(coefs, **sample, upsampling_factor=3)
        stored = int(ssb._backend()._prepared.g_qk.shape[0])
        chunk = upsampling._WORK_BYTES // ((128 * 3) ** 2 * 8 * 3)
        assert stored > chunk
        assert stored % chunk != 0
        if small_work_budget:
            assert stored // chunk >= 100

        def per_detector(prepared, qr, qc, params, dc, probe, coefficients,
                         higher, n, chunk, columns, has_depth=True):
            mx = prepared.mx
            total = mx.zeros((n, n), dtype=mx.complex64)
            # A final single-BF tile exercises constant-address MLX inputs.
            for start in range(0, stored, stored - 1):
                stop = min(start + stored - 1, stored)
                corrected = upsampling._alias_correction_kernel()(
                    inputs=[prepared.g_qk[start:stop], qr, qc,
                            prepared.kx[start:stop], prepared.ky[start:stop],
                            params, dc, probe[start:stop], coefficients],
                    template=[("N", n), ("SOURCE_N", 128),
                              ("G_COLS", prepared.g_qk.shape[-1]),
                              ("K", stop - start), ("HIGHER", higher)],
                    grid=(n, n, stop - start), threadgroup=(32, 8, 1),
                    output_shapes=[(stop - start, n, n)],
                    output_dtypes=[mx.complex64],
                )[0]
                total = total + mx.sum(corrected, axis=0)
                mx.eval(total)
            return total[:, :columns]

        with monkeypatch.context() as ungrouped:
            ungrouped.setattr(upsampling, "_coherent_spectrum", per_detector)
            expected, expected_loss = ssb.preview(coefs, **sample, upsampling_factor=3)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
        assert loss == expected_loss
        repeated, repeated_loss = ssb.preview(coefs, **sample, upsampling_factor=3)
        np.testing.assert_array_equal(repeated, actual)
        assert repeated_loss == loss


@pytest.mark.parametrize("depth_nm", [0.0, 18.0])
def test_coherent_reconstruction_preserves_anisotropic_aperture(depth_nm):
    """A rotated detector with unequal pixel angles retains its soft edge."""
    _require_mps()
    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": math.degrees(-0.7)}
    with _disk_session(detector_sampling=(10.0, 7.0)) as ssb:
        ssb.set_rotation(165.8)
        phase, _ = ssb.preview(
            coefs, sample_tilt_mrad=(-9.0, 5.0), depth_spread_nm=depth_nm,
            upsampling_factor=3, compute_loss=False,
        )
        expected = _zero_inserted_reference(
            ssb._backend()._prepared, 3, (-80.0, 40.0, -0.7),
            (-9.0, 5.0), depth_nm * 10,
        )
        np.testing.assert_allclose(phase, expected, rtol=0, atol=2e-5)


@pytest.mark.parametrize("higher", [False, True])
def test_zero_depth_specialization_preserves_all_phase_bits(monkeypatch, higher):
    """Turning depth off retains the full phase of the general optical model, and with higher orders its loss."""
    _require_mps()
    from quantem.gpu.ssb.mps import upsampling as up

    coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}
    if higher:
        coefs.update(C21_nm=0.47, phi21_deg=21.0)
    with _disk_session() as ssb:
        settings = dict(
            sample_tilt_mrad=(-9.0, 5.0), depth_spread_nm=0.0,
            upsampling_factor=3, compute_loss=higher,
        )
        actual, actual_loss = ssb.preview(coefs, **settings)
        coherent = up._coherent_alias_kernel
        detector = up._alias_correction_kernel
        with monkeypatch.context() as general:
            general.setattr(
                up, "_coherent_alias_kernel", lambda has_depth=True: coherent(True),
            )
            general.setattr(
                up, "_alias_correction_kernel", lambda has_depth=True: detector(True),
            )
            expected, expected_loss = ssb.preview(coefs, **settings)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        # the higher-order phase variance corrects each BF pixel with the per-detector kernel
        assert actual_loss == expected_loss


@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize("detector_sampling", [(10.0, 10.0), (10.0, 7.0)])
def test_separable_two_term_matches_per_term_transfer(monkeypatch, factor, detector_sampling):
    """The separable C10/C12 phasors reproduce the per-term transfer to float32 rounding."""
    _require_mps()
    from quantem.gpu.ssb.mps import upsampling as up

    cases = [
        {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0},
        {"C10_nm": 35.0, "C12_nm": 9.0, "phi12_deg": 75.0},
        {"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0},
        {"C10_nm": 0.3, "C12_nm": 2.0, "phi12_deg": 10.0},
    ]
    with _disk_session(detector_sampling=detector_sampling) as ssb:
        ssb.set_rotation(165.8)
        for coefs in cases:
            actual, _ = ssb.preview(coefs, upsampling_factor=factor, compute_loss=False)
            with monkeypatch.context() as reference:
                reference.setattr(up, "_SEPARABLE", False)
                expected, _ = ssb.preview(coefs, upsampling_factor=factor, compute_loss=False)
            difference = np.angle(np.exp(1j * (actual.astype(np.float64) - expected)))
            assert np.abs(difference).max() < 2e-6, (coefs, np.abs(difference).max())
        prepared = ssb._backend()._prepared
        # At native sampling the Nyquist row has support and its upper half is computed, not mirrored.
        assert (prepared.separable_geometry[1]["upper"].size > 0) == (factor == 1)


@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize("detector_sampling", [(10.0, 10.0), (10.0, 7.0)])
def test_separable_depth_weights_match_per_term_transfer(monkeypatch, factor, detector_sampling):
    """With sample tilt and depth spread the separable kernel reproduces the per-term transfer to float32 rounding."""
    _require_mps()
    from quantem.gpu.ssb.mps import upsampling as up

    cases = [
        ({"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}, (-9.0, 5.0), 18.0),
        ({"C10_nm": 35.0, "C12_nm": 9.0, "phi12_deg": 75.0}, (3.0, -4.0), 60.0),
        ({"C10_nm": 0.3, "C12_nm": 0.0, "phi12_deg": 0.0}, (0.0, 0.0), 2.0),
    ]
    with _disk_session(detector_sampling=detector_sampling) as ssb:
        ssb.set_rotation(165.8)
        for coefs, tilt, depth in cases:
            sample = dict(sample_tilt_mrad=tilt, depth_spread_nm=depth, upsampling_factor=factor, compute_loss=False)
            actual, _ = ssb.preview(coefs, **sample)
            for aliases in (1, 3):
                # one fine row per thread, or every fine row of a source row: the same arithmetic per output
                with monkeypatch.context() as single:
                    single.setattr(up, "_DEPTH_ALIASES", aliases)
                    ssb._backend()._prepared.separable_geometry = None
                    rows, _ = ssb.preview(coefs, **sample)
                np.testing.assert_array_equal(rows.view(np.uint32), actual.view(np.uint32))
            with monkeypatch.context() as reference:
                reference.setattr(up, "_SEPARABLE", False)
                expected, _ = ssb.preview(coefs, **sample)
            difference = np.angle(np.exp(1j * (actual.astype(np.float64) - expected)))
            assert np.abs(difference).max() < 2e-6, (coefs, tilt, depth, np.abs(difference).max())


@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize("depth_nm", [0.0, 18.0])
def test_odd_region_spectrum_matches_full_gpu_transfer(monkeypatch, factor, depth_nm):
    """Odd-sized regions retain every frequency at native and automatic sampling."""
    _require_mps()
    from quantem.gpu.ssb.mps import upsampling as up

    with _disk_session(n=165) as ssb:
        ssb.set_rotation(165.8)
        coefs = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}
        settings = dict(
            sample_tilt_mrad=(-9.0, 5.0), depth_spread_nm=depth_nm,
            upsampling_factor=factor, compute_loss=False,
        )
        with monkeypatch.context() as reference:
            reference.setattr(up, "_SEPARABLE", False)
            reference.setattr(up, "_coherent_columns", lambda prepared, n: n)
            expected, _ = ssb.preview(coefs, **settings)
        actual, _ = ssb.preview(coefs, **settings)
        assert actual.shape == expected.shape == (165 * factor, 165 * factor)
        assert np.isfinite(actual).all()
        difference = np.angle(np.exp(1j * (actual.astype(np.float64) - expected)))
        assert np.abs(difference).max() < 2e-6
        if factor % 2:
            assert ssb._backend()._prepared.separable_geometry[1]["upper"].size == 0
        # The nonseparable half-spectrum optimization must obey the same rule.
        with monkeypatch.context() as half:
            half.setattr(up, "_SEPARABLE", False)
            mirrored, _ = ssb.preview(coefs, **settings)
        np.testing.assert_allclose(mirrored, expected, rtol=0, atol=2e-6)


def test_paired_rows_at_2x_keep_every_phase_bit(monkeypatch):
    """At 2x one thread computes both fine rows of a source row; each output keeps its own arithmetic."""
    _require_mps()
    from quantem.gpu.ssb.mps import upsampling as up

    rows = np.array([0, 1, 2, 126, 127, 128, 129, 254, 255], np.uint32)
    pairs = up._alias_row_pairs(rows, 128)
    assert sorted(pairs[pairs != 0xFFFFFFFF].tolist()) == list(range(len(rows)))
    assert sorted(tuple(pair) for pair in pairs.tolist()) == [(0, 5), (1, 6), (2, 0xFFFFFFFF), (3, 7), (4, 8)]
    cases = [({"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": -40.0}, (0.0, 0.0)),
             ({"C10_nm": 35.0, "C12_nm": 9.0, "phi12_deg": 75.0}, (-3.0, 2.0))]
    with _disk_session(detector_sampling=(10.0, 7.0)) as ssb:
        ssb.set_rotation(165.8)
        for coefs, tilt in cases:
            paired, _ = ssb.preview(coefs, upsampling_factor=2, compute_loss=False, sample_tilt_mrad=tilt)
            with monkeypatch.context() as single:
                single.setattr(up, "_PAIRED", False)
                expected, _ = ssb.preview(coefs, upsampling_factor=2, compute_loss=False, sample_tilt_mrad=tilt)
            np.testing.assert_array_equal(paired.view(np.uint32), expected.view(np.uint32))


def test_separable_phasor_tables_reduce_exactly():
    """Each table phasor equals the float64 phase m * P reduced modulo 2 pi."""
    _require_mps()
    from quantem.gpu.ssb.mps import upsampling as up
    from quantem.gpu.ssb.mps.hardware import require_mlx

    mx = require_mlx()
    rng = np.random.default_rng(3)
    values = rng.uniform(-40.0, 40.0, size=257)
    values[:3] = (0.0, np.pi, -2.0 * np.pi + 1e-9)
    frequencies = (np.fft.fftfreq(1024) * 1024)[::3]
    parts = mx.array(up._split_mod(values))
    t_row, _ = up._separable_table_kernel()(
        inputs=[parts, parts, mx.array(frequencies.astype(np.float32)), mx.array(frequencies.astype(np.float32))],
        template=[("K", values.size), ("ROWS", frequencies.size), ("COLS", frequencies.size)],
        grid=(frequencies.size, values.size, 1), threadgroup=(256, 1, 1),
        output_shapes=[(values.size, frequencies.size)] * 2, output_dtypes=[mx.complex64] * 2,
    )
    expected = np.exp(-1j * np.remainder(values[:, None] * frequencies[None, :], 2.0 * np.pi))
    assert np.abs(np.asarray(t_row) - expected).max() < 1e-6
