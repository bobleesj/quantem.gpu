from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.ssb_precision import (
    LOSS_ATOL,
    LOSS_RTOL,
    PHASE_ATOL,
    PHASE_RTOL,
    PRECISION,
)


def test_mps_phase_loss_default_chunk_is_size_aware(monkeypatch) -> None:
    """MPS exact phase/loss default must avoid oversized 1024 Metal outputs."""
    monkeypatch.delenv("QUANTEM_MPS_SSB_PHASE_CHUNK_BF", raising=False)

    from quantem.gpu.ssb.compute.mps.engine import (
        _default_object_redraw_threadgroup,
        _default_phase_col_k_bf,
        _default_phase_loss_chunk_bf,
    )

    assert _default_phase_loss_chunk_bf((1024, 1024)) <= 512
    assert _default_phase_loss_chunk_bf((512, 512)) >= 512
    assert _default_phase_col_k_bf((512, 512)) == 4096
    assert _default_object_redraw_threadgroup((512, 512)) == 64
    assert _default_object_redraw_threadgroup((1024, 1024)) == 64


def test_mps_exact_nelder_mead_uses_rounded_wide_simplex_steps() -> None:
    """Exact refinement must not collapse C12/phi steps near a good TPE seed."""
    from quantem.gpu.ssb.compute.mps.engine import _nelder_mead_refine

    seed = {"C10": 68.9103129988, "C12": 10.1981172003, "phi12": 0.3873672179}
    evaluated = []

    def evaluate(params):
        evaluated.append(dict(params))
        return 1.0

    _nelder_mead_refine(
        seed,
        0.0,
        evaluate,
        lock=set(),
        max_iter=0,
        initial_step_floor={"C12": 2.0, "phi12": 0.04},
        initial_step_decimals=2,
    )

    assert len(evaluated) == 3
    np.testing.assert_allclose(
        [evaluated[0]["C10"], evaluated[0]["C12"], evaluated[0]["phi12"]],
        [seed["C10"] + 3.45, seed["C12"], seed["phi12"]],
    )
    np.testing.assert_allclose(
        [evaluated[1]["C10"], evaluated[1]["C12"], evaluated[1]["phi12"]],
        [seed["C10"], seed["C12"] + 2.0, seed["phi12"]],
    )
    np.testing.assert_allclose(
        [evaluated[2]["C10"], evaluated[2]["C12"], evaluated[2]["phi12"]],
        [seed["C10"], seed["C12"], seed["phi12"] + 0.04],
    )


def test_mps_exact_refinement_caches_identical_float32_inputs() -> None:
    """Distinct simplex coordinates must share an identical Metal evaluation."""
    from quantem.gpu.ssb.compute.mps.engine import _evaluate_exact_float32_cached

    calls = []
    cache = {}

    def evaluate(params):
        calls.append(dict(params))
        return 0.125

    first = {
        "C10": 73.73451902106731,
        "C12": 14.728482694817712,
        "phi12": 0.47684490099801885,
    }
    repeated = {
        "C10": 73.73451902106733,
        "C12": 14.728482694817712,
        "phi12": 0.4768449009980189,
    }

    assert _evaluate_exact_float32_cached(first, evaluate, cache) == 0.125
    assert _evaluate_exact_float32_cached(repeated, evaluate, cache) == 0.125
    assert len(calls) == 1


def test_load_bf_columns_mps_keeps_exact_sparse_detector_source(tmp_path) -> None:
    """C1: exact BF companion, expect MPS sums and coordinate-only gathers."""
    pytest.importorskip("mlx.core")
    from quantem.gpu.detector import mean_dp
    from quantem.gpu.ssb.compute.mps.engine import (
        _resolve_bf_selection,
        load_bf_columns_mps,
    )

    folder = tmp_path / "showptycho"
    source = folder / "source"
    snapshots = folder / "snapshots"
    source.mkdir(parents=True)
    snapshots.mkdir()
    bf_rows = np.asarray([0, 1, 2], dtype=np.int32)
    bf_cols = np.asarray([1, 2, 3], dtype=np.int32)
    columns = np.asarray(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
            [7, 0, 1, 0, 7, 0, 1, 0, 7, 0, 1, 0, 7, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    (source / "bf_columns.u8").write_bytes(columns.tobytes())
    (snapshots / "cal.json").write_text(json.dumps({
        "scan_region": {"shape": [4, 4]},
        "detector_shape": [4, 5],
        "bf_rows": bf_rows.tolist(),
        "bf_cols": bf_cols.tolist(),
        "bf_center": [1.0, 2.0],
        "bf_radius_px": 2.25,
        "bf_column_companion_path": "source/bf_columns.u8",
        "bf_column_encoding": "uint8",
        "dc_value": [
            float(columns.sum(axis=1).astype(np.complex64).mean().real),
            0.0,
        ],
    }))
    (snapshots / "manifest.json").write_text(json.dumps({
        "source": {"bf_columns": {"max_value": int(columns.max())}},
    }))

    frames = load_bf_columns_mps(folder)

    assert frames._detector_sum is None
    assert frames.dc_value == complex(
        columns.sum(axis=1).astype(np.complex64).mean()
    )

    np.testing.assert_array_equal(
        frames.columns(bf_rows[[2, 0]], bf_cols[[2, 0]]),
        columns[[2, 0]],
    )
    expected_dp = np.zeros((4, 5), dtype=np.float32)
    expected_dp[bf_rows, bf_cols] = columns.sum(axis=1) / 16
    np.testing.assert_array_equal(mean_dp(frames), expected_dp)
    selection = _resolve_bf_selection(
        frames,
        threshold=0.5,
        bf_radius=None,
    )
    np.testing.assert_array_equal(selection.rows, bf_rows)
    np.testing.assert_array_equal(selection.cols, bf_cols)
    assert selection.center_row_col == (1.0, 2.0)
    assert selection.radius_px == 2.25
    assert selection.detected_radius_px == 2.25
    assert selection is frames.selection
    assert frames.nbytes == columns.nbytes


@pytest.mark.parametrize(
    ("offset", "expected_dtype", "expected_suffix"),
    [
        (0, np.dtype(np.uint8), "u8"),
        (300, np.dtype(np.uint16), "u16"),
    ],
)
def test_mps_backend_writes_exact_integer_bf_columns(
    tmp_path,
    offset: int,
    expected_dtype: np.dtype,
    expected_suffix: str,
) -> None:
    """C1: export uses the smallest lossless dtype without changing counts."""
    from quantem.gpu.ssb.bf_selector import BrightfieldDisk
    from quantem.gpu.ssb.compute.mps.backend import MpsSSBBackend
    from quantem.gpu.ssb.compute.mps.engine import MpsBfColumnFrames

    class RawFrames:
        _np_dtype = np.dtype(np.uint16)

        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.freed = False

        def columns_float32(self, rows, cols) -> np.ndarray:
            indices = np.asarray(rows) * 4 + np.asarray(cols)
            return self.values[indices].astype(np.float32)

        def free(self) -> None:
            self.freed = True

    rows = np.asarray([0, 1, 2], dtype=np.int32)
    cols = np.asarray([1, 2, 3], dtype=np.int32)
    selection = BrightfieldDisk(
        rows=rows,
        cols=cols,
        center_row_col=(1.0, 2.0),
        radius_px=2.0,
        detected_radius_px=2.0,
        detector_shape=(3, 4),
    )
    source_values = (
        np.arange(12 * 6, dtype=np.uint16).reshape(12, 6) + offset
    )
    raw_frames = RawFrames(source_values)
    detector_sum = source_values.sum(axis=1, dtype=np.uint64).reshape(3, 4)
    backend = MpsSSBBackend.__new__(MpsSSBBackend)
    backend._frames = raw_frames
    backend._source_data = raw_frames
    backend._selection = selection
    backend._scan_shape = (2, 3)
    backend._detector_sum = detector_sum
    backend._dc_value_override = complex(
        np.complex64(detector_sum[rows, cols].astype(np.float64).mean())
    )
    backend._prepared = object()

    written = backend.export_brightfield(
        raw_frames,
        tmp_path / "bf_columns",
    )

    assert written is not None
    path, elapsed = written
    assert path == (tmp_path / f"bf_columns.{expected_suffix}").resolve()
    assert elapsed >= 0.0
    expected = source_values[rows * 4 + cols]
    actual = np.memmap(path, mode="r", dtype=expected_dtype).reshape(3, 6)
    np.testing.assert_array_equal(actual, expected)
    assert raw_frames.freed
    assert isinstance(backend._frames, MpsBfColumnFrames)
    np.testing.assert_array_equal(backend._frames.detector_sum, detector_sum)
    assert backend._bf_source_max_value == int(expected.max())


def test_mps_engine_implements_backend_contract_for_exact_bf_source(tmp_path) -> None:
    """C1: exact BF columns expose the same neutral contract as CUDA."""
    pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute import SSBProtocol
    from quantem.gpu.ssb.bf_selector import BrightfieldDisk
    from quantem.gpu.ssb.compute.mps.engine import MpsBfColumnFrames
    from quantem.gpu.ssb.compute.mps.backend import MpsSSBBackend

    rows = np.asarray([1, 1, 2, 2], dtype=np.int32)
    cols = np.asarray([1, 2, 1, 2], dtype=np.int32)
    selection = BrightfieldDisk(
        rows=rows,
        cols=cols,
        center_row_col=(1.5, 1.5),
        radius_px=1.0,
        detected_radius_px=1.0,
        detector_shape=(4, 4),
    )
    columns = np.arange(4 * 128 * 128, dtype=np.uint16).reshape(4, -1)
    source = tmp_path / "bf_columns.u16"
    source.write_bytes(columns.tobytes())
    frames = MpsBfColumnFrames(
        source,
        selection=selection,
        scan_shape=(128, 128),
        dtype=np.uint16,
        max_value=int(columns.max()),
    )

    engine = MpsSSBBackend(
        frames,
        voltage_kV=300.0,
        semiangle_mrad=30.0,
        scan_sampling=0.5,
        det_sampling=1.0,
        bf_intensity_threshold=0.5,
        bf_center=None,
        bf_radius=None,
        rotation_angle_deg=4.0,
    )

    assert isinstance(engine, SSBProtocol)
    assert engine.precision == PRECISION
    assert engine.scan_shape == (128, 128)
    assert engine.detector_shape == (4, 4)
    assert engine.num_bf == 4
    state = engine.browser_state()
    assert engine._prepared is None
    assert state.brightfield is selection
    assert state.bf_source_path == source.resolve()
    assert state.bf_source_dtype == np.dtype(np.uint16)
    assert state.bf_source_max_value == int(columns.max())


def test_brightfield_selection_is_validated_and_immutable() -> None:
    """C1a: one typed BF value owns validated row/column geometry."""
    from quantem.gpu.ssb.bf_selector import BrightfieldDisk

    rows = np.asarray([0, 1, 2], dtype=np.int64)
    cols = np.asarray([1, 2, 3], dtype=np.int64)
    selection = BrightfieldDisk(
        rows=rows,
        cols=cols,
        center_row_col=(1.0, 2.0),
        radius_px=2.0,
        detected_radius_px=2.25,
        detector_shape=(4, 5),
    )

    rows[0] = 3
    assert selection.size == 3
    assert selection.rows.dtype == np.int32
    assert selection.rows.tolist() == [0, 1, 2]
    assert selection.cols.tolist() == [1, 2, 3]
    assert selection.center_row_col == (1.0, 2.0)
    with pytest.raises(ValueError, match="read-only"):
        selection.rows[0] = 3


def test_brightfield_selection_rejects_duplicate_coordinates() -> None:
    """C1b: duplicate scientific evidence fails at construction."""
    from quantem.gpu.ssb.bf_selector import BrightfieldDisk

    with pytest.raises(ValueError, match="duplicates"):
        BrightfieldDisk(
            rows=np.asarray([1, 1]),
            cols=np.asarray([2, 2]),
            center_row_col=(1.0, 2.0),
            radius_px=1.0,
            detected_radius_px=1.0,
            detector_shape=(4, 5),
        )


def test_load_bf_columns_mps_requires_authoritative_center(tmp_path) -> None:
    """C1c: incomplete BF metadata fails before loading scientific data."""
    from quantem.gpu.ssb.compute.mps.engine import load_bf_columns_mps

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "cal.json").write_text(json.dumps({
        "scan_region": {"shape": [4, 4]},
        "detector_shape": [4, 5],
        "bf_rows": [0],
        "bf_cols": [1],
        "bf_column_companion_path": "source/bf_columns.u8",
        "bf_column_encoding": "uint8",
    }))

    with pytest.raises(ValueError, match="bf_center"):
        load_bf_columns_mps(tmp_path)


def test_mps_radix8_batch_phase_loss_matches_radix4_reference() -> None:
    """Radix-8 column FFT must preserve phase/loss numerical parity."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _phase_cols512_reduced_batch_kernel,
        _phase_cols512_scalar_loss_batch_from_row_ifft,
        _twiddle_512,
    )

    rng = np.random.default_rng(41)
    row_ifft_np = (
        rng.standard_normal((2, 5, 512, 512))
        + 1j * rng.standard_normal((2, 5, 512, 512))
    ).astype(np.complex64)
    row_ifft = mx.array(row_ifft_np)
    k_bf = 3
    groups = 2
    reference_kernel = _phase_cols512_reduced_batch_kernel(2, 5, k_bf)
    reference_sum, reference_sumsq_tile = reference_kernel(
        inputs=[row_ifft, _twiddle_512(mx)],
        template=[],
        grid=(128, 512, 2 * groups),
        threadgroup=(128, 4, 1),
        output_shapes=[(2, groups, 512, 512), (2, groups, 512, 128)],
        output_dtypes=[mx.float32, mx.float32],
    )
    reference_sum = mx.sum(reference_sum, axis=1)
    reference_sumsq = mx.sum(reference_sumsq_tile, axis=(1, 2, 3))
    result_sum, result_sumsq = (
        _phase_cols512_scalar_loss_batch_from_row_ifft(
            mx,
            row_ifft,
            k_bf=k_bf,
        )
    )
    mx.eval(reference_sum, reference_sumsq, result_sum, result_sumsq)

    sum_result = np.asarray(result_sum)
    sum_reference = np.asarray(reference_sum)
    sum_close = np.isclose(sum_result, sum_reference, rtol=2e-5, atol=2e-4)
    # Different float32 FFT association can put an exactly negative-real
    # random value on opposite sides of atan2's branch cut. Such a pixel
    # differs by one 2*pi wrap, while all non-cut pixels retain tight parity.
    assert np.count_nonzero(~sum_close) <= 1
    if not np.all(sum_close):
        wrapped = np.abs(sum_result[~sum_close] - sum_reference[~sum_close])
        np.testing.assert_allclose(wrapped, 2.0 * np.pi, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(
        np.asarray(result_sumsq),
        np.asarray(reference_sumsq),
        rtol=2e-6,
        atol=1.0,
    )


def test_mps_tiled_row_intermediate_preserves_exact_column_result() -> None:
    """The 8-column 512 cache layout must only permute row-IFFT storage."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _phase_cols512_scalar_loss_batch_from_row_ifft,
    )

    rng = np.random.default_rng(83)
    row_major = (
        rng.standard_normal((2, 3, 512, 512))
        + 1j * rng.standard_normal((2, 3, 512, 512))
    ).astype(np.complex64)
    tiled = row_major.reshape(2, 3, 512, 64, 8).transpose(
        0,
        1,
        3,
        2,
        4,
    ).reshape(2, 3, 512, 512)
    reference = _phase_cols512_scalar_loss_batch_from_row_ifft(
        mx,
        mx.array(row_major),
        k_bf=2,
    )
    result = _phase_cols512_scalar_loss_batch_from_row_ifft(
        mx,
        mx.array(tiled),
        k_bf=2,
        tiled_input=True,
    )
    mx.eval(*reference, *result)

    np.testing.assert_array_equal(
        np.asarray(result[0]),
        np.asarray(reference[0]),
    )
    np.testing.assert_array_equal(
        np.asarray(result[1]),
        np.asarray(reference[1]),
    )


def test_mps_exact_pair_row_storage_uses_bounded_classes(monkeypatch) -> None:
    """Exact pair packs reuse aligned classes without accepting oversized packs."""
    from quantem.gpu.ssb.compute.mps import engine

    monkeypatch.setattr(
        engine,
        "_exact_pair_row_policy_512",
        lambda _batch=2: (300, (288, 320)),
    )
    storage_class = engine._exact_pair_row_storage_bf_512

    assert storage_class(176) == 288
    assert storage_class(256) == 288
    assert storage_class(257) == 288
    assert storage_class(288) == 288
    assert storage_class(289) == 320
    assert storage_class(300) == 320
    with pytest.raises(ValueError, match="321 BF planes"):
        storage_class(321)


def test_mps_exact_pair_row_storage_uses_high_memory_class(monkeypatch) -> None:
    """Large-memory Macs may reduce launches without changing BF boundaries."""
    from quantem.gpu.ssb.compute.mps import engine

    monkeypatch.setattr(
        engine,
        "_exact_pair_row_policy_512",
        lambda _batch=2: (716, (720,)),
    )

    assert engine._exact_pair_row_storage_bf_512(321) == 720
    assert engine._exact_pair_row_storage_bf_512(720) == 720
    with pytest.raises(ValueError, match="721 BF planes"):
        engine._exact_pair_row_storage_bf_512(721)


def test_mps_exact_pair_row_policy_selects_m5_max(monkeypatch) -> None:
    """The measured topology must not spread to unmeasured high-memory chips."""
    from quantem.gpu.ssb.compute.mps import engine

    monkeypatch.setattr(
        engine,
        "_apple_hardware_profile",
        lambda: (128 * 1024**3, "Apple M5 Max"),
    )
    engine._exact_pair_row_policy_512.cache_clear()
    try:
        assert engine._exact_pair_row_policy_512(2) == (2476, (2496,))
        assert engine._exact_pair_row_policy_512(1) == (300, (288, 320))
        monkeypatch.setattr(
            engine,
            "_apple_hardware_profile",
            lambda: (96 * 1024**3, "Apple M5 Max"),
        )
        engine._exact_pair_row_policy_512.cache_clear()
        assert engine._exact_pair_row_policy_512(2) == (716, (720,))
        monkeypatch.setattr(
            engine,
            "_apple_hardware_profile",
            lambda: (128 * 1024**3, "Apple M4 Max"),
        )
        engine._exact_pair_row_policy_512.cache_clear()
        assert engine._exact_pair_row_policy_512(2) == (300, (288, 320))
    finally:
        engine._exact_pair_row_policy_512.cache_clear()


def test_mps_simd_column_stage_is_scoped_to_m5_max(monkeypatch) -> None:
    """The register transpose stays on its measured hardware family."""
    from quantem.gpu.ssb.compute.mps import engine

    monkeypatch.setattr(
        engine,
        "_apple_hardware_profile",
        lambda: (128 * 1024**3, "Apple M5 Max"),
    )
    assert engine._use_simd_radix8_col_stage_512()

    monkeypatch.setattr(
        engine,
        "_apple_hardware_profile",
        lambda: (128 * 1024**3, "Apple M4 Max"),
    )
    assert not engine._use_simd_radix8_col_stage_512()


def test_mps_exact_pair_keeps_512_logical_boundaries_on_large_memory_hosts() -> None:
    """High-memory phase chunks must not overflow pair row storage classes."""
    from quantem.gpu.ssb.compute.mps.engine import (
        _EXACT_PAIR_LOGICAL_BOUNDARY_BF_512,
    )

    assert min(4096, _EXACT_PAIR_LOGICAL_BOUNDARY_BF_512) == 512


def test_mps_1024_candidate_pair_uses_fused_exact_path(monkeypatch) -> None:
    """Native 1024 Optuna pairs must share geometry and evidence reads."""
    from types import SimpleNamespace

    from quantem.gpu.ssb.compute.mps import engine

    expected = np.asarray([0.125, 0.25], dtype=np.float32)

    def fused(prepared, **kwargs):
        assert prepared.scan_shape == (1024, 1024)
        assert kwargs["packed_columns"] is False
        return expected

    monkeypatch.setattr(
        engine,
        "_reconstruct_prepared_small_batch_exact_loss_fused",
        fused,
    )
    prepared = SimpleNamespace(
        scan_shape=(1024, 1024),
        alpha_k2=None,
    )

    result = engine._reconstruct_prepared_batch_exact_loss(
        prepared,
        C10=np.asarray([1.0, 2.0], dtype=np.float32),
        C12=np.asarray([3.0, 4.0], dtype=np.float32),
        phi12=np.asarray([0.1, 0.2], dtype=np.float32),
        chunk_bf=512,
    )

    np.testing.assert_array_equal(result, expected)


def test_mps_column_subrange_matches_standalone_storage() -> None:
    """Packed row storage must not change a boundary's column reduction."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _phase_cols512_scalar_loss_batch_from_row_ifft,
    )

    rng = np.random.default_rng(97)
    packed_np = (
        rng.standard_normal((1, 4, 512, 512))
        + 1j * rng.standard_normal((1, 4, 512, 512))
    ).astype(np.complex64)
    packed = _phase_cols512_scalar_loss_batch_from_row_ifft(
        mx,
        mx.array(packed_np),
        k_bf=2,
        bf_start=1,
        bf_stop=3,
    )
    standalone = _phase_cols512_scalar_loss_batch_from_row_ifft(
        mx,
        mx.array(packed_np[:, 1:3]),
        k_bf=2,
    )
    mx.eval(*packed, *standalone)

    np.testing.assert_array_equal(np.asarray(packed[0]), np.asarray(standalone[0]))
    np.testing.assert_array_equal(np.asarray(packed[1]), np.asarray(standalone[1]))


def test_mps_zero_aperture_bf_skip_preserves_exact_phase_sums() -> None:
    """Analytic positive-DC BF rows must contribute exactly zero phase."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _phase_cols512_sum_from_row_ifft,
        _phase_cols512_scalar_loss_batch_from_row_ifft,
    )

    rng = np.random.default_rng(73)
    row_ifft = np.zeros((1, 3, 512, 512), dtype=np.complex64)
    row_ifft[:, (0, 2)] = (
        rng.standard_normal((1, 2, 512, 512))
        + 1j * rng.standard_normal((1, 2, 512, 512))
    ).astype(np.complex64)
    row_ifft[:, 1, 0, :] = np.complex64(4508.0 + 0.0j)
    row_ifft_mx = mx.array(row_ifft)

    reference = _phase_cols512_scalar_loss_batch_from_row_ifft(
        mx,
        row_ifft_mx,
        k_bf=4096,
    )
    skipped = _phase_cols512_scalar_loss_batch_from_row_ifft(
        mx,
        row_ifft_mx,
        k_bf=4096,
        active_bf=mx.array([1, 0, 1], dtype=mx.uint8),
    )
    phase_only = _phase_cols512_sum_from_row_ifft(
        mx,
        row_ifft_mx[0],
        k_bf=4096,
        active_bf=mx.array([1, 0, 1], dtype=mx.uint8),
    )
    mx.eval(*reference, *skipped, phase_only)

    np.testing.assert_array_equal(np.asarray(skipped[0]), np.asarray(reference[0]))
    np.testing.assert_array_equal(np.asarray(skipped[1]), np.asarray(reference[1]))
    np.testing.assert_array_equal(np.asarray(phase_only), np.asarray(reference[0])[0])
    assert np.isfinite(np.asarray(phase_only)).all()


def test_mps_hermitian_expand_matches_fft2_reference() -> None:
    """MPS half-plane storage must expand to the exact full FFT grid."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import _expand_hermitian_mx, _fft2_hermitian

    rng = np.random.default_rng(17)
    stack = rng.standard_normal((3, 16, 16)).astype(np.float32)
    half = _fft2_hermitian(mx, mx.array(stack))
    full = _expand_hermitian_mx(mx, half, 16)
    mx.eval(full)

    assert tuple(int(v) for v in half.shape) == (3, 16, 9)
    assert tuple(int(v) for v in full.shape) == (3, 16, 16)
    np.testing.assert_allclose(
        np.asarray(full),
        np.fft.fft2(stack).astype(np.complex64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mps_array_frame_columns_match_direct_detector_indexing() -> None:
    """Optimized detector-column gather must preserve exact BF evidence."""
    from quantem.gpu.ssb.compute.mps.engine import _ArrayFrames

    data = np.arange(4 * 5 * 6 * 7, dtype=np.uint16).reshape(4, 5, 6, 7)
    frames = _ArrayFrames(data)
    rows = np.array([0, 2, 5, 3], dtype=np.intp)
    cols = np.array([1, 6, 0, 4], dtype=np.intp)

    result = frames.columns(rows, cols)
    reference = data.reshape(-1, 6, 7)[:, rows, cols].T

    assert result.shape == (4, 20)
    np.testing.assert_array_equal(result, reference)


def test_mps_object_fourier_sum_matches_looped_reference() -> None:
    """MPS object Fourier sum must match the looped corrected-object path."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _PreparedMpsSSB,
        _corrected_from_dynamic_geometry,
        _ifft2_chunked,
        _object_fourier_sum_dynamic,
        _reconstruct_prepared,
    )

    n = 16
    num_bf = 5
    q_row = np.fft.fftfreq(n, 1.0).astype(np.float32)
    q_col = np.fft.fftfreq(n, 1.0).astype(np.float32)
    kx_np = np.linspace(-0.2, 0.2, num_bf, dtype=np.float32)
    ky_np = np.linspace(0.15, -0.15, num_bf, dtype=np.float32)
    rng = np.random.default_rng(22)
    real_stack = rng.standard_normal((num_bf, n, n)).astype(np.float32)
    g_qk = mx.fft.rfft2(mx.array(real_stack))
    dc_mask = np.zeros((n, n), dtype=bool)
    dc_mask[0, 0] = True
    q_row_mx = mx.array(q_row, dtype=mx.float32)
    q_col_mx = mx.array(q_col, dtype=mx.float32)
    prepared = _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=q_row_mx[None, :, None],
        qy=q_col_mx[None, None, :],
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=mx.array(kx_np, dtype=mx.float32),
        ky=mx.array(ky_np, dtype=mx.float32),
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=complex(np.asarray(g_qk[:, 0, 0]).mean()),
        scan_shape=(n, n),
        wavelength=0.0197,
        semiangle_rad=0.0214,
        ang_y_rad=0.0008,
        ang_x_rad=0.0008,
        factor=float(np.pi / 0.0197),
        dc_mask=mx.array(dc_mask),
        num_bf=num_bf,
        alpha_k2=None,
        cos2_k=None,
        sin2_k=None,
        aperture_k=None,
        alpha_m2=None,
        cos2_m=None,
        sin2_m=None,
        ap_m=None,
        alpha_p2=None,
        cos2_p=None,
        sin2_p=None,
        ap_p=None,
    )
    c10 = mx.array([-120.0], dtype=mx.float32)
    c12 = mx.array([55.0], dtype=mx.float32)
    phi12 = 0.3
    corrected = _corrected_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=mx.array([np.cos(2.0 * phi12)], dtype=mx.float32),
        sin2phi12=mx.array([np.sin(2.0 * phi12)], dtype=mx.float32),
    )[0]
    reference = mx.fft.ifft2(mx.sum(corrected, axis=0) / num_bf)
    obj_chunk_reference = mx.fft.ifft2(corrected)
    obj_chunk_chunked = _ifft2_chunked(mx, corrected)
    result = _object_fourier_sum_dynamic(
        prepared,
        C10=-120.0,
        C12=55.0,
        phi12=phi12,
        chunk_bf=3,
    )
    mx.eval(reference, obj_chunk_reference, obj_chunk_chunked, result)

    np.testing.assert_allclose(
        np.asarray(result),
        np.asarray(reference),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(obj_chunk_chunked),
        np.asarray(obj_chunk_reference),
        rtol=1e-5,
        atol=1e-5,
    )

    phase_reference = mx.arctan2(
        mx.imag(obj_chunk_reference),
        mx.real(obj_chunk_reference),
    )
    phase_sum = mx.sum(phase_reference, axis=0)
    phase_sumsq = mx.sum(phase_reference * phase_reference, axis=0)
    mean_phase_reference = phase_sum / num_bf
    loss_reference = mx.mean(
        phase_sumsq / num_bf - mean_phase_reference * mean_phase_reference
    )
    prepared_object, prepared_loss, prepared_phase = _reconstruct_prepared(
        prepared,
        C10=-120.0,
        C12=55.0,
        phi12=phi12,
        chunk_bf=3,
        compute_loss=True,
        compute_object=True,
    )
    mx.eval(mean_phase_reference, loss_reference)
    np.testing.assert_allclose(
        prepared_object,
        np.asarray(mx.sum(obj_chunk_reference, axis=0) / num_bf),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        prepared_phase,
        np.asarray(mean_phase_reference),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        prepared_loss,
        float(np.asarray(loss_reference)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mps_phase_cols512_matches_mlx_reference() -> None:
    """Fused 512-column phase/loss accumulator must match MLX IFFT reference."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _phase_cols512_from_row_ifft,
        _phase_sums_from_complex,
    )

    rng = np.random.default_rng(31)
    row_ifft_np = (
        rng.standard_normal((5, 512, 512))
        + 1j * rng.standard_normal((5, 512, 512))
    ).astype(np.complex64)
    row_ifft = mx.array(row_ifft_np)
    obj_reference = mx.fft.ifft(row_ifft, axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, obj_reference)
    got_sum, got_sumsq = _phase_cols512_from_row_ifft(mx, row_ifft, k_bf=32)
    mx.eval(ref_sum, ref_sumsq, got_sum, got_sumsq)

    np.testing.assert_allclose(
        np.asarray(got_sum),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(got_sumsq),
        np.asarray(ref_sumsq),
        rtol=1e-5,
        atol=1e-3,
    )


def test_mps_phase_cols512_reduced_modes_match_mlx_reference() -> None:
    """Reduced 512-column phase/loss modes must preserve phase and scalar loss."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _phase_cols512_scalar_loss_from_row_ifft,
        _phase_cols512_sum_from_row_ifft,
        _phase_sums_from_complex,
    )

    rng = np.random.default_rng(37)
    row_ifft_np = (
        rng.standard_normal((7, 512, 512))
        + 1j * rng.standard_normal((7, 512, 512))
    ).astype(np.complex64)
    row_ifft = mx.array(row_ifft_np)
    obj_reference = mx.fft.ifft(row_ifft, axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, obj_reference)
    ref_sumsq_scalar = mx.sum(ref_sumsq)
    sum_only = _phase_cols512_sum_from_row_ifft(mx, row_ifft, k_bf=4)
    scalar_sum, scalar_sumsq = _phase_cols512_scalar_loss_from_row_ifft(
        mx,
        row_ifft,
        k_bf=4,
    )
    mx.eval(ref_sum, ref_sumsq_scalar, sum_only, scalar_sum, scalar_sumsq)

    np.testing.assert_allclose(
        np.asarray(sum_only),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(scalar_sum),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(scalar_sumsq),
        np.asarray(ref_sumsq_scalar),
        rtol=1e-5,
        atol=1e-1,
    )


def test_mps_reconstruct_cached_geometry_512_loss_matches_mlx_reference() -> None:
    """Cached-geometry 512 phase/loss must not broadcast scalar sumsq as an image."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _PreparedMpsSSB,
        _compute_geometry,
        _corrected_from_cached_geometry,
        _phase_sums_from_complex,
        _reconstruct_prepared,
    )

    n = 512
    num_bf = 5
    rng = np.random.default_rng(38)
    real_stack = rng.standard_normal((num_bf, n, n)).astype(np.float32)
    g_qk = mx.fft.rfft2(mx.array(real_stack))
    q_row = np.fft.fftfreq(n, 1.0).astype(np.float32)
    q_col = np.fft.fftfreq(n, 1.0).astype(np.float32)
    kx_np = np.linspace(-0.17, 0.13, num_bf, dtype=np.float32)
    ky_np = np.linspace(0.11, -0.15, num_bf, dtype=np.float32)
    q_row_mx = mx.array(q_row, dtype=mx.float32)
    q_col_mx = mx.array(q_col, dtype=mx.float32)
    kx = mx.array(kx_np, dtype=mx.float32)[:, None, None]
    ky = mx.array(ky_np, dtype=mx.float32)[:, None, None]
    qx = q_row_mx[None, :, None]
    qy = q_col_mx[None, None, :]
    wavelength = 0.0197
    semiangle_rad = 0.0214
    ang_y_rad = 0.0008
    ang_x_rad = 0.0008
    alpha_k2, cos2_k, sin2_k, aperture_k = _compute_geometry(
        mx,
        kx,
        ky,
        wavelength,
        semiangle_rad,
        ang_y_rad,
        ang_x_rad,
    )
    alpha_m2, cos2_m, sin2_m, ap_m = _compute_geometry(
        mx,
        qx - kx,
        qy - ky,
        wavelength,
        semiangle_rad,
        ang_y_rad,
        ang_x_rad,
    )
    alpha_p2, cos2_p, sin2_p, ap_p = _compute_geometry(
        mx,
        qx + kx,
        qy + ky,
        wavelength,
        semiangle_rad,
        ang_y_rad,
        ang_x_rad,
    )
    dc_mask = np.zeros((n, n), dtype=bool)
    dc_mask[0, 0] = True
    prepared = _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=qx,
        qy=qy,
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=mx.array(kx_np, dtype=mx.float32),
        ky=mx.array(ky_np, dtype=mx.float32),
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=complex(np.asarray(g_qk[:, 0, 0]).mean()),
        scan_shape=(n, n),
        wavelength=wavelength,
        semiangle_rad=semiangle_rad,
        ang_y_rad=ang_y_rad,
        ang_x_rad=ang_x_rad,
        factor=float(np.pi / wavelength),
        dc_mask=mx.array(dc_mask),
        num_bf=num_bf,
        alpha_k2=alpha_k2,
        cos2_k=cos2_k,
        sin2_k=sin2_k,
        aperture_k=aperture_k,
        alpha_m2=alpha_m2,
        cos2_m=cos2_m,
        sin2_m=sin2_m,
        ap_m=ap_m,
        alpha_p2=alpha_p2,
        cos2_p=cos2_p,
        sin2_p=sin2_p,
        ap_p=ap_p,
    )
    c10 = mx.array([0.0], dtype=mx.float32)
    c12 = mx.array([50.0], dtype=mx.float32)
    cos2phi12 = mx.array([1.0], dtype=mx.float32)
    sin2phi12 = mx.array([0.0], dtype=mx.float32)
    corrected = _corrected_from_cached_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=cos2phi12,
        sin2phi12=sin2phi12,
    )[0]
    obj_reference = mx.fft.ifft2(corrected)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, obj_reference)
    ref_phase = ref_sum / num_bf
    ref_loss = mx.mean(ref_sumsq / num_bf - ref_phase * ref_phase)

    _obj, loss, phase = _reconstruct_prepared(
        prepared,
        C10=0.0,
        C12=50.0,
        phi12=0.0,
        chunk_bf=16,
        compute_loss=True,
        compute_object=False,
    )
    mx.eval(ref_phase, ref_loss)

    np.testing.assert_allclose(phase, np.asarray(ref_phase), rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(loss, float(np.asarray(ref_loss)), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("n", [128, 256, 1024])
def test_mps_phase_cols_small_reduced_modes_match_mlx_reference(n: int) -> None:
    """Reduced 128/256/1024-column phase/loss modes must match MLX IFFT reference."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _phase_cols_small_scalar_loss_from_row_ifft,
        _phase_cols_small_sum_from_row_ifft,
        _phase_sums_from_complex,
    )

    rng = np.random.default_rng(37 + n)
    row_ifft_np = (
        rng.standard_normal((7, n, n))
        + 1j * rng.standard_normal((7, n, n))
    ).astype(np.complex64)
    row_ifft = mx.array(row_ifft_np)
    obj_reference = mx.fft.ifft(row_ifft, axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, obj_reference)
    ref_sumsq_scalar = mx.sum(ref_sumsq)
    sum_only = _phase_cols_small_sum_from_row_ifft(mx, row_ifft, k_bf=4)
    scalar_sum, scalar_sumsq = _phase_cols_small_scalar_loss_from_row_ifft(
        mx,
        row_ifft,
        k_bf=4,
    )
    mx.eval(ref_sum, ref_sumsq_scalar, sum_only, scalar_sum, scalar_sumsq)

    np.testing.assert_allclose(
        np.asarray(sum_only),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(scalar_sum),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(scalar_sumsq),
        np.asarray(ref_sumsq_scalar),
        rtol=1e-5,
        atol=1e-1,
    )


def test_mps_row_ifft512_dynamic_matches_mlx_reference() -> None:
    """Fused 512 correction + row-IFFT must match the MLX reference path."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _PreparedMpsSSB,
        _corrected_from_dynamic_geometry,
        _phase_cols512_from_row_ifft,
        _phase_sums_from_complex,
        _row_ifft512_from_dynamic_geometry,
    )

    n = 512
    num_bf = 3
    q_row = np.fft.fftfreq(n, 1.0).astype(np.float32)
    q_col = np.fft.fftfreq(n, 1.0).astype(np.float32)
    kx_np = np.array([-0.17, 0.03, 0.21], dtype=np.float32)
    ky_np = np.array([0.11, -0.19, 0.07], dtype=np.float32)
    rng = np.random.default_rng(47)
    real_stack = rng.standard_normal((num_bf, n, n)).astype(np.float32)
    g_qk = mx.fft.rfft2(mx.array(real_stack))
    dc_mask = np.zeros((n, n), dtype=bool)
    dc_mask[0, 0] = True
    q_row_mx = mx.array(q_row, dtype=mx.float32)
    q_col_mx = mx.array(q_col, dtype=mx.float32)
    prepared = _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=q_row_mx[None, :, None],
        qy=q_col_mx[None, None, :],
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=mx.array(kx_np, dtype=mx.float32),
        ky=mx.array(ky_np, dtype=mx.float32),
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=complex(np.asarray(g_qk[:, 0, 0]).mean()),
        scan_shape=(n, n),
        wavelength=0.0197,
        semiangle_rad=0.0214,
        ang_y_rad=0.0008,
        ang_x_rad=0.0008,
        factor=float(np.pi / 0.0197),
        dc_mask=mx.array(dc_mask),
        num_bf=num_bf,
        alpha_k2=None,
        cos2_k=None,
        sin2_k=None,
        aperture_k=None,
        alpha_m2=None,
        cos2_m=None,
        sin2_m=None,
        ap_m=None,
        alpha_p2=None,
        cos2_p=None,
        sin2_p=None,
        ap_p=None,
    )
    c10 = mx.array([0.0], dtype=mx.float32)
    c12 = mx.array([0.0], dtype=mx.float32)
    cos2 = mx.array([1.0], dtype=mx.float32)
    sin2 = mx.array([0.0], dtype=mx.float32)
    corrected = _corrected_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=cos2,
        sin2phi12=sin2,
    )[0]
    row_reference = mx.fft.ifft(corrected, axis=-1)
    row_fused = _row_ifft512_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=cos2,
        sin2phi12=sin2,
    ) / n
    ref_obj = mx.fft.ifft(row_reference, axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, ref_obj)
    got_sum, got_sumsq = _phase_cols512_from_row_ifft(mx, row_fused, k_bf=32)
    mx.eval(row_reference, row_fused, ref_sum, ref_sumsq, got_sum, got_sumsq)

    np.testing.assert_allclose(
        np.asarray(row_fused),
        np.asarray(row_reference),
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(got_sum),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(got_sumsq),
        np.asarray(ref_sumsq),
        rtol=1e-5,
        atol=1e-3,
    )


@pytest.mark.parametrize("candidates", [1, 2, 4, 8])
def test_mps_batched_exact_loss512_matches_single_candidate_path(
    candidates: int,
    monkeypatch,
) -> None:
    """Batched 512 exact loss must match the single-candidate MPS path."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _PreparedMpsSSB,
        _reconstruct_prepared,
        _reconstruct_prepared_batch_exact_loss,
    )

    n = 512
    num_bf = 4
    q_row = np.fft.fftfreq(n, 1.0).astype(np.float32)
    q_col = np.fft.fftfreq(n, 1.0).astype(np.float32)
    kx_np = np.array([-0.17, 0.03, 0.21, -0.09], dtype=np.float32)
    ky_np = np.array([0.11, -0.19, 0.07, 0.18], dtype=np.float32)
    rng = np.random.default_rng(49)
    real_stack = rng.standard_normal((num_bf, n, n)).astype(np.float32)
    g_qk = mx.fft.rfft2(mx.array(real_stack))
    dc_mask = np.zeros((n, n), dtype=bool)
    dc_mask[0, 0] = True
    q_row_mx = mx.array(q_row, dtype=mx.float32)
    q_col_mx = mx.array(q_col, dtype=mx.float32)
    prepared = _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=q_row_mx[None, :, None],
        qy=q_col_mx[None, None, :],
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=mx.array(kx_np, dtype=mx.float32),
        ky=mx.array(ky_np, dtype=mx.float32),
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=complex(np.asarray(g_qk[:, 0, 0]).mean()),
        scan_shape=(n, n),
        wavelength=0.0197,
        semiangle_rad=0.0214,
        ang_y_rad=0.0008,
        ang_x_rad=0.0008,
        factor=float(np.pi / 0.0197),
        dc_mask=mx.array(dc_mask),
        num_bf=num_bf,
        alpha_k2=None,
        cos2_k=None,
        sin2_k=None,
        aperture_k=None,
        alpha_m2=None,
        cos2_m=None,
        sin2_m=None,
        ap_m=None,
        alpha_p2=None,
        cos2_p=None,
        sin2_p=None,
        ap_p=None,
    )

    from quantem.gpu.ssb.compute.mps import engine as mps_module

    observed_kernel_batches = []
    observed_probe_batches = []
    original_row_batch = mps_module._row_ifft512_batch_from_dynamic_geometry
    original_probe_batch = mps_module._pk_batch_from_prepared

    def record_row_batch(*args, **kwargs):
        observed_kernel_batches.append(int(np.asarray(kwargs["c10"]).size))
        return original_row_batch(*args, **kwargs)

    monkeypatch.setattr(
        mps_module,
        "_row_ifft512_batch_from_dynamic_geometry",
        record_row_batch,
    )

    def record_probe_batch(*args, **kwargs):
        observed_probe_batches.append(
            (int(kwargs["start"]), int(kwargs["stop"]))
        )
        return original_probe_batch(*args, **kwargs)

    monkeypatch.setattr(
        mps_module,
        "_pk_batch_from_prepared",
        record_probe_batch,
    )

    c10 = np.array(
        [-120.0, 35.0, 72.0, -18.0, 91.0, -63.0, 44.0, 7.0],
        dtype=np.float32,
    )[:candidates]
    c12 = np.array(
        [55.0, 12.0, 24.0, 41.0, 9.0, 68.0, 33.0, 17.0],
        dtype=np.float32,
    )[:candidates]
    phi12 = np.array(
        [0.3, -0.2, 0.12, -0.44, 0.51, -0.37, 0.08, 0.26],
        dtype=np.float32,
    )[:candidates]
    got = _reconstruct_prepared_batch_exact_loss(
        prepared,
        C10=c10,
        C12=c12,
        phi12=phi12,
        chunk_bf=2,
    )
    got_probe_batches = list(observed_probe_batches)
    expected = []
    for values in zip(c10, c12, phi12):
        _object, loss, _phase = _reconstruct_prepared(
            prepared,
            C10=float(values[0]),
            C12=float(values[1]),
            phi12=float(values[2]),
            chunk_bf=2,
            compute_loss=True,
            compute_object=False,
            return_phase=False,
        )
        expected.append(loss)

    np.testing.assert_allclose(
        got,
        np.asarray(expected, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )
    assert observed_kernel_batches
    assert max(observed_kernel_batches) <= 2
    assert got_probe_batches == [(0, num_bf)] * ((candidates + 1) // 2)


def test_mps_small_exact_batches_use_parallel_stream_workers(
    monkeypatch,
) -> None:
    """Native small batches wider than a pair must not serialize loss calls."""
    from types import SimpleNamespace

    from quantem.gpu.ssb.compute.mps import engine as mps

    calls = []

    class CompletedFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class ImmediateExecutor:
        def submit(self, fn, *args):
            calls.append(args[1:4])
            return CompletedFuture(fn(*args))

    monkeypatch.setattr(mps, "_small_exact_executor", ImmediateExecutor)
    monkeypatch.setattr(
        mps,
        "_small_exact_loss_worker",
        lambda _prepared, c10, c12, phi12, _chunk: c10 + c12 + phi12,
    )
    prepared = SimpleNamespace(
        scan_shape=(128, 128),
        alpha_k2=None,
    )

    losses = mps._reconstruct_prepared_batch_exact_loss(
        prepared,
        C10=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        C12=np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
        phi12=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        chunk_bf=512,
    )

    np.testing.assert_array_equal(
        losses,
        np.asarray([4.1, 6.2, 8.3], dtype=np.float32),
    )
    assert calls == [
        (1.0, 3.0, np.float32(0.1)),
        (2.0, 4.0, np.float32(0.2)),
        (3.0, 5.0, np.float32(0.3)),
    ]


@pytest.mark.parametrize("size", [128, 256])
def test_mps_small_exact_pair_uses_fused_path(size: int, monkeypatch) -> None:
    """Native 128/256 candidate pairs must share exact Metal row inputs."""
    from types import SimpleNamespace

    from quantem.gpu.ssb.compute.mps import engine as mps

    calls = []

    def fake_fused(
        prepared,
        *,
        c10_np,
        c12_np,
        phi_np,
        chunk_bf,
        packed_columns,
    ):
        calls.append(
            (
                prepared.scan_shape,
                c10_np,
                c12_np,
                phi_np,
                chunk_bf,
                packed_columns,
            )
        )
        return c10_np + c12_np + phi_np

    monkeypatch.setattr(
        mps,
        "_reconstruct_prepared_small_batch_exact_loss_fused",
        fake_fused,
    )
    prepared = SimpleNamespace(scan_shape=(size, size), alpha_k2=None)
    losses = mps._reconstruct_prepared_batch_exact_loss(
        prepared,
        C10=np.asarray([1.0, 2.0], dtype=np.float32),
        C12=np.asarray([3.0, 4.0], dtype=np.float32),
        phi12=np.asarray([0.1, 0.2], dtype=np.float32),
        chunk_bf=512,
    )

    np.testing.assert_array_equal(losses, np.asarray([4.1, 6.2], dtype=np.float32))
    assert len(calls) == 1
    assert calls[0][0] == (size, size)
    assert calls[0][-2] == 512
    assert calls[0][-1] == (size == 256)


@pytest.mark.parametrize("n", [128, 256, 1024])
def test_mps_row_ifft_small_dynamic_matches_mlx_reference(n: int) -> None:
    """Fused 128/256/1024 correction + row-IFFT must match the MLX reference."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _PreparedMpsSSB,
        _corrected_from_dynamic_geometry,
        _phase_cols_small_scalar_loss_from_row_ifft,
        _phase_sums_from_complex,
        _row_ifft_small_from_dynamic_geometry,
    )

    num_bf = 3
    q_row = np.fft.fftfreq(n, 1.0).astype(np.float32)
    q_col = np.fft.fftfreq(n, 1.0).astype(np.float32)
    kx_np = np.array([-0.17, 0.03, 0.21], dtype=np.float32)
    ky_np = np.array([0.11, -0.19, 0.07], dtype=np.float32)
    rng = np.random.default_rng(47 + n)
    real_stack = rng.standard_normal((num_bf, n, n)).astype(np.float32)
    g_qk = mx.fft.rfft2(mx.array(real_stack))
    dc_mask = np.zeros((n, n), dtype=bool)
    dc_mask[0, 0] = True
    q_row_mx = mx.array(q_row, dtype=mx.float32)
    q_col_mx = mx.array(q_col, dtype=mx.float32)
    prepared = _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=q_row_mx[None, :, None],
        qy=q_col_mx[None, None, :],
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=mx.array(kx_np, dtype=mx.float32),
        ky=mx.array(ky_np, dtype=mx.float32),
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=complex(np.asarray(g_qk[:, 0, 0]).mean()),
        scan_shape=(n, n),
        wavelength=0.0197,
        semiangle_rad=0.0214,
        ang_y_rad=0.0008,
        ang_x_rad=0.0008,
        factor=float(np.pi / 0.0197),
        dc_mask=mx.array(dc_mask),
        num_bf=num_bf,
        alpha_k2=None,
        cos2_k=None,
        sin2_k=None,
        aperture_k=None,
        alpha_m2=None,
        cos2_m=None,
        sin2_m=None,
        ap_m=None,
        alpha_p2=None,
        cos2_p=None,
        sin2_p=None,
        ap_p=None,
    )
    c10 = mx.array([0.0], dtype=mx.float32)
    c12 = mx.array([0.0], dtype=mx.float32)
    cos2 = mx.array([1.0], dtype=mx.float32)
    sin2 = mx.array([0.0], dtype=mx.float32)
    corrected = _corrected_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=cos2,
        sin2phi12=sin2,
    )[0]
    row_reference = mx.fft.ifft(corrected, axis=-1)
    row_fused = _row_ifft_small_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=c10,
        c12=c12,
        cos2phi12=cos2,
        sin2phi12=sin2,
    ) / n
    ref_obj = mx.fft.ifft(row_reference, axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, ref_obj)
    got_sum, got_sumsq_scalar = _phase_cols_small_scalar_loss_from_row_ifft(
        mx,
        row_fused,
        k_bf=32,
    )
    mx.eval(row_reference, row_fused, ref_sum, ref_sumsq, got_sum, got_sumsq_scalar)

    np.testing.assert_allclose(
        np.asarray(row_fused),
        np.asarray(row_reference),
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(got_sum),
        np.asarray(ref_sum),
        rtol=1e-5,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(got_sumsq_scalar),
        np.asarray(mx.sum(ref_sumsq)),
        rtol=1e-5,
        atol=1e-1,
    )


def test_mps_reconstruct_prepared_uses_scalar_loss_for_small_fused_path() -> None:
    """Top-level fused MPS loss accumulation must not broadcast scalar loss tiles."""
    mx = pytest.importorskip("mlx.core")
    from quantem.gpu.ssb.compute.mps.engine import (
        _PreparedMpsSSB,
        _corrected_from_dynamic_geometry,
        _phase_sums_from_complex,
        _reconstruct_prepared,
    )

    n = 128
    num_bf = 5
    q_row = np.fft.fftfreq(n, 1.0).astype(np.float32)
    q_col = np.fft.fftfreq(n, 1.0).astype(np.float32)
    kx_np = np.array([-0.17, 0.03, 0.21, -0.09, 0.13], dtype=np.float32)
    ky_np = np.array([0.11, -0.19, 0.07, 0.18, -0.05], dtype=np.float32)
    rng = np.random.default_rng(53)
    real_stack = rng.standard_normal((num_bf, n, n)).astype(np.float32)
    g_qk = mx.fft.rfft2(mx.array(real_stack))
    dc_mask = np.zeros((n, n), dtype=bool)
    dc_mask[0, 0] = True
    q_row_mx = mx.array(q_row, dtype=mx.float32)
    q_col_mx = mx.array(q_col, dtype=mx.float32)
    prepared = _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=q_row_mx[None, :, None],
        qy=q_col_mx[None, None, :],
        q_row=q_row_mx,
        q_col=q_col_mx,
        kx=mx.array(kx_np, dtype=mx.float32),
        ky=mx.array(ky_np, dtype=mx.float32),
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=complex(np.asarray(g_qk[:, 0, 0]).mean()),
        scan_shape=(n, n),
        wavelength=0.0197,
        semiangle_rad=0.0214,
        ang_y_rad=0.0008,
        ang_x_rad=0.0008,
        factor=float(np.pi / 0.0197),
        dc_mask=mx.array(dc_mask),
        num_bf=num_bf,
        alpha_k2=None,
        cos2_k=None,
        sin2_k=None,
        aperture_k=None,
        alpha_m2=None,
        cos2_m=None,
        sin2_m=None,
        ap_m=None,
        alpha_p2=None,
        cos2_p=None,
        sin2_p=None,
        ap_p=None,
    )

    c10 = 0.0
    c12 = 0.0
    phi12 = 0.0
    _, got_loss, got_phase = _reconstruct_prepared(
        prepared,
        C10=c10,
        C12=c12,
        phi12=phi12,
        chunk_bf=4,
        compute_loss=True,
        compute_object=False,
    )
    corrected = _corrected_from_dynamic_geometry(
        prepared,
        start=0,
        stop=num_bf,
        c10=mx.array([c10], dtype=mx.float32),
        c12=mx.array([c12], dtype=mx.float32),
        cos2phi12=mx.array([1.0], dtype=mx.float32),
        sin2phi12=mx.array([0.0], dtype=mx.float32),
    )[0]
    ref_obj = mx.fft.ifft(mx.fft.ifft(corrected, axis=-1), axis=-2)
    ref_sum, ref_sumsq = _phase_sums_from_complex(mx, ref_obj)
    ref_phase = ref_sum / num_bf
    ref_loss = mx.mean(ref_sumsq / num_bf - ref_phase * ref_phase)
    mx.eval(ref_phase, ref_loss)

    np.testing.assert_allclose(
        got_phase,
        np.asarray(ref_phase),
        rtol=1e-5,
        atol=1e-3,
    )
    assert got_loss is not None
    np.testing.assert_allclose(got_loss, float(np.asarray(ref_loss)), atol=1e-4)


def test_mps_fit_defaults_to_exact_objective(monkeypatch) -> None:
    """Small MPS fit should use the full active BF objective by default."""
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("optuna")
    from quantem.gpu.ssb.compute.mps.optimizer import optimize

    clear_calls = 0
    original_clear_cache = mx.clear_cache

    def clear_cache() -> None:
        nonlocal clear_calls
        clear_calls += 1
        original_clear_cache()

    monkeypatch.setattr(mx, "clear_cache", clear_cache)

    rng = np.random.default_rng(61)
    data = rng.poisson(4, size=(8, 8, 12, 12)).astype(np.uint16)
    retained = []

    result = optimize(
        data,
        voltage_kV=300,
        semiangle_mrad=21.4,
        scan_sampling_A=1.0,
        aberrations={"C10": 0.0, "C12": 10.0, "phi12": 0.0},
        bf_intensity_threshold=0.0,
        bf_radius=3,
        chunk_bf=4,
        n_trials=0,
        refine=None,
        _on_complete=lambda *state: retained.append(state),
    )

    assert result.loss is not None
    assert result.num_bf > 0
    assert result.phase.shape == (8, 8)
    assert result.amplitude.shape == (8, 8)
    assert result.backend == "mps"
    assert clear_calls == 1
    assert len(retained) == 1
    prepared, phase, loss, aberrations = retained[0]
    assert prepared.scan_shape == (8, 8)
    assert phase.shape == (8, 8)
    assert phase.dtype == np.float32
    assert loss == result.loss
    assert aberrations == result.aberrations


def test_public_ssb_workflow_returns_only_shared_results(monkeypatch) -> None:
    """The public workflow must not expose a backend-specific result type."""

    pytest.importorskip("mlx.core")
    pytest.importorskip("optuna")
    from quantem.gpu import SSB, SSBResult

    rng = np.random.default_rng(7)
    data = rng.poisson(4, size=(8, 8, 12, 12)).astype(np.uint16)
    workflow = SSB(
        data,
        backend="mps",
        voltage_kV=300,
        semiangle_mrad=21.4,
        scan_sampling_A=1.0,
        bf_intensity_threshold=0.0,
        bf_radius=3,
    )

    optimized = workflow.fit(
        trials=0,
        refinement=None,
        verbose=False,
    )
    result = workflow.reconstruct()

    assert type(optimized) is SSBResult
    assert type(result) is SSBResult
    assert optimized is result
    assert optimized.backend == result.backend == "mps"
    assert workflow._mps_backend._prepared is not None
    assert result.object_wave.dtype == np.complex64
    assert result.phase.dtype == np.float32

    from quantem.gpu.ssb.compute.mps import backend as mps_backend

    def reject_duplicate_reconstruction(*_args, **_kwargs):
        raise AssertionError("the exact final fit phase should be retained")

    monkeypatch.setattr(
        mps_backend,
        "_reconstruct_prepared",
        reject_duplicate_reconstruction,
    )
    phase, loss = workflow.preview(
        optimized.aberrations,
        compute_loss=True,
    )
    np.testing.assert_array_equal(
        phase,
        workflow._mps_backend._fit_preview_phase,
    )
    assert loss == optimized.loss


def test_mps_auto_bf_applies_detected_probe_disk(monkeypatch) -> None:
    """Automatic BF selection must not include positive pixels outside the probe."""
    from quantem.gpu.ssb.compute.mps import engine as mps

    dp = np.ones((9, 9), dtype=np.float32)
    monkeypatch.setattr(mps, "mean_dp", lambda _data: dp)
    monkeypatch.setattr(mps, "auto_probe", lambda _dp: ((4.0, 4.0), 2.2))

    selection = mps._resolve_bf_selection(
        object(),
        threshold=0.0,
        bf_radius=None,
    )

    distance_sq = (selection.rows.astype(np.float32) - 4.0) ** 2 + (
        selection.cols.astype(np.float32) - 4.0
    ) ** 2
    assert np.all(distance_sq <= np.float32(2.2**2))
    assert selection.size == 13
    assert selection.center_row_col == (4.0, 4.0)
    assert selection.radius_px == pytest.approx(2.2)
    assert selection.detected_radius_px == pytest.approx(2.2)


def test_mps_auto_bf_can_reuse_the_computed_mean_dp(monkeypatch) -> None:
    """Exact fit setup should not launch the mean-DP reduction twice."""
    from quantem.gpu.ssb.compute.mps import engine as mps

    dp = np.ones((9, 9), dtype=np.float32)

    def fail_mean_dp(_data):
        raise AssertionError("mean_dp must not run when evidence is supplied")

    monkeypatch.setattr(mps, "mean_dp", fail_mean_dp)
    monkeypatch.setattr(mps, "auto_probe", lambda _dp: ((4.0, 4.0), 2.2))

    selection = mps._resolve_bf_selection(
        object(),
        threshold=0.0,
        bf_radius=None,
        mean_diffraction=dp,
    )

    assert selection.size == 13


def test_mps_sparse_storage_keeps_logical_chunk_boundaries() -> None:
    """Packed active BF planes must retain the original reduction grouping."""
    from quantem.gpu.ssb.compute.mps.engine import _bf_storage_chunks

    class Prepared:
        num_bf = 12
        bf_storage_indices_np = np.asarray([0, 3, 4, 7, 11], dtype=np.int32)

    assert list(_bf_storage_chunks(Prepared(), 4)) == [(0, 2), (2, 4), (4, 5)]


def test_mps_sparse_row_packs_preserve_reduction_boundaries() -> None:
    """Row packs may share storage without merging exact column reductions."""
    from quantem.gpu.ssb.compute.mps.engine import _bf_storage_chunk_packs

    class Prepared:
        num_bf = 12
        bf_storage_indices_np = np.asarray([0, 3, 4, 7, 11], dtype=np.int32)

    assert _bf_storage_chunk_packs(Prepared(), 4, 3) == [
        [(0, 2)],
        [(2, 4), (4, 5)],
    ]


@pytest.mark.skipif(
    not os.environ.get("QUANTEM_GPU_SSB_MASTER")
    or not os.environ.get("QUANTEM_GPU_SSB_REFERENCE_NPZ"),
    reason=(
        "set QUANTEM_GPU_SSB_MASTER and QUANTEM_GPU_SSB_REFERENCE_NPZ "
        "for real-data CUDA/MPS SSB reference reference agreement"
    ),
)
def test_mps_ssb_fixed_aberration_matches_cuda_reference() -> None:
    """Compare MPS SSB fixed-aberration output against a CUDA reference artifact."""
    from quantem.gpu.io import load
    from quantem.gpu.ssb.compute.mps.engine import reconstruct

    master = Path(os.environ["QUANTEM_GPU_SSB_MASTER"])
    reference_path = Path(os.environ["QUANTEM_GPU_SSB_REFERENCE_NPZ"])
    if not master.exists():
        pytest.skip(f"master not available: {master}")
    if not reference_path.exists():
        pytest.skip(f"reference not available: {reference_path}")

    reference = np.load(reference_path)
    reference_meta = json.loads(str(reference["meta"]))
    result = reconstruct(
        load(str(master), backend="mps", verbose=False).data,
        voltage_kV=300,
        semiangle_mrad=21.4,
        scan_sampling_A=1.0,
        C10=0.0,
        C12=50.0,
        phi12=0.0,
        bf_radius=5,
        chunk_bf=16,
        compute_loss=True,
        verbose=False,
    )

    assert result.num_bf == int(reference_meta["num_bf"])
    assert np.allclose(result.bf_center, reference_meta["bf_center"], atol=1e-4)
    if reference_meta.get("loss_full") is not None:
        np.testing.assert_allclose(
            float(result.loss),
            float(reference_meta["loss_full"]),
            rtol=LOSS_RTOL,
            atol=LOSS_ATOL,
        )

    # Compare phase as a unit phasor so the equivalent -pi/+pi representation
    # does not look like scientific drift. All other error must stay inside the
    # shared float32 backend budget.
    reference_phase = np.asarray(reference["phase"], dtype=np.float32)
    result_phase = np.asarray(result.phase, dtype=np.float32)
    np.testing.assert_allclose(
        np.exp(1j * result_phase).astype(np.complex64),
        np.exp(1j * reference_phase).astype(np.complex64),
        rtol=PHASE_RTOL,
        atol=PHASE_ATOL,
    )
