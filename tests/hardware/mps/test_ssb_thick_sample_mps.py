"""Thick-sample SSB (sample tilt + thickness) on MPS / MLX.

Top: the public workflow recovers a known sample tilt from a simulated tilted crystal and gives no tilt for the untilted
control (same data and bounds as the CUDA test). Bottom: the thick path reduces exactly to standard MPS SSB when the depth
weighting is off.

The simulated crystals are abTEM 4D-STEM of BaTiO3 [001], 15.2 nm, 300 kV, 30 mrad, 64 x 64 scan at 0.25 A (see
``tests/hardware/cuda/test_ssb_thick_sample.py::_simulated_crystal``). Simulating them on a Mac is slow, so this test reads
them from ``QUANTEM_THICK_SIM_DIR`` (a directory holding ``tilted_3_-4_t15/`` and ``untilted_t15/``, each with ``data.npy``
and ``meta.json`` carrying ``det_mrad``); it skips when that is not set.
"""

import gc
import json
import math
import os
import platform
import weakref
from pathlib import Path

import numpy as np
import pytest


def _require_mps():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("MPS SSB requires Apple silicon")
    pytest.importorskip("mlx.core")


def _session(data, det_mrad, scan_A=0.25):
    from quantem.gpu import SSB

    return SSB(data, backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(scan_A, scan_A), detector_sampling_mrad=(det_mrad, det_mrad),
                          scan_detector_rotation_deg=0.0)


def _simulated_crystal(name):
    root = os.environ.get("QUANTEM_THICK_SIM_DIR")
    if not root or not (Path(root) / name / "data.npy").is_file():
        pytest.skip("set QUANTEM_THICK_SIM_DIR to the simulated tilted / untilted BaTiO3 runs")
    folder = Path(root) / name
    meta = json.loads((folder / "meta.json").read_text())
    # abTEM intensities are normalised (frame sum ~1). The host mean_dp used for BF selection sums in uint64
    # (detector/workflow.py), which truncates sub-unit floats to 0 and selects no BF pixels, so express them as a dose
    # of 1e6 electrons per frame; a uniform scale changes neither the phase nor the location of the fit optimum.
    return np.load(folder / "data.npy") * np.float32(1e6), float(meta["det_mrad"])


@pytest.mark.parametrize(("name", "sample_tilt_mrad"), [("tilted_3_-4_t15", (3.0, -4.0)), ("untilted_t15", (0.0, 0.0))])
def test_fit_tilt_recovers_known_tilt(name, sample_tilt_mrad):
    _require_mps()
    data, det_mrad = _simulated_crystal(name)
    ssb = _session(data, det_mrad)
    assert ssb.supports_tilt
    result = ssb.find_aberrations(search="standard", tilt=True, verbose=False)
    # 1 mrad is the scan/detector sampling limit of this simulation (same bound as the CUDA test)
    assert abs(result.sample_tilt_mrad[0] - sample_tilt_mrad[0]) < 1.0
    assert abs(result.sample_tilt_mrad[1] - sample_tilt_mrad[1]) < 1.0
    assert result.tilt_fit_gain > 1.2      # the thick model explains the thick data better than standard SSB


# ---


def _poisson_disk_session():
    rng = np.random.default_rng(7)
    counts = rng.poisson(40.0, size=(128, 128, 32, 32)).astype(np.uint16)
    rr, cc = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rr, cc) < 10] += 200      # bright-field disk
    return _session(counts, det_mrad=3.0, scan_A=0.3)


def test_joint_fit_rejects_scan_with_no_fitting_frequencies():
    """A coarse scan cannot return arbitrary tilt from an empty objective."""
    _require_mps()
    counts = np.ones((128, 128, 32, 32), dtype=np.uint16)
    rr, cc = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rr, cc) < 10] += 200
    with (
        _session(counts, det_mrad=3.0, scan_A=5.0) as ssb,
        pytest.raises(ValueError, match="acquire a finer scan"),
    ):
        ssb.find_aberrations(search="standard", tilt=True, num_trials=8, verbose=False)


def test_batch_fit_matches_reference_fit(monkeypatch):
    """The fused Metal batch objective (half-plane band, shared geometry, skipped non-overlap pairs) equals ``thick_fit``."""
    _require_mps()
    from quantem.gpu.ssb.mps import thick_sample
    from quantem.gpu.ssb.mps.thick_sample import thick_fit, thick_fit_batch

    ssb = _poisson_disk_session()
    backend = ssb._backend()
    backend.cache_rotation(0.0)
    prepared = backend._prepared
    rng = np.random.default_rng(3)
    rows = np.column_stack([rng.uniform(-300, 300, 11), rng.uniform(0, 200, 11), rng.uniform(-1.5, 1.5, 11),
                            rng.uniform(-25, 25, 11), rng.uniform(-25, 25, 11), rng.uniform(20, 600, 11)])
    rows[0, 5], rows[1, 5] = 0.0, 400.0      # zero-depth control, thick crystal, and a partial execution tile
    reference = np.array([thick_fit(prepared, C10=r[0], C12=r[1], phi12=r[2], sample_tilt_mrad=(r[3], r[4]), thickness=r[5]) for r in rows])
    np.testing.assert_allclose(thick_fit_batch(prepared, rows), reference, rtol=1e-4)
    # Diagnostic batches preserve the existing fused kernel's exact values.
    np.testing.assert_array_equal(backend.sample_agreement(rows, (0.2, 0.9)), thick_fit_batch(prepared, rows))
    # The former eight-candidate execution and the four-candidate execution
    # must give the same values, including both correction models and tails.
    current = thick_fit_batch(prepared, rows)
    # Reading each execution tile immediately is the previous scheduling
    # policy. Queuing those tiles must preserve every objective bit.
    monkeypatch.setattr(thick_sample, "_FIT_PENDING_BYTES", 0)
    np.testing.assert_array_equal(thick_fit_batch(prepared, rows), current)
    monkeypatch.setattr(thick_sample, "_FIT_DISPATCH_BATCH", 8)
    np.testing.assert_array_equal(thick_fit_batch(prepared, rows), current)
    # Evaluate both sides even outside their aperture, as in the original
    # fused objective. Skipping zero-weight terms must preserve every bit.
    mx = prepared.mx
    unpruned = mx.fast.metal_kernel(
        name="ssb_thick_fit_unpruned_test",
        input_names=["G", "band_flat", "band_qx", "band_qy", "kx", "ky",
                     "trial", "scalars", "probe_geometry", "probe_chi"],
        output_names=["numer_real", "numer_imag", "denom"],
        source=thick_sample._THICK_FIT_BATCH_SOURCE.replace(
            "if (a1 != 0.0f)", "if (true)"
        ).replace("if (a2 != 0.0f)", "if (true)"),
        header=thick_sample._THICK_FIT_BATCH_HEADER,
    )
    monkeypatch.setattr(thick_sample, "_thick_fit_batch_kernel", lambda: unpruned)
    np.testing.assert_array_equal(thick_fit_batch(prepared, rows), current)


def test_zero_depth_weighting_is_standard_ssb():
    """Thickness below the sinc cutoff gives every weight exactly 1: the thick path must reproduce standard MPS SSB."""
    _require_mps()
    ssb = _poisson_disk_session()
    ssb.reconstruct(aberrations={"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0})
    for aberrations in ({"C10_nm": 30.0, "C12_nm": 5.0, "phi12_deg": math.degrees(0.3)}, {"C10_nm": -40.0, "C12_nm": 12.0, "phi12_deg": math.degrees(-1.0)}):
        standard, standard_loss = ssb.preview(aberrations)
        thick, thick_loss = ssb.preview(aberrations, sample_tilt_mrad=(5.0, -3.0), depth_spread_nm=1e-9)
        np.testing.assert_allclose(thick, standard, atol=5e-6)
        assert abs(thick_loss - standard_loss) <= 1e-5 * abs(standard_loss)


@pytest.mark.parametrize("batch_size", [28, 39, 84])
def test_newton_candidate_tiles_preserve_objectives(monkeypatch, batch_size):
    """Parallel Newton tiles preserve serial values, tails and bounded batches."""
    _require_mps()
    from quantem.gpu.ssb.mps import thick_sample

    rng = np.random.default_rng(47)
    rows = rng.uniform(
        [-300, 0, -1.5, -25, -25, 0],
        [300, 200, 1.5, 25, 25, 600],
        (batch_size, 6),
    )
    rows[0, 5] = 0.0
    with _poisson_disk_session() as ssb:
        backend = ssb._backend()
        backend.cache_rotation(0.0)
        prepared = backend._prepared
        with monkeypatch.context() as serial:
            serial.setattr(thick_sample, "_FIT_PARALLEL_SHAPES", ())
            expected = thick_sample.thick_fit_batch(prepared, rows)
        np.testing.assert_array_equal(thick_sample.thick_fit_batch(prepared, rows), expected)

        # Force multiple dispatches and a partial last tile without changing
        # either BF accumulation order or the complete candidate sequence.
        n_band = len(thick_sample._band(prepared, (0.2, 0.9))[3])
        chunks = (int(prepared.g_qk.shape[0]) + thick_sample._K_CHUNK - 1) // thick_sample._K_CHUNK
        monkeypatch.setattr(thick_sample, "_FIT_PENDING_BYTES", (chunks + 1) * n_band * 3 * 4 * 8)
        np.testing.assert_array_equal(thick_sample.thick_fit_batch(prepared, rows), expected)


def test_fused_thick_preview_matches_reference_model():
    """The interactive thick path (depth weights inside the fused row kernel) equals the MLX element-wise reference model
    at real thickness and tilt, where the weights are far from 1.

    Tolerances are the measured float32 floor of each case. At 3 nm defocus both paths sit ~1e-8 from a float64
    evaluation of the model. At 40 nm defocus, 60 nm thickness and 23 mrad tilt chi reaches ~1e2 rad, and float32 chi
    alone puts both paths 2.5e-6 (reference) to 7e-6 (fused) from float64, on phases of ~5e-3.
    """
    _require_mps()
    from quantem.gpu.ssb.mps.thick_sample import (
        reconstruct_thick,
        reconstruct_thick_reference,
    )

    ssb = _poisson_disk_session()
    backend = ssb._backend()
    backend.cache_rotation(0.0)
    prepared = backend._prepared
    for kw, atol in ((dict(C10=30.0, C12=5.0, phi12=0.3, sample_tilt_mrad=(5.0, -3.0), thickness=150.0), 5e-8),
                     (dict(C10=-400.0, C12=120.0, phi12=-1.0, sample_tilt_mrad=(-20.0, 12.0), thickness=600.0), 1e-5)):
        fused, fused_loss = reconstruct_thick(prepared, **kw)
        reference, reference_loss = reconstruct_thick_reference(prepared, **kw)
        np.testing.assert_allclose(fused, reference, rtol=0, atol=atol)
        assert abs(fused_loss - reference_loss) <= 1e-6 * abs(reference_loss)


def test_drag_subset_context_standard_and_thick():
    """``preview_context`` swaps both previews onto the deterministic BF subset: the full-size subset reproduces the full
    preview exactly, and on a reduced subset the thick path at zero depth weighting still equals the standard path."""
    _require_mps()
    ssb = _poisson_disk_session()
    aberrations = {"C10_nm": 30.0, "C12_nm": 5.0, "phi12_deg": math.degrees(0.3)}
    full, _ = ssb.preview(aberrations, compute_loss=False)
    everything = ssb.preview_context(ssb.num_bright_field_pixels)
    same, _ = ssb.preview(aberrations, compute_loss=False, context=everything)
    np.testing.assert_array_equal(same, full)
    context = ssb.preview_context(ssb.num_bright_field_pixels // 4)
    assert context.num_bright_field_pixels == ssb.num_bright_field_pixels // 4
    standard, standard_loss = ssb.preview(aberrations, context=context)
    thick, thick_loss = ssb.preview(aberrations, context=context,
                                    sample_tilt_mrad=(5.0, -3.0), depth_spread_nm=1e-9)
    np.testing.assert_allclose(thick, standard, atol=5e-6)
    assert abs(thick_loss - standard_loss) <= 1e-5 * abs(standard_loss)
    assert np.abs(standard - full).max() > 1e-3      # the subset really is fewer BF pixels
    after, _ = ssb.preview(aberrations, compute_loss=False)
    np.testing.assert_array_equal(after, full)      # leaving the context restores the full evidence


def test_drag_subset_close_restores_full_evidence_and_releases_the_subset():
    """Viewers release a drag preview context with ``close()``, even while it is entered, as on CUDA."""
    _require_mps()
    ssb = _poisson_disk_session()
    aberrations = {"C10_nm": 30.0, "C12_nm": 5.0, "phi12_deg": math.degrees(0.3)}
    full, _ = ssb.preview(aberrations, compute_loss=False)
    context = ssb.preview_context(ssb.num_bright_field_pixels // 4)
    subset = weakref.ref(context.__enter__()._subset)
    context.close()
    gc.collect()
    assert subset() is None
    after, _ = ssb.preview(aberrations, compute_loss=False)
    np.testing.assert_array_equal(after, full)
    context.close()
