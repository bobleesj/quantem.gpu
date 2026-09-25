"""Thick-sample SSB (sample tilt + thickness) on MPS / MLX.

Top: the public workflow recovers a known sample tilt from a simulated tilted crystal and gives no tilt for the untilted
control (same data and bounds as the CUDA test). Bottom: the thick path reduces exactly to standard MPS SSB when the depth
weighting is off.

The simulated crystals are abTEM 4D-STEM of BaTiO3 [001], 15.2 nm, 300 kV, 30 mrad, 64 x 64 scan at 0.25 A (see
``tests/hardware/cuda/test_ssb_thick_sample.py::_simulated_crystal``). Simulating them on a Mac is slow, so this test reads
them from ``QUANTEM_THICK_SIM_DIR`` (a directory holding ``tilted_3_-4_t15/`` and ``untilted_t15/``, each with ``data.npy``
and ``meta.json`` carrying ``det_mrad``); it skips when that is not set.
"""

import json
import os
import platform
from pathlib import Path

import numpy as np
import pytest


def _require_mps():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("MPS SSB requires Apple silicon")
    pytest.importorskip("mlx.core")


def _session(data, det_mrad, scan_A=0.25):
    from quantem.gpu import SSB

    return SSB.from_array(data, backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=scan_A, det_sampling=det_mrad,
                          rotation_angle_deg=0.0)


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


@pytest.mark.parametrize(("name", "tilt_mrad"), [("tilted_3_-4_t15", (3.0, -4.0)), ("untilted_t15", (0.0, 0.0))])
def test_fit_tilt_recovers_known_tilt(name, tilt_mrad):
    _require_mps()
    data, det_mrad = _simulated_crystal(name)
    ssb = _session(data, det_mrad)
    assert ssb.supports_tilt
    result = ssb.fit(tilt=True, verbose=False)
    # 1 mrad is the scan/detector sampling limit of this simulation (same bound as the CUDA test)
    assert abs(result.tilt_mrad[0] - tilt_mrad[0]) < 1.0
    assert abs(result.tilt_mrad[1] - tilt_mrad[1]) < 1.0
    assert result.tilt_fit_gain > 1.2      # the thick model explains the thick data better than standard SSB


# ---


def _poisson_disk_session():
    rng = np.random.default_rng(7)
    counts = rng.poisson(40.0, size=(128, 128, 32, 32)).astype(np.uint16)
    rr, cc = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rr, cc) < 10] += 200      # bright-field disk
    return _session(counts, det_mrad=3.0, scan_A=0.3)


def test_batch_fit_matches_reference_fit():
    """The fused Metal batch objective (half-plane band, shared geometry, skipped non-overlap pairs) equals ``thick_fit``."""
    _require_mps()
    from quantem.gpu.ssb.backends.mps._thick_sample import thick_fit, thick_fit_batch

    ssb = _poisson_disk_session()
    backend = ssb._backend_protocol
    backend.cache_rotation(0.0)
    prepared = backend._prepared
    rng = np.random.default_rng(3)
    rows = np.column_stack([rng.uniform(-300, 300, 11), rng.uniform(0, 200, 11), rng.uniform(-1.5, 1.5, 11),
                            rng.uniform(-25, 25, 11), rng.uniform(-25, 25, 11), rng.uniform(20, 600, 11)])
    rows[0, 5], rows[1, 5] = 0.0, 400.0      # standard SSB and a thick crystal; 11 rows = one full batch of 8 plus 3
    reference = np.array([thick_fit(prepared, C10=r[0], C12=r[1], phi12=r[2], tilt_mrad=(r[3], r[4]), thickness=r[5]) for r in rows])
    np.testing.assert_allclose(thick_fit_batch(prepared, rows), reference, rtol=1e-4)


def test_zero_depth_weighting_is_standard_ssb():
    """Thickness below the sinc cutoff gives every weight exactly 1: the thick path must reproduce standard MPS SSB."""
    _require_mps()
    ssb = _poisson_disk_session()
    ssb.reconstruct({"C10": 0.0, "C12": 0.0, "phi12": 0.0})
    for aberrations in ({"C10": 30.0, "C12": 5.0, "phi12": 0.3}, {"C10": -40.0, "C12": 12.0, "phi12": -1.0}):
        standard, standard_loss = ssb.preview(aberrations)
        thick, thick_loss = ssb.preview(aberrations, tilt_mrad=(5.0, -3.0), depth_spread_nm=1e-9)
        np.testing.assert_allclose(thick, standard, atol=5e-6)
        assert abs(thick_loss - standard_loss) <= 1e-5 * abs(standard_loss)


def test_fused_thick_preview_matches_reference_model():
    """The interactive thick path (depth weights inside the fused row kernel) equals the MLX element-wise reference model
    at real thickness and tilt, where the weights are far from 1.

    Tolerances are the measured float32 floor of each case. At 3 nm defocus both paths sit ~1e-8 from a float64
    evaluation of the model. At 40 nm defocus, 60 nm thickness and 23 mrad tilt chi reaches ~1e2 rad, and float32 chi
    alone puts both paths 2.5e-6 (reference) to 7e-6 (fused) from float64, on phases of ~5e-3.
    """
    _require_mps()
    from quantem.gpu.ssb.backends.mps._thick_sample import reconstruct_thick, reconstruct_thick_reference

    ssb = _poisson_disk_session()
    backend = ssb._backend_protocol
    backend.cache_rotation(0.0)
    prepared = backend._prepared
    for kw, atol in ((dict(C10=30.0, C12=5.0, phi12=0.3, tilt_mrad=(5.0, -3.0), thickness=150.0), 5e-8),
                     (dict(C10=-400.0, C12=120.0, phi12=-1.0, tilt_mrad=(-20.0, 12.0), thickness=600.0), 1e-5)):
        fused, fused_loss = reconstruct_thick(prepared, **kw)
        reference, reference_loss = reconstruct_thick_reference(prepared, **kw)
        np.testing.assert_allclose(fused, reference, rtol=0, atol=atol)
        assert abs(fused_loss - reference_loss) <= 1e-6 * abs(reference_loss)


def test_drag_subset_context_standard_and_thick():
    """``preview_context`` swaps both previews onto the deterministic BF subset: the full-size subset reproduces the full
    preview exactly, and on a reduced subset the thick path at zero depth weighting still equals the standard path."""
    _require_mps()
    ssb = _poisson_disk_session()
    aberrations = {"C10": 30.0, "C12": 5.0, "phi12": 0.3}
    full, _ = ssb.preview(aberrations, compute_loss=False)
    everything = ssb.preview_context(ssb.num_bf)
    same, _ = ssb.preview(aberrations, compute_loss=False, context=everything)
    np.testing.assert_array_equal(same, full)
    context = ssb.preview_context(ssb.num_bf // 4)
    assert context.num_bf == ssb.num_bf // 4
    standard, standard_loss = ssb.preview(aberrations, context=context)
    thick, thick_loss = ssb.preview(aberrations, context=context,
                                    tilt_mrad=(5.0, -3.0), depth_spread_nm=1e-9)
    np.testing.assert_allclose(thick, standard, atol=5e-6)
    assert abs(thick_loss - standard_loss) <= 1e-5 * abs(standard_loss)
    assert np.abs(standard - full).max() > 1e-3      # the subset really is fewer BF pixels
    after, _ = ssb.preview(aberrations, compute_loss=False)
    np.testing.assert_array_equal(after, full)      # leaving the context restores the full evidence
