"""The phase-variance standard fit's search (``ssb.phase_variance_fit``), shared by the backends, on analytic losses."""

import math

import numpy as np
import pytest


def test_nelder_mead_keeps_first_vertex_for_tied_losses() -> None:
    """Stable tie ordering keeps a flat seeded fit at its input point."""
    from quantem.gpu.ssb.phase_variance_fit import _nelder_mead

    start = {"C10": -20.0, "C12": 22.0, "phi12": 0.46}
    first = _nelder_mead(start, 1.0, lambda params: 1.0, lock=set(), max_iter=4)
    second = _nelder_mead(start, 1.0, lambda params: 1.0, lock=set(), max_iter=4)

    assert first[0] == start
    assert first == second


def test_search_and_locked_refinement_on_a_quadratic_loss() -> None:
    """A seeded fit finds the minimum of a smooth loss, repeats bit for bit, and holds a locked coefficient."""
    pytest.importorskip("optuna")
    from quantem.gpu.ssb.phase_variance_fit import fit_phase_variance

    truth = np.array([-85.0, 30.0, math.radians(25.0)])

    def evaluate_batch(C10, C12, phi12):
        astigmatism = C12 * np.exp(2j * phi12.astype(np.float64)) - truth[1] * np.exp(2j * truth[2])
        return ((C10 - truth[0]) / 100.0) ** 2 + np.abs(astigmatism / 20.0) ** 2

    first = fit_phase_variance(evaluate_batch, num_trials=60, seed=3)
    second = fit_phase_variance(evaluate_batch, num_trials=60, seed=3)
    held = fit_phase_variance(evaluate_batch, num_trials=30, seed=3, search_ranges={"C12_A": 0.0},
                              refine_lock=["C12"])

    assert first["aberrations"]["C10"] == pytest.approx(truth[0], abs=0.5)
    assert first["aberrations"]["C12"] == pytest.approx(truth[1], abs=0.5)
    assert first["trial_records"] == second["trial_records"]
    assert [record["stage"] for record in first["trial_records"][:2]] == ["initial", "search"]
    assert held["aberrations"]["C12"] == 0.0
    assert held["aberrations"]["C10"] == pytest.approx(truth[0], abs=0.5)
