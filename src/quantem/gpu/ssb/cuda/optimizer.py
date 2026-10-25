"""Aberration fits on CUDA: the standard fit (tilt=False) on either objective, and the joint fit (tilt=True).

The default objective of both fits is the least-squares agreement of the SSB model with G(q, k) (``ThickSample``'s fused
``thick_fit_batch_kernel``), searched with the device-resident TPE and trust-region Newton of ``ssb._resident_search``,
the same code as on MPS; the standard fit holds the sample tilt and depth spread at 0, where the model is ordinary SSB
exactly. ``objective="phase_variance"`` minimizes the phase variance with the backend-neutral search of
``ssb.phase_variance_fit`` (MPS's ``optimize_phase_variance`` search), on the fixed-order full reconstruction (``SSBEngine.reconstruct_with_loss``), so seeded fits get the
same loss bits on every evaluation and a fit, a reconstruction and a preview report the same loss for the same
aberrations.
"""

import math

import numpy as np

from quantem.gpu.ssb.cuda.engine import SSBEngine
from quantem.gpu.ssb.mps.optimizer import standard_bounds
from quantem.gpu.ssb.phase_variance_fit import fit_phase_variance
from quantem.gpu.ssb.thick_sample_fit import fit_sample_search

# where a zero-trial refinement starts when the session has no aberrations (as on MPS)
_DEFAULT_START = {"C10": 0.0, "C12": 50.0, "phi12": 0.0}


def fit_standard(
    accel: SSBEngine,
    *,
    aberrations: dict | None,
    search_ranges: dict | None,
    num_trials: int,
    refine: str | None,
    refine_lock: list[str] | None,
    seed: int,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
) -> dict[str, object]:
    """Standard fit (tilt=False) on the agreement objective: ``fit_standard_resident`` with depth spread 0.

    ``search_ranges`` (engine keys; a value holds that coefficient in every trial) bounds the search; ``refine``
    ("newton", "nelder-mead" or None) polishes from the best distinct trials, and from ``aberrations`` (engine units)
    when given; ``refine_lock`` (engine names) holds coefficients at the search's value through Nelder-Mead. The start
    is clamped into the search box because a held coefficient must be every evaluated point's value, the start
    included. ``band_inv_A`` is the joint fit's band: below 0.2 1/A the probe-overlap geometry dominates G, above
    0.9 1/A the lattice signal ends. Returns ``fit_standard_resident``'s record (the fitted C10, C12, phi12, the
    agreement ``fit`` and the trials) with the band.
    """
    from quantem.gpu.ssb._resident_search import fit_standard_resident

    lower, upper = standard_bounds(search_ranges)
    start = None
    if aberrations:
        start = [float(aberrations[name]) for name in ("C10", "C12", "phi12")]
    elif int(num_trials) == 0:
        start = [_DEFAULT_START[name] for name in ("C10", "C12", "phi12")]
    if start is not None:
        # C12 is a magnitude and phi12 has period pi: a start outside (-90, 90] degrees (typed, or a phase-variance
        # fit's) means the wrapped azimuth, which clamping alone would replace by the box edge
        start[1] = abs(start[1])
        if lower[2] <= -math.pi / 2 + 1e-9 and upper[2] >= math.pi / 2 - 1e-9:
            start[2] = (start[2] + math.pi / 2) % math.pi - math.pi / 2
        # the start obeys the search box: a held coefficient is every trial's and the result's value
        start = [min(max(value, low), high) for value, low, high in zip(start, lower, upper, strict=True)]
    lock = tuple(sorted({"C10": 0, "C12": 1, "phi12": 2}[name] for name in (refine_lock or ())))
    fit = fit_standard_resident(
        lambda trial: accel.thick.fit_resident(trial, band_inv_A), backend="cuda", lower=lower, upper=upper,
        trials=int(num_trials), seed=int(seed), polish=refine, lock=lock, start=start,
    )
    return {**fit, "band_inv_A": list(band_inv_A)}


def fit_standard_phase_variance(accel: SSBEngine, **options) -> dict[str, object]:
    """Standard fit (tilt=False) on the phase-variance objective: ``phase_variance_fit.fit_phase_variance`` with
    the engine's loss. ``options`` are that function's (engine units). The search is backend-neutral so that a
    seeded fit visits the same trials on every backend that evaluates it."""

    def evaluate_batch(C10: np.ndarray, C12: np.ndarray, phi12: np.ndarray) -> np.ndarray:
        """Losses of float32 candidates, one fixed-order full reconstruction each (the engine has no batched loss)."""
        return np.asarray([accel.reconstruct_with_loss(float(c10), float(c12), float(phi12_rad))[1]
                           for c10, c12, phi12_rad in zip(C10, C12, phi12, strict=True)], dtype=np.float64)

    return fit_phase_variance(evaluate_batch, **options)


# =========================================================================
#  Thick-sample fit: sample tilt and thickness with the aberrations
# =========================================================================

def fit_sample(
    accel: SSBEngine,
    *,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
    **options,
) -> dict[str, object]:
    """Fit C10, C12, phi12, sample tilt and thickness by maximising ``ThickSample.fit`` (search: ``ssb.thick_sample_fit``)."""
    if options.get("polish", "newton") == "newton" or options.get("polish_starts", 3) == 0:
        from quantem.gpu.ssb._resident_search import fit_resident

        if options.get("polish_starts", 3) == 0:
            options = {**options, "polish": None}
        from quantem.gpu.ssb import _exact_newton

        derivatives = None
        if _exact_newton.EXACT_NEWTON:
            from quantem.gpu.ssb.cuda.exact_derivatives import LowOrderDerivatives

            derivatives = LowOrderDerivatives(accel, band_inv_A)
        return fit_resident(lambda trial: accel.thick.fit_resident(trial, band_inv_A),
                            backend="cuda", band_inv_A=band_inv_A, derivatives=derivatives, **options)
    return fit_sample_search(lambda c10, c12, phi12, tilt, thickness: accel.thick.fit(c10, c12, phi12, tilt, thickness, band_inv_A),
                             objective_batch=lambda rows: accel.thick.fit_batch(rows, band_inv_A),
                             batch_polish=True,
                             refinement_objective_batch=lambda rows: accel.thick.fit_batch(rows, band_inv_A, independent=True),
                             band_inv_A=band_inv_A, **options)
