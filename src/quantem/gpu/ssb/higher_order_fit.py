"""Fit coma, trefoil and spherical aberration with the joint thick-sample SSB model (``find_aberrations(higher_order=...)``).

The joint fit (``thick_sample_fit``, ``_resident_search``) searches C10, C12, phi12, the sample tilt and the depth
spread. This refinement starts from that fit, with the higher-order terms at 0, and maximizes the same model
agreement over all of them together by trust-region Newton steps on finite-difference models (the joint fit's
``_newton_design``, ``_newton_model`` and ``_bounded_step``). It needs no search of its own: on simulated crystals
it reached the true coma (100 nm), trefoil (100 nm) and C30 (10 um) from a joint fit whose tilt was 8 mrad off.

Coordinates z (engine units): C10, C12 (cos 2 phi12, sin 2 phi12), tilt row and col (mrad), depth spread, then
for each fitted term the Cartesian vector C (cos m phi, sin m phi), or C alone for a round term (C30). Vectors are
smooth through C = 0, where the azimuth is undefined.

Why not a search: the agreement is a smooth function of the higher orders with one maximum near the joint fit in
every simulation tried, and adding five dimensions to the joint search's TPE would cost far more trials than the
Newton polish costs evaluations.
"""

import math
import time
from collections.abc import Callable, Sequence

import numpy as np

from quantem.gpu.ssb.higher_order import ORDERS
from quantem.gpu.ssb.thick_sample_fit import (
    DEPTH_SPREAD_RANGE_A,
    _bounded_step,
    _newton_design,
    _newton_model,
)

# The terms validated on simulated crystals (C21 coma, C23 trefoil, C30 spherical aberration). Slots follow
# ``higher_order.ORDERS``; a fit evaluates the polynomial up to the highest fitted slot.
FITTABLE = {"C21": 2, "C23": 3, "C30": 4}
# Finite-difference steps (engine units: A, A, A, mrad, mrad, A) of the joint fit's Newton, then per higher-order
# coordinate: 40 A of coma or trefoil and 1000 A of C30 change the phase at a 30 mrad aperture edge by 0.1 rad.
_LOW_ORDER_STEP = np.array([2.0, 2.0, 2.0, 0.25, 0.25, 5.0])
_STEP = {"C21": 40.0, "C23": 40.0, "C30": 1000.0}
# Search box of the higher-order coordinates (A): 300 nm of coma or trefoil, 20 um of C30.
_LIMIT = {"C21": 3000.0, "C23": 3000.0, "C30": 2.0e5}
_TOLERANCE_FRACTION = 1.0 / 40.0         # the joint fit's tolerance: 0.05 A for a 2 A step


def validate_terms(terms: Sequence[str] | None) -> tuple[str, ...]:
    """The fitted higher-order terms in slot order; raises for an unknown or repeated name."""
    if not terms:
        return ()
    if isinstance(terms, str):
        terms = (terms,)
    names = tuple(terms)
    unknown = [name for name in names if name not in FITTABLE]
    if unknown or len(set(names)) != len(names):
        raise ValueError(
            "higher_order names the aberrations to fit with the joint model: any of 'C21' (coma), 'C23' "
            f"(trefoil) and 'C30' (spherical aberration), each once; got {list(names)!r}."
        )
    return tuple(sorted(names, key=FITTABLE.get))


def _width(terms: tuple[str, ...]) -> int:
    return sum(1 if ORDERS[FITTABLE[name]][1] == 0 else 2 for name in terms)


def kernel_rows(z: np.ndarray, terms: tuple[str, ...]) -> np.ndarray:
    """Coordinates z -> agreement rows: (depth, tilt row, tilt col) and the Cartesian Krivanek coefficients."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    slots = 1 + max(FITTABLE[name] for name in terms)
    rows = np.zeros((len(z), 3 + 2 * slots))
    rows[:, 0] = np.abs(z[:, 5])
    rows[:, 1:3] = z[:, 3:5]
    rows[:, 3] = z[:, 0]                       # C10, round
    rows[:, 5:7] = z[:, 1:3]                   # C12 vector
    column = 6
    for name in terms:
        slot = FITTABLE[name]
        if ORDERS[slot][1] == 0:
            rows[:, 3 + 2 * slot] = z[:, column]
            column += 1
        else:
            rows[:, 3 + 2 * slot:5 + 2 * slot] = z[:, column:column + 2]
            column += 2
    return rows


def to_coordinates(C10, C12, phi12, tilt_mrad, depth_spread_A, higher: dict[str, tuple[float, float]],
                   terms: tuple[str, ...]) -> np.ndarray:
    """Engine-unit parameters (A, rad, mrad) -> z. ``higher`` maps a term to (magnitude A, azimuth rad)."""
    z = [C10, C12 * math.cos(2 * phi12), C12 * math.sin(2 * phi12), tilt_mrad[0], tilt_mrad[1], depth_spread_A]
    for name in terms:
        magnitude, angle = higher.get(name, (0.0, 0.0))
        m = ORDERS[FITTABLE[name]][1]
        z += [magnitude] if m == 0 else [magnitude * math.cos(m * angle), magnitude * math.sin(m * angle)]
    return np.asarray(z, dtype=np.float64)


def from_coordinates(z: np.ndarray, terms: tuple[str, ...]) -> dict[str, float]:
    """z -> engine-unit parameters: C10, C12, phi12, tilt, depth spread and each term's magnitude and azimuth.

    A round term keeps its sign (C30 < 0 is physical); a vector term has a magnitude >= 0 and an azimuth in
    (-pi / m, pi / m], the azimuth's period.
    """
    out = {"C10": float(z[0]), "C12": float(math.hypot(z[1], z[2])), "phi12": 0.5 * math.atan2(z[2], z[1]),
           "tilt_row_mrad": float(z[3]), "tilt_col_mrad": float(z[4]), "depth_spread_A": float(abs(z[5]))}
    column = 6
    for name in terms:
        m = ORDERS[FITTABLE[name]][1]
        if m == 0:
            out[name], out[f"phi{name[1:]}"] = float(z[column]), 0.0
            column += 1
        else:
            out[name] = float(math.hypot(z[column], z[column + 1]))
            out[f"phi{name[1:]}"] = math.atan2(z[column + 1], z[column]) / m
            column += 2
    return out


def num_terms(rows: np.ndarray) -> int:
    """Krivanek terms an agreement row carries: (width - 3) / 2."""
    width = np.shape(rows)[-1]
    if width < 5 or (width - 3) % 2 or (width - 3) // 2 > len(ORDERS):
        raise ValueError(f"A trial row is (thickness, tilt_row, tilt_col) and 2 N Cartesian coefficients; got {width}.")
    return (width - 3) // 2


def packed(rows: np.ndarray) -> np.ndarray:
    """Agreement rows -> the kernels' float32 rows: tilt in rad and each Cartesian coefficient divided by (n + 1)."""
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    terms = num_terms(rows)
    order = np.repeat(np.asarray([n for n, _ in ORDERS[:terms]], dtype=np.float64), 2)
    out = np.empty_like(rows)
    out[:, 0] = rows[:, 0]
    out[:, 1:3] = rows[:, 1:3] * 1e-3
    out[:, 3:] = rows[:, 3:] / (order + 1.0)
    return out.astype(np.float32)


def refine_higher_order(
    agreement: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    terms: tuple[str, ...],
    *,
    tilt: bool,
    tilt_limit_mrad: float = 25.0,
    c10_range: tuple[float, float] = (-300.0, 300.0),
    c12_max: float = 200.0,
    max_iterations: int = 40,
) -> dict[str, object]:
    """Maximise ``agreement(kernel_rows)`` over the low orders and ``terms`` from ``start`` (coordinates z).

    ``tilt=False`` holds the sample tilt and depth spread at the start's values (0 for a standard fit). Each
    iteration evaluates one finite-difference design in one ``agreement`` call: the full Hessian design the first
    time and after a long step (1 + 2 n + n (n - 1) / 2 points; 78 for all five higher-order coordinates and the
    six low-order ones), the gradient and diagonal otherwise (1 + 2 n). Returns the point, its agreement and the
    work done.
    """
    started = time.perf_counter()
    start = np.asarray(start, dtype=np.float64)
    size = len(start)
    step = np.concatenate((_LOW_ORDER_STEP, [_STEP[name] for name in terms for _ in range(_width((name,)))]))
    limits = np.array([_LIMIT[name] for name in terms for _ in range(_width((name,)))])
    lower = np.concatenate(([c10_range[0], -c12_max, -c12_max, -tilt_limit_mrad, -tilt_limit_mrad,
                             DEPTH_SPREAD_RANGE_A[0]], -limits))
    upper = np.concatenate(([c10_range[1], c12_max, c12_max, tilt_limit_mrad, tilt_limit_mrad,
                             DEPTH_SPREAD_RANGE_A[1]], limits))
    free = np.ones(size, dtype=bool)
    if not tilt:
        free[3:6] = False
    index = np.flatnonzero(free)
    # the start sits inside the box (a joint fit's depth spread is never below the floor; a standard fit's 0 is held)
    center = np.clip(start[index], lower[index], upper[index])
    step, tolerance = step[index], step[index] * _TOLERANCE_FRACTION
    lower, upper = lower[index], upper[index]

    def evaluate(points: np.ndarray) -> np.ndarray:
        z = np.repeat(start[None], len(points), axis=0)
        z[:, index] = points
        return np.asarray(agreement(kernel_rows(z, terms)), dtype=np.float64)

    scale = max(abs(float(evaluate(center[None])[0])), 1e-30)
    curvature = np.zeros((len(index), len(index)))
    previous = None
    radius, full, evaluations, passes, converged = 8.0, True, 1, 1, False
    for _ in range(max_iterations):
        design = _newton_design(center, step, full)
        values = evaluate(design) / scale
        evaluations += len(design)
        passes += 1
        if previous is not None and values[0] < previous[1]:
            rejected = float(np.linalg.norm((center - previous[0]) / step))
            center, _, gradient, hessian = previous
            radius = min(radius, rejected) / 4.0
        else:
            gradient, hessian = _newton_model(values, step, curvature)
            moved = math.inf if previous is None else float(np.linalg.norm((center - previous[0]) / step))
            if previous is not None and moved >= 0.99 * radius:
                radius *= 2.0
            curvature = hessian
            full = previous is not None and moved > 2.0
            previous = (center.copy(), float(values[0]), gradient, hessian)
        proposal = np.clip(center + _bounded_step(center, gradient, hessian, step, radius, lower, upper), lower, upper)
        if np.all(np.abs(proposal - center) < tolerance):
            converged = True
            break
        center = proposal
    z = start.copy()
    z[index] = previous[0]
    # the curvature of the agreement at the returned point, in the free coordinates (for standard_errors)
    return {"z": z, "agreement": previous[1] * scale, "start_agreement": scale, "evaluations": evaluations,
            "passes": passes, "converged": converged, "seconds": time.perf_counter() - started,
            "hessian": previous[3] * scale, "free": index}


def standard_errors(fit: dict[str, object], terms: tuple[str, ...], noise_variance: float) -> dict[str, float]:
    """Standard errors (A) of the fitted terms from the agreement's curvature at the polished point.

    With complex Gaussian noise of variance ``noise_variance`` on every G(q, k) (the counts at k for Poisson counts:
    G(0, k)), the profile log-likelihood of the model is the agreement over the independent half of the q plane divided
    by the variance, A / (2 sigma^2) for the full-plane sum A, so the covariance of the coordinates is
    2 sigma^2 (-Hessian of A)^-1. A vector term reports sqrt(var_x + var_y), the rms length of its error vector.
    Infinite where the curvature is not negative definite (not at a maximum).
    """
    hessian = np.asarray(fit["hessian"], dtype=np.float64)
    free = list(np.asarray(fit["free"]))
    try:
        eigenvalues = np.linalg.eigvalsh(-hessian)
        covariance = 2.0 * noise_variance * np.linalg.inv(-hessian) if eigenvalues.min() > 0 else None
    except np.linalg.LinAlgError:
        covariance = None
    out = {}
    column = 6
    for name in terms:
        width = _width((name,))
        positions = [free.index(column + i) if column + i in free else None for i in range(width)]
        if covariance is None or None in positions:
            out[name] = math.inf
        else:
            out[name] = float(math.sqrt(sum(max(covariance[i, i], 0.0) for i in positions)))
        column += width
    return out


def edge_phase(term: str, magnitude_A: float, wavelength_A: float, semiangle_rad: float) -> float:
    """The probe phase (rad) a term of this magnitude adds at the aperture edge: (2 pi / lambda) C alpha^(n+1) / (n+1)."""
    n = ORDERS[FITTABLE[term]][0]
    return 2 * math.pi / wavelength_A * magnitude_A * semiangle_rad ** (n + 1) / (n + 1)


# When the fitted terms are the probe's aberrations (``assess``): the model must explain this share of G's signal
# power over the band, and each term's standard error must change the probe phase at the aperture edge by no more
# than this. On simulated crystals without these aberrations, no fit that returned spurious terms exceeded 0.80, and
# no fit that passed both was wrong (docs/maintainer/2026-10-21-ssb-higher-order-fit.md).
COHERENCE_MIN = 0.9
EDGE_PHASE_ERROR_MAX_RAD = 0.1


def assess(fit: dict[str, object], terms: tuple[str, ...], band: dict[str, float], *, counted: bool,
           wavelength_A: float, semiangle_rad: float) -> dict[str, object]:
    """Whether the polished higher orders can be read as the probe's aberrations.

    ``coherence``: the share of G's signal power over the band that the fitted model explains, (A - F_A) / (P - F_P)
    with A the agreement, P = sum_q w_q sum_k |G(q, k)|^2 (A <= P) and F the noise floors of both. It is 1 for a
    weak-phase object, whose G the SSB model describes exactly, and falls with the dynamical scattering of thick or
    heavy crystals, which extra free terms then partly fit. ``standard_error_nm`` and ``edge_phase_error_rad``: the
    precision of each term from the curvature (``standard_errors``) and the probe phase it is uncertain by at the
    aperture edge. Both need the noise of counted electrons (Poisson: variance G(0, k)); for other input
    (``counted=False``) the floors are taken as 0, which lowers the coherence of noisy data, and the precision is not
    estimated. ``trusted`` = coherence >= COHERENCE_MIN and every edge-phase error <= EDGE_PHASE_ERROR_MAX_RAD.
    """
    noise = band["noise_variance"] if counted else 0.0
    floor_agreement = band["band_weight"] * noise
    signal = band["power"] - floor_agreement * band["num_bf"]
    coherence = float((fit["agreement"] - floor_agreement) / signal) if signal > 0 else 0.0
    errors = standard_errors(fit, terms, noise)
    edge = {name: edge_phase(name, errors[name], wavelength_A, semiangle_rad) for name in terms}
    precise = all(value <= EDGE_PHASE_ERROR_MAX_RAD for value in edge.values())
    reasons = []
    if coherence < COHERENCE_MIN:
        reasons.append(f"the model explains {coherence:.2f} of the bright-field signal (below {COHERENCE_MIN}): "
                       "scattering it does not describe, as in a thick or strongly scattering crystal")
    if not precise:
        worst = max(edge, key=edge.get)
        reasons.append(f"{worst} is uncertain by {edge[worst]:.2f} rad at the aperture edge (above "
                       f"{EDGE_PHASE_ERROR_MAX_RAD}): too little signal for its precision")
    return {"coherence": coherence, "standard_error_nm": {name: errors[name] / 10.0 for name in terms},
            "edge_phase_error_rad": edge, "noise": "poisson counts" if counted else "not estimated",
            "trusted": not reasons, "reasons": reasons}


__all__ = ["COHERENCE_MIN", "EDGE_PHASE_ERROR_MAX_RAD", "FITTABLE", "assess", "edge_phase", "from_coordinates",
           "kernel_rows", "num_terms", "packed", "refine_higher_order", "standard_errors", "to_coordinates",
           "validate_terms"]
