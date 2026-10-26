"""Trust-region Newton of the low-order joint fit on the exact gradient and Hessian of the agreement.

The resident Newton (``_resident_search._newton``) models the agreement from finite-difference stencils: 13 or
28 evaluations per start and iteration. Here every iteration evaluates each active center once and asks for the
exact derivatives only where the step uses them: not at a rejected step, which keeps the previous center's, and
not at the starts that the second iteration's pruning stops. The trust region, bounds, acceptance, start pruning
and the second polish from other basins (``_other_basins``) are those of the resident Newton; the convergence
tolerances are a tenth of its (the steps converge quadratically). The host controller of the native Metal fit
(``DirectPtychoAgreementSearch``), in float64 on the CPU; the objective and its derivatives run on the session's GPU.

Points are (C10, C12, phi12, tilt row, tilt column, depth spread) in engine units (A, A, rad, mrad, mrad, A); the
Newton coordinates z replace (C12, phi12) by (C12 cos 2 phi12, C12 sin 2 phi12).
"""

import math
from dataclasses import dataclass, field

import numpy as np


def _f(value: float) -> float:
    """A float32 literal of the resident kernels, as their double-precision Real holds it."""
    return float(np.float32(value))


STEPS = np.array([2.0, 2.0, 2.0, _f(0.25), _f(0.25), 5.0])
TOLERANCES = np.array([_f(0.05), _f(0.05), _f(0.05), _f(0.005), _f(0.005), _f(0.5)])
EXACT_TOLERANCES = TOLERANCES / 10
DISTINCT = np.array([50.0, 20.0, _f(0.3), 2.0, 2.0, 50.0])
# The second polish replaces the first only when its agreement is higher by this relative margin (BASIN_MARGIN).
BASIN_MARGIN = 1e-6


def encode(p):
    z = np.array(p, dtype=float)
    z[1] = p[1] * math.cos(2 * p[2])
    z[2] = p[1] * math.sin(2 * p[2])
    return z


def _atan2(y, x):
    if abs(x) < _f(1e-30):
        return -math.pi / 2 if y < 0 else math.pi / 2
    angle = math.atan(y / x)
    if x < 0:
        angle += -math.pi if y < 0 else math.pi
    return angle


def decode(z):
    p = np.array(z, dtype=float)
    p[1] = math.sqrt(max(0.0, z[1] * z[1] + z[2] * z[2]))
    p[2] = _atan2(z[2], z[1]) / 2
    p[5] = abs(z[5])
    return p


def joint_trial(z) -> np.ndarray:
    """The joint objective kernels' float32 trial rows of Newton coordinates z.

    The kernels form their astigmatism terms as C12 cos 2 phi12 and C12 sin 2 phi12 from a row's second to fourth
    entries; (1, a, b) gives a and b exactly. Rows: C10, 1, a, b, depth spread [A], tilt row, tilt column [rad].
    """
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    return np.stack([z[:, 0], np.ones(len(z)), z[:, 1], z[:, 2], np.abs(z[:, 5]), z[:, 3] * 1e-3, z[:, 4] * 1e-3],
                    axis=1).astype(np.float32)


def wrap_phi(phi):
    return (phi + math.pi / 2) % math.pi - math.pi / 2


def ranks(scores):
    """Descending rank, ties to the earlier trial (``RANK_BODY``)."""
    scores = np.asarray(scores, dtype=float)
    order = np.lexsort((np.arange(len(scores)), -scores))
    out = np.empty(len(scores), dtype=int)
    out[order] = np.arange(len(scores))
    return out


def trust_step(gradient, hessian, steps, radius):
    """The scaled trust-region step that maximizes the quadratic model (``trust_step``)."""
    a = hessian * np.outer(steps, steps)
    scaled = gradient * steps
    w, v = np.linalg.eigh(a)
    largest = max(_f(-1e30), float(w.max()))
    gradient_norm = float(scaled @ scaled)
    along = v.T @ scaled
    if np.all(np.abs(w) > _f(1e-30)):
        norm = float(np.sum((along / -w) ** 2))
    else:
        norm = _f(1e30)
    mu = 0.0
    if not (largest < 0 and math.sqrt(norm) <= radius):
        low = max(0.0, largest)
        high = low + math.sqrt(gradient_norm) / radius + _f(1e-30)
        low += _f(1e-12) * (abs(low) + 1)
        for _ in range(100):
            middle = (low + high) / 2
            if math.sqrt(float(np.sum((along / (middle - w)) ** 2))) > radius:
                low = middle
            else:
                high = middle
        mu = high
    return (v @ (along / (mu - w))) * steps


def bounded_step(center, gradient, hessian, radius, lower, upper, free):
    """The trust-region step with coordinates that leave the box held at their bound (``bounded_step``)."""
    free = np.array(free, dtype=bool)
    held = np.zeros(6)
    move = held.copy()
    for _ in range(6):
        move = held.copy()
        ids = np.flatnonzero(free)
        fixed = np.flatnonzero(~free)
        if ids.size:
            fg = gradient[ids] + hessian[np.ix_(ids, fixed)] @ held[fixed]
            move[ids] = trust_step(fg, hessian[np.ix_(ids, ids)], STEPS[ids], radius)
        outside = free & ((center + move < lower) | (center + move > upper))
        if not outside.any():
            break
        held[outside] = np.clip(center[outside] + move[outside], lower[outside], upper[outside]) - center[outside]
        free[outside] = False
    return move


@dataclass
class NewtonState:
    center: np.ndarray
    heading: np.ndarray
    origin: np.ndarray
    kept_center: np.ndarray = field(default_factory=lambda: np.zeros(6))
    kept_value: float = 0.0
    kept_gradient: np.ndarray = field(default_factory=lambda: np.zeros(6))
    kept_hessian: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    radius: float = 8.0
    active: bool = True
    has_previous: bool = False
    iterations: int = 0
    evaluations: int = 0
    converged: bool = False
    known: bool = False


def newton(objective, points, scores, lower, upper, *, explicit=(), starts=3, prune=True, known=()):
    """Polish up to three starts, the ``explicit`` points first and then the best distinct trials, 30 iterations.

    ``objective.values(z_rows, all_derivatives)`` returns the agreements at rows of z and
    ``objective.derivatives(indices)`` (value, gradient, Hessian) at rows of its latest ``values`` call.
    ``prune`` keeps only the highest start after the second step; without it every start climbs until it converges
    or meets a higher one. ``lower`` and ``upper`` bound (C10, C12, phi12, tilt row, tilt column, depth spread); a
    coordinate with equal bounds is held there, as is the astigmatism pair when C12 and phi12 both are or the C12
    maximum is 0 (fixed aberrations; a standard fit's tilt and depth spread at 0). Returns the best point (engine
    units), its agreement and the starts' states.
    """
    points = np.asarray(points, dtype=float)
    scores = np.asarray(scores, dtype=float)
    starts = min(3, int(starts))
    rank = ranks(scores)
    by_rank = np.argsort(rank)
    chosen = [np.asarray(p, dtype=float) for p in explicit]
    for index in by_rank:
        if len(chosen) >= starts:
            break
        p = points[index]
        if all(np.max(np.abs(p - kept) / DISTINCT) > 1 for kept in chosen):
            chosen.append(p)
    state = []
    for finished in known:
        kept = NewtonState(**{**finished.__dict__})
        kept.active, kept.known = False, True
        state.append(kept)
    held = np.array(lower, dtype=float) == np.array(upper, dtype=float)
    held[1] = held[2] = bool(upper[1] == 0 or (held[1] and held[2]))
    for p in chosen:
        p = np.where(held, lower, p)
        z = encode(p)
        state.append(NewtonState(center=z, heading=z.copy(), origin=z.copy()))
    scale = max(_f(1e-30), float(np.max(np.abs(scores)))) if scores.size else _f(1e-30)
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)
    lower[1] = -upper[1]
    lower[2] = -upper[1]
    upper[2] = upper[1]
    free = ~held
    for iteration in range(30):
        active = [k for k, s in enumerate(state) if s.active]
        if not active:
            break
        pruning = prune and iteration == 1 and len(active) > 1
        values = np.asarray(objective.values(np.array([state[k].center for k in active]), not pruning), dtype=float)
        if values.shape != (len(active),) or not np.all(np.isfinite(values)):
            raise ValueError("The joint objective returned nonfinite agreements.")
        exact = [None] * len(active)
        kept_values = []
        accepted = []
        for position, k in enumerate(active):
            s = state[k]
            value = values[position] / scale
            rejected = s.has_previous and value < s.kept_value
            kept_values.append(s.kept_value if rejected else value)
            if not rejected:
                accepted.append(position)
        deferred = []
        survivor = -1
        if pruning:
            best = int(np.argmax(kept_values))
            unique = all(i == best or kept_values[i] < kept_values[best] for i in range(len(active)))
            above = all(j in active or not state[j].has_previous or state[j].kept_value < kept_values[best]
                        for j in range(len(state)))
            if unique and above:
                survivor = best
                deferred = [p for p in accepted if p != best]

        def fetch(positions):
            if not positions:
                return
            results = objective.derivatives(positions)
            for position, (value, gradient, hessian) in zip(positions, results):
                if not (np.isfinite(value) and np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian))):
                    raise ValueError("The joint objective returned nonfinite derivatives.")
                exact[position] = (gradient, hessian)

        fetch([p for p in accepted if p not in deferred])

        def advance(position, stopping=False):
            k = active[position]
            s = state[k]
            value = values[position] / scale
            center = s.center.copy()
            moved = float(np.sqrt(np.sum(((center - s.kept_center) / STEPS) ** 2)))
            gradient = np.zeros(6)
            hessian = s.kept_hessian.copy()
            if s.has_previous and value < s.kept_value:
                center = s.kept_center.copy()
                gradient = s.kept_gradient.copy()
                s.radius = min(s.radius, moved) / 4
            elif stopping:
                if s.has_previous and moved >= _f(0.99) * s.radius:
                    s.radius *= 2
                s.kept_center, s.kept_value, s.has_previous = center, value, True
                s.heading = center.copy()
                s.iterations += 1
                s.evaluations += 1
                return
            else:
                gradient = exact[position][0] / scale
                hessian = exact[position][1] / scale
                if s.has_previous and moved >= _f(0.99) * s.radius:
                    s.radius *= 2
                s.kept_center, s.kept_gradient, s.kept_hessian = center, gradient, hessian
                s.kept_value, s.has_previous = value, True
            move = bounded_step(center, gradient, hessian, s.radius, lower, upper, free)
            proposal = np.where(free, np.clip(center + move, lower, upper), center)
            norm = math.hypot(proposal[1], proposal[2])
            if norm > upper[1]:
                proposal[1:3] *= upper[1] / norm
            converged = bool(np.all(np.abs(proposal - center) < EXACT_TOLERANCES))
            if converged:
                s.active, s.converged = False, True
            s.center = center.copy() if converged else proposal
            s.heading = s.center.copy()
            s.iterations += 1
            s.evaluations += 1

        for position in range(len(active)):
            if position not in deferred:
                advance(position)
        if deferred:
            if state[active[survivor]].active:
                for position in deferred:
                    advance(position, stopping=True)
            else:
                fetch(deferred)
                for position in deferred:
                    advance(position)
        for k, s in enumerate(state):
            if not s.active:
                continue
            for j, other in enumerate(state):
                if j == k or not other.has_previous:
                    continue
                near = bool(np.all(np.abs(s.kept_center - other.kept_center) < 4 * STEPS))
                heading = bool(np.all(np.abs(s.heading - other.heading) < 2 * STEPS))
                worse = s.kept_value < other.kept_value or (not s.kept_value > other.kept_value and k > j)
                if (near or heading) and worse:
                    s.active = False
                    break
        if prune and iteration == 1:
            best = -1
            for k, s in enumerate(state):
                if s.active and (best < 0 or s.kept_value > state[best].kept_value):
                    best = k
            for k, s in enumerate(state):
                if k != best:
                    s.active = False
    best = 0
    for k in range(1, len(state)):
        if state[k].has_previous and state[k].kept_value > state[best].kept_value:
            best = k
    return decode(state[best].kept_center), state[best].kept_value * scale, [s for s in state if not s.known]


def alternate_starts(points, scores, polished):
    """The best trial with phi12 mirrored and the best trial in another (phi12, tilt, depth spread) cluster."""
    points = np.asarray(points, dtype=float)
    rank = ranks(scores)
    by_rank = np.argsort(rank)
    best = points[by_rank[0]]
    mirror = best.copy()
    mirror[2] = -best[2]
    chosen = []

    def different(p):
        for q in [polished] + chosen:
            phi = abs(p[2] - q[2])
            if phi > math.pi * _f(0.5):
                phi = math.pi - phi
            distance = phi / DISTINCT[2]
            for d in range(3, 6):
                distance = max(distance, abs(p[d] - q[d]) / DISTINCT[d])
            if distance <= 1:
                return False
        return True

    if different(mirror):
        chosen.append(mirror)
    for index in by_rank:
        if len(chosen) >= 2:
            break
        if different(points[index]):
            chosen.append(points[index])
    return chosen


def other_basins(objective, points, scores, lower, upper, first):
    """Polish the alternate starts, each to its own maximum, and keep that result only when it is clearly higher.

    An alternate that is one of the first polish's starts would retrace that start's deterministic climb and is not
    polished again; the first polish's maximum takes part in the start pruning, so an alternate that comes near it,
    or heads where it is, while lower, stops.
    """
    point, value, first_starts = first
    origins = [s.origin for s in first_starts]
    alternates = [p for p in alternate_starts(points, scores, point)
                  if not any(np.array_equal(encode(p), origin) for origin in origins)]
    if not alternates:
        return first
    known = []
    polished = [s for s in first_starts if s.has_previous]
    if polished:
        marker = max(polished, key=lambda s: s.kept_value)
        marker = NewtonState(**{**marker.__dict__})
        marker.heading = marker.kept_center.copy()
        known = [marker]
    second = newton(objective, points, scores, lower, upper, explicit=alternates, starts=len(alternates),
                    prune=False, known=known)
    starts = first_starts + second[2]
    if second[1] > value * (1.0 + BASIN_MARGIN):
        return second[0], second[1], starts
    return point, value, starts


def fit(objective, points, scores, lower, upper, *, start=None, starts=3, basins=True):
    """The resident fit's polish: Newton from the start and the best trials, then the other basins.

    Returns the point (C12 >= 0, phi12 wrapped to [-pi/2, pi/2)), its agreement on the search objective and the
    starts' diagnostics.
    """
    explicit = [] if start is None else [np.asarray(start, dtype=float)]
    fitted = newton(objective, points, scores, lower, upper, explicit=explicit, starts=starts)
    if basins:
        fitted = other_basins(objective, points, scores, lower, upper, fitted)
    point, _, states = fitted
    point = np.array(point, dtype=float)
    agreement = float(objective.values(encode(point)[None], False)[0])
    point[1] = abs(point[1])
    point[2] = wrap_phi(point[2])
    diagnostics = [{"iterations": s.iterations, "evaluations": s.evaluations, "converged": s.converged}
                   for s in states if s.has_previous]
    return point, agreement, diagnostics
