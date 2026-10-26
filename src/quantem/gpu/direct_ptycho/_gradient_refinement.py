"""Bounded trust-region quasi-Newton refinement of the fifth-order agreement with exact gradients.

The exact full-data gradient defines the optimum; the Hessian and any BF subsample only steer. The initial Hessian
is the symmetrized forward difference of exact gradients on every n-th BF pixel at the start (made negative
definite), which also fixes the trust-region scaling D = sqrt|diag H0|. Accepted steps update it with dense BFGS.
Each solve first holds coordinates on a bound whose gradient points out of the box, then clips and holds
coordinates that leave it. A proposal's value is evaluated first and its gradient only when the agreement
increases. A coarse stage climbs the agreement of every m-th BF pixel first. On the full data a step is accepted
only when the full objective increases, the result is the best full-data point, and the search stops when the
bounded model step predicts a full-data gain below ``gain_tolerance`` (normalized by the initial agreement).

The host controller of the native Metal search (``SSBFifthOrderGradientSearch``), in float64 on the CPU; the
objective and its gradient run on the session's GPU.
"""

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Settings:
    """Search settings; trust radii are in units of D = sqrt|diag H0|."""

    hessian_bf_stride: int = 64
    coarse_bf_stride: int = 8
    coarse_tolerance: float = 1e-7
    initial_radius: float = 1.0
    fine_radius: float = 0.1
    max_radius: float = 64.0
    # The smallest normalized full-data gain a step must predict (one float32 agreement resolves about 1e-7).
    gain_tolerance: float = 1e-8
    # A rejected step that predicted less than this ends the stage: the float64-summed float32 powers vary by a
    # few 1e-8 between nearby points, so such a step cannot be resolved.
    noise_floor: float = 5e-8
    max_objectives: int = 400
    max_iterations: int = 200


def bounded_step(gradient, hessian, radius, z, lower, upper, scaling):
    """Maximize g.d + d'Hd / 2 subject to |D d| <= radius inside the box.

    Coordinates on a bound whose gradient points out of the box are held first; coordinates the solve moves
    outside are clipped, held, and the rest solved again (eigendecomposition and bisection on the shift).
    """
    n = len(gradient)
    free = ~(((z <= lower + 1e-12) & (gradient < 0)) | ((z >= upper - 1e-12) & (gradient > 0)))
    held = np.zeros(n)
    step = held.copy()
    for _ in range(n):
        step = held.copy()
        ids = np.flatnonzero(free)
        fixed = np.flatnonzero(~free)
        if ids.size:
            gs = (gradient[ids] + hessian[np.ix_(ids, fixed)] @ held[fixed]) / scaling[ids]
            # A = -D^-1 H D^-1 (positive definite for an ascent model); minimize -gs.u + u'Au/2, |u| <= r.
            a = -hessian[np.ix_(ids, ids)] / np.outer(scaling[ids], scaling[ids])
            w, v = np.linalg.eigh(a)
            b = v.T @ gs

            def norm(shift):
                return float(np.sqrt(np.sum((b / (w + shift)) ** 2)))

            smallest = float(w.min())
            shift = 0.0
            if not (smallest > 0 and norm(0.0) <= radius):
                lo = max(0.0, -smallest) + 1e-15
                hi = lo + 1
                while norm(hi) > radius:
                    hi *= 2
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if norm(mid) > radius:
                        lo = mid
                    else:
                        hi = mid
                shift = hi
            step[ids] = (v @ (b / (w + shift))) / scaling[ids]
        outside = free & ((z + step < lower) | (z + step > upper))
        if not outside.any():
            return step
        held[outside] = np.clip(z[outside] + step[outside], lower[outside], upper[outside]) - z[outside]
        free[outside] = False
    return step


def projected_gradient(gradient, z, bounds):
    """The gradient with zeros where a bound holds the coordinate and the ascent direction leaves the box."""
    gradient = np.array(gradient, dtype=float)
    gradient[(z <= bounds[:, 0]) & (gradient < 0)] = 0
    gradient[(z >= bounds[:, 1]) & (gradient > 0)] = 0
    return gradient


def refine(evaluate, start, bounds, *, start_value=None, settings=None, log=None, progress=None):
    """Refine ``start`` inside ``bounds`` (n x 2) to the full-data maximum.

    ``evaluate(points, kind, stride)`` returns ``(values, gradients)`` of the agreement divided by the initial full
    agreement: kind ``"value_and_gradient"``, ``"value"`` (gradients None) or ``"gradient"`` (the gradients of the
    points of the latest ``"value"`` call, values as returned then). ``stride`` 1 is the full objective; a larger
    stride the agreement of every n-th BF pixel, which is scaled here to equal the full one at the start. A coordinate
    whose bounds are equal is held at that value: the search runs over the others, and ``evaluate`` still receives
    and returns all n.

    Returns a dict: the best full-data ``coordinates`` and normalized ``agreement``, ``objectives`` (full-data value
    passes), ``gradients`` (full-data gradient passes), ``subsampled`` (subsample candidates), ``iterations``,
    ``stop`` and the ``trace`` of [objective count, best agreement, seconds]. ``progress(count, best)`` is called after
    each full-data value pass.
    """
    settings = settings or Settings()
    start = np.asarray(start, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    n = len(start)
    lower, upper = bounds[:, 0], bounds[:, 1]
    if (n < 6 or bounds.shape != (n, 2) or not np.all(np.isfinite(bounds)) or np.any(upper < lower)
            or np.all(upper == lower) or not np.all(np.isfinite(start)) or np.any(start < lower)
            or np.any(start > upper)):
        raise ValueError("The initial fit must lie inside the higher-order bounds. Check calibration and the initial fit.")
    # forward-difference steps of the starting Hessian: low orders, tilt and depth, then the higher orders
    h = np.array([0.01, 0.01, 0.01, 0.025, 0.025, 0.025] + [0.01] * (n - 6))
    free = upper > lower
    if free.all():
        return _refine(evaluate, start, bounds, h, start_value, settings, log, progress)
    index = np.flatnonzero(free)

    def reduced(points, kind, stride):
        full = np.tile(start, (len(points), 1))
        full[:, index] = points
        values, gradients = evaluate(full, kind, stride)
        return values, None if gradients is None else np.asarray(gradients)[:, index]

    result = _refine(reduced, start[index], bounds[index], h[index], start_value, settings, log, progress)
    coordinates = start.copy()
    coordinates[index] = result["coordinates"]
    return {**result, "coordinates": coordinates}


def _refine(evaluate, start, bounds, h, start_value, settings, log, progress):
    """``refine`` over free coordinates only: ``h`` the forward-difference steps of the starting Hessian."""
    n = len(start)
    lower, upper = bounds[:, 0], bounds[:, 1]
    counts = {"objectives": 0, "gradients": 0, "subsampled": 0}
    best = {"value": -np.inf, "point": start.copy()}
    trace = []
    started = time.perf_counter()
    subsample_scale = {}

    def record(point, value):
        counts["objectives"] += 1
        if value > best["value"]:
            best["value"], best["point"] = float(value), point.copy()
        trace.append([counts["objectives"], best["value"], time.perf_counter() - started])
        if progress is not None:
            progress(counts["objectives"], best["value"])

    def track(points, kind, stride):
        points = np.atleast_2d(points)
        if not np.all(np.isfinite(points)) or np.any(points < lower - 1e-9) or np.any(points > upper + 1e-9):
            raise ValueError("A trust-region candidate left its calibrated bounds.")
        values, gradients = evaluate(points, kind, stride)
        values = np.asarray(values, dtype=float)
        if values.shape != (len(points),) or not np.all(np.isfinite(values)) or np.any(values <= 0) or (
                kind != "value" and (gradients is None or not np.all(np.isfinite(gradients)))):
            raise ValueError("The higher-order objective returned nonfinite scores. "
                             "Check the calibration and fitting band.")
        if stride > 1:
            counts["subsampled"] += len(points)
            scale = subsample_scale.setdefault(stride, 1 / values[0])
            return values * scale, None if gradients is None else np.asarray(gradients) * scale
        if kind != "value":
            counts["gradients"] += len(points)
        if kind != "gradient":
            for point, value in zip(points, values):
                record(point, value)
        return values, gradients

    if start_value is not None:
        record(start, start_value)
    center = start.copy()
    # Forward differences of exact subsample gradients (the first call fixes the subsample's scale at the start).
    offsets = np.where(center + h <= upper, h, -h)
    design = np.tile(center, (n, 1)) + np.diag(offsets)
    column_values, column_gradients = track(np.vstack([center, design]), "value_and_gradient",
                                            settings.hessian_bf_stride)
    differences = (column_gradients[1:] - column_gradients[0]) / offsets[:, None]   # row j: dg / dz_j
    hessian = 0.5 * (differences + differences.T)
    # Negative definite for the ascent model and BFGS.
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    largest = float(np.abs(eigenvalues).max()) if eigenvalues.size else 1.0
    clamped = np.minimum(eigenvalues, -1e-6 * largest)
    hessian = (eigenvectors * clamped) @ eigenvectors.T
    scaling = np.sqrt(np.maximum(np.abs(np.diag(hessian)), 1e-12))
    if log:
        log(f"start, Hessian from {n} gradients on every {settings.hessian_bf_stride}-th BF pixel "
            f"({int(np.sum(eigenvalues > -1e-6 * largest))} positive eigenvalues clamped), "
            f"{time.perf_counter() - started:.3f} s")
    state = {"iterations": 0, "radius": settings.initial_radius}

    def stage(stride, tolerance, value, gradient):
        nonlocal center, hessian
        while state["iterations"] < settings.max_iterations:
            if counts["objectives"] >= settings.max_objectives:
                return "evaluation cap"
            state["iterations"] += 1
            step = bounded_step(gradient, hessian, state["radius"], center, lower, upper, scaling)
            proposal = np.clip(center + step, lower, upper)
            d = proposal - center
            if np.max(np.abs(d)) < 1e-6:
                if log:
                    log(f"{state['iterations']:3d} stride {stride}: small step")
                return "converged"
            predicted = float(gradient @ d + 0.5 * d @ hessian @ d)
            if predicted < tolerance:
                if log:
                    log(f"{state['iterations']:3d} stride {stride} predicted {predicted:+.3e}: converged")
                return "converged"
            trial_value = float(track(proposal, "value", stride)[0][0])
            actual = trial_value - value
            ratio = actual / predicted
            scaled_norm = float(np.sqrt(np.sum((scaling * d) ** 2)))
            if log:
                log(f"{state['iterations']:3d} stride {stride} radius {state['radius']:8.4f} |Dd| {scaled_norm:8.4f} "
                    f"predicted {predicted:+.3e} actual {actual:+.3e} ratio {ratio:+.3f} "
                    f"{'accept' if actual > 0 else 'reject'} {time.perf_counter() - started:.3f} s")
            if actual > 0:
                _, next_gradient = track(proposal, "gradient", stride)
                next_gradient = next_gradient[0]
                y = next_gradient - gradient
                hs = hessian @ d
                sy, shs = float(d @ y), float(d @ hs)
                if sy < -1e-16 and shs < 0:
                    hessian = hessian - np.outer(hs, hs) / shs + np.outer(y, y) / sy
                center, value, gradient = proposal, trial_value, next_gradient
                if ratio > 0.75 and scaled_norm > 0.8 * state["radius"]:
                    state["radius"] = min(state["radius"] * 2, settings.max_radius)
                elif ratio < 0.25:
                    state["radius"] *= 0.5
            else:
                if predicted < settings.noise_floor:
                    if log:
                        log(f"{state['iterations']:3d} stride {stride}: rejected at the noise floor")
                    return "converged"
                state["radius"] *= 0.25
                if state["radius"] < 1e-7:
                    return "trust radius exhausted"
        return "iteration limit"

    if settings.coarse_bf_stride > 1:
        if settings.coarse_bf_stride == settings.hessian_bf_stride:
            coarse_value, coarse_gradient = column_values[0], column_gradients[0]
        else:
            values, gradients = track(center, "value_and_gradient", settings.coarse_bf_stride)
            coarse_value, coarse_gradient = values[0], gradients[0]
        coarse_stop = stage(settings.coarse_bf_stride, settings.coarse_tolerance, coarse_value, coarse_gradient)
        if log:
            log(f"coarse stage (every {settings.coarse_bf_stride}-th BF pixel) {coarse_stop} after "
                f"{state['iterations']} iterations, {time.perf_counter() - started:.3f} s")
        state["radius"] = settings.fine_radius
    values, gradients = track(center, "value_and_gradient", 1)
    stop = stage(1, settings.gain_tolerance, values[0], gradients[0])
    if log:
        log(f"{stop} after {state['iterations']} iterations: {counts['objectives']} objectives, "
            f"{counts['gradients']} gradients, {counts['subsampled']} subsample evaluations, "
            f"{time.perf_counter() - started:.3f} s")
    return {
        "coordinates": best["point"], "agreement": best["value"], "objectives": counts["objectives"],
        "gradients": counts["gradients"], "subsampled": counts["subsampled"], "iterations": state["iterations"],
        "stop": stop, "trace": trace, "hessian_bf_stride": settings.hessian_bf_stride,
        "coarse_bf_stride": settings.coarse_bf_stride,
    }
