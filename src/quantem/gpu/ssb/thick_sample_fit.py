"""Shared constants of the joint (thick-sample) SSB fit and the finite-difference trust-region helpers of the
higher-order subset polish (``higher_order_fit``).

The joint fit itself is the resident search of ``_resident_search.fit_resident`` with the trust-region Newton on the
exact derivatives of ``cuda`` or ``mps.exact_derivatives`` (``_exact_newton``). Units: C10, C12, depth_spread_A in the
engine's C10 unit (Angstrom); phi12 rad; tilt mrad, scan frame (row, col).
"""

import numpy as np

PARAMETERS = ("C10", "C12", "phi12", "tilt_row_mrad", "tilt_col_mrad", "depth_spread_A")
# The depth spreads the joint fit searches, in nm (public) and in the engine's Angstrom: SSB.find_aberrations(tilt=True)
# returns a depth_spread_nm inside it, and its parameter slices span it.
DEPTH_SPREAD_RANGE_NM = (2.0, 60.0)
DEPTH_SPREAD_RANGE_A = (20.0, 600.0)


def _newton_design(center, step, hessian: bool) -> np.ndarray:
    """The points whose values give the gradient (and the Hessian) at center by finite differences.

    Always center and center +- h_i e_i (central gradient, diagonal curvature); with hessian also center + h_i e_i + h_j e_j
    for i < j, whose mixed second difference (f_ij - f_i - f_j + f_0) / (h_i h_j) is the off-diagonal curvature.
    6 parameters: 13 points, or 28 with the Hessian.
    """
    size = len(center)
    points = [center]
    for i in range(size):
        for sign in (1.0, -1.0):
            point = center.copy()
            point[i] += sign * step[i]
            points.append(point)
    if hessian:
        for i in range(size):
            for j in range(i + 1, size):
                point = center.copy()
                point[i] += step[i]
                point[j] += step[j]
                points.append(point)
    return np.array(points)


def _newton_model(values, step, curvature):
    """Gradient and curvature at the design's center from its values (``_newton_design`` order).

    g_i = (f(+i) - f(-i)) / 2 h_i, H_ii = (f(+i) - 2 f_0 + f(-i)) / h_i^2, and with 28 values H_ij = (f(+i+j) - f(+i) - f(+j)
    + f_0) / (h_i h_j); with 13 values the off-diagonal terms are kept from ``curvature`` (the last full design), so a
    gradient step costs 13 evaluations instead of 28 once the polish is near its maximum, where the Hessian barely changes.
    """
    size = len(step)
    plus, minus = values[1:2 * size + 1:2], values[2:2 * size + 2:2]
    gradient = (plus - minus) / (2 * step)
    hessian = curvature.copy()
    hessian[np.diag_indices(size)] = (plus - 2 * values[0] + minus) / step ** 2
    if len(values) > 2 * size + 1:
        mixed = iter(values[2 * size + 1:])
        for i in range(size):
            for j in range(i + 1, size):
                hessian[i, j] = hessian[j, i] = (next(mixed) - plus[i] - plus[j] + values[0]) / (step[i] * step[j])
    return gradient, hessian


def _trust_step(gradient, hessian, step, radius) -> np.ndarray:
    """The step s maximising g.s + s.H.s / 2 within |s / h| <= radius (h the finite-difference steps).

    In scaled coordinates y = s / h the model is gy.y + y.Hy.y / 2 with gy = h g and Hy = h H h. Its maximiser inside the
    ball is y = (mu I - Hy)^-1 gy with mu >= 0 the smallest value that makes mu I - Hy positive definite and |y| <= radius
    (mu = 0 is the plain Newton step when Hy is negative definite and the step fits): Levenberg-Marquardt, found by
    bisection on mu over the eigenvalues of Hy.
    """
    scaled_gradient, scaled_hessian = gradient * step, hessian * step[:, None] * step[None, :]
    eigenvalues, vectors = np.linalg.eigh(scaled_hessian)
    along = vectors.T @ scaled_gradient

    def scaled(mu):
        return vectors @ (along / (mu - eigenvalues))

    low = max(0.0, float(eigenvalues.max()))
    if eigenvalues.max() < 0 and np.linalg.norm(scaled(0.0)) <= radius:
        return scaled(0.0) * step
    # |y(mu)| falls from infinity just above the largest eigenvalue to 0: bisect for |y| = radius
    high = low + np.linalg.norm(scaled_gradient) / radius + 1e-30
    low += 1e-12 * (abs(low) + 1.0)
    for _ in range(100):
        middle = 0.5 * (low + high)
        if np.linalg.norm(scaled(middle)) > radius:
            low = middle
        else:
            high = middle
    return scaled(high) * step


def _bounded_step(center, gradient, hessian, step, radius, lower, upper) -> np.ndarray:
    """The trust step from center that stays inside the box [lower, upper].

    A coordinate the step pushes past a bound is held on the bound (displacement p), and the others are solved again on
    that face, where the model's gradient is g_f + H_fp p: clipping the free step instead would leave the other
    coordinates at the maximum of the unbounded model, which is not the maximum on the face (a depth spread on its floor
    left C10 1 A and the tilt 0.02 mrad short). Held coordinates are chosen afresh at every call, so one leaves its
    bound as soon as the model says so.
    """
    free = np.ones(len(center), dtype=bool)
    held = np.zeros(len(center))
    for _ in range(len(center)):
        move = held.copy()
        if free.any():
            on_face = gradient[free] + hessian[np.ix_(free, ~free)] @ held[~free]
            move[free] = _trust_step(on_face, hessian[np.ix_(free, free)], step[free], radius)
        outside = free & ((center + move < lower) | (center + move > upper))
        if not outside.any():
            break
        held[outside] = np.clip(center + move, lower, upper)[outside] - center[outside]
        free &= ~outside
    return move
