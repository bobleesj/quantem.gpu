"""Bounded finite-difference Newton refinement for the broad SSB model."""

import time

import numpy as np

from .thick_sample_fit import _bounded_step


def stencil(center, h, bounds, full=True):
    """Three-point diagonal and one-sided mixed stencils, entirely inside bounds."""
    n = len(center)
    offsets = []
    points = [center.copy()]
    for i in range(n):
        lo, hi = bounds[i]
        if center[i] - h[i] >= lo and center[i] + h[i] <= hi:
            aa, bb = (h[i], -h[i])
        elif hi - center[i] >= center[i] - lo:
            hh = min(h[i], (hi - center[i]) / 2)
            aa, bb = (hh, 2 * hh)
        else:
            hh = min(h[i], (center[i] - lo) / 2)
            aa, bb = (-hh, -2 * hh)
        assert abs(aa) > 1e-12
        offsets.append((aa, bb))
        for delta in (aa, bb):
            p = center.copy()
            p[i] += delta
            points.append(p)
    if full:
        for i in range(n):
            for j in range(i + 1, n):
                p = center.copy()
                p[i] += offsets[i][0]
                p[j] += offsets[j][0]
                points.append(p)
    return (np.array(points), np.array(offsets))


def model(values, offsets, old):
    n = len(offsets)
    gradient = np.empty(n)
    hessian = old.copy()
    for i, (aa, bb) in enumerate(offsets):
        yy = values[[0, 1 + 2 * i, 2 + 2 * i]]
        matrix = np.array([[1, 1, 1], [0, aa, bb], [0, aa * aa, bb * bb]])
        gradient[i] = np.linalg.solve(matrix, [0, 1, 0]) @ yy
        hessian[i, i] = np.linalg.solve(matrix, [0, 0, 2]) @ yy
    if len(values) > 1 + 2 * n:
        k = 1 + 2 * n
        for i in range(n):
            for j in range(i + 1, n):
                hessian[i, j] = hessian[j, i] = (
                    values[k] - values[1 + 2 * i] - values[1 + 2 * j] + values[0]
                ) / (offsets[i, 0] * offsets[j, 0])
                k += 1
    return (gradient, hessian)


def newton(track, z0, bounds):
    center = z0.copy()
    n = len(center)
    h = np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05] + [0.02] * (n - 6))
    radius = 8.0
    curvature = np.zeros((n, n))
    nit = 0
    accepted = 0
    for nit in range(200):
        points, offsets = stencil(center, h, bounds, full=nit == 0 or accepted % 4 == 0)
        values = track(points)
        gradient, curvature = model(values, offsets, curvature)
        delta = _bounded_step(
            center, gradient, curvature, h, radius, bounds[:, 0], bounds[:, 1]
        )
        proposal = np.clip(center + delta, bounds[:, 0], bounds[:, 1])
        if np.max(np.abs(delta) / h) < 0.01:
            return (nit + 1, "small bounded Newton step")
        score = float(track(proposal)[0])
        pred = float(gradient @ delta + 0.5 * delta @ curvature @ delta)
        actual = score - values[0]
        if actual > 0:
            center = proposal
            accepted += 1
            if pred > 0 and actual / pred > 0.75:
                radius = min(radius * 2, 64.0)
            elif pred <= 0 or actual / pred < 0.25:
                radius *= 0.5
        else:
            radius *= 0.25
        if radius < 0.01:
            return (nit + 1, "trust radius exhausted")
    return (nit + 1, "iteration limit")


class BudgetReached(Exception):
    pass


class Tracker:
    def __init__(self, fun, start, bounds, cap):
        self.fun, self.bounds, self.cap = (fun, bounds, cap)
        self.best_z, self.best, self.nfev = (start.copy(), -np.inf, 0)
        self.started = time.perf_counter()
        self.trace, self.checkpoints = ([], {})

    def __call__(self, points):
        points = np.atleast_2d(points)
        if self.nfev >= self.cap:
            raise BudgetReached
        assert np.all(points >= self.bounds[:, 0] - 1e-09) and np.all(
            points <= self.bounds[:, 1] + 1e-09
        )
        count = min(len(points), self.cap - self.nfev)
        values = self.fun(points[:count])
        assert np.isfinite(values).all()
        for z, score in zip(points[:count], values):
            self.nfev += 1
            if score > self.best:
                self.best, self.best_z = (float(score), z.copy())
            elapsed = time.perf_counter() - self.started
            self.trace.append([self.nfev, self.best, elapsed])
            if self.nfev == 1000:
                self.checkpoints[1000] = (self.best_z.copy(), self.best, elapsed)
        if count < len(points):
            raise BudgetReached
        return values
