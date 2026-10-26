"""Shared dimensionless coordinates for fifth-order SSB fitting."""

import math

import numpy as np

from .higher_order import ORDERS


def monomial_map(wavelength: float) -> np.ndarray:
    """(25, 28): d agreement / d packed coefficient from the monomial sums T_a, 2 <= |a| <= 6, in float64.

    chi(row, col) = (2 pi / wl) sum_i (x^2 + y^2)^p (Re (x + iy)^m c_2i + Im (x + iy)^m c_2i+1) with (x, y) = wl (row,
    col) (``higher_order.CHI_SOURCE``), so d phi / d c_p = 2 pi wl^n_i [P_p(k -+ q) - P_p(k)] with P_p the term's
    monomials row^a col^c. T_a is stored at index (a + c)(a + c + 1) / 2 + c - 3.
    """
    def times(a, b):
        out = {}
        for (i, j), u in a.items():
            for (k, l), v in b.items():
                out[(i + k, j + l)] = out.get((i + k, j + l), 0) + u * v
        return out

    mapping = np.zeros((25, 28))
    for term, (n, m) in enumerate(ORDERS):
        radial = {(0, 0): 1.0}
        for _ in range((n + 1 - m) // 2):
            radial = times(radial, {(2, 0): 1.0, (0, 2): 1.0})
        real, imag = {(0, 0): 1.0}, {}
        for _ in range(m):
            # (re + i im)(x + i y) = (re x - im y) + i (im x + re y)
            new_re, new_im = {}, {}
            for (i, j), v in real.items():
                new_re[(i + 1, j)] = new_re.get((i + 1, j), 0) + v
                new_im[(i, j + 1)] = new_im.get((i, j + 1), 0) + v
            for (i, j), v in imag.items():
                new_re[(i, j + 1)] = new_re.get((i, j + 1), 0) - v
                new_im[(i + 1, j)] = new_im.get((i + 1, j), 0) + v
            real, imag = new_re, new_im
        factor = 2 * math.pi * wavelength ** n
        for part, polynomial in ((0, times(radial, real)), (1, times(radial, imag))):
            for (a, c), value in polynomial.items():
                if value:
                    d = a + c
                    mapping[d * (d + 1) // 2 + c - 3, 2 * term + part] += factor * value
    return mapping


def packed_coefficients(z, wavelength: float, semiangle_rad: float, order: int = 5) -> np.ndarray:
    """(len(z), 14, 2): the packed Cartesian coefficients C_nm / (n + 1) (cos, sin) in A of coordinates z, float64."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    packs = np.zeros((len(z), 14, 2))
    packs[:, 0, 0] = z[:, 0] * 50
    packs[:, 1, 0] = z[:, 1] * 50
    packs[:, 1, 1] = z[:, 2] * 50
    j = 6
    for i, (n, m) in enumerate(ORDERS):
        if n < 2 or n > order:
            continue
        scale = wavelength / (2 * np.pi * semiangle_rad ** (n + 1))
        packs[:, i, 0] = z[:, j] * scale
        j += 1
        if m:
            packs[:, i, 1] = z[:, j] * scale
            j += 1
    return packs


def aperture_phase(z, wavelength: float, semiangle_rad: float, order: int = 5, points: int = 96) -> np.ndarray:
    """(len(z), samples): chi (rad) of coordinates z over the aperture |theta| <= alpha, in float64.

    Every aberration order counts, C10 and C12 included; tilt and depth spread do not enter chi. The samples are the
    centres of a ``points`` x ``points`` grid on the aperture's square that fall inside it (the native
    ``DirectPtychoHigherOrderConsistency.apertureDisagreement`` grid), so a plain mean over them is an area average.
    """
    unit = (np.arange(points) + 0.5) / points * 2 - 1
    x, y = np.meshgrid(unit, unit, indexing="ij")
    inside = x * x + y * y <= 1
    angle = (x[inside] + 1j * y[inside]) * semiangle_rad
    r2 = np.abs(angle) ** 2
    terms = np.empty((28, angle.size))
    for i, (n, m) in enumerate(ORDERS):
        term = r2 ** ((n + 1 - m) // 2) * angle ** m
        terms[2 * i], terms[2 * i + 1] = term.real, term.imag
    packs = packed_coefficients(z, wavelength, semiangle_rad, order).reshape(-1, 28)
    return (2 * np.pi / wavelength) * packs @ terms


def aperture_rms(chi_a, chi_b) -> np.ndarray:
    """Root mean square over the aperture of chi_a - chi_b after removing the piston (``aperture_phase`` rows)."""
    difference = np.asarray(chi_a, dtype=np.float64) - np.asarray(chi_b, dtype=np.float64)
    difference = difference - difference.mean(axis=-1, keepdims=True)
    return np.sqrt(np.mean(difference ** 2, axis=-1))


class Coordinates:
    """Encode physical aberrations without changing backend arithmetic."""

    def gradient(self, raw):
        """d agreement / dz from a gradient pass's 28 sums per candidate.

        ``raw`` rows are [d/d depth spread (A), d/d tilt row, d/d tilt column (rad), T_a for 2 <= |a| <= 6]: the
        exact-gradient kernels' outputs (``cuda.exact_derivatives``), mapped through ``encode``'s scales.
        """
        raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        mapping = getattr(self, "_monomial_map", None)
        if mapping is None:
            mapping = self._monomial_map = monomial_map(float(self.p.wavelength))
        packed = raw[:, 3:] @ mapping
        gradient = np.zeros((len(raw), 28))
        gradient[:, 0] = 50 * packed[:, 0]
        gradient[:, 1] = 50 * packed[:, 2]
        gradient[:, 2] = 50 * packed[:, 3]
        gradient[:, 3] = 5e-3 * raw[:, 1]
        gradient[:, 4] = 5e-3 * raw[:, 2]
        gradient[:, 5] = 100 * raw[:, 0]
        j = 6
        for i, (n, m) in enumerate(ORDERS):
            if n < 2:
                continue
            scale = self.p.wavelength / (2 * np.pi * self.p.semiangle_rad ** (n + 1))
            gradient[:, j] = scale * packed[:, 2 * i]
            j += 1
            if m:
                gradient[:, j] = scale * packed[:, 2 * i + 1]
                j += 1
        return gradient

    def encode(self, z, order):
        z = np.asarray(z)
        rows = np.zeros((len(z), 6))
        rows[:, 0] = z[:, 0] * 100
        rows[:, 1] = np.hypot(z[:, 1], z[:, 2]) * 100
        rows[:, 2] = 0.5 * np.arctan2(z[:, 2], z[:, 1])
        rows[:, 3:5] = z[:, 3:5] * 5
        rows[:, 5] = z[:, 5] * 100
        packs = packed_coefficients(z, self.p.wavelength, self.p.semiangle_rad, order)
        return (rows, packs.astype(np.float32))


def vector(fit, order):
    a = fit.aberrations
    phi = np.deg2rad(a["phi12_deg"]) * 2
    z = [
        a["C10_nm"] / 10,
        a["C12_nm"] / 10 * np.cos(phi),
        a["C12_nm"] / 10 * np.sin(phi),
        fit.sample_tilt_mrad[0] / 5,
        fit.sample_tilt_mrad[1] / 5,
        fit.depth_spread_nm / 10,
    ]
    for n, m in ORDERS:
        if 2 <= n <= order:
            z.extend([0.0] * (2 if m else 1))
    return np.array(z)


def public_params(obj, z, order):
    rows, packs = obj.encode(np.atleast_2d(z), order)
    r = rows[0]
    packs = packs[0]
    a = {"C10_nm": r[0] / 10, "C12_nm": r[1] / 10, "phi12_deg": np.rad2deg(r[2])}
    for i, (n, m) in enumerate(ORDERS):
        if 2 <= n <= order:
            a[f"C{n}{m}_nm"] = float(
                (packs[i, 0] if m == 0 else np.hypot(*packs[i])) * (n + 1) / 10
            )
            if m:
                a[f"phi{n}{m}_deg"] = float(
                    np.rad2deg(np.arctan2(packs[i, 1], packs[i, 0])) / m
                )
    return (a, tuple(r[3:5]), float(r[5] / 10))
