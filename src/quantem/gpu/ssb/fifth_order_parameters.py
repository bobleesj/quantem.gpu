"""Shared dimensionless coordinates for fifth-order SSB fitting."""

import numpy as np

from .higher_order import ORDERS


class Coordinates:
    """Encode physical aberrations without changing backend arithmetic."""

    def encode(self, z, order):
        z = np.asarray(z)
        rows = np.zeros((len(z), 6))
        packs = np.zeros((len(z), 14, 2))
        rows[:, 0] = z[:, 0] * 100
        rows[:, 1] = np.hypot(z[:, 1], z[:, 2]) * 100
        rows[:, 2] = 0.5 * np.arctan2(z[:, 2], z[:, 1])
        rows[:, 3:5] = z[:, 3:5] * 5
        rows[:, 5] = z[:, 5] * 100
        packs[:, 0, 0] = rows[:, 0] / 2
        packs[:, 1, 0] = z[:, 1] * 50
        packs[:, 1, 1] = z[:, 2] * 50
        j = 6
        for i, (n, m) in enumerate(ORDERS):
            if n < 2 or n > order:
                continue
            scale = self.p.wavelength / (2 * np.pi * self.p.semiangle_rad ** (n + 1))
            packs[:, i, 0] = z[:, j] * scale
            j += 1
            if m:
                packs[:, i, 1] = z[:, j] * scale
                j += 1
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
