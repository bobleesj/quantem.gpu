"""GPU-resident fifth-order direct ptychography model agreement on Apple GPUs.

The Metal polynomial and depth model are those of the per-term agreement kernel
below, which the tests also run on its own against a float64 model. The final weighted reduction also stays on the GPU; only one
float32 score per candidate returns to the host optimizer. Coordinates match
the CUDA objective, including sample tilt and depth spread.
"""

from functools import lru_cache

import numpy as np

from quantem.gpu.direct_ptycho._execution import checkpoint
from quantem.gpu.direct_ptycho.fifth_order_parameters import Coordinates
from quantem.gpu.direct_ptycho.higher_order import CHI_SOURCE
from quantem.gpu.direct_ptycho.mps.hardware import require_mlx
from quantem.gpu.direct_ptycho.mps.resident_fit import _probe_geometry
from quantem.gpu.direct_ptycho.mps.thick_sample import (
    _K_CHUNK,
    _THICK_FIT_BATCH_HEADER,
    _band,
)

# Model agreement per (q, BF chunk, trial tile) of the thick-sample direct ptychography model with chi from the Krivanek polynomial.
_AGREEMENT_SOURCE = """
    uint q = thread_position_in_grid.x;
    uint chunk = thread_position_in_grid.y;
    uint base = thread_position_in_grid.z * B;
    if (q >= N_BAND) return;
    float wl = scalars[0], semiangle = scalars[1], ang_y = scalars[2], ang_x = scalars[3], factor = scalars[4];
    float qx = band_qx[q], qy = band_qy[q];
    uint off = band_flat[q];
    float nr[B], ni[B], dd[B];
    for (int b = 0; b < B; ++b) { nr[b] = 0.0f; ni[b] = 0.0f; dd[b] = 0.0f; }
    uint k0 = chunk * K_CHUNK;
    uint k1 = metal::min(uint(NUM_BF), k0 + uint(K_CHUNK));
    for (uint k = k0; k < k1; ++k) {
        float kxv = kx[k], kyv = ky[k];
        float4 gk = float4(probe_geometry[k * 4], probe_geometry[k * 4 + 1],
                           probe_geometry[k * 4 + 2], probe_geometry[k * 4 + 3]);
        float4 gm = tf_geometry(qx - kxv, qy - kyv, wl, semiangle, ang_y, ang_x);
        float4 gp = tf_geometry(qx + kxv, qy + kyv, wl, semiangle, ang_y, ang_x);
        float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;      // no double overlap: gamma = 0
        auto g = G[(size_t)k * (size_t)PLANE + (size_t)off];
        float gre = g.real, gim = g.imag;
        for (int b = 0; b < B; ++b) {
            if (base + b >= TOTAL) break;
            const device float* row = trial + (base + b) * STRIDE;
            float t = row[0], thr = row[1], thc = row[2];
            float chi_k = probe_chi[k * TOTAL + base + b];
            float b1 = 0.0f, b2 = 0.0f, c1 = 0.0f, cp = 0.0f, s1 = 0.0f, sp = 0.0f;
            bool thick = t != 0.0f;
            float shift = 6.283185307179586f * (qx * thr + qy * thc);
            if (a1 != 0.0f) {
                // chi(k - q), not chi(q - k): they differ for the odd terms (coma, trefoil). G(q, k) ~ P(k - q) conj P(k)
                // - P(k) conj P(k + q) for a thin object (Pennycook et al. 2015); chi(q - k) breaks G(-q) = conj G(q).
                float chi_m = ho_chi<NT>(kxv - qx, kyv - qy, wl, row + 3);
                s1 = metal::precise::sincos(chi_m - chi_k, c1);
                b1 = thick ? tf_sinc(0.5f * t * (-factor * (gm.x - gk.x) - shift)) * a1 : a1;
            }
            if (a2 != 0.0f) {
                float chi_p = ho_chi<NT>(qx + kxv, qy + kyv, wl, row + 3);
                sp = metal::precise::sincos(chi_p - chi_k, cp);
                b2 = thick ? tf_sinc(0.5f * t * (factor * (gp.x - gk.x) - shift)) * a2 : a2;
            }
            float gr = b1 * c1 - b2 * cp;
            float gi = -b1 * s1 - b2 * sp;
            nr[b] += gre * gr + gim * gi;        // G conj(gamma)
            ni[b] += gim * gr - gre * gi;
            dd[b] += gr * gr + gi * gi;
        }
    }
    for (int b = 0; b < B; ++b) {
        if (base + b >= TOTAL) break;
        size_t o = ((size_t)chunk * TOTAL + base + b) * N_BAND + q;
        numer_real[o] = nr[b];
        numer_imag[o] = ni[b];
        denom[o] = dd[b];
    }
"""

_HEADER = (
    "#define DIRECT_DEVICE inline\n"
    "#define DIRECT_COEFFICIENTS const device float*\n"
    + _THICK_FIT_BATCH_HEADER + CHI_SOURCE
)


@lru_cache(maxsize=1)
def _probe_kernel():
    """Evaluate candidate probe phases while reusing the static geometry."""
    return require_mlx().fast.metal_kernel(
        name="direct_fifth_order_probe",
        input_names=["kx", "ky", "scalars", "trial"],
        output_names=["probe_chi"],
        header=_HEADER,
        source="""
            uint k = thread_position_in_grid.x;
            uint b = thread_position_in_grid.y;
            if (k >= NUM_BF || b >= TOTAL) return;
            probe_chi[k * TOTAL + b] = direct_chi_full(
                kx[k], ky[k], scalars[0], trial + b * 31 + 3);
        """,
    )


@lru_cache(maxsize=1)
def _agreement_kernel():
    """Reuse the established model with the CUDA/Metal shared polynomial."""
    return require_mlx().fast.metal_kernel(
        name="direct_fifth_order_agreement",
        input_names=[
            "G", "band_flat", "band_qx", "band_qy", "kx", "ky",
            "trial", "scalars", "probe_geometry", "probe_chi",
        ],
        output_names=["numer_real", "numer_imag", "denom"],
        header=_HEADER,
        source=_AGREEMENT_SOURCE.replace("ho_chi<NT>", "direct_chi_full"),
    )


class Objective(Coordinates):
    """Evaluate the joint fifth-order objective from prepared MPS evidence.

    Parameters
    ----------
    prepared : PreparedMpsDirect
        Native-grid Fourier evidence and detector geometry. All stored active
        bright-field pixels contribute; no preview subset is selected here.
    partial_bytes : int, optional
        Target memory budget for candidate partial sums, with a minimum of
        one four-candidate tile. This changes execution batching, not the
        candidate coordinates or the scientific model.
    """

    def __init__(self, prepared, partial_bytes: int = 64 << 20) -> None:
        self.p = prepared
        self.calls = 0
        self.partial_bytes = int(partial_bytes)
        if self.partial_bytes < 1:
            raise ValueError("partial_bytes must be a positive memory budget.")
        self._scalars = prepared.mx.array(
            [prepared.wavelength, prepared.semiangle_rad,
             prepared.ang_y_rad, prepared.ang_x_rad, prepared.factor],
            dtype=prepared.mx.float32,
        )
        self._band_key = None
        self._band_arrays = None

    def evaluate(self, z, order: int, band=(0.2, 0.9)) -> np.ndarray:
        """Return native-grid agreement for each dimensionless candidate."""
        return self.score(*self.encode(np.atleast_2d(z), order), band=band)

    def score(self, primary, packed, band=(0.2, 0.9)) -> np.ndarray:
        """Return model agreement from engine-unit primary and packed optics.

        Primary columns are C10, C12 (angstrom), phi12 (radian), sample tilt
        row/column (mrad), and depth spread (angstrom). Packed optics are the
        14 Cartesian Krivanek coefficient pairs including the prefactor
        ``1 / (n + 1)`` for aberration order ``n``.
        """
        p = self.p
        mx = p.mx
        rows = np.atleast_2d(primary)
        coefficients = np.asarray(packed, dtype=np.float32).reshape(len(rows), 28)
        trial = np.empty((len(rows), 31), dtype=np.float32)
        trial[:, 0] = rows[:, 5]
        trial[:, 1:3] = rows[:, 3:5] * 1e-3
        trial[:, 3:] = coefficients
        key = tuple(float(value) for value in band)
        if self._band_key != key:
            flat, qx, qy, weight = _band(p, key)
            self._band_arrays = (flat, qx, qy, mx.array(weight, dtype=mx.float32))
            self._band_key = key
        flat, qx, qy, weights = self._band_arrays
        n_band = int(flat.size)
        num_bf = int(p.g_qk.shape[0])
        plane = int(p.g_qk.shape[1]) * int(p.g_qk.shape[2])
        chunks = (num_bf + _K_CHUNK - 1) // _K_CHUNK
        scalars = self._scalars
        per_candidate = (chunks + 1) * n_band * 3 * 4
        execution = max(4, (self.partial_bytes // per_candidate // 4) * 4)
        out = np.empty(len(trial), dtype=np.float64)
        for start in range(0, len(trial), execution):
            checkpoint()
            batch = min(execution, len(trial) - start)
            tile = min(4, batch)
            trial_mx = mx.array(trial[start:start + batch].ravel())
            geometry = _probe_geometry(p, scalars)
            (chi,) = _probe_kernel()(
                inputs=[p.kx, p.ky, scalars, trial_mx],
                template=[("NUM_BF", num_bf), ("TOTAL", batch), ("STRIDE", 31), ("NT", 14)],
                grid=(num_bf, batch, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(num_bf, batch)],
                output_dtypes=[mx.float32],
            )
            real, imag, denominator = _agreement_kernel()(
                inputs=[p.g_qk, flat, qx, qy, p.kx, p.ky, trial_mx, scalars, geometry, chi],
                template=[
                    ("B", tile), ("TOTAL", batch), ("N_BAND", n_band),
                    ("NUM_BF", num_bf), ("PLANE", plane),
                    ("K_CHUNK", _K_CHUNK), ("STRIDE", 31), ("NT", 14),
                ],
                grid=(n_band, chunks, (batch + tile - 1) // tile),
                threadgroup=(64, 1, 1),
                output_shapes=[(chunks, batch, n_band)] * 3,
                output_dtypes=[mx.float32] * 3,
            )
            real = mx.sum(real, axis=0)
            imag = mx.sum(imag, axis=0)
            denominator = mx.sum(denominator, axis=0)
            safe_denominator = mx.where(denominator > 0, denominator, 1.0)
            power = mx.where(
                denominator > 0,
                (real * real + imag * imag) / safe_denominator,
                0.0,
            )
            scores = mx.sum(weights[None, :] * power, axis=1)
            out[start:start + batch] = np.asarray(scores)
            self.calls += batch
        return out
