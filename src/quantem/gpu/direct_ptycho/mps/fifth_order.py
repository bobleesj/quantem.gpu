"""GPU-resident fifth-order SSB model agreement on Apple GPUs.

The Metal polynomial and depth model are shared with the existing higher-order
agreement kernel. The final weighted reduction also stays on the GPU; only one
float32 score per candidate returns to the host optimizer. Coordinates match
the CUDA objective, including sample tilt and depth spread.
"""

from functools import lru_cache

import numpy as np

from quantem.gpu.direct_ptycho._execution import checkpoint
from quantem.gpu.direct_ptycho.fifth_order_parameters import Coordinates
from quantem.gpu.direct_ptycho.higher_order import CHI_SOURCE
from quantem.gpu.direct_ptycho.mps.hardware import require_mlx
from quantem.gpu.direct_ptycho.mps.higher_order_agreement import _AGREEMENT_SOURCE
from quantem.gpu.direct_ptycho.mps.resident_fit import _probe_geometry
from quantem.gpu.direct_ptycho.mps.thick_sample import (
    _K_CHUNK,
    _THICK_FIT_BATCH_HEADER,
    _band,
)

_HEADER = (
    "#define SSB_DEVICE inline\n"
    "#define SSB_COEFFICIENTS const device float*\n"
    + _THICK_FIT_BATCH_HEADER + CHI_SOURCE
)


@lru_cache(maxsize=1)
def _probe_kernel():
    """Evaluate candidate probe phases while reusing the static geometry."""
    return require_mlx().fast.metal_kernel(
        name="ssb_fifth_order_probe",
        input_names=["kx", "ky", "scalars", "trial"],
        output_names=["probe_chi"],
        header=_HEADER,
        source="""
            uint k = thread_position_in_grid.x;
            uint b = thread_position_in_grid.y;
            if (k >= NUM_BF || b >= TOTAL) return;
            probe_chi[k * TOTAL + b] = ssb_chi_full(
                kx[k], ky[k], scalars[0], trial + b * 31 + 3);
        """,
    )


@lru_cache(maxsize=1)
def _agreement_kernel():
    """Reuse the established model with the CUDA/Metal shared polynomial."""
    return require_mlx().fast.metal_kernel(
        name="ssb_fifth_order_agreement",
        input_names=[
            "G", "band_flat", "band_qx", "band_qy", "kx", "ky",
            "trial", "scalars", "probe_geometry", "probe_chi",
        ],
        output_names=["numer_real", "numer_imag", "denom"],
        header=_HEADER,
        source=_AGREEMENT_SOURCE.replace("ho_chi<NT>", "ssb_chi_full"),
    )


class Objective(Coordinates):
    """Evaluate the joint fifth-order objective from prepared MPS evidence.

    Parameters
    ----------
    prepared : PreparedMpsSSB
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
