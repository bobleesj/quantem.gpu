"""Native-grid weak-object inversion with separate amplitude and phase.

For T = exp(eta + i phi) ~ 1 + eta + i phi, the non-DC scan
spectrum is G = (t1 + t2) eta_Q + i (t1 - t2) phi_Q, where
t1 = P(k-Q) P*(k), t2 = P(k) P*(k+Q). Both channels are real in
space. The uniform-depth weights and probe polynomial are shared with SSB.
This is a linear model, not multislice inversion or quantitative absorption.
"""

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from quantem.gpu.direct_ptycho.cuda.kernels.engine import _THICK_GEOMETRY_SOURCE
from quantem.gpu.direct_ptycho.higher_order import CHI_SOURCE, pack_coefficients

if TYPE_CHECKING:
    from quantem.gpu.direct_ptycho.cuda.engine import SSBEngine


_transfer = cp.ElementwiseKernel(
    "float32 qr, float32 qc, float32 kr, float32 kc, float32 wl, "
    "float32 alpha, float32 dr, float32 dc, float32 depth, "
    "float32 tr, float32 tc, raw float32 coefficients",
    "complex64 ha, complex64 hp",
    r"""
    float4 k = thick_probe_geometry(kr, kc, wl, alpha, dr, dc);
    float4 m = thick_probe_geometry(kr-qr, kc-qc, wl, alpha, dr, dc);
    float4 p = thick_probe_geometry(kr+qr, kc+qc, wl, alpha, dr, dc);
    float ck = ssb_chi_full(kr, kc, wl, &coefficients[0]);
    float cm = ssb_chi_full(kr-qr, kc-qc, wl, &coefficients[0]);
    float cp_ = ssb_chi_full(kr+qr, kc+qc, wl, &coefficients[0]);
    float shift = 6.283185307f * (qr*tr + qc*tc);
    float w1 = thick_sinc(0.5f * depth * (-3.141592654f/wl*(m.x-k.x)-shift));
    float w2 = thick_sinc(0.5f * depth * (3.141592654f/wl*(p.x-k.x)-shift));
    complex<float> t1 = m.w*k.w*w1*complex<float>(cosf(cm-ck), -sinf(cm-ck));
    complex<float> t2 = p.w*k.w*w2*complex<float>(cosf(cp_-ck), sinf(cp_-ck));
    ha = t1+t2;
    hp = complex<float>(0, 1)*(t1-t2);
    """,
    "ssb_weak_object_transfer",
    preamble=(
        "#define SSB_DEVICE __device__\n"
        "#define SSB_COEFFICIENTS const float*\n" + CHI_SOURCE + _THICK_GEOMETRY_SOURCE
    ),
)


def transfer(
    engine: "SSBEngine",
    rows: cp.ndarray,
    cols: cp.ndarray,
    magnitudes_A: np.ndarray,
    angles_rad: np.ndarray,
    tilt_mrad: tuple[float, float],
    depth_A: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Return amplitude/phase transfer matrices, (frequency, BF pixel)."""
    return transfer_frequencies(
        engine,
        engine._cache["qx_1d"][rows],
        engine._cache["qy_1d"][cols],
        magnitudes_A,
        angles_rad,
        tilt_mrad,
        depth_A,
    )


def transfer_frequencies(
    engine: "SSBEngine",
    qr: cp.ndarray,
    qc: cp.ndarray,
    magnitudes_A: np.ndarray,
    angles_rad: np.ndarray,
    tilt_mrad: tuple[float, float],
    depth_A: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Evaluate the transfer at physical frequencies, including scan aliases."""
    cache = engine._cache
    coefficients = cp.asarray(pack_coefficients(magnitudes_A, angles_rad))
    return _transfer(
        qr[..., None],
        qc[..., None],
        cache["kx_bf"][None, :],
        cache["ky_bf"][None, :],
        np.float32(engine.wavelength),
        np.float32(cache["semiangle_rad"]),
        np.float32(cache["ang_y_rad"]),
        np.float32(cache["ang_x_rad"]),
        np.float32(depth_A),
        np.float32(tilt_mrad[0] * 1e-3),
        np.float32(tilt_mrad[1] * 1e-3),
        coefficients,
    )
