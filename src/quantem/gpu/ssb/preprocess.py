"""Preprocessing helpers used by the SSB compute engine."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from quantem.gpu.detector import detect_bf_radius, dp_mean, virtual_image

__all__ = ["Preview", "bf_df_dpc", "dp_mean", "virtual_image"]


class Preview(NamedTuple):
    """Quick-look bundle from :func:`bf_df_dpc`.

    All arrays are CPU NumPy copies suitable for widget display. ``dpc_result``
    keeps the full SSB/DPC result object for callers that need calibration
    details or persistence metadata.
    """

    mean_dp: "np.ndarray"
    bf: "np.ndarray"
    df: "np.ndarray"
    dpc_phase: "np.ndarray"
    com_row: "np.ndarray"
    com_col: "np.ndarray"
    dpc_result: "object"
    bf_center: tuple[float, float]
    bf_radius: float
    rotation_angle_deg: float
    curl_angles_deg: "np.ndarray | None" = None
    curl_normal: "np.ndarray | None" = None
    curl_transpose: "np.ndarray | None" = None
    autofit_dpc_phase: "np.ndarray | None" = None
    autofit_com_row: "np.ndarray | None" = None
    autofit_com_col: "np.ndarray | None" = None
    autofit_rotation_angle_deg: "float | None" = None


def bf_df_dpc(
    data,
    rotation_angle_deg: float | None = None,
    df_outer_ratio: float = 3.0,
) -> Preview:
    """Mean DP + BF + DF + DPC in one call.

    This is the shared preview path used by live calibration, screening, and
    browse workflows. The full 4D block stays in its native dtype; the helper
    performs integer detector reductions and returns only small CPU products.
    """
    import cupy as cp

    from quantem.gpu.ssb.api import dpc as _dpc

    mean_dp = dp_mean(data)
    (cr, cc), bf_r = detect_bf_radius(mean_dp)
    bf = virtual_image(data, cr, cc, radius=bf_r)
    df = virtual_image(
        data, cr, cc, inner_radius=bf_r, outer_radius=bf_r * df_outer_ratio
    )
    cp.get_default_memory_pool().free_all_blocks()

    dpc_result = _dpc(
        data,
        rotation_angle_deg=rotation_angle_deg,
        verbose=False,
    )
    cp.get_default_memory_pool().free_all_blocks()

    def _as_np(arr):
        return cp.asnumpy(arr) if arr is not None else None

    return Preview(
        mean_dp=cp.asnumpy(mean_dp),
        bf=cp.asnumpy(bf),
        df=cp.asnumpy(df),
        dpc_phase=cp.asnumpy(dpc_result.phase),
        com_row=cp.asnumpy(dpc_result.com_k_row_aligned),
        com_col=cp.asnumpy(dpc_result.com_k_col_aligned),
        dpc_result=dpc_result,
        bf_center=(float(cr), float(cc)),
        bf_radius=float(bf_r),
        rotation_angle_deg=float(dpc_result.rotation_angle_deg),
        curl_angles_deg=_as_np(dpc_result.curl_angles_deg),
        curl_normal=_as_np(dpc_result.curl_normal),
        curl_transpose=_as_np(dpc_result.curl_transpose),
        autofit_dpc_phase=_as_np(dpc_result.autofit_phase),
        autofit_com_row=_as_np(dpc_result.autofit_com_k_row_aligned),
        autofit_com_col=_as_np(dpc_result.autofit_com_k_col_aligned),
        autofit_rotation_angle_deg=(
            float(dpc_result.autofit_rotation_angle_deg)
            if dpc_result.autofit_rotation_angle_deg is not None else None
        ),
    )
