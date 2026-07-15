"""Apple GPU SSB preview backend using MLX.

The input path is the chunk-backed MPS loader: each BF detector pixel is
streamed from the resident Metal chunks, transformed with MLX FFT on Apple GPU,
corrected, and accumulated without materializing the full 4D stack or using
Torch. The fixed-preview path and a small Optuna/Nelder-Mead free-fit path share
the same reconstruction/loss core.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from quantem.gpu.detector import mean_dp
from quantem.gpu.ssb.optics.physics import electron_wavelength_angstrom


@dataclass
class MpsSSBPreviewResult:
    """Result from :func:`ssb_preview` or :func:`ssb_fit`."""

    object_wave: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray
    bf_center: tuple[float, float]
    bf_radius: float
    num_bf: int
    elapsed: float
    aberrations: dict[str, float] | None = None
    loss: float | None = None
    n_trials: int | None = None
    optuna_trials: list[dict] | None = None
    refine_method: str | None = None


@dataclass
class _PreparedMpsSSB:
    """Device-resident BF FFT stack and geometry for repeated SSB loss calls."""

    mx: object
    g_qk: object
    qx: object
    qy: object
    kx_np: np.ndarray
    ky_np: np.ndarray
    dc_value: complex
    scan_shape: tuple[int, int]
    wavelength: float
    semiangle_rad: float
    ang_y_rad: float
    ang_x_rad: float
    factor: float
    dc_mask: object
    num_bf: int
    alpha_k2: object | None
    cos2_k: object | None
    sin2_k: object | None
    aperture_k: object | None
    alpha_m2: object | None
    cos2_m: object | None
    sin2_m: object | None
    ap_m: object | None
    alpha_p2: object | None
    cos2_p: object | None
    sin2_p: object | None
    ap_p: object | None


class _ArrayFrames:
    """Flat detector-column view over a 4D crop-first array.

    ``ssb_preview`` and ``ssb_fit`` stream one detector pixel over all scan
    positions.  Full no-bin MPS loads provide that through ``ChunkedFrames``;
    crop-first MPS loads return a Metal-backed ndarray-like object instead.
    This adapter gives both inputs the same ``column(row, col)`` contract.
    """

    _is_gpu_frames = True

    def __init__(self, data):
        arr = np.asarray(data)
        if arr.ndim == 4:
            self.scan_shape = (int(arr.shape[0]), int(arr.shape[1]))
            self.det_shape = (int(arr.shape[2]), int(arr.shape[3]))
            self._flat = arr.reshape(-1, *self.det_shape)
        elif arr.ndim == 3:
            self.scan_shape = None
            self.det_shape = (int(arr.shape[1]), int(arr.shape[2]))
            self._flat = arr
        else:
            raise TypeError(
                "MPS SSB preview expects 3D/4D detector data or chunk-backed "
                f"MPS data, got shape {getattr(arr, 'shape', None)}."
            )
        self.shape = tuple(int(x) for x in self._flat.shape)
        self.ndim = 3
        self.dtype = self._flat.dtype

    def __array__(self, dtype=None):
        arr = np.asarray(self._flat)
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    def reshape(self, *shape, **kwargs):
        return self._flat.reshape(*shape, **kwargs)

    def column(self, row: int, col: int) -> np.ndarray:
        return np.asarray(self._flat[:, int(row), int(col)])


def _require_mlx():
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MPS SSB preview requires MLX on Apple Silicon. Install with "
            "`python -m pip install mlx` in the Mac environment."
        ) from exc
    return mx


def _as_chunked_frames(data):
    if getattr(data, "_is_gpu_frames", False):
        return data
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    if hasattr(data, "chunks"):
        from quantem.gpu.compute.mps import ChunkedFrames

        return ChunkedFrames(data)
    if hasattr(data, "ndim") and int(getattr(data, "ndim")) in (3, 4):
        return _ArrayFrames(data)
    raise TypeError(
        "MPS SSB preview expects chunk-backed MPS data from "
        "`quantem.gpu.io.hdf5.load(..., backend='mps')` or a crop-first "
        "3D/4D MPS/NumPy array."
    )


def _scan_shape(frames) -> tuple[int, int]:
    shape = getattr(frames, "scan_shape", None)
    if shape is not None:
        return int(shape[0]), int(shape[1])
    n = int(frames.shape[0])
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError("scan_shape is required for non-square frame counts.")
    return side, side


def _spatial_frequencies(shape: tuple[int, int], sampling: tuple[float, float]):
    return (
        np.fft.fftfreq(shape[0], sampling[0]).astype(np.float32),
        np.fft.fftfreq(shape[1], sampling[1]).astype(np.float32),
    )


def _compute_geometry(mx, dx, dy, wavelength, semiangle_rad, ang_y_rad, ang_x_rad):
    dx2 = dx * dx
    dy2 = dy * dy
    r2 = dx2 + dy2
    r = mx.sqrt(r2)
    alpha = r * wavelength
    alpha2 = alpha * alpha
    inv_r2 = mx.where(r2 > 1e-30, 1.0 / r2, 0.0)
    cos2phi = (dx2 - dy2) * inv_r2
    sin2phi = 2.0 * dx * dy * inv_r2
    denom_num2 = (dx * ang_y_rad) ** 2 + (dy * ang_x_rad) ** 2
    inv_r = mx.where(r > 1e-15, 1.0 / r, 0.0)
    denom = mx.sqrt(denom_num2) * inv_r
    edge = mx.where(denom > 1e-15, (semiangle_rad - alpha) / denom + 0.5, 1.0)
    aperture = mx.clip(edge, 0.0, 1.0)
    return alpha2, cos2phi, sin2phi, aperture


def _exp_neg_i(mx, chi):
    return mx.cos(chi) - (1j * mx.sin(chi))


def _as_sampling(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    return float(value[0]), float(value[1])


def _bf_pixels(
    data,
    threshold: float,
    bf_radius: float | None,
    center_override: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float, float]:
    dp = mean_dp(data)
    _detected_center, detected_radius = _detect_bf_radius_numpy(dp)
    mask = dp > float(dp.max()) * float(threshold)
    rr, cc = np.nonzero(mask)
    if rr.size == 0:
        raise ValueError(
            f"No bright-field pixels found with threshold={threshold:.2f}."
        )
    if center_override is not None:
        center = (
            float(center_override[0]),
            float(center_override[1]),
        )
    else:
        weights = dp[rr, cc].astype(np.float32, copy=False)
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            center = (
                float((rr.astype(np.float32) * weights).sum() / weight_sum),
                float((cc.astype(np.float32) * weights).sum() / weight_sum),
            )
        else:
            center = (float(rr.mean()), float(cc.mean()))
    selected_radius = float(detected_radius if bf_radius is None else bf_radius)
    if bf_radius is not None:
        dist2 = (rr.astype(np.float32) - center[0]) ** 2 + (
            cc.astype(np.float32) - center[1]
        ) ** 2
        keep = dist2 <= float(bf_radius) ** 2
        rr = rr[keep]
        cc = cc[keep]
    if rr.size == 0:
        raise ValueError(
            f"No BF pixels selected with threshold={threshold} and radius={bf_radius}."
        )
    return (
        rr.astype(np.int32),
        cc.astype(np.int32),
        center,
        selected_radius,
        float(detected_radius),
    )


def _detect_bf_radius_numpy(
    mean_dp_array: np.ndarray,
    threshold_ratio: float = 0.1,
) -> tuple[tuple[int, int], int]:
    """NumPy mirror of :func:`quantem.gpu.detector.detect_bf_radius`."""
    dp = np.asarray(mean_dp_array, dtype=np.float32)
    if dp.ndim != 2:
        raise ValueError(f"Expected 2D diffraction pattern, got shape {dp.shape}.")
    n_k_row, n_k_col = dp.shape
    dp_max = float(np.nanmax(dp))
    if not np.isfinite(dp_max) or dp_max <= 0:
        raise ValueError("Diffraction pattern has no positive finite values.")
    mask = dp > threshold_ratio * dp_max
    if not bool(mask.any()):
        raise ValueError(f"No pixels above threshold ({threshold_ratio:.0%} of max).")
    mask_f = mask.astype(np.float32)
    total = float(mask_f.sum())
    row_coords = np.arange(n_k_row, dtype=np.float32).reshape(-1, 1)
    col_coords = np.arange(n_k_col, dtype=np.float32).reshape(1, -1)
    row_center = max(0, min(int(round(float((row_coords * mask_f).sum() / total))), n_k_row - 1))
    col_center = max(0, min(int(round(float((col_coords * mask_f).sum() / total))), n_k_col - 1))
    dr = np.arange(n_k_row, dtype=np.float32) - row_center
    dc = np.arange(n_k_col, dtype=np.float32) - col_center
    rr, cc = np.meshgrid(dr, dc, indexing="ij")
    radii = np.rint(np.sqrt(rr * rr + cc * cc)).astype(np.int32).reshape(-1)
    max_r = min(row_center, col_center, n_k_row - row_center, n_k_col - col_center)
    if max_r < 2:
        return (row_center, col_center), max(1, min(n_k_row, n_k_col) // 4)
    valid = radii < max_r
    profile = np.bincount(radii[valid], weights=dp.reshape(-1)[valid], minlength=max_r).astype(np.float32)
    counts = np.bincount(radii[valid], minlength=max_r).astype(np.float32)
    nonzero = counts > 0
    profile[nonzero] /= counts[nonzero]
    if profile.size > 5:
        sigma = 2.0
        ksize = int(6 * sigma + 1) | 1
        x = np.arange(ksize, dtype=np.float32) - ksize // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
        kernel /= kernel.sum()
        padded = np.pad(profile, ksize // 2, mode="edge")
        profile_smooth = np.convolve(padded, kernel, mode="valid")[:profile.size]
        half_max = float(profile_smooth[:5].mean()) * 0.5
        below_half = np.flatnonzero(profile_smooth < half_max)
        radius = int(below_half[0]) if below_half.size else int(profile.size) // 2
    else:
        radius = min(n_k_row, n_k_col) // 4
    return (row_center, col_center), max(1, int(radius))


def _ranges_from_start(
    start: dict[str, float],
    search_ranges: dict | None,
) -> dict[str, tuple[float, float] | float]:
    if search_ranges is not None:
        return dict(search_ranges)
    return {
        "C10_nm": (-400.0, 400.0),
        "C12_nm": (0.0, 100.0),
        "phi12_deg": (-90.0, 90.0),
    }


def _suggest_or_fixed(trial, ranges: dict, key: str, default: float) -> float:
    value = ranges.get(key, default)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if lo == hi:
            return lo
        return float(trial.suggest_float(key, lo, hi))
    return float(value)


def _loss_from_phase_stack(mx, obj_chunk):
    phase = mx.arctan2(mx.imag(obj_chunk), mx.real(obj_chunk))
    return mx.sum(phase, axis=0), mx.sum(phase * phase, axis=0)


@lru_cache(maxsize=16)
def _phase_sums_kernel(batch: int, chunk: int, ny: int, nx: int):
    mx = _require_mlx()
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        uint total = BATCH * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint batch = elem / PLANE;
        uint pixel = elem - batch * PLANE;
        size_t base = ((size_t)batch * (size_t)CHUNK * (size_t)PLANE) + (size_t)pixel;
        float s = 0.0f;
        float sq = 0.0f;
        for (uint bf = 0; bf < CHUNK; ++bf) {{
            auto z = obj[base + (size_t)bf * (size_t)PLANE];
            float a = metal::atan2(z.imag, z.real);
            s += a;
            sq += a * a;
        }}
        sum_out[elem] = s;
        sumsq_out[elem] = sq;
    """
    return mx.fast.metal_kernel(
        name=f"ssb_phase_sums_n{int(batch)}_b{int(chunk)}_{int(ny)}_{int(nx)}",
        input_names=["obj"],
        output_names=["sum_out", "sumsq_out"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _phase_sums_from_complex(mx, obj_chunk):
    """Metal fused atan2/sum/sumsq over BF pixels for a chunked object stack."""
    shape = tuple(int(x) for x in obj_chunk.shape)
    if len(shape) == 3:
        chunk, ny, nx = shape
        obj = obj_chunk[None, :, :, :]
        squeeze = True
        batch = 1
    elif len(shape) == 4:
        batch, chunk, ny, nx = shape
        obj = obj_chunk
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D object chunk, got shape {shape}.")
    kernel = _phase_sums_kernel(int(batch), int(chunk), int(ny), int(nx))
    outputs = kernel(
        inputs=[obj],
        template=[],
        grid=(int(batch) * int(ny) * int(nx), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(int(batch), int(ny), int(nx)), (int(batch), int(ny), int(nx))],
        output_dtypes=[mx.float32, mx.float32],
    )
    if squeeze:
        return outputs[0][0], outputs[1][0]
    return outputs[0], outputs[1]


@lru_cache(maxsize=16)
def _corrected_kernel(batch: int, chunk: int, ny: int, nx: int):
    mx = _require_mlx()
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint BATCH = {int(batch)};
        constexpr uint CHUNK = {int(chunk)};
        constexpr uint NY = {int(ny)};
        constexpr uint NX = {int(nx)};
        constexpr uint PLANE = NY * NX;
        uint total = BATCH * CHUNK * PLANE;
        if (elem >= total) {{
            return;
        }}
        uint batch = elem / (CHUNK * PLANE);
        uint rem = elem - batch * CHUNK * PLANE;
        uint bf = rem / PLANE;
        uint pixel = rem - bf * PLANE;
        uint geom_idx = bf * PLANE + pixel;

        if (pixel == 0) {{
            corrected[elem].real = scalars[1];
            corrected[elem].imag = scalars[2];
            return;
        }}

        float c10v = c10[batch];
        float c12v = c12[batch];
        float cos2v = cos2phi12[batch];
        float sin2v = sin2phi12[batch];
        float factor = scalars[0];

        float cos_term_k = cos2_k[bf] * cos2v + sin2_k[bf] * sin2v;
        float chi_k = factor * alpha_k2[bf] * (c12v * cos_term_k + c10v);
        float pk_amp = aperture_k[bf];
        float pkr = pk_amp * metal::cos(chi_k);
        float pki = -pk_amp * metal::sin(chi_k);

        float cos_term_m = cos2_m[geom_idx] * cos2v + sin2_m[geom_idx] * sin2v;
        float chi_m = factor * alpha_m2[geom_idx] * (c12v * cos_term_m + c10v);
        float pm_amp = ap_m[geom_idx];
        float pmr = pm_amp * metal::cos(chi_m);
        float pmi = -pm_amp * metal::sin(chi_m);

        float cos_term_p = cos2_p[geom_idx] * cos2v + sin2_p[geom_idx] * sin2v;
        float chi_p = factor * alpha_p2[geom_idx] * (c12v * cos_term_p + c10v);
        float pp_amp = ap_p[geom_idx];
        float ppr = pp_amp * metal::cos(chi_p);
        float ppi = -pp_amp * metal::sin(chi_p);

        float gamma_r = (pmr * pkr + pmi * pki) - (ppr * pkr + ppi * pki);
        float gamma_i = (pmi * pkr - pmr * pki) - (ppr * pki - ppi * pkr);
        float mag = metal::sqrt(gamma_r * gamma_r + gamma_i * gamma_i);
        float inv_mag = 1.0f / metal::max(mag, 1.0e-8f);
        float conj_gamma_r = gamma_r * inv_mag;
        float conj_gamma_i = -gamma_i * inv_mag;

        auto gz = g[geom_idx];
        corrected[elem].real = gz.real * conj_gamma_r - gz.imag * conj_gamma_i;
        corrected[elem].imag = gz.real * conj_gamma_i + gz.imag * conj_gamma_r;
    """
    return mx.fast.metal_kernel(
        name=f"ssb_corrected_n{int(batch)}_b{int(chunk)}_{int(ny)}_{int(nx)}",
        input_names=[
            "g",
            "alpha_k2",
            "cos2_k",
            "sin2_k",
            "aperture_k",
            "alpha_m2",
            "cos2_m",
            "sin2_m",
            "ap_m",
            "alpha_p2",
            "cos2_p",
            "sin2_p",
            "ap_p",
            "c10",
            "c12",
            "cos2phi12",
            "sin2phi12",
            "scalars",
        ],
        output_names=["corrected"],
        source=source,
        compile_options={"math_mode": "fast"},
    )


def _corrected_from_cached_geometry(
    prepared: _PreparedMpsSSB,
    *,
    start: int,
    stop: int,
    c10,
    c12,
    cos2phi12,
    sin2phi12,
):
    """Fused Metal correction for cached-geometry MPS sparse objectives."""
    mx = prepared.mx
    batch = int(c10.shape[0])
    chunk = int(stop) - int(start)
    ny, nx = prepared.scan_shape
    kernel = _corrected_kernel(batch, chunk, int(ny), int(nx))
    scalars = mx.array(
        [
            float(prepared.factor),
            float(prepared.dc_value.real),
            float(prepared.dc_value.imag),
        ],
        dtype=mx.float32,
    )
    outputs = kernel(
        inputs=[
            prepared.g_qk[start:stop],
            prepared.alpha_k2[start:stop],
            prepared.cos2_k[start:stop],
            prepared.sin2_k[start:stop],
            prepared.aperture_k[start:stop],
            prepared.alpha_m2[start:stop],
            prepared.cos2_m[start:stop],
            prepared.sin2_m[start:stop],
            prepared.ap_m[start:stop],
            prepared.alpha_p2[start:stop],
            prepared.cos2_p[start:stop],
            prepared.sin2_p[start:stop],
            prepared.ap_p[start:stop],
            c10,
            c12,
            cos2phi12,
            sin2phi12,
            scalars,
        ],
        template=[],
        grid=(batch * chunk * int(ny) * int(nx), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, chunk, int(ny), int(nx))],
        output_dtypes=[mx.complex64],
    )
    return outputs[0]


def _prepare_selection(
    frames,
    *,
    scan_shape: tuple[int, int],
    det_shape: tuple[int, int],
    bf_row: np.ndarray,
    bf_col: np.ndarray,
    center: tuple[float, float],
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling: tuple[float, float],
    det_sampling: tuple[float, float],
    rotation_angle_deg: float,
    chunk_bf: int,
) -> _PreparedMpsSSB:
    """Precompute BF-column FFTs and static geometry for MPS SSB fitting."""
    mx = _require_mlx()
    wavelength = float(electron_wavelength_angstrom(float(voltage_kV) * 1e3))
    reciprocal_sampling = (
        det_sampling[0] * 1e-3 / wavelength,
        det_sampling[1] * 1e-3 / wavelength,
    )
    sampling = (
        1.0 / (reciprocal_sampling[0] * det_shape[0]),
        1.0 / (reciprocal_sampling[1] * det_shape[1]),
    )
    q_row_np, q_col_np = _spatial_frequencies(scan_shape, scan_sampling)

    recip_y = 1.0 / (sampling[0] * det_shape[0])
    recip_x = 1.0 / (sampling[1] * det_shape[1])
    kx_np = (bf_row.astype(np.float32) - center[0]) * recip_y
    ky_np = (bf_col.astype(np.float32) - center[1]) * recip_x
    if rotation_angle_deg:
        angle = math.radians(-float(rotation_angle_deg))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        kx_np, ky_np = kx_np * cos_a + ky_np * sin_a, -kx_np * sin_a + ky_np * cos_a
    kx_np = np.asarray(kx_np, dtype=np.float32)
    ky_np = np.asarray(ky_np, dtype=np.float32)

    g_chunks = []
    chunk_bf = max(1, int(chunk_bf))
    for start in range(0, int(bf_row.size), chunk_bf):
        stop = min(start + chunk_bf, int(bf_row.size))
        rows = bf_row[start:stop]
        cols = bf_col[start:stop]
        stack_np = np.stack(
            [frames.column(int(r), int(c)).reshape(scan_shape) for r, c in zip(rows, cols)],
            axis=0,
        ).astype(np.complex64, copy=False)
        g_chunk = mx.fft.fft2(mx.array(stack_np))
        mx.eval(g_chunk)
        g_chunks.append(g_chunk)
    g_qk = g_chunks[0] if len(g_chunks) == 1 else mx.concatenate(g_chunks, axis=0)
    mx.eval(g_qk)
    dc_value = complex(np.asarray(g_qk[:, 0, 0]).mean())

    qx = mx.array(q_row_np, dtype=mx.float32)[None, :, None]
    qy = mx.array(q_col_np, dtype=mx.float32)[None, None, :]
    semiangle_rad = float(semiangle_mrad) * 1e-3
    ang_y_rad = float(det_sampling[0]) * 1e-3
    ang_x_rad = float(det_sampling[1]) * 1e-3
    alpha_k2 = cos2_k = sin2_k = aperture_k = None
    alpha_m2 = cos2_m = sin2_m = ap_m = None
    alpha_p2 = cos2_p = sin2_p = ap_p = None
    # Cache static geometry when the selected BF set is small enough.  For the
    # Samsung bf_radius=5 path this is ~650 MB of float32 geometry and removes
    # repeated sqrt/aperture work from every optimizer batch.
    geometry_values = int(bf_row.size) * int(scan_shape[0]) * int(scan_shape[1])
    if geometry_values <= 32_000_000:
        kx = mx.array(kx_np, dtype=mx.float32)[:, None, None]
        ky = mx.array(ky_np, dtype=mx.float32)[:, None, None]
        alpha_k2, cos2_k, sin2_k, aperture_k = _compute_geometry(
            mx, kx, ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        alpha_m2, cos2_m, sin2_m, ap_m = _compute_geometry(
            mx, qx - kx, qy - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        alpha_p2, cos2_p, sin2_p, ap_p = _compute_geometry(
            mx, qx + kx, qy + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        mx.eval(
            alpha_k2, cos2_k, sin2_k, aperture_k,
            alpha_m2, cos2_m, sin2_m, ap_m,
            alpha_p2, cos2_p, sin2_p, ap_p,
        )

    dc_mask_np = np.zeros(scan_shape, dtype=bool)
    dc_mask_np[0, 0] = True
    return _PreparedMpsSSB(
        mx=mx,
        g_qk=g_qk,
        qx=qx,
        qy=qy,
        kx_np=kx_np,
        ky_np=ky_np,
        dc_value=dc_value,
        scan_shape=scan_shape,
        wavelength=wavelength,
        semiangle_rad=semiangle_rad,
        ang_y_rad=ang_y_rad,
        ang_x_rad=ang_x_rad,
        factor=math.pi / wavelength,
        dc_mask=mx.array(dc_mask_np),
        num_bf=int(bf_row.size),
        alpha_k2=alpha_k2,
        cos2_k=cos2_k,
        sin2_k=sin2_k,
        aperture_k=aperture_k,
        alpha_m2=alpha_m2,
        cos2_m=cos2_m,
        sin2_m=sin2_m,
        ap_m=ap_m,
        alpha_p2=alpha_p2,
        cos2_p=cos2_p,
        sin2_p=sin2_p,
        ap_p=ap_p,
    )


def _reconstruct_prepared(
    prepared: _PreparedMpsSSB,
    *,
    C10: float,
    C12: float,
    phi12: float,
    chunk_bf: int,
    compute_loss: bool,
    compute_object: bool,
) -> tuple[np.ndarray | None, float | None, np.ndarray | None]:
    """Run SSB correction from a prepared BF FFT stack."""
    mx = prepared.mx
    accumulator = (
        mx.zeros(prepared.scan_shape, dtype=mx.complex64)
        if compute_object else None
    )
    # CUDA's fixed SSB output is the mean of per-BF phase images, not the
    # phase of the averaged complex object wave. Keep that contract for parity.
    phase_sum = mx.zeros(prepared.scan_shape, dtype=mx.float32)
    phase_sumsq = mx.zeros(prepared.scan_shape, dtype=mx.float32) if compute_loss else None
    cos2phi12 = math.cos(2.0 * float(phi12))
    sin2phi12 = math.sin(2.0 * float(phi12))
    chunk_bf = max(1, int(chunk_bf))

    for start in range(0, prepared.num_bf, chunk_bf):
        stop = min(start + chunk_bf, prepared.num_bf)
        g_qk = prepared.g_qk[start:stop]
        if prepared.alpha_k2 is not None:
            alpha_k2 = prepared.alpha_k2[start:stop]
            cos2_k = prepared.cos2_k[start:stop]
            sin2_k = prepared.sin2_k[start:stop]
            aperture_k = prepared.aperture_k[start:stop]
        else:
            kx = mx.array(prepared.kx_np[start:stop], dtype=mx.float32)[:, None, None]
            ky = mx.array(prepared.ky_np[start:stop], dtype=mx.float32)[:, None, None]
            alpha_k2, cos2_k, sin2_k, aperture_k = _compute_geometry(
                mx,
                kx,
                ky,
                prepared.wavelength,
                prepared.semiangle_rad,
                prepared.ang_y_rad,
                prepared.ang_x_rad,
            )
        cos_term_k = cos2_k * cos2phi12 + sin2_k * sin2phi12
        chi_k = prepared.factor * alpha_k2 * (
            float(C12) * cos_term_k + float(C10)
        )
        pk = aperture_k * _exp_neg_i(mx, chi_k)

        if prepared.alpha_m2 is not None:
            alpha_m2 = prepared.alpha_m2[start:stop]
            cos2_m = prepared.cos2_m[start:stop]
            sin2_m = prepared.sin2_m[start:stop]
            ap_m = prepared.ap_m[start:stop]
            alpha_p2 = prepared.alpha_p2[start:stop]
            cos2_p = prepared.cos2_p[start:stop]
            sin2_p = prepared.sin2_p[start:stop]
            ap_p = prepared.ap_p[start:stop]
        else:
            alpha_m2, cos2_m, sin2_m, ap_m = _compute_geometry(
                mx,
                prepared.qx - kx,
                prepared.qy - ky,
                prepared.wavelength,
                prepared.semiangle_rad,
                prepared.ang_y_rad,
                prepared.ang_x_rad,
            )
            alpha_p2, cos2_p, sin2_p, ap_p = _compute_geometry(
                mx,
                prepared.qx + kx,
                prepared.qy + ky,
                prepared.wavelength,
                prepared.semiangle_rad,
                prepared.ang_y_rad,
                prepared.ang_x_rad,
            )
        chi_m = prepared.factor * alpha_m2 * (
            float(C12) * (cos2_m * cos2phi12 + sin2_m * sin2phi12) + float(C10)
        )
        chi_p = prepared.factor * alpha_p2 * (
            float(C12) * (cos2_p * cos2phi12 + sin2_p * sin2phi12) + float(C10)
        )
        pm = ap_m * _exp_neg_i(mx, chi_m)
        pp = ap_p * _exp_neg_i(mx, chi_p)
        gamma = pm * mx.conjugate(pk) - mx.conjugate(pp) * pk
        gamma = gamma / mx.maximum(mx.abs(gamma), 1e-8)
        corrected = g_qk * mx.conjugate(gamma)
        corrected = mx.where(
            prepared.dc_mask[None, :, :],
            mx.array(prepared.dc_value, dtype=mx.complex64),
            corrected,
        )
        obj_chunk = mx.fft.ifft2(corrected)
        if compute_object:
            accumulator = accumulator + mx.sum(obj_chunk, axis=0)
        chunk_sum, chunk_sumsq = _loss_from_phase_stack(mx, obj_chunk)
        phase_sum = phase_sum + chunk_sum
        if compute_loss:
            phase_sumsq = phase_sumsq + chunk_sumsq
        mx.eval(
            *[
                arr for arr in (accumulator, phase_sum, phase_sumsq)
                if arr is not None
            ]
        )

    object_wave = None
    if compute_object:
        object_wave_mx = accumulator / prepared.num_bf
        mx.eval(object_wave_mx)
        object_wave = np.asarray(object_wave_mx).astype(np.complex64, copy=False)
    mean_phase_mx = phase_sum / prepared.num_bf
    mx.eval(mean_phase_mx)
    mean_phase = np.asarray(mean_phase_mx).astype(np.float32, copy=False)
    loss = None
    if compute_loss:
        var_per_pixel = phase_sumsq / prepared.num_bf - mean_phase_mx * mean_phase_mx
        loss = float(np.asarray(mx.mean(var_per_pixel)))
    return object_wave, loss, mean_phase


def _cuda_sparse_row_indices(scan_shape: tuple[int, int]) -> np.ndarray:
    """Rows matching CUDA sparse optimizer staging for supported sizes."""
    ny, nx = (int(scan_shape[0]), int(scan_shape[1]))
    if ny != nx or ny not in (128, 256, 512):
        raise ValueError(
            "MPS SSB fit currently supports CUDA-parity sparse optimizer "
            f"objective only for square 128x128, 256x256, or 512x512 scans; got {scan_shape}. "
            "Use ssb_preview for fixed-aberration reconstruction or add a "
            "size-specific sparse objective before enabling free-fit."
        )
    offsets = range(8) if ny == 128 else range(4) if ny == 256 else range(2)
    groups = ny // 8
    return np.asarray(
        [group * 8 + offset for group in range(groups) for offset in offsets],
        dtype=np.int32,
    )


def _cuda_sparse_row_mask(mx, scan_shape: tuple[int, int]):
    """Mask matching CUDA sparse optimizer row staging for supported sizes."""
    ny = int(scan_shape[0])
    rows = np.zeros((ny,), dtype=np.float32)
    rows[_cuda_sparse_row_indices(scan_shape)] = 1.0
    return mx.array(rows, dtype=mx.float32)[None, None, :, None]


def _reconstruct_prepared_batch_cuda_sparse(
    prepared: _PreparedMpsSSB,
    *,
    C10: np.ndarray,
    C12: np.ndarray,
    phi12: np.ndarray,
    chunk_bf: int,
) -> np.ndarray:
    """Evaluate the CUDA sparse-row optimizer objective on MPS."""

    mx = prepared.mx
    c10_np = np.asarray(C10, dtype=np.float32).reshape(-1)
    c12_np = np.asarray(C12, dtype=np.float32).reshape(-1)
    phi_np = np.asarray(phi12, dtype=np.float32).reshape(-1)
    if c10_np.size == 0:
        return np.empty((0,), dtype=np.float32)
    if c12_np.size != c10_np.size or phi_np.size != c10_np.size:
        raise ValueError("C10, C12, and phi12 must have matching lengths.")

    batch = int(c10_np.size)
    phase_sum = mx.zeros((batch, *prepared.scan_shape), dtype=mx.float32)
    phase_sumsq = mx.zeros((batch, *prepared.scan_shape), dtype=mx.float32)
    c10_values = mx.array(c10_np, dtype=mx.float32)
    c12_values = mx.array(c12_np, dtype=mx.float32)
    cos2phi12_values = mx.array(np.cos(2.0 * phi_np).astype(np.float32))
    sin2phi12_values = mx.array(np.sin(2.0 * phi_np).astype(np.float32))
    c10 = c10_values[:, None, None, None]
    c12 = c12_values[:, None, None, None]
    cos2phi12 = cos2phi12_values[:, None, None, None]
    sin2phi12 = sin2phi12_values[:, None, None, None]
    chunk_bf = max(1, int(chunk_bf))
    row_mask = _cuda_sparse_row_mask(mx, prepared.scan_shape)

    for start in range(0, prepared.num_bf, chunk_bf):
        stop = min(start + chunk_bf, prepared.num_bf)
        if prepared.alpha_k2 is not None:
            corrected = _corrected_from_cached_geometry(
                prepared,
                start=start,
                stop=stop,
                c10=c10_values,
                c12=c12_values,
                cos2phi12=cos2phi12_values,
                sin2phi12=sin2phi12_values,
            )
        else:
            g_qk = prepared.g_qk[start:stop][None, :, :, :]
            kx = mx.array(prepared.kx_np[start:stop], dtype=mx.float32)[None, :, None, None]
            ky = mx.array(prepared.ky_np[start:stop], dtype=mx.float32)[None, :, None, None]
            alpha_k2, cos2_k, sin2_k, aperture_k = _compute_geometry(
                mx,
                kx,
                ky,
                prepared.wavelength,
                prepared.semiangle_rad,
                prepared.ang_y_rad,
                prepared.ang_x_rad,
            )
            cos_term_k = cos2_k * cos2phi12 + sin2_k * sin2phi12
            chi_k = prepared.factor * alpha_k2 * (c12 * cos_term_k + c10)
            pk = aperture_k * _exp_neg_i(mx, chi_k)

            alpha_m2, cos2_m, sin2_m, ap_m = _compute_geometry(
                mx,
                prepared.qx[None, :, :, :] - kx,
                prepared.qy[None, :, :, :] - ky,
                prepared.wavelength,
                prepared.semiangle_rad,
                prepared.ang_y_rad,
                prepared.ang_x_rad,
            )
            alpha_p2, cos2_p, sin2_p, ap_p = _compute_geometry(
                mx,
                prepared.qx[None, :, :, :] + kx,
                prepared.qy[None, :, :, :] + ky,
                prepared.wavelength,
                prepared.semiangle_rad,
                prepared.ang_y_rad,
                prepared.ang_x_rad,
            )
            chi_m = prepared.factor * alpha_m2 * (
                c12 * (cos2_m * cos2phi12 + sin2_m * sin2phi12) + c10
            )
            chi_p = prepared.factor * alpha_p2 * (
                c12 * (cos2_p * cos2phi12 + sin2_p * sin2phi12) + c10
            )
            pm = ap_m * _exp_neg_i(mx, chi_m)
            pp = ap_p * _exp_neg_i(mx, chi_p)
            gamma = pm * mx.conjugate(pk) - mx.conjugate(pp) * pk
            gamma = gamma / mx.maximum(mx.abs(gamma), 1e-8)
            corrected = g_qk * mx.conjugate(gamma)
            corrected = mx.where(
                prepared.dc_mask[None, None, :, :],
                mx.array(prepared.dc_value, dtype=mx.complex64),
                corrected,
            )
        if prepared.scan_shape == (256, 256):
            # CUDA's 256 sparse row kernel writes first-stage rows transposed
            # as out[pos, row] before the variance kernel performs stage two.
            row_fft = mx.fft.ifft(corrected * row_mask, axis=-1)
            obj_chunk = mx.fft.ifft(mx.swapaxes(row_fft, -1, -2), axis=-1)
        else:
            obj_chunk = mx.fft.ifft2(corrected * row_mask)
        chunk_sum, chunk_sumsq = _phase_sums_from_complex(mx, obj_chunk)
        phase_sum = phase_sum + chunk_sum
        phase_sumsq = phase_sumsq + chunk_sumsq
        mx.eval(phase_sum, phase_sumsq)

    mean_phase = phase_sum / prepared.num_bf
    var_per_pixel = phase_sumsq / prepared.num_bf - mean_phase * mean_phase
    losses = mx.mean(var_per_pixel, axis=(1, 2))
    mx.eval(losses)
    return np.asarray(losses).astype(np.float32, copy=False)


def _nelder_mead_refine(
    best: dict[str, float],
    best_loss: float,
    evaluate,
    *,
    lock: set[str],
    xatol: float = 0.1,
    fatol: float = 1e-8,
    max_iter: int = 300,
) -> tuple[dict[str, float], float]:
    """Pure-Python Nelder-Mead matching the CUDA optimizer's simplex policy."""
    keys = [key for key in ("C10", "C12", "phi12") if key not in lock]
    if not keys:
        return dict(best), float(best_loss)
    x0 = np.array([best[key] for key in keys], dtype=np.float64)
    n = int(x0.size)
    simplex = np.empty((n + 1, n), dtype=np.float64)
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1, i] += max(abs(x0[i]) * 0.05, 0.00025)

    def params_from_x(x: np.ndarray) -> dict[str, float]:
        params = dict(best)
        for i, key in enumerate(keys):
            value = float(x[i])
            if key == "C12":
                value = max(0.0, value)
            params[key] = value
        return params

    f_values = np.empty(n + 1, dtype=np.float64)
    f_values[0] = float(best_loss)
    for i in range(1, n + 1):
        params = params_from_x(simplex[i])
        f_values[i] = evaluate(params)

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    for _ in range(max_iter):
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]
        x_spread = float(np.max(np.abs(simplex[-1] - simplex[0])))
        f_spread = float(abs(f_values[-1] - f_values[0]))
        if x_spread < xatol and f_spread < fatol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = evaluate(params_from_x(x_r))

        if f_values[0] <= f_r < f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
            continue

        if f_r < f_values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = evaluate(params_from_x(x_e))
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
            continue

        if f_r < f_values[-1]:
            x_c = centroid + rho * (x_r - centroid)
            f_c = evaluate(params_from_x(x_c))
            if f_c <= f_r:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue
        else:
            x_c = centroid - rho * (centroid - simplex[-1])
            f_c = evaluate(params_from_x(x_c))
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
                continue

        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            f_values[i] = evaluate(params_from_x(simplex[i]))

    best_idx = int(np.argmin(f_values))
    return params_from_x(simplex[best_idx]), float(f_values[best_idx])


def _reconstruct_selection(
    frames,
    *,
    scan_shape: tuple[int, int],
    det_shape: tuple[int, int],
    bf_row: np.ndarray,
    bf_col: np.ndarray,
    center: tuple[float, float],
    radius: float,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling: tuple[float, float],
    det_sampling: tuple[float, float],
    C10: float,
    C12: float,
    phi12: float,
    rotation_angle_deg: float,
    chunk_bf: int,
    compute_loss: bool,
    verbose: bool,
) -> tuple[np.ndarray, float | None, np.ndarray]:
    mx = _require_mlx()
    wavelength = float(electron_wavelength_angstrom(float(voltage_kV) * 1e3))
    reciprocal_sampling = (
        det_sampling[0] * 1e-3 / wavelength,
        det_sampling[1] * 1e-3 / wavelength,
    )
    sampling = (
        1.0 / (reciprocal_sampling[0] * det_shape[0]),
        1.0 / (reciprocal_sampling[1] * det_shape[1]),
    )
    q_row_np, q_col_np = _spatial_frequencies(scan_shape, scan_sampling)

    recip_y = 1.0 / (sampling[0] * det_shape[0])
    recip_x = 1.0 / (sampling[1] * det_shape[1])
    kx_np = (bf_row.astype(np.float32) - center[0]) * recip_y
    ky_np = (bf_col.astype(np.float32) - center[1]) * recip_x
    if rotation_angle_deg:
        angle = math.radians(-float(rotation_angle_deg))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        kx_np, ky_np = kx_np * cos_a + ky_np * sin_a, -kx_np * sin_a + ky_np * cos_a
    kx_np = np.asarray(kx_np, dtype=np.float32)
    ky_np = np.asarray(ky_np, dtype=np.float32)

    qx = mx.array(q_row_np, dtype=mx.float32)[None, :, None]
    qy = mx.array(q_col_np, dtype=mx.float32)[None, None, :]
    dc_value = complex(0.0)
    accumulator = mx.zeros(scan_shape, dtype=mx.complex64)
    # Match CUDA fixed-preview semantics: mean of per-BF phases.
    phase_sum = mx.zeros(scan_shape, dtype=mx.float32)
    phase_sumsq = mx.zeros(scan_shape, dtype=mx.float32) if compute_loss else None
    semiangle_rad = float(semiangle_mrad) * 1e-3
    ang_y_rad = float(det_sampling[0]) * 1e-3
    ang_x_rad = float(det_sampling[1]) * 1e-3
    factor = math.pi / wavelength
    cos2phi12 = math.cos(2.0 * float(phi12))
    sin2phi12 = math.sin(2.0 * float(phi12))
    dc_mask_np = np.zeros(scan_shape, dtype=bool)
    dc_mask_np[0, 0] = True
    dc_mask = mx.array(dc_mask_np)

    for start in range(0, int(bf_row.size), int(chunk_bf)):
        stop = min(start + int(chunk_bf), int(bf_row.size))
        rows = bf_row[start:stop]
        cols = bf_col[start:stop]
        stack_np = np.stack(
            [frames.column(int(r), int(c)).reshape(scan_shape) for r, c in zip(rows, cols)],
            axis=0,
        ).astype(np.complex64, copy=False)
        stack = mx.array(stack_np)
        g_qk = mx.fft.fft2(stack)
        if start == 0:
            dc_value = complex(np.asarray(g_qk[:, 0, 0]).mean())

        kx = mx.array(kx_np[start:stop], dtype=mx.float32)[:, None, None]
        ky = mx.array(ky_np[start:stop], dtype=mx.float32)[:, None, None]
        alpha_k2, cos2_k, sin2_k, aperture_k = _compute_geometry(
            mx, kx, ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        cos_term_k = cos2_k * cos2phi12 + sin2_k * sin2phi12
        chi_k = factor * alpha_k2 * (float(C12) * cos_term_k + float(C10))
        pk = aperture_k * _exp_neg_i(mx, chi_k)

        alpha_m2, cos2_m, sin2_m, ap_m = _compute_geometry(
            mx, qx - kx, qy - ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        alpha_p2, cos2_p, sin2_p, ap_p = _compute_geometry(
            mx, qx + kx, qy + ky, wavelength, semiangle_rad, ang_y_rad, ang_x_rad,
        )
        chi_m = factor * alpha_m2 * (
            float(C12) * (cos2_m * cos2phi12 + sin2_m * sin2phi12) + float(C10)
        )
        chi_p = factor * alpha_p2 * (
            float(C12) * (cos2_p * cos2phi12 + sin2_p * sin2phi12) + float(C10)
        )
        pm = ap_m * _exp_neg_i(mx, chi_m)
        pp = ap_p * _exp_neg_i(mx, chi_p)
        gamma = pm * mx.conjugate(pk) - mx.conjugate(pp) * pk
        gamma = gamma / mx.maximum(mx.abs(gamma), 1e-8)
        corrected = g_qk * mx.conjugate(gamma)
        corrected = mx.where(dc_mask[None, :, :], mx.array(dc_value, dtype=mx.complex64), corrected)
        obj_chunk = mx.fft.ifft2(corrected)
        accumulator = accumulator + mx.sum(obj_chunk, axis=0)
        chunk_sum, chunk_sumsq = _loss_from_phase_stack(mx, obj_chunk)
        phase_sum = phase_sum + chunk_sum
        if compute_loss:
            phase_sumsq = phase_sumsq + chunk_sumsq
        mx.eval(
            *[
                arr for arr in (accumulator, phase_sum, phase_sumsq)
                if arr is not None
            ]
        )
        if verbose:
            print(f"MPS SSB BF {stop}/{bf_row.size}")

    object_wave_mx = accumulator / int(bf_row.size)
    loss = None
    mean_phase_mx = phase_sum / int(bf_row.size)
    if compute_loss:
        var_per_pixel = phase_sumsq / int(bf_row.size) - mean_phase_mx * mean_phase_mx
        loss = float(np.asarray(mx.mean(var_per_pixel)))
    mx.eval(object_wave_mx, mean_phase_mx)
    object_wave = np.asarray(object_wave_mx).astype(np.complex64, copy=False)
    mean_phase = np.asarray(mean_phase_mx).astype(np.float32, copy=False)
    return object_wave, loss, mean_phase


def ssb_preview(
    data,
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | tuple[float, float],
    det_sampling: float | tuple[float, float] | None = None,
    C10: float = 0.0,
    C12: float = 0.0,
    phi12: float = 0.0,
    rotation_angle_deg: float = 0.0,
    bf_intensity_threshold: float = 0.5,
    bf_center: tuple[float, float] | None = None,
    bf_radius: float | None = None,
    chunk_bf: int = 16,
    verbose: bool = False,
    compute_loss: bool = False,
) -> MpsSSBPreviewResult:
    """Reconstruct an SSB preview on Apple GPU without CuPy or Torch.

    Parameters mirror the CUDA SSB constructor where possible. The output is
    copied back to NumPy for display/review.
    """

    t0 = time.perf_counter()
    frames = _as_chunked_frames(data)
    scan_shape = _scan_shape(frames)
    det_shape = tuple(int(x) for x in frames.shape[-2:])
    scan_sampling = _as_sampling(scan_sampling_A)

    bf_row, bf_col, center, radius, detected_radius = _bf_pixels(
        frames,
        bf_intensity_threshold,
        bf_radius,
        center_override=bf_center,
    )
    if det_sampling is None:
        det_px = (2.0 * float(semiangle_mrad)) / detected_radius
        det_sampling = (det_px, det_px)
    else:
        det_sampling = _as_sampling(det_sampling)
    object_wave, loss, phase = _reconstruct_selection(
        frames,
        scan_shape=scan_shape,
        det_shape=det_shape,
        bf_row=bf_row,
        bf_col=bf_col,
        center=center,
        radius=radius,
        voltage_kV=voltage_kV,
        semiangle_mrad=semiangle_mrad,
        scan_sampling=scan_sampling,
        det_sampling=det_sampling,
        C10=C10,
        C12=C12,
        phi12=phi12,
        rotation_angle_deg=rotation_angle_deg,
        chunk_bf=chunk_bf,
        compute_loss=compute_loss,
        verbose=verbose,
    )
    amplitude = np.abs(object_wave).astype(np.float32)
    return MpsSSBPreviewResult(
        object_wave=object_wave,
        phase=phase,
        amplitude=amplitude,
        bf_center=center,
        bf_radius=radius,
        num_bf=int(bf_row.size),
        elapsed=time.perf_counter() - t0,
        aberrations={"C10": float(C10), "C12": float(C12), "phi12": float(phi12)},
        loss=loss,
    )


def ssb_fit(
    data,
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | tuple[float, float],
    det_sampling: float | tuple[float, float] | None = None,
    aberrations: dict | None = None,
    search_ranges: dict | None = None,
    n_trials: int = 24,
    refine: str | None = "nmead",
    refine_lock: list[str] | None = None,
    rotation_angle_deg: float = 0.0,
    bf_intensity_threshold: float = 0.5,
    bf_center: tuple[float, float] | None = None,
    bf_radius: float | None = None,
    chunk_bf: int = 16,
    optuna_batch_size: int = 16,
    seed: int = 42,
    verbose: bool = False,
) -> MpsSSBPreviewResult:
    """Free-fit C10/C12/phi12 on Apple GPU, then reconstruct the best SSB phase.

    This is a compact MLX optimizer for Mac workflows. For supported SSB scan
    sizes it evaluates the same sparse-row phase-variance objective as the CUDA
    SSB optimizer.
    """
    _require_mlx()
    import optuna

    t0 = time.perf_counter()
    frames = _as_chunked_frames(data)
    scan_shape = _scan_shape(frames)
    det_shape = tuple(int(x) for x in frames.shape[-2:])
    scan_sampling = _as_sampling(scan_sampling_A)
    bf_row, bf_col, center, radius, detected_radius = _bf_pixels(
        frames,
        bf_intensity_threshold,
        bf_radius,
        center_override=bf_center,
    )
    if det_sampling is None:
        det_px = (2.0 * float(semiangle_mrad)) / detected_radius
        det_sampling = (det_px, det_px)
    else:
        det_sampling = _as_sampling(det_sampling)

    prepared = _prepare_selection(
        frames,
        scan_shape=scan_shape,
        det_shape=det_shape,
        bf_row=bf_row,
        bf_col=bf_col,
        center=center,
        voltage_kV=voltage_kV,
        semiangle_mrad=semiangle_mrad,
        scan_sampling=scan_sampling,
        det_sampling=det_sampling,
        rotation_angle_deg=rotation_angle_deg,
        chunk_bf=chunk_bf,
    )

    start = {"C10": 0.0, "C12": 50.0, "phi12": 0.0}
    if aberrations:
        start.update({k: float(v) for k, v in aberrations.items() if k in start})
    ranges = _ranges_from_start(start, search_ranges)
    trials: list[dict] = []

    def evaluate(C10: float, C12: float, phi12: float) -> float:
        loss = _reconstruct_prepared_batch_cuda_sparse(
            prepared,
            C10=np.asarray([C10], dtype=np.float32),
            C12=np.asarray([C12], dtype=np.float32),
            phi12=np.asarray([phi12], dtype=np.float32),
            chunk_bf=chunk_bf,
        )[0]
        return float(loss)

    def evaluate_batch(params: list[dict[str, float]]) -> np.ndarray:
        c10 = np.asarray([p["C10"] for p in params], dtype=np.float32)
        c12 = np.asarray([p["C12"] for p in params], dtype=np.float32)
        phi = np.asarray([p["phi12"] for p in params], dtype=np.float32)
        return _reconstruct_prepared_batch_cuda_sparse(
            prepared,
            C10=c10,
            C12=c12,
            phi12=phi,
            chunk_bf=chunk_bf,
        )

    best = dict(start)
    best_loss = evaluate(best["C10"], best["C12"], best["phi12"])
    trials.append({"params": dict(best), "loss": best_loss})

    if n_trials > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(seed)),
        )

        n_completed = 0
        batch_size = max(1, int(optuna_batch_size))
        while n_completed < int(n_trials):
            current = min(batch_size, int(n_trials) - n_completed)
            optuna_trials = [study.ask() for _ in range(current)]
            trial_params = []
            for trial in optuna_trials:
                C10 = _suggest_or_fixed(trial, ranges, "C10_nm", best["C10"])
                C12 = _suggest_or_fixed(trial, ranges, "C12_nm", best["C12"])
                phi12 = math.radians(_suggest_or_fixed(
                    trial, ranges, "phi12_deg", math.degrees(best["phi12"])
                ))
                trial_params.append({"C10": C10, "C12": C12, "phi12": phi12})
            losses = evaluate_batch(trial_params)
            for trial, params, loss in zip(optuna_trials, trial_params, losses):
                loss_value = float(loss)
                study.tell(trial, loss_value)
                trials.append({"params": dict(params), "loss": loss_value})
            n_completed += current

        if study.best_trial is not None and float(study.best_value) < best_loss:
            params = study.best_trial.params
            best = {
                "C10": float(params.get("C10_nm", best["C10"])),
                "C12": float(params.get("C12_nm", best["C12"])),
                "phi12": math.radians(float(params.get("phi12_deg", math.degrees(best["phi12"])))),
            }
            best_loss = float(study.best_value)

    if refine in ("nmead", "nelder-mead"):
        lock = set(refine_lock or [])
        def refine_eval(params: dict[str, float]) -> float:
            loss = evaluate(params["C10"], params["C12"], params["phi12"])
            trials.append({"params": dict(params), "loss": loss})
            return loss

        best, best_loss = _nelder_mead_refine(
            best,
            best_loss,
            refine_eval,
            lock=lock,
        )
    elif refine is not None:
        raise ValueError(f"refine must be 'nmead' or None, got {refine!r}")

    object_wave, _full_loss, phase = _reconstruct_prepared(
        prepared,
        C10=best["C10"],
        C12=best["C12"],
        phi12=best["phi12"],
        chunk_bf=chunk_bf,
        compute_loss=False,
        compute_object=True,
    )
    final_loss = evaluate(best["C10"], best["C12"], best["phi12"])
    amplitude = np.abs(object_wave).astype(np.float32)
    return MpsSSBPreviewResult(
        object_wave=object_wave,
        phase=phase,
        amplitude=amplitude,
        bf_center=center,
        bf_radius=radius,
        num_bf=int(bf_row.size),
        elapsed=time.perf_counter() - t0,
        aberrations=best,
        loss=float(final_loss if final_loss is not None else best_loss),
        n_trials=int(n_trials),
        optuna_trials=trials,
        refine_method=refine,
    )


__all__ = ["MpsSSBPreviewResult", "ssb_fit", "ssb_preview"]
