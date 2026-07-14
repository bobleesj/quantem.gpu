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

import numpy as np

from quantem.gpu.detector import auto_probe, mean_dp
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
    raise TypeError(
        "MPS SSB preview expects chunk-backed MPS data from "
        "`quantem.gpu.io.hdf5.load(..., backend='mps')`."
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
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], float]:
    dp = mean_dp(data)
    center, detected_radius = auto_probe(dp)
    radius = float(detected_radius if bf_radius is None else bf_radius)
    mask = dp > float(dp.max()) * float(threshold)
    rr, cc = np.nonzero(mask)
    dist2 = (rr.astype(np.float32) - center[0]) ** 2 + (
        cc.astype(np.float32) - center[1]
    ) ** 2
    keep = dist2 <= radius ** 2
    rr = rr[keep].astype(np.int32)
    cc = cc[keep].astype(np.int32)
    if rr.size == 0:
        raise ValueError(
            f"No BF pixels selected with threshold={threshold} and radius={radius}."
        )
    return rr, cc, (float(center[0]), float(center[1])), radius


def _ranges_from_start(
    start: dict[str, float],
    search_ranges: dict | None,
) -> dict[str, tuple[float, float] | float]:
    if search_ranges is not None:
        return dict(search_ranges)
    return {
        "C10_nm": (float(start["C10"]) - 150.0, float(start["C10"]) + 150.0),
        "C12_nm": (max(0.0, float(start["C12"]) - 80.0), float(start["C12"]) + 80.0),
        "phi12_deg": (
            math.degrees(float(start["phi12"])) - 90.0,
            math.degrees(float(start["phi12"])) + 90.0,
        ),
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
) -> tuple[np.ndarray, float | None]:
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
    phase_sum = mx.zeros(scan_shape, dtype=mx.float32) if compute_loss else None
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
        if compute_loss:
            chunk_sum, chunk_sumsq = _loss_from_phase_stack(mx, obj_chunk)
            phase_sum = phase_sum + chunk_sum
            phase_sumsq = phase_sumsq + chunk_sumsq
            mx.eval(accumulator, phase_sum, phase_sumsq)
        else:
            mx.eval(accumulator)
        if verbose:
            print(f"MPS SSB BF {stop}/{bf_row.size}")

    object_wave_mx = accumulator / int(bf_row.size)
    loss = None
    if compute_loss:
        mean_phase = phase_sum / int(bf_row.size)
        var_per_pixel = phase_sumsq / int(bf_row.size) - mean_phase * mean_phase
        loss = float(np.asarray(mx.mean(var_per_pixel)))
    mx.eval(object_wave_mx)
    object_wave = np.asarray(object_wave_mx).astype(np.complex64, copy=False)
    return object_wave, loss


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

    bf_row, bf_col, center, radius = _bf_pixels(frames, bf_intensity_threshold, bf_radius)
    if det_sampling is None:
        det_px = (2.0 * float(semiangle_mrad)) / radius
        det_sampling = (det_px, det_px)
    else:
        det_sampling = _as_sampling(det_sampling)
    object_wave, loss = _reconstruct_selection(
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
    phase = np.angle(object_wave).astype(np.float32)
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
    bf_radius: float | None = None,
    chunk_bf: int = 16,
    verbose: bool = False,
) -> MpsSSBPreviewResult:
    """Free-fit C10/C12/phi12 on Apple GPU, then reconstruct the best SSB phase.

    This is a compact MLX optimizer for Mac workflows. It uses the same
    per-BF-pixel phase-variance loss family as the CUDA SSB engine, but not the
    batched CUDA kernels.
    """
    _require_mlx()
    import optuna

    t0 = time.perf_counter()
    frames = _as_chunked_frames(data)
    scan_shape = _scan_shape(frames)
    det_shape = tuple(int(x) for x in frames.shape[-2:])
    scan_sampling = _as_sampling(scan_sampling_A)
    bf_row, bf_col, center, radius = _bf_pixels(frames, bf_intensity_threshold, bf_radius)
    if det_sampling is None:
        det_px = (2.0 * float(semiangle_mrad)) / radius
        det_sampling = (det_px, det_px)
    else:
        det_sampling = _as_sampling(det_sampling)

    start = {"C10": 0.0, "C12": 50.0, "phi12": 0.0}
    if aberrations:
        start.update({k: float(v) for k, v in aberrations.items() if k in start})
    ranges = _ranges_from_start(start, search_ranges)
    trials: list[dict] = []

    def evaluate(C10: float, C12: float, phi12: float) -> float:
        _obj, loss = _reconstruct_selection(
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
            compute_loss=True,
            verbose=False,
        )
        return float(loss)

    best = dict(start)
    best_loss = evaluate(best["C10"], best["C12"], best["phi12"])
    trials.append({"params": dict(best), "loss": best_loss})

    if n_trials > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")

        def objective(trial) -> float:
            C10 = _suggest_or_fixed(trial, ranges, "C10_nm", best["C10"])
            C12 = _suggest_or_fixed(trial, ranges, "C12_nm", best["C12"])
            phi12 = math.radians(_suggest_or_fixed(
                trial, ranges, "phi12_deg", math.degrees(best["phi12"])
            ))
            loss = evaluate(C10, C12, phi12)
            trials.append(
                {"params": {"C10": C10, "C12": C12, "phi12": phi12}, "loss": loss}
            )
            return loss

        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=verbose)
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
        steps = {"C10": 10.0, "C12": 5.0, "phi12": math.radians(5.0)}
        for _ in range(3):
            improved = False
            for key, step in steps.items():
                if key in lock:
                    continue
                for direction in (-1.0, 1.0):
                    cand = dict(best)
                    cand[key] += direction * step
                    if key == "C12":
                        cand[key] = max(0.0, cand[key])
                    loss = evaluate(cand["C10"], cand["C12"], cand["phi12"])
                    trials.append({"params": dict(cand), "loss": loss})
                    if loss < best_loss:
                        best, best_loss, improved = cand, loss, True
            for key in steps:
                steps[key] *= 0.5
            if not improved:
                continue
    elif refine is not None:
        raise ValueError(f"refine must be 'nmead' or None, got {refine!r}")

    object_wave, final_loss = _reconstruct_selection(
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
        C10=best["C10"],
        C12=best["C12"],
        phi12=best["phi12"],
        rotation_angle_deg=rotation_angle_deg,
        chunk_bf=chunk_bf,
        compute_loss=True,
        verbose=verbose,
    )
    phase = np.angle(object_wave).astype(np.float32)
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
