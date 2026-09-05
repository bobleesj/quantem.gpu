"""Exact full-BF aberration optimization for the MPS backend."""
from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from quantem.gpu.detector import mean_dp
from quantem.gpu.ssb.results import SSBResult

from .engine import (
    MpsBfColumnFrames,
    _PreparedMpsSSB,
    _as_chunked_frames,
    _as_sampling,
    _default_object_redraw_chunk_bf,
    _default_object_setup_chunk_bf,
    _effective_phase_loss_chunk_bf,
    _evaluate_exact_float32_cached,
    _nelder_mead_refine,
    _object_fourier_sum_dynamic,
    _prepare_selection,
    _ranges_from_start,
    _reconstruct_prepared,
    _reconstruct_prepared_batch_exact_loss,
    _require_mlx,
    _resolve_bf_selection,
    _scan_shape,
    _suggest_or_fixed,
)


def optimize(
    data,
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | tuple[float, float],
    det_sampling: float | tuple[float, float] | None = None,
    aberrations: dict | None = None,
    search_ranges: dict | None = None,
    n_trials: int = 200,
    refine: str | None = "nelder-mead",
    refine_lock: list[str] | None = None,
    rotation_angle_deg: float = 0.0,
    bf_intensity_threshold: float = 0.0,
    bf_center: tuple[float, float] | None = None,
    bf_radius: float | None = None,
    chunk_bf: int = 16,
    optuna_batch_size: int = 2,
    seed: int = 42,
    verbose: bool = False,
    _on_complete: Callable[
        [_PreparedMpsSSB, np.ndarray, float, dict[str, float]], None
    ]
    | None = None,
) -> SSBResult:
    """Free-fit C10/C12/phi12 on Apple GPU, then reconstruct the best SSB phase.

    This is a compact MLX optimizer for Mac workflows. Every candidate uses
    the same full active-BF phase-variance loss as the final reconstruction.
    """
    mx = _require_mlx()
    import optuna

    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    frames = _as_chunked_frames(data)
    scan_shape = _scan_shape(frames)
    scan_sampling = _as_sampling(scan_sampling_A)
    stored_dc = (
        frames.dc_value
        if isinstance(frames, MpsBfColumnFrames)
        else None
    )
    dp = None if stored_dc is not None else mean_dp(frames)
    selection = _resolve_bf_selection(
        frames,
        bf_intensity_threshold,
        bf_radius,
        center_override=bf_center,
        mean_diffraction=dp,
    )
    if det_sampling is None:
        det_px = (
            2.0 * float(semiangle_mrad) / selection.detected_radius_px
        )
        det_sampling = (det_px, det_px)
    else:
        det_sampling = _as_sampling(det_sampling)

    requested_chunk_bf = max(1, int(chunk_bf))
    setup_chunk_bf = max(requested_chunk_bf, _default_object_setup_chunk_bf())
    compact_inactive = True
    dc_value_override = stored_dc
    if compact_inactive:
        # mean_dp is detector_sum / n_frames. A power-of-two frame count and
        # detector sums no larger than 2**24 make the round trip exact in float32.
        # Otherwise preparation reads every selected column to retain its FFT DC.
        n_frames = int(scan_shape[0]) * int(scan_shape[1])
        if dc_value_override is None:
            if dp is None:
                raise RuntimeError(
                    "MPS SSB has neither detector sums nor DC metadata."
                )
            recovered_dc = (
                dp[selection.rows, selection.cols].astype(np.float32, copy=False)
                * np.float32(n_frames)
            )
            dc_round_trip_is_exact = (
                n_frames > 0
                and n_frames & (n_frames - 1) == 0
                and bool(np.all(recovered_dc <= np.float32(2**24)))
                and bool(np.all(recovered_dc == np.floor(recovered_dc)))
            )
            if dc_round_trip_is_exact:
                dc_value_override = complex(
                    recovered_dc.astype(np.complex64).mean()
                )
        if dc_value_override is not None:
            # Compact preparation can gather every aperture-active plane in one
            # Metal dispatch and import it into MLX once. The optimizer retains
            # its original logical 512-BF reduction boundaries independently.
            setup_chunk_bf = max(setup_chunk_bf, selection.size)
    prepared = _prepare_selection(
        frames,
        scan_shape=scan_shape,
        selection=selection,
        voltage_kV=voltage_kV,
        semiangle_mrad=semiangle_mrad,
        scan_sampling=scan_sampling,
        det_sampling=det_sampling,
        rotation_angle_deg=rotation_angle_deg,
        chunk_bf=setup_chunk_bf,
        compact_inactive=compact_inactive,
        dc_value_override=dc_value_override,
    )
    # Preparation leaves large, unused FFT/gather temporaries in MLX's buffer
    # cache.  Release those before the exact optimizer allocates its paired row
    # intermediates; active prepared arrays such as G_qk remain resident.
    mx.clear_cache()
    timings["preparation_seconds"] = time.perf_counter() - t0
    fit_chunk_bf = _effective_phase_loss_chunk_bf(requested_chunk_bf, scan_shape)

    start = {"C10": 0.0, "C12": 50.0, "phi12": 0.0}
    if aberrations:
        start.update({k: float(v) for k, v in aberrations.items() if k in start})
    ranges = _ranges_from_start(start, search_ranges)
    trials: list[dict] = []

    def evaluate(C10: float, C12: float, phi12: float) -> float:
        loss = _reconstruct_prepared_batch_exact_loss(
            prepared,
            C10=np.asarray([C10], dtype=np.float32),
            C12=np.asarray([C12], dtype=np.float32),
            phi12=np.asarray([phi12], dtype=np.float32),
            chunk_bf=fit_chunk_bf,
        )[0]
        return float(loss)

    def evaluate_batch(params: list[dict[str, float]]) -> np.ndarray:
        c10 = np.asarray([p["C10"] for p in params], dtype=np.float32)
        c12 = np.asarray([p["C12"] for p in params], dtype=np.float32)
        phi = np.asarray([p["phi12"] for p in params], dtype=np.float32)
        return _reconstruct_prepared_batch_exact_loss(
            prepared,
            C10=c10,
            C12=c12,
            phi12=phi,
            chunk_bf=fit_chunk_bf,
        )

    best = dict(start)
    initial_started = time.perf_counter()
    best_loss = evaluate(best["C10"], best["C12"], best["phi12"])
    timings["initial_loss_seconds"] = time.perf_counter() - initial_started
    trials.append({"params": dict(best), "loss": best_loss})

    optuna_started = time.perf_counter()
    if n_trials > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(seed)),
        )

        from tqdm.auto import tqdm

        n_completed = 0
        batch_size = max(1, int(optuna_batch_size))
        progress = tqdm(
            total=int(n_trials),
            desc="SSB optimize",
            disable=not verbose,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}]"
            ),
        )
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
            progress.update(current)
        progress.close()

        if study.best_trial is not None and float(study.best_value) < best_loss:
            params = study.best_trial.params
            best = {
                "C10": float(params.get("C10_nm", best["C10"])),
                "C12": float(params.get("C12_nm", best["C12"])),
                "phi12": math.radians(float(params.get("phi12_deg", math.degrees(best["phi12"])))),
            }
            best_loss = float(study.best_value)
    timings["optuna_seconds"] = time.perf_counter() - optuna_started

    refine_started = time.perf_counter()
    refine_nfev = 0
    if refine == "nelder-mead":
        lock = set(refine_lock or [])
        refine_cache: dict[tuple[float, float, float, float], float] = {}

        def refine_eval(params: dict[str, float]) -> float:
            loss = _evaluate_exact_float32_cached(
                params,
                lambda current: evaluate(
                    current["C10"], current["C12"], current["phi12"]
                ),
                refine_cache,
            )
            trials.append({"params": dict(params), "loss": loss})
            return loss

        best, best_loss = _nelder_mead_refine(
            best,
            best_loss,
            refine_eval,
            lock=lock,
            fatol=3e-6,
            max_iter=80,
            initial_step_floor={"C12": 2.0, "phi12": 0.04},
            initial_step_decimals=2,
        )
        refine_nfev = len(refine_cache)
    elif refine is not None:
        raise ValueError(f"refine must be 'nelder-mead' or None, got {refine!r}")
    timings["refinement_seconds"] = time.perf_counter() - refine_started

    final_object_started = time.perf_counter()
    final_chunk_bf = _effective_phase_loss_chunk_bf(chunk_bf, scan_shape)
    object_wave_mx = _object_fourier_sum_dynamic(
        prepared,
        C10=best["C10"],
        C12=best["C12"],
        phi12=best["phi12"],
        chunk_bf=_default_object_redraw_chunk_bf(),
    )
    mx = _require_mlx()
    mx.eval(object_wave_mx)
    object_wave = np.asarray(object_wave_mx).astype(np.complex64, copy=False)
    timings["final_object_seconds"] = time.perf_counter() - final_object_started
    final_loss_started = time.perf_counter()
    _object_wave, _full_loss, phase = _reconstruct_prepared(
        prepared,
        C10=best["C10"],
        C12=best["C12"],
        phi12=best["phi12"],
        chunk_bf=final_chunk_bf,
        compute_loss=True,
        compute_object=False,
    )
    timings["final_phase_loss_seconds"] = time.perf_counter() - final_loss_started
    final_loss = _full_loss if _full_loss is not None else best_loss
    elapsed = time.perf_counter() - t0
    final_loss_value = float(final_loss if final_loss is not None else best_loss)
    if phase is None:
        raise RuntimeError("MPS optimizer did not produce its final exact phase.")
    if _on_complete is not None:
        _on_complete(prepared, phase, final_loss_value, dict(best))
    normalized_trials: list[dict[str, object]] = []
    for trial in trials:
        params = dict(trial["params"])
        normalized_trials.append(
            {
                "params": {
                    "C10_nm": float(params["C10"]),
                    "C12_nm": float(params["C12"]),
                    "phi12_deg": math.degrees(float(params["phi12"])),
                },
                "loss": float(trial["loss"]),
            }
        )
    return SSBResult(
        object_wave=object_wave,
        backend="mps",
        aberrations=dict(best),
        rotation_angle_deg=float(rotation_angle_deg),
        loss=final_loss_value,
        elapsed=elapsed,
        timings=timings,
        n_trials=int(n_trials),
        num_bf=selection.size,
        refine_method=refine,
        refine_nfev=refine_nfev,
        refine_elapsed=timings["refinement_seconds"],
        voltage_kV=float(voltage_kV),
        semiangle_mrad=float(semiangle_mrad),
        scan_sampling_A=scan_sampling_A,
        bf_center=selection.center_row_col,
        bf_radius=selection.radius_px,
        detected_bf_radius=selection.detected_radius_px,
        optuna_trials=normalized_trials,
    )


__all__ = ["optimize"]
