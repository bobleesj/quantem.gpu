"""Reproducible Direct Ptycho workflow timings using the installed compute backend."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io as text_io
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantem.gpu import SSB


def _synchronize(backend: str) -> None:
    """Finish all queued work before reading the wall clock."""
    import torch

    if backend == "cuda":
        import cupy as cp

        cp.cuda.Device().synchronize()
        torch.cuda.synchronize()
    else:
        import mlx.core as mx

        mx.synchronize()
        torch.mps.synchronize()


def _host_phase(phase, backend: str) -> np.ndarray:
    """Download a phase image explicitly for report hashes and comparisons."""
    if backend == "cuda":
        import cupy as cp

        return cp.asnumpy(phase)
    return np.asarray(phase)


def _timed(call: Callable, backend: str) -> tuple[object, float]:
    _synchronize(backend)
    start = time.perf_counter()
    value = call()
    _synchronize(backend)
    return value, time.perf_counter() - start


def _environment(backend: str) -> dict:
    """Record hardware and imported code without home paths or host names."""
    import torch

    import quantem.gpu

    if backend == "cuda":
        hardware = torch.cuda.get_device_name()
    else:
        hardware = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    versions = {}
    for name in ("quantem.gpu", "torch", "numpy", "cupy", "cupy-cuda13x", "mlx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    root = Path(quantem.gpu.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".cu", ".metal", ".msl"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return {
        "backend": backend,
        "hardware": hardware,
        "system": platform.system(),
        "python": platform.python_version(),
        "versions": versions,
        "compute_source_sha256": digest.hexdigest(),
        "installed_wheel": "site-packages" in root.parts,
    }


def _fit_record(result, backend: str, trials: int, branches: int) -> dict:
    """Fail if a timed fit silently changes the requested numerical work."""
    optimizer = result.optimizer
    if (
        optimizer.get("device") != backend
        or optimizer.get("host_numerical_search") is not False
    ):
        raise RuntimeError(
            "The joint search must run on the selected GPU. Install the matching GPU extra."
        )
    if result.refinement != "newton" or len(result.trial_records) != trials * branches:
        raise RuntimeError(
            "The timed fit did not complete the requested trials and Newton refinement."
        )
    return {
        "rotation_branches": branches,
        "completed_trials": len(result.trial_records),
        "refinement": result.refinement,
        "optimizer": optimizer,
        "aberrations": dict(result.aberrations),
        "sample_tilt_mrad": list(result.sample_tilt_mrad),
        "depth_spread_nm": float(result.depth_spread_nm),
        "rotation_deg": float(result.scan_detector_rotation_deg),
        "agreement": float(result.sample_agreement),
        "stage_seconds": dict(result.timings),
    }


def _summary(seconds: list[float]) -> dict:
    """Keep the first observation separate from repeated warm observations."""
    warm = seconds[1:]
    return {
        "first_s": seconds[0],
        "warm_n": len(warm),
        "warm_p50_s": statistics.median(warm) if warm else None,
        "warm_p90_s": float(np.percentile(warm, 90)) if warm else None,
        "min_s": min(seconds),
        "max_s": max(seconds),
        "seconds": seconds,
    }


def _regions(shape: tuple[int, int], sizes: tuple[int, ...]) -> list[tuple[str, tuple]]:
    """Choose centered regions, then a translated and resized small region."""
    rows, cols = shape
    if not sizes or any(size < 2 or size > min(shape) for size in sizes):
        raise ValueError(
            f"Region sizes must be between 2 and {min(shape)}; got {sizes}."
        )
    result = []
    for size in sizes:
        row, col = (rows - size) // 2, (cols - size) // 2
        result.append((f"region_{size}", (row, row + size, col, col + size)))
    size = min(sizes)
    row, col = (rows - size) // 2, (cols - size) // 2
    step = max(1, size // 4)
    row, col = min(row + step, rows - size), min(col + step, cols - size)
    result.append((f"move_{size}", (row, row + size, col, col + size)))
    result.append((f"resize_{size - 1}x{size}", (row, row + size - 1, col, col + size)))
    return result


def run_ssb_preview(
    ssb: SSB,
    *,
    upsampling: tuple[int, ...] = (1, 2, 3, 4, 8),
    runs: int = 6,
    defocus_step_nm: float = 0.1,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Measure changed-defocus redraws of an existing full-field Direct Ptycho session.

    Retain the session's aberrations, rotation, tilt, depth and complete
    bright-field support. Only C10 changes along a deterministic trajectory;
    the session itself is not modified. No fitting or loss diagnostic runs.
    Each output grid is measured separately, because changing upsampling
    changes the reconstructed frequency coverage.

    Parameters
    ----------
    ssb : quantem.gpu.DirectPtycho
        Calibrated session, normally restored from a reviewed joint fit.
    upsampling : tuple of int
        Output factors to measure. Supported factors are 1, 2, 3, 4 and 8.
    runs : int
        First observation plus warm observations, at least two.
    defocus_step_nm : float
        Nonzero C10 increment between successive observations. Each factor
        uses the same trajectory, starting at the session's current C10.
    progress : callable, optional
        Receives one completion message per output factor.

    Returns
    -------
    dict
        Hardware/source identity, unchanged scientific settings, preparation
        time, synchronized reconstruction and host-delivery distributions,
        and full-array hashes. Browser transfer and paint are excluded.
        MPS preview already returns a host array, so its reconstruction time
        includes host delivery; CUDA download time is measured separately.
        Hashes identify outputs, but do not by themselves establish parity
        against an independent reference.

    Examples
    --------
    >>> from quantem.gpu.benchmark import run_ssb_preview
    >>> report = run_ssb_preview(ssb, upsampling=(1, 4), runs=11)  # doctest: +SKIP
    """
    if runs < 2:
        raise ValueError("Use runs >= 2 for a first observation and warm timings.")
    if not upsampling or any(
        type(value) is not int or value not in (1, 2, 3, 4, 8) for value in upsampling
    ):
        raise ValueError("Choose upsampling factors from (1, 2, 3, 4, 8).")
    if not np.isfinite(defocus_step_nm) or defocus_step_nm == 0:
        raise ValueError("Use a finite, nonzero defocus_step_nm to measure new images.")
    backend = ssb.backend
    if backend not in ("mps", "cuda"):
        raise ValueError("Open the SSB session on MPS or CUDA before benchmarking.")
    if ssb.aberrations is None:
        raise ValueError("Restore or fit the SSB calibration before measuring redraws.")
    optics = dict(ssb.aberrations)
    sample = {
        "sample_tilt_mrad": tuple(ssb.sample_tilt_mrad or (0.0, 0.0)),
        "depth_spread_nm": float(ssb.depth_spread_nm or 0.0),
    }
    bright_field_pixels, preparation_s = _timed(
        lambda: ssb.num_bright_field_pixels, backend
    )
    report = {
        "schema": "quantem.gpu.ssb-preview-benchmark/1",
        "environment": _environment(backend),
        "scan_shape": list(ssb.scan_shape),
        "bright_field_pixels": bright_field_pixels,
        "aberrations": optics,
        "sample": sample,
        "rotation_deg": float(ssb.scan_detector_rotation_deg),
        "calibration": {
            "voltage_kV": ssb.voltage_kV,
            "semiangle_mrad": ssb.semiangle_mrad,
            "scan_sampling_A": list(ssb.scan_sampling_A),
            "detector_sampling_mrad": list(ssb.detector_sampling_mrad),
            "scan_detector_rotation_flipped": ssb.scan_detector_rotation_flipped,
        },
        "phase_estimator": "phase_of_mean",
        "preparation_s": preparation_s,
        "reconstruction_includes_host_delivery": backend == "mps",
        "browser_paint_measured": False,
        "measurements": {},
    }
    for factor in upsampling:
        compute_times, download_times, delivery_times, samples = [], [], [], []
        for index in range(runs):
            values = {**optics, "C10_nm": optics["C10_nm"] + index * defocus_step_nm}
            (phase, _), compute_s = _timed(
                partial(
                    ssb.preview,
                    values,
                    compute_loss=False,
                    upsampling_factor=factor,
                    output="native" if backend == "cuda" else "numpy",
                    **sample,
                ),
                backend,
            )
            started = time.perf_counter()
            phase = _host_phase(phase, backend)
            download_s = time.perf_counter() - started
            if not np.isfinite(phase).all():
                raise RuntimeError(
                    f"Nonfinite phase at upsampling={factor}; inspect calibration."
                )
            compute_times.append(compute_s)
            download_times.append(download_s)
            delivery_times.append(compute_s + download_s)
            samples.append(
                {
                    "C10_nm": values["C10_nm"],
                    "phase_sha256": hashlib.sha256(phase.tobytes()).hexdigest(),
                }
            )
        report["measurements"][str(factor)] = {
            "output_shape": list(phase.shape),
            "reconstruction": _summary(compute_times),
            "host_download": _summary(download_times) if backend == "cuda" else None,
            "phase_delivery": _summary(delivery_times),
            "samples": samples,
        }
        if progress is not None:
            progress(
                f"{factor}x: {statistics.median(delivery_times[1:]) * 1000:.2f} ms warm phase delivery"
            )
    return report


def run_ssb(
    source: str | Path,
    *,
    backend: str = "auto",
    runs: int = 6,
    region_sizes: tuple[int, ...] = (256, 128),
    num_trials: int = 200,
    seed: int = 42,
    calibration: dict | None = None,
    progress: Callable[[str], None] | None = None,
    require_installed: bool = False,
    reference_phase: str | Path | None = None,
) -> dict:
    """Measure full-field joint fits and independent regional corrections.

    Each measurement performs one seeded search and Newton refinement. An
    estimated full-field rotation tests both 180-degree branches; a supplied
    rotation is respected. Regional fits hold that rotation, jointly refit
    aberrations, tilt and depth, reconstruct the full scan at Auto sampling,
    and crop the corrected view. No scan binning or BF subsampling is used.

    Parameters
    ----------
    source : str or pathlib.Path
        Local acquisition accepted by ``quantem.gpu.io.load``.
    backend : str
        ``auto``, ``cuda`` or ``mps``. CPU fallback is not permitted.
    runs : int
        Number of independent measurements, including the first observation.
        Six gives one first observation and five warm measurements. These are
        benchmark repetitions, not multiple searches per user interaction.
    region_sizes : tuple of int
        Square scan sizes in original pixels. The smallest is also moved and
        resized by one row to measure non-power-of-two dispatch.
    num_trials : int
        Trials in each search, before Newton refinement.
    seed : int
        The same seed is reused in every measurement.
    calibration : dict, optional
        Overrides for ``voltage_kV``, ``semiangle_mrad``, ``scan_sampling_A``
        and ``detector_sampling_mrad``. Saved values are reused otherwise;
        missing detector sampling is fitted automatically.
    progress : callable, optional
        Receives progress labels without source paths or acquisition names.
    require_installed : bool
        Reject a development checkout when qualifying an installed wheel.
    reference_phase : str or pathlib.Path, optional
        Frozen Auto full-field phase in a NumPy file from the same backend,
        input and calibration. Compare without alignment or resampling.

    Returns
    -------
    dict
        JSON-compatible timings, fit settings, hardware, source-code hash,
        dimensions and repeat parity. No source paths, input names, host names
        or images are included. Browser paint time is not measured here.

    Examples
    --------
    >>> from quantem.gpu.benchmark import run_ssb
    >>> report = run_ssb("scan.qem")  # doctest: +SKIP
    """
    from quantem.gpu import DirectPtycho, io
    from quantem.gpu.device import resolve

    if runs < 2 or num_trials < 1:
        raise ValueError("Use runs >= 2 and num_trials >= 1 to measure repeated fits.")
    selected = resolve(backend)
    if selected not in {"cuda", "mps"}:
        raise RuntimeError(
            "SSB benchmarking needs CUDA or Apple MPS. Install the matching GPU extra."
        )
    environment = _environment(selected)
    reference = (
        np.load(reference_phase, allow_pickle=False) if reference_phase else None
    )
    reference_checks = []
    if require_installed and not environment["installed_wheel"]:
        raise RuntimeError(
            "Install the candidate wheel in a fresh environment and unset PYTHONPATH."
        )
    # Loader notices can contain private filenames. Only explicit report
    # fields are exported; the acquisition and its metadata remain untouched.
    with contextlib.redirect_stdout(text_io.StringIO()):
        data, load_s = _timed(
            partial(io.load, source, backend=selected, verbose=False), selected
        )
    timings: dict[str, list[float]] = {}
    records: dict[str, list[dict]] = {}
    references: dict[str, np.ndarray] = {}
    parity: dict[str, list[dict]] = {}

    def keep(name: str, seconds: float, phase, record: dict) -> None:
        phase = _host_phase(phase, selected)
        if not np.isfinite(phase).all():
            raise RuntimeError(f"{name} produced nonfinite reconstructed phase.")
        if name not in references:
            references[name] = phase.copy()
        reference = references[name]
        delta = phase.astype(np.float64) - reference
        checks = {
            "bitwise_equal": bool(np.array_equal(phase, reference)),
            "max_abs_rad": float(np.max(np.abs(delta))),
            "rms_rad": float(np.sqrt(np.mean(delta * delta))),
        }
        if not np.allclose(phase, reference, atol=1e-6, rtol=1e-6):
            raise RuntimeError(
                f"{name} changed between identical seeded runs: {checks}"
            )
        timings.setdefault(name, []).append(seconds)
        records.setdefault(name, []).append(
            {
                **record,
                "phase_sha256": hashlib.sha256(
                    np.ascontiguousarray(phase).tobytes()
                ).hexdigest(),
            }
        )
        parity.setdefault(name, []).append(checks)

    try:
        for name, value in (calibration or {}).items():
            if name not in {
                "voltage_kV",
                "semiangle_mrad",
                "scan_sampling_A",
                "detector_sampling_mrad",
            }:
                raise ValueError(
                    f"Unknown calibration {name!r}; supply a documented acquisition calibration."
                )
            setattr(data.calibration, name, value)
        calibration_s = 0.0
        if (
            data.calibration.detector_sampling_mrad is None
            and data.calibration.detector_sampling_inv_A is None
        ):
            with contextlib.redirect_stdout(text_io.StringIO()):
                _, calibration_s = _timed(
                    data.calibration.fit_detector_sampling, selected
                )
        shape = tuple(int(size) for size in data.shape)
        regions = _regions(shape[:2], region_sizes)
        session_times = []
        for index in range(runs):
            if progress:
                progress(f"Measurement {index + 1}/{runs}: full field")
            with contextlib.redirect_stdout(text_io.StringIO()):
                ssb, session_s = _timed(partial(DirectPtycho, data, backend=selected), selected)
            session_times.append(session_s)
            with ssb:
                fitted, seconds = _timed(
                    partial(
                        ssb.find_aberrations,
                        search="standard",
                        num_trials=num_trials,
                        seed=seed,
                        compare_standard=False,
                        verbose=False,
                    ),
                    selected,
                )
                branches = (
                    len(fitted.rotation_branches.branches)
                    if fitted.rotation_branches
                    else 1
                )
                record = _fit_record(fitted, selected, num_trials, branches)
                keep("full_field_fit", seconds, fitted.phase, record)
                output, output_s = _timed(
                    partial(
                        ssb.reconstruct,
                        fitted,
                        upsampling_factor="auto",
                        compute_loss=False,
                    ),
                    selected,
                )
                keep("full_field_reconstruction", output_s, output.phase, record)
                if reference is not None:
                    if reference.shape != output.phase.shape:
                        raise ValueError(
                            "The reference phase must use the same full field and Auto sampling."
                        )
                    host_phase = _host_phase(output.phase, selected)
                    difference = host_phase.astype(np.float64) - reference
                    reference_checks.append(
                        {
                            "bitwise_equal": bool(
                                np.array_equal(host_phase, reference)
                            ),
                            "max_abs_rad": float(np.max(np.abs(difference))),
                            "rms_rad": float(np.sqrt(np.mean(difference * difference))),
                        }
                    )
                    if not np.allclose(host_phase, reference, atol=1e-6, rtol=1e-6):
                        raise RuntimeError(
                            "Full-field phase differs from the frozen reference. Check source, calibration and precision; do not replace the reference automatically."
                        )
                factor = int(output.phase.shape[0] // ssb.scan_shape[0])
                row_offset, col_offset = (
                    (padded - measured) // 2
                    for padded, measured in zip(ssb.scan_shape, shape[:2])
                )
                bf_pixels = ssb.num_bright_field_pixels
                used_calibration = {
                    "voltage_kV": ssb.voltage_kV,
                    "semiangle_mrad": ssb.semiangle_mrad,
                    "scan_sampling_A": list(ssb.scan_sampling_A),
                    "detector_sampling_mrad": list(ssb.detector_sampling_mrad),
                    "bright_field_center_px": list(ssb.bright_field_center_px),
                    "bright_field_radius_px": ssb.bright_field_radius_px,
                }
                for label, bounds in regions:
                    if progress:
                        progress(f"Measurement {index + 1}/{runs}: {label}")
                    _synchronize(selected)
                    start = time.perf_counter()
                    regional, prepare_s = _timed(
                        partial(
                            ssb.select_region,
                            bounds,
                            aberrations=fitted.aberrations,
                            scan_detector_rotation_deg=fitted.scan_detector_rotation_deg,
                        ),
                        selected,
                    )
                    with regional:
                        local, fit_s = _timed(
                            partial(
                                regional.find_aberrations,
                                search="standard",
                                num_trials=num_trials,
                                seed=seed,
                                check_rotation=False,
                                compare_standard=False,
                                compute_loss=False,
                                verbose=False,
                            ),
                            selected,
                        )
                        local_bf = regional.num_bright_field_pixels
                    (phase, _), reconstruct_s = _timed(
                        partial(
                            ssb.preview,
                            local.aberrations,
                            sample_tilt_mrad=local.sample_tilt_mrad,
                            depth_spread_nm=local.depth_spread_nm,
                            upsampling_factor=factor,
                            compute_loss=False,
                        ),
                        selected,
                    )
                    row_start, row_stop, col_start, col_stop = bounds
                    crop = _host_phase(phase, selected)[
                        (row_start + row_offset) * factor : (row_stop + row_offset)
                        * factor,
                        (col_start + col_offset) * factor : (col_stop + col_offset)
                        * factor,
                    ].copy()
                    _synchronize(selected)
                    elapsed = time.perf_counter() - start
                    if local_bf != bf_pixels or tuple(phase.shape) != tuple(
                        n * factor for n in ssb.scan_shape
                    ):
                        raise RuntimeError(
                            "Regional correction changed the BF support or full-scan output shape."
                        )
                    record = _fit_record(local, selected, num_trials, 1)
                    record.update(
                        bounds=list(bounds),
                        prepare_s=prepare_s,
                        fit_s=fit_s,
                        reconstruction_s=reconstruct_s,
                    )
                    keep(label, elapsed, crop, record)
        return {
            "schema": "quantem.gpu.ssb-benchmark/1",
            "environment": environment,
            "reference_parity": reference_checks,
            "acquisition": {
                "shape": list(shape),
                "dtype": str(data.dtype),
                "bf_pixels": bf_pixels,
                "output_factor": factor,
            },
            "calibration": used_calibration,
            "settings": {
                "runs": runs,
                "seed": seed,
                "trials_per_branch": num_trials,
                "refinement": "newton",
                "region_sizes": list(region_sizes),
                "full_field_loss_diagnostic": True,
                "regional_loss_diagnostic": False,
            },
            "load_s": load_s,
            "calibration_s": calibration_s,
            "session": _summary(session_times),
            "measurements": {
                name: {
                    **_summary(values),
                    "fits": records[name],
                    "repeat_parity": parity[name],
                }
                for name, values in timings.items()
            },
            "timing_scope": "Synchronized wall time; regional values include preparation, joint fit, full-scan reconstruction and crop. Browser paint excluded. First observations are not cold-disk measurements.",
        }
    finally:
        data.close()
