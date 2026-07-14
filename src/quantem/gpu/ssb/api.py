"""
Functional API for quantem.gpu SSB reconstructions.

- :func:`dpc` - Differential Phase Contrast
- :func:`ssb` - Single-sideband ptychography
- :func:`ssb_series` - SSB time series or defocus series with calibration
- :func:`defocus_sweep` - sweep defocus (C10)

Example::

    from quantem.gpu import load, dpc, ssb

    data, _ = load('scan_master.h5')
    dpc_result = dpc(data)
    ssb_result = ssb(data, voltage_kV=300, semiangle_mrad=30, scan_sampling_A=0.5,
                     rotation_angle_deg=dpc_result.rotation_angle_deg)
"""

import gc
import time
from typing import Literal

import cupy as cp

from quantem.gpu.ssb.results import DefocusSweepResult, DPCResult, SSBResult

_SSB_KEYS = {"C10", "C12", "phi12"}


def free_gpu() -> None:
    """Release cached CuPy memory used by SSB compute helpers."""
    gc.collect()
    cp.fft.config.get_plan_cache().clear()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _infer_scan_shape(data, scan_shape: tuple[int, int] | None) -> tuple[int, int]:
    """Infer scan_shape from data if not provided.

    For 4D data, uses the first two dimensions.
    For 3D data, assumes square scan if scan_shape is None.
    """
    if data.ndim == 4:
        return (data.shape[0], data.shape[1])
    if scan_shape is not None:
        scan_row, scan_col = scan_shape
        if scan_row * scan_col != data.shape[0]:
            raise ValueError(
                f"scan_shape={scan_shape} implies {scan_row * scan_col} frames, but "
                f"data has {data.shape[0]} frames (shape {data.shape})."
            )
        return scan_shape
    # 3D data, no scan_shape: assume square
    import math
    n = data.shape[0]
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(
            f"Cannot infer square scan_shape from {n} frames. "
            f"Provide scan_shape explicitly."
        )
    return (side, side)


# =========================================================================
#  Aberration conversion
# =========================================================================


def _resolve_aberrations(aberrations: dict | None) -> dict | None:
    """Validate SSB aberration dict.

    Raises on partial dicts.
    """
    if aberrations is None:
        return None
    missing = _SSB_KEYS - aberrations.keys()
    if missing:
        raise ValueError(
            f"Aberrations require {{C10, C12, phi12}} in nm/rad. "
            f"Missing: {missing}."
        )
    return aberrations.copy()


def _resolve_bf_subsample_ratio(
    full_num_bf: int,
    bf_subsample: int | float | None,
) -> float | None:
    """Resolve a public BF subsample value to the engine's ratio API.

    Public callers usually think in pixels: ``2000`` means optimize/refine
    against about 2000 bright-field pixels. The SSB engine takes a fraction
    in ``(0, 1]``. Keeping this conversion here avoids accidentally running
    full-BF optimization on large microscope scans.
    """
    if bf_subsample is None:
        return None
    value = float(bf_subsample)
    if value <= 0.0:
        raise ValueError(f"bf_subsample must be positive or None, got {bf_subsample!r}")
    if value <= 1.0:
        return value
    if full_num_bf <= 0:
        raise ValueError(f"full_num_bf must be positive, got {full_num_bf}")
    return min(1.0, value / float(full_num_bf))


# =========================================================================
#  DPC
# =========================================================================


def dpc(
    data: cp.ndarray,
    scan_shape: tuple[int, int] | None = None,
    *,
    rotation_angle_deg: float | None = None,
    rotation_steps: int = 180,
    normalize_com: bool = True,
    zero_mean: bool = True,
    plot_rotation: bool = False,
    plot_com: bool = False,
    verbose: bool = True,
) -> DPCResult:
    """Differential Phase Contrast reconstruction.

    Computes the center-of-mass (CoM) of each diffraction pattern, finds the
    optimal scan-detector rotation angle, and integrates the aligned CoM field
    to produce a phase image.

    Parameters
    ----------
    data : cp.ndarray
        3D ``(N, k_row, k_col)`` or 4D ``(scan_row, scan_col, k_row, k_col)``.
    scan_shape : tuple[int, int] or None
        Scan grid shape. Inferred from 4D data or assumes square for 3D.
    rotation_angle_deg : float or None
        Force rotation angle in degrees. If None, auto-detect by minimizing
        the curl of the CoM field over ``rotation_steps`` test angles.
    rotation_steps : int
        Number of rotation angles to test during auto-detection (default 180).
    normalize_com : bool
        Subtract mean from CoM (default True).
    zero_mean : bool
        Subtract mean from phase (default True).
    plot_rotation : bool
        Plot curl vs rotation angle (default False).
    plot_com : bool
        Plot aligned CoM fields (default False).
    verbose : bool
        Print progress (default True).

    Returns
    -------
    DPCResult
        .phase : cp.ndarray
            2D phase image ``(scan_row, scan_col)``.
        .rotation_angle_deg : float
            Rotation angle in degrees (auto-detected or forced).
        .com_k_row_aligned : cp.ndarray
            CoM row component after rotation alignment.
        .com_k_col_aligned : cp.ndarray
            CoM col component after rotation alignment.
        .elapsed : float
            Wall-clock time in seconds.

    Examples
    --------
    Auto-detect rotation and reconstruct::

        from quantem.gpu import load, dpc
        data, _ = load('scan_master.h5')
        result = dpc(data)
        print(f'Rotation: {result.rotation_angle_deg:.1f} deg')

    Force rotation angle from a previous calibration::

        result = dpc(data, rotation_angle_deg=-8.0, plot_rotation=True)
    """
    from quantem.gpu.ssb.dpc import DPC

    scan_shape = _infer_scan_shape(data, scan_shape)
    obj = DPC(data, scan_shape)
    del data
    obj.preprocess(
        rotation_steps=rotation_steps,
        normalize_com=normalize_com,
        plot_rotation=plot_rotation,
        plot_com=plot_com,
    )
    autofit_phase = None
    autofit_com_row = None
    autofit_com_col = None
    autofit_rotation = None
    if rotation_angle_deg is not None:
        # Reconstruct at the AUTO-FIT angle first so the screening dashboard
        # can show both auto and forced phases side-by-side. ~0.5s of extra
        # work, worth it for the QC comparison. Aggressive cleanup between
        # the two reconstructs because dual DPC roughly doubles peak memory
        # at 512² scan size and the GPU can be tight.
        obj.reconstruct(zero_mean=zero_mean, show_results=False)
        autofit_phase = obj.phase.copy()
        autofit_com_row = obj.com_k_row_aligned.copy()
        autofit_com_col = obj.com_k_col_aligned.copy()
        autofit_rotation = obj.rotation_angle_deg
        # Drop the temporary reconstruction buffers before the second pass.
        obj.phase = None
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        # Re-rotate raw CoM at the forced angle (preprocess used auto-detected angle)
        import math
        from quantem.gpu.ssb.dpc import _rotate_vector_batch
        angle_rad = cp.array([math.radians(rotation_angle_deg)], dtype=cp.float32)
        raw_row = obj.com_k_row_aligned  # already 2D from preprocess
        raw_col = obj.com_k_col_aligned
        # Undo auto rotation, apply forced rotation
        auto_rad = math.radians(obj.rotation_angle_deg)
        forced_rad = math.radians(rotation_angle_deg)
        delta = forced_rad - auto_rad
        cos_d, sin_d = math.cos(delta), math.sin(delta)
        obj.com_k_row_aligned = raw_row * cos_d - raw_col * sin_d
        obj.com_k_col_aligned = raw_row * sin_d + raw_col * cos_d
        obj.rotation_angle_deg = rotation_angle_deg
    obj.reconstruct(zero_mean=zero_mean, show_results=False)
    result = DPCResult(
        phase=obj.phase.copy(),
        com_k_row_aligned=obj.com_k_row_aligned.copy(),
        com_k_col_aligned=obj.com_k_col_aligned.copy(),
        rotation_angle_deg=obj.rotation_angle_deg,
        elapsed=obj.elapsed,
        curl_angles_deg=obj.curl_angles_deg.copy() if obj.curl_angles_deg is not None else None,
        curl_normal=obj.curl_normal.copy() if obj.curl_normal is not None else None,
        curl_transpose=obj.curl_transpose.copy() if obj.curl_transpose is not None else None,
        autofit_phase=autofit_phase,
        autofit_com_k_row_aligned=autofit_com_row,
        autofit_com_k_col_aligned=autofit_com_col,
        autofit_rotation_angle_deg=autofit_rotation,
    )
    obj.__dict__.clear()
    del obj
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result


# =========================================================================
#  SSB (Direct Ptychography)
# =========================================================================


def ssb(
    data: cp.ndarray,
    scan_shape: tuple[int, int] | None = None,
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | tuple[float, float],
    det_sampling: float | tuple[float, float] | None = None,
    aberrations: dict | None = None,
    search_ranges: dict | None = None,
    n_trials: int = 200,
    refine: str | None = 'nmead',
    refine_lock: list[str] | None = None,
    rotation_angle_deg: float = 0.0,
    source_path: str | None = None,
    verbose: bool = True,
    bf_subsample: int | float | None = 2000,
    bf_radius: int | None = None,
) -> SSBResult:
    """Single-sideband ptychographic reconstruction.

    Runs Optuna-based aberration optimization followed by optional
    Nelder-Mead or grid refinement. Returns a complex object wave whose
    phase encodes the projected potential.

    Parameters
    ----------
    data : cp.ndarray
        3D ``(N, det_row, det_col)`` or 4D ``(scan_row, scan_col, det_row, det_col)``.
    scan_shape : tuple[int, int] or None
        Required when *data* is 3D and non-square.
    voltage_kV : float
        Accelerating voltage in kV.
    semiangle_mrad : float
        Probe convergence semiangle in mrad.
    scan_sampling_A : float or tuple[float, float]
        Real-space scan pixel size in angstroms. Scalar for isotropic,
        tuple ``(row_A, col_A)`` for anisotropic.
    det_sampling : float or tuple[float, float] or None
        Detector angular sampling in mrad/px. Auto-detected from the
        bright-field disk radius if None.
    aberrations : dict or None
        Starting aberrations ``{C10, C12, phi12}`` in nm / radians.
        If provided, Optuna searches around these values.
    search_ranges : dict or None
        Optuna search ranges. Keys: ``"C10_nm"``, ``"C12_nm"``,
        ``"phi12_deg"``. Values: ``(low, high)`` tuple to search or a
        fixed ``float`` to lock. If None, uses default ranges.
    n_trials : int
        Number of Optuna trials (default 200). Set 0 to skip optimization.
    refine : str or None
        Post-optimization refinement method: ``'nmead'`` (Nelder-Mead,
        default), ``'grid'``, or ``None`` to skip refinement.
    refine_lock : list[str] or None
        Aberration keys to freeze during refinement
        (e.g. ``["C12", "phi12"]`` to refine only defocus).
    rotation_angle_deg : float
        Scan-detector rotation angle in degrees (default 0.0).
    source_path : str or None
        Source file path stored on the result for traceability.
    verbose : bool
        Print progress bars and summaries (default True).
    bf_subsample : int, float, or None, default 2000
        Run optimize() and refine() on a uniform-stride BF-pixel subset of
        this approximate size instead of the full BF disk. Values in
        ``(0, 1]`` are treated as a fraction of the BF disk; values above
        ``1`` are treated as a target BF-pixel count. ~4.5× faster on
        gold_02 with aberration parity within 0.05 nm on C10/C12 (phi is
        weakly constrained and may drift ~0.5°). Set to None to force
        full-BF optimization (legacy path, ~4.5× slower).
    bf_radius : int or None
        Optional radius, in detector pixels, for the BF disk used by SSB.
        Use this to keep large 512x512 microscope scans within GPU memory by
        reconstructing from the central BF disk.

    Returns
    -------
    SSBResult
        .object_wave : cp.ndarray
            Complex transmission function ``(scan_row, scan_col)``.
        .phase : cp.ndarray
            Phase image (property, derived from ``object_wave``).
        .aberrations : dict
            Optimized aberrations ``{C10, C12, phi12}`` in nm / radians.
        .loss : float
            Best variance loss value.
        .elapsed : float
            Wall-clock time in seconds.

    Examples
    --------
    Full optimization from scratch::

        from quantem.gpu import load, ssb
        data, _ = load('scan_master.h5')
        result = ssb(data, voltage_kV=300, semiangle_mrad=30,
                     scan_sampling_A=0.5)

    With DPC-calibrated rotation and locked astigmatism::

        result = ssb(data, voltage_kV=300, semiangle_mrad=30,
                     scan_sampling_A=0.5,
                     rotation_angle_deg=dpc_result.rotation_angle_deg,
                     refine_lock=["C12", "phi12"])
    """
    from quantem.gpu.ssb import SSB as _SSB

    t0 = time.perf_counter()
    starting = _resolve_aberrations(aberrations)
    engine = _SSB(
        data=data,
        semiangle=semiangle_mrad,
        scan_sampling=scan_sampling_A,
        det_sampling=det_sampling,
        voltage_kV=voltage_kV,
        scan_shape=scan_shape,
        bf_radius=bf_radius,
        aberrations=starting,
        rotation_angle_deg=rotation_angle_deg,
    )
    # Release the raw data reference so the 19 GB block can be freed.
    # Critical for L40S 48 GB budget on 512x512 scans.
    del data
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    bf_subsample_ratio = _resolve_bf_subsample_ratio(
        int(len(engine.bf_inds_row)), bf_subsample,
    )

    if n_trials > 0:
        engine.optimize(
            n_trials=n_trials, aberrations=search_ranges, verbose=verbose,
            bf_subsample=bf_subsample_ratio,
        )
    if refine in ("nmead", "nelder-mead"):
        engine.refine(verbose=verbose, lock=refine_lock, bf_subsample=bf_subsample_ratio)
    elif refine == "grid":
        engine.grid_search(verbose=verbose)
    elif refine is not None:
        raise ValueError(f"refine must be 'nmead', 'grid', or None, got {refine!r}")

    result = engine.result()
    result.elapsed = time.perf_counter() - t0
    result.source_path = source_path
    del engine
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result


# =========================================================================
#  SSB Series
# =========================================================================


def _format_time(seconds: float) -> str:
    """Format elapsed time per CLAUDE.md rules."""
    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}m {secs}s"


def ssb_series(
    files: str | list[str],
    *,
    calibration: SSBResult,
    mode: Literal["time", "defocus"],
    verbose: bool = True,
) -> list[SSBResult]:
    """Run SSB on multiple scans using a pre-calibrated result.

    For **time series** (``mode="time"``): all aberrations locked, no
    optimization. Each scan is reconstructed with the calibration's exact
    aberrations, giving maximum throughput.

    For **defocus ptychography** (``mode="defocus"``): astigmatism (C12,
    phi12) locked, defocus (C10) re-optimized within +/-200 nm of the
    calibration value using 20 Optuna trials + Nelder-Mead refinement.

    Parameters
    ----------
    files : str or list[str]
        Folder path (discovers all ``*_master.h5`` files) or explicit
        list of file paths.
    calibration : SSBResult
        Calibration result from :func:`ssb`. Must have ``voltage_kV``,
        ``semiangle_mrad``, ``scan_sampling_A``, ``rotation_angle_deg``,
        and ``aberrations`` populated.
    mode : {"time", "defocus"}
        ``"time"`` locks all aberrations. ``"defocus"`` re-optimizes C10.
    verbose : bool
        Print per-scan progress (default True).

    Returns
    -------
    list[SSBResult]
        One GPU-resident result per successfully processed scan. Failed
        scans are skipped with a printed warning.

    Examples
    --------
    Time series with a calibration from the first scan::

        from quantem.gpu import load, ssb, ssb_series
        data, _ = load('scan_001_master.h5')
        cal = ssb(data, voltage_kV=300, semiangle_mrad=30,
                  scan_sampling_A=0.5)
        del data
        results = ssb_series('scans/', calibration=cal, mode='time')
    """
    import math
    from pathlib import Path
    from quantem.gpu.io.hdf5 import discover_masters, load
    from quantem.gpu.ssb.api import free_gpu

    # Validate calibration
    for attr in ("voltage_kV", "semiangle_mrad", "scan_sampling_A"):
        if getattr(calibration, attr) is None:
            raise ValueError(
                f"calibration.{attr} is None. "
                f"Re-run ssb() to populate microscope parameters."
            )

    # Resolve file list
    if isinstance(files, str):
        paths = discover_masters(files, verbose=False)
    else:
        paths = list(files)
    if not paths:
        return []

    # Build mode-specific kwargs for ssb()
    ab = calibration.aberrations
    starting = {"C10": float(ab["C10"]), "C12": float(ab["C12"]), "phi12": float(ab["phi12"])}
    if mode == "time":
        series_kwargs = {"aberrations": starting, "n_trials": 0, "refine": None}
    elif mode == "defocus":
        c10 = float(ab["C10"])
        series_kwargs = {
            "aberrations": starting,
            "search_ranges": {
                "C10_nm": (c10 - 200.0, c10 + 200.0),
                "C12_nm": float(ab["C12"]),
                "phi12_deg": math.degrees(float(ab["phi12"])),
            },
            "n_trials": 20,
            "refine": "nmead",
            "refine_lock": ["C12", "phi12"],
        }
    else:
        raise ValueError(f"mode must be 'time' or 'defocus', got {mode!r}")

    results: list[SSBResult] = []
    mode_label = "time series" if mode == "time" else "defocus series"
    if verbose:
        print(f"SSB {mode_label}: {len(paths)} scans, calibration C10={ab['C10']:.1f} nm")

    t_total = time.perf_counter()
    for idx, filepath in enumerate(paths):
        stem = Path(filepath).stem.replace("_master", "")
        data = None
        try:
            data, _ = load(filepath, verbose=False)
            result = ssb(
                data,
                voltage_kV=calibration.voltage_kV,
                semiangle_mrad=calibration.semiangle_mrad,
                scan_sampling_A=calibration.scan_sampling_A,
                rotation_angle_deg=calibration.rotation_angle_deg,
                source_path=filepath,
                verbose=False,
                **series_kwargs,
            )
            del data
            data = None
            results.append(result)
            loss_str = f"loss={result.loss:.6f}" if result.loss is not None else ""
            c10_str = f"C10={result.aberrations['C10']:.1f}" if mode == "defocus" else ""
            detail = "  ".join(s for s in [c10_str, loss_str] if s)
            if verbose:
                print(f"  [{idx + 1:>2}/{len(paths)}] {stem}  {_format_time(result.elapsed)}  {detail}")
        except (RuntimeError, MemoryError, OSError, ValueError) as e:
            if verbose:
                print(f"  [{idx + 1:>2}/{len(paths)}] {stem}  FAILED: {e}")
        finally:
            del data
            free_gpu()
        cp.fft.config.get_plan_cache().clear()

    elapsed = time.perf_counter() - t_total
    if verbose:
        print(f"  Done: {len(results)} of {len(paths)} scans in {_format_time(elapsed)}")
    return results


# =========================================================================
#  Defocus Sweep
# =========================================================================


def defocus_sweep(
    data: cp.ndarray,
    scan_shape: tuple[int, int] | None = None,
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    scan_sampling_A: float | tuple[float, float],
    det_sampling: float | tuple[float, float] | None = None,
    aberrations: dict | None = None,
    rotation_angle_deg: float = 0.0,
    c10_range_nm: tuple[float, float] = (-100, 100),
    n_steps: int = 21,
    show: bool = True,
    n_cols: int = 4,
    axsize: tuple[int, int] = (3, 3),
    verbose: bool = True,
) -> DefocusSweepResult:
    """Sweep defocus (C10) to visualize the aberration landscape.

    Reconstructs phase images at evenly-spaced C10 values and computes the
    variance loss at each point. When *aberrations* is provided, C12 and
    phi12 are locked to those values during the sweep, useful for refining
    defocus after a full optimization.

    Parameters
    ----------
    data : cp.ndarray
        3D ``(N, det_row, det_col)`` or 4D ``(scan_row, scan_col, det_row, det_col)``.
    scan_shape : tuple[int, int] or None
        Required when *data* is 3D and non-square.
    voltage_kV : float
        Accelerating voltage in kV.
    semiangle_mrad : float
        Probe convergence semiangle in mrad.
    scan_sampling_A : float or tuple[float, float]
        Real-space scan pixel size in angstroms.
    det_sampling : float or tuple[float, float] or None
        Detector angular sampling in mrad/px. Auto-detected if None.
    aberrations : dict or None
        Locked aberrations ``{C10, C12, phi12}`` in nm / radians. If provided,
        C12 and phi12 are held fixed while C10 is swept. The C10 value in
        the dict is ignored (overridden by ``c10_range_nm``).
    rotation_angle_deg : float
        Scan-detector rotation angle in degrees (default 0.0).
    c10_range_nm : tuple[float, float]
        Min and max defocus in nm (default ``(-100, 100)``).
    n_steps : int
        Number of defocus values to evaluate (default 21).
    verbose : bool
        Print progress (default True).

    Returns
    -------
    DefocusSweepResult
        .c10_values_nm : np.ndarray
            Defocus values swept, in nm.
        .losses : np.ndarray
            Variance loss at each defocus value.
        .images : np.ndarray
            Phase images ``(n_steps, scan_row, scan_col)``.
        .best_c10_nm : float
            Defocus with the lowest loss, in nm.
        .best_loss : float
            Lowest loss value found.

    Examples
    --------
    Sweep defocus after full SSB optimization::

        from quantem.gpu import load, ssb, defocus_sweep
        data, _ = load('scan_master.h5')
        result = ssb(data, voltage_kV=300, semiangle_mrad=30,
                     scan_sampling_A=0.5)
        sweep = defocus_sweep(data, voltage_kV=300, semiangle_mrad=30,
                              scan_sampling_A=0.5,
                              aberrations=result.aberrations,
                              c10_range_nm=(-50, 50))

    Or use the convenience method on SSBResult::

        sweep = result.defocus_sweep(data, c10_range_nm=50)
    """
    from quantem.gpu.ssb import SSB as _SSB

    coefs = _resolve_aberrations(aberrations)

    engine = _SSB(
        data=data,
        semiangle=semiangle_mrad,
        scan_sampling=scan_sampling_A,
        det_sampling=det_sampling,
        voltage_kV=voltage_kV,
        scan_shape=scan_shape,
        aberrations=coefs,
        rotation_angle_deg=rotation_angle_deg,
    )
    del data
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    result = engine._defocus_sweep(c10_range_nm=c10_range_nm, n_steps=n_steps)
    del engine
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    if show:
        print(f"Best C10: {result.best_c10_nm:.1f} nm (loss: {result.best_loss:.6f})")
        from quantem.core.visualization import show_2d
        images = list(result.images)
        labels = result.labels
        rows = [images[i:i + n_cols] for i in range(0, len(images), n_cols)]
        title_rows = [labels[i:i + n_cols] for i in range(0, len(labels), n_cols)]
        show_2d(rows, title=title_rows, axsize=axsize)

    return result
