"""Regularized direct reconstruction using the existing calibrated session."""

import hashlib
import json
import math
import time
from contextlib import nullcontext
from functools import wraps

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.ssb.persistence import load_result, result_paths
from quantem.gpu.ssb.results import SSBResult
from quantem.gpu.ssb.units import validate_aberrations

# drag-window tiles kept per control: a drag back and forth across one tile edge needs two
TILES = 3


def settings(
    method,
    phase_regularization,
    amplitude_regularization,
    smoothness_inv_A,
    detector_weighting,
):
    """Resolve only the controls that belong to the selected reconstruction."""
    if method not in ("ssb", "phase", "complex"):
        raise ValueError(f"method must be 'ssb', 'phase' or 'complex'; got {method!r}.")
    if method == "ssb":
        if any(
            value is not None
            for value in (
                phase_regularization,
                amplitude_regularization,
                smoothness_inv_A,
                detector_weighting,
            )
        ):
            raise ValueError(
                "Regularization controls need method='phase' or 'complex'; omit them for method='ssb'."
            )
        return {}
    if method == "phase" and amplitude_regularization is not None:
        raise ValueError(
            "amplitude_regularization needs method='complex'; omit it for method='phase'."
        )
    strengths = {
        "phase_regularization": 0.03
        if phase_regularization is None
        else float(phase_regularization)
    }
    if method == "complex":
        strengths["amplitude_regularization"] = (
            0.03
            if amplitude_regularization is None
            else float(amplitude_regularization)
        )
    for name, value in strengths.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive; got {value!r}.")
    smoothness = 1.0 if smoothness_inv_A is None else float(smoothness_inv_A)
    if not math.isfinite(smoothness) or smoothness < 0:
        raise ValueError(
            f"smoothness_inv_A must be nonnegative and finite; use 0 for a flat prior, got {smoothness!r}."
        )
    weighting = "uniform" if detector_weighting is None else detector_weighting
    if weighting not in ("uniform", "inverse_mean"):
        raise ValueError(
            f"detector_weighting must be 'uniform' or 'inverse_mean'; got {weighting!r}."
        )
    return dict(
        method=method,
        **strengths,
        smoothness_inv_A=smoothness,
        detector_weighting=weighting,
    )


def solve(session, coefs, tilt, depth, factor, policy, *, compute_loss, force=False):
    """Reuse a prepared system while every fixed input agrees.

    One prepared system per method is kept for the current data and optics, so
    switching phase -> complex -> phase prepares each once (about 92 + 170 MB on
    the 512 x 512 test field). Any change to the optics, sampling or prior drops
    both. CUDA prepares and solves on the session's GPU (``ssb.cuda.direct``);
    MPS prepares on the Apple GPU and solves in float64 on the CPU
    (``ssb.mps.direct``), from the same model and layout.
    """
    if session._streamed is not None:
        raise NotImplementedError(
            "Regularized direct reconstruction needs a resident region; use select_region(...) first."
        )
    if session.backend == "cuda":
        from quantem.gpu.ssb.cuda.direct import PreparedDirect
    else:
        from quantem.gpu.ssb.mps.direct import PreparedDirect

    engine, optics, key = _inputs(session, coefs, tilt, depth, factor, policy)
    cached = getattr(session, "_prepared_direct", None)
    if cached is None or cached[0] != key:
        # Retain the current optics only, so exploring changed optics cannot grow memory without bound.
        session._prepared_direct = None
        cached = session._prepared_direct = key, {}
    prepared = cached[1].get(policy["method"])
    if force or prepared is None:
        cached[1].pop(policy["method"], None)
        prepared = PreparedDirect.prepare(
            engine,
            optics,
            factor,
            complex_object=policy["method"] == "complex",
            smoothness_inv_A=policy["smoothness_inv_A"],
            detector_weighting=policy["detector_weighting"],
        )
        cached[1][policy["method"]] = prepared
    return prepared.solve(
        policy["phase_regularization"],
        policy.get("amplitude_regularization"),
        compute_loss=compute_loss,
    )


def _on_scan_gpu(function):
    """Run with the session's GPU current, as the session's own methods do: drags arrive from a widget, not a method."""
    @wraps(function)
    def on_scan_gpu(session, *args):
        with nullcontext() if session._device is None else cp.cuda.Device(session._device):
            return function(session, *args)
    return on_scan_gpu


@_on_scan_gpu
def drag(session, control, coefs, tilt, depth, factor, policy):
    """The float32 phase (on the device) while ``control`` is dragged, every other input held.

    Serves: ShowPtycho slider drags, C10, C12 and phi12 typically, at 3x on
    the full scan. Regularized methods are served from a `DragWindow` along
    that control (``ssb.cuda.direct`` or ``ssb.mps.direct``), from the prepared
    system when the optics are those already prepared (a strength drag), and
    otherwise exactly by `solve`. SSB on CUDA
    (``policy`` from ``settings("ssb", ...)``) on a fine grid is served from
    a `WaveWindow`: the single-sideband terms interpolated, the triple-overlap
    terms exact. A
    window is valid while every other input is unchanged and the value lies
    inside it; otherwise a new one is centred on this value. `advance` builds
    windows one node at a time between frames, with exact frames meanwhile,
    and `prepare` asks for windows before a grab. Committed frames and saved
    results never come from a window, so they remain the exact model.
    """
    method = policy.get("method", "ssb")
    if not _windowed(session, factor, policy):
        return _exact(session, coefs, tilt, depth, factor, policy)
    if method != "ssb":
        key = _inputs(session, coefs, tilt, depth, factor, policy)[2]
        cached = getattr(session, "_prepared_direct", None)
        if cached is not None and cached[0] == key and method in cached[1]:
            return _exact(session, coefs, tilt, depth, factor, policy)
    window = _window(session, control, coefs, tilt, depth, factor, policy, first=True)
    if not window.series.ready:
        return _exact(session, coefs, tilt, depth, factor, policy)
    value = float(coefs.get(control, 0.0))
    if method == "ssb":
        return window.phase(value)
    wave, _ = window.solve(value, policy["phase_regularization"], policy.get("amplitude_regularization"),
                           complex_object=method == "complex")
    return _phase(session, wave)


@_on_scan_gpu
def prepare(session, controls, coefs, tilt, depth, factor, policy):
    """Ask for windows along ``controls`` at the current optics, so the next grab of one starts on window frames.

    Serves: the idle time after a slider is released, while the pointer moves
    to the next one. Nothing is computed here; `advance` builds them.
    """
    if _windowed(session, factor, policy):
        for control in controls:
            _window(session, control, coefs, tilt, depth, factor, policy)


@_on_scan_gpu
def advance(session) -> bool:
    """Add one node to the first incomplete window (the one dragged last first); True when none is left."""
    windows = [tile for _, tiles, _ in getattr(session, "_direct_windows", {}).values() for tile in tiles]
    pending = [window for window in windows if not window.series.ready]
    if pending:
        pending[0].step()
    return all(window.series.ready for window in windows)


def _windowed(session, factor, policy) -> bool:
    """Whether drags of this method can use a window: a resident region, and for SSB CUDA's fine-grid coherent sum."""
    if session._streamed is not None:
        return False
    if policy.get("method", "ssb") == "ssb":
        return session.backend == "cuda" and factor > 1
    return True


def _window(session, control, coefs, tilt, depth, factor, policy, *, first=False):
    """The session's window along ``control`` covering its value, new (unbuilt) unless a valid one exists.

    Windows of one control are tiles on a fixed grid, centred at
    c0 + 2 h k from the first one (c0 kept while the held inputs are), so a drag that crosses a tile edge and
    comes back reuses both instead of building a window per crossing; the
    `TILES` most recently used are kept. Windows whose held inputs no longer
    match these are dropped first: a complex one holds about 4 GB on the test
    field. ``first`` puts this control and tile ahead in `advance`'s order.
    """
    method = policy.get("method", "ssb")
    complex_object = method == "complex"
    kind = "ssb" if method == "ssb" else "direct"
    windows = getattr(session, "_direct_windows", None)
    if windows is None:
        windows = session._direct_windows = {}
    for name, (held, *_) in list(windows.items()):
        if held != (kind, *_inputs(session, {**coefs, name: None}, tilt, depth, factor, policy)[2]):
            del windows[name]
    held = (kind, *_inputs(session, {**coefs, control: None}, tilt, depth, factor, policy)[2])
    value = float(coefs.get(control, 0.0))
    _, tiles, anchor = windows.get(control, (held, [], value))
    if complex_object and tiles and not tiles[0].complex_object:
        tiles, anchor = [], value
    window = next((tile for tile in tiles if tile.series.covers(value)), None)
    if window is None:
        center = value
        if tiles and not math.isinf(tiles[0].series.half_width):
            width = 2 * tiles[0].series.half_width
            center = anchor + width * round((value - anchor) / width)
        tile_coefs = {**coefs, control: center}
        engine, optics, _ = _inputs(session, tile_coefs, tilt, depth, factor, policy)
        if kind == "ssb":
            from quantem.gpu.ssb.cuda.thick import WaveWindow

            window = WaveWindow(engine.thick, tile_coefs, control, tilt, depth, factor)
        else:
            if session.backend == "cuda":
                from quantem.gpu.ssb.cuda.direct import DragWindow
            else:
                from quantem.gpu.ssb.mps.direct import DragWindow

            window = DragWindow(engine, optics, control, factor, complex_object=complex_object,
                                smoothness_inv_A=policy["smoothness_inv_A"],
                                detector_weighting=policy["detector_weighting"])
    windows[control] = held, [window] + [tile for tile in tiles if tile is not window][:TILES - 1], anchor
    if first:
        session._direct_windows = {control: windows[control], **{name: other for name, other in windows.items() if name != control}}
    return window


def _exact(session, coefs, tilt, depth, factor, policy):
    """The exact phase, as ``preview(..., output="native")`` computes it."""
    method = policy.get("method", "ssb")
    if method == "ssb":
        phase, _ = session.preview(coefs, compute_loss=False, upsampling_factor=factor, sample_tilt_mrad=tilt or (0.0, 0.0),
                                   depth_spread_nm=depth or 0.0, output="native")
        return phase
    wave, _ = solve(session, coefs, tilt, depth, factor, policy, compute_loss=False)
    return _phase(session, wave)


def _phase(session, wave):
    """The float32 phase of a direct wave: on the device for CUDA, a NumPy array for MPS."""
    if session.backend == "cuda":
        return cp.angle(wave).astype(cp.float32)
    from quantem.gpu.ssb.mps.direct import wave_phase

    return wave_phase(np.asarray(wave, np.complex64))


def _inputs(session, coefs, tilt, depth, factor, policy):
    """The engine, the optics and the key of every input a prepared system depends on.

    The engine is the CUDA accelerator, or the MPS backend with its G(q, k) prepared at the session's rotation.
    """
    if session.backend == "cuda":
        engine = session._backend()._get_accelerator()
        spectra = engine.G_qk
    else:
        engine = session._backend()
        engine.cache_rotation(math.radians(session.scan_detector_rotation_deg))
        spectra = engine._prepared.g_qk
    optics = {
        "aberrations": coefs,
        "sample_tilt_mrad": tilt or (0.0, 0.0),
        "depth_spread_nm": depth or 0.0,
    }
    key = (
        id(engine),
        id(spectra),
        session.scan_detector_rotation_deg,
        tuple(sorted(coefs.items())),
        tuple(optics["sample_tilt_mrad"]),
        optics["depth_spread_nm"],
        factor,
        policy.get("smoothness_inv_A"),
        policy.get("detector_weighting"),
    )
    return engine, optics, key


def run(
    session,
    parameters,
    *,
    aberrations,
    factor,
    policy,
    compute_loss,
    save_dir,
    force,
    verbose,
):
    """Apply one regularized method with explicit provenance and saved reuse."""
    if parameters is not None and not isinstance(parameters, SSBResult):
        raise TypeError(
            "Pass the result of find_aberrations(), or coefficients through aberrations={...}."
        )
    if parameters is None and aberrations is None and not session._aberrations_explicit:
        parameters = session.find_aberrations(verbose=verbose)
    correction = parameters
    if correction is not None:
        coefs, tilt, depth = (
            dict(correction.aberrations),
            correction.sample_tilt_mrad,
            correction.depth_spread_nm,
        )
        if correction.scan_detector_rotation_deg != session.scan_detector_rotation_deg:
            session.set_rotation(correction.scan_detector_rotation_deg)
    else:
        coefs, tilt, depth = (
            dict(session.aberrations),
            session.sample_tilt_mrad,
            session.depth_spread_nm,
        )
    coefs = validate_aberrations({**coefs, **(aberrations or {})}, higher_order=True)
    options = dict(
        policy,
        aberrations=coefs,
        sample_tilt_mrad=tilt,
        depth_spread_nm=depth,
        upsampling_factor=factor,
        compute_loss=bool(compute_loss),
    )
    paths = signature = None
    if save_dir is not None:
        signature = session._result_signature("direct", options)
        key = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()[:16]
        paths = result_paths(save_dir, f"direct-{policy['method']}-{key}")
        if not force and session.source_kind != "array":
            reused = load_result(
                paths=paths, signature=signature, backend=session.backend
            )
            if reused is not None:
                if verbose:
                    print(f"Loaded an exact saved direct result: {paths[0]}")
                return session._accept_result(reused)
    started = time.perf_counter()
    wave, diagnostics = solve(
        session,
        coefs,
        tilt,
        depth,
        factor,
        policy,
        compute_loss=compute_loss,
        force=force,
    )
    if session.backend == "cuda":
        cp.cuda.get_current_stream().synchronize()
    result = SSBResult(
        object_wave=wave,
        backend=session.backend,
        aberrations=coefs,
        sample_tilt_mrad=tilt,
        depth_spread_nm=depth,
        scan_detector_rotation_deg=session.scan_detector_rotation_deg,
        elapsed_s=time.perf_counter() - started,
        num_bright_field_pixels=session.num_bright_field_pixels,
        voltage_kV=session.voltage_kV,
        semiangle_mrad=session.semiangle_mrad,
        object_sampling_A=session._output_sampling(factor),
        upsampling_factor=factor,
        phase_estimator=policy["method"],
        amplitude_estimated=policy["method"] == "complex",
        reconstruction_method=policy["method"],
        reconstruction_options=dict(policy),
        reconstruction_diagnostics=diagnostics,
        phase_clim=correction.phase_limits if correction is not None else None,
        fit_scan_region=correction.fit_scan_region if correction is not None else None,
        **session._fit_evidence(correction, coefs, tilt, depth),
    )
    # The relative prediction error is not the conventional phase-variance loss.
    result = session._accept_result(result)
    if paths is not None:
        result = session._save_result(result, paths=paths, signature=signature)
    if verbose:
        print(
            f"Direct {policy['method']}: {tuple(wave.shape)}, {result.elapsed_s:.3f} s; effective depth {depth or 0:.3f} nm."
        )
    return result
