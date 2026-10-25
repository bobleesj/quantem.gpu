"""Regularized direct reconstruction using the existing calibrated session."""

import hashlib
import json
import math
import time

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.ssb.persistence import load_result, result_paths
from quantem.gpu.ssb.results import SSBResult
from quantem.gpu.ssb.units import validate_aberrations


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
    both.
    """
    if session.backend != "cuda":
        raise NotImplementedError(
            "Regularized direct reconstruction currently requires CUDA; use method='ssb' on MPS."
        )
    if session._streamed is not None:
        raise NotImplementedError(
            "Regularized direct reconstruction needs a resident region; use select_region(...) first."
        )
    from quantem.gpu.ssb.cuda.direct import PreparedDirect

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


def drag(session, control, coefs, tilt, depth, factor, policy):
    """The wave while ``control`` is dragged, every other input held.

    Served from the session's `DragWindow` along that control when it is
    complete and covers the value, from the prepared system when the optics
    are those already prepared (a strength drag), and otherwise exactly by
    `solve`. A window for another control, other fixed inputs, or a value
    outside it is replaced by a new one centred on this value; `advance` then
    builds it one node at a time between frames. Committed frames and saved
    results always use `solve`, so they remain the exact model.
    """
    from quantem.gpu.ssb.cuda.direct import DragWindow

    if session.backend != "cuda" or session._streamed is not None:
        return solve(session, coefs, tilt, depth, factor, policy, compute_loss=False)
    engine, optics, key = _inputs(session, coefs, tilt, depth, factor, policy)
    cached = getattr(session, "_prepared_direct", None)
    if cached is not None and cached[0] == key and policy["method"] in cached[1]:
        return solve(session, coefs, tilt, depth, factor, policy, compute_loss=False)
    held = _inputs(session, {**coefs, control: None}, tilt, depth, factor, policy)[2]
    complex_object = policy["method"] == "complex"
    window = getattr(session, "_direct_window", None)
    value = float(coefs.get(control, 0.0))
    if (window is None or window[0] != held or window[1].control != control
            or not window[1].covers(value) or (complex_object and not window[1].complex_object)):
        # drop the old window first: a complex one holds about 2 GB on the test field
        session._direct_window = None
        window = session._direct_window = held, DragWindow(
            engine, optics, control, factor, complex_object=complex_object,
            smoothness_inv_A=policy["smoothness_inv_A"], detector_weighting=policy["detector_weighting"],
        )
    if window[1].ready:
        return window[1].solve(value, policy["phase_regularization"], policy.get("amplitude_regularization"),
                               complex_object=complex_object)
    return solve(session, coefs, tilt, depth, factor, policy, compute_loss=False)


def advance(session) -> bool:
    """Add one node to the pending drag window; True when there is nothing left to build."""
    window = getattr(session, "_direct_window", None)
    return window is None or window[1].step()


def _inputs(session, coefs, tilt, depth, factor, policy):
    """The engine, the optics and the key of every input a prepared system depends on."""
    engine = session._backend()._get_accelerator()
    optics = {
        "aberrations": coefs,
        "sample_tilt_mrad": tilt or (0.0, 0.0),
        "depth_spread_nm": depth or 0.0,
    }
    key = (
        id(engine),
        id(engine.G_qk),
        session.scan_detector_rotation_deg,
        tuple(sorted(coefs.items())),
        tuple(optics["sample_tilt_mrad"]),
        optics["depth_spread_nm"],
        factor,
        policy["smoothness_inv_A"],
        policy["detector_weighting"],
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
    if session.backend != "cuda":
        raise NotImplementedError(
            "Regularized direct reconstruction currently requires CUDA; use method='ssb' on MPS."
        )
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
