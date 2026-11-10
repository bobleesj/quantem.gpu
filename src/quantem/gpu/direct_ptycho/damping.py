"""Conditional detector holdout for regularized direct phase reconstruction."""

import time

import numpy as np


def validate(session, parameters, phase_damps, *, scan_basis, upsample, block_size):
    """Select damping with fixed optics and two complementary detector masks."""
    from quantem.gpu.direct_ptycho import direct_workflow

    if session.backend != "cuda":
        raise NotImplementedError("Detector-held-out damping validation currently requires CUDA.")
    from quantem.gpu.direct_ptycho.cuda.direct import PreparedDirect
    import cupy as cp

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer in detector pixels.")
    values = np.asarray(phase_damps if phase_damps is not None else
                        np.r_[np.geomspace(1e-5, 3., 33), .03], dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("Provide at least two finite positive phase_damps.")
    values = np.unique(values)
    if len(values) < 2:
        raise ValueError("Provide at least two distinct phase_damps.")
    started = time.perf_counter()
    # This also resolves fitted optics through the established workflow.
    baseline = session.run(parameters, method="phase", phase_damp=.03,
                                   scan_basis=scan_basis, upsample=upsample,
                                   compute_loss=False)
    policy = baseline.reconstruction_options
    engine, optics, _ = direct_workflow._inputs(
        session, baseline.aberrations, baseline.sample_tilt_mrad,
        baseline.depth_spread_nm, baseline.upsample, policy,
    )
    partition = ((cp.asarray(engine.bf_inds_row) // block_size
                  + cp.asarray(engine.bf_inds_col) // block_size) % 2)
    cases = [PreparedDirect.prepare(
        engine, optics, baseline.upsample, complex_object=False,
        smoothness_inv_A=1., detector_weighting="uniform", detector_mask=partition == half,
    ) for half in (0, 1)]
    if any(float(case.signal_power) <= 0 for case in cases):
        raise ValueError("Validation needs nonzero non-DC diffraction signal in both detector subsets.")
    records = []
    for damping in values:
        errors = []
        for train, held in ((cases[0], cases[1]), (cases[1], cases[0])):
            wave, _ = train.solve(float(damping), None, compute_loss=False)
            errors.append(held.phase_error(cp.angle(wave)))
        records.append(dict(phase_damp=float(damping), a_to_b=errors[0],
                            b_to_a=errors[1], mean=float(np.mean(errors))))
    selected = min(records, key=lambda row: row["mean"])
    result = session.run(baseline, method="phase", scan_basis=scan_basis,
                                 phase_damp=selected["phase_damp"],
                                 upsample=baseline.upsample, compute_loss=False)
    cp.cuda.get_current_stream().synchronize()
    result.reconstruction_diagnostics["damping_validation"] = dict(
        records=records, selected_phase_damp=selected["phase_damp"],
        elapsed_s=time.perf_counter() - started, block_size=block_size,
        detector_pixels=[int(cp.sum(partition == half)) for half in (0, 1)],
        model="linear weak-phase scan-Fourier transfer; uniform detector weighting",
        normalization="non-DC squared prediction error / held-out non-DC signal power",
        conditioning="Optics, geometry and full bright-field incident-intensity estimate are shared and fixed.",
        limitations=("Detector pixels may be correlated. Selection is conditional validation, "
                     "not an independent final test or a resolution measurement."),
        reconstruction="Returned phase uses all bright-field pixels at the selected damping.",
    )
    return result


def plot(result, *, axsize):
    """Render recorded holdout scores without rerunning the reconstruction."""
    import matplotlib.pyplot as plt

    evidence = result.reconstruction_diagnostics.get("damping_validation")
    if not evidence:
        raise ValueError("No damping validation is recorded. Run ptycho.validate_damping() first.")
    records = sorted(evidence["records"], key=lambda row: row["phase_damp"])
    x = [row["phase_damp"] for row in records]
    figure, axes = plt.subplots(1, 2, figsize=(2 * axsize[0], axsize[0] * .8),
                               layout="constrained")
    for axis, keys in zip(axes, (("mean",), ("a_to_b", "b_to_a")), strict=True):
        for key in keys:
            axis.semilogx(x, [row[key] for row in records], ".-", label=key.replace("_", " "))
        axis.axvline(evidence["selected_phase_damp"], color="0.4", ls="--")
        axis.set(xlabel="Phase damping", ylabel="Held-out relative squared error")
        axis.ticklabel_format(axis="y", useOffset=False)
        axis.grid(alpha=.2)
        axis.legend()
    axes[0].set_title(f"Mean · selected {evidence['selected_phase_damp']:.4g}")
    axes[1].set_title("Both detector prediction directions")
    figure.suptitle("Fixed optics · detector holdout · lower is better\n"
                   "A shallow minimum does not establish precise damping or resolution")
    plt.close(figure)
    return figure
