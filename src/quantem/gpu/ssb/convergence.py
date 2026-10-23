"""Plot recorded SSB objectives without evaluating the model again."""

import math

import numpy as np

from quantem.gpu.display._layout import resolve_columns


def plot_convergence(result, *, axsize, rows=None, columns=None, compare=None, x="auto"):
    """Return a closed Matplotlib figure of saved search and refinement evidence."""
    import matplotlib.pyplot as plt

    if x not in ("auto", "time", "evaluations"):
        raise ValueError(f"Use x='auto', 'time' or 'evaluations'; got {x!r}.")
    if compare is not None:
        return _plot_comparison(result, compare, axsize=axsize, x=x)
    optimizer = result.optimizer
    coarse = optimizer.get("coarse_scores")
    groups = {}
    if (coarse is not None and len(coarse)) or optimizer.get("refinement_trace"):
        groups["broad"] = coarse
    else:
        for record in result.trial_records or ():
            if record.get("loss") is not None:
                key = (record.get("search", 0), record.get("objective", result.objective or "loss"))
                groups.setdefault(key, []).append(record)
    if not groups:
        raise ValueError("This result has no recorded search history. Use the result returned by "
                         "ssb.find_aberrations(), or reload a fit saved with its history.")
    if "broad" in groups and x == "time" and not _broad_history(result)[-1]:
        raise ValueError("This fit has no complete elapsed-time history. Use x='evaluations', or rerun it.")
    count = len(groups)
    ncols = resolve_columns(count, rows=rows, columns=columns, default=min(count, 2))
    nrows = math.ceil(count / ncols)
    figure, axes = plt.subplots(nrows, ncols, squeeze=False, layout="constrained",
                               figsize=(axsize[0] * ncols, axsize[1] * nrows))
    for ax, (key, records) in zip(axes.flat, groups.items()):
        if key == "broad":
            axis = ("time" if _broad_history(result)[-1] else "evaluations") if x == "auto" else x
            _plot_broad(ax, result, x=axis)
            ax.set(title=("Initial fit → joint refinement" if optimizer.get("search") == "standard"
                          else "Broad search + refinement"),
                   ylabel="Normalized agreement (higher is better)")
            stop = optimizer.get("stop")
            if stop:
                ax.text(0, -0.28, f"Stop: {stop}", transform=ax.transAxes, fontsize="small", wrap=True)
        else:
            if x == "time":
                plt.close(figure)
                raise ValueError("This standard fit has no elapsed-time history. Use x='evaluations'.")
            search, objective = key
            losses = np.asarray([item["loss"] for item in records], dtype=float)
            best = np.minimum.accumulate(np.where(np.isfinite(losses), losses, np.inf))
            evaluations = np.arange(1, len(records) + 1)
            ax.plot(evaluations, np.where(np.isfinite(best), best, np.nan), label="Best recorded loss")
            refinement = [i + 1 for i, item in enumerate(records) if item.get("stage") == "refinement"]
            if refinement:
                ax.axvline(refinement[0], color="0.6", linestyle=":", linewidth=1, label="Refinement begins")
            elif result.refinement:
                ax.text(0, -0.28, "Refinement trajectory was not recorded.",
                        transform=ax.transAxes, fontsize="small")
            label = {"negative_thick_agreement": "Model agreement", "agreement": "Model agreement",
                     "phase_variance": "Phase variance"}.get(objective, objective)
            ax.set(title=f"Search {search} · {label}", ylabel="Recorded loss (lower is better)")
            ax.set_xlabel("Recorded objective evaluations")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize="small")
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    plt.close(figure)
    return figure


def _broad_history(result):
    """Read scalar histories without accessing the reconstructed wave."""
    optimizer = result.optimizer
    scores = np.asarray(optimizer.get("coarse_scores", []), dtype=float)
    trace = optimizer.get("refinement_trace")
    if trace is None:
        trace = optimizer.get("newton_trace", [])
    trace = np.asarray(trace, dtype=float).reshape(-1, 3)
    seconds = np.asarray(optimizer.get("coarse_elapsed_s", []), dtype=float)
    offset = optimizer.get("refinement_start_s")
    timed = (optimizer.get("convergence_clock") == "fit_start"
             and len(seconds) == len(scores) and (len(seconds) > 0 or len(trace) > 0)
             and np.isfinite(seconds).all() and np.all(np.diff(seconds) >= 0)
             and (not len(seconds) or seconds[0] >= 0) and offset is not None and np.isfinite(offset)
             and float(offset) >= (seconds[-1] if len(seconds) else 0)
             and (not len(trace) or (np.isfinite(trace[:, 2]).all()
                                    and np.all(np.diff(trace[:, 2]) >= 0) and trace[0, 2] >= 0)))
    return scores, trace, seconds, offset, timed


def _comparison_fits(result, compare):
    """Require comparable scientific contexts and keep each fit once."""
    from collections.abc import Mapping
    from quantem.gpu.ssb.results import SSBResult

    entries = [(f"{result.num_trials} trials", result)]
    if isinstance(compare, SSBResult):
        entries.append((f"{compare.num_trials} trials", compare))
    elif isinstance(compare, Mapping):
        entries = [(f"{key} trials" if isinstance(key, int) else str(key), fit)
                   for key, fit in compare.items()]
        if not any(fit is result for _, fit in entries):
            entries.insert(0, (f"{result.num_trials} trials", result))
    else:
        raise ValueError("compare must be a fit or a mapping such as {200: fit200, 400: fit400}.")
    unique = []
    seen = set()
    fields = ("objective", "num_bright_field_pixels", "source_path", "fit_scan_region",
              "voltage_kV", "semiangle_mrad", "object_sampling_A", "scan_detector_rotation_deg",
              "scan_detector_rotation_flipped")
    for label, fit in entries:
        if not isinstance(fit, SSBResult):
            raise ValueError(f"Comparison {label!r} is not an SSB fit. Supply fitted results.")
        if id(fit) in seen:
            continue
        seen.add(id(fit))
        if not len(fit.optimizer.get("coarse_scores", [])) and not fit.optimizer.get("refinement_trace"):
            raise ValueError("Convergence overlays need fifth-order search histories. "
                             "Show standard searches separately with fit.show('convergence').")
        if (any(getattr(fit, name) != getattr(result, name) for name in fields)
                or tuple(fit.object_wave.shape) != tuple(result.object_wave.shape)
                or any(fit.optimizer.get(name) != result.optimizer.get(name)
                       for name in ("model", "fit_band_inv_A", "primary_seed", "primary_trials"))):
            raise ValueError("Compare fits of the same data, calibration, region and objective. "
                             "Use trials=(200, 400, 600) on one SSB session.")
        # Geometry is captured at fit time and survives saving; it includes detector sampling.
        first_context = (result.calibration_context or {}).get("session_identity")
        next_context = (fit.calibration_context or {}).get("session_identity")
        if first_context != next_context:
            raise ValueError("Fit acquisition/calibration identities differ. Compare fits from one session.")
        unique.append((label, fit))
    return unique


def _plot_broad(ax, result, *, x, label=None, color=None, scale=1.0):
    """Draw TPE and refinement on a common recorded clock or evaluation count."""
    scores, trace, seconds, offset, timed = _broad_history(result)
    if x == "time" and not timed:
        raise ValueError("This fit has no complete elapsed-time history. Use x='evaluations', "
                         "or rerun the fit to record times.")
    best = np.maximum.accumulate(np.where(np.isfinite(scores), scores, -np.inf))
    method = {"newton": "Newton", "nelder-mead": "Nelder–Mead"}.get(result.refinement, "Refinement")
    prefix = "" if label is None else f"{label} · "
    if len(scores):
        line, = ax.plot(seconds if x == "time" else np.arange(1, len(scores) + 1),
                        np.where(np.isfinite(best), best * scale, np.nan), linestyle="--",
                        color=color, label=f"{prefix}TPE")
        color = line.get_color()
    if len(trace):
        ax.plot(float(offset) + trace[:, 2] if x == "time" else len(scores) + trace[:, 0],
                trace[:, 1] * scale, color=color, label=f"{prefix}{method}")
    if not len(scores) and x == "time" and float(offset) > 0:
        ax.axvspan(0, float(offset), color="0.7", alpha=0.2, label="Initial fit + setup")
    ax.set_xlabel("Elapsed fit time (s)" if x == "time" else "Recorded fifth-order objective evaluations")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize="small")
    ax.set_xlim(left=0)


def _plot_comparison(result, compare, *, axsize, x):
    """Overlay comparable fits, undoing their individual score normalization."""
    import matplotlib.pyplot as plt

    entries = _comparison_fits(result, compare)
    if x == "auto":
        x = "time" if all(_broad_history(fit)[-1] for _, fit in entries) else "evaluations"
    if x == "time" and not all(_broad_history(fit)[-1] for _, fit in entries):
        raise ValueError("Some fits lack elapsed times. Use x='evaluations' or rerun those fits.")
    scales = [fit.optimizer.get("normalization") for _, fit in entries]
    if any(scale is None or not np.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("Comparisons require each fit's objective normalization. Rerun the broad fits.")
    figure, ax = plt.subplots(figsize=axsize, layout="constrained")
    for (label, fit), scale in zip(entries, scales, strict=True):
        _plot_broad(ax, fit, x=x, label=label, scale=scale)
    ax.set(title="Search budget comparison", ylabel="Model agreement (higher is better)")
    stops = [f"{label}: {fit.optimizer.get('stop', 'not recorded')}" for label, fit in entries]
    ax.text(0, -.28, "Stop: " + "\n".join(stops), transform=ax.transAxes, fontsize="small")
    plt.close(figure)
    return figure
