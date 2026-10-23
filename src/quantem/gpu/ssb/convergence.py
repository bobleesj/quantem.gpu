"""Plot recorded SSB objectives without evaluating the model again."""

import math

import numpy as np

from quantem.gpu.display._layout import resolve_columns


def plot_convergence(result, *, axsize, rows=None, columns=None):
    """Return a closed Matplotlib figure of saved search and refinement evidence."""
    import matplotlib.pyplot as plt

    optimizer = result.optimizer
    coarse = optimizer.get("coarse_scores")
    groups = {}
    if coarse is not None and len(coarse):
        groups["broad"] = coarse
    else:
        for record in result.trial_records or ():
            if record.get("loss") is not None:
                key = (record.get("search", 0), record.get("objective", result.objective or "loss"))
                groups.setdefault(key, []).append(record)
    if not groups:
        raise ValueError("This result has no recorded search history. Use the result returned by "
                         "ssb.find_aberrations(), or reload a fit saved with its history.")
    count = len(groups)
    ncols = resolve_columns(count, rows=rows, columns=columns, default=min(count, 2))
    nrows = math.ceil(count / ncols)
    figure, axes = plt.subplots(nrows, ncols, squeeze=False, layout="constrained",
                               figsize=(axsize[0] * ncols, axsize[1] * nrows))
    for ax, (key, records) in zip(axes.flat, groups.items()):
        if key == "broad":
            scores = np.asarray(records, dtype=float)
            best = np.maximum.accumulate(np.where(np.isfinite(scores), scores, -np.inf))
            ax.plot(np.arange(1, len(scores) + 1), np.where(np.isfinite(best), best, np.nan),
                    label="TPE · best recorded")
            trace = optimizer.get("refinement_trace")
            if trace is None:
                trace = optimizer.get("newton_trace")
            if trace is not None and len(trace):
                trace = np.asarray(trace, dtype=float)
                method = {"newton": "Newton", "nelder-mead": "Nelder–Mead"}.get(result.refinement, "Refinement")
                ax.plot(len(scores) + trace[:, 0], trace[:, 1], label=f"{method} · best recorded")
                ax.axvline(len(scores), color="0.6", linestyle=":", linewidth=1)
            ax.set(title="Broad search + refinement", ylabel="Normalized agreement (higher is better)")
            stop = optimizer.get("stop")
            if stop:
                ax.text(0, -0.28, f"Stop: {stop}", transform=ax.transAxes, fontsize="small", wrap=True)
        else:
            search, objective = key
            losses = np.asarray([item["loss"] for item in records], dtype=float)
            best = np.minimum.accumulate(np.where(np.isfinite(losses), losses, np.inf))
            x = np.arange(1, len(records) + 1)
            ax.plot(x, np.where(np.isfinite(best), best, np.nan), label="Best recorded loss")
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
