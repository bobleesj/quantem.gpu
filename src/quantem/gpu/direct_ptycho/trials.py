"""Select and display recorded SSB search attempts without retaining image stacks."""

import math
from copy import deepcopy

import numpy as np

from quantem.gpu.optics.physics import wrap_degrees
from quantem.gpu.direct_ptycho.results import SSBResult, host_array


def resolve_trial_count(trials: int | None, num_trials: int | None, *, default: int = 200) -> int:
    """Resolve the search budget while preserving the legacy keyword."""
    if trials is not None and num_trials is not None:
        raise TypeError("Use trials=200, or the legacy num_trials keyword, not both.")
    value = trials if trials is not None else num_trials
    return default if value is None else value


def label_trials(result: SSBResult, *, search: int) -> None:
    """Give every trial record its index, search pass and physical rotation.

    Backends record only the optimized parameters. Without these keys
    ``show_trials`` and ``reconstruct(trial_ids=...)`` cannot replay a trial
    at its own rotation and sample geometry.
    """
    for index, record in enumerate(result.trial_records or ()):
        rotation = wrap_degrees(record["params"].pop("scan_detector_rotation_deg", result.scan_detector_rotation_deg), 360.0)
        record.update(trial=index, search=search, scan_detector_rotation_deg=rotation)
        record.setdefault("stage", "search")
        record.setdefault("objective", "phase_variance")
        record.setdefault("sample_tilt_mrad", [0.0, 0.0])
        record.setdefault("depth_spread_nm", 0.0)


def select_records(records: list[dict], trial_ids) -> list[dict]:
    """Select stable trial IDs in requested order, preserving their full geometry."""
    by_id = {record["trial"]: record for record in records or ()}
    ids = list(trial_ids)
    if not ids:
        raise ValueError("Select at least one trial from aberrations.trials.index.")
    missing = [index for index in ids if index not in by_id]
    if missing:
        raise ValueError(f"Unknown trial IDs {missing}; select IDs from aberrations.trials.index.")
    return [deepcopy(by_id[index]) for index in ids]


def choose_trials(records: list[dict], *, best=None, last=None, first=None) -> list[dict]:
    """Choose completed trials, ranking only within one search and objective."""
    choices = [count for count in (best, last, first) if count is not None]
    if len(choices) != 1 or type(choices[0]) is not int or choices[0] <= 0:
        raise ValueError("Use one positive count: best=5, last=5, or first=5.")
    completed = [record for record in records or () if record.get("stage", "search") == "search" and math.isfinite(record["loss"])]
    if not completed:
        raise ValueError("No completed trials were recorded; run find_aberrations(trials=200).")
    if best is not None:
        latest = max(record["search"] for record in completed)
        group = [record for record in completed if record["search"] == latest]
        if len({record["objective"] for record in group}) != 1:
            raise ValueError("Trials use different objectives; select their IDs explicitly for inspection.")
        return sorted(group, key=lambda record: record["loss"])[:best]
    return completed[-last:] if last is not None else completed[:first]


def plot_trials(result, selection: str, *, axsize: tuple[float, float] = (4.5, 4.5),
                columns: int = 2, sampling: str | None = None):
    """Return one static figure with shared phase contrast and the recorded parameters.

    ``sampling`` names the output factor in the title (``"4× auto"``); by default the result's factor.
    """
    import matplotlib.pyplot as plt
    from quantem.core.visualization import show_2d

    phase = host_array(result.phase)
    records = result.trial_records
    count = len(records)
    columns = min(columns, count)
    rows = math.ceil(count / columns)
    table_height = .8 + .3 * count
    height = axsize[1] * rows + table_height + .9
    figure = plt.figure(figsize=(axsize[0] * columns, height))
    grid = figure.add_gridspec(
        rows + 1, columns, height_ratios=[axsize[1]] * rows + [table_height],
        left=.025, right=.98, top=1 - .85 / height, bottom=.025,
        hspace=.14, wspace=.08,
    )
    axes = np.array([[figure.add_subplot(grid[row, col]) for col in range(columns)] for row in range(rows)])
    panels = list(phase) + [np.zeros_like(phase[0])] * (rows * columns - count)
    titles = [f"Trial {record['trial']}" for record in records] + [""] * (len(panels) - count)
    lower, upper = np.quantile(phase, [.01, .99])
    bar = {"sampling": float(result.object_sampling_A[1]),
           "units": "Å", "fontsize": 13}
    show_2d(
        [panels[start:start + columns] for start in range(0, len(panels), columns)],
        figax=(figure, axes),
        title=[titles[start:start + columns] for start in range(0, len(titles), columns)],
        cmap="inferno", vmin=float(lower), vmax=float(upper), scalebar=bar,
        cbar=False, tight_layout=False, title_fontsize=16,
    )
    for axis in axes.ravel()[count:]:
        axis.set_visible(False)
    table_axis = figure.add_subplot(grid[-1, :])
    table_axis.axis("off")
    labels = ["Trial", "C10\n(nm)", "C12\n(nm)", "Angle\n(deg)", "Tilt row/col\n(mrad)",
              "Depth spread\n(nm)", "Rotation\n(deg)", "Search loss"]
    values = []
    for record in records:
        params = record["params"]
        values.append([record["trial"], f"{params['C10_nm']:.2f}", f"{params['C12_nm']:.2f}",
                       f"{params['phi12_deg']:.1f}",
                       f"{record['sample_tilt_mrad'][0]:.1f}, {record['sample_tilt_mrad'][1]:.1f}",
                       f"{record['depth_spread_nm']:.2f}", f"{record['scan_detector_rotation_deg']:.1f}", f"{record['loss']:.5g}"])
    table = table_axis.table(cellText=values, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10 if columns == 2 else 8)
    table.scale(1, 1.8)
    names = {"phase_variance": "native phase variance", "negative_thick_agreement": "negative thick-sample agreement"}
    objectives = ", ".join(sorted({names.get(record["objective"], record["objective"]) for record in records}))
    sampling = f"{result.upsampling_factor}×" if sampling is None else sampling
    figure.suptitle(f"{selection} {count} search trials · {sampling} · phase (rad)", fontsize=18, y=1 - .08 / height)
    figure.text(.5, 1 - .5 / height,
                f"Shared contrast: {lower:.3g} to {upper:.3g} rad (pooled 1–99%) · {objectives}",
                ha="center", fontsize=11)
    plt.close(figure)
    return figure
