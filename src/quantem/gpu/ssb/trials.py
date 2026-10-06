"""Select and display recorded SSB search attempts without retaining image stacks."""

import math
from copy import deepcopy

import numpy as np

from quantem.gpu.ssb.results import SSBResult, host_array, split_rotation


def label_trials(result: SSBResult, *, search: int) -> None:
    """Give every trial record its index, search pass and public rotation.

    Backends record only the optimized parameters. Without these keys
    ``show_trials`` and ``reconstruct(trials=...)`` cannot replay a trial
    at its own rotation and sample geometry.
    """
    for index, record in enumerate(result.trial_records or ()):
        angle, reversed_com = split_rotation(
            record["params"].pop("rotation_angle_deg", result.physical_rotation_deg)
        )
        record.update(trial=index, search=search, rotation_angle_deg=angle,
                      com_reversed=reversed_com)
        record.setdefault("stage", "search")
        record.setdefault("objective", "phase_variance")
        record.setdefault("tilt_row_mrad", 0.0)
        record.setdefault("tilt_col_mrad", 0.0)
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


def plot_trials(result, selection: str, *, axsize: tuple[float, float] = (6, 6)):
    """Return one static figure with shared phase contrast and the recorded parameters."""
    import matplotlib.pyplot as plt
    from quantem.core.visualization import show_2d

    phase = host_array(result.phase)
    records = result.trial_records
    count = len(records)
    columns = min(2, count)
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
    sampling = np.asarray(result.scan_sampling_A)
    bar = {"sampling": float(sampling if sampling.ndim == 0 else sampling[1]),
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
        rotation = record["rotation_angle_deg"] + 180 * record["com_reversed"]
        values.append([record["trial"], f"{params['C10_nm']:.2f}", f"{params['C12_nm']:.2f}",
                       f"{params['phi12_deg']:.1f}",
                       f"{record['tilt_row_mrad']:.1f}, {record['tilt_col_mrad']:.1f}",
                       f"{record['depth_spread_nm']:.2f}", f"{rotation:.1f}", f"{record['loss']:.5g}"])
    table = table_axis.table(cellText=values, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10 if columns == 2 else 8)
    table.scale(1, 1.8)
    names = {"phase_variance": "native phase variance", "negative_thick_agreement": "negative thick-sample agreement"}
    objectives = ", ".join(sorted({names.get(record["objective"], record["objective"]) for record in records}))
    figure.suptitle(f"{selection} {count} search trials · phase (rad)", fontsize=18, y=1 - .08 / height)
    figure.text(.5, 1 - .5 / height,
                f"Shared contrast: {lower:.3g} to {upper:.3g} rad (pooled 1–99%) · {objectives}",
                ha="center", fontsize=11)
    plt.close(figure)
    return figure
