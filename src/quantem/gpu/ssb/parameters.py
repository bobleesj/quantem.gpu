"""Measured, conditional objective slices around a joint SSB solution."""

import time
from collections.abc import Callable
from textwrap import fill
from typing import TYPE_CHECKING

import numpy as np

from quantem.gpu.ssb.thick_sample_fit import DEPTH_SPREAD_RANGE_A

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from quantem.gpu.ssb.results import SSBResult


def measure_slices(
    evaluate: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    *,
    rotation_deg: float,
    band_inv_A: tuple[float, float],
) -> dict[str, object]:
    """Measure two bounded cuts without changing the fit or its objective."""
    started = time.perf_counter()
    point = np.asarray(point, dtype=np.float64)
    reference = float(evaluate(point[None])[0])
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("The fitted model has no positive finite agreement; parameter slices cannot be normalized.")
    # Include the fitted point exactly. All other parameters stay at this point;
    # these are conditional cuts, not projections of the optimizer's trials.
    designs = (
        ("focus_depth", 0, 5, (-300.0, 300.0), DEPTH_SPREAD_RANGE_A, 10.0, 10.0),
        ("tilt", 3, 4, (-25.0, 25.0), (-25.0, 25.0), 1.0, 1.0),
    )
    slices = {}
    evaluations = 1
    for name, column_axis, row_axis, column_range, row_range, column_unit, row_unit in designs:
        columns = np.unique(np.r_[np.linspace(*column_range, 9), point[column_axis]])
        rows = np.unique(np.r_[np.linspace(*row_range, 9), point[row_axis]])
        column_grid, row_grid = np.meshgrid(columns, rows)
        parameters = np.tile(point, (column_grid.size, 1))
        parameters[:, column_axis] = column_grid.ravel()
        parameters[:, row_axis] = row_grid.ravel()
        # Bound device scratch memory to the same batch size as the search.
        agreement = np.concatenate(
            [
                np.asarray(evaluate(parameters[start : start + 8]), dtype=np.float64)
                for start in range(0, len(parameters), 8)
            ]
        ).reshape(column_grid.shape)
        if not np.isfinite(agreement).all():
            raise ValueError(
                "Parameter slices contain non-finite model agreement; the fitted result remains unchanged."
            )
        evaluations += len(parameters)
        slices[name] = {
            "column_values": (columns / column_unit).tolist(),
            "row_values": (rows / row_unit).tolist(),
            "relative_agreement": (agreement / reference).tolist(),
            "fitted_point": [float(point[column_axis] / column_unit), float(point[row_axis] / row_unit)],
        }
    return {
        "schema": "quantem.ssb-parameter-slices/1",
        "objective": "least_squares_model_agreement",
        "normalization": "agreement_at_fitted_point",
        "band_inv_A": list(band_inv_A),
        "scan_detector_rotation_deg": float(rotation_deg),
        "fitted_agreement": reference,
        "evaluations": evaluations,
        "elapsed_s": time.perf_counter() - started,
        "slices": slices,
    }


def plot_parameters(
    result: "SSBResult",
    *,
    view: str = "parameters",
    include_phase: bool = False,
    axsize: tuple[float, float] | None = None,
) -> "Figure":
    """Show recorded slices with physical units and the fitted point."""
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    evidence = result.parameter_slices
    if not evidence:
        raise ValueError(
            "No measured parameter slices are saved. Run ssb.measure_parameters(result) while the session is open."
        )
    descriptions = (
        ("focus_depth", "Focus and depth", "C₁₀ at mid-depth (nm)", "Depth spread (nm)"),
        ("tilt", "Sample tilt", "Tilt row (mrad)", "Tilt col (mrad)"),
    )
    if view != "parameters":
        key = "focus_depth" if view == "depth" else "tilt"
        descriptions = tuple(item for item in descriptions if item[0] == key)
    panels = len(descriptions) + int(include_phase)
    panel_size = (5.2, 4.5) if axsize is None else axsize
    figure, axes = plt.subplots(1, panels, figsize=(panel_size[0] * panels, panel_size[1]), layout="constrained")
    axes = np.atleast_1d(axes)
    if include_phase:
        from quantem.gpu.ssb.results import host_array

        phase = host_array(result.phase)
        sampling_nm = np.asarray(result.object_sampling_A) / 10.0
        phase_image = axes[0].imshow(
            phase,
            cmap="inferno",
            vmin=result.phase_limits[0],
            vmax=result.phase_limits[1],
            extent=(0, phase.shape[1] * sampling_nm[1], phase.shape[0] * sampling_nm[0], 0),
        )
        axes[0].set(title=f"SSB phase · {result.upsampling_factor}× (rad)", xlabel="Scan column (nm)", ylabel="Scan row (nm)")
        figure.colorbar(phase_image, ax=axes[0], label="Phase (rad)", shrink=0.78)
    maximum = max(1.0, *(np.max(item["relative_agreement"]) for item in evidence["slices"].values()))
    parameter_axes = axes[int(include_phase):]
    for axis, (key, title, column_label, row_label) in zip(parameter_axes, descriptions, strict=True):
        data = evidence["slices"][key]
        mesh = axis.pcolormesh(
            data["column_values"],
            data["row_values"],
            data["relative_agreement"],
            cmap="inferno",
            shading="nearest",
            vmin=0,
            vmax=maximum,
        )
        axis.plot(
            *data["fitted_point"],
            marker="+",
            color="white",
            markersize=13,
            markeredgewidth=2,
            path_effects=[path_effects.Stroke(linewidth=4, foreground="black"), path_effects.Normal()],
        )
        axis.set(
            title=title,
            xlabel=column_label,
            ylabel=row_label,
            xlim=(data["column_values"][0], data["column_values"][-1]),
            ylim=(data["row_values"][0], data["row_values"][-1]),
        )
    figure.colorbar(mesh, ax=list(parameter_axes), label="Agreement / fitted agreement", shrink=0.78)
    tilt = result.sample_tilt_mrad
    focus_label = f"C₁₀ {result.aberrations['C10_nm']:+.2f} nm"
    tilt_label = f"tilt ({tilt[0]:+.2f}, {tilt[1]:+.2f}) mrad"
    depth_label = f"depth spread {result.depth_spread_nm:.2f} nm"
    if view == "depth":
        summary = f"{focus_label}  ·  {depth_label}"
    elif view == "tilt":
        summary = tilt_label.capitalize()
    else:
        summary = f"{focus_label}  ·  {tilt_label}  ·  {depth_label}"
    caption = f"Conditional slice at {evidence['scan_detector_rotation_deg']:.2f}°; other parameters fixed."
    if view != "tilt":
        caption += " Depth spread is a model parameter."
    title_width = max(32, int(figure.get_figwidth() * 10))
    figure.suptitle(
        fill(summary, title_width, break_long_words=False, break_on_hyphens=False)
        + "\n" + fill(caption, title_width, break_long_words=False, break_on_hyphens=False),
        fontsize=11,
    )
    plt.close(figure)
    return figure
