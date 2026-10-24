"""Reconstructed focus/depth comparisons around a jointly fitted SSB model."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from quantem.gpu.display._layout import panel_inches, resolve_columns

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from quantem.widget import Show2D, Show3D

    from quantem.gpu.ssb.results import SSBResult
    from quantem.gpu.ssb.workflow import DirectPtycho


def show_focus(
    session: "DirectPtycho",
    fit: "SSBResult",
    *,
    region: tuple[int, int, int, int] | None = None,
    mode: Literal["interactive", "static"] = "interactive",
    layout: Literal["grid", "stack"] = "grid",
    size: Literal["small", "medium", "large", "xlarge"] | None = None,
    focus_offsets_nm: Sequence[float] = (-5.0, 0.0, 5.0),
    depth_spread_nm: float | Sequence[float] | None = None,
    axsize: tuple[float, float] | None = None,
    columns: int | None = None,
    rows: int | None = None,
    cmap: str = "inferno",
    save: str | Path | None = None,
    max_cols: int | None = None,
) -> "Show2D | Show3D | Figure":
    """Reconstruct full-field phase for every pair, then crop the displayed region."""
    if mode not in ("interactive", "static"):
        raise ValueError(f"Use mode='interactive' or 'static' for focus images; got {mode!r}.")
    if save is not None and mode != "static":
        raise ValueError("save exports a static figure; use ssb.show('focus', mode='static', save='focus.png').")
    if layout not in ("grid", "stack"):
        raise ValueError(f"Use layout='grid' for Show2D or 'stack' for Show3D; got {layout!r}.")
    width_inches = panel_inches("medium" if size is None else size)
    if axsize is not None and (size is not None or mode != "static"):
        raise ValueError("Choose size='small', 'medium', 'large' or 'xlarge', or axsize=(width, height) with mode='static'.")
    offsets = np.asarray(focus_offsets_nm, dtype=float)
    if offsets.ndim != 1 or not offsets.size or not np.isfinite(offsets).all():
        raise ValueError("Give finite focus_offsets_nm, for example (-5, 0, 5), relative to the joint fit.")
    if fit.sample_tilt_mrad is None or fit.depth_spread_nm is None:
        raise ValueError("Focus/depth images require a joint fit. Run ssb.find_aberrations(tilt=True) first.")
    depths = (
        np.unique(np.maximum(0.0, fit.depth_spread_nm + np.array([-5.0, 0.0, 5.0])))
        if depth_spread_nm is None else np.atleast_1d(np.asarray(depth_spread_nm, dtype=float))
    )
    if depths.ndim != 1 or not depths.size or not np.isfinite(depths).all() or (depths < 0).any():
        raise ValueError("Give finite, nonnegative depth_spread_nm values, for example (5, 10, 15).")
    panel_count = len(depths) if mode == "interactive" and layout == "stack" else len(depths) * len(offsets)
    default_columns = (len(depths) if layout == "stack" else len(offsets)) if mode == "interactive" else 2
    columns = resolve_columns(
        panel_count, columns=columns, rows=rows, default=default_columns, legacy_columns=max_cols,
    )
    if mode == "interactive":
        try:
            from quantem.widget import Show2D, Show3D
        except ImportError as exc:
            raise ImportError("Interactive focus images need quantem.widget; install it, or use mode='static'.") from exc

    factor = session._resolve_upsampling_factor("auto")
    sampling = session._output_sampling(factor)
    crop = (slice(None), slice(None))
    if region is not None:
        row_start, row_stop, col_start, col_stop = region
        crop = (slice(row_start * factor, row_stop * factor), slice(col_start * factor, col_stop * factor))
    images, labels = [], []
    rotation = session.scan_detector_rotation_deg
    try:
        session.set_rotation(fit.scan_detector_rotation_deg)
        for depth in depths:
            row_images, row_labels = [], []
            for offset in offsets:
                aberrations = {**fit.aberrations, "C10_nm": fit.aberrations["C10_nm"] + float(offset)}
                phase, _ = session.preview(
                    aberrations, sample_tilt_mrad=fit.sample_tilt_mrad,
                    depth_spread_nm=float(depth), upsampling_factor=factor,
                    compute_loss=False,
                )
                row_images.append(phase[crop].copy())
                fitted = offset == 0 and np.isclose(depth, fit.depth_spread_nm, rtol=0, atol=1e-12)
                row_labels.append(
                    f"C₁₀ {aberrations['C10_nm']:+.2f} nm · depth {depth:.2f} nm"
                    + (" · joint fit" if fitted else "")
                )
            images.append(row_images)
            labels.append(row_labels)
    finally:
        session.set_rotation(rotation)

    title = f"SSB focus/depth · {factor}× auto · phase (rad)"
    if region is not None:
        title += f" · rows {region[0]}:{region[1]}, columns {region[2]}:{region[3]}"
    lower, upper = fit.phase_limits
    if mode == "interactive":
        if layout == "stack":
            viewer = Show3D(
                [np.stack(row) for row in images],
                panel_titles=[
                    f"Depth {depth:.2f} nm"
                    + (" · fitted depth" if np.isclose(depth, fit.depth_spread_nm, rtol=0, atol=1e-12) else "")
                    for depth in depths
                ],
                labels=[f"C₁₀ {fit.aberrations['C10_nm'] + offset:+.2f} nm" for offset in offsets],
                title=title, dim_label="Defocus", cmap=cmap,
                sampling=sampling, units="Å", columns=columns,
                size=size or "medium", panel_gap_px=0,
                vmin=lower, vmax=upper, auto_contrast=False,
                link_contrast=True, display_bin=1, avg_window=1,
            )
            viewer.slice_idx = int(np.argmin(np.abs(offsets)))
            return viewer
        return Show2D(
            [image for row in images for image in row],
            labels=[label for row in labels for label in row], title=title,
            sampling=sampling, units="Å", columns=columns,
            size=size or "medium", cmap=cmap,
            vmin=lower, vmax=upper, auto_contrast=False,
            link_contrast=True, link_zoom=True, link_pan=True, display_bin=1,
        )

    import matplotlib.pyplot as plt
    from quantem.core.visualization import show_2d

    frames = [image for row in images for image in row]
    captions = [label.replace(" · ", "\n", 1) for row in labels for label in row]
    grid = [frames[index:index + columns] for index in range(0, len(frames), columns)]
    titles = [captions[index:index + columns] for index in range(0, len(frames), columns)]
    panel_size = (width_inches, width_inches) if axsize is None else axsize
    aspect = frames[0].shape[0] / frames[0].shape[1]
    width = min(panel_size[0], panel_size[1] / aspect)
    height = width * aspect
    # Match subplot geometry to the data; leave only the two-line caption band.
    # Set DPI at creation: closing a figure restores its original canvas DPI.
    dpi = max(plt.rcParams["figure.dpi"], np.ceil(1.1 * frames[0].shape[1] / width))
    figure, axes = plt.subplots(
        len(grid), columns, squeeze=False, dpi=dpi,
        figsize=(columns * width, len(grid) * (height + 0.5) + 0.4),
    )
    show_2d(
        grid, title=titles, figax=(figure, axes), cmap=cmap,
        norm={"interval_type": "manual", "vmin": lower, "vmax": upper},
        scalebar={"sampling": sampling[1], "units": "Å"},
    )
    title = title.replace(" · rows", "\nrows")
    if columns == 1:
        title = title.replace(" · phase", "\nphase")
    figure.suptitle(title)
    figure.set_layout_engine("compressed", w_pad=0.02, h_pad=0.02, wspace=0, hspace=0)
    if save is not None:
        figure.savefig(save, dpi="figure", bbox_inches="tight")
    plt.close(figure)
    return figure
