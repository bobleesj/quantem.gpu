"""Reconstructed focus/depth comparisons around a jointly fitted SSB model."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from quantem.gpu.ssb.results import SSBResult
    from quantem.gpu.ssb.workflow import SSB
    from quantem.widget import Show2D


def show_focus(
    session: "SSB",
    fit: "SSBResult",
    *,
    region: tuple[int, int, int, int] | None = None,
    mode: Literal["interactive", "static"] = "interactive",
    size: Literal["small", "medium", "large"] | None = None,
    focus_offsets_nm: Sequence[float] = (-5.0, 0.0, 5.0),
    depth_spread_nm: float | Sequence[float] | None = None,
    axsize: tuple[float, float] | None = None,
) -> "Show2D | Figure":
    """Reconstruct full-field phase for every pair, then crop the displayed region."""
    if mode not in ("interactive", "static"):
        raise ValueError(f"Use mode='interactive' or 'static' for focus images; got {mode!r}.")
    scales = {None: 1.0, "small": 0.75, "medium": 1.0, "large": 1.5}
    if size not in scales or (axsize is not None and (size is not None or mode != "static")):
        raise ValueError("Choose size='small', 'medium' or 'large', or axsize=(width, height) with mode='static'.")
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
    if mode == "interactive":
        try:
            from quantem.widget import Show2D
        except ImportError as exc:
            raise ImportError("Focus images use Show2D: install quantem.widget, or use mode='static'.") from exc

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
        return Show2D(
            [image for row in images for image in row],
            labels=[label for row in labels for label in row], title=title,
            sampling=sampling, units="Å", ncols=len(offsets),
            panel_width_px=200 if size == "small" else round(300 * scales[size]),
            vmin=lower, vmax=upper, auto_contrast=False,
            link_contrast=True, link_zoom=True, link_pan=True, display_bin=1,
        )

    import matplotlib.pyplot as plt
    from quantem.core.visualization import show_2d

    panel_size = tuple(4 * scales[size] for _ in range(2)) if axsize is None else axsize
    figure, _ = show_2d(
        images, title=[[label.replace(" · ", "\n", 1) for label in row] for row in labels],
        axsize=panel_size, cmap="inferno",
        norm={"interval_type": "manual", "vmin": lower, "vmax": upper},
        scalebar={"sampling": sampling[1], "units": "Å"},
    )
    figure.suptitle(title)
    figure.tight_layout()
    plt.close(figure)
    return figure
