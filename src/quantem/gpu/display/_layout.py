"""Display geometry shared by static detector and SSB views."""

import math
import numbers

PANEL_WIDTHS = {"small": 300, "medium": 450, "large": 600, "xlarge": 900}


def panel_inches(size: str | int) -> float:
    """Translate a preset or pixel width at the static 100-DPI reference."""
    if isinstance(size, str) and size in PANEL_WIDTHS:
        return PANEL_WIDTHS[size] / 100
    if isinstance(size, numbers.Integral) and not isinstance(size, bool) and size > 0:
        return int(size) / 100
    raise ValueError(
        "size must be 'small', 'medium', 'large', 'xlarge', or a positive integer "
        f"panel width in pixels; use size=900, got {size!r}."
    )


def resolve_columns(
    count: int, *, columns: int | None = None, rows: int | None = None,
    default: int = 2,
) -> int:
    """Resolve a panel grid from ``columns`` or ``rows``."""
    for name, value in (("columns", columns), ("rows", rows)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer; use {name}=2, got {value!r}.")
    if rows is not None and columns is not None and rows * columns < count:
        raise ValueError(
            f"rows={rows} and columns={columns} cannot hold {count} panels; "
            "increase the layout or omit rows to calculate it automatically."
        )
    if columns is None:
        columns = max(1, math.ceil(count / rows)) if rows is not None else default
    return min(int(columns), count)


def panel_figure(rows: int, columns: int, width: float, height: float, *, title_lines: int,
                 top_in: float = 0.0, bottom_in: float = 0.3):
    """Matplotlib grid whose gaps are fixed in inches, independent of panel size.

    ``hspace`` and ``wspace`` are fractions of the panel, so a 13-inch panel
    left about 5 inches under a three-line title. Each row here reserves
    ``0.22 * title_lines + 0.2`` inches for its titles and each column gap
    0.9 inches for a colorbar (the last column's too), whatever ``width`` and ``height`` are.
    ``top_in`` holds a figure heading and ``bottom_in`` a footnote.
    """
    import matplotlib.pyplot as plt

    title_in = .22 * title_lines + .2
    gap_in = .9
    figure_width = columns * width + columns * gap_in + .15
    figure_height = rows * (height + title_in) + top_in + bottom_in
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(figure_width, figure_height))
    figure.subplots_adjust(
        left=.15 / figure_width, right=1 - gap_in / figure_width,
        top=1 - (top_in + title_in) / figure_height, bottom=bottom_in / figure_height,
        hspace=title_in / height, wspace=gap_in / width,
    )
    return figure, axes


def scalebar(sampling: float, units: str, rows: int, width: float) -> dict:
    """``show_2d`` scale bar that stays legible at every panel size.

    ``show_2d`` draws the bar ``width_px`` image pixels thick, so the default of
    one pixel is a hairline on a 1536-pixel phase image. Thickness is 0.8 % of
    the image rows (the same fraction of the panel at any zoom) and the label
    is ``1.6 * width`` points for a panel ``width`` inches wide, 10 to 28 pt.
    """
    return {"sampling": sampling, "units": units, "width_px": max(1, math.ceil(.008 * rows)),
            "fontsize": min(28.0, max(10.0, 1.6 * width))}


def zoom_bounds(shape: tuple[int, int], zoom: float) -> tuple[int, int, int, int]:
    """Central display crop; magnification changes no scientific arrays."""
    if isinstance(zoom, bool) or not isinstance(zoom, numbers.Real) or not math.isfinite(zoom) or zoom < 1:
        raise ValueError(f"zoom must be a finite number at least 1; use zoom=3, got {zoom!r}.")
    rows, cols = shape
    height, width = max(1, round(rows / zoom)), max(1, round(cols / zoom))
    row, col = (rows - height) // 2, (cols - width) // 2
    return row, row + height, col, col + width
