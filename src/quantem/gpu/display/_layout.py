"""Display geometry shared by static detector and SSB views."""

import math
import numbers
import warnings

PANEL_WIDTHS = {"small": 300, "medium": 450, "large": 600, "xlarge": 900}


def panel_inches(size: str) -> float:
    """Translate a named panel width to the static 100-DPI reference."""
    if not isinstance(size, str) or size not in PANEL_WIDTHS:
        raise ValueError("Use size='small', 'medium', 'large' or 'xlarge'.")
    return PANEL_WIDTHS[size] / 100


def resolve_columns(
    count: int, *, columns: int | None = None, rows: int | None = None,
    default: int = 2, legacy_columns: int | None = None,
) -> int:
    """Resolve a panel grid, retaining the published max_cols alias."""
    if legacy_columns is not None:
        if columns is not None or rows is not None:
            raise ValueError("Use columns or rows, or the legacy max_cols, not both.")
        warnings.warn(
            "max_cols is deprecated; use columns instead. It remains supported during the transition.",
            DeprecationWarning, stacklevel=3,
        )
        columns = legacy_columns
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
