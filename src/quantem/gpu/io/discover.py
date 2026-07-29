"""Discovery of 4D-STEM master files."""
from __future__ import annotations

from .load import discover_masters


def discover(
    folder: str,
    *,
    pattern: str = "*_master.h5",
    recursive: bool = True,
    scan_shape: tuple[int, int] | None = None,
    verbose: bool = True,
) -> list[str]:
    """Find readable candidate masters below a folder.

    Parameters
    ----------
    folder
        Root folder to search.
    pattern
        Filename glob; defaults to Arina master files.
    recursive
        Search child folders when ``True``.
    scan_shape
        Optional ``(scan_row, scan_col)`` frame-count filter.
    verbose
        Print the selected files.

    Returns
    -------
    list[str]
        Sorted absolute paths.
    """
    return discover_masters(
        folder,
        pattern=pattern,
        recursive=recursive,
        scan_shape=scan_shape,
        verbose=verbose,
    )
