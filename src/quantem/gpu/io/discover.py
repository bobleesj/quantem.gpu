"""Discovery of 4D-STEM master files."""

from pathlib import Path

import h5py

from quantem.gpu.formats.hdf5.frames import detector_sources


def discover(
    folder: str,
    *,
    pattern: str = "*_master.h5",
    recursive: bool = True,
    scan_shape: tuple[int, int] | None = None,
    verbose: bool = True,
) -> list[str]:
    """Find readable candidate masters below a folder.

    A session folder often mixes scan sizes; ``scan_shape`` keeps only the
    masters whose recorded frame count matches, reading HDF5 headers only.

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

    Raises
    ------
    FileNotFoundError
        If ``folder`` does not exist.
    ValueError
        If no files match the pattern.
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    glob_method = root.rglob if recursive else root.glob
    paths = sorted(str(p) for p in glob_method(pattern))
    if not paths:
        raise ValueError(
            f"No files matching '{pattern}' found in {folder}"
        )
    if scan_shape is not None:
        expected_frames = scan_shape[0] * scan_shape[1]
        filtered = [path for path in paths if _frame_count(path) == expected_frames]
        skipped = len(paths) - len(filtered)
        paths = filtered
        if verbose and skipped > 0:
            print(f"  Filtered: {len(paths)} files matching {scan_shape[0]}x{scan_shape[1]} "
                  f"(skipped {skipped})")
    if verbose:
        # pad indices only as wide as the largest one, so a short list reads [0] not [ 0]
        width = len(str(max(len(paths) - 1, 0)))
        for index, path in enumerate(paths):
            print(f"  [{index:>{width}}] {path.split('/')[-1]}")
        print(f"\nFound {len(paths)} files in {root.name}/")
    return paths


def _frame_count(filepath: str) -> int | None:
    """Total frame count of a master from its headers only; None when unreadable.

    Arina records ``ntrigger * nimages`` frames in the master; other masters
    are counted from the shapes of the detector datasets they link.
    """
    try:
        with h5py.File(filepath, "r") as master:
            specific = "entry/instrument/detector/detectorSpecific/"
            nimages = int(master[specific + "nimages"][()]) if specific + "nimages" in master else 1
            ntrigger = int(master[specific + "ntrigger"][()]) if specific + "ntrigger" in master else None
            if ntrigger is not None:
                return nimages * ntrigger
            sources = detector_sources(master)
        if not sources:
            return None
        total = 0
        for source in sources:
            with h5py.File(source.path, "r") as handle:
                total += handle[source.dataset_path].shape[0]
        return total
    except (OSError, KeyError, ValueError, TypeError):
        return None
