"""Header-only acquisition metadata readers; no resident allocation."""

import json
import math
from pathlib import Path

import h5py
import numpy as np


def read_pixel_mask(filepath):
    """Return the Arina pixel_mask array from a master HDF5.

    The Arina detector writes a 2-D `pixel_mask` dataset under
    `entry/instrument/detector/detectorSpecific/` enumerating hardware
    dead pixels (>0 = bad). This is the ONLY sanctioned reader - other
    modules must go through here instead of opening h5py directly, so
    the Arina schema stays in one place.

    Parameters
    ----------
    filepath : str or Path
        Path to an Arina master HDF5 file.

    Returns
    -------
    np.ndarray or None
        Raw (H, W) mask array as stored in the HDF5, or None if the
        file is missing/unreadable or has no `pixel_mask` dataset.
    """
    try:
        with h5py.File(str(Path(filepath)), "r") as handle:
            key = "entry/instrument/detector/detectorSpecific/pixel_mask"
            if key not in handle:
                return None
            return handle[key][:]
    except (OSError, KeyError):
        return None


def get_metadata(filepath: str) -> dict:
    """Read all scalar metadata from an HDF5 master file.

    Returns a flat dict that mixes two layers:

    **Derived, named fields** (always present as keys; value is ``None`` when
    the source field is missing from the file):

    - ``scan_shape`` : tuple[int, int] or None
        Scan grid as ``(height, width)``. Derived from ``ntrigger`` assuming
        a square scan. If ``ntrigger`` is not a perfect square, this is
        ``None`` and the caller must pass ``scan_shape=`` to ``load()``
        explicitly.
    - ``n_frames`` : int or None
        Total frame count (``ntrigger``).
    - ``dwell_time_us`` : float or None
        Per-frame dwell in microseconds (``frame_time * 1e6``).
    - ``detector_shape`` : tuple[int, int] or None
        Detector pixel count as ``(height, width)``.
    - ``detector_name`` : str or None
        Human-readable detector description, e.g. ``"Dectris ARINA Si"``.
    - ``saturation`` : int or None
        ADU ceiling before the detector saturates.

    **Raw HDF5 scalars** (schema-agnostic): every scalar dataset in the file
    keyed by its full HDF5 path, e.g.
    ``metadata["entry/instrument/detector/frame_time"]``. Arrays of more
    than 100 elements are skipped. This is the escape hatch when you need a
    field the derived layer does not cover.

    .. note::

        Scope-side parameters (``voltage_kV``, ``semiangle``,
        ``scan_sampling``, ``camera_length``, ``rotation``) are NOT in the
        h5 master - they must be passed to ``ssb()`` explicitly or loaded
        from a site config. If a field is in this dict, it came from the
        file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 master file.

    Returns
    -------
    dict
        Mixed dict of derived named fields and raw h5-path scalars.

    Examples
    --------
    ```python
    m = get_metadata("scan_master.h5")
    m["scan_shape"]       # (512, 512)
    m["dwell_time_us"]    # 49.8
    m["detector_name"]    # detector model string
    # any raw HDF5 scalar is also available by its full path:
    m["entry/instrument/detector/count_time"]   # 9.95e-05
    ```
    """
    metadata: dict = {}
    with h5py.File(filepath, "r") as handle:
        def visit(name, item):
            if not isinstance(item, h5py.Dataset):
                return
            if item.size > 100:
                return  # skip large arrays (flatfield, pixel_mask, etc.)
            if "data_" in name:
                return  # skip data chunk links
            try:
                metadata[name] = _plain_value(item[()])
            except (TypeError, ValueError, OSError, UnicodeDecodeError):
                return  # Skip non-scalar/non-readable datasets
        handle.visititems(visit)

        def copy_attributes(attributes):
            for key, value in attributes.items():
                metadata.setdefault(key, _plain_value(value))

        copy_attributes(handle.attrs)
        data_group = handle.get("entry/data")
        if data_group is not None:
            copy_attributes(data_group.attrs)

        data_ds = handle.get("entry/data/data")
        if data_ds is None and data_group is not None:
            for key in sorted(data_group.keys()):
                if key.startswith("data_"):
                    try:
                        data_ds = data_group[key]
                    except (OSError, KeyError):
                        data_ds = None
                    break
        if data_ds is not None:
            if "scan_shape" in data_ds.attrs:
                metadata.setdefault("scan_shape", tuple(int(size) for size in data_ds.attrs["scan_shape"]))
            if "det_shape" in data_ds.attrs:
                metadata.setdefault("detector_shape", tuple(int(size) for size in data_ds.attrs["det_shape"]))
            metadata.setdefault("source_dtype", str(data_ds.dtype))
            if data_ds.ndim >= 3:
                metadata.setdefault("n_frames", int(np.prod(data_ds.shape[:-2])))
    _derive_fields(metadata)
    _decode_quantem_records(metadata)
    return metadata


def _plain_value(value):
    """Decode bytes and unwrap 0-D arrays so metadata holds plain Python values."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _decode_quantem_records(metadata: dict) -> None:
    """Expose versioned QuantEM JSON attributes under stable public keys."""
    for attribute, key in (
        ("quantem_maped_merge_v1", "maped_merge"),
        ("quantem_maped_summary_v1", "maped_summary"),
    ):
        value = metadata.get(attribute)
        if isinstance(value, bytes):
            value = value.decode()
        if not isinstance(value, str):
            continue
        try:
            record = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("version") == 1:
            metadata[key] = record


def _derive_fields(metadata: dict) -> None:
    """Promote raw h5-path scalars into named fields on the metadata dict.

    Every derived field is set unconditionally - missing sources land as
    ``None`` so the key is always present and code can do ``meta["scan_shape"]``
    without defensive ``.get()`` calls.
    """
    ntrigger = metadata.get("entry/instrument/detector/detectorSpecific/ntrigger")
    n_frames = int(ntrigger) if ntrigger is not None else metadata.get("n_frames")
    n_frames = int(n_frames) if n_frames is not None else None

    scan_shape = metadata.get("scan_shape")
    if scan_shape is not None:
        scan_shape = tuple(int(size) for size in scan_shape)
    elif n_frames is not None:
        side = math.isqrt(n_frames)
        scan_shape = (side, side) if side * side == n_frames else None

    frame_time = metadata.get("entry/instrument/detector/frame_time")
    dwell_time_us = float(frame_time) * 1e6 if frame_time is not None else None

    detector_rows = metadata.get("entry/instrument/detector/detectorSpecific/y_pixels_in_detector")
    detector_cols = metadata.get("entry/instrument/detector/detectorSpecific/x_pixels_in_detector")
    detector_shape = metadata.get("detector_shape")
    if detector_shape is not None:
        detector_shape = tuple(int(size) for size in detector_shape)
    elif detector_rows is not None and detector_cols is not None:
        detector_shape = (int(detector_rows), int(detector_cols))

    detector_name = metadata.get("entry/instrument/detector/description")
    saturation_raw = metadata.get("entry/instrument/detector/saturation_value")
    saturation = int(saturation_raw) if saturation_raw is not None else None

    metadata["scan_shape"] = scan_shape
    metadata["n_frames"] = n_frames
    metadata["dwell_time_us"] = dwell_time_us
    metadata["detector_shape"] = detector_shape
    metadata["detector_name"] = detector_name
    metadata["saturation"] = saturation
