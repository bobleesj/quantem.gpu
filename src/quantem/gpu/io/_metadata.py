"""Header-only acquisition metadata readers; no resident allocation."""

from __future__ import annotations

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
    from pathlib import Path
    try:
        with h5py.File(str(Path(filepath)), "r") as f:
            key = "entry/instrument/detector/detectorSpecific/pixel_mask"
            if key not in f:
                return None
            return f[key][:]
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
    with h5py.File(filepath, "r") as f:
        def _visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.size > 100:
                return  # skip large arrays (flatfield, pixel_mask, etc.)
            if "data_" in name:
                return  # skip data chunk links
            try:
                val = obj[()]
                if isinstance(val, bytes):
                    val = val.decode()
                elif isinstance(val, np.ndarray) and val.ndim == 0:
                    val = val.item()
                metadata[name] = val
            except (TypeError, ValueError, OSError, UnicodeDecodeError):
                return  # Skip non-scalar/non-readable datasets
        f.visititems(_visit)

        def _copy_attrs(attrs):
            for key, val in attrs.items():
                if isinstance(val, bytes):
                    val = val.decode()
                elif isinstance(val, np.ndarray) and val.ndim == 0:
                    val = val.item()
                metadata.setdefault(key, val)

        _copy_attrs(f.attrs)
        data_group = f.get("entry/data")
        if data_group is not None:
            _copy_attrs(data_group.attrs)

        data_ds = f.get("entry/data/data")
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
                metadata.setdefault("scan_shape", tuple(int(x) for x in data_ds.attrs["scan_shape"]))
            if "det_shape" in data_ds.attrs:
                metadata.setdefault("detector_shape", tuple(int(x) for x in data_ds.attrs["det_shape"]))
            metadata.setdefault("source_dtype", str(data_ds.dtype))
            if data_ds.ndim >= 3:
                metadata.setdefault("n_frames", int(np.prod(data_ds.shape[:-2])))
    _derive_fields(metadata)
    return metadata


def _derive_fields(metadata: dict) -> None:
    """Promote raw h5-path scalars into named fields on the metadata dict.

    Every derived field is set unconditionally - missing sources land as
    ``None`` so the key is always present and code can do ``meta["scan_shape"]``
    without defensive ``.get()`` calls.
    """
    import math

    ntrigger = metadata.get("entry/instrument/detector/detectorSpecific/ntrigger")
    n_frames = int(ntrigger) if ntrigger is not None else metadata.get("n_frames")
    n_frames = int(n_frames) if n_frames is not None else None

    scan_shape = metadata.get("scan_shape")
    if scan_shape is not None:
        scan_shape = tuple(int(x) for x in scan_shape)
    elif n_frames is not None:
        side = math.isqrt(n_frames)
        scan_shape = (side, side) if side * side == n_frames else None

    frame_time = metadata.get("entry/instrument/detector/frame_time")
    dwell_time_us = float(frame_time) * 1e6 if frame_time is not None else None

    y_pix = metadata.get("entry/instrument/detector/detectorSpecific/y_pixels_in_detector")
    x_pix = metadata.get("entry/instrument/detector/detectorSpecific/x_pixels_in_detector")
    detector_shape = metadata.get("detector_shape")
    if detector_shape is not None:
        detector_shape = tuple(int(x) for x in detector_shape)
    elif y_pix is not None and x_pix is not None:
        detector_shape = (int(y_pix), int(x_pix))

    detector_name = metadata.get("entry/instrument/detector/description")
    saturation_raw = metadata.get("entry/instrument/detector/saturation_value")
    saturation = int(saturation_raw) if saturation_raw is not None else None

    metadata["scan_shape"] = scan_shape
    metadata["n_frames"] = n_frames
    metadata["dwell_time_us"] = dwell_time_us
    metadata["detector_shape"] = detector_shape
    metadata["detector_name"] = detector_name
    metadata["saturation"] = saturation


def read_emd_metadata(emd_path) -> dict:
    """Extract scope-side fields from a Velox EMD file.

    Reads the first image's ``Data/Image/<hash>/Metadata`` JSON (Velox
    stores metadata as a uint8 byte vector per frame). Returns a dict
    with whichever of the following keys were found; missing keys are
    omitted so callers can merge via ``dict.update`` without clobbering:

    - ``stem_magnification``    : float, e.g. 5_100_000 for 5.1 Mx
    - ``field_of_view_nm``      : float, FullScanFieldOfView.x in nm
    - ``voltage_kV``            : float, AccelerationVoltage / 1000
    - ``semi_angle_mrad``       : float, probe semiangle when exposed

    Returns ``{}`` on any parse failure so callers can always `.update()`
    the result into an existing config dict without guarding. The EMD
    format version varies across microscope builds, so missing-field
    handling is the common path, not the edge case.
    """
    import json as _json
    from pathlib import Path as _Path
    path = _Path(emd_path)
    if not path.is_file():
        return {}
    try:
        with h5py.File(path, "r") as f:
            if "Data/Image" not in f:
                return {}
            image_group = f["Data/Image"]
            first_hash = next(iter(image_group.keys()), None)
            if first_hash is None:
                return {}
            meta_ds = image_group[first_hash].get("Metadata")
            if meta_ds is None:
                return {}
            # Velox stores metadata as a (nbytes, nframes) uint8 JSON buffer.
            # Frame 0 is sufficient; per-frame blobs are near-identical.
            raw = meta_ds[:, 0] if meta_ds.ndim == 2 else meta_ds[()]
            raw_bytes = bytes(np.asarray(raw).tolist()).rstrip(b"\x00")
            doc = _json.loads(raw_bytes)
    except (OSError, ValueError, KeyError, _json.JSONDecodeError):
        return {}

    out: dict = {}
    optics = doc.get("Optics") or {}
    custom = doc.get("CustomProperties") or {}

    # Velox wraps most scalars as {"type": "double", "value": "5100000"};
    # AccelerationVoltage is historically a bare string. Accept both.
    def _as_float(v):
        if isinstance(v, dict):
            v = v.get("value")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    mag = _as_float(custom.get("StemMagnification"))
    if mag is not None:
        out["stem_magnification"] = mag

    fov = optics.get("FullScanFieldOfView")
    if isinstance(fov, dict):
        fov_x = _as_float(fov.get("x"))
        if fov_x is not None:
            # Velox reports FOV in metres; screener works in nm.
            out["field_of_view_nm"] = fov_x * 1e9

    voltage = _as_float(optics.get("AccelerationVoltage"))
    if voltage is not None:
        out["voltage_kV"] = voltage / 1000.0

    semi = _as_float(optics.get("ConvergenceSemiAngle") or optics.get("SemiConvergenceAngle"))
    if semi is not None:
        # Velox stores the convergence angle in radians.
        out["semi_angle_mrad"] = semi * 1000.0
    return out


def find_emd_sibling(master_path) -> Path | None:
    """Locate a Velox EMD next to an Arina master file.

    Arina writes ``<stem>_master.h5`` alongside data chunk files; when
    the operator also exports the scan to Velox, the EMD usually lands
    in the same folder. Strategy:

    1. Prefer a file named ``<stem>.emd`` (strict match).
    2. Fall back to any ``*.emd`` in the same directory - Dectris
       operators often batch-rename after the fact.

    Returns ``None`` when no EMD sibling is found.
    """
    from pathlib import Path as _Path
    master = _Path(master_path)
    folder = master.parent
    stem = master.stem
    stem = stem.removesuffix("_master")
    candidates = list(folder.glob(f"{stem}.emd")) + list(folder.glob(f"{stem}*.emd"))
    if candidates:
        return candidates[0]
    others = list(folder.glob("*.emd"))
    return others[0] if len(others) == 1 else None
