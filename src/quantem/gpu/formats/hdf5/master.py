"""Header-only acquisition metadata readers; no resident allocation."""

import json
import math
from pathlib import Path

import h5py
import numpy as np

from quantem.gpu.formats.hdf5.nxem_metadata import _nxem_companion, _read_nxem_metadata
from quantem.gpu.formats.hdf5.stem_metadata import _read_stem_metadata
from quantem.gpu.formats.qem.metadata import HOT_PIXEL_SCHEMA, maped_records

HOT_PIXEL_CORRECTION_ATTRIBUTE = "quantem_hot_pixel_correction"


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
        Dwell per scan position in microseconds, as a saved ``.qem`` copy
        records it: the exposure ``count_time``, else the frame period
        ``frame_time``, of any master.
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

        An exactly paired ``<acquisition>.h5`` containing ``STEM Metadata``
        supplies microscope voltage, camera length and scan information.
        Uncalibrated scan steps are retained as ``scan_sampling_nominal_A``;
        only a source-marked calibrated step becomes ``scan_sampling_A``.
        Missing detector-angle calibration must still be supplied or measured.

        A matching ``<acquisition>_em_metadata.h5`` NXem companion takes
        precedence over the legacy STEM companion. Its explicit units provide
        scan spacing, angular detector spacing, voltage and convergence angle
        when present and positive. Zero placeholders remain unavailable.

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
            if item.ndim >= 3 or item.size > 100:
                return  # skip diffraction data and large calibration arrays
            if "data_" in name:
                return  # skip data chunk links
            try:
                metadata[name] = _plain_value(item[()])
            except (TypeError, ValueError, OSError, UnicodeDecodeError):
                return  # Skip non-scalar/non-readable datasets
        handle.visititems(visit)

        def copy_attributes(attributes):
            for key, value in attributes.items():
                # These describe the HDF5 file's own layout, not the acquisition: the NeXus class of the
                # group, the name of its default dataset, the frame range of each linked data file. No
                # reader uses them and the .qem format lists none; copied, they were saved into the
                # header of every .qem made from the master.
                if key not in ("NX_class", "signal", "data_file_frame_ranges"):
                    metadata.setdefault(key, _plain_value(value))

        copy_attributes(handle.attrs)
        data_group = handle.get("entry/data")
        if data_group is not None:
            copy_attributes(data_group.attrs)

        detector_dataset = handle.get("entry/data/data")
        if detector_dataset is None and data_group is not None:
            for key in sorted(data_group.keys()):
                if key.startswith("data_"):
                    try:
                        detector_dataset = data_group[key]
                    except (OSError, KeyError):
                        detector_dataset = None
                    break
        if detector_dataset is not None:
            if "scan_shape" in detector_dataset.attrs:
                metadata.setdefault("scan_shape", tuple(int(size) for size in detector_dataset.attrs["scan_shape"]))
            if "detector_shape" in detector_dataset.attrs:
                metadata.setdefault("detector_shape", tuple(int(size) for size in detector_dataset.attrs["detector_shape"]))
            metadata.setdefault("source_dtype", str(detector_dataset.dtype))
            if detector_dataset.ndim >= 3:
                metadata.setdefault("n_frames", int(np.prod(detector_dataset.shape[:-2])))
    saved_correction = metadata.get(HOT_PIXEL_CORRECTION_ATTRIBUTE)
    if saved_correction is not None:
        record = json.loads(saved_correction)
        if (not isinstance(record, dict) or record.get("schema") != HOT_PIXEL_SCHEMA
                or record.get("method") not in {"none", "zero", "median"}
                or type(record.get("applied")) is not bool):
            raise ValueError("Unsupported saved detector correction; reopen the original acquisition.")
        metadata[HOT_PIXEL_CORRECTION_ATTRIBUTE] = record
        metadata["hot_pixel_correction"] = record
    recorded_shape = metadata.get("scan_shape")
    _derive_fields(metadata)
    reader = _read_nxem_metadata if _nxem_companion(Path(filepath)) else _read_stem_metadata
    metadata.update(reader(Path(filepath), metadata, recorded_shape))
    # a merged result's records, under the keys every reader gives them
    metadata.update(maped_records(metadata))
    return metadata


def _plain_value(value):
    """Decode bytes and unwrap 0-D arrays so metadata holds plain Python values."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


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

    # The saved copy records exposure before frame period; reading the same rule here keeps
    # the value from changing when an acquisition is saved and reopened. The rule holds for
    # every detector and for the numpy scalars h5py returns: a master of another detector
    # or with a float32 time reported no dwell time at all.
    exposure_s = metadata.get("entry/instrument/detector/count_time")
    if exposure_s is None:
        exposure_s = metadata.get("entry/instrument/detector/frame_time")
    dwell_time_us = float(exposure_s) * 1e6 if exposure_s is not None else None

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
