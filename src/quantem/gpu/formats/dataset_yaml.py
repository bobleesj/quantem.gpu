"""A session's ``dataset.yaml``: the file quantem.live owns and an operator writes once per session.

quantem.gpu reads three things from it that a master file does not record: the beam (``microscope``), the scan
step of each acquisition and the scan-to-detector rotation an operator measured. Only schema ``quantem.dataset/2``
is read. An older file names the rotation ``screen_rotation_deg`` / ``rotation_deg`` and stores the step as one
number, so reading it with these names would find no rotation and no step without saying so; quantem.live's
``tools/rewrite_dataset_yaml.py`` rewrites it once.
"""

import math

DATASET_SCHEMA = "quantem.dataset/2"


def checked_dataset(document: object, path: object) -> dict:
    """Return the parsed ``dataset.yaml`` when it carries ``DATASET_SCHEMA``; raise ``ValueError`` naming the rewrite.

    Examples
    --------
    >>> checked_dataset({"schema": "quantem.dataset/2"}, "dataset.yaml")["schema"]
    'quantem.dataset/2'
    """
    found = document.get("schema") if isinstance(document, dict) else None
    if found != DATASET_SCHEMA:
        raise ValueError(
            f"{path}: dataset.yaml schema {found!r} is not read; this reader takes {DATASET_SCHEMA!r}. "
            "Rewrite it once with quantem.live's tools/rewrite_dataset_yaml.py."
        )
    return document


def scan_sampling_A(document: dict, entry: dict | None, path: object) -> tuple[float, float] | None:
    """The scan step of one acquisition in angstrom, ``(row, col)``, or None when the session records none.

    quantem.live's order: the file entry's own ``scan_sampling_A``, else the ``calibrations`` row of the entry's
    ``mag``, else the session's ``calibrations.default``. Each is a ``[row, col]`` pair of positive numbers.

    Examples
    --------
    >>> scan_sampling_A({"calibrations": {"default": {"scan_sampling_A": [0.5, 0.4]}}}, None, "dataset.yaml")
    (0.5, 0.4)
    """
    calibrations = document.get("calibrations") or {}
    entry = entry or {}
    if not isinstance(calibrations, dict):
        raise ValueError(f"{path}: calibrations must be a mapping")
    rows = [("files entry", entry)]
    mag = entry.get("mag")
    if mag is not None:
        if not isinstance(mag, str):
            raise ValueError(f"{path}: a files entry's mag names a calibrations row, got {mag!r}")
        rows.append((f"calibrations.{mag}", calibrations.get(mag) or {}))
    rows.append(("calibrations.default", calibrations.get("default") or {}))
    for where, row in rows:
        step = row.get("scan_sampling_A") if isinstance(row, dict) else None
        if step is None:
            continue
        if (not isinstance(step, list) or len(step) != 2
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
                           for value in step)):
            raise ValueError(f"{path}: {where}.scan_sampling_A must be [row, col] in angstrom, got {step!r}")
        return float(step[0]), float(step[1])
    return None


def scan_detector_rotation_deg(document: dict, entry: dict | None) -> float | None:
    """The scan-to-detector rotation of one acquisition in degrees: the file entry's own, else the session's.

    A per-file ``0.0`` is a value and wins over the session's ``microscope.scan_detector_rotation_deg``.

    Examples
    --------
    >>> scan_detector_rotation_deg({"microscope": {"scan_detector_rotation_deg": 12.0}}, {"scan_detector_rotation_deg": 0.0})
    0.0
    """
    own = (entry or {}).get("scan_detector_rotation_deg")
    if own is not None:
        return float(own)
    session = (document.get("microscope") or {}).get("scan_detector_rotation_deg")
    return None if session is None else float(session)
