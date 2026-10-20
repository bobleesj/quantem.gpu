"""A session's ``dataset.yaml``: the file quantem.live owns and an operator writes once per session.

quantem.gpu reads three things from it that a master file does not record: the beam (``microscope``), the scan
step of each acquisition and the scan-to-detector rotation an operator measured. Only schema ``quantem.dataset/2``
is read. An older file names the rotation ``screen_rotation_deg`` / ``rotation_deg`` and stores the step as one
number, so reading it with these names would find no rotation and no step without saying so; quantem.live's
``tools/rewrite_dataset_yaml.py`` rewrites it once.
"""

import math
import re

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
        step = _step(row, where, path) if isinstance(row, dict) else None
        if step is not None:
            return step
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


def master_names(master: str) -> set[str]:
    """The names a ``files`` entry's ``master`` may give for one acquisition: the file's own name, and for a ``.qem``
    working copy the ``<stem>_master.h5`` it was converted from.

    Examples
    --------
    >>> sorted(master_names("scan_018.qem"))
    ['scan_018.qem', 'scan_018_master.h5']
    """
    name = str(master).rsplit("/", 1)[-1]
    names = {name}
    if name.endswith(".qem"):
        names.add(name[: -len(".qem")] + "_master.h5")
    return names


def file_entry(document: dict, master: str) -> dict | None:
    """The ``files`` entry of one acquisition, by quantem.live's rules; None when the session lists none.

    First the entry whose ``master`` names this file (``master_names``). Else an entry keyed by the file's trailing
    number, either exactly (``"06"``, ``6``) or as an inclusive range (``"06-24"``), that names no other master:
    sessions reuse numbers across series, so a numbered entry naming another master never matches. The first
    matching key wins.

    Examples
    --------
    >>> doc = {"files": {2: {"master": "scan_02_master.h5", "mag": "a"}, "27-29": {"mag": "b"}}}
    >>> file_entry(doc, "scan_02_master.h5")["mag"], file_entry(doc, "series_28_master.h5")["mag"]
    ('a', 'b')
    >>> file_entry(doc, "other_02_master.h5") is None
    True
    """
    files = document.get("files") or {}
    if not isinstance(files, dict):
        return None
    names = master_names(master)
    for entry in files.values():
        if isinstance(entry, dict) and entry.get("master") in names:
            return entry
    stem = min(names, key=len)
    stem = stem.removesuffix(".qem").removesuffix(".h5").removesuffix("_master")
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return None
    number = int(numbers[-1])
    for key, entry in files.items():
        if not isinstance(entry, dict) or (entry.get("master") and entry.get("master") not in names):
            continue
        text = str(key)
        if "-" in text and text.replace("-", "").isdigit():
            low, high = (int(part) for part in text.split("-", 1))
            if low <= number <= high:
                return entry
        elif text.isdigit() and int(text) == number:
            return entry
    return None


def acquisition_scan_sampling_A(
    document: dict,
    master: str,
    *,
    magnification_MX: float | None = None,
    path: object = "dataset.yaml",
) -> tuple[tuple[float, float], str] | None:
    """The scan step ``(row, col)`` in angstrom of one acquisition and where it came from; None when none is recorded.

    quantem.live's order, shared by every reader of a session: the file entry's own ``scan_sampling_A``
    (``"dataset.yaml file entry"``), then the ``calibrations`` row of its ``mag`` (``"dataset.yaml <mag>"``), then,
    for a file with no calibrated entry but a recorded ``magnification_MX``, the nearest calibrated magnification
    (``mag_5p1`` = 5.1 MX) scaled by 1/magnification at fixed scan pixels (``"scaled from dataset.yaml <mag> to
    <MX> MX"``), then the session's ``calibrations.default`` row (``"dataset.yaml session"``). Every value is a
    ``[row, col]`` pair of positive numbers; any other raises ``ValueError``.

    Examples
    --------
    >>> doc = {"calibrations": {"mag_5p1": {"scan_sampling_A": [0.373, 0.373]}}}
    >>> step, source = acquisition_scan_sampling_A(doc, "scan_04_master.h5", magnification_MX=3.6)
    >>> round(step[0], 4), source
    (0.5284, 'scaled from dataset.yaml mag_5p1 to 3.6 MX')
    """
    calibrations = document.get("calibrations") or {}
    if not isinstance(calibrations, dict):
        raise ValueError(f"{path}: calibrations must be a mapping")
    entry = file_entry(document, master)
    if entry is not None:
        own = _step(entry, "files entry", path)
        if own is not None:
            return own, "dataset.yaml file entry"
        mag = entry.get("mag")
        if mag is not None:
            row = calibrations.get(mag)
            step = _step(row, f"calibrations.{mag}", path) if isinstance(row, dict) else None
            if step is not None:
                return step, f"dataset.yaml {mag}"
    if magnification_MX:
        calibrated = []
        for name, row in calibrations.items():
            found = re.fullmatch(r"mag_(\d+)p(\d+)", str(name))
            step = _step(row, f"calibrations.{name}", path) if found and isinstance(row, dict) else None
            if step is not None:
                calibrated.append((float(f"{found.group(1)}.{found.group(2)}"), step, name))
        if calibrated:
            mag, (row_step, col_step), name = min(calibrated, key=lambda c: abs(c[0] - float(magnification_MX)))
            scale = mag / float(magnification_MX)
            return ((row_step * scale, col_step * scale),
                    f"scaled from dataset.yaml {name} to {float(magnification_MX):g} MX")
    default = calibrations.get("default")
    step = _step(default, "calibrations.default", path) if isinstance(default, dict) else None
    return None if step is None else (step, "dataset.yaml session")


def _step(row: dict, where: str, path: object) -> tuple[float, float] | None:
    step = row.get("scan_sampling_A")
    if step is None:
        return None
    if (not isinstance(step, list) or len(step) != 2
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
                       for value in step)):
        raise ValueError(f"{path}: {where}.scan_sampling_A must be [row, col] in angstrom, got {step!r}")
    return float(step[0]), float(step[1])
