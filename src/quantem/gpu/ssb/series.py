"""SSB reconstruction of an acquisition series (``SSB.reconstruct_series``) and the Live layout it reads and writes.

A series directory holds raw ``*_master.h5`` acquisitions and, under ``live/screen/<dataset>`` (or the ``save_dir``
a caller names, such as Live's ``<output>/live/screen``), the products QuantEM Live already made: ``config.json`` with the fitted SSB settings, bright- and dark-field images, and saved SSB results.
These functions find the acquisitions in natural order, resolve each acquisition's physical settings (Live result,
then ``dataset.yaml``, then the caller), and keep fresh fits discoverable in the same layout, so a series run reuses
Live's work and Live reuses the series run's work.
"""

import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from quantem.gpu import screening
from quantem.gpu.formats import dataset_yaml
from quantem.gpu.ssb.contract import RefineMethod
from quantem.gpu.ssb.persistence import FIT_SCHEMA, RESULT_SCHEMA, SCREEN_ACQUISITION_SCHEMA
from quantem.gpu.ssb.results import (
    SSBResult,
    SSBSeriesResult,
    host_array,
    retired_estimator_note,
)
from quantem.gpu.ssb.trials import resolve_trial_count
from quantem.gpu.ssb.units import sampling_pair, validate_aberrations


def reconstruct_series(
    cls,
    source_directory: str | Path,
    *,
    first_frame: int,
    last_frame: int,
    probe_reference_frame: int | None = None,
    voltage_kV: float | None = None,
    semiangle_mrad: float | None = None,
    scan_sampling_A: float | tuple[float, float] | None = None,
    scan_detector_rotation_deg: float | None = None,
    scan_detector_rotation_flipped: bool | None = None,
    trials: int | None = None,
    refinement: RefineMethod = "nelder-mead",
    backend: Literal["auto", "cuda", "mps"] = "auto",
    save_dir: str | Path | None = None,
    progress: bool = True,
    num_trials: int | None = None,
) -> SSBSeriesResult:
    """Reconstruct an SSB series with independent or fixed-probe fitting (``SSB.reconstruct_series``; ``cls`` is ``SSB``).

    Existing QuantEM Live products under ``save_dir`` (default
    ``<source>/live/screen``) are discovered and reused automatically. Missing
    products run through :class:`SSB` and are saved to the same location. When
    no Live screening exists, physical settings are read from ``dataset.yaml``
    or from explicit keyword arguments; an acquisition with no recorded
    rotation has it estimated by ``SSB.open``. A stored fit counts only when it carries
    the current fit schema (``stored_fit``); any other is treated as absent, so
    a probe reference stored by an earlier version is fitted again.

    Parameters
    ----------
    source_directory : str or Path
        Directory containing raw ``*_master.h5`` acquisitions.
    first_frame : int
        Identifier of the first acquisition to process. The identifier is
        the trailing integer in the dataset name.
    last_frame : int
        Identifier of the last acquisition to process, inclusive. Every
        acquisition between the two endpoints in natural acquisition
        order is included, even when identifiers are not consecutive.
    probe_reference_frame : int, optional
        Acquisition identifier whose fitted aberrations and rotation define
        the fixed probe. When omitted, fit each acquisition independently.
    voltage_kV, semiangle_mrad, scan_detector_rotation_deg : float, optional
        Physical microscope settings; each wins over prior Live metadata and
        ``dataset.yaml``. A rotation none of them records is estimated per
        acquisition (``SSB.open``).
    scan_detector_rotation_flipped : bool, optional
        Whether the detector row and column axes are exchanged with the
        rotation. None takes the saved fit's flag, else unflipped for a
        recorded rotation; an estimated rotation brings its own.
    scan_sampling_A : float or tuple[float, float], optional
        Scan step in Angstrom, ``[row, col]`` or one number for both axes, used on the same terms.
    trials : int, optional
        Search trials per rotation branch of each missing acquisition. Default 200.
    num_trials : int, optional
        Compatibility alias for ``trials``. Supply only one of them.
    refinement : {"nelder-mead", None}, default "nelder-mead"
        Independent-fit refinement method.
    backend : {"auto", "cuda", "mps"}, default "auto"
        Accelerated SSB backend. CPU is never used.
    save_dir : str or Path, optional
        Directory holding one folder per acquisition (``<dataset>/config.json``
        and its saved fits), the layout ``live screen --output OUT`` writes as
        ``OUT/live/screen``. Default ``<source_directory>/live/screen``.
    progress : bool, default True
        Show one acquisition-level progress bar.

    Returns
    -------
    SSBSeriesResult
        One independent-fit or fixed-probe phase stack, with its bright- and
        dark-field companions and a ``metadata()`` table.
    """
    num_trials = resolve_trial_count(trials, num_trials)
    source_root = Path(source_directory).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"SSB source directory not found: {source_root}")
    fixed_probe = probe_reference_frame is not None
    results_root = (source_root / "live" / "screen" if save_dir is None
                    else Path(save_dir).expanduser().resolve())
    selected_paths, paths_by_frame = select_acquisitions(
        source_root, int(first_frame), int(last_frame), results_root,
    )
    frame_numbers = tuple(frame_number(path) for path in selected_paths)
    document = read_dataset_yaml(source_root)
    physical = {"voltage_kV": voltage_kV, "semiangle_mrad": semiangle_mrad,
                "scan_sampling_A": None if scan_sampling_A is None else sampling_pair("scan_sampling_A", scan_sampling_A),
                "scan_detector_rotation_deg": scan_detector_rotation_deg,
                "scan_detector_rotation_flipped": scan_detector_rotation_flipped}

    reference_dataset = None
    if fixed_probe:
        reference_frame = int(probe_reference_frame)
        if reference_frame not in paths_by_frame:
            raise IndexError(
                f"Probe reference frame {reference_frame} was not found in "
                f"{source_root}."
            )
        reference_path = paths_by_frame[reference_frame]
        reference_dataset = reference_path.name
        reference_fit = saved_fit(reference_path)
        if reference_fit is None:
            if progress:
                print(
                    "No matching probe fit found for "
                    f"frame {reference_frame}; fitting it now.",
                    flush=True,
                )
            settings = acquisition_settings(reference_path, document, **physical)
            source = source_root / f"{reference_dataset}_master.h5"
            if not source.is_file():
                raise FileNotFoundError(
                    f"Raw probe-reference acquisition not found: {source}"
                )
            with cls.open(str(source), backend=backend, **open_settings(settings)) as ssb:
                reference_result = ssb.find_aberrations(
                    search="primary", num_trials=num_trials,
                    refinement=refinement,
                    save_dir=reference_path / "ssb-fit",
                    verbose=False,
                )
                detector_sampling_mrad = ssb.detector_sampling_mrad
            write_fit_metadata(reference_path, settings, reference_result, detector_sampling_mrad)
            reference_fit = {
                "aberrations": reference_result.aberrations,
                "scan_detector_rotation_deg": reference_result.scan_detector_rotation_deg,
                "scan_detector_rotation_flipped": reference_result.scan_detector_rotation_flipped,
            }
        reference_aberrations = dict(reference_fit["aberrations"])
        reference_rotation = float(reference_fit["scan_detector_rotation_deg"])
        reference_flipped = bool(reference_fit.get("scan_detector_rotation_flipped", False))

    iterator = selected_paths
    if progress:
        from tqdm.auto import tqdm

        print(
            "Checking exact saved SSB results and reconstructing any "
            "missing acquisitions.",
            flush=True,
        )
        iterator = tqdm(iterator, desc="SSB series", unit="frame")
    phases = []
    bright_fields = []
    dark_fields = []
    records = []
    for frame, screen_path in zip(frame_numbers, iterator, strict=True):
        settings = acquisition_settings(screen_path, document, **physical)
        source = source_root / f"{screen_path.name}_master.h5"
        if not source.is_file():
            raise FileNotFoundError(f"Raw SSB acquisition not found: {source}")
        if fixed_probe:
            settings = {**settings, "scan_detector_rotation_deg": reference_rotation,
                        "scan_detector_rotation_flipped": reference_flipped}
        with cls.open(str(source), backend=backend, **open_settings(settings)) as ssb:
            if fixed_probe:
                result = ssb.run(
                    aberrations=reference_aberrations,
                    save_dir=screen_path / "ssb-locked",
                    verbose=False,
                )
            else:
                result = ssb.find_aberrations(
                    search="primary", num_trials=num_trials,
                    refinement=refinement,
                    save_dir=screen_path / "ssb-fit",
                    verbose=False,
                )
            detector_sampling_mrad = ssb.detector_sampling_mrad
        if not fixed_probe:
            write_fit_metadata(screen_path, settings, result, detector_sampling_mrad)
        # one float32 stack whatever array type the backend's phase is
        phases.append(np.asarray(host_array(result.phase), dtype=np.float32))
        bright_field, dark_field = virtual_images(
            screen_path,
            source,
            backend=backend,
            # the rotation the reconstruction used, estimated or recorded, with its detector axes
            scan_detector_rotation_deg=float(result.scan_detector_rotation_deg),
            scan_detector_rotation_flipped=bool(result.scan_detector_rotation_flipped),
        )
        bright_fields.append(bright_field)
        dark_fields.append(dark_field)
        if fixed_probe:
            aberrations = dict(reference_aberrations)
        else:
            fitted = saved_fit(screen_path)
            aberrations = dict(fitted["aberrations"] if fitted is not None else result.aberrations)
        records.append(
            {
                "frame": frame,
                "dataset": screen_path.name,
                "C10 (nm)": float(aberrations["C10_nm"]),
                "C12 (nm)": float(aberrations["C12_nm"]),
                "phi12 (deg)": float(aberrations["phi12_deg"]),
                "loss": None if result.loss is None else float(result.loss),
                "result": "saved" if result.reused else "computed",
            }
        )
    return SSBSeriesResult(
        phase=np.stack(phases),
        bright_field=np.stack(bright_fields),
        dark_field=np.stack(dark_fields),
        frames=frame_numbers,
        datasets=tuple(path.name for path in selected_paths),
        master_names=tuple(f"{path.name}_master.h5" for path in selected_paths),
        probe_reference_frame=int(probe_reference_frame) if fixed_probe else None,
        probe_reference_dataset=reference_dataset,
        records=tuple(records),
        source_directory=source_root,
        results_directory=results_root,
        requested_backend=backend,
        num_trials=num_trials,
        refinement=refinement,
    )


def select_acquisitions(
    source_root: Path,
    first_frame: int,
    last_frame: int,
    results_root: Path,
) -> tuple[tuple[Path, ...], dict[int, Path]]:
    """Return the screening paths (under ``results_root``) from ``first_frame`` to ``last_frame``, and every path by frame.

    Frame numbers are the trailing integers of the dataset names; the selection runs in natural acquisition order and
    includes both ends, even when the numbers between them are not consecutive.
    """
    if first_frame > last_frame:
        raise ValueError(
            f"first_frame must be at most last_frame; got "
            f"{first_frame} > {last_frame}."
        )
    paths = screen_paths(source_root, results_root)
    by_frame = paths_by_frame(paths)
    if first_frame not in by_frame:
        raise IndexError(
            f"First acquisition frame {first_frame} was not found in "
            f"{source_root}."
        )
    if last_frame not in by_frame:
        raise IndexError(
            f"Last acquisition frame {last_frame} was not found in "
            f"{source_root}."
        )
    first_index = paths.index(by_frame[first_frame])
    last_index = paths.index(by_frame[last_frame])
    if first_index > last_index:
        raise ValueError(
            "first_frame must precede last_frame in acquisition order; "
            f"got {first_frame} after {last_frame}."
        )
    return paths[first_index : last_index + 1], by_frame


def screen_paths(source_root: Path, results_root: Path) -> tuple[Path, ...]:
    """One screening path under ``results_root`` per raw or screened acquisition, in natural order.

    ``results_root`` is the directory of per-acquisition folders (``<source>/live/screen`` by default, or Live's
    ``<output>/live/screen``); it is created only when it is used to store a result.
    """
    masters = tuple(sorted(source_root.glob("*_master.h5"), key=acquisition_order))
    datasets = {master.name.removesuffix("_master.h5") for master in masters}
    if results_root.is_dir():
        datasets.update(
            path.name
            for path in results_root.iterdir()
            if path.is_dir() and (path / "config.json").is_file()
        )
    if not datasets:
        raise FileNotFoundError(
            f"No prior QuantEM screening results or *_master.h5 acquisitions "
            f"were found in {source_root}."
        )
    return tuple(
        results_root / dataset
        for dataset in sorted(datasets, key=lambda name: acquisition_order(Path(name)))
    )


def acquisition_order(path: Path) -> tuple[object, ...]:
    """Natural sort key of one dataset path, so ``scan_10`` follows ``scan_9``."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def paths_by_frame(paths: tuple[Path, ...]) -> dict[int, Path]:
    """Map acquisition frame numbers to paths, rejecting two datasets with the same number."""
    mapped: dict[int, Path] = {}
    for path in paths:
        frame = frame_number(path)
        if frame in mapped:
            raise ValueError(
                f"Acquisition frame {frame} is ambiguous between "
                f"{mapped[frame].name!r} and {path.name!r}."
            )
        mapped[frame] = path
    return mapped


def frame_number(path: Path) -> int:
    """Return the trailing acquisition number of one dataset name."""
    match = re.search(r"(\d+)$", path.name)
    if match is None:
        raise ValueError(
            f"Cannot identify an acquisition frame from {path.name!r}; "
            "dataset names must end with a frame number."
        )
    return int(match.group(1))


def read_dataset_yaml(source_root: Path) -> dict[str, object]:
    """Read the optional session metadata used when no Live result exists yet; empty when absent.

    A file of another schema than ``quantem.dataset/2`` raises instead of supplying no rotation and no step.
    """
    path = source_root / "dataset.yaml"
    if not path.is_file():
        return {}
    return dataset_yaml.checked_dataset(yaml.safe_load(path.read_text(encoding="utf-8")), path)


def acquisition_settings(
    screen_path: Path,
    document: dict[str, object],
    *,
    voltage_kV: float | None,
    semiangle_mrad: float | None,
    scan_sampling_A: tuple[float, float] | None,
    scan_detector_rotation_deg: float | None,
    scan_detector_rotation_flipped: bool | None = None,
) -> dict[str, object]:
    """Return the physical settings for one acquisition.

    Each value comes from the caller, else the saved Live SSB fit, else ``dataset.yaml``; a beam or scan step none of
    them gives is an error that names the three places to fix it. A rotation none of them gives stays None, and
    ``SSB.open`` estimates it with its detector axes. The flip is the caller's, else the saved fit's when the
    rotation is the saved one, else unflipped for a recorded rotation.
    """
    config_path = screen_path / "config.json"
    saved = stored_fit(json.loads(config_path.read_text())) if config_path.is_file() else {}
    session = _dataset_yaml_settings(document, screen_path.name)
    values = {
        "voltage_kV": voltage_kV,
        "semiangle_mrad": semiangle_mrad,
        "scan_sampling_A": scan_sampling_A,
        "scan_detector_rotation_deg": scan_detector_rotation_deg,
    }
    for name, value in values.items():
        if value is None:
            values[name] = saved.get(name, session.get(name))
    missing = [name for name, value in values.items() if value is None and name != "scan_detector_rotation_deg"]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Cannot reconstruct {screen_path.name}: missing {joined}. Run "
            "QuantEM Live screening once, add the values to dataset.yaml, or "
            "pass the missing physical parameters to reconstruct_series()."
        )
    values["bright_field_radius_px"] = saved.get("bright_field_radius_px")
    if values["scan_detector_rotation_deg"] is None:
        values["scan_detector_rotation_flipped"] = scan_detector_rotation_flipped    # estimated with the rotation
    elif scan_detector_rotation_flipped is not None:
        values["scan_detector_rotation_flipped"] = bool(scan_detector_rotation_flipped)
    elif scan_detector_rotation_deg is None and saved.get("scan_detector_rotation_deg") is not None:
        values["scan_detector_rotation_flipped"] = bool(saved.get("scan_detector_rotation_flipped", False))
    else:
        values["scan_detector_rotation_flipped"] = False
    return values


def open_settings(settings: dict[str, object]) -> dict[str, object]:
    """``SSB.open`` keywords for one acquisition's resolved settings.

    A saved bright-field radius is passed as it was fitted: rounding it to whole pixels (53.4 to 53) reconstructed with
    a different disk than the stored fit. A rotation of None is estimated by ``SSB.open``.
    """
    rotation = settings["scan_detector_rotation_deg"]
    return {
        "voltage_kV": float(settings["voltage_kV"]),
        "semiangle_mrad": float(settings["semiangle_mrad"]),
        "scan_sampling_A": tuple(settings["scan_sampling_A"]),
        "scan_detector_rotation_deg": None if rotation is None else float(rotation),
        "scan_detector_rotation_flipped": settings["scan_detector_rotation_flipped"],
        "bright_field_radius_px": None if settings["bright_field_radius_px"] is None else float(settings["bright_field_radius_px"]),
    }


# Keys quantem.live's session rotation vote (settle_session_rotation) and branch move (reverse_com) add to a fit block.
LIVE_BRANCH_DECISION = ("column_sign", "session_moved_180", "session_rotation", "rotation_disagrees")


def write_fit_metadata(
    screen_path: Path,
    settings: dict[str, object],
    result: SSBResult,
    detector_sampling_mrad: tuple[float, float],
) -> None:
    """Record a fresh GPU fit in ``config.json`` so Live screening and later series runs find it.

    The block holds every calibration the fit used, including the detector sampling, so an exported phase states the
    numbers that made it rather than ones derived again from another radius. It carries ``"schema": FIT_SCHEMA``,
    without which no reader accepts it. Fields Live added to a block of the
    current schema are kept, except quantem.live's 180-degree decision on the earlier fit (``LIVE_BRANCH_DECISION``),
    which a new fit invalidates; a block of any other schema is replaced whole, so none of its values pass as current.
    ``detector_sampling_mrad`` is the session's after ``SSB.open``, which holds one on every backend; a block
    without it cannot be exported.

    The file is quantem.live's screening record. A record this function creates carries
    ``"schema": SCREEN_ACQUISITION_SCHEMA`` at its top level: without it Live leaves the acquisition out of its
    Screen list. In a record that exists nothing outside ``computed.ssb`` changes, and one with another top-level
    schema, or none, raises ``ValueError`` before anything is written: a fit stored into a record of an earlier
    layout would sit beside values Live no longer reads under their names.
    """
    config_path = screen_path / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        found = config.get("schema") if isinstance(config, dict) else None
        if found != SCREEN_ACQUISITION_SCHEMA:
            raise ValueError(
                f"{config_path} is a screening record with schema {found!r}; this version stores a fit only in a "
                f"record with schema {SCREEN_ACQUISITION_SCHEMA!r}. Nothing was written. Rewrite the record once "
                "with quantem.live's tools/rewrite_screen_config.py."
            )
    else:
        config = {"schema": SCREEN_ACQUISITION_SCHEMA}
    computed = config.setdefault("computed", {})
    # quantem.live's per-session 180-degree decision describes the fit it was made on: kept beside a new rotation it
    # would say this fit was moved, voted or flagged when it was not
    kept = {key: value for key, value in stored_fit(config).items() if key not in LIVE_BRANCH_DECISION}
    computed["ssb"] = {
        **kept,
        **settings,
        "schema": FIT_SCHEMA,
        "aberrations": dict(result.aberrations),
        "scan_detector_rotation_deg": float(result.scan_detector_rotation_deg),
        "scan_detector_rotation_flipped": bool(result.scan_detector_rotation_flipped),
        "loss": None if result.loss is None else float(result.loss),
        "bright_field_radius_px": result.bright_field_radius_px,
        "bright_field_center_px": None if result.bright_field_center_px is None else list(result.bright_field_center_px),
        "detector_sampling_mrad": list(detector_sampling_mrad),
    }
    screen_path.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def stored_fit(config: dict[str, object], block: str = "ssb") -> dict[str, object]:
    """Return one fit block of a screening ``config.json`` when the current engine wrote it; else an empty dict.

    The block is ``computed.ssb`` (or a sibling Live keeps beside it, named by ``block``), and it counts only when its
    ``"schema"`` equals ``FIT_SCHEMA``. A block with another schema or none is the same as no block: nothing in it is
    read, converted or kept. Without this one test, values fitted under an earlier convention (another automatic
    detector sampling, Angstrom stored as nm, an azimuth in radians) reconstruct a different image with no error. QuantEM Live reads its
    stored fits through this function too, so both packages apply the same rule.
    """
    stored = (config.get("computed") or {}).get(block) or {}
    return stored if stored.get("schema") == FIT_SCHEMA else {}


def screen_record_fit(config: dict[str, object]) -> dict[str, object]:
    """The fit a QuantEM Live screening record's result is: its tilt-corrected fit, else ``computed.ssb``.

    A record of Live's default mode holds the tilt-corrected fit in ``computed.ssb`` and it is returned as it is
    (``stored_fit(config)``, the same dict). A ``--ssb-mode both`` record holds the ordinary fit there and the
    tilt-corrected one in ``computed.ssb_tilt``; that block's optics, rotation, sample tilt and depth spread are then
    returned over ``computed.ssb``'s beam, sampling and bright-field calibration, which the two fits share. A failed
    tilt fit, or one of another schema, is no result. Records are read as written.
    """
    plain = stored_fit(config)
    tilt = stored_fit(config, "ssb_tilt")
    if not plain or tilt.get("state") != "complete" or tilt.get("sample_tilt_mrad") is None:
        return plain
    keys = ("aberrations", "scan_detector_rotation_deg", "scan_detector_rotation_flipped", "sample_tilt_mrad",
            "depth_spread_nm", "fit_gain", "loss", "phase_estimator", "objective")
    return {**plain, **{key: tilt[key] for key in keys if key in tilt}}


def saved_fit(screen_path: Path) -> dict[str, object] | None:
    """Return the fitted aberrations and physical rotation from Live or a GPU save; else None.

    A fit of another schema is not an error and is not returned: the caller fits again, as for a folder with no fit.
    """
    config_path = screen_path / "config.json"
    if config_path.is_file():
        settings = stored_fit(json.loads(config_path.read_text()))
        if "aberrations" in settings and "scan_detector_rotation_deg" in settings:
            return {**settings, "aberrations": validate_aberrations(settings["aberrations"])}
    metadata_path = screen_path / "ssb-fit" / "ssb-find_aberrations.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        result = metadata.get("result") or {}
        aberrations = result.get("aberrations")
        rotation = result.get("scan_detector_rotation_deg")
        if metadata.get("schema") == RESULT_SCHEMA and aberrations is not None and rotation is not None:
            return {
                "aberrations": validate_aberrations(aberrations),
                "scan_detector_rotation_deg": float(rotation),
                "scan_detector_rotation_flipped": bool(result.get("scan_detector_rotation_flipped", False)),
            }
    return None


def virtual_images(
    screen_path: Path,
    source: Path,
    *,
    backend: str,
    scan_detector_rotation_deg: float,
    scan_detector_rotation_flipped: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the acquisition's bright- and dark-field images, from Live's screening or a fresh screening run.

    A fresh run saves every screening product beside the SSB result, as Live would.
    """
    bright_field_path = screen_path / "bf.npy"
    dark_field_path = screen_path / "df.npy"
    if bright_field_path.is_file() and dark_field_path.is_file():
        return (
            np.asarray(np.load(bright_field_path), dtype=np.float32),
            np.asarray(np.load(dark_field_path), dtype=np.float32),
        )
    products = screening.prepare(
        source,
        backend=backend,
        scan_detector_rotation_deg=scan_detector_rotation_deg,
        scan_detector_rotation_flipped=scan_detector_rotation_flipped,
        cache=False,
    )
    screen_path.mkdir(parents=True, exist_ok=True)
    arrays = {
        "mean_dp.npy": products.mean_diffraction_pattern,
        "bf.npy": products.bright_field,
        "df.npy": products.dark_field,
        "com_row.npy": products.com_row,
        "com_col.npy": products.com_col,
        "dpc_phase.npy": products.dpc_phase,
    }
    for name, array in arrays.items():
        np.save(screen_path / name, np.asarray(array, dtype=np.float32))
    return (
        np.asarray(products.bright_field, dtype=np.float32),
        np.asarray(products.dark_field, dtype=np.float32),
    )


def _dataset_yaml_settings(
    document: dict[str, object],
    dataset: str,
) -> dict[str, object]:
    """Resolve one dataset's physical SSB settings from a ``quantem.dataset/2`` file (``formats.dataset_yaml``).

    The beam is the session's ``microscope``; the scan step ``[row, col]`` and the scan-to-detector rotation are the
    file entry's own, else the session's, matched and ordered as quantem.live does (``dataset_yaml.file_entry``:
    the entry naming ``<dataset>_master.h5``, else its number or a range such as ``"06-24"``;
    ``dataset_yaml.acquisition_scan_sampling_A``: file entry, its ``mag`` row, session default). Matching only the
    trailing number as an exact key gave a range-keyed or name-keyed file the session default instead of its own
    step. An empty document (no file) gives no values.
    """
    microscope = document.get("microscope")
    microscope = microscope if isinstance(microscope, dict) else {}
    master = f"{dataset}_master.h5"
    entry = dataset_yaml.file_entry(document, master)
    step = dataset_yaml.acquisition_scan_sampling_A(document, master, path="dataset.yaml")
    return {
        "voltage_kV": microscope.get("voltage_kV"),
        "semiangle_mrad": microscope.get("semiangle_mrad"),
        "scan_sampling_A": None if step is None else step[0],
        "scan_detector_rotation_deg": dataset_yaml.scan_detector_rotation_deg(document, entry),
    }


def screen_record_settings(record: str | Path) -> dict[str, object]:
    """The ``SSB.open`` keywords a QuantEM Live screening record holds for its acquisition's stored SSB fit.

    ``record`` is one acquisition's screening folder (``<output>/live/screen/<dataset>``) or its ``config.json``.
    The keywords are everything the fit used: the beam (``voltage_kV``, ``semiangle_mrad``), ``scan_sampling_A`` and
    ``detector_sampling_mrad`` as ``(row, col)`` pairs, the physical ``scan_detector_rotation_deg`` with
    ``scan_detector_rotation_flipped``, the bright-field centre and radius, and the fitted ``aberrations``
    (``C10_nm``, ``C12_nm``, ``phi12_deg``). The fit is the record's result, its tilt-corrected fit when it has one
    (``screen_record_fit``); its sample tilt and depth spread are not ``SSB.open`` keywords and are
    ``screen_record_sample``'s. Opening with all of them reproduces the fit's geometry; opening with
    the rotation alone (``series.open_settings``) let the session derive another detector sampling and disk.

    Raises
    ------
    ValueError
        The record holds no fit of the current schema (``stored_fit``), or a value the fit needs is missing.

    Examples
    --------
    >>> screen_record_settings("OUT/live/screen/scan_018")["scan_detector_rotation_deg"]  # doctest: +SKIP
    159.78
    """
    path = Path(record).expanduser()
    config_path = path if path.name == "config.json" else path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fit = screen_record_fit(config)
    if not fit:
        raise ValueError(f"{config_path} holds no SSB fit of schema {FIT_SCHEMA!r}; screen it again with QuantEM Live.")
    required = ("voltage_kV", "semiangle_mrad", "scan_sampling_A", "detector_sampling_mrad", "aberrations",
                "scan_detector_rotation_deg")
    missing = [name for name in required if fit.get(name) is None]
    if missing:
        raise ValueError(f"{config_path}: the stored SSB fit has no {', '.join(missing)}.")
    center = fit.get("bright_field_center_px")
    radius = fit.get("bright_field_radius_px")
    return {
        "voltage_kV": float(fit["voltage_kV"]),
        "semiangle_mrad": float(fit["semiangle_mrad"]),
        "scan_sampling_A": tuple(float(value) for value in fit["scan_sampling_A"]),
        "detector_sampling_mrad": tuple(float(value) for value in fit["detector_sampling_mrad"]),
        "aberrations": validate_aberrations(fit["aberrations"]),
        "scan_detector_rotation_deg": float(fit["scan_detector_rotation_deg"]),
        "scan_detector_rotation_flipped": bool(fit.get("scan_detector_rotation_flipped", False)),
        "bright_field_center_px": None if center is None else tuple(float(value) for value in center),
        "bright_field_radius_px": None if radius is None else float(radius),
    }


def screen_record_sample(record: str | Path) -> dict[str, object] | None:
    """The sample tilt and depth spread of a screening record's tilt-corrected fit, or None for a tilt-0 fit.

    ``{"sample_tilt_mrad": (row, col), "depth_spread_nm": float}`` (scan frame, mrad and nm), the session attributes
    ``open_screen_record`` sets so the session reconstructs the tilt-corrected fit, not the same optics at tilt 0.
    """
    path = Path(record).expanduser()
    config_path = path if path.name == "config.json" else path / "config.json"
    fit = screen_record_fit(json.loads(config_path.read_text(encoding="utf-8")))
    if fit.get("sample_tilt_mrad") is None or fit.get("depth_spread_nm") is None:
        return None
    return {"sample_tilt_mrad": tuple(float(value) for value in fit["sample_tilt_mrad"]),
            "depth_spread_nm": float(fit["depth_spread_nm"])}


def open_screen_record(
    record: str | Path,
    *,
    source: str | Path | None = None,
    backend: Literal["auto", "cuda", "mps"] = "auto",
    verbose: bool = False,
):
    """Open an ``SSB`` session on a QuantEM Live screened acquisition, at its stored fit (``screen_record_settings``).

    The bridge from a microscope session to the notebook: the screening record names the raw source
    (``source_master``) and every calibration of its fit, so the session opens with the fitted aberrations, the
    physical rotation and its flip, the bright-field disk and the detector and scan sampling, and the sample tilt
    and depth spread of a tilt-corrected fit (the record's result when it has one, ``screen_record_fit``), and
    ``ssb.show()`` opens ShowPtycho there. Pass ``source`` when the raw file has moved.

    Examples
    --------
    >>> from quantem.gpu.ssb.series import open_screen_record  # doctest: +SKIP
    >>> ssb = open_screen_record("OUT/live/screen/scan_018")  # doctest: +SKIP
    >>> ssb.run()  # doctest: +SKIP
    >>> ssb.show(show_fft=True, save_dir="ssb-results")  # doctest: +SKIP
    """
    from quantem.gpu.ssb.workflow import DirectPtycho

    path = Path(record).expanduser()
    config_path = path if path.name == "config.json" else path / "config.json"
    settings = screen_record_settings(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    note = retired_estimator_note(screen_record_fit(config).get("phase_estimator"), str(config_path))
    if note is not None:
        # the record's optics are used as they were fitted; only its stored image belongs to the retired estimator
        print(note)
    if source is None:
        source = config.get("source_master")
        if not source:
            raise ValueError(f"{config_path} names no source_master; pass source=.")
    if not Path(source).exists():
        raise FileNotFoundError(f"Raw source {source} of {config_path} not found; pass source= where it now is.")
    ssb = DirectPtycho.open(str(source), backend=backend, verbose=verbose, **settings)
    sample = screen_record_sample(config_path)
    if sample is not None:
        # the tilt-corrected fit's sample model: without it the session reconstructs its mid-depth C10 at tilt 0
        ssb.sample_tilt_mrad = sample["sample_tilt_mrad"]
        ssb.depth_spread_nm = sample["depth_spread_nm"]
    return ssb
