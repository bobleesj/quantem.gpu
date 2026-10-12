"""SSB reconstruction of an acquisition series (``SSB.reconstruct_series``) and the Live layout it reads and writes.

A series directory holds raw ``*_master.h5`` acquisitions and, under ``live/screen/<dataset>``, the products QuantEM
Live already made: ``config.json`` with the fitted SSB settings, bright- and dark-field images, and saved SSB results.
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
)
from quantem.gpu.ssb.units import validate_aberrations


def reconstruct_series(
    cls,
    source_directory: str | Path,
    *,
    first_frame: int,
    last_frame: int,
    probe_reference_frame: int | None = None,
    voltage_kV: float | None = None,
    semiangle_mrad: float | None = None,
    scan_sampling_A: tuple[float, float] | None = None,
    scan_detector_rotation_deg: float | None = None,
    num_trials: int = 200,
    refinement: RefineMethod = "nelder-mead",
    backend: Literal["auto", "cuda", "mps"] = "auto",
    progress: bool = True,
) -> SSBSeriesResult:
    """Reconstruct an SSB series with independent or fixed-probe fitting (``SSB.reconstruct_series``; ``cls`` is ``SSB``).

    Existing QuantEM Live products under ``<source>/live/screen`` are
    discovered and reused automatically. Missing products run through
    :class:`SSB` and are saved to the same standard location. When no Live
    screening exists, physical settings are read from ``dataset.yaml`` or
    from explicit keyword arguments. A stored fit counts only when it carries
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
        Physical microscope settings used only when prior Live metadata or
        ``dataset.yaml`` does not already provide them.
    scan_sampling_A : tuple[float, float], optional
        Scan step as one ``[row, col]`` pair, used on the same terms.
    num_trials : int, default 200
        Independent SSB fit trials per missing acquisition.
    refinement : {"nelder-mead", None}, default "nelder-mead"
        Independent-fit refinement method.
    backend : {"auto", "cuda", "mps"}, default "auto"
        Accelerated SSB backend. CPU is never used.
    progress : bool, default True
        Show one acquisition-level progress bar.

    Returns
    -------
    SSBSeriesResult
        One independent-fit or fixed-probe phase stack, with its bright- and
        dark-field companions and a ``metadata()`` table.
    """
    source_root = Path(source_directory).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"SSB source directory not found: {source_root}")
    fixed_probe = probe_reference_frame is not None
    results_root, selected_paths, paths_by_frame = select_acquisitions(
        source_root, int(first_frame), int(last_frame),
    )
    frame_numbers = tuple(frame_number(path) for path in selected_paths)
    document = read_dataset_yaml(source_root)
    physical = {"voltage_kV": voltage_kV, "semiangle_mrad": semiangle_mrad,
                "scan_sampling_A": scan_sampling_A, "scan_detector_rotation_deg": scan_detector_rotation_deg}

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
            with cls.open(str(source), backend=backend,
                          **open_settings(settings, float(settings["scan_detector_rotation_deg"]))) as ssb:
                reference_result = ssb.find_aberrations(
                    num_trials=num_trials,
                    refinement=refinement,
                    save_to=reference_path / "ssb-fit",
                    verbose=False,
                )
                detector_sampling_mrad = ssb.detector_sampling_mrad
            write_fit_metadata(reference_path, settings, reference_result, detector_sampling_mrad)
            reference_fit = {
                "aberrations": reference_result.aberrations,
                "scan_detector_rotation_deg": reference_result.scan_detector_rotation_deg,
            }
        reference_aberrations = dict(reference_fit["aberrations"])
        reference_rotation = float(reference_fit["scan_detector_rotation_deg"])

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
        rotation = reference_rotation if fixed_probe else float(settings["scan_detector_rotation_deg"])
        with cls.open(str(source), backend=backend, **open_settings(settings, rotation)) as ssb:
            if fixed_probe:
                result = ssb.reconstruct(
                    aberrations=reference_aberrations,
                    save_to=screen_path / "ssb-locked",
                    verbose=False,
                )
            else:
                result = ssb.find_aberrations(
                    num_trials=num_trials,
                    refinement=refinement,
                    save_to=screen_path / "ssb-fit",
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
            scan_detector_rotation_deg=float(settings["scan_detector_rotation_deg"]),
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
) -> tuple[Path, tuple[Path, ...], dict[int, Path]]:
    """Return the screening root, the screening paths from ``first_frame`` to ``last_frame``, and every path by frame.

    Frame numbers are the trailing integers of the dataset names; the selection runs in natural acquisition order and
    includes both ends, even when the numbers between them are not consecutive.
    """
    if first_frame > last_frame:
        raise ValueError(
            f"first_frame must be at most last_frame; got "
            f"{first_frame} > {last_frame}."
        )
    results_root, paths = screen_paths(source_root)
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
    return results_root, paths[first_index : last_index + 1], by_frame


def screen_paths(source_root: Path) -> tuple[Path, tuple[Path, ...]]:
    """Return the screening root and one screening path per raw or screened acquisition, in natural order."""
    results_root = source_root / "live" / "screen"
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
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root, tuple(
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
) -> dict[str, object]:
    """Return complete physical settings for one acquisition.

    Each value comes from the caller, else the saved Live SSB fit, else ``dataset.yaml``; a value none of them gives
    is an error that names the three places to fix it.
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
    missing = [name for name, value in values.items() if value is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Cannot reconstruct {screen_path.name}: missing {joined}. Run "
            "QuantEM Live screening once, add the values to dataset.yaml, or "
            "pass the missing physical parameters to reconstruct_series()."
        )
    values["bright_field_radius_px"] = saved.get("bright_field_radius_px")
    # a stored fit says whether its detector axes were exchanged; SSB.open refuses a True instead of ignoring it
    values["detector_transposed"] = bool(saved.get("detector_transposed", False))
    return values


def open_settings(settings: dict[str, object], scan_detector_rotation_deg: float) -> dict[str, object]:
    """``SSB.open`` keywords for one acquisition's resolved settings (a saved BF radius is rounded to whole pixels)."""
    return {
        "voltage_kV": float(settings["voltage_kV"]),
        "semiangle_mrad": float(settings["semiangle_mrad"]),
        "scan_sampling_A": tuple(settings["scan_sampling_A"]),
        "scan_detector_rotation_deg": scan_detector_rotation_deg,
        "detector_transposed": settings["detector_transposed"],
        "bright_field_radius_px": None if settings["bright_field_radius_px"] is None else round(float(settings["bright_field_radius_px"])),
    }


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
    current schema are kept; a block of any other schema is replaced whole, so none of its values pass as current.
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
    computed["ssb"] = {
        **stored_fit(config),
        **settings,
        "schema": FIT_SCHEMA,
        "aberrations": dict(result.aberrations),
        "scan_detector_rotation_deg": float(result.scan_detector_rotation_deg),
        "detector_transposed": bool(result.detector_transposed),
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
            }
    return None


def virtual_images(
    screen_path: Path,
    source: Path,
    *,
    backend: str,
    scan_detector_rotation_deg: float,
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
    file entry's own, else the session's, in quantem.live's order. An empty document (no file) gives no values.
    """
    microscope = document.get("microscope")
    files = document.get("files")
    microscope = microscope if isinstance(microscope, dict) else {}
    files = files if isinstance(files, dict) else {}
    match = re.search(r"(\d+)$", dataset)
    frame = None if match is None else int(match.group(1))
    entry = files.get(frame, files.get(str(frame)))
    entry = entry if isinstance(entry, dict) else None
    return {
        "voltage_kV": microscope.get("voltage_kV"),
        "semiangle_mrad": microscope.get("semiangle_mrad"),
        "scan_sampling_A": dataset_yaml.scan_sampling_A(document, entry, "dataset.yaml"),
        "scan_detector_rotation_deg": dataset_yaml.scan_detector_rotation_deg(document, entry),
    }
