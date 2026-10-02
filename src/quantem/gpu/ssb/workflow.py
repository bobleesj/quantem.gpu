"""Backend-neutral scientist-facing SSB workflow.

This module owns the only public stateful SSB entry point. Backend modules own
device preparation and kernels, but they do not define a second user API.
"""
from __future__ import annotations

import json
import math
import re
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal, Self

import numpy as np

from quantem.gpu.device import resolve
from quantem.gpu.io.integrity import SourceIntegrity
from quantem.gpu.io.models import _release_owned_storage
from quantem.gpu.io.representation import DataRepresentation

from ._persistence import (
    SCHEMA,
    json_value,
    load_result,
    path_signature,
    result_paths,
    save_result,
    software_signature,
)
from .backends.protocol import SSBProtocol
from .results import (COLUMN_SIGN_MIN, SSBResult, SSBSeriesResult, column_sign, draw_column_histogram,
                      physical_rotation_deg, split_rotation)

RefineMethod = Literal["nelder-mead"] | None


def _mps_brightfield_sources(
    source: str,
    calibration: str | None,
) -> tuple[Path, ...]:
    """Return exact BF-column locations in source-authoritative order."""

    source_path = Path(source).expanduser().resolve()
    candidates: list[Path] = []
    calibration_path = (
        Path(calibration).expanduser().resolve()
        if calibration is not None
        else None
    )
    if calibration_path is not None:
        exact_export_calibration = calibration_path.parent / "snapshots" / "cal.json"
        if exact_export_calibration.is_file():
            candidates.append(exact_export_calibration)
    if source_path.is_dir():
        candidates.append(source_path)
    else:
        source_parent = source_path.parent
        if source_parent.name == "source":
            candidates.append(source_parent.parent)
        candidates.append(source_parent)
    if calibration_path is not None:
        candidates.append(calibration_path)

    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _validate_aberrations(
    aberrations: dict[str, float] | None,
) -> dict[str, float]:
    """Return a complete, owned C10/C12/phi12 mapping."""

    if aberrations is None:
        return {"C10": 0.0, "C12": 0.0, "phi12": 0.0}
    required = {"C10", "C12", "phi12"}
    missing = required - aberrations.keys()
    extra = aberrations.keys() - required
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unknown {sorted(extra)}")
        raise ValueError(
            "SSB aberrations must contain exactly C10, C12, and phi12 "
            f"(nm, nm, radians); {', '.join(details)}."
        )
    return {name: float(aberrations[name]) for name in ("C10", "C12", "phi12")}


# ---------------------------------------------------------------------------
# Aberration units: the public API is nm, the engines work in Angstrom.
# Every backend evaluates chi = (pi / lambda[A]) alpha^2 (C10 + C12 cos 2(phi - phi12)) with no conversion, so the numbers
# it takes and returns are Angstrom; before 2026-09-24 they were passed through and reported as "nm" (known-defocus check:
# abTEM C10 = -100 A came back as C10 = -100.26 "nm"). The SSB session now converts at this boundary so C10 / C12 and the
# sample thickness are true nm everywhere users see them, while kernels, backend parity tests and fixtures keep Angstrom.
# Records must declare nm explicitly; older records require a fresh fit.
# ---------------------------------------------------------------------------

_ENGINE_PER_NM = 10.0


def _in_notebook() -> bool:
    """True inside a Jupyter kernel, where fit() draws its histograms."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config
ABERRATION_UNIT = "nm"


def _aberrations_to_engine(aberrations: dict[str, float]) -> dict[str, float]:
    """nm -> engine Angstrom for C10 / C12; phi12 (rad) unchanged."""
    return {**aberrations, "C10": float(aberrations["C10"]) * _ENGINE_PER_NM, "C12": float(aberrations["C12"]) * _ENGINE_PER_NM}


def _aberrations_from_engine(aberrations: dict[str, float]) -> dict[str, float]:
    """engine Angstrom -> nm for C10 / C12; phi12 (rad) unchanged."""
    return {**aberrations, "C10": float(aberrations["C10"]) / _ENGINE_PER_NM, "C12": float(aberrations["C12"]) / _ENGINE_PER_NM}


def _search_ranges_to_engine(search_ranges: dict[str, object] | None) -> dict[str, object] | None:
    """Scale the nm keys of an Optuna search-range dict (``C10_nm``, ``C12_nm``: range tuple or fixed value) into Angstrom."""
    if search_ranges is None:
        return None
    scaled = dict(search_ranges)
    for key in ("C10_nm", "C12_nm"):
        if key in scaled and scaled[key] is not None:
            value = scaled[key]
            scaled[key] = tuple(float(v) * _ENGINE_PER_NM for v in value) if isinstance(value, (tuple, list)) else float(value) * _ENGINE_PER_NM
    return scaled


def _result_from_engine(result: SSBResult) -> SSBResult:
    """Convert a result computed by a backend (Angstrom) to the public nm units, in place."""
    result.aberrations = _aberrations_from_engine(result.aberrations)
    if result.optuna_trials:
        converted = []
        for trial in result.optuna_trials:
            params = dict(trial.get("params") or {})
            for key in ("C10_nm", "C12_nm"):
                if key in params:
                    params[key] = float(params[key]) / _ENGINE_PER_NM
            converted.append({**trial, "params": params})
        result.optuna_trials = converted
    return result


def _resolve_backend(
    backend: Literal["auto", "cuda", "mps", "webgpu"],
) -> Literal["cuda", "mps", "webgpu"]:
    """Resolve one accelerated SSB backend without a CPU fallback."""

    requested = str(backend).lower()
    if requested == "webgpu":
        return "webgpu"
    if requested not in {"auto", "cuda", "mps"}:
        raise ValueError(
            f"Unknown SSB backend {backend!r}. Use 'auto', 'cuda', 'mps', or "
            "'webgpu'."
        )
    selected = resolve(requested)
    if selected == "cpu":
        raise RuntimeError(
            "SSB requires CUDA, MPS, or browser WebGPU. CPU is test-only and "
            "is never selected as a scientific fallback."
        )
    return selected


def _bright_field_crop(
    loaded,
    backend: str,
    threshold: float,
    bf_radius: float | None,
    *,
    calibrate_detector: bool = False,
):
    """Decode a BF crop while retaining full-detector CUDA calibration.

    Return counts, cropped disk centre, disk radius, and calibration radius.

    SSB reads nothing outside the bright-field disk, but decoding the whole 4D cube (19 GB for 512^2 x 192^2 uint16) is
    what the GPU loader forbids. The disk is found on the full-detector mean pattern with the backend's own rule
    (pixels above ``threshold`` x max, within the radius around the centroid of pixels above mean + std, or within
    ``bf_radius`` around the weighted centroid), so the crop plus the pinned centre selects exactly the pixels a
    full-detector session would (tests/hardware/cuda/test_ssb_open_encoded.py).
    """
    from quantem.gpu.detector.workflow import mean_dp

    dp = np.asarray(mean_dp(loaded), dtype=np.float64)
    calibration_radius = None
    if backend == "cuda" and calibrate_detector:
        import cupy as cp
        from quantem.gpu.detector.backends.cuda.probe import detect_bf_radius

        # Detector calibration must see the full radial profile. A tight BF
        # crop can remove the disk edge and change the detected pixel size.
        _, calibration_radius = detect_bf_radius(cp.asarray(dp))
    selected = dp > dp.max() * float(threshold)
    if bf_radius is None:
        probe = dp > dp.mean() + dp.std()
        total = int(probe.sum())
        if total == 0:
            raise ValueError("No bright-field disk found in the mean diffraction pattern.")
        rows, cols = np.nonzero(probe)
        center = (float(rows.mean()), float(cols.mean()))
        radius = math.sqrt(total / math.pi)
    else:
        rows, cols = np.nonzero(selected)
        weights = dp[rows, cols]
        center = (float((rows * weights).sum() / weights.sum()), float((cols * weights).sum() / weights.sum()))
        radius = float(bf_radius)
    det_rows, det_cols = dp.shape
    # one pixel of margin beyond the disk so float rounding of the centre never drops an edge pixel
    row0, col0 = max(0, math.floor(center[0] - radius) - 1), max(0, math.floor(center[1] - radius) - 1)
    row1, col1 = min(det_rows, math.ceil(center[0] + radius) + 2), min(det_cols, math.ceil(center[1] + radius) + 2)
    counts = loaded.read(detector_region=(row0, row1, col0, col1))
    if backend == "cuda":
        import cupy as cp

        counts = cp.from_dlpack(counts)
    else:
        counts = counts.cpu().numpy()
    loaded.close()
    return counts, (center[0] - row0, center[1] - col0), radius, calibration_radius


def _mps_data_with_scan_shape(data: object, scan_shape: tuple[int, int] | None):
    """Apply an explicit scan shape to an in-memory MPS array without copying."""

    if scan_shape is None or not isinstance(data, np.ndarray) or data.ndim != 3:
        return data
    rows, cols = (int(scan_shape[0]), int(scan_shape[1]))
    if rows * cols != int(data.shape[0]):
        raise ValueError(
            f"scan_shape={scan_shape} describes {rows * cols} frames, but data "
            f"contains {data.shape[0]}."
        )
    return data.reshape(rows, cols, data.shape[1], data.shape[2])


def _host_phase(result: SSBResult) -> np.ndarray:
    """Return one SSB phase as a float32 host array."""
    if isinstance(result, tuple):
        return result[0]
    phase = result.phase
    get = getattr(phase, "get", None)
    return np.asarray(get() if callable(get) else phase, dtype=np.float32)


def _result_loss(result: SSBResult, fallback: object) -> float | None:
    """Return one saved or newly computed SSB loss."""
    loss = result.loss
    if loss is None:
        loss = fallback
    return None if loss is None else float(loss)


def _series_virtual_images(
    screen_path: Path,
    source: Path,
    *,
    backend: str,
    rotation_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate the bright-field and dark-field series views."""
    bright_field_path = screen_path / "bf.npy"
    dark_field_path = screen_path / "df.npy"
    if bright_field_path.is_file() and dark_field_path.is_file():
        return (
            np.asarray(np.load(bright_field_path), dtype=np.float32),
            np.asarray(np.load(dark_field_path), dtype=np.float32),
        )

    from quantem.gpu.screening import prepare

    products = prepare(
        source,
        backend=backend,
        rotation_angle_deg=rotation_angle_deg,
        cache=False,
    )
    screen_path.mkdir(parents=True, exist_ok=True)
    arrays = {
        "mean_dp.npy": products.mean_dp,
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


def _series_sort_key(path: Path) -> tuple[object, ...]:
    """Return a natural acquisition order for one dataset path."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def _series_paths(source_root: Path) -> tuple[Path, tuple[Path, ...]]:
    """Return every raw or screened acquisition in natural order."""
    results_root = source_root / "live" / "screen"
    masters = tuple(sorted(source_root.glob("*_master.h5"), key=_series_sort_key))
    datasets = {
        master.name.removesuffix("_master.h5")
        for master in masters
    }
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
        for dataset in sorted(datasets, key=lambda name: _series_sort_key(Path(name)))
    )


def _series_frame(path: Path) -> int:
    """Return the trailing acquisition identifier from one dataset name."""
    match = re.search(r"(\d+)$", path.name)
    if match is None:
        raise ValueError(
            f"Cannot identify an acquisition frame from {path.name!r}; "
            "dataset names must end with a frame number."
        )
    return int(match.group(1))


def _series_frame_map(paths: tuple[Path, ...]) -> dict[int, Path]:
    """Map acquisition identifiers to paths and reject ambiguous names."""
    mapped: dict[int, Path] = {}
    for path in paths:
        frame = _series_frame(path)
        if frame in mapped:
            raise ValueError(
                f"Acquisition frame {frame} is ambiguous between "
                f"{mapped[frame].name!r} and {path.name!r}."
            )
        mapped[frame] = path
    return mapped


def _dataset_yaml(source_root: Path) -> dict[str, object]:
    """Read optional session metadata used when no Live result exists yet."""
    path = source_root / "dataset.yaml"
    if not path.is_file():
        return {}
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else {}


def _yaml_series_settings(
    document: dict[str, object],
    dataset: str,
) -> dict[str, object]:
    """Resolve physical SSB settings for one dataset from session metadata."""
    microscope = document.get("microscope")
    files = document.get("files")
    calibrations = document.get("calibrations")
    microscope = microscope if isinstance(microscope, dict) else {}
    files = files if isinstance(files, dict) else {}
    calibrations = calibrations if isinstance(calibrations, dict) else {}
    match = re.search(r"(\d+)$", dataset)
    frame = None if match is None else int(match.group(1))
    file_settings = files.get(frame, files.get(str(frame), {}))
    file_settings = file_settings if isinstance(file_settings, dict) else {}
    calibration = calibrations.get(file_settings.get("mag"), {})
    calibration = calibration if isinstance(calibration, dict) else {}
    return {
        "voltage_kV": microscope.get("voltage_kV"),
        "semiangle_mrad": microscope.get(
            "semiangle_mrad",
            microscope.get("semi_angle_mrad"),
        ),
        "scan_sampling_A": calibration.get("scan_sampling_A"),
        "rotation_angle_deg": file_settings.get(
            "rotation_deg",
            microscope.get("screen_rotation_deg"),
        ),
    }


def _series_settings(
    source_root: Path,
    screen_path: Path,
    document: dict[str, object],
    *,
    voltage_kV: float | None,
    semiangle_mrad: float | None,
    scan_sampling_A: float | tuple[float, float] | None,
    rotation_angle_deg: float | None,
) -> dict[str, object]:
    """Return complete physical settings from Live, session metadata, or input."""
    config_path = screen_path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    computed = config.get("computed") if isinstance(config, dict) else {}
    computed = computed if isinstance(computed, dict) else {}
    saved = computed.get("ssb")
    saved = saved if isinstance(saved, dict) else {}
    session = _yaml_series_settings(document, screen_path.name)
    values = {
        "voltage_kV": voltage_kV,
        "semiangle_mrad": semiangle_mrad,
        "scan_sampling_A": scan_sampling_A,
        "rotation_angle_deg": rotation_angle_deg,
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
    values["bf_radius"] = saved.get("bf_radius")
    if rotation_angle_deg is None and saved.get("rotation_angle_deg") is not None:
        # a saved fit stores the angle below 180 plus com_reversed; the series passes the physical angle to open()
        values["rotation_angle_deg"] = physical_rotation_deg(float(values["rotation_angle_deg"]), bool(saved.get("com_reversed", False)))
    return values


def _write_series_fit_metadata(
    screen_path: Path,
    settings: dict[str, object],
    result: SSBResult,
) -> None:
    """Keep fresh GPU fits discoverable by the regular Live screening layout."""
    config_path = screen_path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    config = config if isinstance(config, dict) else {}
    computed = config.setdefault("computed", {})
    computed["ssb"] = {
        **dict(computed.get("ssb") or {}),
        **settings,
        "aberrations": dict(result.aberrations),
        "aberration_unit": ABERRATION_UNIT,
        "rotation_angle_deg": float(result.rotation_angle_deg),
        "com_reversed": bool(result.com_reversed),
        "loss": None if result.loss is None else float(result.loss),
        "bf_radius": result.bf_radius,
    }
    screen_path.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _screen_fit_settings(screen_path: Path) -> dict[str, object] | None:
    """Return fitted aberrations and the physical rotation (CoM reversal folded in) from Live or a current GPU save."""
    config_path = screen_path / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        settings = (config.get("computed") or {}).get("ssb") or {}
        if "aberrations" in settings and "rotation_angle_deg" in settings:
            if settings.get("aberration_unit") != ABERRATION_UNIT:
                raise ValueError("Saved SSB fit must declare aberration_unit='nm'. Rerun the probe fit to save a current record.")
            rotation = physical_rotation_deg(float(settings["rotation_angle_deg"]), bool(settings.get("com_reversed", False)))
            return {**settings, "aberrations": _validate_aberrations(settings["aberrations"]), "aberration_unit": ABERRATION_UNIT,
                    "rotation_angle_deg": rotation}
    metadata_path = screen_path / "ssb-fit" / "ssb-fit.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        result = metadata.get("result") or {}
        aberrations = result.get("aberrations")
        rotation = result.get("rotation_angle_deg")
        if rotation is not None:
            rotation = physical_rotation_deg(float(rotation), bool(result.get("com_reversed", False)))
        if aberrations is not None and rotation is not None:
            if metadata.get("schema") != SCHEMA:
                raise ValueError(f"Saved SSB fit must use schema {SCHEMA}. Rerun the probe fit to save a current record.")
            return {
                "aberrations": _validate_aberrations(aberrations),
                "aberration_unit": ABERRATION_UNIT,
                "rotation_angle_deg": rotation,
            }
    return None


class SSB:
    """One SSB workflow with identical CUDA and MPS scientific semantics.

    Parameters use explicit public units on every backend. Full automatically
    detected bright-field evidence, exact phase-variance fitting, float32 real
    storage, and complex64 object storage are invariant defaults.

    WebGPU uses the same parameter and result contract through the exported
    browser workflow. It cannot execute inside the Python process; requesting
    it here fails deterministically and points to the canonical CLI boundary.
    """

    def __init__(
        self,
        data: object,
        *,
        backend: Literal["auto", "cuda", "mps", "webgpu"] = "auto",
        voltage_kV: float,
        semiangle_mrad: float,
        scan_sampling_A: float | tuple[float, float],
        scan_shape: tuple[int, int] | None = None,
        det_sampling: float | tuple[float, float] | None = None,
        aberrations: dict[str, float] | None = None,
        rotation_angle_deg: float = 0.0,
        com_reversed: bool = False,
        bf_intensity_threshold: float = 0.0,
        bf_radius: float | None = None,
        source_path: str | None = None,
        bf_center: tuple[float, float] | None = None,
    ) -> None:
        self.backend = _resolve_backend(backend)
        if self.backend == "mps":
            from quantem.gpu.io.backends.mps.packed import MPSCompactV3Resident

            if isinstance(data, MPSCompactV3Resident):
                raise NotImplementedError(
                    "Packed MPS detector sources are not supported by SSB. "
                    "Use a dense detector array or an exact BF-column source."
                )
        if self.backend == "webgpu":
            raise RuntimeError(
                "WebGPU SSB executes in the browser. Use `quantem showptycho "
                "--backend webgpu` so the CLI can export the same SSB plan and "
                "collect the shared result schema."
            )
        self._data = _mps_data_with_scan_shape(data, scan_shape)
        self._scan_shape = scan_shape
        self.voltage_kV = float(voltage_kV)
        self.semiangle_mrad = float(semiangle_mrad)
        self.scan_sampling_A = scan_sampling_A
        self.det_sampling = det_sampling
        self._aberrations_explicit = aberrations is not None
        self.aberrations = _validate_aberrations(aberrations)
        self._fit_start_aberrations = dict(self.aberrations)
        self._fit_start_aberrations_explicit = self._aberrations_explicit
        # public form: angle below 180 plus whether the CoM is reversed; the engines get the physical angle
        self.rotation_angle_deg, self.com_reversed = split_rotation(rotation_angle_deg, com_reversed)
        self.bf_intensity_threshold = float(bf_intensity_threshold)
        self.bf_radius = bf_radius
        # (row, col) detector pixels; set by SSB.open when it decodes only the bright-field crop of an encoded source
        self.bf_center = None if bf_center is None else (float(bf_center[0]), float(bf_center[1]))
        # latest fit(tilt=True): sample tilt (row, col) mrad, scan frame, and depth spread nm; None after a standard fit
        self.tilt_mrad: tuple[float, float] | None = None
        self.depth_spread_nm: float | None = None
        self.source_path = source_path
        self.calibration_path: str | None = None
        self.source_manifest_path: str | None = None
        self.source_storage_path = source_path
        self.source_kind: Literal["array", "detector", "bf_columns"] = "array"
        self.source_dtype = str(data.dtype)
        self.source_bytes = int(data.nbytes)
        self.source_detector_bin = int(getattr(data, "det_bin", 1) or 1)
        self.source_provenance = json_value(
            getattr(data, "source_provenance", None)
        )
        self.source_load_seconds: float | None = None
        self._cuda_session = None
        self._mps_backend = None
        self._reconstruction: SSBResult | None = None
        self.best_loss = float("inf")
        self.trial_history: list[dict[str, object]] = []

    @classmethod
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
        rotation_angle_deg: float | None = None,
        trials: int = 200,
        refinement: RefineMethod = "nelder-mead",
        backend: Literal["auto", "cuda", "mps"] = "auto",
        progress: bool = True,
    ) -> SSBSeriesResult:
        """Reconstruct an SSB series with independent or fixed-probe fitting.

        Existing QuantEM Live products under ``<source>/live/screen`` are
        discovered and reused automatically. Missing products run through
        :class:`SSB` and are saved to the same standard location. When no Live
        screening exists, physical settings are read from ``dataset.yaml`` or
        from explicit keyword arguments.

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
        voltage_kV, semiangle_mrad, scan_sampling_A, rotation_angle_deg : float, optional
            Physical microscope settings used only when prior Live metadata or
            ``dataset.yaml`` does not already provide them.
        trials : int, default 200
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
            One independent-fit or fixed-probe phase stack with display,
            metrics, and metadata methods.
        """
        source_root = Path(source_directory).expanduser().resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"SSB source directory not found: {source_root}")
        fixed_probe = probe_reference_frame is not None
        first_frame = int(first_frame)
        last_frame = int(last_frame)
        if first_frame > last_frame:
            raise ValueError(
                f"first_frame must be at most last_frame; got "
                f"{first_frame} > {last_frame}."
            )
        results_root, screen_paths = _series_paths(source_root)
        paths_by_frame = _series_frame_map(screen_paths)
        if first_frame not in paths_by_frame:
            raise IndexError(
                f"First acquisition frame {first_frame} was not found in "
                f"{source_root}."
            )
        if last_frame not in paths_by_frame:
            raise IndexError(
                f"Last acquisition frame {last_frame} was not found in "
                f"{source_root}."
            )
        first_index = screen_paths.index(paths_by_frame[first_frame])
        last_index = screen_paths.index(paths_by_frame[last_frame])
        if first_index > last_index:
            raise ValueError(
                "first_frame must precede last_frame in acquisition order; "
                f"got {first_frame} after {last_frame}."
            )
        selected_paths = screen_paths[first_index : last_index + 1]
        frame_numbers = tuple(_series_frame(path) for path in selected_paths)
        document = _dataset_yaml(source_root)

        reference_dataset = None
        reference_aberrations = None
        reference_rotation = None
        if fixed_probe:
            reference_frame = int(probe_reference_frame)
            if reference_frame not in paths_by_frame:
                raise IndexError(
                    f"Probe reference frame {reference_frame} was not found in "
                    f"{source_root}."
                )
            reference_path = paths_by_frame[reference_frame]
            reference_dataset = reference_path.name
            reference_fit = _screen_fit_settings(reference_path)
            if reference_fit is None:
                if progress:
                    print(
                        "No matching probe fit found for "
                        f"frame {reference_frame}; fitting it now.",
                        flush=True,
                    )
                settings = _series_settings(
                    source_root,
                    reference_path,
                    document,
                    voltage_kV=voltage_kV,
                    semiangle_mrad=semiangle_mrad,
                    scan_sampling_A=scan_sampling_A,
                    rotation_angle_deg=rotation_angle_deg,
                )
                source = source_root / f"{reference_dataset}_master.h5"
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Raw probe-reference acquisition not found: {source}"
                    )
                with cls.open(
                    str(source),
                    backend=backend,
                    voltage_kV=float(settings["voltage_kV"]),
                    semiangle_mrad=float(settings["semiangle_mrad"]),
                    scan_sampling_A=settings["scan_sampling_A"],
                    rotation_angle_deg=float(settings["rotation_angle_deg"]),
                    bf_radius=(
                        None
                        if settings["bf_radius"] is None
                        else round(float(settings["bf_radius"]))
                    ),
                ) as ssb:
                    reference_result = ssb.fit(
                        trials=trials,
                        refinement=refinement,
                        save_to=reference_path / "ssb-fit",
                        verbose=False,
                    )
                _write_series_fit_metadata(
                    reference_path,
                    settings,
                    reference_result,
                )
                reference_fit = {
                    "aberrations": reference_result.aberrations,
                    "rotation_angle_deg": reference_result.physical_rotation_deg,
                }
            reference_aberrations = dict(reference_fit["aberrations"])
            reference_rotation = float(reference_fit["rotation_angle_deg"])

        iterator = selected_paths
        if progress:
            from tqdm.auto import tqdm

            print(
                "Checking exact saved SSB results and reconstructing any "
                "missing acquisitions.",
                flush=True,
            )
            iterator = tqdm(
                iterator,
                desc="SSB series",
                unit="frame",
            )

        phases = []
        bright_fields = []
        dark_fields = []
        records = []
        for frame, screen_path in zip(
            frame_numbers,
            iterator,
            strict=True,
        ):
            settings = _series_settings(
                source_root,
                screen_path,
                document,
                voltage_kV=voltage_kV,
                semiangle_mrad=semiangle_mrad,
                scan_sampling_A=scan_sampling_A,
                rotation_angle_deg=rotation_angle_deg,
            )
            source = source_root / f"{screen_path.name}_master.h5"
            if not source.is_file():
                raise FileNotFoundError(f"Raw SSB acquisition not found: {source}")
            with cls.open(
                str(source),
                backend=backend,
                voltage_kV=float(settings["voltage_kV"]),
                semiangle_mrad=float(settings["semiangle_mrad"]),
                scan_sampling_A=settings["scan_sampling_A"],
                rotation_angle_deg=(
                    reference_rotation
                    if fixed_probe
                    else float(settings["rotation_angle_deg"])
                ),
                bf_radius=(
                    None
                    if settings["bf_radius"] is None
                    else round(float(settings["bf_radius"]))
                ),
            ) as ssb:
                if fixed_probe:
                    result = ssb.reconstruct(
                        reference_aberrations,
                        save_to=screen_path / "ssb-locked",
                        verbose=False,
                    )
                else:
                    result = ssb.fit(
                        trials=trials,
                        refinement=refinement,
                        save_to=screen_path / "ssb-fit",
                        verbose=False,
                    )
            if not fixed_probe:
                _write_series_fit_metadata(screen_path, settings, result)
            state = "saved" if result.reused else "computed"

            phases.append(_host_phase(result))
            bright_field, dark_field = _series_virtual_images(
                screen_path,
                source,
                backend=backend,
                rotation_angle_deg=float(settings["rotation_angle_deg"]),
            )
            bright_fields.append(bright_field)
            dark_fields.append(dark_field)
            if fixed_probe:
                aberrations = dict(reference_aberrations)
            else:
                fitted = _screen_fit_settings(screen_path)
                aberrations = dict(
                    fitted["aberrations"] if fitted is not None else result.aberrations
                )
            records.append(
                {
                    "frame": frame,
                    "dataset": screen_path.name,
                    "C10 (nm)": float(aberrations["C10"]),
                    "C12 (nm)": float(aberrations["C12"]),
                    "phi12 (rad)": float(aberrations["phi12"]),
                    "loss": _result_loss(result, settings.get("loss")),
                    "result": state,
                }
            )

        return SSBSeriesResult(
            phase=np.stack(phases).astype(
                np.float32,
                copy=False,
            ),
            bright_field=np.stack(bright_fields).astype(np.float32, copy=False),
            dark_field=np.stack(dark_fields).astype(np.float32, copy=False),
            frames=frame_numbers,
            datasets=tuple(path.name for path in selected_paths),
            master_names=tuple(
                f"{path.name}_master.h5" for path in selected_paths
            ),
            probe_reference_frame=(
                int(probe_reference_frame) if fixed_probe else None
            ),
            probe_reference_dataset=reference_dataset,
            records=tuple(records),
            source_directory=source_root,
            results_directory=results_root,
            requested_backend=backend,
            trials=trials,
            refinement=refinement,
        )

    @classmethod
    def open(
        cls,
        source: str,
        *,
        backend: Literal["auto", "cuda", "mps", "webgpu"] = "auto",
        dtype: str | None = None,
        voltage_kV: float,
        semiangle_mrad: float,
        scan_sampling_A: float | tuple[float, float],
        scan_shape: tuple[int, int] | None = None,
        det_sampling: float | tuple[float, float] | None = None,
        aberrations: dict[str, float] | None = None,
        rotation_angle_deg: float = 0.0,
        com_reversed: bool = False,
        bf_intensity_threshold: float = 0.0,
        bf_radius: int | None = None,
        calibration: str | None = None,
        expected_source_sha256: str | None = None,
        source_integrity: SourceIntegrity | None = None,
        verbose: bool = False,
    ) -> SSB:
        """Open one lossless 4D-STEM source and prepare an SSB session.

        Exact BF-column storage is chosen automatically when it is available;
        otherwise detector counts are loaded with the explicitly requested
        storage dtype. Leave ``dtype=None`` for native detector precision.
        This storage choice never changes the float32/complex64 optimization
        precision or scientific objective.

        ``expected_source_sha256`` and ``source_integrity`` have the same
        meanings as in ``io.load``. GPU acquisitions use ANS-encoded storage.
        Older prepared-packed files must be re-exported from their original
        acquisitions as .qem files before opening them here.
        """

        selected = _resolve_backend(backend)
        if DataRepresentation.detect_source(source) is DataRepresentation.PACKED:
            raise NotImplementedError(
                "Packed acquisition files cannot be opened as ANS residents. "
                "Reopen the original acquisition and save a new .qem copy."
            )
        if selected == "webgpu":
            raise RuntimeError(
                "Browser WebGPU sources are opened by the exported SSB runtime."
            )
        data = None
        bf_center = None
        source_kind: Literal["detector", "bf_columns"]
        source_dtype: str
        source_bytes: int
        source_load_seconds: float
        source_storage_path: str
        if selected == "mps" and expected_source_sha256 is None and source_integrity is None:
            from .backends.mps.engine import (
                _BfColumnCompanionNotDeclared,
                load_bf_columns_mps,
            )

            for candidate in _mps_brightfield_sources(source, calibration):
                try:
                    frames = load_bf_columns_mps(candidate, verbose=verbose)
                except _BfColumnCompanionNotDeclared:
                    continue
                data = frames
                source_kind = "bf_columns"
                source_storage_path = str(frames.source_path)
                source_dtype = str(frames.dtype)
                source_bytes = int(frames.nbytes)
                source_load_seconds = float(frames.load_seconds)
                break
        if data is None:
            from quantem.gpu.io import load

            load_started = time.perf_counter()
            loaded = load(
                source,
                backend=selected,
                detector_bin=1,
                dtype=dtype,
                verbose=verbose,
                expected_source_sha256=expected_source_sha256,
                source_integrity=source_integrity,
            )
            data, bf_center, bf_radius, calibration_radius = _bright_field_crop(
                loaded, selected, bf_intensity_threshold, bf_radius,
                calibrate_detector=det_sampling is None,
            )
            if det_sampling is None and calibration_radius is not None:
                det_sampling = (2.0 * semiangle_mrad) / calibration_radius
            source_kind = "detector"
            source_storage_path = str(source)
            source_dtype = str(loaded.dtype)
            source_bytes = loaded.logical_bytes
            source_load_seconds = time.perf_counter() - load_started
        session = cls(
            data,
            backend=selected,
            voltage_kV=voltage_kV,
            semiangle_mrad=semiangle_mrad,
            scan_sampling_A=scan_sampling_A,
            scan_shape=scan_shape,
            det_sampling=det_sampling,
            aberrations=aberrations,
            rotation_angle_deg=rotation_angle_deg,
            com_reversed=com_reversed,
            bf_intensity_threshold=bf_intensity_threshold,
            bf_radius=bf_radius,
            source_path=str(source),
            bf_center=bf_center,
        )
        session.source_kind = source_kind
        auto_calibration = (
            session.source_provenance.get("calibration_path")
            if isinstance(session.source_provenance, dict)
            else None
        )
        auto_manifest = (
            session.source_provenance.get("manifest_path")
            if isinstance(session.source_provenance, dict)
            else None
        )
        session.calibration_path = calibration or auto_calibration
        session.source_manifest_path = auto_manifest
        session.source_storage_path = source_storage_path
        session.source_dtype = source_dtype
        session.source_bytes = source_bytes
        session.source_detector_bin = int(getattr(data, "det_bin", 1) or 1)
        session.source_provenance = json_value(
            getattr(data, "source_provenance", None)
        )
        if data is not None and source_kind == "detector":
            correction = loaded.metadata.get("hot_pixel_correction")
            if correction is not None:
                session.source_provenance = dict(session.source_provenance or {})
                session.source_provenance["hot_pixel_correction"] = correction
        session.source_load_seconds = source_load_seconds
        return session

    @classmethod
    def from_array(
        cls,
        data: object,
        *,
        backend: Literal["auto", "cuda", "mps", "webgpu"] = "auto",
        voltage_kV: float,
        semiangle_mrad: float,
        scan_sampling_A: float | tuple[float, float],
        scan_shape: tuple[int, int] | None = None,
        det_sampling: float | tuple[float, float] | None = None,
        aberrations: dict[str, float] | None = None,
        rotation_angle_deg: float = 0.0,
        com_reversed: bool = False,
        bf_intensity_threshold: float = 0.0,
        bf_radius: int | None = None,
        source_path: str | None = None,
    ) -> SSB:
        """Create an SSB session from an existing lossless detector array."""

        return cls(
            data,
            backend=backend,
            voltage_kV=voltage_kV,
            semiangle_mrad=semiangle_mrad,
            scan_sampling_A=scan_sampling_A,
            scan_shape=scan_shape,
            det_sampling=det_sampling,
            aberrations=aberrations,
            rotation_angle_deg=rotation_angle_deg,
            com_reversed=com_reversed,
            bf_intensity_threshold=bf_intensity_threshold,
            bf_radius=bf_radius,
            source_path=source_path,
        )

    def _prepare_cuda(self):
        """Construct the private CUDA implementation once."""

        if self._cuda_session is None:
            from .backends.cuda.backend import CudaSSBBackend

            self._cuda_session = CudaSSBBackend(
                data=self._data,
                semiangle=self.semiangle_mrad,
                scan_sampling=self.scan_sampling_A,
                det_sampling=self.det_sampling,
                voltage_kV=self.voltage_kV,
                scan_shape=self._scan_shape,
                bf_intensity_threshold=self.bf_intensity_threshold,
                bf_radius=self.bf_radius,
                bf_center=self.bf_center,
                aberrations=(
                    _aberrations_to_engine(self.aberrations) if self._aberrations_explicit else None
                ),
                rotation_angle_deg=self.physical_rotation_deg,
            )
        return self._cuda_session

    def _source_signature(self) -> dict[str, object]:
        """Return the complete source identity used for saved-result reuse."""

        if self.source_path is None:
            raise ValueError(
                "save_to requires source_path when SSB.from_array() is used. "
                "Pass the detector source path so saved results cannot be "
                "reused for a different array accidentally."
            )
        if not Path(self.source_path).expanduser().exists():
            raise FileNotFoundError(
                f"Cannot save an SSB result because source_path does not exist: "
                f"{Path(self.source_path).expanduser()}"
            )
        try:
            from quantem.gpu.io import inspect

            inspected = inspect(self.source_path, scan_shape=self._scan_shape)
            signature = json_value(inspected.source_signature)
        except (OSError, RuntimeError, TypeError, ValueError):
            signature = path_signature(self.source_path)

        storage = None
        if (
            self.source_storage_path is not None
            and Path(self.source_storage_path).expanduser().resolve()
            != Path(self.source_path).expanduser().resolve()
        ):
            storage = path_signature(self.source_storage_path)
        calibration = (
            None
            if self.calibration_path is None
            else path_signature(self.calibration_path)
        )
        manifest = (
            None
            if self.source_manifest_path is None
            else path_signature(self.source_manifest_path)
        )
        return {
            "detector": signature,
            "storage": storage,
            "calibration": calibration,
            "manifest": manifest,
        }

    def _result_signature(
        self,
        operation: Literal["fit", "reconstruct"],
        settings: dict[str, object],
    ) -> dict[str, object]:
        """Build the exact scientific identity of one SSB operation."""

        return json_value(
            {
                "schema": SCHEMA,
                "software": software_signature(),
                "operation": operation,
                "source": self._source_signature(),
                "data": {
                    "kind": self.source_kind,
                    "dtype": self.source_dtype,
                    "bytes": self.source_bytes,
                    "shape": tuple(int(value) for value in self._data.shape),
                    "scan_shape": self._scan_shape,
                    "detector_bin": self.source_detector_bin,
                    "source_provenance": self.source_provenance,
                },
                "instrument": {
                    "voltage_kV": self.voltage_kV,
                    "semiangle_mrad": self.semiangle_mrad,
                    "scan_sampling_A": self.scan_sampling_A,
                    "det_sampling": self.det_sampling,
                },
                "ssb": {
                    "backend": self.backend,
                    # physical angle: saves made before the (angle, com_reversed) split still match exactly
                    "rotation_angle_deg": self.physical_rotation_deg,
                    "bf_intensity_threshold": self.bf_intensity_threshold,
                    "bf_radius": self.bf_radius,
                    "bf_center": self.bf_center,
                },
                "settings": settings,
            }
        )

    def _save_result(
        self,
        result: SSBResult,
        *,
        paths: tuple[Path, Path],
        signature: dict[str, object],
    ) -> SSBResult:
        """Persist one result and attach its complete provenance."""

        return save_result(
            result,
            paths=paths,
            signature=signature,
            input_metadata={
                "source_path": self.source_path,
                "source_storage_path": self.source_storage_path,
                "source_kind": self.source_kind,
                "source_dtype": self.source_dtype,
                "source_bytes": self.source_bytes,
                "source_detector_bin": self.source_detector_bin,
                "source_provenance": self.source_provenance,
                "source_load_seconds": self.source_load_seconds,
                "calibration_path": self.calibration_path,
                "source_manifest_path": self.source_manifest_path,
            },
        )

    def _accept_result(self, result: SSBResult) -> SSBResult:
        """Update session state from a computed or reused result."""

        result.source_path = self.source_path
        if result.physical_rotation_deg != self.physical_rotation_deg and self._backend_ready():
            # a reversed or reused result can sit on the other 180-degree branch; previews must use its rotation
            self.set_rotation(result.rotation_angle_deg, com_reversed=result.com_reversed)
        self.aberrations = dict(result.aberrations)
        self.tilt_mrad, self.depth_spread_nm = result.tilt_mrad, result.depth_spread_nm
        self._aberrations_explicit = True
        self.rotation_angle_deg, self.com_reversed = result.rotation_angle_deg, result.com_reversed
        self.best_loss = (
            float(result.loss) if result.loss is not None else float("inf")
        )
        self.trial_history = [dict(trial) for trial in result.optuna_trials or ()]
        self._reconstruction = result
        return result

    def _resolve_rotation_branch(self, result: SSBResult, *, refinement: RefineMethod, tilt: bool,
                                 tilt_limit_mrad: float, seed: int, verbose: bool) -> SSBResult:
        """Keep the CoM direction whose phase has bright atom columns (see ``fit(check_rotation=...)``).

        The CoM curl fixes the scan-detector rotation only up to 180 degrees; the other branch is the same angle with
        every CoM vector reversed. When the fitted atom columns are dark, the CoM is reversed (the angle stays) and the
        fit is repeated there, starting from every aberration flipped in sign (C10 -> -C10, C12 -> -C12 at the same
        angle; the tilt too): the other branch is the conjugate object, whose aberration phase is the negative of this
        one. A tilt-aware fit repeats its search.
        """
        sign = column_sign(result.phase)
        result.column_sign = sign
        show = verbose and _in_notebook()
        angle = float(result.rotation_angle_deg)
        state = "CoM reversed" if result.com_reversed else "CoM as measured"
        limit = draw_column_histogram(result.phase, f"fit at rotation {angle:.1f}°, {state} · column sign {sign:+.2f}") if show else None
        if sign >= -COLUMN_SIGN_MIN:
            if verbose and sign < COLUMN_SIGN_MIN:
                print(f"SSB: column sign {sign:+.2f} is too weak to decide the 180° question; rotation stays {angle:.1f}°, {state}.")
            return result
        reversed_now = not result.com_reversed
        start = {"C10": -result.aberrations["C10"], "C12": -result.aberrations["C12"], "phi12": result.aberrations["phi12"]}
        if verbose:
            print(f"SSB: atom columns are dark at rotation {angle:.1f}°, {state} (column sign {sign:+.2f}).\n"
                  f"     CoM {'reversed' if reversed_now else 'restored'} for this session, rotation stays {angle:.1f}°; "
                  f"refitting with every aberration flipped (C10 {start['C10']:+.2f} nm, C12 {start['C12']:+.2f} nm).")
        self.set_rotation(angle, com_reversed=reversed_now)
        backend = self._backend_protocol
        if tilt:
            refit = self._fit_tilt(result.n_trials or 200, refinement, tilt_limit_mrad, seed, False)
        else:
            backend.reconstruct_result(_aberrations_to_engine(start), compute_loss=False)   # the refit starts here
            refit = _result_from_engine(backend.fit(trials=0, refinement=refinement or "nelder-mead", search_ranges=None,
                                                    refine_lock=None, seed=seed, verbose=False))
            refit.n_trials, refit.optuna_trials = result.n_trials, result.optuna_trials
        refit.column_sign = column_sign(refit.phase)
        refit.rotation_flipped = True
        if refit.elapsed is not None and result.elapsed is not None:
            refit.elapsed += result.elapsed   # time of the whole fit: first pass + refit
        state = "CoM reversed" if refit.com_reversed else "CoM as measured"
        if show:
            draw_column_histogram(refit.phase, f"refit at rotation {angle:.1f}°, {state} · column sign {refit.column_sign:+.2f}", limit=limit)
        if verbose:
            print(f"SSB: refit at rotation {angle:.1f}°, {state}: C10 {refit.aberrations['C10']:+.2f} nm, "
                  f"column sign {refit.column_sign:+.2f}. fit(check_rotation=False) keeps the rotation you give.")
        return refit

    def _backend_ready(self) -> bool:
        """True once a backend holds prepared data, so changing its rotation does not trigger a load."""
        return self._cuda_session is not None or self._mps_backend is not None

    @property
    def _backend_protocol(self) -> SSBProtocol:
        """Return the sole strict backend implementation for this session."""

        if self.backend == "cuda":
            backend = self._prepare_cuda()
        else:
            if self._mps_backend is None:
                from .backends.mps.backend import MpsSSBBackend

                self._mps_backend = MpsSSBBackend(
                    self._data,
                    voltage_kV=self.voltage_kV,
                    semiangle_mrad=self.semiangle_mrad,
                    scan_sampling=self.scan_sampling_A,
                    det_sampling=self.det_sampling,
                    bf_intensity_threshold=self.bf_intensity_threshold,
                    bf_center=self.bf_center,
                    bf_radius=self.bf_radius,
                    rotation_angle_deg=self.physical_rotation_deg,
                    aberrations=(
                        _aberrations_to_engine(self.aberrations) if self._aberrations_explicit else None
                    ),
                )
            backend = self._mps_backend
        if not isinstance(backend, SSBProtocol):
            raise TypeError(
                f"The {self.backend} implementation does not satisfy SSBProtocol."
            )
        return backend

    def fit(
        self,
        *,
        tilt: bool = False,
        trials: int = 200,
        refinement: RefineMethod = "nelder-mead",
        search_ranges: dict[str, tuple[float, float] | float] | None = None,
        refine_lock: list[str] | None = None,
        tilt_limit_mrad: float = 25.0,
        check_rotation: bool = True,
        seed: int = 42,
        save_to: str | Path | None = None,
        force: bool = False,
        verbose: bool = True,
    ) -> SSBResult:
        """Optimize C10/C12/phi12 and return the final reconstruction.

        ``tilt=True`` also fits the sample tilt and a depth spread for a thick, tilted crystal, jointly with the
        aberrations in one search of ``trials`` trials plus Nelder-Mead (fitting the tilt after a standard fit gets stuck:
        C10 has to move to the defocus at mid-depth at the same time). The phase-variance loss of the standard fit does
        not see tilt, so this search maximises the least-squares agreement of the thick-sample model with the data. The
        result's ``tilt_mrad`` (row, col; scan frame, within ``tilt_limit_mrad``) and ``depth_spread_nm`` hold the fit; CUDA and
        MPS. On two full 512 x 512 acquisitions 100 trials already converged every seed and 200-400 gave the same tilt
        to 0.02 mrad (docs/maintainer/2026-09-24-ssb-units-and-thick-sample.md).

        ``check_rotation=True`` (default) also settles the 180-degree ambiguity of the scan-detector rotation: the rotation
        search cannot tell omega from omega + 180 degrees and both fit equally well (the second is the conjugate object with
        the opposite aberration phase). Atom columns carry positive phase, so when the fitted phase has a negative column
        sign (``result.column_sign`` < -0.2) the CoM is reversed for the session (``com_reversed``; the angle stays below
        180 degrees), said in one line, and the fit is repeated there with every aberration flipped in sign. In a notebook the phase histogram is drawn after the fit (and after the
        refit). Near zero (no resolved columns) the rotation is kept. ``check_rotation=False`` keeps the rotation as given.

        Set ``save_to`` to reuse an exact prior result when the detector source,
        calibration, backend, physical parameters, and fit settings all match.
        Changed settings recompute automatically. Set ``force=True`` to recompute
        an otherwise matching result.
        """

        if tilt and (search_ranges is not None or refine_lock is not None):
            raise ValueError("search_ranges and refine_lock apply to the standard fit; tilt=True searches its own ranges.")
        if trials < 0:
            raise ValueError(f"trials must be non-negative, got {trials}.")
        if refinement not in {"nelder-mead", None}:
            raise ValueError("refinement must be 'nelder-mead' or None.")
        paths = None
        signature = None
        if save_to is not None:
            paths = result_paths(save_to, "fit")
            signature = self._result_signature(
                "fit",
                {
                    "trials": int(trials),
                    "refinement": refinement,
                    "search_ranges": search_ranges,
                    "refine_lock": refine_lock,
                    "seed": int(seed),
                    "starting_aberrations": self._fit_start_aberrations,
                    "starting_aberrations_explicit": (
                        self._fit_start_aberrations_explicit
                    ),
                    **({"tilt": True, "tilt_limit_mrad": float(tilt_limit_mrad)} if tilt else {}),
                    "check_rotation": bool(check_rotation),
                },
            )
            if not force:
                reused = load_result(
                    paths=paths,
                    signature=signature,
                    backend=self.backend,
                )
                if reused is not None:
                    if verbose:
                        print(f"Matching SSB result found; loading {paths[0]}")
                    return self._accept_result(reused)
            if verbose and any(path.exists() for path in paths):
                print("Saved SSB settings changed; running fit end to end")
        if tilt:
            result = self._accept_result(self._fit_tilt(int(trials), refinement, float(tilt_limit_mrad), int(seed), verbose))
        else:
            result = self._backend_protocol.fit(
                trials=int(trials),
                refinement=refinement,
                search_ranges=_search_ranges_to_engine(search_ranges),
                refine_lock=refine_lock,
                seed=int(seed),
                verbose=verbose,
            )
            result = self._accept_result(_result_from_engine(result))
        if check_rotation:
            result = self._accept_result(self._resolve_rotation_branch(
                result, refinement=refinement, tilt=tilt, tilt_limit_mrad=float(tilt_limit_mrad), seed=int(seed),
                verbose=verbose))
        if paths is not None and signature is not None:
            result = self._save_result(result, paths=paths, signature=signature)
            if verbose:
                print(f"SSB result saved to {paths[0]}")
        return result

    def reconstruct(
        self,
        aberrations: dict[str, float] | None = None,
        *,
        compute_loss: bool = True,
        save_to: str | Path | None = None,
        force: bool = False,
        verbose: bool = False,
    ) -> SSBResult:
        """Reconstruct the complex object wave at fixed aberrations.

        Set ``compute_loss=False`` when only the exact reconstructed object is
        needed. This skips the separate post-reconstruction loss calculation;
        it does not change the object wave or phase.
        """

        coefs = (
            self.aberrations
            if aberrations is None
            else _validate_aberrations(aberrations)
        )
        paths = None
        signature = None
        if save_to is not None:
            paths = result_paths(save_to, "reconstruct")
            signature = self._result_signature(
                "reconstruct",
                {"aberrations": coefs, "compute_loss": bool(compute_loss)},
            )
            if not force:
                reused = load_result(
                    paths=paths,
                    signature=signature,
                    backend=self.backend,
                )
                if reused is not None:
                    if verbose:
                        print(f"Matching SSB result found; loading {paths[0]}")
                    return self._accept_result(reused)
            if verbose and any(path.exists() for path in paths):
                print("Saved SSB settings changed; running reconstruction end to end")

        if (
            not force
            and save_to is None
            and aberrations is None
            and self._reconstruction is not None
            and (not compute_loss or self._reconstruction.loss is not None)
        ):
            result = self._reconstruction
        else:
            result = self._backend_protocol.reconstruct_result(
                _aberrations_to_engine(coefs),
                compute_loss=compute_loss,
            )
            result = self._accept_result(_result_from_engine(result))
        if paths is not None and signature is not None:
            result = self._save_result(result, paths=paths, signature=signature)
            if verbose:
                print(f"SSB result saved to {paths[0]}")
        return result

    def preview(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool = True,
        upsampling_factor: int = 1,
        higher_order_magnitudes: np.ndarray | None = None,
        higher_order_angles: np.ndarray | None = None,
        context: AbstractContextManager | None = None,
        tilt_mrad: tuple[float, float] = (0.0, 0.0),
        depth_spread_nm: float = 0.0,
        phase_estimator: Literal["mean_phase", "phase_of_mean"] = "mean_phase",
    ) -> tuple[np.ndarray, float | None]:
        """Reconstruct a transient phase image for an interactive viewer.

        The returned array has the full selected output resolution. "Preview"
        means it is not saved and does not replace the fitted calibration or
        stored result; it does not imply a lower-quality reconstruction.

        ``aberrations`` C10 / C12 in nm, phi12 in rad. A positive ``depth_spread_nm`` switches to the thick-sample model:
        each bright-field pixel's correction is averaged over that depth, with the crystal leaning by ``tilt_mrad``
        (row, col; scan frame), as fitted by ``fit(tilt=True)``. With no depth spread the tilt has no effect and this is
        standard SSB. CUDA and MPS backends.

        ``upsampling_factor`` selects 1, 2, 3, 4, or 8 times finer output sampling.
        Values above one support C10/C12 with or without tilt/depth correction
        on CUDA. The diagnostic loss and aberration search remain on the native
        scan grid. Sampling never fits parameters or interpolates detector data.

        ``phase_estimator="mean_phase"`` preserves the existing average of
        per-detector phases. Experimental ``"phase_of_mean"`` instead takes
        the phase after averaging corrected complex waves, on CUDA with C10/C12
        and optional tilt/depth. This choice applies at every sampling factor,
        including 1x, so changing sampling never switches estimators. It can
        reduce sampling-dependent contrast compression, but quantitative phase
        amplitude is not validated. Loss still uses the existing native-grid
        per-detector phase variance; fitting is unaffected.

        Examples
        --------
        Fit all correction parameters once, then change only output sampling:

        >>> fitted = ssb.fit(tilt=True)  # native-grid joint fit
        >>> phase, loss = ssb.preview(
        ...     fitted.aberrations, tilt_mrad=fitted.tilt_mrad,
        ...     depth_spread_nm=fitted.depth_spread_nm, upsampling_factor=4,
        ... )

        Compare the experimental estimator explicitly, without changing the fit:

        >>> candidate, native_loss = ssb.preview(
        ...     fitted.aberrations, tilt_mrad=fitted.tilt_mrad,
        ...     depth_spread_nm=fitted.depth_spread_nm, upsampling_factor=4,
        ...     phase_estimator="phase_of_mean",
        ... )
        """

        coefs = _aberrations_to_engine(_validate_aberrations(aberrations))
        if phase_estimator not in ("mean_phase", "phase_of_mean"):
            raise ValueError(
                "phase_estimator must be 'mean_phase' or 'phase_of_mean'; "
                f"got {phase_estimator!r}."
            )
        if type(upsampling_factor) is not int or upsampling_factor not in (1, 2, 3, 4, 8):
            raise ValueError("upsampling_factor must be 1, 2, 3, 4, or 8.")
        if (higher_order_magnitudes is None) != (higher_order_angles is None):
            raise ValueError("Higher-order magnitudes and angles must be provided together.")
        if higher_order_magnitudes is not None:
            magnitudes = np.asarray(higher_order_magnitudes, dtype=np.float32)
            angles = np.asarray(higher_order_angles, dtype=np.float32)
            if magnitudes.shape != (14,) or angles.shape != (14,):
                raise ValueError("Higher-order SSB arrays must each have shape (14,).")
            # An angle has no physical effect when its coefficient is zero.
            uses_depth_kernel = (
                upsampling_factor > 1
                or depth_spread_nm > 0
                or phase_estimator == "phase_of_mean"
            )
            if uses_depth_kernel and not np.any(magnitudes[2:]):
                # Preserve the explicit primary coefficients in the packed API.
                coefs = {**coefs, "C10": float(magnitudes[0]) * _ENGINE_PER_NM,
                         "C12": float(magnitudes[1]) * _ENGINE_PER_NM,
                         "phi12": float(angles[1])}
                higher_order_magnitudes = higher_order_angles = None
        if upsampling_factor != 1 or phase_estimator == "phase_of_mean":
            if higher_order_magnitudes is not None:
                raise ValueError(
                    "Upsampling and wave averaging support C10/C12 with "
                    "tilt/depth; turn off higher-order magnitudes."
                )
            backend = self._backend_protocol
            if not hasattr(backend, "preview_upsampled"):
                raise NotImplementedError(
                    "Upsampled or wave-average SSB preview requires the CUDA "
                    "backend; use backend='cuda'."
                )
            options = dict(
                upsampling_factor=upsampling_factor, compute_loss=compute_loss,
                tilt_mrad=tuple(float(value) for value in tilt_mrad),
                thickness=float(depth_spread_nm) * _ENGINE_PER_NM,
                phase_estimator=phase_estimator,
            )
            if context is None:
                return backend.preview_upsampled(coefs, **options)
            with context:
                return backend.preview_upsampled(coefs, **options)
        if float(depth_spread_nm) > 0.0:
            sample = {"tilt_row_mrad": float(tilt_mrad[0]), "tilt_col_mrad": float(tilt_mrad[1]),
                      "thickness": float(depth_spread_nm) * _ENGINE_PER_NM}
            if higher_order_magnitudes is not None:
                raise ValueError("The thick-sample preview does not combine with higher-order aberrations yet.")
            backend = self._backend_protocol
            if not hasattr(backend, "preview_sample"):
                raise NotImplementedError("Sample tilt / thickness SSB is implemented for the CUDA and MPS backends only.")
            if context is None:
                return backend.preview_sample(coefs, sample, compute_loss=compute_loss)
            with context:
                return backend.preview_sample(coefs, sample, compute_loss=compute_loss)
        if (higher_order_magnitudes is None) != (higher_order_angles is None):
            raise ValueError(
                "higher_order_magnitudes and higher_order_angles must be "
                "provided together."
            )
        magnitudes = (
            None
            if higher_order_magnitudes is None
            # all 14 magnitudes (C10, C12 in slots 0-1, then C21..C56) are nm; the engines take Angstrom
            else np.asarray(higher_order_magnitudes, dtype=np.float32) * np.float32(_ENGINE_PER_NM)
        )
        angles = (
            None
            if higher_order_angles is None
            else np.asarray(higher_order_angles, dtype=np.float32)
        )
        if magnitudes is not None and (
            magnitudes.shape != (14,) or angles is None or angles.shape != (14,)
        ):
            raise ValueError("Higher-order SSB arrays must each have shape (14,).")
        backend = self._backend_protocol
        if context is None:
            return backend.preview(
                coefs,
                compute_loss=compute_loss,
                higher_order_magnitudes=magnitudes,
                higher_order_angles=angles,
            )
        with context:
            return backend.preview(
                coefs,
                compute_loss=compute_loss,
                higher_order_magnitudes=magnitudes,
                higher_order_angles=angles,
            )

    @property
    def supports_tilt(self) -> bool:
        """True when this session's backend implements the thick-sample model (``fit(tilt=True)``, ``preview(tilt_mrad=...)``)."""
        return hasattr(self._backend_protocol, "fit_sample")

    def _fit_tilt(self, trials: int, refinement: RefineMethod, tilt_limit_mrad: float, seed: int, verbose: bool) -> SSBResult:
        """Joint thick-sample fit (backend ``fit_sample``, engine units) -> an nm SSBResult at the fitted parameters."""
        backend = self._backend_protocol
        if not hasattr(backend, "fit_sample"):
            raise NotImplementedError("fit(tilt=True) is implemented for the CUDA and MPS backends only.")
        started = time.perf_counter()
        fit = backend.fit_sample(trials=trials, tilt_limit_mrad=tilt_limit_mrad, seed=seed, verbose=verbose,
                                 polish_starts=0 if refinement is None else 3)
        aberrations = _aberrations_from_engine({key: float(fit[key]) for key in ("C10", "C12", "phi12")})
        tilt_mrad = (float(fit["tilt_row_mrad"]), float(fit["tilt_col_mrad"]))
        depth_spread_nm = float(fit["thickness"]) / _ENGINE_PER_NM
        phase, loss = self.preview(aberrations, tilt_mrad=tilt_mrad, depth_spread_nm=depth_spread_nm)
        # the thick-sample path recovers the phase only; the transmission amplitude is not estimated
        if self.backend == "cuda":
            import cupy as cp

            object_wave = cp.exp(1j * cp.asarray(phase))
        else:
            object_wave = np.exp(1j * np.asarray(phase))
        return SSBResult(object_wave=object_wave, backend=self.backend, aberrations=aberrations, tilt_mrad=tilt_mrad,
                         depth_spread_nm=depth_spread_nm, tilt_fit_gain=float(fit["gain"]),
                         rotation_angle_deg=self.rotation_angle_deg, com_reversed=self.com_reversed,
                         loss=None if loss is None else float(loss),
                         elapsed=time.perf_counter() - started, n_trials=trials, num_bf=self.num_bf,
                         refine_method=refinement, voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
                         scan_sampling_A=self.scan_sampling_A)

    def preview_context(self, num_bf: int):
        """Prepare a backend-owned reduced-BF interaction context."""

        return self._backend_protocol.preview_context(int(num_bf))

    def browser_state(self):
        """Return compact backend-neutral state for browser WebGPU."""

        return self._backend_protocol.browser_state()

    def export_brightfield(
        self,
        path_stem: str | Path,
    ) -> tuple[str, float] | None:
        """Persist exact raw-count bright-field columns when supported."""

        written = self._backend_protocol.export_brightfield(self._data, path_stem)
        if written is None:
            return None
        path, elapsed = written
        return str(path), float(elapsed)

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Prepared scan shape in public ``(row, col)`` order."""

        return self._backend_protocol.scan_shape

    @property
    def num_bf(self) -> int:
        """Number of pixels in the complete detected bright-field disk."""

        return self._backend_protocol.num_bf

    @property
    def physical_rotation_deg(self) -> float:
        """Rotation the engines use: ``rotation_angle_deg + 180`` when the CoM is reversed."""
        return physical_rotation_deg(self.rotation_angle_deg, self.com_reversed)

    def set_rotation(self, rotation_angle_deg: float, com_reversed: bool = False) -> None:
        """Set scan-to-detector rotation (any angle; stored as below 180 plus ``com_reversed``) and refresh geometry."""

        self.rotation_angle_deg, self.com_reversed = split_rotation(rotation_angle_deg, com_reversed)
        self._backend_protocol.cache_rotation(math.radians(self.physical_rotation_deg))

    def close(self) -> None:
        """Release backend-owned GPU state."""

        data = self._data
        self._data = None
        backend_owned_data = (
            self._cuda_session is not None or self._mps_backend is not None
        )
        if self._cuda_session is not None:
            self._cuda_session.close()
            self._cuda_session = None
        if self._mps_backend is not None:
            backend = self._mps_backend
            self._mps_backend = None
            # The MPS backend's final allocator flush must run after the
            # workflow releases its own reference to the shared source.
            backend.close()
        if not backend_owned_data:
            _release_owned_storage(data)

    def __enter__(self) -> Self:
        """Return this prepared SSB session."""

        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        """Release backend resources when leaving a context manager."""

        self.close()
__all__ = ["SSB"]
