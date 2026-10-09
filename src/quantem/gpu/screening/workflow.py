"""Cached BF/DF/CoM/rotation products for rapid 4D-STEM dataset review.

Screening answers "is this scan worth reconstructing?" before any iterative
work starts. One call loads the acquisition into encoded GPU storage, derives
the mean diffraction pattern, bright-field and dark-field images, centre of
mass, scan-detector rotation and integrated DPC phase, and saves them in a
small cache beside the master so the next look at the same scan is instant.
"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CACHE_VERSION = 3
_EXACT_COUNT_PRODUCT_FIELDS = (
    "total_intensity",
    "annular_bright_field",
    "annular_dark_field",
)
# Detector bands as (inner, outer) multiples of the fitted bright-field radius,
# in the order BF, ABF, ADF, DF that the cache metadata records.
_BAND_RADII = ((0.0, 1.0), (0.5, 1.0), (1.0, 2.0), (1.0, np.inf))


@dataclass
class ScreeningResult:
    """Small screening products for instant UI and ptychography setup.

    ``total_intensity``, ``annular_bright_field`` and ``annular_dark_field``
    are exact uint64 count maps. They are ``None`` only when reopening a cache
    that was written without them.
    """

    mean_dp: np.ndarray
    bright_field: np.ndarray
    dark_field: np.ndarray
    dpc_phase: np.ndarray
    com_row: np.ndarray
    com_col: np.ndarray
    probe_center: tuple[float, float]
    probe_radius: float
    rotation_deg: float
    transposed: bool
    metadata: dict[str, object]
    cache_path: Path | None = None
    from_cache: bool = False
    elapsed_s: float = 0.0
    total_intensity: np.ndarray | None = None
    annular_bright_field: np.ndarray | None = None
    annular_dark_field: np.ndarray | None = None


def prepare(
    source: str | Path,
    *,
    backend: str = "auto",
    scan_shape: tuple[int, int] | None = None,
    rotation_angle_deg: float | None = None,
    cache: bool = True,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    rotation_steps: int = 90,
    verbose: bool = False,
) -> ScreeningResult:
    """Load or build the screening products of one 4D-STEM acquisition.

    A cache miss loads the acquisition once into encoded CUDA or MPS storage
    and derives every product from exact integer detector sums: the mean
    diffraction pattern, the fitted bright-field disk, per-position total,
    BF, ABF, ADF and DF counts, and the centre of mass from exact detector
    moments. The rotation that minimizes the curl of the centre-of-mass field
    and the integrated DPC phase follow on the host. Integer products are
    identical on CUDA and MPS. Flagged detector pixels count as zero. The raw
    acquisition remains the evidence source for reconstruction.

    Parameters
    ----------
    source : str or Path
        HDF5 master (for example an Arina ``*_master.h5``) or another
        acquisition that ``io.load`` reads into integer counts.
    backend : str
        ``"auto"``, ``"cuda"`` or ``"mps"``; only a cache miss needs a GPU.
    scan_shape : tuple of int, optional
        ``(rows, cols)`` scan positions when the file does not record them.
    rotation_angle_deg : float, optional
        Scan-detector rotation in degrees. ``None`` fits it from
        ``rotation_steps`` angles in ``[0, 180)`` degrees, testing transposed
        detector axes too.
    cache, cache_dir, refresh : bool, str or Path, bool
        Reuse and save ``<master stem>.screening-v3.npz``, by default in
        ``.quantem_gpu_cache`` beside the master. ``refresh`` rebuilds it.
        A cache is reused only while every source file is unchanged.
    rotation_steps : int
        Number of rotation angles tried when fitting the rotation.
    verbose : bool
        Print the load report and the product timing.

    Returns
    -------
    ScreeningResult
        Scan-shaped ``(row, col)`` images, the detector-shaped mean pattern,
        the probe disk ``(row, col)`` centre and radius in detector pixels,
        and provenance in ``metadata``.

    Examples
    --------
    >>> products = screening.prepare("scan_master.h5")
    >>> products.bright_field.shape, products.probe_radius
    """
    master_path = Path(source).expanduser()
    if not master_path.exists():
        raise FileNotFoundError(f"HDF5 master not found: {master_path}")
    cache_path = _cache_path(master_path, cache_dir)
    if cache and not refresh:
        products = _prepare_cache(cache_path, master_path)
        if products is not None:
            return _with_rotation(products, rotation_angle_deg)

    # A cache hit stays cheap: device, raw I/O and detector modules are
    # imported only by a cache miss.
    from quantem.gpu import device

    resolved_backend = device.resolve(backend)
    if resolved_backend not in {"cuda", "mps"}:
        raise RuntimeError(
            "prepare cache generation requires backend='cuda' "
            f"or backend='mps'; backend={resolved_backend!r} was selected. "
            "Existing caches can still be read by CPU-facing callers."
        )
    products = _build_products(
        master_path,
        backend=resolved_backend,
        scan_shape=scan_shape,
        rotation_steps=int(rotation_steps),
        verbose=verbose,
    )
    products.cache_path = cache_path if cache else None
    if cache:
        _save_cache(products, cache_path)
    return _with_rotation(products, rotation_angle_deg)


# --- building the products from one encoded acquisition ---


def _build_products(
    master: Path,
    *,
    backend: str,
    scan_shape: tuple[int, int] | None,
    rotation_steps: int,
    verbose: bool,
) -> ScreeningResult:
    """Derive every screening product from the encoded resident ``io.load`` returns.

    Both GPU backends answer the same detector-session queries, so one builder
    serves CUDA and MPS. The exact detector total gives the mean pattern and
    the probe fit; one batch of exact binary-mask sums then gives, per scan
    position, the total and the four bands, and the session's centre of mass
    divides exact detector moments by the exact total. Flagged pixels are
    zeroed at load (``io.load`` defaults to a local median), as screening
    products have always counted them.
    """
    from quantem.gpu import detector
    from quantem.gpu.dpc.workflow import find_optimal_rotation
    from quantem.gpu.io import load

    started = time.perf_counter()
    source = _source_fingerprint(master)
    with load(
        str(master),
        backend=backend,
        scan_shape=scan_shape,
        hot_pixel_correction="zero",
        verbose=verbose,
    ) as loaded:
        # A live acquisition may still be writing; products of a moving
        # source must not be cached under its final identity.
        if _source_fingerprint(master) != source:
            raise RuntimeError(
                "The 4D-STEM source changed while screening products were being "
                "computed. Wait for acquisition to finish, then retry."
            )
        if loaded.dtype.kind != "u":
            raise ValueError(
                "Screening products are exact detector-count sums; "
                f"{master.name} stores {loaded.dtype} intensities. Use "
                "quantem.gpu.detector and quantem.gpu.dpc on the loaded data instead."
            )
        loaded_s = time.perf_counter() - started
        session = detector.prepare(loaded)
        scan_shape = session.scan_shape
        detector_shape = session.detector_shape
        frame_count = session.num_frames
        # The exact total divided in float64 and rounded once, like every mean pattern.
        mean_dp = (session.detector_total() / frame_count).astype(np.float32)
        center, radius = detector.fit_probe(mean_dp)
        masks = [np.ones(detector_shape, dtype=bool)]
        masks += [
            detector.detector_mask(center, inner * radius, outer * radius, detector_shape)
            for inner, outer in _BAND_RADII
        ]
        sums = session.masked_sums_exact(np.stack(masks))
        com_row, com_col = session.center_of_mass()
        working_dtype = loaded.dtype.name
    reduced_s = time.perf_counter() - started - loaded_s

    total, bright_field, annular_bright_field, annular_dark_field, dark_field = sums
    com_row -= float(com_row.mean())
    com_col -= float(com_col.mean())
    rotation_started = time.perf_counter()
    _, _, rotation_deg, transposed = find_optimal_rotation(
        com_row,
        com_col,
        rotation_steps=rotation_steps,
    )
    phase_started = time.perf_counter()
    phase = _dpc_phase(com_row, com_col, float(rotation_deg), bool(transposed))
    elapsed_s = time.perf_counter() - started
    if verbose:
        print(
            f"Screening products in {elapsed_s:.2f} s: probe radius {radius:.1f} px, "
            f"rotation {rotation_deg:.1f} deg{' (transposed)' if transposed else ''}."
        )

    parameters = {
        "scan_shape": [int(value) for value in scan_shape],
        "detector_shape": [int(value) for value in detector_shape],
        "dtype": working_dtype,
        "hot_pixel_correction": "zero",
        "rotation_steps": int(rotation_steps),
        "center": [float(center[0]), float(center[1])],
        "radius_px": float(radius),
        "rotation_deg": float(rotation_deg),
        "transposed": bool(transposed),
        "backend": backend,
    }
    timing = {
        "load_s": float(loaded_s),
        "reduce_s": float(reduced_s),
        "rotation_s": float(phase_started - rotation_started),
        "idpc_s": float(time.perf_counter() - phase_started),
        "elapsed_s": float(elapsed_s),
    }
    return ScreeningResult(
        mean_dp=mean_dp,
        bright_field=bright_field.astype(np.float32),
        dark_field=dark_field.astype(np.float32),
        dpc_phase=phase,
        com_row=com_row,
        com_col=com_col,
        probe_center=(float(center[0]), float(center[1])),
        probe_radius=float(radius),
        rotation_deg=float(rotation_deg),
        transposed=bool(transposed),
        metadata={
            "version": _CACHE_VERSION,
            "source": source,
            "parameters": parameters,
            "timing": timing,
            "exact_accumulation": {
                "dtype": "uint64",
                "published_uint64_products": list(_EXACT_COUNT_PRODUCT_FIELDS),
                "detector_band_order": ["BF", "ABF", "ADF", "DF"],
                "detector_band_radius_multipliers": [
                    [inner, None if np.isinf(outer) else outer]
                    for inner, outer in _BAND_RADII
                ],
                "mean_dp_divisor": int(frame_count),
                "coordinate_order": "row-column",
            },
            "mode": "cached-screening-products",
            "note": (
                "Products are exact uint64 detector sums of the encoded "
                "acquisition; the raw source remains the reconstruction "
                "evidence."
            ),
        },
        from_cache=False,
        elapsed_s=elapsed_s,
        total_intensity=total,
        annular_bright_field=annular_bright_field,
        annular_dark_field=annular_dark_field,
    )


def _with_rotation(
    result: ScreeningResult,
    rotation_angle_deg: float | None,
) -> ScreeningResult:
    """Apply an explicit DPC rotation without touching the raw evidence.

    A user who knows the scan-detector rotation (for example from a
    calibration at this camera length) overrides the fitted one; only the
    phase depends on it, so the cached centre of mass is reused.
    """
    if rotation_angle_deg is None:
        return result
    result.rotation_deg = float(rotation_angle_deg)
    result.transposed = False
    result.dpc_phase = _dpc_phase(
        result.com_row,
        result.com_col,
        result.rotation_deg,
        False,
    )
    return result


def _dpc_phase(
    com_row: np.ndarray,
    com_col: np.ndarray,
    rotation_deg: float,
    transposed: bool,
) -> np.ndarray:
    """Rotate the CoM field into scan axes and integrate it to the float32 iDPC phase.

    Fitted and forced rotations reach the phase through this one float32
    rotation, so a cached phase and a recomputed one agree bit for bit.
    ``dpc.run`` cannot replace it: it recomputes the centre of mass from the
    data, and rotates in float64 for a forced angle.
    """
    from quantem.gpu.dpc import integrate

    source_row = com_col if transposed else com_row
    source_col = com_row if transposed else com_col
    angle = np.radians(rotation_deg)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    aligned_row = (cosine * source_row - sine * source_col).astype(np.float32)
    aligned_col = (sine * source_row + cosine * source_col).astype(np.float32)
    gradient_row, gradient_col = (
        (aligned_col, aligned_row) if transposed else (aligned_row, aligned_col)
    )
    return integrate(gradient_row, gradient_col)


# --- the product cache beside the master ---


def _cache_path(
    master: str | Path,
    cache_dir: str | Path | None = None,
) -> Path:
    """Name the cache after the master and the cache version.

    The version in the name keeps a reader from ever reopening a cache
    written in another layout; by default the cache sits beside the master.
    """
    master_path = Path(master).expanduser()
    if cache_dir is None:
        cache_root = master_path.parent / ".quantem_gpu_cache"
    else:
        cache_root = Path(cache_dir).expanduser()
    return cache_root / f"{master_path.stem}.screening-v{_CACHE_VERSION}.npz"


def _source_fingerprint(master: Path) -> dict[str, object]:
    """Return the complete master-and-shard source identity.

    The cache is keyed by this identity, so products are reused only while
    the master and every external shard are unchanged. A source whose
    inspection has no signature falls back to the master's size and time.
    """
    from quantem.gpu.io import inspect as inspect_source

    inspection = inspect_source(str(master))
    signature = inspection.source_signature
    if signature:
        return dict(signature)
    stat = master.stat()
    return {
        "master": str(master.resolve()),
        "files": [
            {
                "path": str(master.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        ],
        "datasets": [],
        "expectation": {"frames": None, "basis": None},
    }


def _cache_matches(metadata: dict[str, object], master: Path) -> bool:
    """Accept a cache only for the current version and an unchanged source."""
    if int(metadata.get("version", -1)) != _CACHE_VERSION:
        return False
    cached_source = metadata.get("source")
    stat_match = _strong_cached_source_match(cached_source, master)
    if stat_match is not None:
        return stat_match
    return cached_source == _source_fingerprint(master)


def _strong_cached_source_match(
    cached_source: object,
    master: Path,
) -> bool | None:
    """Validate a cached source identity using fail-closed file statistics.

    Complete HDF5 inspections retain size, modification/change times, device,
    and inode for the master and every external shard. When all strong fields
    are present, comparing them is equivalent to rediscovering the same HDF5
    links but avoids reparsing every source header on each cache reopen.

    ``None`` asks the caller to use full HDF5 inspection for a reduced
    signature. A malformed complete signature returns ``False``.
    """
    if not isinstance(cached_source, dict):
        return False
    files = cached_source.get("files")
    cached_master = cached_source.get("master")
    if not isinstance(files, list) or not files or not isinstance(cached_master, str):
        return False
    required = {"path", "size", "mtime_ns", "ctime_ns", "device", "inode"}
    if any(not isinstance(item, dict) or not required <= item.keys() for item in files):
        return None

    requested_master = os.path.abspath(os.path.expanduser(os.fspath(master)))
    if cached_master != requested_master:
        # Relative HDF5 external links resolve from the requested master
        # spelling's parent. Two aliases can therefore name the same master
        # inode but select different shards. Reinspect instead of treating
        # canonical-path equality as scientific-source equality.
        return None

    observed_paths: set[str] = set()
    for item in files:
        path = item["path"]
        if not isinstance(path, str) or path in observed_paths:
            return False
        observed_paths.add(path)
        file_path = Path(path)
        try:
            stat = file_path.stat()
        except OSError:
            return False
        observed = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
        }
        try:
            file_changed = any(
                int(item[name]) != value for name, value in observed.items()
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if file_changed:
            return False

        symlink_fields = {
            "symlink_target",
            "symlink_mtime_ns",
            "symlink_ctime_ns",
        }
        cached_symlink_fields = symlink_fields.intersection(item)
        if cached_symlink_fields and cached_symlink_fields != symlink_fields:
            return False
        if "symlink_unreadable" in item:
            return None
        if file_path.is_symlink() != bool(cached_symlink_fields):
            return False
        if cached_symlink_fields:
            try:
                link_stat = file_path.lstat()
                link_target = file_path.readlink()
            except OSError:
                return False
            try:
                symlink_changed = (
                    str(link_target) != item["symlink_target"]
                    or int(link_stat.st_mtime_ns)
                    != int(item["symlink_mtime_ns"])
                    or int(link_stat.st_ctime_ns)
                    != int(item["symlink_ctime_ns"])
                )
            except (TypeError, ValueError, OverflowError):
                return False
            if symlink_changed:
                return False
    return cached_master in observed_paths


def _metadata_array(metadata: dict[str, object]) -> np.ndarray:
    """Store metadata as one JSON string so reopening never needs pickle."""
    return np.asarray(json.dumps(metadata, separators=(",", ":")), dtype=np.str_)


def _save_cache(products: ScreeningResult, path: Path) -> None:
    """Write the products as one ``.npz`` that reopens without pickle.

    Exact count maps are stored as uint64 and only as a complete set, so a
    reopened cache never mixes exact counts with rounded float products.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata_json": _metadata_array(products.metadata),
        "mean_dp": np.asarray(products.mean_dp, dtype=np.float32),
        "bright_field": np.asarray(products.bright_field, dtype=np.float32),
        "dark_field": np.asarray(products.dark_field, dtype=np.float32),
        "dpc_phase": np.asarray(products.dpc_phase, dtype=np.float32),
        "com_row": np.asarray(products.com_row, dtype=np.float32),
        "com_col": np.asarray(products.com_col, dtype=np.float32),
    }
    exact_values = [
        getattr(products, name) for name in _EXACT_COUNT_PRODUCT_FIELDS
    ]
    if any(value is not None for value in exact_values):
        if any(value is None for value in exact_values):
            raise ValueError(
                "Exact screening cache products must include total_intensity, "
                "annular_bright_field, and annular_dark_field together."
            )
        expected_shape = np.asarray(products.bright_field).shape
        for name, value in zip(
            _EXACT_COUNT_PRODUCT_FIELDS,
            exact_values,
            strict=True,
        ):
            array = np.asarray(value)
            if array.dtype != np.dtype(np.uint64):
                raise TypeError(
                    f"{name} must preserve exact uint64 detector counts; "
                    f"got {array.dtype}."
                )
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {array.shape}, expected scan shape "
                    f"{expected_shape}."
                )
            payload[name] = array
    np.savez(path, **payload)


def _prepare_cache(path: Path, master: Path) -> ScreeningResult | None:
    """Reopen a current cache for an unchanged source, or return ``None`` to rebuild."""
    if not path.exists():
        return None
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].reshape(())))
        # A cache without the phase predates phase caching; rebuild it.
        if not _cache_matches(metadata, master) or "dpc_phase" not in data.files:
            return None
        parameters = metadata["parameters"]
        com_row = np.asarray(data["com_row"], dtype=np.float32)
        exact_present = [
            name in data.files for name in _EXACT_COUNT_PRODUCT_FIELDS
        ]
        if any(exact_present) and not all(exact_present):
            return None
        exact_products: dict[str, np.ndarray | None] = dict.fromkeys(_EXACT_COUNT_PRODUCT_FIELDS)
        if all(exact_present):
            for name in _EXACT_COUNT_PRODUCT_FIELDS:
                array = np.asarray(data[name])
                if array.dtype != np.dtype(np.uint64) or array.shape != com_row.shape:
                    return None
                exact_products[name] = array
        return ScreeningResult(
            mean_dp=np.asarray(data["mean_dp"], dtype=np.float32),
            bright_field=np.asarray(data["bright_field"], dtype=np.float32),
            dark_field=np.asarray(data["dark_field"], dtype=np.float32),
            dpc_phase=np.asarray(data["dpc_phase"], dtype=np.float32),
            com_row=com_row,
            com_col=np.asarray(data["com_col"], dtype=np.float32),
            probe_center=tuple(float(value) for value in parameters["center"]),
            probe_radius=float(parameters["radius_px"]),
            rotation_deg=float(parameters["rotation_deg"]),
            transposed=bool(parameters["transposed"]),
            metadata=metadata,
            cache_path=path,
            from_cache=True,
            elapsed_s=time.perf_counter() - started,
            **exact_products,
        )
