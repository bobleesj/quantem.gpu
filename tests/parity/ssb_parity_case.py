"""Exact SSB parity cases, declaration export, and metric comparison.

The strict gate compares three implementations at identical inputs:

- the public Python ``quantem.gpu.SSB`` MPS path;
- the native Swift ``MetalSSBKernels`` path through a standalone harness; and
- the independent float64 reference in :mod:`ssb_double_reference`.

Identical inputs are guaranteed by exporting one authoritative artifact per
case: the production exact BF-column companion (``(BF, scan)`` exact integer
counts plus its ``snapshots/cal.json`` declaration). The Python MPS path loads
it through ``SSB.open(..., backend="mps")``; the native harness loads the same
``.u16`` payload and the same declared BF pixel list. Nothing is binned,
cropped, scaled, or approximated in any measured path.

Generated data lives outside the repository under the runs root
(``QUANTEM_SSB_PARITY_RUNS``, default
``/path/to/local/perf-lab/ssb-audit/parity-runs``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ssb_double_reference import (
    brightfield_disk,
    electron_wavelength_angstrom,
    ssb_reference,
)

__all__ = [
    "ARINA_MASTER",
    "artifact_root",
    "CASES",
    "DOCUMENTED_BF_COUNT",
    "DOCUMENTED_CENTER",
    "DOCUMENTED_DET_SAMPLING_MRAD",
    "DOCUMENTED_RADIUS",
    "SSBParityCase",
    "case_root",
    "compare_products",
    "detector_dead_pixels",
    "ensure_case_directory",
    "export_case",
    "load_case_declaration",
    "reference_products",
    "runs_root",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Real acquisition retained by the MAPED/ARINA reference set.
ARINA_MASTER = Path(
    os.environ.get("QUANTEM_SSB_PARITY_SOURCE", "arina-fixture-b_master.h5")
).expanduser()

#: Frozen documented geometry of the retained Reference-512 disk.
DOCUMENTED_CENTER = (94.88451385498047, 96.35952758789062)
DOCUMENTED_RADIUS = 53.35992814757164
DOCUMENTED_BF_COUNT = 8937
#: Frozen detector sampling of the Reference-512 preset
#: (``tests/parity/fixtures/ssb_reference_512_mps.json``).
DOCUMENTED_DET_SAMPLING_MRAD = 1.0909090909090908

DEFAULT_RUNS_ROOT = Path.home() / "perf-lab/ssb-audit/parity-runs"


def detector_dead_pixels(source: Path) -> np.ndarray:
    """Return the acquisition's hardware dead pixels as an ``(N, 2)`` array."""

    from quantem.gpu.io._metadata import read_pixel_mask

    mask = read_pixel_mask(str(source))
    if mask is None:
        return np.zeros((0, 2), dtype=np.int64)
    return np.argwhere(np.asarray(mask) != 0).astype(np.int64)


def runs_root() -> Path:
    """Return the generated-data root for parity runs."""

    return Path(os.environ.get("QUANTEM_SSB_PARITY_RUNS", DEFAULT_RUNS_ROOT))


@dataclass(frozen=True)
class SSBParityCase:
    """One fixed-input strict parity case.

    ``scan_region`` is a ``(row0, row1, col0, col1)`` real-acquisition crop
    that must have a supported square side (128, 256, or 512).
    """

    name: str
    description: str
    scan_region: tuple[int, int, int, int]
    bf_center: tuple[float, float] = DOCUMENTED_CENTER
    bf_radius: float = DOCUMENTED_RADIUS
    det_sampling_mrad: float = DOCUMENTED_DET_SAMPLING_MRAD
    voltage_kV: float = 300.0
    semiangle_mrad: float = 30.0
    scan_sampling_A: float = 0.264
    rotation_angle_deg: float = 158.88268568029937
    aberrations: tuple[tuple[float, float, float], ...] = (
        (73.18188621458395, 14.020962948808993, 0.4700365259977606),
        (-42.5, 27.3, 1.13),
    )
    #: Settings retained for the report but excluded from the strict gate.
    #: ``(0, 50, 0)`` is the frozen optimizer start. Its aberration phase is a
    #: pure quadratic form in ``k``, so the SSB normalization denominator
    #: ``gamma`` has a zero curve through the discrete scan grid and the
    #: objective is not a stable float32 quantity there (measured
    #: single-precision floor 1.5e-2 relative L2). No implementation can
    #: reproduce the double-precision objective at that setting; measuring it
    #: as a precision gate would be meaningless either way.
    diagnostic_aberrations: tuple[tuple[float, float, float], ...] = (
        (0.0, 50.0, 0.0),
    )
    #: Name of the case whose exported artifact this case measures, when the
    #: two share the same acquisition region and bright-field selection and
    #: differ only in aberration settings. ``None`` means "use this case's own
    #: name". Sharing never duplicates or regenerates the payload, so two cases
    #: can never disagree about the bytes they measure.
    artifact_name: str | None = None
    source: Path = ARINA_MASTER

    @property
    def scan_side(self) -> int:
        """Square scan side of this case."""

        row0, row1, col0, col1 = (int(v) for v in self.scan_region)
        if row1 - row0 != col1 - col0:
            raise ValueError(
                f"Case {self.name!r} must use a square scan region, got "
                f"{self.scan_region}."
            )
        return row1 - row0

    @property
    def bf_rows_cols(self) -> tuple[np.ndarray, np.ndarray]:
        """Pinned full-disk BF pixel list, dead pixels excluded.

        The documented automatic policy is "positive-count pixels inside the
        calibrated disk". The one hardware dead pixel inside the disk
        (``(78, 74)``) reads zero everywhere, so the exact same set is reached
        either by excluding the acquisition's ``pixel_mask`` entries or by
        dropping zero-count pixels. This function excludes the declared mask
        so the pinned list never depends on a counts threshold.
        """

        rows, cols = brightfield_disk(
            self.detector_shape_hint,
            self.bf_center,
            self.bf_radius,
        )
        dead = detector_dead_pixels(self.source)
        if dead.size == 0:
            return rows, cols
        dead_index = set(zip(dead[:, 0].tolist(), dead[:, 1].tolist()))
        keep = np.fromiter(
            (
                (int(row), int(col)) not in dead_index
                for row, col in zip(rows.tolist(), cols.tolist())
            ),
            dtype=bool,
            count=rows.size,
        )
        return rows[keep].astype(np.int32), cols[keep].astype(np.int32)

    @property
    def detector_shape_hint(self) -> tuple[int, int]:
        """Native detector shape of the ARINA reference acquisition."""

        return (192, 192)


CASES: dict[str, SSBParityCase] = {
    "arina-128-full-disk": SSBParityCase(
        name="arina-128-full-disk",
        description=(
            "128x128 real ARINA scan crop, full documented BF disk, exact "
            "uint16 counts; exercises the streamed/dynamic MPS loss kernels "
            "and the compact-inactive storage path."
        ),
        scan_region=(0, 128, 0, 128),
    ),
    "arina-128-inner-disk": SSBParityCase(
        name="arina-128-inner-disk",
        description=(
            "128x128 real ARINA scan crop, inner radius-24 disk (1810 BF, all "
            "aperture-active); exercises the cached-geometry MPS correction "
            "kernel."
        ),
        scan_region=(0, 128, 0, 128),
        bf_radius=24.0,
    ),
    "arina-128-recorded-c10": SSBParityCase(
        name="arina-128-recorded-c10",
        description=(
            "128x128 real ARINA crop sweeping the C10 values of the recorded "
            "cached-versus-streamed loss disagreement "
            "(experiments/20260913-ssb-loss-diagnosis: C10 = 0, 55, "
            "155.96977). C12 and phi12 are zero here: the record names only "
            "C10 and does not state its scan region, so this case reproduces "
            "the recorded configuration family rather than the recorded loss "
            "values themselves. Reuses the full-disk artifact, so the only "
            "variable is the aberration phase."
        ),
        scan_region=(0, 128, 0, 128),
        artifact_name="arina-128-full-disk",
        aberrations=(
            (0.0, 0.0, 0.0),
            (55.0, 0.0, 0.0),
            (155.96977, 0.0, 0.0),
        ),
    ),
    "arina-512-full-disk": SSBParityCase(
        name="arina-512-full-disk",
        description=(
            "Full 512x512 real ARINA acquisition, full documented BF disk, "
            "exact uint16 counts; the production scale."
        ),
        scan_region=(0, 512, 0, 512),
    ),
}


def case_root(case: SSBParityCase) -> Path:
    """Return the generated case directory."""

    return runs_root() / "strict-parity" / case.name


def artifact_root(case: SSBParityCase) -> Path:
    """Return the directory holding this case's exported exact inputs.

    A case may reuse another case's artifact when only its aberration settings
    differ, so the measured bytes are identical by construction rather than by
    re-extraction.
    """

    return case_root(case) if case.artifact_name is None else (
        runs_root() / "strict-parity" / case.artifact_name
    )


def _source_dir(case: SSBParityCase) -> Path:
    return case_root(case) / "source"


def _declaration(case: SSBParityCase) -> dict[str, object]:
    rows, cols = case.bf_rows_cols
    side = case.scan_side
    return {
        "bf_rows": [int(value) for value in rows],
        "bf_cols": [int(value) for value in cols],
        "bf_center": [float(case.bf_center[0]), float(case.bf_center[1])],
        "bf_radius_px": float(case.bf_radius),
        "detector_shape": [192, 192],
        "scan_region": {"shape": [side, side]},
        "bf_column_companion": True,
        "source_transport": "bf_columns",
        "bf_excluded_detector_pixels": [
            int(row) * 192 + int(col)
            for row, col in detector_dead_pixels(case.source)
        ],
    }


def ensure_case_directory(case: SSBParityCase) -> Path:
    """Return a case directory that holds ``case.json`` and the exact ``source``.

    A case that reuses another case's artifact gets its own ``case.json``
    (identical scientific inputs, its own aberration settings) and a symlink to
    the owner's ``source`` directory, so every consumer - the public MPS path
    and the standalone native harness alike - reads one copy of the bytes and
    the settings that belong to the case under test.
    """

    root = case_root(case)
    if case.artifact_name is None:
        return root
    owner = artifact_root(case)
    payload = owner / "source" / "bf_columns.u16"
    if not payload.is_file():
        raise FileNotFoundError(
            f"Case {case.name!r} reuses the artifact of {case.artifact_name!r}, "
            f"but {payload} does not exist. Export that case first: "
            f"python tests/parity/ssb_parity_gate.py --case {case.artifact_name} "
            "--export"
        )
    root.mkdir(parents=True, exist_ok=True)
    link = root / "source"
    if not link.exists():
        link.symlink_to(owner / "source")
    declaration = json.loads((owner / "case.json").read_text(encoding="utf-8"))
    declaration["name"] = case.name
    declaration["description"] = case.description
    declaration["aberrations"] = [list(values) for values in case.aberrations]
    (root / "case.json").write_text(
        json.dumps(declaration, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


def export_case(
    case: SSBParityCase,
    *,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    """Create the exact-input artifact tree for one case.

    Writes ``source/snapshots/cal.json``, ``source/manifest.json``, the exact
    ``source/bf_columns.u16`` payload, and a ``case.json`` record of every
    pinned scientific input. The payload is extracted from the real
    acquisition with the canonical MPS loader, then verified byte-exactly
    against a direct HDF5 read of the same frames where the file layout allows
    it (see :func:`verify_counts_against_hdf5`).
    """

    if case.artifact_name is not None:
        # A sharing case never re-extracts: its exact inputs are the owner's.
        return ensure_case_directory(case)
    root = case_root(case)
    source = _source_dir(case)
    snapshots = source / "snapshots"
    if (source / "bf_columns.u16").is_file() and not force:
        return root
    snapshots.mkdir(parents=True, exist_ok=True)
    rows, cols = case.bf_rows_cols
    side = case.scan_side
    dead = detector_dead_pixels(case.source)
    if dead.size:
        dead_index = set(zip(dead[:, 0].tolist(), dead[:, 1].tolist()))
        selected = set(zip(rows.tolist(), cols.tolist()))
        overlap = sorted(selected & dead_index)
        if overlap:
            raise ValueError(
                f"Case {case.name!r} selects hardware dead pixels {overlap}; "
                "the documented disk excludes them."
            )
    counts = extract_bf_columns(case, rows, cols, verbose=verbose)
    payload_bytes = int(counts.nbytes)
    if counts.dtype != np.dtype(np.uint16):
        raise TypeError(
            f"Exact ARINA counts must stay uint16, got {counts.dtype}."
        )
    column_sums = np.asarray(counts, dtype=np.int64).reshape(rows.size, -1).sum(axis=1)
    zero_columns = int(np.count_nonzero(column_sums == 0))
    if zero_columns:
        raise ValueError(
            f"Case {case.name!r} selected {zero_columns} BF pixels with zero "
            "counts across the whole scan; the documented policy keeps only "
            "positive-count pixels."
        )
    exactness = verify_counts_against_hdf5(case, counts, rows, cols)
    if not exactness["exact"]:
        raise ValueError(
            f"Case {case.name!r} exported counts that are not the acquisition's "
            f"raw uint16 values: {exactness['mismatching_scan_positions']} scan "
            f"positions differ from a direct h5py read (first: "
            f"{exactness['first_mismatches']}). The gate must measure exact "
            "counts; fix the extraction instead of loosening the check."
        )
    (source / "bf_columns.u16").write_bytes(
        np.ascontiguousarray(counts).tobytes(order="C")
    )
    declaration = _declaration(case)
    (snapshots / "cal.json").write_text(
        json.dumps(declaration, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema": "quantem-ssb-parity-case/1",
        "calibration": "snapshots/cal.json",
        "case": case.name,
        "description": case.description,
        "source": {
            "path": str(case.source),
            "scan_region": list(case.scan_region),
            "bf_columns": {
                "kind": "bf_columns",
                "order": "bf,scan",
                "path": "bf_columns.u16",
                "encoding": "u16",
                "dtype": "uint16",
                "shape": [int(rows.size), side * side],
                "scan_shape": [side, side],
                "detector_shape": [192, 192],
                "bytes": payload_bytes,
                "bytes_per_bf": side * side * 2,
                "bits_per_value": 16,
                "max_value": int(counts.max()),
                "detector_bin": 1,
            },
        },
    }
    (source / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    scientific = {
        "name": case.name,
        "description": case.description,
        "source": str(case.source),
        "scan_region": list(case.scan_region),
        "scan_shape": [side, side],
        "detector_shape": [192, 192],
        "bf_center": list(case.bf_center),
        "bf_radius_px": float(case.bf_radius),
        "bf_count": int(rows.size),
        "documented_bf_count": DOCUMENTED_BF_COUNT,
        "voltage_kV": case.voltage_kV,
        "semiangle_mrad": case.semiangle_mrad,
        "scan_sampling_A": case.scan_sampling_A,
        "rotation_angle_deg": case.rotation_angle_deg,
        "aberrations": [list(values) for values in case.aberrations],
        "counts_dtype": "uint16",
        "counts_max": int(counts.max()),
        "counts_sum": int(np.asarray(counts, dtype=np.int64).sum()),
        "counts_column_sums_exact": [int(value) for value in column_sums],
        "brightfield_center": list(case.bf_center),
        "brightfield_radius_px": float(case.bf_radius),
        "brightfield_policy": "documented_full_disk_minus_hardware_dead_pixels",
    }
    scientific.update(shared_geometry(case, source))
    scientific["counts_exactness"] = exactness
    scientific["metal_geometry"] = metal_geometry(scientific)
    scientific["metal_geometry_unrotated"] = {
        "brightfieldKX": scientific["brightfield_kx_unrotated"],
        "brightfieldKY": scientific["brightfield_ky_unrotated"],
        "brightfieldAlphaSquared": scientific["brightfield_alpha_squared_unrotated"],
        "brightfieldAperture": scientific["brightfield_aperture_unrotated"],
        "brightfieldCos2Phi": scientific["brightfield_cos2phi_unrotated"],
        "brightfieldSin2Phi": scientific["brightfield_sin2phi_unrotated"],
    }
    (root / "case.json").write_text(
        json.dumps(scientific, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


def acquisition_scan_shape(case: SSBParityCase) -> tuple[int, int] | None:
    """Return the source acquisition's native scan shape, if discoverable."""

    from quantem.gpu.io import inspect

    info = inspect(str(case.source))
    shape = getattr(info, "scan_shape", None)
    if shape is None:
        return None
    return (int(shape[0]), int(shape[1]))


def scan_region_is_whole_acquisition(case: SSBParityCase) -> bool:
    """Return whether the case region covers the complete native acquisition.

    The native (encoded/H5-to-ANS) loader preserves whole acquisitions only, so
    a case that asks for the entire scan must not pass a selection. Deciding
    this from the file's own scan shape keeps the exported bytes identical to a
    full-acquisition load instead of silently narrowing it.
    """

    native = acquisition_scan_shape(case)
    if native is None:
        return False
    row0, row1, col0, col1 = (int(v) for v in case.scan_region)
    return (row0, row1, col0, col1) == (0, native[0], 0, native[1])


def extract_bf_columns(
    case: SSBParityCase,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Return exact ``(BF, scan, scan)`` uint16 counts from the acquisition."""

    from quantem.gpu.io import load

    row0, row1, col0, col1 = (int(v) for v in case.scan_region)
    side = case.scan_side
    whole = scan_region_is_whole_acquisition(case)
    loaded = load(
        str(case.source),
        backend="mps",
        representation="encoded" if side >= 256 else "dense",
        detector_bin=1,
        scan_region=None if whole else (row0, row1, col0, col1),
        dtype=None,
        hot_pixel_correction="none",
        verbose=verbose,
    )
    data = loaded.data
    if side >= 256:
        # ``decode_scan_range_device`` addresses contiguous *flat scan
        # positions*, not scan rows, and returns ``(stop - first, 192, 192)``.
        # The payload is stored ``(BF, scan)``, so each decoded block
        # transposes into its own flat slice.
        scan_count = side * side
        out = np.empty((rows.size, scan_count), dtype=np.uint16)
        step = 32 * side
        for first in range(0, scan_count, step):
            stop = min(first + step, scan_count)
            chunk = data.decode_scan_range_device(first, stop).to_numpy()
            block = np.asarray(chunk).reshape(stop - first, 192, 192)[:, rows, cols]
            out[:, first:stop] = block.T
        columns = out.reshape(rows.size, side, side)
    else:
        dense = np.asarray(data).reshape(-1, 192, 192)
        columns = np.ascontiguousarray(dense[:, rows, cols].T).reshape(
            rows.size, side, side
        )
    release = getattr(data, "release", None)
    if callable(release):
        release()
    return np.ascontiguousarray(columns)


def verify_counts_against_hdf5(
    case: SSBParityCase,
    columns: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> dict[str, object]:
    """Verify exported counts byte-exactly against raw HDF5 frames.

    This reads the master acquisition with h5py alone - no ``quantem.gpu``
    loader, no hot-pixel correction, no conversion - and compares the raw
    uint16 detector values of the pinned BF pixels at every scan position with
    the payload the gate measures. It is the independent proof that the gate's
    inputs are the acquisition's exact raw counts.

    The scan is raster order over the acquisition's own scan width, so one scan
    row of a region is one contiguous frame range and can be read in bulk.
    """

    import h5py
    import hdf5plugin  # noqa: F401 - registers the acquisition's bitshuffle filter

    side = case.scan_side
    row0, _row1, col0, col1 = (int(v) for v in case.scan_region)
    native_rows, native_cols = acquisition_scan_shape(case) or (side, side)
    if col1 > native_cols:
        raise ValueError(
            f"Case {case.name!r} asks for columns {col0}:{col1}, beyond the "
            f"acquisition width {native_cols}."
        )
    expected = columns.reshape(rows.size, side, side)
    mismatching_positions = 0
    first_mismatch: list[int] | None = None
    with h5py.File(str(case.source), "r") as handle:
        group = handle["/entry/data"]
        datasets = []
        offset = 0
        for name in sorted(group):
            dataset = group[name]
            if not hasattr(dataset, "shape"):
                continue
            datasets.append((offset, offset + int(dataset.shape[0]), dataset))
            offset += int(dataset.shape[0])
        total_frames = offset
        for scan_row in range(side):
            absolute_row = row0 + scan_row
            first = absolute_row * native_cols + col0
            last = absolute_row * native_cols + col1
            if last > total_frames:
                raise ValueError(
                    f"Case {case.name!r} needs frames {first}:{last} but the "
                    f"acquisition holds {total_frames}."
                )
            block = np.empty((last - first, 192, 192), dtype=np.uint16)
            for low, high, dataset in datasets:
                take_low, take_high = max(first, low), min(last, high)
                if take_high <= take_low:
                    continue
                block[take_low - first : take_high - first] = dataset[
                    take_low - low : take_high - low
                ]
            measured = np.ascontiguousarray(block[:, rows, cols].T)
            reference = expected[:, scan_row, :]
            if not np.array_equal(measured, reference):
                differing = np.flatnonzero((measured != reference).any(axis=1))
                mismatching_positions += int(differing.size)
                if first_mismatch is None:
                    first_mismatch = [int(value) for value in differing[:8]]
    return {
        "exact": mismatching_positions == 0,
        "mismatching_scan_positions": int(mismatching_positions),
        "first_mismatches": first_mismatch,
        "source": str(case.source),
        "read": "h5py_raw_uint16_no_loader_full_scan",
        "frames_compared": int(side * side),
        "values_compared": int(side * side * rows.size),
    }


def _reciprocal_geometry(
    case: SSBParityCase,
    *,
    rotate: bool,
) -> dict[str, np.ndarray]:
    """Return the documented float32 probe geometry for the pinned BF pixels.

    The arithmetic mirrors the production derivation exactly: the reciprocal
    coordinates are formed in double precision from the calibrated center and
    detector sampling, the calibrated scan/detector rotation is applied in
    double precision, and only then is the result rounded to float32. The
    derived probe terms then follow the documented float32 formula
    (``alpha^2 = (|k| * wavelength)^2``, ``cos2phi``/``sin2phi`` from the
    rounded coordinates, and the one-sampling-wide linear disk edge).
    """

    rows, cols = case.bf_rows_cols
    center = case.bf_center
    wavelength = electron_wavelength_angstrom(case.voltage_kV)
    det_rad = case.det_sampling_mrad * 1e-3
    kx = (rows.astype(np.float64) - center[0]) * det_rad / wavelength
    ky = (cols.astype(np.float64) - center[1]) * det_rad / wavelength
    if rotate and case.rotation_angle_deg:
        angle = np.radians(-float(case.rotation_angle_deg))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        kx, ky = kx * cos_a + ky * sin_a, -kx * sin_a + ky * cos_a
    kx64, ky64 = kx.copy(), ky.copy()
    kx32 = kx.astype(np.float32)
    ky32 = ky.astype(np.float32)
    r2 = kx32.astype(np.float64) ** 2 + ky32.astype(np.float64) ** 2
    r = np.sqrt(r2)
    alpha2 = (r * wavelength) ** 2
    inv_r2 = np.where(r2 > 1e-30, 1.0 / np.where(r2 > 0, r2, 1.0), 0.0)
    cos2 = (kx32**2 - ky32**2) * inv_r2
    sin2 = 2.0 * kx32 * ky32 * inv_r2
    denom = np.sqrt((kx32 * det_rad) ** 2 + (ky32 * det_rad) ** 2) / np.where(
        r > 1e-15, r, 1.0
    )
    edge = np.where(
        denom > 1e-15,
        (case.semiangle_mrad * 1e-3 - r * wavelength) / np.where(denom > 0, denom, 1.0)
        + 0.5,
        1.0,
    )
    aperture = np.clip(edge, 0.0, 1.0)
    return {
        "kx_exact": kx64,
        "ky_exact": ky64,
        "kx": kx32,
        "ky": ky32,
        "alpha_squared": alpha2.astype(np.float32),
        "aperture": aperture.astype(np.float32),
        "cos2phi": cos2.astype(np.float32),
        "sin2phi": sin2.astype(np.float32),
    }


def shared_geometry(case: SSBParityCase, source: Path) -> dict[str, object]:
    """Return the declared geometry shared by all three implementations.

    ``brightfield_kx_exact`` / ``brightfield_ky_exact`` are the double-precision
    reciprocal coordinates of the pinned BF pixels with the calibrated rotation
    applied in double precision. They are the oracle's inputs.
    ``brightfield_kx`` / ``brightfield_ky`` and the derived
    ``brightfield_alpha_squared`` / ``brightfield_aperture`` /
    ``brightfield_cos2phi`` / ``brightfield_sin2phi`` are the same quantities
    rounded to the float32 precision the production backends carry. Feeding the
    float32 pair to the oracle isolates geometry rounding from transform and
    phase arithmetic.
    """

    rows, _cols = case.bf_rows_cols
    side = case.scan_side
    step = (float(case.scan_sampling_A), float(case.scan_sampling_A))
    rotated = _reciprocal_geometry(case, rotate=True)
    unrotated = _reciprocal_geometry(case, rotate=False)
    qx = np.fft.fftfreq(side, step[0]).astype(np.float32)
    qy = np.fft.fftfreq(side, step[1]).astype(np.float32)
    counts = np.memmap(
        source / "bf_columns.u16",
        dtype=np.uint16,
        mode="r",
        shape=(rows.size, side, side),
    )
    dc_exact = float(
        np.asarray(counts, dtype=np.float64).reshape(rows.size, -1).sum(axis=1).mean()
    )
    del counts
    return {
        "wavelength_angstrom": float(electron_wavelength_angstrom(case.voltage_kV)),
        "det_sampling_mrad": float(case.det_sampling_mrad),
        "det_sampling_rad": [
            float(case.det_sampling_mrad * 1e-3),
            float(case.det_sampling_mrad * 1e-3),
        ],
        "scan_sampling_A": [step[0], step[1]],
        "dc_value_exact": float(dc_exact),
        "dc_value_float32": float(np.float32(dc_exact)),
        "brightfield_kx_exact": rotated["kx_exact"].tolist(),
        "brightfield_ky_exact": rotated["ky_exact"].tolist(),
        "brightfield_kx": rotated["kx"].tolist(),
        "brightfield_ky": rotated["ky"].tolist(),
        "brightfield_alpha_squared": rotated["alpha_squared"].tolist(),
        "brightfield_aperture": rotated["aperture"].tolist(),
        "brightfield_cos2phi": rotated["cos2phi"].tolist(),
        "brightfield_sin2phi": rotated["sin2phi"].tolist(),
        "brightfield_kx_unrotated": unrotated["kx"].tolist(),
        "brightfield_ky_unrotated": unrotated["ky"].tolist(),
        "brightfield_alpha_squared_unrotated": unrotated["alpha_squared"].tolist(),
        "brightfield_aperture_unrotated": unrotated["aperture"].tolist(),
        "brightfield_cos2phi_unrotated": unrotated["cos2phi"].tolist(),
        "brightfield_sin2phi_unrotated": unrotated["sin2phi"].tolist(),
        "qx_by_row": qx.tolist(),
        "qy_by_column": qy.tolist(),
        "detector_dead_pixels": [
            [int(row), int(col)] for row, col in detector_dead_pixels(case.source)
        ],
        "active_bf_count": int(np.count_nonzero(rotated["aperture"] > 0.0)),
    }


def metal_geometry(case_meta: dict[str, object]) -> dict[str, object]:
    """Return the ``MetalSSBGeometry`` JSON consumed by the Swift harness.

    The calibrated rotation is already folded into these float32 arrays, so the
    harness declares ``referenceRotationDegrees = 0`` and the native engine
    applies no second rotation.
    """

    return {
        "brightfieldKX": case_meta["brightfield_kx"],
        "brightfieldKY": case_meta["brightfield_ky"],
        "brightfieldAlphaSquared": case_meta["brightfield_alpha_squared"],
        "brightfieldAperture": case_meta["brightfield_aperture"],
        "brightfieldCos2Phi": case_meta["brightfield_cos2phi"],
        "brightfieldSin2Phi": case_meta["brightfield_sin2phi"],
        "qxByRow": case_meta["qx_by_row"],
        "qyByColumn": case_meta["qy_by_column"],
        "wavelengthAngstroms": np.float32(case_meta["wavelength_angstrom"]).item(),
        "semiangleRadians": np.float32(
            float(case_meta["semiangle_mrad"]) * 1e-3
        ).item(),
        "angularSamplingYRadians": np.float32(
            case_meta["det_sampling_rad"][0]
        ).item(),
        "angularSamplingXRadians": np.float32(
            case_meta["det_sampling_rad"][1]
        ).item(),
        "dcValue": [
            float(np.float32(case_meta["dc_value_exact"])),
            float(np.float32(0.0)),
        ],
        "referenceRotationDegrees": float(np.float32(0.0)),
    }


def load_case_declaration(case: SSBParityCase) -> dict[str, object]:
    """Return the exported ``case.json`` for one case."""

    path = case_root(case) / "case.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Case {case.name!r} has no exported inputs at {path}. Run "
            "`scripts/check_ssb_parity.sh --cpu-oracle --export` or call export_case()."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def reference_products(
    case: SSBParityCase,
    meta: dict[str, object],
    c10: float,
    c12: float,
    phi12: float,
    *,
    geometry: str = "exact",
    dtype=np.float64,
) -> tuple[object, dict[str, object]]:
    """Return the independent reference result for one aberration setting.

    ``geometry="exact"`` gives the oracle its double-precision reciprocal
    coordinates. ``geometry="declared"`` gives it the float32 coordinates the
    production backends carry, which separates geometry rounding from the
    transform and phase arithmetic.
    """

    rows, cols = case.bf_rows_cols
    side = case.scan_side
    if geometry == "exact":
        kx_key, ky_key = "brightfield_kx_exact", "brightfield_ky_exact"
    elif geometry == "declared":
        kx_key, ky_key = "brightfield_kx", "brightfield_ky"
    else:
        raise ValueError(
            f"geometry must be 'exact' or 'declared', got {geometry!r}."
        )
    counts = np.memmap(
        _source_dir(case) / "bf_columns.u16",
        dtype=np.uint16,
        mode="r",
        shape=(rows.size, side, side),
    )
    det_rad = tuple(float(v) for v in meta["det_sampling_rad"])
    result = ssb_reference(
        np.asarray(counts),
        rows,
        cols,
        kx=np.asarray(meta[kx_key], dtype=np.float64),
        ky=np.asarray(meta[ky_key], dtype=np.float64),
        qx=np.asarray(meta["qx_by_row"], dtype=np.float64),
        qy=np.asarray(meta["qy_by_column"], dtype=np.float64),
        wavelength=float(meta["wavelength_angstrom"]),
        semiangle_rad=float(meta["semiangle_mrad"]) * 1e-3,
        det_sampling_rad=det_rad,
        c10=c10,
        c12=c12,
        phi12=phi12,
        dc_value=complex(float(meta["dc_value_exact"]), 0.0),
        dtype=dtype,
        chunk_bf=512 if side <= 128 else 64,
        rotation_angle_deg=0.0,
    )
    return result, {
        "c10": float(c10),
        "c12": float(c12),
        "phi12": float(phi12),
        "geometry": geometry,
        "dtype": str(np.dtype(dtype)),
        **result.summary(),
    }


def compare_products(
    reference_object: np.ndarray,
    reference_loss: float,
    reference_phase: np.ndarray,
    object_wave: np.ndarray,
    loss: float | None,
    mean_phase: np.ndarray | None,
    *,
    object_phase: np.ndarray | None = None,
    core_ratio: float = 1e-3,
) -> dict[str, float]:
    """Return the strict error metrics of one product set against the oracle.

    Two different phase products are reported, because they are different
    physical quantities and must never be compared against each other:

    ``object_phase_*`` is the phase image a reconstruction displays, the
    argument of the complex object. Every backend's object supports it, so it
    is the cross-backend phase metric. Pass ``object_phase`` to measure a
    backend's own phase kernel (the native Metal ``ssb_object_phase`` output);
    omit it to derive the phase from ``object_wave``.

    ``mean_phase_*`` is the objective's own phase product, the mean over the
    bright-field set of the per-BF inverse-transform phase. The MPS public
    ``preview`` returns it; the native Metal API exposes its variance (loss)
    but not the mean itself, so Metal entries leave ``mean_phase`` as ``None``
    rather than comparing a different quantity under the same name.
    """

    reference_object = np.asarray(reference_object, dtype=np.complex128)
    object_wave = np.asarray(object_wave, dtype=np.complex128)
    if object_wave.shape != reference_object.shape:
        raise ValueError(
            f"Object shape {object_wave.shape} does not match reference "
            f"{reference_object.shape}."
        )
    difference = object_wave - reference_object
    scale = float(np.max(np.abs(reference_object)))
    norm = float(np.linalg.norm(reference_object.reshape(-1)))
    metrics = {
        "object_scale": scale,
        "object_max_abs_error": float(np.max(np.abs(difference))),
        "object_max_abs_error_relative": float(np.max(np.abs(difference)) / scale),
        "object_relative_l2": float(np.linalg.norm(difference.reshape(-1)) / norm),
    }
    if mean_phase is not None:
        phase_difference = np.angle(
            np.exp(1j * (np.asarray(mean_phase, dtype=np.float64) - reference_phase))
        )
        magnitude = np.abs(reference_object)
        core = magnitude >= core_ratio * scale
        metrics.update(
            {
                "mean_phase_max_error_radians": float(np.max(np.abs(phase_difference))),
                "mean_phase_max_error_radians_core": float(
                    np.max(np.abs(phase_difference[core]))
                ),
                "mean_phase_core_pixels": int(np.count_nonzero(core)),
                "mean_phase_rms_error_radians_core": float(
                    np.sqrt(np.mean(phase_difference[core] ** 2))
                ),
            }
        )
    magnitude = np.abs(reference_object)
    core = magnitude >= core_ratio * scale
    measured_object_phase = (
        np.angle(object_wave)
        if object_phase is None
        else np.asarray(object_phase, dtype=np.float64)
    )
    object_phase_difference = np.angle(
        np.exp(1j * (measured_object_phase - np.angle(reference_object)))
    )
    metrics.update(
        {
            "object_phase_max_error_radians": float(
                np.max(np.abs(object_phase_difference))
            ),
            "object_phase_max_error_radians_core": float(
                np.max(np.abs(object_phase_difference[core]))
            ),
            "object_phase_rms_error_radians_core": float(
                np.sqrt(np.mean(object_phase_difference[core] ** 2))
            ),
            "phase_core_pixels": int(np.count_nonzero(core)),
            "object_magnitude_ratio_min": float(
                float(np.min(magnitude[core])) / scale if scale else float("nan")
            ),
        }
    )
    if loss is not None:
        metrics.update(
            {
                "loss": float(loss),
                "reference_loss": float(reference_loss),
                "loss_abs_error": float(abs(loss - reference_loss)),
                "loss_relative_error": float(
                    abs(loss - reference_loss) / abs(reference_loss)
                ),
            }
        )
    return metrics


def write_reference(case: SSBParityCase, meta: dict[str, object]) -> Path:
    """Compute and store the float64 oracle for every case aberration."""

    root = case_root(case)
    reference_dir = root / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, (c10, c12, phi12) in enumerate(case.aberrations):
        result, summary = reference_products(case, meta, c10, c12, phi12)
        np.save(reference_dir / f"object-{index}.npy", result.object_wave)
        np.save(reference_dir / f"phase-{index}.npy", result.mean_phase)
        summaries.append(summary)
    (reference_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    return reference_dir
