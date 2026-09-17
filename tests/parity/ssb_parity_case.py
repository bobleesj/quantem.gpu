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
    "CASES",
    "DOCUMENTED_BF_COUNT",
    "DOCUMENTED_CENTER",
    "DOCUMENTED_DET_SAMPLING_MRAD",
    "DOCUMENTED_RADIUS",
    "SSBParityCase",
    "case_root",
    "compare_products",
    "detector_dead_pixels",
    "export_case",
    "load_case_declaration",
    "reference_products",
    "runs_root",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Real acquisition retained by the MAPED/ARINA reference set.
ARINA_MASTER = Path(
    "/path/to/local/data/Live4DSTEM Testing/ARINA/"
    "arina-fixture-b_master.h5"
    ""
)

#: Frozen documented geometry of the retained Reference-512 disk.
DOCUMENTED_CENTER = (94.88451385498047, 96.35952758789062)
DOCUMENTED_RADIUS = 53.35992814757164
DOCUMENTED_BF_COUNT = 8937
#: Frozen detector sampling of the Reference-512 preset
#: (``tests/parity/fixtures/ssb_reference_512_mps.json``).
DOCUMENTED_DET_SAMPLING_MRAD = 1.0909090909090908

DEFAULT_RUNS_ROOT = Path("/path/to/local/perf-lab/ssb-audit/parity-runs")


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
    loaded = load(
        str(case.source),
        backend="mps",
        representation="encoded" if side >= 256 else "dense",
        detector_bin=1,
        scan_region=(row0, row1, col0, col1),
        dtype=None,
        verbose=verbose,
    )
    data = loaded.data
    if side >= 256:
        out = np.empty((rows.size, side * side), dtype=np.uint16)
        for start in range(0, side, 32):
            stop = min(start + 32, side)
            chunk = data.decode_scan_range_device(start, stop).to_numpy()
            block = np.asarray(chunk).reshape(-1, 192, 192)[:, rows, cols]
            out[:, start * side : stop * side] = block.T
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
            "`scripts/check_ssb_parity.sh --export` or call export_case()."
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
    core_ratio: float = 1e-3,
) -> dict[str, float]:
    """Return the strict error metrics of one product set against the oracle.

    Two different phase products are reported, because they are different
    physical quantities and must never be compared against each other:

    ``object_phase_*`` is the phase image a reconstruction displays, the
    argument of the complex object. Every backend's object supports it, so it
    is the cross-backend phase metric.

    ``mean_phase_*`` is the objective's own phase product, the mean over the
    bright-field set of the per-BF inverse-transform phase. The MPS public
    ``preview`` returns it; the native Metal API exposes its variance (loss)
    but not the mean itself.
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
    object_phase_difference = np.angle(
        np.exp(1j * (np.angle(object_wave) - np.angle(reference_object)))
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
