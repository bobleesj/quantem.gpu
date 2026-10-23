"""Backend-neutral scientist-facing SSB workflow.

This module owns the only public stateful SSB entry point. Backend modules own
device preparation and kernels, but they do not define a second user API.
"""

import hashlib
import math
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import torch

from quantem.gpu import detector, io
from quantem.gpu.device import resolve
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.geometry.frames import opposite_rotation_deg
from quantem.gpu.formats.qem.metadata import BRIGHT_FIELD_DISK_KEY
from quantem.gpu.formats.qem.snapshot import INTEGER_CODEC
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.io import metadata_display as labels
from quantem.gpu.io.metadata_display import calibration_rows
from quantem.gpu.io.representation import DataRepresentation
from quantem.gpu.optics.physics import wavelength_A_from_kV, wrap_degrees
from quantem.gpu.ssb import series
from quantem.gpu.ssb._execution import cancellation_scope
from quantem.gpu.ssb.branches import DARK_FIELD_INNER_RADIUS, RotationBranches, compare_branches
from quantem.gpu.ssb.brightfield import bright_field_crop_region, crop_bright_field
from quantem.gpu.ssb.contract import FitObjective, RefineMethod, SSBProtocol
from quantem.gpu.ssb.higher_order_fit import (
    assess,
    from_coordinates,
    refine_higher_order,
    to_coordinates,
    validate_terms,
)
from quantem.gpu.ssb.mps.backend import MpsSSBBackend
from quantem.gpu.ssb.mps.frames import EncodedBrightFieldCrop, MpsBfColumnFrames, find_bf_columns
from quantem.gpu.ssb.mps.optimizer import newton_supported, standard_bounds
from quantem.gpu.ssb.mps.reconstruct import fit_loss_tiles
from quantem.gpu.ssb.persistence import (
    RESULT_SCHEMA,
    json_value,
    load_result,
    result_paths,
    save_result,
    portable_source,
    software_signature,
    source_identity,
)
from quantem.gpu.ssb.results import SSBResult, host_array
from quantem.gpu.ssb.trials import (
    choose_trials,
    label_trials,
    plot_trials,
    resolve_trial_count,
    select_records,
)
from quantem.gpu.ssb.units import (
    ENGINE_PER_NM,
    aberrations_from_engine,
    aberrations_to_engine,
    higher_order_arrays,
    refine_lock_to_engine,
    result_from_engine,
    sampling_pair,
    search_ranges_to_engine,
    validate_aberrations,
)


def _counted_electrons(dtype: str | None) -> bool:
    """Whether the session's source holds integer detector counts (electrons), whose noise is Poisson."""
    try:
        return dtype is not None and np.dtype(str(dtype).removeprefix("torch.")).kind in "iu"
    except TypeError:
        return False


def _default_phase_estimator(aberrations: dict[str, float]) -> str:
    """Use coherent reconstruction for every aberration on both backends."""
    return "phase_of_mean"


def _browser_qem_source(data: object) -> dict[str, object] | None:
    """Retain the identity of an unchanged, native-count QEM acquisition."""
    if not isinstance(data, Dataset4dstemGPU):
        return None
    metadata = data.metadata
    if (data.ndim != 4 or data.representation is DataRepresentation.DENSE
            or metadata.get("container") != "quantem.qem"
            or metadata.get("resident_profile") != INTEGER_CODEC
            or str(data.dtype).removeprefix("torch.") not in {"uint8", "uint16"}
            or not metadata.get("source_path")
            or not metadata.get("qem_header_sha256")):
        return None
    path = Path(metadata["source_path"]).expanduser().resolve()
    try:
        stat = path.stat()
        with path.open("rb") as handle:
            if handle.read(56)[24:].hex() != metadata["qem_header_sha256"]:
                return None
    except OSError:
        return None
    return {
        "path": path, "scan_shape": tuple(data.shape[:2]),
        "detector_shape": tuple(data.shape[2:]),
        "dtype": str(data.dtype).removeprefix("torch."),
        "signature": (stat.st_size, stat.st_mtime_ns),
        "header_sha256": metadata["qem_header_sha256"],
        "valid_pixels": np.asarray(data.data.valid_pixels, dtype=bool).copy(),
    }


def _estimate_rotation(data, scan_shape, flipped: bool | None) -> tuple[float, bool]:
    """Scan-detector rotation, known modulo 180 degrees, and whether it is flipped, from the centre-of-mass curl.

    A thin phase object has a curl-free centre-of-mass field once the scan and detector axes agree; the angle is the
    exact minimum of the mean squared curl (``dpc.workflow.find_optimal_rotation``, the search DPC and screening use).
    The curl cannot tell an angle from the angle plus 180 degrees; ``find_aberrations`` settles that from the sign of
    the atom columns. A given ``flipped`` is kept and only the angle is fitted.
    """
    from quantem.gpu.dpc.workflow import center_of_mass, find_optimal_rotation

    try:
        if isinstance(data, MpsBfColumnFrames):
            com_row, com_col = _bright_field_center_of_mass(data)
        else:
            com_row, com_col = center_of_mass(data, scan_shape=scan_shape)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(
            "SSB could not estimate the scan-detector rotation from this input; pass "
            "scan_detector_rotation_deg (and scan_detector_rotation_flipped) explicitly."
        ) from exc
    _, _, angle, flipped = find_optimal_rotation(com_row, com_col, scan_detector_rotation_flipped=flipped)
    return float(angle), bool(flipped)


def _bright_field_center_of_mass(frames: "MpsBfColumnFrames") -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass of each pattern over its bright-field pixels, the only ones exact column storage keeps.

    The bright-field disk carries the DPC shift, so its centre of mass fixes the rotation as the full pattern's does.
    """
    columns = frames._columns  # (bright-field pixels, scan positions)
    rows = np.asarray(frames.selection.rows, np.float64)
    cols = np.asarray(frames.selection.cols, np.float64)
    weights = np.stack([np.ones_like(rows), rows, cols]).astype(np.float32)
    moments = np.empty((3, columns.shape[1]), np.float64)
    for start in range(0, columns.shape[1], 8192):
        moments[:, start:start + 8192] = weights @ np.asarray(columns[:, start:start + 8192], np.float32)
    total = np.where(moments[0] > 0, moments[0], 1.0)
    shape = frames.scan_shape
    return ((moments[1] / total).reshape(shape).astype(np.float32),
            (moments[2] / total).reshape(shape).astype(np.float32))


def _announce_rotation(angle_deg: float, flipped: bool) -> None:
    """The one line a session prints when it estimated the rotation itself."""
    print(f"SSB: scan-detector rotation {angle_deg:.1f}°, rotation flipped: {'yes' if flipped else 'no'} "
          "(estimated from the centre-of-mass curl; find_aberrations settles the 180° choice).")


def _announce_detector_sampling(detector_sampling_mrad, semiangle_mrad: float, *, recordable: bool) -> None:
    """The one line a session prints when it estimated the detector sampling itself, and how to state it instead."""
    explicit = ("data.calibration.fit_detector_sampling() records it and shows the fit"
                if recordable else "give detector_sampling_mrad= to state it")
    print(f"SSB: detector step {float(detector_sampling_mrad[0]):.4f} mrad per pixel, estimated from the bright-field "
          f"disk edge at the {float(semiangle_mrad):g} mrad semiangle (detector.fit_bright_field_disk); {explicit}.")


def _recorded_bright_field_center_px(metadata: dict[str, object]) -> tuple[float, float] | None:
    """The bright-field centre (row, col) ``data.calibration.fit_detector_sampling()`` recorded, or None."""
    disk = metadata.get(BRIGHT_FIELD_DISK_KEY)
    if not isinstance(disk, dict) or disk.get("bright_field_center_px") is None:
        return None
    return tuple(float(value) for value in disk["bright_field_center_px"])


def _source_calibration(
    metadata: dict[str, object], voltage_kV: float | None,
    semiangle_mrad: float | None, scan_sampling_A: tuple[float, float] | None,
) -> tuple[float, float, tuple[float, float]]:
    """Resolve physical inputs from explicit values, then acquisition metadata."""
    values = {"voltage_kV": voltage_kV, "semiangle_mrad": semiangle_mrad, "scan_sampling_A": scan_sampling_A}
    for name, value in values.items():
        if value is None:
            values[name] = metadata.get(name)
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(
            f"SSB acquisition metadata is missing {', '.join(missing)}. "
            "Pass those calibrated values to SSB(data, ...) or SSB.open(...); scan_sampling_A is in Angstrom, one "
            "number for both axes or (row, col). "
            "A diffraction disk fit cannot determine the real-space scan step."
        )
    return (float(values["voltage_kV"]), float(values["semiangle_mrad"]),
            sampling_pair("scan_sampling_A", values["scan_sampling_A"]))


def _tilt_constraints(
    search_ranges: dict[str, object] | None, fixed_aberrations: dict[str, float] | None,
) -> tuple[dict[str, object], dict[str, float] | None]:
    """The joint (tilt=True) fit's constraints as search bounds in engine units, and the fully fixed optics if any.

    The joint search bounds C10 by a range and C12 from zero to a maximum, and searches phi12 over its whole period.
    So it takes a ``C10_nm`` range or value, a ``C12_nm`` range starting at 0 (or C12_nm = 0, which leaves phi12
    without effect), and, in ``fixed_aberrations``, ``C10_nm`` alone, ``C12_nm=0`` alone, or all three coefficients.
    Anything else raises with what to pass instead.
    """
    ranges = dict(search_ranges or {})
    held = dict(fixed_aberrations or {})
    if set(held) == {"C10_nm", "C12_nm", "phi12_deg"}:
        if ranges:
            raise ValueError("fixed_aberrations holds all three coefficients; drop search_ranges, which has nothing to search.")
        return {}, held
    unknown = (set(ranges) | set(held)) - {"C10_nm", "C12_nm", "phi12_deg"}
    if unknown:
        raise ValueError(f"search_ranges and fixed_aberrations take C10_nm, C12_nm and phi12_deg; got {sorted(unknown)}.")
    both = set(ranges) & set(held)
    if both:
        raise ValueError(f"{sorted(both)} is in both search_ranges and fixed_aberrations; give each coefficient once.")
    for name, value in held.items():
        ranges[name] = float(value)
    bounds: dict[str, object] = {}
    if "phi12_deg" in ranges:
        raise ValueError(
            "tilt=True searches phi12_deg over its whole period and cannot bound or hold it alone. Hold all three "
            "with fixed_aberrations={'C10_nm': ..., 'C12_nm': ..., 'phi12_deg': ...} (and check_rotation=False), hold "
            "C12_nm=0 to remove astigmatism, or bound phi12_deg in a standard fit (tilt=False).")
    if "C10_nm" in ranges:
        value = ranges["C10_nm"]
        low, high = (value, value) if np.ndim(value) == 0 else value
        if not (math.isfinite(float(low)) and math.isfinite(float(high)) and float(low) <= float(high)):
            raise ValueError(f"C10_nm takes a finite (low, high) range in nm with low <= high, or one value; got {value!r}.")
        bounds["c10_range"] = (float(low) * ENGINE_PER_NM, float(high) * ENGINE_PER_NM)
    if "C12_nm" in ranges:
        value = ranges["C12_nm"]
        low, high = (0.0, 0.0) if np.ndim(value) == 0 and float(value) == 0.0 else (
            (None, None) if np.ndim(value) == 0 else value)
        if low is None or float(low) != 0.0 or not (math.isfinite(float(high)) and float(high) >= 0.0):
            raise ValueError(
                f"tilt=True bounds C12_nm from 0: give search_ranges={{'C12_nm': (0, maximum_nm)}} or hold it with "
                f"fixed_aberrations={{'C12_nm': 0.0}}; got {value!r}. A nonzero held C12_nm needs phi12_deg and "
                "C10_nm too: fixed_aberrations with all three and check_rotation=False.")
        bounds["c12_max"] = float(high) * ENGINE_PER_NM
    return bounds, None


def _standard_constraints(
    search_ranges: dict[str, object] | None, refine_lock: list[str] | None, fixed_aberrations: dict[str, float] | None,
) -> tuple[dict[str, object] | None, list[str] | None]:
    """A standard fit's (tilt=False) ``fixed_aberrations`` as what they mean there: held in every trial
    (a ``search_ranges`` value) and through the refinement (``refine_lock``)."""
    if not fixed_aberrations:
        return search_ranges, refine_lock
    unknown = set(fixed_aberrations) - {"C10_nm", "C12_nm", "phi12_deg"}
    if unknown:
        raise ValueError(f"fixed_aberrations takes C10_nm, C12_nm and phi12_deg; got {sorted(unknown)}.")
    if set(fixed_aberrations) == {"C10_nm", "C12_nm", "phi12_deg"}:
        raise ValueError("A standard fit (tilt=False) with all three coefficients held has nothing to fit: "
                         "reconstruct(aberrations=...) uses them as given.")
    both = set(search_ranges or {}) & set(fixed_aberrations)
    if both:
        raise ValueError(f"{sorted(both)} is in both search_ranges and fixed_aberrations; give each coefficient once.")
    ranges = {**(search_ranges or {}), **{name: float(value) for name, value in fixed_aberrations.items()}}
    lock = list(dict.fromkeys([*(refine_lock or []), *fixed_aberrations]))
    return ranges, lock


def _recorded_detector_sampling_mrad(metadata: dict[str, object], voltage_kV: float) -> tuple[float, float] | None:
    """The acquisition's recorded detector sampling in mrad per pixel, (row, col), or None.

    ``detector_sampling_mrad`` as recorded, else ``detector_sampling_inv_A`` (1/Angstrom per pixel, what
    ``data.calibration.detector_sampling_inv_A = ...`` sets) times the electron wavelength at ``voltage_kV``:
    an angle is ``lambda k``, so mrad per pixel = 1000 * lambda[A] * k[1/A].
    """
    if metadata.get("detector_sampling_mrad") is not None:
        return sampling_pair("detector_sampling_mrad", metadata["detector_sampling_mrad"])
    if metadata.get("detector_sampling_inv_A") is None:
        return None
    wavelength_A = wavelength_A_from_kV(voltage_kV)
    row, col = sampling_pair("detector_sampling_inv_A", metadata["detector_sampling_inv_A"])
    return (1e3 * wavelength_A * row, 1e3 * wavelength_A * col)


def _announce_rotation_branches(result: SSBResult) -> None:
    """The one line a fit prints when it compared both 180-degree branches of an estimated rotation."""
    if result.rotation_branches is not None:
        print(f"SSB: {result.rotation_branches.summary}")


def _dark_field_of(source, semiangle_mrad: float, detector_sampling_mrad, center_px=None) -> np.ndarray:
    """Dark field of a full-detector source from just outside the bright-field disk to the detector edge.

    The inner radius is ``DARK_FIELD_INNER_RADIUS`` times the bright-field radius, the semiangle over the detector
    step (the larger of the two axes' radii); without a detector step, the disk edge of the mean pattern. The disk
    centre is ``center_px`` (full-detector pixels) when known, else fitted.
    """
    if detector_sampling_mrad is None:
        radius_px = detector.disk_edge_radius(detector.mean(source))
    else:
        radius_px = float(semiangle_mrad) / float(min(sampling_pair("detector_sampling_mrad", detector_sampling_mrad)))
    geometry = {} if center_px is None else {"bright_field_center_px": center_px, "bright_field_edge_radius_px": radius_px}
    return np.asarray(detector.df(source, inner_radius_px=DARK_FIELD_INNER_RADIUS * radius_px, **geometry),
                      dtype=np.float32)


def _announce_pixel_correction(provenance: dict[str, object] | None) -> None:
    """The one line ``SSB.open`` prints when it replaced flagged detector pixels before fitting."""
    correction = (provenance or {}).get("hot_pixel_correction")
    if not isinstance(correction, dict) or not correction.get("applied") or not correction.get("pixel_count"):
        return
    count = int(correction["pixel_count"])
    window = correction.get("largest_window_px") or 3
    median = "their 3 x 3 neighbor median" + (f" (windows up to {window} x {window} inside clusters)" if window > 3 else "")
    method = {"median": median, "zero": "zero"}.get(correction.get("method"), correction.get("method"))
    print(f"SSB: {count} flagged detector pixel{'s' if count != 1 else ''} replaced by {method} "
          "(recorded in source_provenance); to keep the recorded values, load with "
          'io.load(path, hot_pixel_correction="none") and use SSB(data).')


def _on_scan_gpu(method):
    """Run a session method with the scan's GPU as the current CuPy device (CUDA sessions).

    CuPy allocates and launches on its current device, and the FFT kernel sets are compiled per device, so a
    session on a second GPU would otherwise compute with the first GPU's kernels on memory it does not own.
    """
    @wraps(method)
    def on_scan_gpu(self, *args, **kwargs):
        with nullcontext() if self._device is None else cp.cuda.Device(self._device):
            return method(self, *args, **kwargs)
    return on_scan_gpu


if TYPE_CHECKING:
    from quantem.gpu.ssb.rotation_defocus import RotationDefocusComparison
    import pandas as pd

def _keeps_crop_encoded(data: Dataset4dstemGPU, backend: str) -> bool:
    """Whether an MPS session keeps this acquisition's bright-field crop encoded (``EncodedBrightFieldCrop``).

    Integer counts are read exactly from the encoded source while G(q, k) is built; float sources keep the decoded
    float32 crop their FFT input needs.
    """
    from quantem.gpu.formats.float8 import is_float8

    return backend == "mps" and not is_float8(data.dtype) and np.dtype(data.dtype).kind == "u"


def _streams_scan(data: Dataset4dstemGPU, backend: str) -> bool:
    """Select large integer ANS scans without changing smaller/float workflows."""
    if backend != "cuda":
        return False
    from quantem.gpu.resident.cuda.counts import StreamedCounts

    return max(data.shape[:2]) > 1024 or (
        data.shape[:2] == (1024, 1024) and type(data.data) is StreamedCounts
    )


class SSB:
    """Single-sideband ptychography of one 4D-STEM acquisition on CUDA or MPS.

    SSB recovers the object phase from the bright-field disk: each bright-field detector pixel sees the sample through
    a slightly different probe tilt, and once the probe aberrations are corrected for every pixel their phase images
    agree. ``find_aberrations`` fits C10_nm, C12_nm and phi12_deg (with the sample tilt and a depth spread by default) by
    maximising the least-squares agreement of that model with the data; ``tilt=False`` holds the tilt and depth spread
    at 0, and ``tilt=False, objective="phase_variance"`` is the older search that minimizes the phase variance over
    bright-field pixels instead. ``reconstruct`` and ``preview`` apply known aberrations and report that phase
    variance as the diagnostic ``loss``. The full
    automatically detected bright-field disk, float32 real and complex64 object storage are fixed.

    C10/C12 reconstruction averages corrected complex waves before taking the
    phase (``phase_of_mean``) on CUDA and MPS. Both support finer output sampling
    through ``upsampling_factor``. Bright-field selection, loss reduction order
    and Nelder-Mead settings remain backend-specific, so fitted values and images
    need not agree bit for bit.

    Parameters
    ----------
    data
        The dataset ``io.load`` returns, or a NumPy, CuPy or Torch array of counts, 4D ``(scan_row, scan_col,
        det_row, det_col)`` or 3D ``(scan, det_row, det_col)`` with ``scan_shape``. Of a dataset only the
        bright-field crop is read: CUDA decodes it; MPS keeps integer counts encoded and decodes their bright-field
        columns a few scan rows at a time while it builds the transform, so keep the dataset open while the session
        is in use.
    backend : {"auto", "cuda", "mps"}
        Accelerator; there is no CPU fallback.
    voltage_kV : float, optional
        Accelerating voltage in kV; defaults to the loaded dataset's metadata.
    semiangle_mrad : float, optional
        Probe convergence semiangle in mrad; defaults to the loaded dataset's metadata.
    scan_sampling_A : float or (float, float), optional
        Scan step in Angstrom, (row, col), or one number for both axes; defaults to the loaded dataset's metadata.
        The session stores and reports it as the (row, col) pair.
        These three values remain required for an array without metadata.
    scan_shape : (int, int), optional
        Scan shape of a 3D frame stack.
    detector_sampling_mrad : float or (float, float), optional
        Detector sampling in mrad per pixel, (row, col), or one number for both axes. When omitted, ``semiangle_mrad`` divided by the radius in pixels at which
        the bright-field disk of the full-detector mean pattern falls to half its plateau (``disk_edge_radius``).
        A loaded dataset's recorded detector sampling takes priority over that
        estimate (``detector_sampling_inv_A``, 1/Angstrom per pixel, is converted with the electron wavelength);
        explicit keyword values take priority over metadata.
    aberrations : dict, optional
        Starting ``C10_nm``, ``C12_nm`` (nm) and ``phi12_deg`` (degrees).
    scan_detector_rotation_deg : float, optional
        Physical scan-to-detector rotation in degrees; kept in [0, 360). Omitted, the acquisition's recorded rotation
        is used, else it is estimated from the curl of the centre-of-mass field, which knows it modulo 180 degrees,
        and one line says so; ``find_aberrations`` (and the automatic fit of ``reconstruct()``) then fits both
        180-degree branches and keeps the one with positive phase at the atomic columns
        (``compare_rotation_branches``). A given or recorded rotation is never moved. Give your calibrated angle
        to skip the estimate and the second fit.
    scan_detector_rotation_flipped : bool, optional
        True when the detector's row and column axes are swapped relative to the scan, which no rotation can
        produce. Omitted, it is estimated with the rotation, or False when the rotation is given. Detector
        calibration and reported centres remain in the original (row, col) frame.
    bright_field_intensity_threshold : float
        Fraction of the mean-pattern maximum a bright-field pixel must exceed.
    bright_field_radius_px : float, optional
        Bright-field radius in detector pixels; the detected disk when omitted.
    source_path : str, optional
        The file the array came from, recorded with saved results.
    bright_field_center_px : (float, float), optional
        Bright-field centre (row, col) in pixels of the full detector; detected when omitted.
    scan_region : (int, int, int, int), optional
        ``(row_start, row_stop, col_start, col_stop)`` of a dataset ``io.load`` returns: the session holds those scan
        positions alone, so only their bright-field columns are decoded and only their transform is built (a fit of a
        192 x 192 region of a 512 x 512 scan costs a seventh of the whole scan's). The bright-field disk is still found
        on the whole scan's mean pattern; the rotation and the detector sampling are the caller's to give from the whole
        scan.
    mean_pattern : array, optional
        The whole scan's mean pattern (``detector.mean(data)``) when the caller has it already, so an encoded dataset
        does not compute it a second time.

    An encoded dataset stays open and the caller's. A result saved from it with ``find_aberrations(save_dir=...)`` is
    identified by the data (shape, dtype, the mean pattern's hash, two integer sums of the decoded counts and the scan
    region), with no file to inspect, and reused by a later session on the same data.

    Examples
    --------
    >>> ssb = SSB(patterns, voltage_kV=300, semiangle_mrad=30,
    ...           scan_sampling_A=(0.99, 0.99), detector_sampling_mrad=(0.6081, 0.6081))
    >>> aberrations = ssb.find_aberrations()
    >>> result = ssb.reconstruct(aberrations)

    Load and inspect calibration separately before opening an interactive view:

    >>> data = io.load("acquisition.qem")
    >>> data.metadata_table()
    >>> ssb = SSB(data, scan_detector_rotation_deg=12.3)
    >>> # data stays open for other algorithms until data.close().
    """

    def __init__(
        self,
        data: object,
        *,
        backend: Literal["auto", "cuda", "mps"] = "auto",
        voltage_kV: float | None = None,
        semiangle_mrad: float | None = None,
        scan_sampling_A: float | tuple[float, float] | None = None,
        scan_shape: tuple[int, int] | None = None,
        detector_sampling_mrad: float | tuple[float, float] | None = None,
        aberrations: dict[str, float] | None = None,
        scan_detector_rotation_deg: float | None = None,
        scan_detector_rotation_flipped: bool | None = None,
        bright_field_intensity_threshold: float = 0.0,
        bright_field_radius_px: float | None = None,
        source_path: str | None = None,
        bright_field_center_px: tuple[float, float] | None = None,
        scan_region: tuple[int, int, int, int] | None = None,
        mean_pattern=None,
    ) -> None:
        supplied = {
            "voltage_kV": voltage_kV, "semiangle_mrad": semiangle_mrad,
            "scan_sampling_A": scan_sampling_A, "detector_sampling_mrad": detector_sampling_mrad,
            "scan_detector_rotation_deg": scan_detector_rotation_deg,
        }
        self.backend = resolve(backend)
        source_dataset = data if isinstance(data, Dataset4dstemGPU) else None
        self._browser_qem = _browser_qem_source(source_dataset)
        metadata = {} if source_dataset is None else source_dataset.metadata
        voltage_kV, semiangle_mrad, scan_sampling_A = _source_calibration(
            metadata, voltage_kV, semiangle_mrad, scan_sampling_A)
        self._detector_sampling_recorded = False
        detector_sampling_estimated = False
        if source_dataset is not None:
            if detector_sampling_mrad is None:
                detector_sampling_mrad = _recorded_detector_sampling_mrad(metadata, voltage_kV)
                self._detector_sampling_recorded = detector_sampling_mrad is not None
                detector_sampling_estimated = detector_sampling_mrad is None
            if bright_field_center_px is None:
                # the centre data.calibration.fit_detector_sampling() recorded: the one SSB's own rule finds
                bright_field_center_px = _recorded_bright_field_center_px(metadata)
            if source_path is None:
                source_path = metadata.get("source_path")
        self.scan_detector_rotation_source = "given"
        if scan_detector_rotation_deg is None and metadata.get("scan_detector_rotation_deg") is not None:
            scan_detector_rotation_deg = float(metadata["scan_detector_rotation_deg"])
            if scan_detector_rotation_flipped is None:
                scan_detector_rotation_flipped = bool(metadata.get("scan_detector_rotation_flipped", False))
            self.scan_detector_rotation_source = "metadata"
        if scan_detector_rotation_deg is None:
            scan_detector_rotation_deg, scan_detector_rotation_flipped = _estimate_rotation(
                data, scan_shape, scan_detector_rotation_flipped)
            self.scan_detector_rotation_source = "estimated"
            _announce_rotation(scan_detector_rotation_deg, scan_detector_rotation_flipped)
        elif scan_detector_rotation_flipped is None:
            scan_detector_rotation_flipped = False
        self._streamed = None
        self._owned_dataset = None
        # first detector (row, col) of the data the session holds: nonzero when only the bright-field crop was decoded
        self._crop_origin = (0, 0)
        self.scan_region = None if scan_region is None else tuple(int(value) for value in scan_region)
        # what a saved result of a session on an encoded dataset is filed under: the data themselves, there is no file
        self.resident_identity: dict[str, object] | None = None
        if isinstance(data, Dataset4dstemGPU):
            if data.ndim != 4:
                raise ValueError("SSB needs one 4D acquisition; select a dataset from the series first.")
            if data.representation is not DataRepresentation.DENSE:
                if self.scan_region is None and _streams_scan(data, self.backend):
                    from quantem.gpu.ssb.cuda.streamed import StreamedSSB

                    if data.device is None or data.device.type != "cuda":
                        raise ValueError("CUDA SSB needs the acquisition resident on a CUDA device.")
                    with cp.cuda.Device(data.device.index):
                        self._streamed = StreamedSSB(
                            data, mean_pattern=mean_pattern, center=bright_field_center_px,
                            radius=bright_field_radius_px, threshold=bright_field_intensity_threshold,
                            semiangle_mrad=semiangle_mrad, detector_sampling_mrad=detector_sampling_mrad,
                            scan_detector_rotation_flipped=scan_detector_rotation_flipped,
                        )
                    bright_field_center_px = self._streamed.center
                    bright_field_radius_px = self._streamed.radius
                    detector_sampling_mrad = self._streamed.detector_sampling_mrad
                else:
                    data, bright_field_center_px, bright_field_radius_px, detector_sampling_mrad, mean_pattern = self._decode_resident(
                        data, bright_field_center_px, bright_field_radius_px, detector_sampling_mrad,
                        bright_field_intensity_threshold, semiangle_mrad, mean_pattern,
                        identify=source_path is None,
                    )
                if detector_sampling_estimated:
                    _announce_detector_sampling(detector_sampling_mrad, semiangle_mrad, recordable=True)
            elif self.scan_region is not None:
                row_start, row_stop, col_start, col_stop = self.scan_region
                data = data.data[row_start:row_stop, col_start:col_stop]
            else:
                data = data.data
        elif scan_region is not None:
            raise ValueError("scan_region selects scan positions of a dataset io.load returns; slice an array before passing it.")
        # The backends read dense arrays, exact BF columns, or a borrowed large
        # ANS source whose detector columns are decoded during reconstruction.
        backend_array = cp.ndarray if self.backend == "cuda" else MpsBfColumnFrames | EncodedBrightFieldCrop
        if self._streamed is None and not isinstance(data, np.ndarray | backend_array | torch.Tensor):
            raise TypeError(
                f"SSB reads the dataset io.load returns or a NumPy, CuPy or Torch array; got "
                f"{type(data).__name__}. Pass the loaded dataset itself, SSB(data, ...), not its "
                "encoded storage data.data, or open the file with SSB.open(path, ...)."
            )
        if scan_shape is not None and data.ndim == 3 and not isinstance(data, backend_array):
            rows, cols = int(scan_shape[0]), int(scan_shape[1])
            if rows * cols != int(data.shape[0]):
                raise ValueError(
                    f"scan_shape={scan_shape} describes {rows * cols} frames, but data "
                    f"contains {data.shape[0]}."
                )
            data = data.reshape(rows, cols, data.shape[1], data.shape[2])
        # the full-detector source whose dark field locates the atomic columns of the 180-degree branch test: the
        # caller's dataset, or a 4D array; SSB.open and region sessions set their own
        self._dark_field: np.ndarray | None = None
        self._dark_field_source: object = source_dataset if source_dataset is not None else (
            data if self._streamed is None and not isinstance(data, MpsBfColumnFrames) and data.ndim == 4 else None)
        self._dark_field_region: tuple[int, int, int, int] | None = None
        self._data = (
            data.transpose_detector() if isinstance(data, MpsBfColumnFrames) else data.swapaxes(-1, -2)
        ) if scan_detector_rotation_flipped and self._streamed is None else data
        self._device = None
        if self.backend == "cuda":
            if self._streamed is not None:
                self._device = self._streamed.device
            elif isinstance(data, cp.ndarray):
                self._device = data.device.id
            elif isinstance(data, torch.Tensor) and data.is_cuda:
                self._device = data.device.index
            else:
                self._device = cp.cuda.runtime.getDevice()
        self._scan_shape = scan_shape
        self.voltage_kV = float(voltage_kV)
        self.semiangle_mrad = float(semiangle_mrad)
        self.scan_sampling_A = sampling_pair("scan_sampling_A", scan_sampling_A)
        self.detector_sampling_mrad = None if detector_sampling_mrad is None else sampling_pair("detector_sampling_mrad", detector_sampling_mrad)
        self._aberrations_explicit = aberrations is not None
        self._aberrations = validate_aberrations(aberrations, higher_order=True)
        self.scan_detector_rotation_deg = wrap_degrees(scan_detector_rotation_deg, 360.0)
        self.scan_detector_rotation_flipped = bool(scan_detector_rotation_flipped)
        self.bright_field_intensity_threshold = float(bright_field_intensity_threshold)
        self.bright_field_radius_px = bright_field_radius_px
        self.bright_field_center_px = None if bright_field_center_px is None else (float(bright_field_center_px[0]), float(bright_field_center_px[1]))
        self.sample_tilt_mrad: tuple[float, float] | None = None
        self.depth_spread_nm: float | None = None
        self._compare_standard = True
        self.source_path = source_path
        self.calibration_path: str | None = None
        self.source_manifest_path: str | None = None
        self.source_storage_path = source_path
        self.source_kind: Literal["array", "resident", "detector", "bf_columns"] = "array" if self.resident_identity is None else "resident"
        self.source_dtype = str(data.dtype)
        self.source_bytes = int(data.logical_bytes if isinstance(data, Dataset4dstemGPU) else data.nbytes)
        self.source_detector_bin = 1
        self.source_provenance: dict[str, object] | None = None
        self.source_load_seconds: float | None = None
        if source_dataset is not None:
            self.source_kind = "resident"
            self.source_dtype = str(source_dataset.dtype)
            self.source_bytes = source_dataset.logical_bytes
            correction = source_dataset.metadata.get("hot_pixel_correction")
            self.source_provenance = None if correction is None else {"hot_pixel_correction": correction}
        self._compute: SSBProtocol | None = None
        self._reconstruction: SSBResult | None = None
        self.best_loss = float("inf")
        self.trial_history: list[dict[str, object]] = []
        self._aberration_search: SSBResult | None = None
        self._calibration_file = calibration_rows(metadata)
        self._calibration_supplied = {name for name, value in supplied.items() if value is not None}
        self._calibration_initial = {name: getattr(self, name) for name in supplied}

    @property
    def aberrations(self) -> dict[str, float]:
        """The session's aberrations: ``C10_nm`` / ``C12_nm`` in nm, ``phi12_deg`` in degrees (higher orders by name).

        They are known once given to ``SSB(aberrations=...)``, fitted by ``find_aberrations``, or assigned here, and
        ``reconstruct()`` then applies them instead of fitting. Assign a whole mapping to state a known or restored
        calibration (``ssb.aberrations = {"C10_nm": 15.8, "C12_nm": 4.3, "phi12_deg": 17.4}``); assign None to forget
        them, so the next ``reconstruct()`` fits again. ``find_aberrations`` starts its refinement from known
        aberrations.

        Examples
        --------
        >>> ssb.aberrations = saved["aberrations"]  # doctest: +SKIP
        >>> result = ssb.reconstruct()  # applies them; no fit  # doctest: +SKIP
        """
        return self._aberrations

    @aberrations.setter
    def aberrations(self, aberrations: dict[str, float] | None) -> None:
        self._aberrations = validate_aberrations(aberrations, higher_order=True)
        self._aberrations_explicit = aberrations is not None

    def calibration_table(self) -> "pd.DataFrame":
        """Compare loaded file calibration with the settings this session uses.

        The table retains the metadata seen at construction and distinguishes
        explicit keywords (even equal values), overridden values, defaults and
        probe estimates. It never changes the acquisition or starts a fit.
        A detector estimate is pending until the first backend preparation.
        ShowPtycho displays this table automatically when first shown.

        Returns
        -------
        pandas.DataFrame
            File value, value used, units and the source of each value used.
            File values are the loader's effective metadata, including saved
            calibration overrides; original source records stay in the dataset.
            Numbers are rounded to six significant figures for reading; the
            session's attributes (``ssb.detector_sampling_mrad``, ...) keep the
            full values.

        Examples
        --------
        >>> data = io.load("scan.qem")
        >>> ssb = SSB(data, scan_detector_rotation_deg=12.3)
        >>> ssb.calibration_table()
        """
        import pandas as pd

        rows = {}
        for name in ("voltage_kV", "semiangle_mrad", "scan_sampling_A", "detector_sampling_mrad",
                     "scan_detector_rotation_deg"):
            label = labels.CALIBRATION_LABELS[name]
            recorded = self._calibration_file[label]
            value = getattr(self, name)
            source = recorded["Metadata"]
            if name in self._calibration_supplied:
                source = labels.EXPLICIT_OVERRIDE
            elif name == "scan_detector_rotation_deg":
                source = {"estimated": labels.ESTIMATED_ROTATION,
                          "metadata": recorded["Metadata"]}.get(self.scan_detector_rotation_source, labels.DEFAULT)
            elif source == labels.NOT_RECORDED or (
                name == "detector_sampling_mrad" and recorded["Unit"] != labels.CALIBRATION_UNITS[name]
                and not self._detector_sampling_recorded
            ):
                source = labels.ESTIMATED
            initial = self._calibration_initial[name]
            if initial is not None and not np.array_equal(value, initial):
                source = labels.MODIFIED_IN_SESSION
            if name == "detector_sampling_mrad" and value is None:
                if self._compute is not None:
                    value = (self._compute._det_sampling if self.backend == "mps"
                             else self._compute.angular_sampling)
                    if self.scan_detector_rotation_flipped:
                        value = value[::-1]
                else:
                    value = labels.PENDING_PROBE_FIT
            if isinstance(value, (list, tuple)):
                value = tuple(value)
            rows[label] = {
                "File value": labels.display_value(recorded["Value"]), "File unit": recorded["Unit"],
                "Value used": labels.display_value(value),
                "Unit used": labels.CALIBRATION_UNITS[name] if name == "detector_sampling_mrad" else recorded["Unit"],
                "Source": source,
            }
        rotation = rows[labels.CALIBRATION_LABELS["scan_detector_rotation_deg"]]
        flipped = self._calibration_file.get(labels.CALIBRATION_LABELS["scan_detector_rotation_flipped"], {})
        rows[labels.CALIBRATION_LABELS["scan_detector_rotation_flipped"]] = {
            "File value": flipped.get("Value", labels.NOT_RECORDED), "File unit": "",
            "Value used": "yes" if self.scan_detector_rotation_flipped else "no", "Unit used": "",
            "Source": rotation["Source"] if rotation["Source"] != labels.MODIFIED_IN_SESSION else (
                labels.EXPLICIT_OVERRIDE if "scan_detector_rotation_deg" in self._calibration_supplied else labels.DEFAULT),
        }
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("SSB launch calibration")

    def _decode_resident(self, data, bright_field_center_px, bright_field_radius_px,
                         detector_sampling_mrad, threshold, semiangle_mrad, mean_pattern, *, identify=True):
        """Take one fit-sized bright-field crop and identify its exact resident evidence.

        MPS keeps an integer crop encoded (``EncodedBrightFieldCrop``): its columns are decoded only while G(q, k) is
        built, and its identity sums come from exact masked sums of the encoded counts. CUDA, and float sources on
        MPS, decode the crop.
        """
        shape, dtype = tuple(int(value) for value in data.shape), str(data.dtype)
        if _keeps_crop_encoded(data, self.backend):
            bright_field_center_px, bright_field_radius_px, calibration_radius, region, mean_pattern = bright_field_crop_region(
                data, threshold, bright_field_radius_px, calibrate_detector=detector_sampling_mrad is None,
                bright_field_center_px=bright_field_center_px, mean_pattern=mean_pattern,
            )
            self._crop_origin = (region[0], region[2])
            data = EncodedBrightFieldCrop(data, region, scan_region=self.scan_region)
        else:
            data, bright_field_center_px, bright_field_radius_px, calibration_radius, self._crop_origin, mean_pattern = crop_bright_field(
                data, self.backend, threshold, bright_field_radius_px,
                calibrate_detector=detector_sampling_mrad is None, bright_field_center_px=bright_field_center_px,
                scan_region=self.scan_region, mean_pattern=mean_pattern,
            )
        if detector_sampling_mrad is None:
            detector_sampling_mrad = (semiangle_mrad / calibration_radius,) * 2
        # File-backed sessions use source_signature, including the recorded
        # preprocessing. Their saved identity never uses resident checksums.
        if not identify:
            return data, bright_field_center_px, bright_field_radius_px, detector_sampling_mrad, mean_pattern
        if isinstance(data, EncodedBrightFieldCrop):
            # the same two sums of the per-pattern totals, modulo 2**64, without decoding the crop
            per_frame = data.scan_sums().reshape(-1)
            positions = np.arange(1, per_frame.size + 1, dtype=np.uint64)
        elif self.backend == "cuda":
            frames = data.reshape(data.shape[0] * data.shape[1], -1)
            per_frame = frames.sum(axis=1, dtype=cp.uint64)
            positions = cp.arange(1, per_frame.size + 1, dtype=cp.uint64)
        else:
            frames = data.reshape(data.shape[0] * data.shape[1], -1)
            per_frame = frames.sum(dim=1, dtype=torch.int64)
            positions = torch.arange(1, per_frame.numel() + 1, dtype=torch.int64, device=per_frame.device)
        self.resident_identity = {
            "shape": list(shape), "dtype": dtype,
            "mean_pattern_sha256": hashlib.sha256(np.ascontiguousarray(mean_pattern, dtype=np.float64).tobytes()).hexdigest(),
            "bright_field_sums": [int(per_frame.sum()) % 2**64, int((per_frame * positions).sum()) % 2**64],
        }
        return data, bright_field_center_px, bright_field_radius_px, detector_sampling_mrad, mean_pattern

    def _backend_input(self):
        """Return dense counts only for operations needing the <=1024 backend."""
        if self._streamed is None:
            return self._data
        data, _, _, _, self._crop_origin, _ = crop_bright_field(
            self._data, self.backend, self.bright_field_intensity_threshold,
            self.bright_field_radius_px,
            bright_field_center_px=self.bright_field_center_px,
            mean_pattern=self._streamed.mean_pattern,
        )
        return data.swapaxes(-1, -2) if self.scan_detector_rotation_flipped else data

    # one implementation in ``series``, bound here so the public spelling is ``SSB.reconstruct_series``
    reconstruct_series = classmethod(series.reconstruct_series)

    @classmethod
    def open(
        cls,
        source: str,
        *,
        backend: Literal["auto", "cuda", "mps"] = "auto",
        dtype: str | None = None,
        voltage_kV: float | None = None,
        semiangle_mrad: float | None = None,
        scan_sampling_A: float | tuple[float, float] | None = None,
        scan_shape: tuple[int, int] | None = None,
        detector_sampling_mrad: float | tuple[float, float] | None = None,
        aberrations: dict[str, float] | None = None,
        scan_detector_rotation_deg: float | None = None,
        scan_detector_rotation_flipped: bool | None = None,
        bright_field_intensity_threshold: float = 0.0,
        bright_field_radius_px: float | None = None,
        bright_field_center_px: tuple[float, float] | None = None,
        calibration: str | None = None,
        verbose: bool = False,
    ) -> Self:
        """Open one lossless 4D-STEM source and prepare an SSB session.

        Exact BF-column storage is chosen automatically when it is available;
        otherwise detector counts are loaded with the explicitly requested
        storage dtype. Leave ``dtype=None`` for native detector precision.
        This storage choice never changes the float32/complex64 optimization
        precision or scientific objective.

        GPU acquisitions use ANS-encoded storage.

        File-based SSB loads with ``io.load``'s default detector preprocessing:
        recorded invalid pixels receive the median of their valid neighbors on
        the GPU, and one line says how many.
        The original file is unchanged, and the correction is recorded in
        ``source_provenance``. To choose another policy, load with ``io.load``
        and pass that dataset to ``SSB(data)``.

        Omitted voltage, convergence semiangle and scan step come from the
        acquisition's calibrated metadata. Missing physical values raise an
        error rather than being guessed. Detector sampling uses an explicit
        value, then stored calibration, then the bright-field disk edge and
        the known convergence semiangle. Explicit values always take priority.

        Pass a saved ``bright_field_center_px``, ``bright_field_radius_px``
        and ``detector_sampling_mrad`` together when replaying a fit. The
        centre remains in the original full-detector ``(row, col)`` axes,
        including when ``scan_detector_rotation_flipped=True``.
        """

        supplied = {
            "voltage_kV": voltage_kV, "semiangle_mrad": semiangle_mrad,
            "scan_sampling_A": scan_sampling_A, "detector_sampling_mrad": detector_sampling_mrad,
            "scan_detector_rotation_deg": scan_detector_rotation_deg,
        }
        selected = resolve(backend)
        rotation_estimated = rotation_recorded = detector_sampling_recorded = False
        dark_field = None
        crop_origin = (0, 0)
        owned = None
        source_detector_bin = 1
        browser_qem = None
        # on MPS an exact BF-column export beside the source replaces decoding the detector
        frames = find_bf_columns(source, calibration, verbose=verbose) if selected == "mps" else None
        if frames is not None:
            if frames.det_bin > 1:
                # exact columns are the native counts; a binned export is a reduction and says so
                print(f"SSB: reading exported bright-field columns binned {frames.det_bin} x {frames.det_bin} on the "
                      f"detector ({frames.source_path}) instead of {Path(source).name}; move the export aside to fit "
                      "native detector pixels.")
            file_calibration = calibration_rows(frames.metadata)
            data = frames
            if scan_detector_rotation_deg is None:
                scan_detector_rotation_deg, scan_detector_rotation_flipped = _estimate_rotation(
                    frames, scan_shape, scan_detector_rotation_flipped)
                rotation_estimated = True
            voltage_kV, semiangle_mrad, scan_sampling_A = _source_calibration(
                frames.metadata, voltage_kV, semiangle_mrad, scan_sampling_A)
            source_kind = "bf_columns"
            source_storage_path = str(frames.source_path)
            source_dtype = str(frames.dtype)
            source_bytes = int(frames.nbytes)
            source_load_seconds = float(frames.load_seconds)
            source_detector_bin = frames.det_bin
            source_provenance = json_value(frames.source_provenance)
            if detector_sampling_mrad is None:
                # The backend's own rule for exact columns, applied here so the session holds the sampling the
                # backend then uses: the export's calibration, else its disk edge at the semiangle. A session left
                # without it stored a fit with no detector sampling, which could not be exported.
                detector_sampling_mrad = frames.detector_sampling_mrad or (semiangle_mrad / frames.selection.bright_field_edge_radius_px,) * 2
        else:
            load_started = time.perf_counter()
            loaded = io.load(
                source,
                backend=selected,
                scan_shape=scan_shape,
                dtype=dtype,
                verbose=verbose,
            )
            file_calibration = calibration_rows(loaded.metadata)
            browser_qem = _browser_qem_source(loaded)
            try:
                voltage_kV, semiangle_mrad, scan_sampling_A = _source_calibration(
                    loaded.metadata, voltage_kV, semiangle_mrad, scan_sampling_A)
                if detector_sampling_mrad is None:
                    detector_sampling_mrad = _recorded_detector_sampling_mrad(loaded.metadata, voltage_kV)
                    detector_sampling_recorded = detector_sampling_mrad is not None
                if bright_field_center_px is None:
                    bright_field_center_px = _recorded_bright_field_center_px(loaded.metadata)
                if scan_detector_rotation_deg is None and loaded.metadata.get("scan_detector_rotation_deg") is not None:
                    scan_detector_rotation_deg = float(loaded.metadata["scan_detector_rotation_deg"])
                    if scan_detector_rotation_flipped is None:
                        scan_detector_rotation_flipped = bool(loaded.metadata.get("scan_detector_rotation_flipped", False))
                    rotation_recorded = True
                if scan_detector_rotation_deg is None:
                    # the whole detector, before the bright-field crop below
                    scan_detector_rotation_deg, scan_detector_rotation_flipped = _estimate_rotation(
                        loaded, scan_shape, scan_detector_rotation_flipped)
                    rotation_estimated = True
                if _streams_scan(loaded, selected):
                    owned = loaded
                    data = loaded
                else:
                    if _keeps_crop_encoded(loaded, selected):
                        # the session owns the encoded acquisition and decodes its bright-field columns from it
                        bright_field_center_px, bright_field_radius_px, calibration_radius, region, _ = bright_field_crop_region(
                            loaded, bright_field_intensity_threshold, bright_field_radius_px,
                            calibrate_detector=detector_sampling_mrad is None,
                            bright_field_center_px=bright_field_center_px,
                        )
                        crop_origin = (region[0], region[2])
                        data = EncodedBrightFieldCrop(loaded, region)
                        owned = loaded
                    else:
                        data, bright_field_center_px, bright_field_radius_px, calibration_radius, crop_origin, _ = crop_bright_field(
                            loaded, selected, bright_field_intensity_threshold, bright_field_radius_px,
                            calibrate_detector=detector_sampling_mrad is None,
                            bright_field_center_px=bright_field_center_px,
                        )
                    if rotation_estimated:
                        # the 180-degree branch test locates atomic columns on the dark field, which the
                        # bright-field crop does not keep
                        dark_field = _dark_field_of(loaded, semiangle_mrad, detector_sampling_mrad or (
                            semiangle_mrad / calibration_radius,) * 2, bright_field_center_px)
                source_dtype = str(loaded.dtype)
                source_bytes = loaded.logical_bytes
                correction = loaded.metadata.get("hot_pixel_correction")
            finally:
                if owned is None:
                    loaded.close()
            if detector_sampling_mrad is None and (owned is None or isinstance(data, EncodedBrightFieldCrop)):
                detector_sampling_mrad = (semiangle_mrad / calibration_radius,) * 2
                _announce_detector_sampling(detector_sampling_mrad, semiangle_mrad, recordable=False)
            source_provenance = None if correction is None else {"hot_pixel_correction": correction}
            source_kind = "detector"
            source_storage_path = str(source)
            source_load_seconds = time.perf_counter() - load_started
        try:
            session = cls(
                data,
                backend=selected,
                voltage_kV=voltage_kV,
                semiangle_mrad=semiangle_mrad,
                scan_sampling_A=scan_sampling_A,
                scan_shape=scan_shape,
                detector_sampling_mrad=detector_sampling_mrad,
                aberrations=aberrations,
                scan_detector_rotation_deg=scan_detector_rotation_deg,
                scan_detector_rotation_flipped=scan_detector_rotation_flipped,
                bright_field_intensity_threshold=bright_field_intensity_threshold,
                bright_field_radius_px=bright_field_radius_px,
                source_path=str(source),
                bright_field_center_px=bright_field_center_px,
            )
        except Exception:
            if owned is not None:
                owned.close()
            raise
        if source_kind != "detector" or owned is None:
            # exported bright-field columns and a bright-field crop keep no dark field: the file is read again for it
            # when a branch comparison needs it (an estimated rotation computed it before the crop)
            session._dark_field_source = Path(source)
        elif isinstance(data, EncodedBrightFieldCrop):
            # an encoded crop's session owns the whole acquisition
            session._dark_field_source = owned
        session._dark_field = dark_field
        if rotation_recorded:
            session.scan_detector_rotation_source = "metadata"
        if rotation_estimated:
            session.scan_detector_rotation_source = "estimated"
            _announce_rotation(scan_detector_rotation_deg, scan_detector_rotation_flipped)
        _announce_pixel_correction(source_provenance)
        session._calibration_file = file_calibration
        session._detector_sampling_recorded = detector_sampling_recorded
        session._calibration_supplied = {name for name, value in supplied.items() if value is not None}
        session._owned_dataset = owned
        session._crop_origin = crop_origin
        session.source_kind = source_kind
        session.source_detector_bin = source_detector_bin
        session.source_provenance = source_provenance
        # exported BF columns name the calibration and manifest they were made with
        recorded = source_provenance or {}
        session.calibration_path = calibration or recorded.get("calibration_path")
        session.source_manifest_path = recorded.get("manifest_path")
        session.source_storage_path = source_storage_path
        session.source_dtype = source_dtype
        session.source_bytes = source_bytes
        session.source_load_seconds = source_load_seconds
        session._browser_qem = browser_qem
        return session

    @_on_scan_gpu
    def select_region(
        self,
        scan_region: tuple[int, int, int, int],
        *,
        aberrations: dict[str, float] | None = None,
        scan_detector_rotation_deg: float | None = None,
    ) -> Self:
        """Open an independent fit session over existing scan evidence.

        Dense resident counts are borrowed without reopening or decoding their
        file. The detector calibration and full bright-field disk are shared;
        fitting or closing the returned session leaves this session unchanged.
        Encoded sources decode the requested positions through their existing
        loader. Rectangular MPS regions retain every measured position and use
        the backend's ordinary mean-pattern padding.

        Parameters
        ----------
        scan_region : tuple of int
            ``(row_start, row_stop, col_start, col_stop)`` relative to this scan.
        aberrations : dict, optional
            Starting optics in the usual nm/degree units; defaults to this
            session's optics. These parameters remain free in a joint fit.
        scan_detector_rotation_deg : float, optional
            Shared scan-to-detector rotation; defaults to this session's value.

        Returns
        -------
        SSB
            A separately owned fitting session. Use it as a context manager.

        Examples
        --------
        >>> with ssb.select_region((32, 160, 64, 192)) as region:
        ...     fitted = region.find_aberrations(tilt=True, trials=200)
        """
        if self._data is None:
            raise RuntimeError("This SSB session is closed. Reopen the acquisition before selecting a region.")
        if len(scan_region) != 4 or any(type(value) is not int for value in scan_region):
            raise ValueError("Use scan_region=(row_start, row_stop, col_start, col_stop), in integer scan pixels.")
        dense = self._streamed is None and not isinstance(self._data, MpsBfColumnFrames)
        shape = self._data.shape[:2] if len(self._data.shape) == 4 else self.scan_shape
        row_start, row_stop, col_start, col_stop = scan_region
        if not (0 <= row_start < row_stop <= shape[0] and 0 <= col_start < col_stop <= shape[1]):
            raise ValueError(f"scan_region={scan_region} is outside the {shape[0]} x {shape[1]} scan.")
        full_field = tuple(scan_region) == (0, shape[0], 0, shape[1])
        if not full_field:
            self._release_cached_buffers((row_stop - row_start, col_stop - col_start))
        options = {
            "backend": self.backend, "voltage_kV": self.voltage_kV,
            "semiangle_mrad": self.semiangle_mrad, "scan_sampling_A": self.scan_sampling_A,
            "detector_sampling_mrad": self.detector_sampling_mrad,
            "bright_field_center_px": self.bright_field_center_px,
            "bright_field_radius_px": self.bright_field_radius_px,
            "bright_field_intensity_threshold": self.bright_field_intensity_threshold,
            "scan_detector_rotation_flipped": self.scan_detector_rotation_flipped, "source_path": self.source_path,
            "aberrations": self.aberrations if aberrations is None else aberrations,
            "scan_detector_rotation_deg": self.scan_detector_rotation_deg
            if scan_detector_rotation_deg is None else scan_detector_rotation_deg,
        }
        if dense:
            if len(self._data.shape) != 4:
                raise ValueError("Region selection needs explicit scan rows and columns. Open a 4D acquisition with SSB.open().")
            if any(options[name] is None for name in (
                "bright_field_center_px", "bright_field_radius_px", "detector_sampling_mrad"
            )):
                pattern = detector.mean(self._data)
                center, radius = detector.fit_probe(pattern)
                if options["bright_field_center_px"] is None:
                    center = center[::-1] if self.scan_detector_rotation_flipped else center
                    options["bright_field_center_px"] = tuple(float(value) + origin for value, origin in zip(center, self._crop_origin))
                if options["bright_field_radius_px"] is None:
                    options["bright_field_radius_px"] = radius
                if options["detector_sampling_mrad"] is None:
                    options["detector_sampling_mrad"] = detector.detector_sampling_mrad(pattern, self.semiangle_mrad)
            counts = self._data[row_start:row_stop, col_start:col_stop]
            # __init__ applies the detector convention once. The parent's
            # resident array has already had it applied.
            if self.scan_detector_rotation_flipped:
                counts = counts.swapaxes(-1, -2)
            regional = type(self)(counts, **options)
            regional._crop_origin = self._crop_origin
        elif self._streamed is not None:
            regional = type(self)(self._data, scan_region=scan_region,
                                  mean_pattern=self._streamed.mean_pattern, **options)
        else:
            if not self.source_path:
                raise ValueError("These detector columns have no original acquisition. Open the original file with SSB.open() to fit regions.")
            with io.load(self.source_path, backend=self.backend, verbose=False) as acquisition:
                regional = type(self)(acquisition, scan_region=scan_region, **options)
        origin = self.scan_region or (0, 0, 0, 0)
        regional.scan_region = (row_start + origin[0], row_stop + origin[0],
                                col_start + origin[2], col_stop + origin[2])
        if scan_detector_rotation_deg is None:
            # an estimated rotation stays estimated: the region's fit settles its 180-degree branch
            regional.scan_detector_rotation_source = self.scan_detector_rotation_source
        regional._dark_field, regional._dark_field_source = None, self
        regional._dark_field_region = tuple(scan_region)
        if full_field and isinstance(self._compute, MpsSSBBackend):
            compute = regional._backend()
            compute.inherit_full_field_preparation(self._compute)
            if compute._prepared is None:
                # Changed geometry needs its own preparation. Avoid holding
                # two full Fourier stacks when sharing is not applicable.
                self._release_cached_buffers()
        for name in ("source_kind", "source_storage_path", "source_dtype", "source_bytes",
                     "source_detector_bin", "source_provenance", "calibration_path",
                     "source_manifest_path", "resident_identity"):
            setattr(regional, name, getattr(self, name))
        return regional

    def _result_signature(
        self,
        operation: Literal["find_aberrations", "reconstruct"],
        settings: dict[str, object],
    ) -> dict[str, object]:
        """Build the exact scientific identity of one SSB operation."""

        return json_value(
            {
                "schema": RESULT_SCHEMA,
                "software": software_signature(),
                "operation": operation,
                "reconstruction": {
                    "default_estimator": "phase_of_mean",
                    # "k-q": the probe phase of the first sideband at k - q (odd orders were at q - k before)
                    "higher_order_model": "krivanek14 k-q",
                    "upsampling_dc_policy": "single_origin",
                },
                "source": (
                    {"detector": self.resident_identity, "storage": None, "calibration": None, "manifest": None}
                    if self.source_path is None and self.resident_identity is not None else
                    source_identity(
                        self.source_path, scan_shape=self._scan_shape,
                        hot_pixel_correction=(self.source_provenance or {}).get("hot_pixel_correction"),
                        storage_path=self.source_storage_path,
                        calibration_path=self.calibration_path, manifest_path=self.source_manifest_path,
                    )
                ),
                "data": {
                    "kind": self.source_kind,
                    "dtype": self.source_dtype,
                    "bytes": self.source_bytes,
                    "shape": tuple(int(value) for value in self._data.shape),
                    "scan_shape": self._scan_shape,
                    "detector_bin": self.source_detector_bin,
                    **({"scan_region": self.scan_region} if self.scan_region is not None else {}),
                    **({"resident": self.resident_identity} if self.resident_identity is not None else {}),
                },
                "instrument": {
                    "voltage_kV": self.voltage_kV,
                    "semiangle_mrad": self.semiangle_mrad,
                    "scan_sampling_A": self.scan_sampling_A,
                    "detector_sampling_mrad": self.detector_sampling_mrad,
                },
                "ssb": {
                    "backend": self.backend,
                    "scan_detector_rotation_deg": self.scan_detector_rotation_deg,
                    "scan_detector_rotation_flipped": self.scan_detector_rotation_flipped,
                    "bright_field_intensity_threshold": self.bright_field_intensity_threshold,
                    "bright_field_radius_px": self.bright_field_radius_px,
                    "bright_field_center_px": self.bright_field_center_px,
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

        # a results folder is shared or moved whole: files are named relative to it, never by absolute path
        folder = paths[0].parent
        return save_result(
            result,
            paths=paths,
            signature=signature,
            input_metadata={
                "source": portable_source(self.source_path, folder),
                "source_storage": portable_source(self.source_storage_path, folder),
                "source_kind": self.source_kind,
                "source_dtype": self.source_dtype,
                "source_bytes": self.source_bytes,
                "source_detector_bin": self.source_detector_bin,
                "source_hot_pixel_correction": (self.source_provenance or {}).get("hot_pixel_correction"),
                "source_load_seconds": self.source_load_seconds,
                "calibration": portable_source(self.calibration_path, folder),
                "source_manifest": portable_source(self.source_manifest_path, folder),
            },
        )

    def _accept_result(self, result: SSBResult) -> SSBResult:
        """Update session state from a computed or reused result."""

        result.source_path = self.source_path
        result.scan_detector_rotation_flipped = self.scan_detector_rotation_flipped
        if result.bright_field_center_px is None:
            result.bright_field_center_px = self.bright_field_center_px
        # Only a backend that already holds prepared data is re-rotated; otherwise the change would trigger a load.
        if result.scan_detector_rotation_deg != self.scan_detector_rotation_deg and self._compute is not None:
            # a reused result, or one whose branch the polarity check changed, can sit on the other 180-degree branch; previews must use its rotation
            self.set_rotation(result.scan_detector_rotation_deg)
        self.aberrations = dict(result.aberrations)
        self.sample_tilt_mrad, self.depth_spread_nm = result.sample_tilt_mrad, result.depth_spread_nm
        self.scan_detector_rotation_deg = result.scan_detector_rotation_deg
        self.best_loss = (
            float(result.loss) if result.loss is not None else float("inf")
        )
        self._reconstruction = result
        from quantem.gpu.ssb.calibration import capture_calibration

        capture_calibration(self, result)
        branches = result.rotation_branches
        if (branches is not None and result.fit_scan_region is None
                and (branches._panel_source is None or branches._panel_source() is None)):
            # this session reconstructs the branch panels on demand (a region fit's region session made its own)
            branches.bind_session(self._rotation_branch_panels)
        return result

    def _dark_field_image(self) -> np.ndarray | None:
        """The scan's dark-field image that locates atomic columns for the 180-degree branch test, or None.

        Computed once from the full-detector source (``_dark_field_of``): the caller's dataset or array, or the file
        a ``SSB.open`` session read; a region session slices its parent's. A released dataset has none, and the
        phase tail alone then decides.
        """
        if self._dark_field is None and self._dark_field_source is not None:
            source = self._dark_field_source
            if isinstance(source, SSB):
                image = source._dark_field_image()
                region = self._dark_field_region
                if image is not None and region is not None:
                    image = image[region[0]:region[1], region[2]:region[3]]
            elif getattr(getattr(source, "data", None), "is_released", False):
                image = None
            elif isinstance(source, Path):
                with io.load(source, backend=self.backend, scan_shape=self._scan_shape, verbose=False) as acquisition:
                    image = _dark_field_of(acquisition, self.semiangle_mrad, self.detector_sampling_mrad,
                                           self.bright_field_center_px)
            else:
                image = _dark_field_of(source, self.semiangle_mrad, self.detector_sampling_mrad,
                                       self.bright_field_center_px)
                if self.scan_region is not None and isinstance(source, Dataset4dstemGPU):
                    row_start, row_stop, col_start, col_stop = self.scan_region
                    image = image[row_start:row_stop, col_start:col_stop]
            self._dark_field = image
        return self._dark_field

    def _finish_fit(self, result: SSBResult) -> SSBResult:
        """Reconstruct a fit's phase at the native scan sampling with the default estimator and keep it on the result."""
        estimator = _default_phase_estimator(result.aberrations)
        if result.phase_estimator == estimator and result.upsampling_factor == 1:
            result.phase_clim = result.phase_limits
            return self._accept_result(result)
        started = time.perf_counter()
        displayed = self.reconstruct(result, phase_estimator=estimator, compute_loss=False, force=True)
        result.timings["final_native_reconstruction_s"] = time.perf_counter() - started
        result.object_wave = displayed.object_wave
        result.phase_clim = result.phase_limits
        result.phase_estimator = displayed.phase_estimator
        result.amplitude_estimated = False
        return self._accept_result(result)

    def _fit_both_branches(self, fit: Callable[[], SSBResult], *, keep_first: bool,
                           measure_gain: bool = False) -> SSBResult:
        """Fit the session's rotation and that rotation plus 180 degrees independently, then keep one branch.

        Each branch runs ``fit`` from the same start (the session's aberrations, tilt and depth spread before the
        first fit) with the same settings, and its phase is reconstructed at the native scan sampling. The branch
        is chosen by ``branches.compare_branches``: positive phase at the dark-field atomic columns, else the larger
        phase tail. ``keep_first`` keeps the session's rotation (a given one) and records what the evidence selects.
        The kept fit carries the comparison as ``rotation_branches`` and both searches' trials.
        ``measure_gain`` (joint fits whose branches skipped the standard comparison) measures the kept fit's
        ``tilt_fit_gain`` once, at its rotation, after the decision.
        """
        start = (self._aberrations, self._aberrations_explicit, self.sample_tilt_mrad, self.depth_spread_nm)
        first_rotation = self.scan_detector_rotation_deg
        fits = []
        for search, rotation in enumerate((first_rotation, opposite_rotation_deg(first_rotation))):
            self._aberrations, self._aberrations_explicit, self.sample_tilt_mrad, self.depth_spread_nm = start
            self.set_rotation(rotation)
            result = fit()
            label_trials(result, search=search)
            fits.append(self._finish_fit(result))
        branches = compare_branches(fits, dark_field=self._dark_field_image(), scan_sampling_A=self.scan_sampling_A,
                                    keep_first=keep_first)
        kept = fits[branches.kept]
        kept.trial_records = list(fits[0].trial_records or ()) + list(fits[1].trial_records or ())
        for index, trial in enumerate(kept.trial_records):
            trial["trial"] = index
        if all(fit.elapsed_s is not None for fit in fits):
            kept.elapsed_s = sum(fit.elapsed_s for fit in fits)
        kept.rotation_branches = branches
        self.set_rotation(kept.scan_detector_rotation_deg)
        if measure_gain:
            self._measure_tilt_fit_gain(kept)
        return self._accept_result(kept)

    def _measure_tilt_fit_gain(self, result: SSBResult) -> None:
        """Set a joint fit's ``tilt_fit_gain``: its agreement over the best standard-SSB agreement.

        The standard comparison (200 trials at depth spread 0) depends only on the rotation, the seed and the
        search bounds, not on the joint search, so run after the branch decision at the kept fit's rotation it
        gives the gain the kept branch's own fit would have reported, at the cost of one comparison instead of two.
        """
        started = time.perf_counter()
        standard = self._backend().fit_sample(verbose=False, compare_standard=True, standard_only=True,
                                              **self._tilt_fit_options)
        standard_fit = float(standard["standard_fit"])
        result.tilt_fit_gain = (float(result.sample_agreement) / standard_fit if standard_fit > 0
                                else float("nan"))
        result.timings["standard_comparison_s"] = time.perf_counter() - started

    def _refine_rotation(self, kept: SSBResult, fit: Callable[[int], SSBResult], *, keep: bool, measure_gain: bool,
                         verbose: bool) -> SSBResult:
        """``find_aberrations(refine_rotation=True)``: refine an estimated rotation around ``kept`` (rotation_refinement.py).

        Each angle is one polish, ``fit(0)`` from the nearest angle's fit, after ``set_rotation`` (G(q, k) is not
        rebuilt). The polishes skip the standard comparison; a kept ``tilt_fit_gain`` is measured again at the
        refined angle. A given or recorded rotation (or ``apply=False``) is left as it is.
        """
        from quantem.gpu.ssb.rotation_refinement import refine_rotation, refinement_summary

        if self.scan_detector_rotation_source != "estimated" or keep:
            if verbose:
                why = "kept (apply=False)" if keep else {"metadata": "recorded with the acquisition"}.get(
                    self.scan_detector_rotation_source, self.scan_detector_rotation_source)
                print(f"SSB: rotation {kept.scan_detector_rotation_deg:.2f}° is {why}; refine_rotation leaves it as it is.")
            return kept

        def polish(rotation_deg: float, start: SSBResult) -> SSBResult:
            self.aberrations = dict(start.aberrations)
            self.sample_tilt_mrad, self.depth_spread_nm = start.sample_tilt_mrad, start.depth_spread_nm
            self.set_rotation(rotation_deg)
            return fit(0)

        compare_standard, self._compare_standard = self._compare_standard, False
        try:
            refined, record = refine_rotation(kept, polish)
        finally:
            self._compare_standard = compare_standard
        if refined is not kept:
            refined.trial_records, refined.rotation_branches = kept.trial_records, kept.rotation_branches
            refined.num_trials = kept.num_trials
            if kept.elapsed_s is not None:
                refined.elapsed_s = kept.elapsed_s + record["seconds"]
            self.set_rotation(refined.scan_detector_rotation_deg)
            if measure_gain and kept.tilt_fit_gain is not None:
                self._measure_tilt_fit_gain(refined)
            refined = self._finish_fit(refined)
        refined.timings["rotation_refinement_s"] = record["seconds"]
        refined.rotation_estimate_deg, refined.rotation_refined = kept.scan_detector_rotation_deg, True
        refined.rotation_refinement = record
        if verbose:
            print(f"SSB: {refinement_summary(record)}")
        return self._accept_result(refined)

    def _release_cached_buffers(self, region_shape: tuple[int, int] | None = None) -> None:
        """Release scratch buffers and budget the overview FFT against a region fit.

        Without a region shape, release the prepared stack as before. With a
        shape, the backend keeps it only when sufficient fitting headroom is
        available; a later full-field correction can then reuse the same bits.
        """
        if isinstance(self._compute, MpsSSBBackend):
            if region_shape is None:
                self._compute.release_prepared()
            else:
                self._compute.release_for_region(region_shape)

    def _center_in_data(self) -> tuple[float, float] | None:
        """The bright-field centre in pixels of the array the session holds, which the backends index.

        The public centre counts from the first pixel of the full detector; a session that decoded only the
        bright-field crop holds an array that starts ``_crop_origin`` later.
        """
        if self.bright_field_center_px is None:
            return None
        center = (self.bright_field_center_px[0] - self._crop_origin[0],
                  self.bright_field_center_px[1] - self._crop_origin[1])
        return center[::-1] if self.scan_detector_rotation_flipped else center

    def _sampling_in_data(self) -> tuple[float, float] | None:
        """Detector sampling in the axes passed to the numerical backend."""
        sampling = self.detector_sampling_mrad
        return sampling[::-1] if sampling is not None and self.scan_detector_rotation_flipped else sampling

    def _in_detector_frame(self, result: SSBResult) -> SSBResult:
        """Move a result's bright-field centre from the array the session holds to full-detector pixels, in place.

        Backends index the array they were given. Without the shift a session built from an encoded acquisition
        reports a centre inside its crop, a different point from the one a dense session reports.
        """
        if result.bright_field_center_px is not None:
            if self.scan_detector_rotation_flipped:
                result.bright_field_center_px = result.bright_field_center_px[::-1]
            result.bright_field_center_px = (result.bright_field_center_px[0] + self._crop_origin[0],
                                             result.bright_field_center_px[1] + self._crop_origin[1])
        result.scan_detector_rotation_flipped = self.scan_detector_rotation_flipped
        return result

    def _backend(self) -> SSBProtocol:
        """The session's CUDA or MPS implementation, built on first use.

        Building it reads the bright-field evidence and transforms it (seconds and gigabytes on a full scan), so a
        session that is only configured, saved or closed never pays for it.
        """
        if self._streamed is not None and max(self._streamed.scan_shape) > 1024:
            raise NotImplementedError(
                "This large resident uses streamed full-scan reconstruction. "
                "Fit with find_aberrations(scan_region=(row_start, row_stop, col_start, col_stop)), "
                "then reconstruct(parameters, compute_loss=False)."
            )
        if self._compute is None:
            if self.backend == "cuda":
                # CuPy loads with the CUDA implementation only, so MPS sessions run without it
                from quantem.gpu.ssb.cuda.backend import CudaSSBBackend

                # Preserve <=1024 loss, higher-order and subset APIs; their
                # Fourier stack is prepared only when those operations need it.
                data = self._backend_input()
                self._compute = CudaSSBBackend(
                    data=data, semiangle=self.semiangle_mrad, scan_sampling=self.scan_sampling_A,
                    detector_sampling_mrad=self._sampling_in_data(), voltage_kV=self.voltage_kV, scan_shape=self._scan_shape,
                    bright_field_intensity_threshold=self.bright_field_intensity_threshold, bright_field_radius_px=self.bright_field_radius_px,
                    bright_field_center_px=self._center_in_data(), scan_detector_rotation_deg=self.scan_detector_rotation_deg,
                    aberrations=aberrations_to_engine(self.aberrations) if self._aberrations_explicit else None,
                )
            else:
                self._compute = MpsSSBBackend(
                    self._data, voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
                    scan_sampling=self.scan_sampling_A, detector_sampling_mrad=self._sampling_in_data(),
                    bright_field_intensity_threshold=self.bright_field_intensity_threshold, bright_field_center_px=self._center_in_data(),
                    bright_field_radius_px=self.bright_field_radius_px, scan_detector_rotation_deg=self.scan_detector_rotation_deg,
                )
            if self.detector_sampling_mrad is None:
                # Retain the backend's probe estimate in the working calibration
                # so region sessions and Save use the value actually computed.
                sampling = (self._compute._det_sampling if self.backend == "mps"
                            else self._compute.angular_sampling)
                self.detector_sampling_mrad = tuple(sampling[::-1] if self.scan_detector_rotation_flipped else sampling)
                self._calibration_initial["detector_sampling_mrad"] = self.detector_sampling_mrad
                _announce_detector_sampling(self.detector_sampling_mrad, self.semiangle_mrad, recordable=False)
        return self._compute

    def _resolve_upsampling_factor(self, upsampling_factor: int | Literal["auto"]) -> int:
        """``upsampling_factor`` as an output factor: ``"auto"`` is ``ssb_upsampling_factor`` for this calibration."""
        if upsampling_factor == "auto":
            from quantem.gpu.optics.physics import ssb_upsampling_factor

            upsampling_factor = ssb_upsampling_factor(
                voltage_kV=self.voltage_kV,
                semiangle_mrad=self.semiangle_mrad,
                scan_sampling_A=self.scan_sampling_A,
            )
        if type(upsampling_factor) is not int or upsampling_factor not in (1, 2, 3, 4, 8):
            raise ValueError("upsampling_factor must be 'auto', 1, 2, 3, 4, or 8.")
        return upsampling_factor

    @_on_scan_gpu
    def _rotation_branch_panels(self, branches: RotationBranches, upsampling_factor) -> tuple[int, tuple] | None:
        """Both branches' fits at ``upsampling_factor``, for ``RotationBranches.upsampled_phases``; None once closed.

        Each branch is reconstructed at its own rotation, aberrations, sample tilt and depth spread, with the default
        estimator, from the G(q, k) the session holds (``reconstruct(trial_ids=...)`` replays a trial the same way);
        the session's rotation is restored and its fit is unchanged.
        """
        if self._data is None:
            return None
        factor = self._resolve_upsampling_factor(upsampling_factor)
        saved_rotation = self.scan_detector_rotation_deg
        phases = []
        try:
            for branch in branches.branches:
                self.set_rotation(branch.scan_detector_rotation_deg)
                phase, _ = self._phase(
                    branch.aberrations, upsampling_factor=factor, compute_loss=False,
                    sample_tilt_mrad=branch.sample_tilt_mrad or (0.0, 0.0), depth_spread_nm=branch.depth_spread_nm or 0.0,
                    phase_estimator=_default_phase_estimator(branch.aberrations),
                )
                phases.append(np.array(host_array(phase), dtype=np.float32))
        finally:
            self.set_rotation(saved_rotation)
        if isinstance(self._compute, MpsSSBBackend):
            self._compute.release_cache()
        return factor, tuple(phases)

    def _reconstruct_trials(self, parameters, trial_ids, *, upsampling_factor, phase_estimator):
        """Replay selected records without replacing the current search or correction."""
        parameters = self._aberration_search if parameters is None else parameters
        if parameters is None:
            raise ValueError("Run ssb.find_aberrations() before reconstructing trials.")
        records = select_records(parameters.trial_records, trial_ids)
        estimator = phase_estimator or _default_phase_estimator(records[0]["params"])
        saved_rotation = self.scan_detector_rotation_deg
        waves = []
        for record in records:
            self.set_rotation(record["scan_detector_rotation_deg"])
            phase, _ = self._phase(
                record["params"],
                sample_tilt_mrad=tuple(record["sample_tilt_mrad"]),
                depth_spread_nm=record["depth_spread_nm"],
                upsampling_factor=upsampling_factor, phase_estimator=estimator, compute_loss=False,
            )
            # Native mean-phase kernels reuse a scratch buffer. Own each
            # wave before reconstructing the next trial on the same stream.
            waves.append(self._unit_wave(phase))
        self.set_rotation(saved_rotation)
        return SSBResult(
            object_wave=(cp if self.backend == "cuda" else np).stack(waves), backend=self.backend, trial_records=records,
            scan_detector_rotation_flipped=self.scan_detector_rotation_flipped,
            voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
            object_sampling_A=self._output_sampling(upsampling_factor), upsampling_factor=upsampling_factor, phase_estimator=estimator,
            amplitude_estimated=False, source_path=self.source_path,
            metadata={"trial_ids": [record["trial"] for record in records]},
        )

    def show(self, view: Literal["interactive", "depth", "tilt", "parameters", "focus"] = "interactive", *,
             size: Literal["small", "medium", "large", "xlarge"] | None = None,
             rows: int | None = None, columns: int | None = None,
             cmap: str | None = None, save: str | Path | None = None,
             region: tuple[int, int, int, int] | None = None,
             mode: Literal["interactive", "static"] | None = None, **options):
        """Open ShowPtycho on this session: the live phase and its FFT, aberration sliders, Save.

        ``options`` go to ``quantem.widget.ShowPtycho``: ``show_fft``, ``save_dir``, ``local_tilt``, ... The widget
        drives this session while it is open. Needs ``quantem.widget``.

        ``show("depth")`` and ``show("tilt")`` return static figures of the
        measured agreement landscape around the latest joint fit. They reuse
        saved parameter slices, or measure them on this session's GPU once.
        If no fit exists, the ordinary joint fit runs first. All unplotted
        parameters remain fixed; these are not spatial maps or uncertainties.

        ``show("focus")`` displays actual reconstructed phase images in Show2D,
        varying focus and model depth spread around the joint fit. Use
        ``layout="stack"`` for Show3D: a depth panel per stack and a shared
        defocus slider, initially at the requested offset nearest zero. Use
        ``mode="static"`` for the same images through ``show_2d``. The saved
        astigmatism, specimen tilt and rotation are retained. These are
        conditional image comparisons, not independent depth slices or refits.

        Parameters
        ----------
        view : {"interactive", "depth", "tilt", "parameters", "focus"}, optional
            The existing ShowPtycho widget, the focus-depth landscape, the
            row-column tilt landscape, both landscapes, or reconstructed images.
        size : {"small", "medium", "large", "xlarge"} or None, optional
            Plot or focus-gallery panel size only; medium by default. Interactive
            focus panels use 300, 450, 600 or 900 CSS pixels. Does not change
            data or sampling.
        rows, columns : int, optional
            Images down or across focus and parameter galleries. Set either
            to infer the other; if both are supplied, every panel must fit.
        cmap : str, optional
            Colormap for focus images and static landscapes, inferno by default.
            ShowPtycho's live view retains its own Color control.
        save : str or pathlib.Path, optional
            Export a static figure. For focus images, also set mode="static".
            This is distinct from save_dir, which stores ShowPtycho calibration.
        region : tuple of int or None, optional
            Static diagnostic region in original scan pixels, as
            ``(row_start, row_stop, col_start, col_stop)`` with exclusive stops.
            This runs one independent joint regional fit with shared detector
            calibration and full-field rotation; it leaves the full-field fit
            unchanged. To reuse a regional fit for several views, retain the
            session returned by ``select_region`` and call its ``show`` method.
            Focus images always reconstruct the full parent field before cropping.
        mode : {"interactive", "static"} or None, optional
            For ``view="focus"``, an interactive Show2D/Show3D widget, or a
            static ``show_2d`` figure containing every frame. Both interactive
            layouts use the same static comparison. Landscape views remain static.
        **options
            ShowPtycho options for the interactive view, or SSBResult.show
            options such as ``size="large"`` for a static diagnostic.
            Focus images accept ``focus_offsets_nm=(-5, 0, 5)`` relative to
            fitted C10 and ``depth_spread_nm`` as an absolute scalar or sequence.
            The default depths are the fitted value and ±5 nm, bounded at zero.
            ``layout="grid"`` uses Show2D; ``layout="stack"`` uses Show3D
            with focus frames and depth panels in interactive mode.
            Focus comparisons accept ``columns`` or ``rows`` to reflow the
            displayed panels. Static views default to two columns; interactive
            views retain one depth row or one stack per depth. ``max_cols``
            remains a deprecated alias. Display layout never changes sampling.

        Returns
        -------
        quantem.widget.ShowPtycho, Show2D, Show3D or matplotlib.figure.Figure
            A live widget, or a closed figure that renders once in a notebook.
            ``save="focus.png"`` retains static focus images' native pixel
            density, independent of the chosen display size.

        Examples
        --------
        >>> ssb = SSB(data)  # doctest: +SKIP
        >>> ssb.reconstruct()  # doctest: +SKIP
        >>> ssb.show(show_fft=True, save_dir="ssb-results")  # doctest: +SKIP
        >>> ssb.show("depth", size="large")  # doctest: +SKIP
        >>> ssb.show("tilt", region=(64, 192, 64, 192))  # doctest: +SKIP
        >>> ssb.show("focus")  # doctest: +SKIP
        >>> ssb.show("focus", layout="stack")  # doctest: +SKIP
        >>> ssb.show("focus", columns=2, size="medium")  # doctest: +SKIP
        >>> ssb.show("focus", mode="static", depth_spread_nm=15)  # doctest: +SKIP
        """
        if view not in ("interactive", "depth", "tilt", "parameters", "focus"):
            raise ValueError(f"Choose 'interactive', 'depth', 'tilt', 'parameters' or 'focus'; got {view!r}.")
        if mode is not None and view != "focus":
            raise ValueError("Use mode with ssb.show('focus'); depth and tilt landscapes are already static.")
        if view == "focus" and save is not None and mode != "static":
            raise ValueError("save exports a static figure; use ssb.show('focus', mode='static', save='focus.png').")
        if view != "interactive":
            if size is not None and (size not in ("small", "medium", "large", "xlarge") or options.get("axsize") is not None):
                raise ValueError("Use size='small', 'medium', 'large' or 'xlarge', or axsize=(width, height), not both.")
            result = self._aberration_search
            if result is None:
                result = self.find_aberrations(search="standard", compare_standard=False,
                                              check_rotation=self._dark_field_region is None)
            if view == "focus":
                from quantem.gpu.ssb.focus import show_focus

                source, crop = self, region
                if region is not None:
                    with self.select_region(region, aberrations=result.aberrations,
                                            scan_detector_rotation_deg=result.scan_detector_rotation_deg) as selected:
                        result = selected.find_aberrations(
                            search="standard", check_rotation=False, compare_standard=False, compute_loss=False)
                while isinstance(source._dark_field_source, SSB) and source._dark_field_region is not None:
                    parent_bounds = source._dark_field_region
                    if crop is None:
                        crop = parent_bounds
                    else:
                        crop = (crop[0] + parent_bounds[0], crop[1] + parent_bounds[0],
                                crop[2] + parent_bounds[2], crop[3] + parent_bounds[2])
                    source = source._dark_field_source
                return show_focus(source, result, region=crop, mode=mode or "interactive", size=size,
                                  rows=rows, columns=columns, cmap="inferno" if cmap is None else cmap, save=save, **options)
            if region is not None:
                with self.select_region(region, aberrations=result.aberrations,
                                        scan_detector_rotation_deg=result.scan_detector_rotation_deg) as selected:
                    selected.find_aberrations(search="standard", check_rotation=False, compare_standard=False, compute_loss=False)
                    return selected.show(view, size=size, rows=rows, columns=columns, cmap=cmap, save=save, **options)
            if not result.parameter_slices:
                self.measure_parameters(result)
            figure = result.show(view, size=size, rows=rows, columns=columns, cmap="inferno" if cmap is None else cmap, **options)
            if self._dark_field_region is not None:
                row_start, row_stop, col_start, col_stop = self._dark_field_region
                figure.suptitle(
                    f"Rows {row_start}:{row_stop}, columns {col_start}:{col_stop}\n"
                    + figure.get_suptitle(), fontsize=11,
                )
            if save is not None:
                figure.savefig(save, dpi="figure", bbox_inches="tight")
            return figure
        if any(value is not None for value in (size, region, rows, columns, cmap, save)):
            raise ValueError(
                "Use size, rows, columns, cmap and save with ssb.show('focus') or a static landscape "
                "('depth', 'tilt', 'parameters'); use mode='static' to save focus images. "
                "ShowPtycho has its own Color control and responsive layout; save_dir stores calibration."
            )
        try:
            from quantem.widget import ShowPtycho
        except ImportError as exc:
            raise ImportError("ssb.show() opens ShowPtycho from quantem.widget: pip install quantem.widget") from exc
        return ShowPtycho(self, **options)

    def show_trials(self, *, best: int | None = None, last: int | None = None,
                    first: int | None = None, upsampling_factor: int | Literal["auto"] = "auto",
                    size: Literal["small", "medium", "large", "xlarge"] | None = None,
                    columns: int | None = None, rows: int | None = None,
                    axsize: tuple[float, float] | None = None):
        """Show selected attempts as phase images with a compact parameter table.

        Parameters
        ----------
        best, last, first
            Supply exactly one positive count. ``best`` selects the lowest
            search loss within the latest recorded search; ``last`` and ``first``
            select completed attempts in time order. Local refinement is not
            an Optuna trial and is displayed separately in the final report.
        upsampling_factor
            Output factor of the shown phases, ``"auto"`` by default (``reconstruct``'s rule; the title names it).
            The trials were searched, and their losses measured, on the native grid; this never reruns the search.
        size
            Per-panel size: small, medium (default), large or xlarge.
        columns, rows
            Images across or down the grid. Default is two columns with rows
            calculated automatically. If both are given, all images must fit.
        axsize
            Advanced panel dimensions in inches instead of size.

        Returns
        -------
        matplotlib.figure.Figure
            Shared-contrast phase panels, calibrated scale bars and trial table.
            The figure renders once as a bare notebook expression.

        Examples
        --------
        >>> aberrations = ssb.find_aberrations(tilt=True)
        >>> ssb.show_trials(best=5)
        >>> ssb.show_trials(last=5)
        >>> ssb.show_trials(first=5)
        >>> ssb.show_trials(best=5, size="medium", columns=2)
        """
        from quantem.gpu.display._layout import panel_inches, resolve_columns

        if size is not None and axsize is not None:
            raise ValueError("Use size or axsize, not both.")
        if axsize is None:
            width = panel_inches("medium" if size is None else size)
            axsize = (width, width)
        if self._aberration_search is None:
            raise ValueError("Run ssb.find_aberrations() before showing trials.")
        records = choose_trials(self._aberration_search.trial_records, best=best, last=last, first=first)
        columns = resolve_columns(len(records), columns=columns, rows=rows)
        result = self.reconstruct(self._aberration_search,
                                  trial_ids=[record["trial"] for record in records], upsampling_factor=upsampling_factor)
        kind = "Best" if best is not None else "Last" if last is not None else "First"
        sampling = f"{result.upsampling_factor}×" + (" auto" if upsampling_factor == "auto" else "")
        return plot_trials(result, kind, axsize=axsize, columns=columns, sampling=sampling)

    @_on_scan_gpu
    def find_aberrations(
        self,
        *,
        search: Literal["standard", "broad", "primary"] = "standard",
        tilt: bool = True,
        objective: FitObjective = "agreement",
        scan_region: tuple[int, int, int, int] | None = None,
        trials: int | Sequence[int] | None = None,
        refinement: RefineMethod | Literal["auto"] = "auto",
        search_ranges: dict[str, tuple[float, float] | float] | None = None,
        refine_lock: list[str] | None = None,
        tilt_limit_mrad: float = 25.0,
        check_rotation: bool | Literal["both"] = True,
        seed: int | None = None,
        save_dir: str | Path | None = None,
        force: bool = False,
        verbose: bool = True,
        compare_standard: bool = True,
        fixed_aberrations: dict[str, float] | None = None,
        refine_rotation: bool = False,
        compute_loss: bool | None = None,
        num_trials: int | None = None,
        higher_order: Sequence[str] | None = None,
        show: bool = True,
        on_initial_fit: Callable[[SSBResult], None] | None = None,
    ) -> SSBResult | dict[int, SSBResult]:
        """Fit aberrations, sample tilt and depth with one search interface.

        ``search="standard"`` (default) uses 200 TPE trials for the initial
        low-order optics/tilt/depth fit, including its Newton polish, then
        jointly refines all terms through fifth order with bounded Newton.
        No extra higher-order TPE search is performed. ``trials`` changes the
        initial budget and ``seed`` defaults to 42.

        In a live notebook, ``show=True`` displays the initial fit at Auto
        sampling and updates that same phase view after refinement. Set
        ``show=False`` to suppress it. Command-line calls do not create plots.
        Display time is recorded separately and excluded from the fit clock.
        Applications can receive the initial native result through
        ``on_initial_fit(result)``; this callback runs once before fifth-order
        refinement and does not indicate that the final fit is complete.
        Treat the callback result as read-only; do not change session geometry
        or fitting parameters in the callback.
        The returned result remains the final fit on the native grid.

        ``search="broad"`` retains the same 200-trial initialization, followed
        by 400 additional fifth-order TPE trials and joint refinement.
        ``trials`` changes this additional budget; its seed defaults to 17.
        ``refinement="nelder-mead"`` selects the same bounded simplex path in
        either fifth-order strategy. Newton has a 5000-evaluation ceiling;
        Nelder-Mead has 3000. Both may stop earlier; a budget cap does not
        establish convergence or a global optimum.

        Both fifth-order strategies use the selected CUDA or MPS device. They share the
        ``broad_fifth_agreement`` model, bounds, precision and reconstruction.
        Scientific objectives run on the selected GPU; TPE and the float64 refinement
        controller run on the CPU. Fit at native sampling and reconstruct
        with ``upsampling_factor="auto"`` for display.

        ``search="primary"`` keeps the earlier low-order fit (200 trials,
        seed 42) on CUDA and MPS. Use it for custom bounds, fixed coefficients,
        ``higher_order`` subsets or the phase-variance cross-check below.
        These advanced controls are not silently applied to the fifth-order
        model. In earlier versions this initial-only route was ``standard``.
        The model interpretation and rotation choice still need validation.

        The remaining controls describe ``search="primary"`` unless noted.

        SSB in quantem is model-based by default. For each trial set of aberrations, sample tilt and depth spread the
        fit predicts the bright-field data G(q, k) the detector should have recorded and keeps the parameters whose
        prediction agrees best with the measurement (least squares). This model-based, tilt-corrected fit is the
        default everywhere and the one to use; ``tilt`` and ``objective`` select one of three methods:

        ============================================================  =====================  ==============================
        Call                                                          Method                 Use it for
        ============================================================  =====================  ==============================
        ``find_aberrations()``                                        model-based,           **the default**: every fit
                                                                      tilt-corrected
        ``find_aberrations(search='primary', tilt=False)``                              model-based, tilt and  thin, untilted samples, or to
                                                                      depth spread at 0      see what tilt correction changes
        ``find_aberrations(search='primary', tilt=False, objective="phase_variance")``  phase variance, the    a backup: reproducing older
                                                                      earlier SSB search     results, cross-checking a fit
        ============================================================  =====================  ==============================

        ``objective`` is ``"agreement"`` (default, the model) or ``"phase_variance"``, which is kept for reproducing
        older results and cross-checks: a backup, not an equal alternative. It chooses the aberrations under which the
        bright-field pixels' corrected phase images agree best (the least phase variance across them), so it scores
        images rather than predicting the data, cannot see the sample tilt or depth spread (with ``tilt=True`` it
        raises) and is about 10x slower; on simulated crystals with known aberrations the model-based fit lands about
        3x closer to the true C10 (docs/maintainer/2026-10-20-ssb-standard-fit-objective.md).

        Each method runs one search of ``trials`` candidates (default 200) per rotation branch, plus local refinement.
        ``refinement="auto"`` (default) selects Newton for the agreement fits and Nelder-Mead for the phase variance and for an agreement fit that
        locks a coefficient, bounds phi12 or holds a nonzero C12 (Newton cannot). Explicit ``"nelder-mead"`` uses the
        simplex in an agreement fit too; None skips local refinement. Saved settings record the selected optimizer,
        not ``"auto"``. ``result.objective`` records the objective; ``result.loss`` is the phase variance at the fit
        whatever the objective, unless ``compute_loss=False`` skips that diagnostic. CUDA and MPS run all three.

        For broad search, pass ``trials=(200, 400, 600)`` to compare independent budgets. This
        returns a dictionary keyed by trial count. Each fit starts from the
        same session calibration and optics, borrows the loaded data and owns
        its preparation; the parent session is unchanged. Fits run sequentially
        and do not share search histories. With ``save_dir``, each budget uses
        its own ``trials-<count>`` subdirectory. Native fit results remain
        available for Auto reconstruction and calibration export.

        >>> fits = ssb.find_aberrations(search="broad", trials=(200, 400, 600), seed=17)
        >>> fits[200].show("convergence", compare=fits)
        >>> image = ssb.reconstruct(fits[400], upsampling_factor="auto")

        ``num_trials`` remains a compatibility alias for ``trials``; supply only one of them.

        ``compute_loss=None`` skips the separate diagnostic for fifth-order fits
        and preserves the diagnostic for ``search="primary"``. Explicit True
        evaluates it for either route. ``compute_loss=False`` keeps the same search, refinement, fitted parameters and reconstructed phase,
        without evaluating the separate phase-variance diagnostic after an agreement fit. ``result.loss`` is then
        None; ``sample_agreement`` and every trial remain available. For example, repeated regional fits can use
        ``fit = ssb.find_aberrations(compute_loss=False)`` and reconstruct with ``ssb.reconstruct(fit)`` later when
        the diagnostic is wanted. This option does not reduce data or change the fitted objective.

        The joint fit searches all six parameters at once (fitting the tilt after a standard fit gets stuck: C10
        has to move to the defocus at mid-depth at the same time). The result's ``sample_tilt_mrad`` (row, col;
        scan frame, within ``tilt_limit_mrad``) and ``depth_spread_nm`` hold the fit. On two full 512 x 512
        acquisitions 100 trials already converged every seed and 200-400 gave the same tilt to 0.02 mrad
        (docs/maintainer/2026-09-24-ssb-units-and-thick-sample.md).
        ``refinement="newton"`` polishes with Newton steps on finite-difference models whose evaluations share
        passes over the data, a few passes instead of hundreds (both backends evaluate the trials of a pass in one batch).
        ``compare_standard=False`` skips the standard-SSB search that only measures ``tilt_fit_gain`` (left None).
        ``trials=0`` with ``tilt=True`` searches nothing and refines from the session's ``aberrations``,
        ``sample_tilt_mrad`` and ``depth_spread_nm``: set them to the previous scan's fit and the next scan of a session
        costs one polish.

        Constrain the joint fit (``tilt=True``) with ``search_ranges`` and ``fixed_aberrations``: a ``C10_nm`` range
        or value, a ``C12_nm`` range from 0 (``(0, maximum_nm)``) or ``C12_nm=0``, which removes astigmatism
        (``phi12_deg`` is then reported as 0). For example
        ``find_aberrations(search="primary", tilt=True, search_ranges={"C10_nm": (-50, 50)}, fixed_aberrations={"C12_nm": 0.0})``.
        The joint search covers phi12 over its whole period, so a phi12 range raises; bound it in a standard fit.
        Both 180-degree branches are fitted with these same bounds. C10 changes sign between the branches, so a C10
        range that excludes one sign also favours one branch.

        ``fixed_aberrations`` with all three of C10_nm, C12_nm and phi12_deg holds optics common across
        local tilt fits. Supply a previously calibrated thick-sample optical
        model, ``tilt=True`` and ``check_rotation=False``. The same batched
        Newton solver optimizes only tilt and depth spread on CUDA and MPS.
        This conditional fit does not determine whether the common optics are
        correct. Use ``compare_standard=False`` to omit the optional thin fit.
        A standard fit (``tilt=False``) searches C10_nm over (-40, 40), C12_nm over (0, 10) and phi12_deg over its
        period unless ``search_ranges`` bounds them; a value holds that coefficient in every trial, ``refine_lock``
        holds it through Nelder-Mead at the search's value, and ``fixed_aberrations`` does both:
        ``find_aberrations(search='primary', tilt=False, fixed_aberrations={"C12_nm": 0.0})``.

        ``check_rotation=True`` (default) settles the 180-degree ambiguity of an estimated scan-detector rotation. The
        centre-of-mass curl cannot tell omega from omega + 180 degrees and both fit equally well (the second is the
        conjugate object with C10 of the opposite sign). So the fit runs twice, at the estimate and at the estimate
        plus 180 degrees, two independent searches with the same settings, and keeps the branch whose atomic columns,
        located on the scan's dark field, have positive phase; when no columns are resolved, the branch with the
        larger phase tail (``compare_rotation_branches`` explains both tests). The fit then costs two searches: two
        joint fits took 9 to 25 s in all on 512 x 512 scans on an M-series Mac, two MPS standard fits (``tilt=False``)
        3 to 10 s, two phase-variance fits about 90 s. ``result.rotation_branches`` records both fits and the
        evidence, ``result.show("rotation")`` draws it, and one line (with ``verbose``) says which branch was kept.
        A rotation given to ``SSB(...)`` or recorded with the acquisition is the user's calibration: it is fitted
        once and never moved. ``ssb.compare_rotation_branches()`` still fits both branches of it and reports.
        ``check_rotation=False`` fits once at the session's rotation, estimated or not.
        ``check_rotation="both"`` explicitly compares both 180-degree branches
        even for a supplied angle, before the fifth-order refinement.

        ``refine_rotation=True`` also refines an estimated rotation within the fit, after the 180-degree branch
        decision: a 1-D search from the estimate (1-degree steps, at most 10 degrees, then Brent to 0.05 degrees)
        that polishes the fit at each angle from the nearest angle's fit, with the same objective and tilt mode, and
        keeps the angle of the highest agreement. Use it when the centre-of-mass estimate is suspect: weak curl
        contrast, thick or tilted crystals, or before iterative ptychography, which needs the angle more than SSB
        does. It costs 7 to 12 polishes, no new search: on the 512 x 512 gold scan 11 polishes took 12.5 s on an M5
        Max. It finds the best angle within the fit's model, which need not improve the image: on gold it moved the
        angle by +2.9 degrees and raised the agreement by 10%, while the phase image changed little (correlation 0.98
        with the image at the estimate; column width, roundness and contrast unchanged); on a simulated crystal it
        recovered a 2-degree error to within 0.01 degrees. ``result.rotation_estimate_deg`` keeps the estimate,
        ``result.rotation_refined`` says whether the angle was refined, ``result.rotation_refinement`` records the
        agreement at both angles and the search, and one line (with ``verbose``) reports the change. A given or
        recorded rotation is never refined.

        ``scan_region=(row_start, row_stop, col_start, col_stop)``
        restricts fitting to that scan region, with exclusive stops. Detector
        geometry is estimated from the full input unless supplied explicitly.
        The returned phase and rotation diagnostics describe the region;
        ``ssb.reconstruct(result)`` applies its parameters to the full scan.
        The region must be square with 128, 256, 512 or 1024 samples per side;
        it is never silently padded or cropped. Large CUDA integer ANS sources
        decode only that region's bright-field crop. Their full reconstruction
        streams detector columns and requires ``compute_loss=False``; the fit's
        diagnostic loss describes the region, not the full field of view.

        ``save_dir`` is a results directory; the fit is written there as ``ssb-find_aberrations.npz`` and ``.json``.
        Set it to reuse an exact prior result from ``SSB.open`` or from an encoded dataset when the detector
        source, calibration, backend, physical parameters, and fit settings all match.
        Changed settings recompute automatically. Set ``force=True`` to recompute
        an otherwise matching result. Direct array inputs are saved but always
        recomputed: a source path does not identify an array's crop or mutations.
        Zero trials start from the current session coefficients: the standard search refines them, and ``tilt=True``
        refines them with the session's sample tilt and depth spread.

        ``higher_order=("C21", "C23", "C30")`` also fits coma (``C21_nm``, ``phi21_deg``), trefoil (``C23_nm``,
        ``phi23_deg``) and spherical aberration (``C30_nm``), any subset, with the same model agreement: after the
        fit above (both branches, rotation refinement), one Newton polish from it moves the higher orders together
        with C10, C12, phi12 and, with ``tilt=True``, the sample tilt and depth spread. Coma and trefoil are not
        confused with the sample tilt (correlations below 0.25 on simulated tilted crystals); C30 is strongly
        correlated with C10 (about -0.9), and fitting it moves C10. The result's ``aberrations`` hold the fitted
        terms, so ``reconstruct(result)`` applies them. They are the probe's aberrations only where the SSB model
        holds: on simulated crystals they matched the truth to a few percent for a light-element crystal up to
        8 nm and for SrTiO3 up to about 1 nm; thicker or heavier crystals returned terms that fit dynamical
        scattering instead (docs/maintainer/2026-10-21-ssb-higher-order-fit.md). ``result.optimizer["higher_order"]``
        says whether they can be read as the probe's (``trusted``, and the numbers in ``guard``): the fitted model must
        explain at least 90 % of G's signal power over the fit band (``coherence``; simulated thick crystals without
        these aberrations reached at most 0.80 where their fit returned spurious terms), and each term's standard
        error, from the agreement's curvature and the Poisson noise of counted electrons, may move the probe phase at
        the aperture edge by at most 0.1 rad. An untrusted fit prints why and keeps the terms. One Newton polish after the fit: about 0.2 s on a
        64 x 64 scan, 5 to 7 s on a 512 x 512 scan with 8,807 bright-field pixels. MPS and CUDA; the agreement
        objective only, and not with ``fixed_aberrations``.
        """

        if search not in ("standard", "broad", "primary"):
            raise ValueError("search must be 'standard', 'broad' or 'primary'.")
        if check_rotation not in (True, False, "both"):
            raise ValueError("check_rotation must be True, False or 'both'.")
        if isinstance(trials, Sequence) and not isinstance(trials, (str, bytes)):
            if search != "broad":
                raise ValueError("Trial-budget comparisons currently require search='broad'. "
                                 "Run standard fits individually to review their histories.")
            if num_trials is not None:
                raise ValueError("Supply trials or num_trials, not both.")
            budgets = list(trials)
            if (not budgets or any(isinstance(n, bool) or not isinstance(n, (int, np.integer))
                                   or n < 1 for n in budgets)
                    or len(set(budgets)) != len(budgets)):
                raise ValueError("Use distinct trial counts, for example trials=(200, 400, 600).")
            field_rows, field_cols = self.input_scan_shape
            region = (0, field_rows, 0, field_cols)
            fits = {}
            for budget in budgets:
                # Separate sessions prevent a preceding fit from becoming a warm start.
                with self.select_region(region) as independent:
                    fits[int(budget)] = independent.find_aberrations(
                        search=search, trials=int(budget), seed=seed, refinement=refinement,
                        tilt=tilt, objective=objective, scan_region=scan_region,
                        search_ranges=search_ranges, refine_lock=refine_lock,
                        tilt_limit_mrad=tilt_limit_mrad, check_rotation=check_rotation,
                        save_dir=None if save_dir is None else Path(save_dir) / f"trials-{budget}",
                        force=force, verbose=verbose, compare_standard=compare_standard,
                        fixed_aberrations=fixed_aberrations, refine_rotation=refine_rotation,
                        compute_loss=compute_loss, higher_order=higher_order,
                        show=show, on_initial_fit=on_initial_fit,
                    )
            return fits
        if search in ("standard", "broad"):
            if scan_region is not None:
                if len(scan_region) != 4 or any(type(value) is not int for value in scan_region):
                    raise ValueError("Use scan_region=(row_start, row_stop, col_start, col_stop), in integer scan pixels.")
                row_start, row_stop, col_start, col_stop = scan_region
                rows, columns = self.input_scan_shape
                side = row_stop - row_start
                if not (0 <= row_start < row_stop <= rows
                        and 0 <= col_start < col_stop <= columns
                        and side == col_stop - col_start and side in (128, 256, 512, 1024)):
                    raise ValueError(f"scan_region={scan_region} must be inside {rows} × {columns} and square with 128, 256, 512 or 1024 scan pixels per side.")

                def initial_region(initial):
                    # The application sees parent-relative provenance; the
                    # child keeps its native fit over its own entire region.
                    on_initial_fit(replace(initial, fit_scan_region=tuple(scan_region)))

                with self.select_region(scan_region) as regional:
                    result = regional.find_aberrations(
                        search=search, tilt=tilt, objective=objective,
                        trials=trials, num_trials=num_trials, refinement=refinement,
                        search_ranges=search_ranges, refine_lock=refine_lock,
                        tilt_limit_mrad=tilt_limit_mrad, check_rotation=check_rotation,
                        seed=seed, save_dir=save_dir, force=force, verbose=verbose,
                        compare_standard=compare_standard, fixed_aberrations=fixed_aberrations,
                        refine_rotation=refine_rotation, compute_loss=compute_loss,
                        higher_order=higher_order, show=show,
                        on_initial_fit=initial_region if on_initial_fit is not None else None,
                    )
                    if result.rotation_branches is not None:
                        # Keep the region's Auto panels before its GPU evidence closes.
                        with suppress(ValueError):
                            result.rotation_branches.upsampled_phases()
                    result.fit_scan_region = tuple(scan_region)
                    if result.saved_path is not None and result.metadata.get("signature"):
                        result = regional._save_result(
                            result, paths=(result.saved_path, result.saved_path.with_suffix(".json")),
                            signature=result.metadata["signature"],
                        )
                # select_region already preserves the parent's detector convention.
                self._aberration_search = self._accept_result(result)
                self._reconstruction = None
                self.trial_history = list(result.trial_records or ())
                return result
            if (not tilt or objective != "agreement"
                    or search_ranges is not None or refine_lock is not None
                    or fixed_aberrations is not None or higher_order is not None
                    or refinement not in ("auto", "newton", "nelder-mead") or tilt_limit_mrad != 25.0):
                raise ValueError(
                    "Fifth-order search uses its joint model and bounds. "
                    "Use search='primary' for custom ranges, "
                    "fixed coefficients or higher_order subsets."
                )
            from .fifth_order_fit import fit_fifth_order

            count = (400 if search == "broad" else 200) if trials is None and num_trials is None else resolve_trial_count(trials, num_trials)
            return fit_fifth_order(
                self, search=search, trials=count,
                seed=(17 if search == "broad" else 42) if seed is None else seed,
                save_dir=save_dir, force=force, verbose=verbose,
                primary_options={"check_rotation": check_rotation, "refine_rotation": refine_rotation},
                refinement="newton" if refinement == "auto" else refinement,
                compute_loss=False if compute_loss is None else compute_loss,
                show=show, on_initial_fit=on_initial_fit,
            )
        if on_initial_fit is not None:
            raise ValueError("on_initial_fit is for standard or broad two-stage fits.")
        return self._find_aberrations(
            "apply" if check_rotation == "both" else "auto" if check_rotation else "off",
            tilt=tilt, objective=objective, scan_region=scan_region,
            num_trials=resolve_trial_count(trials, num_trials),
            refinement=refinement, search_ranges=search_ranges, refine_lock=refine_lock,
            tilt_limit_mrad=tilt_limit_mrad, seed=42 if seed is None else seed, save_dir=save_dir, force=force, verbose=verbose,
            compare_standard=compare_standard, fixed_aberrations=fixed_aberrations,
            refine_rotation=refine_rotation, compute_loss=True if compute_loss is None else compute_loss,
            higher_order=higher_order)

    @_on_scan_gpu
    def compare_rotation_defocus(
        self, *, show: bool = False, trials: int | None = None, refinement: RefineMethod | Literal["auto"] = "auto",
        search_ranges: dict[str, tuple[float, float] | float] | None = None,
        seed: int = 42, tilt: bool = True, verbose: bool = True,
        num_trials: int | None = None,
    ) -> "RotationDefocusComparison":
        """Compare two rotations and both defocus signs without selecting a winner.

        Four independent fits use the session's full field, calibration, valid
        BF pixels and detector convention. The session's rotation and fitted
        optics are preserved. Each phase image uses Auto output sampling.

        Parameters
        ----------
        show : bool, optional
            Display the phase images and histograms immediately in a notebook.
            Default False returns the comparison with a pandas-style table
            representation. ``report()`` returns the pandas DataFrame.
        trials : int, optional
            Coarse-search budget per case, before local refinement. Default 200.
        num_trials : int, optional
            Compatibility alias for ``trials``. Supply only one of them.
        refinement : str, optional
            Local refinement, as in ``find_aberrations``; default ``"auto"``.
        search_ranges : dict, optional
            The joint-fit default ranges unless supplied. C10_nm must span zero
            and is split into negative and positive intervals, including zero.
        seed : int, optional
            Same search seed for each independent case.
        tilt : bool, optional
            Fit tilt and depth spread, default True.
        verbose : bool, optional
            Show one tqdm progress bar with completed cases, current rotation,
            C10 interval, preparation/fitting/reconstruction stage, elapsed time
            and estimated remaining time. Default True. Updates occur between
            stages, not within the GPU search or Newton refinement. Set False
            for unattended runs.

        Returns
        -------
        RotationDefocusComparison
            Native fits and Auto reconstructions. ``show()`` displays matched
            phase images and histograms. Status stays unresolved: appearance
            and loss alone do not establish the physical rotation. Defocus
            f = -C10 only when expressed at the same model reference plane.

        Examples
        --------
        >>> comparison = ssb.compare_rotation_defocus(show=True)  # doctest: +SKIP
        >>> comparison.report()  # doctest: +SKIP
        """
        from quantem.gpu.ssb.rotation_defocus import compare_rotation_defocus

        comparison = compare_rotation_defocus(
            self, num_trials=resolve_trial_count(trials, num_trials), refinement=refinement,
            search_ranges=search_ranges, seed=seed, tilt=tilt, verbose=verbose,
        )
        if show:
            from IPython.display import display

            display(comparison.show())
        return comparison

    def compare_rotation_branches(self, *, apply: bool | None = None, **find_aberrations_options) -> RotationBranches:
        """Fit both 180-degree branches of the rotation independently and say which one the phase supports.

        Each branch is one ``find_aberrations(**find_aberrations_options)`` fit. The default, and the fit to decide the
        branch on, is the model-based, tilt-corrected fit; ``tilt`` and ``objective`` pass through, and the phase
        variance is a backup:

        ============================================================  =====================  ==============================
        Call                                                          Method                 Use it for
        ============================================================  =====================  ==============================
        ``find_aberrations()``                                        model-based,           **the default**: every fit
                                                                      tilt-corrected
        ``find_aberrations(search='primary', tilt=False)``                              model-based, tilt and  thin, untilted samples, or to
                                                                      depth spread at 0      see what tilt correction changes
        ``find_aberrations(search='primary', tilt=False, objective="phase_variance")``  phase variance, the    a backup: reproducing older
                                                                      earlier SSB search     results, cross-checking a fit
        ============================================================  =====================  ==============================

        The centre-of-mass curl knows the scan-detector rotation only modulo 180 degrees. This runs
        ``find_aberrations(**find_aberrations_options)`` twice, at the session's rotation and at that rotation plus
        180 degrees: two fresh searches with the same settings (tilt, ranges, trials, seed), neither started from
        the other. Each fit's phase is reconstructed at the native scan sampling, and the branch is chosen:

        1. **Atomic columns.** The columns are located on the same scan's dark field (``detector.df`` from 1.1 times
           the bright-field radius to the detector edge): its local maxima at least 2 Å apart, the 20% with the
           largest prominence. Dark field is incoherent, so the positions are the same on both branches. Atoms push
           the phase forward, so on the right branch the high-passed phase at the columns is positive. The test is
           decisive when, on either branch, the mean column phase is at least 3 standard errors from zero and at
           least 0.25 of its spread; the branch with the more positive columns is kept.
        2. **Phase tail.** Otherwise, or when the session has no dark field, the branch with the larger mean-centred
           third-moment ratio (``phase_skewness``) is kept.

        The branch is decided on these tilt-corrected fits (the default ``tilt=True``): correcting the sample tilt
        and depth spread makes the column contrast clearer, while both branches still fit equally well.

        Both fits cost a full search each: two joint fits (the default) took 9 to 25 s in all on 512 x 512 scans on
        an M-series Mac, two MPS standard fits (``tilt=False``) 3 to 10 s, two phase-variance fits about 90 s.

        Parameters
        ----------
        apply : bool, optional
            Move the session to the selected branch. Omitted, the selected branch is applied when the session
            estimated its rotation, and a rotation given to ``SSB(...)`` or recorded with the acquisition is kept:
            the comparison then only reports. Either way the session ends settled, holding the kept branch's fit,
            so ``reconstruct()`` applies it without fitting again.
        **find_aberrations_options
            Any ``find_aberrations`` keyword except ``check_rotation``: ``tilt``, ``objective``, ``trials``,
            ``search_ranges``, ``fixed_aberrations``, ``seed``, ``scan_region``, ``save_dir``, ``verbose``, ...
            ``refine_rotation=True`` refines the applied branch's angle after the decision (not with ``apply=False``).

        Returns
        -------
        RotationBranches
            Both fits (aberrations, loss, column test, phase tail, native phase), the atomic columns, the selected
            and the kept branch and the test that decided. ``show()`` draws both fits reconstructed at
            ``upsampling_factor="auto"`` by this session above the evidence, ``report()`` tabulates it.
            The kept fit, with the same comparison as ``rotation_branches``, is the session's latest fit.

        Examples
        --------
        >>> branches = ssb.compare_rotation_branches()  # doctest: +SKIP
        >>> branches.show()  # doctest: +SKIP
        >>> branches.report()  # doctest: +SKIP
        >>> result = ssb.reconstruct()  # the kept fit, no third search  # doctest: +SKIP
        """
        if "check_rotation" in find_aberrations_options:
            raise TypeError("compare_rotation_branches always compares both branches; call "
                            "find_aberrations(check_rotation=False) for one fit at the session's rotation.")
        if apply is None:
            apply = self.scan_detector_rotation_source == "estimated"
        find_aberrations_options["num_trials"] = resolve_trial_count(
            find_aberrations_options.pop("trials", None), find_aberrations_options.pop("num_trials", None),
        )
        return self._find_aberrations("apply" if apply else "keep", **find_aberrations_options).rotation_branches

    def _find_aberrations(
        self,
        branches: Literal["auto", "off", "apply", "keep"],
        *,
        tilt: bool = True,
        objective: FitObjective = "agreement",
        scan_region: tuple[int, int, int, int] | None = None,
        num_trials: int = 200,
        refinement: RefineMethod | Literal["auto"] = "auto",
        search_ranges: dict[str, tuple[float, float] | float] | None = None,
        refine_lock: list[str] | None = None,
        tilt_limit_mrad: float = 25.0,
        seed: int = 42,
        save_dir: str | Path | None = None,
        force: bool = False,
        verbose: bool = True,
        compare_standard: bool = True,
        fixed_aberrations: dict[str, float] | None = None,
        refine_rotation: bool = False,
        compute_loss: bool = True,
        higher_order: Sequence[str] | None = None,
    ) -> SSBResult:
        """``find_aberrations`` with the 180-degree branch handling explicit.

        ``branches``: ``"auto"`` compares both branches when the session estimated its rotation and applies the
        selected one, and fits once otherwise; ``"off"`` fits once; ``"apply"`` compares and applies; ``"keep"``
        compares and keeps the session's rotation.
        """
        if objective not in ("agreement", "phase_variance"):
            raise ValueError(f"objective is 'agreement' (the default, the least-squares agreement of the SSB model with "
                             f"the data) or 'phase_variance' (the older SSB search, tilt=False only); got {objective!r}.")
        if tilt and objective == "phase_variance":
            raise ValueError(
                "objective='phase_variance' cannot fit the sample tilt: the phase variance over bright-field pixels "
                "does not see it. Use find_aberrations(search='primary', tilt=False, objective='phase_variance') for the older SSB "
                "search, or the default find_aberrations() for the tilt-corrected model agreement.")
        if tilt and refine_lock is not None:
            raise ValueError(
                "refine_lock applies to the standard fit (tilt=False). In the joint fit (tilt=True, the default) hold "
                "coefficients with fixed_aberrations: C10_nm alone, C12_nm=0 alone, or all three (with "
                "check_rotation=False).")
        if num_trials < 0:
            raise ValueError(f"trials must be non-negative, got {num_trials}.")
        higher_order = validate_terms(higher_order)
        if higher_order:
            if objective != "agreement":
                raise ValueError("higher_order fits coma, trefoil and C30 with the model agreement; it does not combine "
                                 "with objective='phase_variance'.")
            if fixed_aberrations is not None or refine_lock is not None:
                raise ValueError("higher_order refines C10, C12 and phi12 together with the higher orders; it does not "
                                 "combine with fixed_aberrations or refine_lock.")
        if tilt and int(num_trials) == 0 and (self.sample_tilt_mrad is None or self.depth_spread_nm is None):
            raise ValueError(
                "find_aberrations(tilt=True, trials=0) refines from the session's aberrations, sample_tilt_mrad and "
                "depth_spread_nm: set the last two (a previous scan's fit) first, or use trials=200."
            )
        tilt_bounds: dict[str, object] = {}
        requested = {"search_ranges": search_ranges, "fixed_aberrations": fixed_aberrations}
        if not tilt:
            search_ranges, refine_lock = _standard_constraints(search_ranges, refine_lock, fixed_aberrations)
            fixed_aberrations = None
        # Newton polishes the agreement; the phase variance and a locked or bounded agreement fit use Nelder-Mead
        newton_standard = (not tilt and objective == "agreement"
                           and newton_supported(search_ranges_to_engine(search_ranges), refine_lock))
        if refinement == "auto":
            refinement = "newton" if tilt or newton_standard else "nelder-mead"
        if tilt:
            tilt_bounds, fixed_aberrations = _tilt_constraints(search_ranges, fixed_aberrations)
            search_ranges = None
        if fixed_aberrations is not None:
            fixed_aberrations = validate_aberrations(fixed_aberrations)
            if not all(math.isfinite(v) for v in fixed_aberrations.values()):
                raise ValueError("fixed_aberrations must contain finite coefficients.")
            if refinement != "newton" or branches != "off":
                raise ValueError(
                    "Holding all three coefficients with fixed_aberrations needs refinement='newton' (or 'auto') and "
                    "check_rotation=False: with the optics fixed there is no branch to choose. Hold C10_nm alone or "
                    "C12_nm=0 alone to keep the rotation check.")
        if refinement not in {"nelder-mead", "newton", None}:
            raise ValueError("refinement must be 'auto', 'nelder-mead', 'newton' or None.")
        if refine_rotation and refinement is None:
            raise ValueError("refine_rotation polishes the fit at each angle; it needs refinement ('auto', 'newton' or "
                             "'nelder-mead'), not None.")
        if refinement == "newton" and not tilt and not newton_standard:
            raise ValueError(
                "refinement='newton' polishes the agreement objective; the phase-variance search refines with "
                "'nelder-mead'." if objective == "phase_variance" else
                "refinement='newton' in a standard fit (tilt=False) bounds C10 and the C12 maximum only; use "
                "'nelder-mead' to lock a coefficient, bound phi12 or hold a nonzero C12_nm.")
        if branches == "auto":
            # a rotation given to SSB(...) or recorded with the acquisition is the user's calibration and is kept
            branches = "apply" if self.scan_detector_rotation_source == "estimated" else "off"
        paths = None
        signature = None
        if save_dir is not None:
            paths = result_paths(save_dir, "find_aberrations")
            signature = self._result_signature(
                "find_aberrations",
                {
                    "num_trials": int(num_trials),
                    "refinement": refinement,
                    "search_ranges": requested["search_ranges"] if tilt else search_ranges,
                    "refine_lock": refine_lock,
                    "seed": int(seed),
                    "starting_aberrations": self.aberrations,
                    "starting_aberrations_explicit": self._aberrations_explicit,
                    **({"tilt": True, "tilt_limit_mrad": float(tilt_limit_mrad)} if tilt else {}),
                    "objective": objective,
                    **({"compute_loss": False} if not compute_loss else {}),
                    **({"compare_standard": False} if tilt and not compare_standard else {}),
                    **({"refined_from": {"sample_tilt_mrad": self.sample_tilt_mrad, "depth_spread_nm": self.depth_spread_nm}}
                       if tilt and int(num_trials) == 0 else {}),
                    "rotation_branches": branches,
                    "scan_region": scan_region,
                    **({"fixed_aberrations": fixed_aberrations} if fixed_aberrations is not None else
                       {"fixed_aberrations": requested["fixed_aberrations"]} if requested["fixed_aberrations"] else {}),
                    **({"refine_rotation": True} if refine_rotation else {}),
                    **({"higher_order": list(higher_order)} if higher_order else {}),
                },
            )
            if not force and self.source_kind != "array":
                reused = load_result(
                    paths=paths,
                    signature=signature,
                    backend=self.backend,
                )
                if reused is not None:
                    if verbose:
                        print(f"Matching SSB result found; loading {paths[0]}")
                    self._aberration_search = self._accept_result(reused)
                    if reused.fit_scan_region is not None:
                        self._reconstruction = None
                    self.trial_history = list(reused.trial_records or ())
                    return reused
            if verbose and any(path.exists() for path in paths):
                print("Saved SSB settings changed; running fit end to end")
        if self._streamed is not None and scan_region is None:
            rows, columns = self._streamed.scan_shape
            if rows == columns == 1024:
                scan_region = (0, rows, 0, columns)
            else:
                raise ValueError(
                    "Fit a small region of the large acquisition with "
                    "scan_region=(row_start, row_stop, col_start, col_stop), "
                    "then reconstruct the full scan with compute_loss=False."
                )
        if scan_region is not None:
            if len(scan_region) != 4 or any(type(value) is not int for value in scan_region):
                raise ValueError("Use scan_region=(row_start, row_stop, col_start, col_stop), in integer scan pixels.")
            if len(self._data.shape) != 4:
                raise NotImplementedError("Region fitting needs a dense 4D scan source. Open the original acquisition with io.load and pass it to SSB.")
            row_start, row_stop, col_start, col_stop = scan_region
            rows, columns = self._data.shape[:2]
            side = row_stop - row_start
            if not (0 <= row_start < row_stop <= rows and
                    0 <= col_start < col_stop <= columns and
                    side == col_stop - col_start and side in (128, 256, 512, 1024)):
                raise ValueError(f"scan_region={scan_region} must be inside {rows} × {columns} and square with 128, 256, 512 or 1024 scan pixels per side.")
            # The region keeps the full scan's bright-field disk and detector calibration.
            center, radius, detector_sampling_mrad = self._center_in_data(), self.bright_field_radius_px, self._sampling_in_data()
            region_options = {}
            if self._streamed is not None:
                center, detector_sampling_mrad = self.bright_field_center_px, self.detector_sampling_mrad
                region_options = {"scan_region": scan_region, "mean_pattern": self._streamed.mean_pattern,
                                  "scan_detector_rotation_flipped": self.scan_detector_rotation_flipped}
            if center is None or radius is None or detector_sampling_mrad is None:
                mean_pattern = detector.mean(self._data)
                detected_center, detected_radius = detector.fit_probe(mean_pattern)
                center = detected_center if center is None else center
                radius = detected_radius if radius is None else radius
                if detector_sampling_mrad is None:
                    detector_sampling_mrad = detector.detector_sampling_mrad(mean_pattern, self.semiangle_mrad)
            self._release_cached_buffers()
            with SSB(
                self._data if self._streamed is not None else self._data[row_start:row_stop, col_start:col_stop],
                backend=self.backend, voltage_kV=self.voltage_kV,
                semiangle_mrad=self.semiangle_mrad, scan_sampling_A=self.scan_sampling_A,
                detector_sampling_mrad=detector_sampling_mrad,
                # a refit with no trials starts from the session's aberrations, however they were set
                aberrations=self.aberrations if self._aberrations_explicit or (tilt and int(num_trials) == 0) else None,
                scan_detector_rotation_deg=self.scan_detector_rotation_deg,
                bright_field_center_px=center, bright_field_radius_px=radius,
                bright_field_intensity_threshold=self.bright_field_intensity_threshold,
                **region_options,
            ) as regional:
                if tilt and int(num_trials) == 0:
                    regional.sample_tilt_mrad, regional.depth_spread_nm = self.sample_tilt_mrad, self.depth_spread_nm
                regional._dark_field, regional._dark_field_source = None, self
                regional._dark_field_region = tuple(scan_region)
                # refine_rotation refines only an estimated rotation, the region's as the full scan's
                regional.scan_detector_rotation_source = self.scan_detector_rotation_source
                result = regional._find_aberrations(
                    branches, tilt=tilt, objective=objective, num_trials=num_trials, refinement=refinement,
                    search_ranges=requested["search_ranges"] if tilt else search_ranges, refine_lock=refine_lock,
                    tilt_limit_mrad=tilt_limit_mrad,
                    seed=seed, verbose=verbose, compare_standard=compare_standard,
                    fixed_aberrations=(fixed_aberrations if fixed_aberrations is not None else
                                       requested["fixed_aberrations"] if tilt else None),
                    refine_rotation=refine_rotation, compute_loss=compute_loss, higher_order=higher_order,
                )
                if result.rotation_branches is not None:
                    # the branch panels need the region's G(q, k), which closes with the region session
                    with suppress(ValueError):
                        result.rotation_branches.upsampled_phases()
            result.fit_scan_region = tuple(scan_region)
            # the regional session indexed this session's array, which starts at the crop origin
            self._accept_result(result if self._streamed is not None else self._in_detector_frame(result))
            self._reconstruction = None
            if paths is not None:
                result = self._save_result(result, paths=paths, signature=signature)
            self._aberration_search = result
            self.trial_history = list(result.trial_records or ())
            return result
        if tilt:
            self._compare_standard = bool(compare_standard)
        # The standard comparison only measures tilt_fit_gain. With two branch fits it runs once, for the kept
        # branch after the decision, instead of in both fits (resident Newton fits; the same gain).
        gain_once = tilt and branches != "off" and bool(compare_standard) and refinement == "newton"
        if gain_once:
            self._compare_standard = False

        def fit(trials: int = int(num_trials)) -> SSBResult:
            """One fit at the session's rotation, from the session's aberrations (searched when none are known)."""
            if tilt:
                result = self._fit_tilt(
                    trials, refinement, float(tilt_limit_mrad), int(seed), verbose,
                    fixed_aberrations=fixed_aberrations, bounds=tilt_bounds, compute_loss=compute_loss)
                result.objective = objective
                return self._accept_result(result)
            result = result_from_engine(self._backend().fit(
                aberrations=aberrations_to_engine(self.aberrations) if self._aberrations_explicit else None,
                trials=trials, refinement=refinement,
                search_ranges=search_ranges_to_engine(search_ranges), refine_lock=refine_lock_to_engine(refine_lock),
                seed=int(seed), verbose=verbose, objective=objective, compute_loss=compute_loss,
            ))
            result.objective = objective
            if result.aberrations["C12_nm"] == 0.0:
                # no astigmatism (C12 held at 0): its azimuth has no effect on the image and is stated as 0
                result.aberrations["phi12_deg"] = 0.0
            return self._accept_result(self._in_detector_frame(result))

        # the fit's loss and phase evaluations use small row IFFT tiles on MPS (reconstruct.fit_loss_tiles)
        with fit_loss_tiles() if self.backend == "mps" else nullcontext():
            if branches == "off":
                result = fit()
                label_trials(result, search=0)
                result = self._finish_fit(result)
            else:
                try:
                    result = self._fit_both_branches(fit, keep_first=branches == "keep", measure_gain=gain_once)
                finally:
                    if gain_once:
                        self._compare_standard = True
                if verbose:
                    _announce_rotation_branches(result)
            if refine_rotation:
                result = self._refine_rotation(result, fit, keep=branches == "keep",
                                               measure_gain=tilt and refinement == "newton", verbose=verbose)
            if higher_order:
                if tilt:
                    limits = {key: value for key, value in tilt_bounds.items() if key in ("c10_range", "c12_max")}
                else:
                    # the standard search box (C10 +-40 nm, C12 0 to 10 nm unless search_ranges bounds them)
                    low, high = standard_bounds(search_ranges_to_engine(search_ranges))
                    limits = {"c10_range": (low[0], high[0]), "c12_max": high[1]}
                result = self._fit_higher_order(result, higher_order, tilt=tilt, limits=limits,
                                                tilt_limit_mrad=float(tilt_limit_mrad), verbose=verbose,
                                                compute_loss=compute_loss)
        if not compute_loss:
            result.loss = None
            self.best_loss = float("inf")
        if isinstance(self._compute, MpsSSBBackend):
            # the fits reuse their batch buffers through MLX's allocator cache, across both branches and the phase
            # diagnostic (up to ~20 GB on a 512 x 512 scan); once the fit is done the session keeps only G(q, k)
            self._compute.release_cache()
        if paths is not None and signature is not None:
            result = self._save_result(result, paths=paths, signature=signature)
            if verbose:
                print(f"SSB result saved to {paths[0]}")
        self._aberration_search = result
        self.trial_history = list(result.trial_records or ())
        return result

    @_on_scan_gpu
    def measure_parameters(self, parameters: SSBResult | None = None) -> dict[str, object]:
        """Measure focus-depth and tilt objective slices without refitting.

        The same prepared evidence and joint objective evaluate up to 201
        points. All unplotted parameters stay fixed. These conditional cuts
        describe local interpretation, not uncertainty or a global optimum.

        Parameters
        ----------
        parameters : SSBResult or None, optional
            A joint fit from this session. None uses its latest fit. Measure
            regional fits in the same regional session used for fitting.

        Returns
        -------
        dict
            Recorded axes in nm and mrad, relative agreement, the fitted
            point and evaluation time. Also attached to the result and its
            saved record. Values above one exceed the fitted agreement;
            they do not replace the fitted parameters.

        Examples
        --------
        >>> fit = ssb.find_aberrations(tilt=True)
        >>> slices = ssb.measure_parameters(fit)
        >>> fit.show()  # phase with measured parameters; no further compute
        >>> fit.show(show_parameters=False)
        """
        from quantem.gpu.ssb.parameters import measure_slices

        result = self._aberration_search if parameters is None else parameters
        if result is None or result.sample_tilt_mrad is None or result.depth_spread_nm is None:
            raise ValueError("Measure a joint fit: run ssb.find_aberrations(tilt=True) first.")
        if result.fit_scan_region is not None:
            raise ValueError("Open the selected scan region as its own SSB session before measuring parameter slices.")
        mismatch = "Measure this fit in its original SSB source and calibration, or fit the current session first."
        if result.source_path is not None and (
            self.source_path is None
            or Path(result.source_path).expanduser().resolve() != Path(self.source_path).expanduser().resolve()
        ):
            raise ValueError(mismatch)
        signature = result.metadata.get("signature")
        if result.saved_path is not None and not signature:
            raise ValueError("The saved fit lacks provenance. Fit the current session before measuring parameter slices.")
        # A path cannot establish the contents of a caller-owned array.
        # As with fit reuse, only this session's live fit is trusted. Anonymous
        # unsaved corrections explicitly apply parameters to the current data.
        if (self.source_kind == "array" and (signature or result.source_path is not None)
                and result is not self._aberration_search):
            raise ValueError("Array fits must be measured in the session that fitted them. Fit this session first.")
        if signature:
            current = self._result_signature("find_aberrations", {})
            if any(signature.get(key) != current[key] for key in ("source", "data", "instrument")):
                raise ValueError(mismatch)
            # The fit can select the opposite rotation branch; evaluation below
            # uses its recorded angle. Every other preparation setting must match.
            saved_preparation = dict(signature.get("ssb", {}))
            current_preparation = dict(current["ssb"])
            saved_preparation.pop("scan_detector_rotation_deg", None)
            current_preparation.pop("scan_detector_rotation_deg")
            if saved_preparation != current_preparation:
                raise ValueError(mismatch)
        point = np.array([
            result.aberrations["C10_nm"] * ENGINE_PER_NM,
            result.aberrations["C12_nm"] * ENGINE_PER_NM,
            math.radians(result.aberrations["phi12_deg"]),
            *result.sample_tilt_mrad, result.depth_spread_nm * ENGINE_PER_NM,
        ])
        band = (0.2, 0.9)
        rotation = self.scan_detector_rotation_deg
        try:
            self.set_rotation(result.scan_detector_rotation_deg)
            backend = self._backend()
            evaluate = lambda rows: backend.sample_agreement(rows, band)
            if result.objective == "broad_fifth_agreement":
                from .fifth_order_fit import parameter_evaluator

                evaluate = parameter_evaluator(self, result, band)
            evidence = measure_slices(
                evaluate, point,
                rotation_deg=result.scan_detector_rotation_deg, band_inv_A=band,
            )
        finally:
            self.set_rotation(rotation)
        result.parameter_slices = evidence
        result.timings["parameter_slices_s"] = evidence["elapsed_s"]
        if result.saved_path is not None and result.metadata.get("signature"):
            self._save_result(result, paths=(result.saved_path, result.saved_path.with_suffix(".json")),
                              signature=result.metadata["signature"])
        return evidence

    @_on_scan_gpu
    def reconstruct(
        self,
        parameters: SSBResult | None = None,
        *,
        aberrations: dict[str, float] | None = None,
        trial_ids: object = None,
        upsampling_factor: int | Literal["auto"] = 1,
        phase_estimator: Literal["mean_phase", "phase_of_mean", "complex_wave"] | None = None,
        compute_loss: bool = True,
        save_dir: str | Path | None = None,
        force: bool = False,
        verbose: bool = False,
    ) -> SSBResult:
        """Reconstruct the phase with the session's aberrations, fitting them first when none are known.

        A new session with no aberrations given runs ``find_aberrations()`` once
        on CUDA or MPS.
        The default uses the joint fifth-order model on CUDA and MPS.
        Use ``search="broad"`` for extra exploration before refinement:

        ============================================================  =====================  ==============================
        Call                                                          Method                 Use it for
        ============================================================  =====================  ==============================
        ``find_aberrations()``                                        model-based,           **the default**: every fit
                                                                      tilt-corrected
        ``find_aberrations(search='primary', tilt=False)``                              model-based, tilt and  thin, untilted samples, or to
                                                                      depth spread at 0      see what tilt correction changes
        ``find_aberrations(search='primary', tilt=False, objective="phase_variance")``  phase variance, the    a backup: reproducing older
                                                                      earlier SSB search     results, cross-checking a fit
        ============================================================  =====================  ==============================

        Later calls reuse that fit. On an estimated rotation that fit compares both 180-degree branches with two
        independent joint fits (9 to 25 s in all for a 512 x 512 scan on an M-series Mac) and prints one line naming the
        branch kept; a given or recorded rotation is fitted once and kept. To choose another method or constrain the fit
        (``tilt``, ``objective``, search ranges, fixed terms, seed), call ``find_aberrations(...)`` or
        ``compare_rotation_branches(...)`` yourself first.
        Given ``parameters`` or ``aberrations`` are used as they are.

        Parameters
        ----------
        parameters
            Result of ``find_aberrations``. Carries aberrations, specimen tilt,
            depth spread, and scan-detector rotation. None uses session values.
        aberrations
            Partial coefficient overrides: ``C10_nm``/``C12_nm`` in nm, ``phi12_deg`` in degrees.
            Unspecified coefficients and sample geometry are preserved.
        trial_ids
            Trial IDs from ``parameters.trials.index``. Reconstruct these
            attempts using their original rotation and sample geometry;
            return a phase stack without changing the active session.
        upsampling_factor
            Output factor: 1, 2, 3, 4, or 8. Scan positions and measured
            diffraction patterns are unchanged. Output pixel size decreases
            by this factor, preserving the physical field of view.
            ``"auto"`` chooses the smallest supported factor whose output
            Nyquist frequency covers the ideal circular-aperture SSB support,
            2 * semiangle / wavelength, on both scan axes. This is a grid
            criterion, not a measured resolution or a guarantee of useful
            signal at the cutoff. All 14 aberration terms support upsampling
            and tilt/depth correction on CUDA and MPS.
        phase_estimator
            Defaults to ``phase_of_mean`` on CUDA and MPS at every output
            factor, including higher-order aberrations.
            The loss is the native-grid phase variance over bright-field
            pixels, a diagnostic (the default fits maximise the model
            agreement instead), not a score computed from the displayed phase.
            ``complex_wave`` explicitly retains amplitude for native, thin-sample
            reconstruction, including coherent temporal averaging.
        compute_loss
            Evaluate diagnostic loss on the native grid, regardless of output
            factor. False skips this diagnostic without changing the image.
        save_dir
            Results directory: the result and its correction metadata are written there as
            ``ssb-reconstruct.npz`` and ``ssb-reconstruct.json``.
            Exact saved matches from ``SSB.open`` reload without reconstruction.
            Direct array inputs always recompute because their crop or values
            can change without changing the source path.
        force
            Recompute even when an exact saved result exists.
        verbose
            Print output sampling, shape, correction, and persistence status.

        Returns
        -------
        SSBResult
            Calibrated reconstruction. Upsampled, tilt-aware, and wave-average
            paths recover phase only: their object wave has unit amplitude,
            explicitly recorded by ``amplitude_estimated=False``.

        Examples
        --------
        >>> result = ssb.reconstruct()  # fits the aberrations once, then reconstructs
        >>> aberrations = ssb.find_aberrations(tilt=True)
        >>> result = ssb.reconstruct(aberrations, upsampling_factor=4, save_dir="results/4x")
        """
        upsampling_factor = self._resolve_upsampling_factor(upsampling_factor)
        if phase_estimator not in (None, "mean_phase", "phase_of_mean", "complex_wave"):
            raise ValueError("phase_estimator must be 'mean_phase', 'phase_of_mean', or 'complex_wave'.")
        if parameters is not None and not isinstance(parameters, SSBResult):
            raise TypeError("Pass the result of find_aberrations(), or supply coefficients with aberrations={...}.")
        if trial_ids is not None:
            if phase_estimator == "complex_wave":
                raise ValueError("Trial inspection returns phase images; choose mean_phase or phase_of_mean.")
            if aberrations is not None or save_dir is not None:
                raise ValueError("Replay trials without coefficient overrides or save_dir; save the returned result separately.")
            return self._reconstruct_trials(parameters, trial_ids, upsampling_factor=upsampling_factor, phase_estimator=phase_estimator)
        if parameters is None and aberrations is None and not self._aberrations_explicit:
            print("SSB: no aberrations given or fitted yet; fitting with search='standard' first.")
            parameters = self.find_aberrations(verbose=verbose)
            if not verbose:
                _announce_rotation_branches(parameters)
        correction = parameters
        if correction is not None:
            coefs = dict(correction.aberrations)
            tilt = correction.sample_tilt_mrad
            depth = correction.depth_spread_nm
            if correction.scan_detector_rotation_deg != self.scan_detector_rotation_deg:
                self.set_rotation(correction.scan_detector_rotation_deg)
        else:
            coefs = dict(self.aberrations)
            tilt = self.sample_tilt_mrad
            depth = self.depth_spread_nm
        coefs = validate_aberrations({**coefs, **(aberrations or {})}, higher_order=True)
        estimator = phase_estimator or _default_phase_estimator(coefs)
        higher = higher_order_arrays(coefs)
        if estimator == "complex_wave" and higher is not None and np.any(higher[0][2:]):
            raise ValueError("Higher-order reconstruction supports phase_of_mean or mean_phase; choose either phase estimator.")
        if estimator == "complex_wave" and (upsampling_factor != 1 or depth or any(tilt or ())):
            raise ValueError("complex_wave requires upsampling_factor=1 and no sample tilt/depth; choose a phase estimator instead.")
        settings = {
            "aberrations": coefs, "compute_loss": bool(compute_loss),
            "upsampling_factor": upsampling_factor, "phase_estimator": estimator,
            "sample_tilt_mrad": tilt, "depth_spread_nm": depth,
        }
        paths = signature = None
        if save_dir is not None:
            paths = result_paths(save_dir, "reconstruct")
            signature = self._result_signature("reconstruct", settings)
            if not force and self.source_kind != "array":
                reused = load_result(paths=paths, signature=signature, backend=self.backend)
                if reused is not None:
                    if verbose:
                        print(f"Loading matching {upsampling_factor}x SSB result: {paths[0]}")
                    return self._accept_result(reused)

        if verbose:
            rows, cols = self.scan_shape
            print(f"Reconstructing {upsampling_factor}x: {rows * upsampling_factor} x {cols * upsampling_factor}; "
                  f"{estimator}; tilt {tilt or (0.0, 0.0)} mrad; "
                  f"depth spread {depth or 0.0:.3f} nm; loss at native 1x")
        cached = self._reconstruction
        if (not force and save_dir is None and parameters is None and aberrations is None
                and cached is not None and cached.upsampling_factor == upsampling_factor
                and cached.phase_estimator == estimator
                and cached.scan_detector_rotation_deg == self.scan_detector_rotation_deg
                and cached.aberrations == coefs
                and cached.sample_tilt_mrad == tilt
                and cached.depth_spread_nm == depth
                and (not compute_loss or cached.loss is not None)):
            return cached
        evidence = self._fit_evidence(correction, coefs, tilt, depth)
        if estimator == "complex_wave":
            result = self._in_detector_frame(result_from_engine(self._backend().reconstruct_result(
                aberrations_to_engine(coefs), compute_loss=compute_loss,
            )))
            result.phase_estimator = estimator
            result.amplitude_estimated = True
            result.phase_clim = correction.phase_limits if correction is not None else None
            result.fit_scan_region = correction.fit_scan_region if correction is not None else None
            for name, value in evidence.items():
                setattr(result, name, value)
            result = self._accept_result(result)
            if paths is not None and signature is not None:
                result = self._save_result(result, paths=paths, signature=signature)
            return result
        started = time.perf_counter()
        phase, loss = self._phase(
            coefs, upsampling_factor=upsampling_factor, phase_estimator=estimator,
            sample_tilt_mrad=tilt or (0.0, 0.0), depth_spread_nm=depth or 0.0,
            compute_loss=compute_loss,
        )
        if isinstance(self._compute, MpsSSBBackend):
            # slider previews reuse their chunk buffers through MLX's cache (the loss's row IFFT is 8.6 GB at 4096
            # bright-field pixels); a finished reconstruction keeps only G(q, k)
            self._compute.release_cache()
        result = SSBResult(
            object_wave=self._unit_wave(phase), backend=self.backend, aberrations=dict(coefs),
            sample_tilt_mrad=tilt, depth_spread_nm=depth,
            scan_detector_rotation_deg=self.scan_detector_rotation_deg,
            loss=loss, elapsed_s=time.perf_counter() - started, num_bright_field_pixels=self.num_bright_field_pixels,
            voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
            object_sampling_A=self._output_sampling(upsampling_factor), upsampling_factor=upsampling_factor,
            phase_estimator=estimator, amplitude_estimated=False,
            phase_clim=correction.phase_limits if correction is not None else None,
            fit_scan_region=correction.fit_scan_region if correction is not None else None,
            **evidence,
        )
        result = self._accept_result(result)
        if paths is not None and signature is not None:
            result = self._save_result(result, paths=paths, signature=signature)
            if verbose:
                print(f"SSB result saved to {paths[0]}")
        return result

    def _fit_evidence(self, correction: SSBResult | None, coefs: dict[str, float], tilt, depth) -> dict[str, object]:
        """The objective and the 180-degree branch comparison of the fit a reconstruction applies, for its result.

        The fit is ``correction``, else the session's latest ``find_aberrations``. Both describe its own
        aberrations, sample geometry and rotation, so they are carried only when the reconstruction uses exactly
        those; changed coefficients reconstruct another image, and the result then states neither.
        """
        fit = self._aberration_search if correction is None else correction
        if fit is None or (fit.objective is None and fit.rotation_branches is None and not fit.rotation_refined):
            return {}
        if (fit.scan_detector_rotation_deg != self.scan_detector_rotation_deg
                or validate_aberrations(dict(fit.aberrations), higher_order=True) != coefs
                or fit.sample_tilt_mrad != tilt or fit.depth_spread_nm != depth):
            return {}
        return {"objective": fit.objective, "rotation_branches": fit.rotation_branches,
                "rotation_estimate_deg": fit.rotation_estimate_deg, "rotation_refined": fit.rotation_refined,
                "rotation_refinement": fit.rotation_refinement}

    @_on_scan_gpu
    def preview(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool = True,
        upsampling_factor: int = 1,
        context: AbstractContextManager | None = None,
        sample_tilt_mrad: tuple[float, float] = (0.0, 0.0),
        depth_spread_nm: float = 0.0,
        phase_estimator: Literal["mean_phase", "phase_of_mean"] | None = None,
        output: Literal["numpy", "native"] = "numpy",
        cancel: Callable[[], bool] | None = None,
    ) -> tuple[object, float | None]:
        """Reconstruct a transient phase image for an interactive viewer.

        The returned array has the full selected output resolution. "Preview"
        means it is not saved and does not replace the fitted calibration or
        stored result; it does not imply a lower-quality reconstruction.

        Interactive applications may provide ``cancel``, a predicate that is
        true when this request is obsolete. Cancellation raises
        ``concurrent.futures.CancelledError`` and never returns a partial
        image or loss. MPS checks between completed GPU batches; other
        backends check before and after computation. A running GPU kernel is
        not interrupted. Omitting this option leaves computation unchanged.

        ``output="numpy"`` returns a float32 host array for display. On CUDA, ``output="native"`` returns the CuPy phase
        on the session's device, without exporting it to the host, so diagnostics can reduce a series of phase images
        on the GPU (a through-focus series of 30 previews). Other backends require ``"numpy"``.

        ``aberrations`` holds ``C10_nm`` / ``C12_nm`` in nm and ``phi12_deg`` in degrees. It may also name higher
        orders, each with its unit in the key: ``C21_nm`` with ``phi21_deg``, ``C23_nm`` with ``phi23_deg``, ``C30_nm``,
        ``C32_nm``, ``C34_nm``, ``C41_nm``, ``C43_nm``, ``C45_nm``, ``C50_nm``, ``C52_nm``, ``C54_nm``, ``C56_nm`` and
        their ``phi.._deg`` (the round ``C30`` and ``C50`` have none). Naming any of them selects the 14-term
        aberration kernel. A positive ``depth_spread_nm`` switches to the thick-sample model:
        each bright-field pixel's correction is averaged over that depth, with the crystal leaning by ``sample_tilt_mrad``
        (row, col; scan frame), as fitted by ``find_aberrations(tilt=True)``. With no depth spread the tilt has no effect and this is
        standard SSB. CUDA and MPS backends.

        ``upsampling_factor`` selects 1, 2, 3, 4, or 8 times finer output sampling.
        Values above one support all 14 terms with or without tilt/depth correction
        on CUDA and MPS. The diagnostic loss and aberration search remain on the native
        scan grid. Sampling never fits parameters or interpolates detector data.

        By default, CUDA and MPS reconstruction averages corrected complex
        waves before taking phase (``phase_of_mean``), with or without sample
        tilt, at every output factor. Explicit ``mean_phase`` retains the
        historical per-detector phase average for comparisons. Higher-order
        coefficients use the same coherent estimator. Upsampling excludes replicas
        of the scan mean and retains one normalization at q=0. Quantitative phase
        amplitude is not established by stronger image contrast. Diagnostic
        loss still uses native-grid per-detector phase variance; fitting is
        unaffected.

        Examples
        --------
        Fit all correction parameters once, then change only output sampling:

        >>> fitted = ssb.find_aberrations(tilt=True)  # native-grid joint fit
        >>> phase, loss = ssb.preview(
        ...     dict(fitted), sample_tilt_mrad=fitted.sample_tilt_mrad,
        ...     depth_spread_nm=fitted.depth_spread_nm, upsampling_factor=4,
        ... )

        Compare the historical estimator explicitly, without changing the fit:

        >>> candidate, native_loss = ssb.preview(
        ...     dict(fitted), sample_tilt_mrad=fitted.sample_tilt_mrad,
        ...     depth_spread_nm=fitted.depth_spread_nm, upsampling_factor=4,
        ...     phase_estimator="mean_phase",
        ... )
        """

        if output not in ("numpy", "native"):
            raise ValueError(f"Use output='numpy' or 'native'; got {output!r}.")
        if output == "native" and self.backend != "cuda":
            raise NotImplementedError("Native SSB preview output requires backend='cuda'; use output='numpy' on this backend.")
        with cancellation_scope(cancel):
            phase, loss = self._phase(
                aberrations, compute_loss=compute_loss,
                upsampling_factor=upsampling_factor, context=context,
                sample_tilt_mrad=sample_tilt_mrad, depth_spread_nm=depth_spread_nm,
                phase_estimator=phase_estimator,
            )
        if output == "native":
            # the mean-phase path returns the engine's reused buffer, which the next preview overwrites
            return phase.copy(), loss
        if self.backend == "cuda":
            phase = cp.asnumpy(phase)
        return np.asarray(phase, dtype=np.float32), loss

    def _phase(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool = True,
        upsampling_factor: int = 1,
        context: AbstractContextManager | None = None,
        sample_tilt_mrad: tuple[float, float] = (0.0, 0.0),
        depth_spread_nm: float = 0.0,
        phase_estimator: Literal["mean_phase", "phase_of_mean"] | None = None,
    ) -> tuple[object, float | None]:
        """Compute the shared phase without exporting CUDA arrays to the host."""
        aberrations = validate_aberrations(aberrations, higher_order=True)
        coefs = aberrations_to_engine(aberrations)
        # 14-slot magnitudes (nm) and azimuths (rad) when the mapping names a higher order; else both None
        higher_order_magnitudes, higher_order_angles = higher_order_arrays(aberrations) or (None, None)
        if phase_estimator is None:
            phase_estimator = _default_phase_estimator(aberrations)
        if phase_estimator not in ("mean_phase", "phase_of_mean"):
            raise ValueError(
                "phase_estimator must be 'mean_phase' or 'phase_of_mean'; "
                f"got {phase_estimator!r}."
            )
        if type(upsampling_factor) is not int or upsampling_factor not in (1, 2, 3, 4, 8):
            raise ValueError("upsampling_factor must be 1, 2, 3, 4, or 8.")
        if higher_order_magnitudes is not None:
            # An angle has no physical effect when its coefficient is zero.
            uses_depth_kernel = (
                upsampling_factor > 1
                or depth_spread_nm > 0
                or phase_estimator == "phase_of_mean"
            )
            if uses_depth_kernel and not np.any(higher_order_magnitudes[2:]):
                # the two-term kernels then read the same float32 coefficients the 14-term kernel would
                coefs = {**coefs, "C10": float(higher_order_magnitudes[0]) * ENGINE_PER_NM,
                         "C12": float(higher_order_magnitudes[1]) * ENGINE_PER_NM,
                         "phi12": float(higher_order_angles[1])}
                higher_order_magnitudes = higher_order_angles = None
        scope = nullcontext() if context is None else context
        if self._streamed is not None:
            bounded = (not compute_loss and phase_estimator == "phase_of_mean"
                       and higher_order_magnitudes is None and context is None)
            if not bounded and max(self._streamed.scan_shape) > 1024:
                raise NotImplementedError(
                    "Streamed full-scan SSB supports C10/C12 phase_of_mean with compute_loss=False. "
                    "The regional fit retains its own diagnostic loss; it is not a full-scan loss."
                )
            if bounded:
                return self._streamed.phase(
                    coefs, voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
                    scan_sampling_A=self.scan_sampling_A, rotation_deg=self.scan_detector_rotation_deg,
                    upsampling_factor=upsampling_factor, sample_tilt_mrad=sample_tilt_mrad,
                    depth_spread_A=float(depth_spread_nm) * ENGINE_PER_NM,
                )
        if (upsampling_factor != 1 or phase_estimator == "phase_of_mean"
                or (higher_order_magnitudes is not None and (depth_spread_nm > 0 or self.backend == "mps"))):
            backend = self._backend()
            higher = {} if higher_order_magnitudes is None else {
                "higher_order_magnitudes": higher_order_magnitudes * np.float32(ENGINE_PER_NM),
                "higher_order_angles": higher_order_angles,
            }
            with scope:
                return backend.preview_upsampled(
                    coefs, upsampling_factor=upsampling_factor, compute_loss=compute_loss,
                    sample_tilt_mrad=tuple(float(value) for value in sample_tilt_mrad),
                    depth_spread_A=float(depth_spread_nm) * ENGINE_PER_NM,
                    phase_estimator=phase_estimator,
                    **higher,
                )
        if float(depth_spread_nm) > 0.0:
            sample = {"tilt_row_mrad": float(sample_tilt_mrad[0]), "tilt_col_mrad": float(sample_tilt_mrad[1]),
                      "depth_spread_A": float(depth_spread_nm) * ENGINE_PER_NM}
            if higher_order_magnitudes is not None:
                raise ValueError("The thick-sample preview does not combine with higher-order aberrations yet.")
            backend = self._backend()
            with scope:
                return backend.preview_sample(coefs, sample, compute_loss=compute_loss)
        backend = self._backend()
        with scope:
            return backend.preview(
                coefs,
                compute_loss=compute_loss,
                # all 14 magnitudes (C10, C12 in slots 0-1, then C21..C56) are nm; the engines take Angstrom
                higher_order_magnitudes=None if higher_order_magnitudes is None else higher_order_magnitudes * np.float32(ENGINE_PER_NM),
                higher_order_angles=higher_order_angles,
            )

    @property
    def supports_tilt(self) -> bool:
        """True when this session's backend implements the thick-sample model (``find_aberrations(tilt=True)``, ``preview(sample_tilt_mrad=...)``).

        Both Python backends, CUDA and MPS, implement it.
        """
        return self.backend in ("cuda", "mps")

    def _fit_tilt(self, trials: int, refinement: RefineMethod, tilt_limit_mrad: float, seed: int, verbose: bool,
                  *, fixed_aberrations: dict[str, float] | None = None,
                  bounds: dict[str, object] | None = None,
                  compute_loss: bool = True) -> SSBResult:
        """Joint thick-sample fit (backend ``fit_sample``, engine units) -> an nm SSBResult at the fitted parameters."""
        preparation_started = time.perf_counter()
        backend = self._backend()
        backend_preparation_s = time.perf_counter() - preparation_started
        started = time.perf_counter()
        start = None
        if trials == 0:
            # the previous scan's fit, in engine units, is the one point the polish starts from
            engine = aberrations_to_engine(self.aberrations)
            start = (engine["C10"], engine["C12"], engine["phi12"], *self.sample_tilt_mrad, self.depth_spread_nm * ENGINE_PER_NM)
        fixed = {} if fixed_aberrations is None else {"fixed_aberrations": aberrations_to_engine(fixed_aberrations)}
        options = dict(trials=trials, tilt_limit_mrad=tilt_limit_mrad, seed=seed,
                       polish_starts=0 if refinement is None else 3, polish=refinement or "nelder-mead",
                       start=start, **fixed, **(bounds or {}))
        # the standard comparison of a later _measure_tilt_fit_gain uses this fit's seed and search bounds
        self._tilt_fit_options = options
        fit = backend.fit_sample(verbose=verbose, compare_standard=self._compare_standard, **options)
        aberrations = aberrations_from_engine(fit)
        if (bounds or {}).get("c12_max") == 0.0:
            # C12 held at zero: the azimuth has no effect on the image, so it is stated as 0, not the search's last draw
            aberrations["phi12_deg"] = 0.0
        sample_tilt_mrad = (float(fit["tilt_row_mrad"]), float(fit["tilt_col_mrad"]))
        depth_spread_nm = float(fit["depth_spread_A"]) / ENGINE_PER_NM
        diagnostic_started = time.perf_counter()
        estimator = "mean_phase" if compute_loss else _default_phase_estimator(aberrations)
        phase, loss = self._phase(aberrations, sample_tilt_mrad=sample_tilt_mrad, depth_spread_nm=depth_spread_nm,
                                 phase_estimator=estimator, compute_loss=compute_loss)
        timings = dict(fit.get("timings", {}))
        timings["backend_preparation_s"] = backend_preparation_s
        timing = "native_phase_diagnostic_s" if compute_loss else "final_native_reconstruction_s"
        timings[timing] = time.perf_counter() - diagnostic_started
        # the thick-sample path recovers the phase only; the transmission amplitude is not estimated
        object_wave = self._unit_wave(phase)
        records = []
        for trial in fit["trial_records"]:
            values = trial["params"]
            records.append({
                "number": trial["number"], "loss": trial["loss"],
                "objective": "negative_thick_agreement", "band_inv_A": fit["band_inv_A"],
                "params": {"C10_nm": values["C10"] / ENGINE_PER_NM, "C12_nm": values["C12"] / ENGINE_PER_NM,
                           "phi12_deg": math.degrees(values["phi12"])},
                "sample_tilt_mrad": [values["tilt_row_mrad"], values["tilt_col_mrad"]],
                "depth_spread_nm": values["depth_spread_A"] / ENGINE_PER_NM,
            })
        return SSBResult(object_wave=object_wave, trial_records=records, backend=self.backend, aberrations=aberrations, sample_tilt_mrad=sample_tilt_mrad,
                         timings=timings, optimizer=dict(fit.get("optimizer", {})), depth_spread_nm=depth_spread_nm, sample_agreement=float(fit["fit"]), tilt_fit_gain=None if fit["gain"] is None else float(fit["gain"]),
                         amplitude_estimated=False, phase_estimator=estimator,
                         scan_detector_rotation_deg=self.scan_detector_rotation_deg,
                         loss=None if loss is None else float(loss),
                         elapsed_s=time.perf_counter() - started, num_trials=trials, num_bright_field_pixels=self.num_bright_field_pixels,
                         refinement=refinement, voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
                         object_sampling_A=self.scan_sampling_A)

    def _fit_higher_order(self, result: SSBResult, terms: tuple[str, ...], *, tilt: bool,
                          limits: dict[str, object], tilt_limit_mrad: float, verbose: bool,
                          compute_loss: bool = True) -> SSBResult:
        """Polish a finished fit with the higher-order ``terms`` free (``higher_order_fit.refine_higher_order``).

        Starts from ``result`` with the terms at 0, at its rotation, inside the fit's C10 range and C12 maximum
        (``limits``). ``tilt=False`` holds the tilt and depth spread at 0. Returns a new result at the polished point with its native phase and loss, carrying the fit's
        rotation evidence and trials.
        """
        started = time.perf_counter()
        backend = self._backend()
        engine = aberrations_to_engine(result.aberrations)
        sample_tilt = result.sample_tilt_mrad or (0.0, 0.0)
        depth_A = (result.depth_spread_nm or 0.0) * ENGINE_PER_NM
        start = to_coordinates(engine["C10"], engine["C12"], engine["phi12"], sample_tilt, depth_A, {}, terms)
        fit = refine_higher_order(backend.higher_order_agreement, start, terms, tilt=tilt,
                                  tilt_limit_mrad=tilt_limit_mrad, **limits)
        guard = assess(fit, terms, backend.band_power(), counted=_counted_electrons(self.source_dtype),
                       wavelength_A=float(wavelength_A_from_kV(self.voltage_kV)),
                       semiangle_rad=float(self.semiangle_mrad) * 1e-3)
        values = from_coordinates(fit["z"], terms)
        aberrations = {"C10_nm": values["C10"] / ENGINE_PER_NM, "C12_nm": values["C12"] / ENGINE_PER_NM,
                       "phi12_deg": math.degrees(values["phi12"])}
        for name in terms:
            aberrations[f"{name}_nm"] = values[name] / ENGINE_PER_NM
            if name[2] != "0":
                aberrations[f"phi{name[1:]}_deg"] = math.degrees(values[f"phi{name[1:]}"])
        if tilt:
            sample_tilt_mrad = (values["tilt_row_mrad"], values["tilt_col_mrad"])
            depth_spread_nm = values["depth_spread_A"] / ENGINE_PER_NM
        else:
            sample_tilt_mrad, depth_spread_nm = result.sample_tilt_mrad, result.depth_spread_nm
        refined_s = time.perf_counter() - started
        phase, loss = self._phase(aberrations, sample_tilt_mrad=sample_tilt_mrad or (0.0, 0.0),
                                  depth_spread_nm=depth_spread_nm or 0.0,
                                  phase_estimator="mean_phase" if compute_loss else "phase_of_mean",
                                  compute_loss=compute_loss)
        record = {"terms": list(terms), "start_agreement": float(fit["start_agreement"]),
                  "agreement": float(fit["agreement"]), "evaluations": int(fit["evaluations"]),
                  "passes": int(fit["passes"]), "converged": bool(fit["converged"]), "seconds": refined_s,
                  "trusted": guard["trusted"], "guard": guard,
                  "host_numerical_search": True, "optimizer_backend": "numpy",
                  "agreement_backend": self.backend}
        refined = replace(
            result, object_wave=self._unit_wave(phase), aberrations=aberrations, sample_tilt_mrad=sample_tilt_mrad,
            depth_spread_nm=depth_spread_nm, loss=None if loss is None else float(loss),
            phase_estimator="mean_phase" if compute_loss else "phase_of_mean",
            amplitude_estimated=False,
            sample_agreement=float(fit["agreement"]),
            timings={**result.timings, "higher_order_s": time.perf_counter() - started},
            optimizer={**result.optimizer, "higher_order": record, "host_numerical_search": True},
            elapsed_s=None if result.elapsed_s is None else result.elapsed_s + (time.perf_counter() - started),
        )
        if verbose:
            gain = record["agreement"] / record["start_agreement"]
            fitted = ", ".join(f"{name} {aberrations[f'{name}_nm']:.1f} nm" for name in terms)
            print(f"SSB: higher orders {fitted}; agreement x{gain:.3f}, coherence {guard['coherence']:.2f}, "
                  f"{record['evaluations']} evaluations in {refined_s:.2f} s.")
        if not guard["trusted"]:
            # announced whatever verbose says: the result keeps the terms, but they are fit parameters
            print("SSB: the fitted higher orders are not the probe's aberrations here: " + "; ".join(guard["reasons"])
                  + ". They stay in result.aberrations; optimizer['higher_order']['guard'] has the numbers.")
        return self._finish_fit(refined)

    def _unit_wave(self, phase):
        """The unit-amplitude wave exp(i phase) of a phase-only reconstruction, on the session's device."""
        return cp.exp(1j * phase) if self.backend == "cuda" else np.exp(1j * np.asarray(phase))

    def _output_sampling(self, upsampling_factor: int) -> tuple[float, float]:
        """Output pixel size in Angstrom, (row, col), at ``upsampling_factor`` times the scan grid."""
        return tuple((np.asarray(self.scan_sampling_A) / upsampling_factor).tolist())

    @_on_scan_gpu
    def preview_context(self, num_bright_field_pixels: int):
        """Prepare a backend-owned reduced-BF interaction context."""

        return self._backend().preview_context(int(num_bright_field_pixels))

    @_on_scan_gpu
    def browser_state(self):
        """Return compact backend-neutral state for browser WebGPU."""

        state = self._backend().browser_state()
        source = self._browser_qem
        if source is None or self.scan_region is not None or tuple(source["scan_shape"]) != tuple(state.scan_shape):
            return state
        try:
            stat = source["path"].stat()
            with source["path"].open("rb") as handle:
                if handle.read(56)[24:].hex() != source["header_sha256"]:
                    return state
        except OSError:
            return state
        if (stat.st_size, stat.st_mtime_ns) != source["signature"]:
            return state
        rows, cols = state.brightfield.rows, state.brightfield.cols
        if self.scan_detector_rotation_flipped:
            rows, cols = cols, rows
        rows, cols = rows + self._crop_origin[0], cols + self._crop_origin[1]
        if (np.any(rows < 0) or np.any(cols < 0)
                or np.any(rows >= source["detector_shape"][0])
                or np.any(cols >= source["detector_shape"][1])
                or not source["valid_pixels"][rows, cols].all()):
            # Dense SSB masks invalid pixels after loading. Its exact BF
            # export preserves that preprocessing; the original QEM does not.
            return state
        return replace(state, qem_source={
            name: source[name] for name in ("path", "scan_shape", "detector_shape", "dtype", "header_sha256")
        } | {"detector_origin_px": self._crop_origin,
             "scan_detector_rotation_flipped": self.scan_detector_rotation_flipped})

    @_on_scan_gpu
    def export_brightfield(
        self,
        path_stem: str | Path,
    ) -> tuple[str, float] | None:
        """Persist exact raw-count bright-field columns when supported."""

        written = self._backend().export_brightfield(self._backend_input(), path_stem)
        if written is None:
            return None
        path, elapsed_s = written
        return str(path), float(elapsed_s)

    @property
    def input_scan_shape(self) -> tuple[int, int]:
        """Measured scan dimensions before backend padding, in ``(row, col)`` order.

        A selected region reports its own dimensions. ``scan_shape`` instead
        describes the prepared Fourier grid, which may include padding.
        Reading this metadata does not prepare or reconstruct the acquisition.
        """
        if self._data is None:
            raise RuntimeError("This SSB session is closed. Reopen the acquisition to inspect its scan.")
        if isinstance(self._data, MpsBfColumnFrames):
            shape = self._data.scan_shape
        elif self._data.ndim == 4:
            shape = self._data.shape[:2]
        elif self._scan_shape is not None:
            shape = self._scan_shape
        else:
            side = math.isqrt(int(self._data.shape[0]))
            if side * side != self._data.shape[0]:
                raise ValueError("A nonsquare flattened scan needs scan_shape=(rows, cols).")
            shape = (side, side)
        return tuple(int(size) for size in shape)

    @property
    @_on_scan_gpu
    def scan_shape(self) -> tuple[int, int]:
        """Prepared scan shape in public ``(row, col)`` order."""

        return self._streamed.scan_shape if self._streamed is not None else self._backend().scan_shape

    @property
    @_on_scan_gpu
    def num_bright_field_pixels(self) -> int:
        """Number of pixels in the complete detected bright-field disk."""

        return self._streamed.num_bright_field_pixels if self._streamed is not None else self._backend().num_bright_field_pixels

    @_on_scan_gpu
    def set_rotation(self, scan_detector_rotation_deg: float) -> None:
        """Set the physical scan-to-detector rotation (degrees, kept in [0, 360)) and refresh geometry."""

        self.scan_detector_rotation_deg = wrap_degrees(scan_detector_rotation_deg, 360.0)
        if self._streamed is None or self._compute is not None:
            self._backend().cache_rotation(math.radians(self.scan_detector_rotation_deg))

    @_on_scan_gpu
    def close(self) -> None:
        """Release the session's GPU state; data passed to ``SSB(data)`` stays the caller's.

        The session owns only what it made: the bright-field crop decoded from a
        loaded dataset, backends' Fourier data, and the encoded acquisition
        retained by an ``SSB.open`` session that reads it encoded (a large CUDA
        scan, or integer counts on MPS). A borrowed array or dataset is never
        released here.
        """

        self._data = None
        self._streamed = None
        if self._owned_dataset is not None:
            self._owned_dataset.close()
            self._owned_dataset = None
        # the MPS backend's final allocator flush must run after the workflow has dropped the source and the backend
        backend, self._compute = self._compute, None
        if backend is not None:
            backend.close()

    def __enter__(self) -> Self:
        """Return this prepared SSB session."""

        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        """Release backend resources when leaving a context manager."""

        self.close()


__all__ = ["SSB"]
