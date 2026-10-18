"""Backend-neutral scientist-facing SSB workflow.

This module owns the only public stateful SSB entry point. Backend modules own
device preparation and kernels, but they do not define a second user API.
"""

import hashlib
import math
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import torch

from quantem.gpu import detector, io
from quantem.gpu.device import resolve
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.io.metadata_display import calibration_rows
from quantem.gpu.io.representation import DataRepresentation
from quantem.gpu.optics.physics import wrap_degrees
from quantem.gpu.ssb import series
from quantem.gpu.ssb._execution import cancellation_scope
from quantem.gpu.ssb.brightfield import crop_bright_field, disk_edge_radius
from quantem.gpu.ssb.contract import RefineMethod, SSBProtocol
from quantem.gpu.ssb.mps.backend import MpsSSBBackend
from quantem.gpu.ssb.mps.frames import MpsBfColumnFrames, find_bf_columns
from quantem.gpu.ssb.persistence import (
    RESULT_SCHEMA,
    json_value,
    load_result,
    result_paths,
    save_result,
    software_signature,
    source_signature,
)
from quantem.gpu.ssb.results import (
    COLUMN_SIGN_MIN,
    SSBResult,
    column_sign,
    draw_column_histogram,
    host_array,
)
from quantem.gpu.ssb.trials import (
    choose_trials,
    label_trials,
    plot_trials,
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


def _default_phase_estimator(aberrations: dict[str, float]) -> str:
    """Use coherent reconstruction for every aberration on both backends."""
    return "phase_of_mean"


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
            "Pass those calibrated values to SSB(data, ...) or SSB.open(...); scan_sampling_A is (row, col) in Angstrom. "
            "A diffraction disk fit cannot determine the real-space scan step."
        )
    return (float(values["voltage_kV"]), float(values["semiangle_mrad"]),
            sampling_pair("scan_sampling_A", values["scan_sampling_A"]))


def _in_notebook() -> bool:
    """True inside a Jupyter kernel, where fit() draws its histograms."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config


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
    import pandas as pd

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
    agree. ``find_aberrations`` fits C10_nm, C12_nm and phi12_deg by minimising that disagreement (the phase variance over
    bright-field pixels); ``reconstruct`` and ``preview`` apply known aberrations. The full automatically detected
    bright-field disk, the exact phase-variance objective, float32 real and complex64 object storage are fixed.

    C10/C12 reconstruction averages corrected complex waves before taking the
    phase (``phase_of_mean``) on CUDA and MPS. Both support finer output sampling
    through ``upsampling_factor``. Bright-field selection, loss reduction order
    and Nelder-Mead settings remain backend-specific, so fitted values and images
    need not agree bit for bit.

    Parameters
    ----------
    data
        The dataset ``io.load`` returns (only its bright-field crop is decoded), or a NumPy, CuPy or Torch array of
        counts, 4D ``(scan_row, scan_col, det_row, det_col)`` or 3D ``(scan, det_row, det_col)`` with ``scan_shape``.
    backend : {"auto", "cuda", "mps"}
        Accelerator; there is no CPU fallback.
    voltage_kV : float, optional
        Accelerating voltage in kV; defaults to the loaded dataset's metadata.
    semiangle_mrad : float, optional
        Probe convergence semiangle in mrad; defaults to the loaded dataset's metadata.
    scan_sampling_A : (float, float), optional
        Scan step in Angstrom, (row, col); defaults to the loaded dataset's metadata.
        These three values remain required for an array without metadata.
    scan_shape : (int, int), optional
        Scan shape of a 3D frame stack.
    detector_sampling_mrad : (float, float), optional
        Detector sampling in mrad per pixel, (row, col). When omitted, ``semiangle_mrad`` divided by the radius in pixels at which
        the bright-field disk of the full-detector mean pattern falls to half its plateau (``disk_edge_radius``).
        A loaded dataset's recorded detector sampling takes priority over that
        estimate; explicit keyword values take priority over metadata.
    aberrations : dict, optional
        Starting ``C10_nm``, ``C12_nm`` (nm) and ``phi12_deg`` (degrees).
    scan_detector_rotation_deg : float, optional
        Physical scan-to-detector rotation in degrees; kept in [0, 360).
        Pass the recorded ``data.metadata["scan_detector_rotation_deg"]`` or
        your calibrated angle explicitly. Omission keeps a 0-degree starting
        default, labeled as such in ``calibration_table()``.
        An estimate known only modulo 180 degrees
        (``dpc.run``, ``screening.prepare``) may be given: ``find_aberrations(check_rotation=True)`` settles the branch.
    detector_transposed : bool
        Exchange the two detector axes before applying the rotation. Detector
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

    An encoded dataset stays open and the caller's. A result saved from it with ``find_aberrations(save_to=...)`` is
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
        scan_sampling_A: tuple[float, float] | None = None,
        scan_shape: tuple[int, int] | None = None,
        detector_sampling_mrad: tuple[float, float] | None = None,
        aberrations: dict[str, float] | None = None,
        scan_detector_rotation_deg: float | None = None,
        detector_transposed: bool = False,
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
        metadata = {} if source_dataset is None else source_dataset.metadata
        voltage_kV, semiangle_mrad, scan_sampling_A = _source_calibration(
            metadata, voltage_kV, semiangle_mrad, scan_sampling_A)
        if source_dataset is not None:
            if detector_sampling_mrad is None:
                detector_sampling_mrad = metadata.get("detector_sampling_mrad")
            if source_path is None:
                source_path = metadata.get("source_path")
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
                            detector_transposed=detector_transposed,
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
            elif self.scan_region is not None:
                row_start, row_stop, col_start, col_stop = self.scan_region
                data = data.data[row_start:row_stop, col_start:col_stop]
            else:
                data = data.data
        elif scan_region is not None:
            raise ValueError("scan_region selects scan positions of a dataset io.load returns; slice an array before passing it.")
        # The backends read dense arrays, exact BF columns, or a borrowed large
        # ANS source whose detector columns are decoded during reconstruction.
        backend_array = cp.ndarray if self.backend == "cuda" else MpsBfColumnFrames
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
        self._data = (
            data.transpose_detector() if isinstance(data, MpsBfColumnFrames) else data.swapaxes(-1, -2)
        ) if detector_transposed and self._streamed is None else data
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
        self.aberrations = validate_aberrations(aberrations)
        self.scan_detector_rotation_deg = wrap_degrees(
            0.0 if scan_detector_rotation_deg is None else scan_detector_rotation_deg, 360.0)
        self.detector_transposed = bool(detector_transposed)
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

    def calibration_table(self) -> "pd.DataFrame":
        """Compare loaded file calibration with the settings this session uses.

        The table retains the metadata seen at construction and distinguishes
        explicit keywords (even equal values), saved overrides, defaults and
        probe estimates. It never changes the acquisition or starts a fit.
        A detector estimate is pending until the first backend preparation.
        ShowPtycho displays this table automatically when first shown.

        Returns
        -------
        pandas.DataFrame
            File value, value used, units and the source of each value used.
            File values are the loader's effective metadata, including saved
            calibration overrides; original source records stay in the dataset.

        Examples
        --------
        >>> data = io.load("scan.qem")
        >>> ssb = SSB(data, scan_detector_rotation_deg=12.3)
        >>> ssb.calibration_table()
        """
        import pandas as pd

        fields = {
            "voltage_kV": "Accelerating voltage",
            "semiangle_mrad": "Convergence semiangle",
            "scan_sampling_A": "Scan step (row, col)",
            "detector_sampling_mrad": "Detector step (row, col)",
            "scan_detector_rotation_deg": "Scan-detector rotation",
        }
        rows = {}
        for name, label in fields.items():
            recorded = self._calibration_file[label]
            value = getattr(self, name)
            source = recorded["Metadata"]
            if name in self._calibration_supplied:
                source = "Explicit override"
            elif name == "scan_detector_rotation_deg":
                source = "Default"
            elif source == "Not recorded" or (
                name == "detector_sampling_mrad" and recorded["Unit"] != "mrad per pixel"
            ):
                source = "Estimated"
            initial = self._calibration_initial[name]
            if initial is not None and not np.array_equal(value, initial):
                source = "Modified in session"
            if name == "detector_sampling_mrad" and value is None:
                if self._compute is not None:
                    value = (self._compute._det_sampling if self.backend == "mps"
                             else self._compute.angular_sampling)
                    if self.detector_transposed:
                        value = value[::-1]
                else:
                    value = "Pending probe fit"
            if isinstance(value, (list, tuple)):
                value = tuple(value)
            rows[label] = {
                "File value": recorded["Value"], "File unit": recorded["Unit"],
                "Value used": value,
                "Unit used": "mrad per pixel" if name == "detector_sampling_mrad" else recorded["Unit"],
                "Source": source,
            }
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("SSB launch calibration")

    def _decode_resident(self, data, bright_field_center_px, bright_field_radius_px,
                         detector_sampling_mrad, threshold, semiangle_mrad, mean_pattern, *, identify=True):
        """Decode one fit-sized crop and identify its exact resident evidence."""
        shape, dtype = tuple(int(value) for value in data.shape), str(data.dtype)
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
        frames = data.reshape(data.shape[0] * data.shape[1], -1)
        if self.backend == "cuda":
            per_frame = frames.sum(axis=1, dtype=cp.uint64)
            positions = cp.arange(1, per_frame.size + 1, dtype=cp.uint64)
        else:
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
        return data.swapaxes(-1, -2) if self.detector_transposed else data

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
        scan_sampling_A: tuple[float, float] | None = None,
        scan_shape: tuple[int, int] | None = None,
        detector_sampling_mrad: tuple[float, float] | None = None,
        aberrations: dict[str, float] | None = None,
        scan_detector_rotation_deg: float | None = None,
        detector_transposed: bool = False,
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

        Omitted voltage, convergence semiangle and scan step come from the
        acquisition's calibrated metadata. Missing physical values raise an
        error rather than being guessed. Detector sampling uses an explicit
        value, then stored calibration, then the bright-field disk edge and
        the known convergence semiangle. Explicit values always take priority.

        Pass a saved ``bright_field_center_px``, ``bright_field_radius_px``
        and ``detector_sampling_mrad`` together when replaying a fit. The
        centre remains in the original full-detector ``(row, col)`` axes,
        including when ``detector_transposed=True``.
        """

        supplied = {
            "voltage_kV": voltage_kV, "semiangle_mrad": semiangle_mrad,
            "scan_sampling_A": scan_sampling_A, "detector_sampling_mrad": detector_sampling_mrad,
            "scan_detector_rotation_deg": scan_detector_rotation_deg,
        }
        selected = resolve(backend)
        crop_origin = (0, 0)
        owned = None
        source_detector_bin = 1
        # on MPS an exact BF-column export beside the source replaces decoding the detector
        frames = find_bf_columns(source, calibration, verbose=verbose) if selected == "mps" else None
        if frames is not None:
            file_calibration = calibration_rows(frames.metadata)
            data = frames
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
            try:
                voltage_kV, semiangle_mrad, scan_sampling_A = _source_calibration(
                    loaded.metadata, voltage_kV, semiangle_mrad, scan_sampling_A)
                if detector_sampling_mrad is None:
                    detector_sampling_mrad = loaded.metadata.get("detector_sampling_mrad")
                if _streams_scan(loaded, selected):
                    owned = loaded
                    data = loaded
                else:
                    data, bright_field_center_px, bright_field_radius_px, calibration_radius, crop_origin, _ = crop_bright_field(
                        loaded, selected, bright_field_intensity_threshold, bright_field_radius_px,
                        calibrate_detector=detector_sampling_mrad is None,
                        bright_field_center_px=bright_field_center_px,
                    )
                source_dtype = str(loaded.dtype)
                source_bytes = loaded.logical_bytes
                correction = loaded.metadata.get("hot_pixel_correction")
            finally:
                if owned is None:
                    loaded.close()
            if detector_sampling_mrad is None and owned is None:
                detector_sampling_mrad = (semiangle_mrad / calibration_radius,) * 2
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
                detector_transposed=detector_transposed,
                bright_field_intensity_threshold=bright_field_intensity_threshold,
                bright_field_radius_px=bright_field_radius_px,
                source_path=str(source),
                bright_field_center_px=bright_field_center_px,
            )
        except Exception:
            if owned is not None:
                owned.close()
            raise
        session._calibration_file = file_calibration
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
        ...     fitted = region.find_aberrations(tilt=True, num_trials=200)
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
        options = {
            "backend": self.backend, "voltage_kV": self.voltage_kV,
            "semiangle_mrad": self.semiangle_mrad, "scan_sampling_A": self.scan_sampling_A,
            "detector_sampling_mrad": self.detector_sampling_mrad,
            "bright_field_center_px": self.bright_field_center_px,
            "bright_field_radius_px": self.bright_field_radius_px,
            "bright_field_intensity_threshold": self.bright_field_intensity_threshold,
            "detector_transposed": self.detector_transposed, "source_path": self.source_path,
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
                    center = center[::-1] if self.detector_transposed else center
                    options["bright_field_center_px"] = tuple(float(value) + origin for value, origin in zip(center, self._crop_origin))
                if options["bright_field_radius_px"] is None:
                    options["bright_field_radius_px"] = radius
                if options["detector_sampling_mrad"] is None:
                    options["detector_sampling_mrad"] = (self.semiangle_mrad / disk_edge_radius(pattern),) * 2
            counts = self._data[row_start:row_stop, col_start:col_stop]
            # __init__ applies the detector convention once. The parent's
            # resident array has already had it applied.
            if self.detector_transposed:
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
                    "higher_order_model": "krivanek14",
                    "upsampling_dc_policy": "single_origin",
                },
                "source": (
                    {"detector": self.resident_identity, "storage": None, "calibration": None, "manifest": None}
                    if self.source_path is None and self.resident_identity is not None else
                    source_signature(
                        self.source_path, scan_shape=self._scan_shape, storage_path=self.source_storage_path,
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
                    "source_provenance": self.source_provenance,
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
                    "detector_transposed": self.detector_transposed,
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
        result.detector_transposed = self.detector_transposed
        if result.bright_field_center_px is None:
            result.bright_field_center_px = self.bright_field_center_px
        # Only a backend that already holds prepared data is re-rotated; otherwise the change would trigger a load.
        if result.scan_detector_rotation_deg != self.scan_detector_rotation_deg and self._compute is not None:
            # a reused result, or one whose branch the polarity check changed, can sit on the other 180-degree branch; previews must use its rotation
            self.set_rotation(result.scan_detector_rotation_deg)
        self.aberrations = dict(result.aberrations)
        self.sample_tilt_mrad, self.depth_spread_nm = result.sample_tilt_mrad, result.depth_spread_nm
        self._aberrations_explicit = True
        self.scan_detector_rotation_deg = result.scan_detector_rotation_deg
        self.best_loss = (
            float(result.loss) if result.loss is not None else float("inf")
        )
        self._reconstruction = result
        return result

    def _resolve_rotation_branch(self, result: SSBResult, *, refinement: RefineMethod, tilt: bool,
                                 tilt_limit_mrad: float, seed: int, verbose: bool) -> SSBResult:
        """Keep the CoM direction whose phase has bright atom columns (see ``find_aberrations(check_rotation=...)``).

        The CoM curl fixes the scan-detector rotation only up to 180 degrees; the other branch is the angle plus 180
        degrees, every CoM vector reversed. When fitted atom columns are dark, the
        rotation moves to that branch, with C10, C12 and tilt flipped in sign at
        the same astigmatism angle. The other branch is the conjugate object.
        The joint model's objective is invariant under this transformation.
        CUDA reuses the fitted parameters and reconstructs the other branch;
        the other fitting paths retain their existing refinement behavior.
        """
        sign = column_sign(result.phase)
        result.column_sign = sign
        result.initial_scan_detector_rotation_deg = result.scan_detector_rotation_deg
        result.initial_column_sign = sign
        initial_phase = host_array(result.phase).copy()
        result.rotation_check_phases = np.stack((initial_phase, initial_phase))
        show = verbose and _in_notebook()
        input_rotation = result.initial_scan_detector_rotation_deg
        if not math.isfinite(sign) or sign >= -COLUMN_SIGN_MIN:
            if verbose and (not math.isfinite(sign) or sign < COLUMN_SIGN_MIN):
                print(f"SSB: polarity score {sign:+.2f} is inconclusive; keeping the suggested rotation {input_rotation:.2f}°.")
            return result
        limit = draw_column_histogram(result.phase, f"Suggested rotation {input_rotation:.2f}° · polarity score {sign:+.2f}") if show else None
        # the other branch; the subtraction is exact where adding 180 and wrapping at 360 would round
        opposite_rotation = input_rotation - 180.0 if input_rotation >= 180.0 else input_rotation + 180.0
        start = {"C10_nm": -result.aberrations["C10_nm"], "C12_nm": -result.aberrations["C12_nm"], "phi12_deg": result.aberrations["phi12_deg"]}
        reuse_joint_fit = tilt and self.backend == "cuda"
        if verbose:
            print(f"SSB: negative phase polarity at the suggested rotation {input_rotation:.2f}° (score {sign:+.2f}).\n"
                  f"     Testing {opposite_rotation:.2f}° under the bright-column assumption."
                  + (" Reusing the joint fit; no second search." if reuse_joint_fit else " Refining the opposite branch."))
        self.set_rotation(opposite_rotation)
        backend = self._backend()
        if reuse_joint_fit:
            opposite = replace(
                result, aberrations=start, scan_detector_rotation_deg=opposite_rotation,
                sample_tilt_mrad=tuple(-value for value in result.sample_tilt_mrad),
            )
            # Recompute the native phase and diagnostic at the selected branch.
            # Simply negating stored pixels would miss discrete Nyquist effects.
            reconstruction = self.reconstruct(
                opposite, phase_estimator=result.phase_estimator, force=True,
            )
            refit = replace(
                opposite, object_wave=reconstruction.object_wave,
                loss=reconstruction.loss, elapsed_s=reconstruction.elapsed_s,
            )
        elif tilt:
            refit = self._fit_tilt(result.num_trials or 200, refinement, tilt_limit_mrad, seed, False)
        else:
            refit = self._in_detector_frame(result_from_engine(backend.fit(
                aberrations=aberrations_to_engine(start), trials=0,
                refinement=refinement or "nelder-mead", search_ranges=None,
                refine_lock=None, seed=seed, verbose=False,
            )))
            refit.num_trials = result.num_trials
            refit.trial_records = []
        if not reuse_joint_fit:
            label_trials(refit, search=1)
            refit.trial_records = list(result.trial_records or ()) + list(refit.trial_records or ())
        for index, trial in enumerate(refit.trial_records):
            trial["trial"] = index
        refit.column_sign = column_sign(refit.phase)
        refit.initial_scan_detector_rotation_deg = input_rotation
        refit.initial_column_sign = sign
        refit.rotation_check_phases = np.stack((initial_phase, host_array(refit.phase)))
        if refit.elapsed_s is not None and result.elapsed_s is not None:
            refit.elapsed_s += result.elapsed_s   # First pass plus branch selection.
        if show:
            draw_column_histogram(refit.phase, f"SSB-selected rotation {refit.scan_detector_rotation_deg:.2f}° · polarity score {refit.column_sign:+.2f}", limit=limit)
        if verbose:
            print(f"SSB: selected rotation {refit.scan_detector_rotation_deg:.2f}°; C10 {refit.aberrations['C10_nm']:+.2f} nm, "
                  f"polarity score {refit.column_sign:+.2f}. find_aberrations(check_rotation=False) keeps the input rotation.")
        return refit

    def _center_in_data(self) -> tuple[float, float] | None:
        """The bright-field centre in pixels of the array the session holds, which the backends index.

        The public centre counts from the first pixel of the full detector; a session that decoded only the
        bright-field crop holds an array that starts ``_crop_origin`` later.
        """
        if self.bright_field_center_px is None:
            return None
        center = (self.bright_field_center_px[0] - self._crop_origin[0],
                  self.bright_field_center_px[1] - self._crop_origin[1])
        return center[::-1] if self.detector_transposed else center

    def _sampling_in_data(self) -> tuple[float, float] | None:
        """Detector sampling in the axes passed to the numerical backend."""
        sampling = self.detector_sampling_mrad
        return sampling[::-1] if sampling is not None and self.detector_transposed else sampling

    def _in_detector_frame(self, result: SSBResult) -> SSBResult:
        """Move a result's bright-field centre from the array the session holds to full-detector pixels, in place.

        Backends index the array they were given. Without the shift a session built from an encoded acquisition
        reports a centre inside its crop, a different point from the one a dense session reports.
        """
        if result.bright_field_center_px is not None:
            if self.detector_transposed:
                result.bright_field_center_px = result.bright_field_center_px[::-1]
            result.bright_field_center_px = (result.bright_field_center_px[0] + self._crop_origin[0],
                                             result.bright_field_center_px[1] + self._crop_origin[1])
        result.detector_transposed = self.detector_transposed
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
                self.detector_sampling_mrad = tuple(sampling[::-1] if self.detector_transposed else sampling)
                self._calibration_initial["detector_sampling_mrad"] = self.detector_sampling_mrad
        return self._compute

    def _reconstruct_trials(self, parameters, trials, *, upsampling_factor, phase_estimator):
        """Replay selected records without replacing the current search or correction."""
        parameters = self._aberration_search if parameters is None else parameters
        if parameters is None:
            raise ValueError("Run ssb.find_aberrations() before reconstructing trials.")
        records = select_records(parameters.trial_records, trials)
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
            detector_transposed=self.detector_transposed,
            voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
            object_sampling_A=self._output_sampling(upsampling_factor), upsampling_factor=upsampling_factor, phase_estimator=estimator,
            amplitude_estimated=False, source_path=self.source_path,
            metadata={"trial_ids": [record["trial"] for record in records]},
        )

    def show_trials(self, *, best: int | None = None, last: int | None = None,
                    first: int | None = None, upsampling_factor: int = 1,
                    axsize: tuple[float, float] = (6, 6)):
        """Show selected attempts as phase images with a compact parameter table.

        Parameters
        ----------
        best, last, first
            Supply exactly one positive count. ``best`` selects the lowest
            search loss within the latest recorded search; ``last`` and ``first``
            select completed attempts in time order. Local refinement is not
            an Optuna trial and is displayed separately in the final report.
        upsampling_factor
            Output sampling factor; never reruns the parameter search.
        axsize
            Width and height of each phase panel in inches. At most two
            columns are shown; additional trials continue on the next row.

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
        """
        if self._aberration_search is None:
            raise ValueError("Run ssb.find_aberrations() before showing trials.")
        records = choose_trials(self._aberration_search.trial_records, best=best, last=last, first=first)
        result = self.reconstruct(self._aberration_search,
                                  trials=[record["trial"] for record in records], upsampling_factor=upsampling_factor)
        kind = "Best" if best is not None else "Last" if last is not None else "First"
        return plot_trials(result, kind, axsize=axsize)

    @_on_scan_gpu
    def find_aberrations(
        self,
        *,
        tilt: bool = False,
        scan_region: tuple[int, int, int, int] | None = None,
        num_trials: int = 200,
        refinement: RefineMethod | Literal["auto"] = "auto",
        search_ranges: dict[str, tuple[float, float] | float] | None = None,
        refine_lock: list[str] | None = None,
        tilt_limit_mrad: float = 25.0,
        check_rotation: bool = True,
        seed: int = 42,
        save_to: str | Path | None = None,
        force: bool = False,
        verbose: bool = True,
        compare_standard: bool = True,
        fixed_aberrations: dict[str, float] | None = None,
    ) -> SSBResult:
        """Optimize C10_nm/C12_nm/phi12_deg and return the final reconstruction.

        ``tilt=True`` also fits the sample tilt and a depth spread for a thick, tilted crystal, jointly with the
        aberrations in one search of ``num_trials`` trials plus local refinement (fitting the tilt after a standard fit gets stuck:
        C10 has to move to the defocus at mid-depth at the same time). The phase-variance loss of the standard fit does
        not see tilt, so this search maximises the least-squares agreement of the thick-sample model with the data. The
        result's ``sample_tilt_mrad`` (row, col; scan frame, within ``tilt_limit_mrad``) and ``depth_spread_nm`` hold the fit; CUDA and
        MPS. On two full 512 x 512 acquisitions 100 trials already converged every seed and 200-400 gave the same tilt
        to 0.02 mrad (docs/maintainer/2026-09-24-ssb-units-and-thick-sample.md).
        ``refinement="auto"`` (default) selects Newton for tilt fits and Nelder-Mead for standard fits.
        Explicit ``"nelder-mead"`` retains the previous optimizer; None skips local refinement.
        The 200-trial coarse search is unchanged. Saved settings record the selected optimizer, not ``"auto"``.
        ``refinement="newton"`` (tilt fits) polishes with Newton steps on finite-difference models whose evaluations share
        passes over the data, a few passes instead of hundreds (both backends evaluate the trials of a pass in one batch).
        ``compare_standard=False`` skips the standard-SSB search that only measures ``tilt_fit_gain`` (left None).
        ``num_trials=0`` with ``tilt=True`` searches nothing and refines from the session's ``aberrations``,
        ``sample_tilt_mrad`` and ``depth_spread_nm``: set them to the previous scan's fit and the next scan of a session
        costs one polish.

        ``fixed_aberrations`` holds C10_nm, C12_nm and phi12_deg common across
        local tilt fits. Supply a previously calibrated thick-sample optical
        model, ``tilt=True`` and ``check_rotation=False``. The same batched
        Newton solver optimizes only tilt and depth spread on CUDA and MPS.
        This conditional fit does not determine whether the common optics are
        correct. Use ``compare_standard=False`` to omit the optional thin fit.

        ``check_rotation=True`` (default) also settles the 180-degree ambiguity of the scan-detector rotation: the rotation
        search cannot tell omega from omega + 180 degrees and both fit equally well (the second is the conjugate object with
        the opposite aberration phase). Atom columns carry positive phase, so when the fitted phase has a negative column
        sign (``result.column_sign`` < -0.2) the session's rotation moves by 180 degrees (``scan_detector_rotation_deg``;
        ``initial_scan_detector_rotation_deg`` keeps the angle before the check), with C10, C12, and tilt reversed in sign. A CUDA joint fit
        reuses those fitted parameters and reconstructs the opposite branch
        without a second search. Other fitting paths retain their refinement.
        The inexpensive sign check always runs; extra reconstruction/refinement
        runs only for a finite score below -0.2. Bright-column polarity is a
        physical assumption, not a guarantee of absolute rotation: unresolved
        structure and contrast inversion in thick specimens can be ambiguous.
        With verbose notebook output, before/after histograms are drawn only
        when the rotation changes. The saved result also supports
        ``show("rotation", histogram=True)`` for an explained comparison.
        Near zero (no resolved columns) the rotation is kept.
        ``check_rotation=False`` keeps the rotation as given.

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

        Set ``save_to`` to reuse an exact prior result from ``SSB.open`` or from an encoded dataset when the detector
        source, calibration, backend, physical parameters, and fit settings all match.
        Changed settings recompute automatically. Set ``force=True`` to recompute
        an otherwise matching result. Direct array inputs are saved but always
        recomputed: a source path does not identify an array's crop or mutations.
        Zero trials start from the current session coefficients: the standard search refines them, and ``tilt=True``
        refines them with the session's sample tilt and depth spread.
        """

        if tilt and (search_ranges is not None or refine_lock is not None):
            raise ValueError("search_ranges and refine_lock apply to the standard fit; tilt=True searches its own ranges.")
        if num_trials < 0:
            raise ValueError(f"num_trials must be non-negative, got {num_trials}.")
        if tilt and int(num_trials) == 0 and (self.sample_tilt_mrad is None or self.depth_spread_nm is None):
            raise ValueError(
                "find_aberrations(tilt=True, num_trials=0) refines from the session's aberrations, sample_tilt_mrad and "
                "depth_spread_nm: set the last two (a previous scan's fit) first, or use num_trials=200."
            )
        if refinement == "auto":
            refinement = "newton" if tilt else "nelder-mead"
        if fixed_aberrations is not None:
            fixed_aberrations = validate_aberrations(fixed_aberrations)
            if not all(math.isfinite(v) for v in fixed_aberrations.values()):
                raise ValueError("fixed_aberrations must contain finite coefficients.")
            if not tilt or refinement != "newton" or check_rotation:
                raise ValueError("Use fixed_aberrations with tilt=True, refinement='newton' (or 'auto'), and check_rotation=False.")
        if refinement not in {"nelder-mead", "newton", None}:
            raise ValueError("refinement must be 'auto', 'nelder-mead', 'newton' or None.")
        if refinement == "newton" and not tilt:
            raise ValueError("refinement='newton' polishes the tilt fit; the standard fit refines with 'nelder-mead'.")
        paths = None
        signature = None
        if save_to is not None:
            paths = result_paths(save_to, "find_aberrations")
            signature = self._result_signature(
                "find_aberrations",
                {
                    "num_trials": int(num_trials),
                    "refinement": refinement,
                    "search_ranges": search_ranges,
                    "refine_lock": refine_lock,
                    "seed": int(seed),
                    "starting_aberrations": self.aberrations,
                    "starting_aberrations_explicit": self._aberrations_explicit,
                    **({"tilt": True, "tilt_limit_mrad": float(tilt_limit_mrad)} if tilt else {}),
                    **({"compare_standard": False} if tilt and not compare_standard else {}),
                    **({"refined_from": {"sample_tilt_mrad": self.sample_tilt_mrad, "depth_spread_nm": self.depth_spread_nm}}
                       if tilt and int(num_trials) == 0 else {}),
                    "check_rotation": bool(check_rotation),
                    "scan_region": scan_region,
                    **({"fixed_aberrations": fixed_aberrations} if fixed_aberrations is not None else {}),
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
                                  "detector_transposed": self.detector_transposed}
            if center is None or radius is None or detector_sampling_mrad is None:
                mean_pattern = detector.mean(self._data)
                detected_center, detected_radius = detector.fit_probe(mean_pattern)
                center = detected_center if center is None else center
                radius = detected_radius if radius is None else radius
                if detector_sampling_mrad is None:
                    detector_sampling_mrad = (self.semiangle_mrad / disk_edge_radius(mean_pattern),) * 2
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
                result = regional.find_aberrations(
                    tilt=tilt, num_trials=num_trials, refinement=refinement,
                    search_ranges=search_ranges, refine_lock=refine_lock,
                    tilt_limit_mrad=tilt_limit_mrad, check_rotation=check_rotation,
                    seed=seed, verbose=verbose, compare_standard=compare_standard,
                    fixed_aberrations=fixed_aberrations,
                )
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
            result = self._accept_result(self._fit_tilt(int(num_trials), refinement, float(tilt_limit_mrad), int(seed), verbose,
                                                      fixed_aberrations=fixed_aberrations))
        else:
            result = self._backend().fit(
                aberrations=(
                    aberrations_to_engine(self.aberrations)
                    if self._aberrations_explicit else None
                ),
                trials=int(num_trials),
                refinement=refinement,
                search_ranges=search_ranges_to_engine(search_ranges),
                refine_lock=refine_lock_to_engine(refine_lock),
                seed=int(seed),
                verbose=verbose,
            )
            result = self._accept_result(self._in_detector_frame(result_from_engine(result)))
        label_trials(result, search=0)
        if check_rotation:
            result = self._accept_result(self._resolve_rotation_branch(
                result, refinement=refinement, tilt=tilt, tilt_limit_mrad=float(tilt_limit_mrad), seed=int(seed),
                verbose=verbose))
        estimator = _default_phase_estimator(result.aberrations)
        reconstruction_started = time.perf_counter()
        displayed = self.reconstruct(result, phase_estimator=estimator, compute_loss=False, force=True)
        result.timings["final_native_reconstruction_s"] = time.perf_counter() - reconstruction_started
        result.object_wave = displayed.object_wave
        result.phase_clim = result.phase_limits
        result.phase_estimator = displayed.phase_estimator
        result.amplitude_estimated = False
        self._accept_result(result)
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
            evidence = measure_slices(
                lambda rows: backend.sample_agreement(rows, band), point,
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
        trials: object = None,
        upsampling_factor: int | Literal["auto"] = 1,
        phase_estimator: Literal["mean_phase", "phase_of_mean", "complex_wave"] | None = None,
        compute_loss: bool = True,
        save_to: str | Path | None = None,
        force: bool = False,
        verbose: bool = False,
    ) -> SSBResult:
        """Reconstruct with fixed correction parameters, without refitting.

        Parameters
        ----------
        parameters
            Result of ``find_aberrations``. Carries aberrations, specimen tilt,
            depth spread, and scan-detector rotation. None uses session values.
        aberrations
            Partial coefficient overrides: ``C10_nm``/``C12_nm`` in nm, ``phi12_deg`` in degrees.
            Unspecified coefficients and sample geometry are preserved.
        trials
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
            The loss is the native-grid search
            diagnostic, not a score computed from the displayed phase.
            ``complex_wave`` explicitly retains amplitude for native, thin-sample
            reconstruction, including coherent temporal averaging.
        compute_loss
            Evaluate diagnostic loss on the native grid, regardless of output
            factor. False skips this diagnostic without changing the image.
        save_to
            Directory or .npz file for the result and correction metadata.
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
        >>> aberrations = ssb.find_aberrations(tilt=True)
        >>> result = ssb.reconstruct(aberrations, upsampling_factor=4, save_to="results/4x")
        """
        if upsampling_factor == "auto":
            from quantem.gpu.optics.physics import ssb_upsampling_factor

            upsampling_factor = ssb_upsampling_factor(
                voltage_kV=self.voltage_kV,
                semiangle_mrad=self.semiangle_mrad,
                scan_sampling_A=self.scan_sampling_A,
            )
        if type(upsampling_factor) is not int or upsampling_factor not in (1, 2, 3, 4, 8):
            raise ValueError("upsampling_factor must be 'auto', 1, 2, 3, 4, or 8.")
        if phase_estimator not in (None, "mean_phase", "phase_of_mean", "complex_wave"):
            raise ValueError("phase_estimator must be 'mean_phase', 'phase_of_mean', or 'complex_wave'.")
        if parameters is not None and not isinstance(parameters, SSBResult):
            raise TypeError("Pass the result of find_aberrations(), or supply coefficients with aberrations={...}.")
        if trials is not None:
            if phase_estimator == "complex_wave":
                raise ValueError("Trial inspection returns phase images; choose mean_phase or phase_of_mean.")
            if aberrations is not None or save_to is not None:
                raise ValueError("Replay trials without coefficient overrides or save_to; save the returned result separately.")
            return self._reconstruct_trials(parameters, trials, upsampling_factor=upsampling_factor, phase_estimator=phase_estimator)
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
        if save_to is not None:
            paths = result_paths(save_to, "reconstruct")
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
        if (not force and save_to is None and parameters is None and aberrations is None
                and cached is not None and cached.upsampling_factor == upsampling_factor
                and cached.phase_estimator == estimator
                and cached.scan_detector_rotation_deg == self.scan_detector_rotation_deg
                and cached.aberrations == coefs
                and cached.sample_tilt_mrad == tilt
                and cached.depth_spread_nm == depth
                and (not compute_loss or cached.loss is not None)):
            return cached
        if estimator == "complex_wave":
            result = self._in_detector_frame(result_from_engine(self._backend().reconstruct_result(
                aberrations_to_engine(coefs), compute_loss=compute_loss,
            )))
            result.phase_estimator = estimator
            result.amplitude_estimated = True
            result.phase_clim = correction.phase_limits if correction is not None else None
            result.fit_scan_region = correction.fit_scan_region if correction is not None else None
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
        )
        result = self._accept_result(result)
        if paths is not None and signature is not None:
            result = self._save_result(result, paths=paths, signature=signature)
            if verbose:
                print(f"SSB result saved to {paths[0]}")
        return result

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
                    thickness=float(depth_spread_nm) * ENGINE_PER_NM,
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
                    thickness=float(depth_spread_nm) * ENGINE_PER_NM,
                    phase_estimator=phase_estimator,
                    **higher,
                )
        if float(depth_spread_nm) > 0.0:
            sample = {"tilt_row_mrad": float(sample_tilt_mrad[0]), "tilt_col_mrad": float(sample_tilt_mrad[1]),
                      "thickness": float(depth_spread_nm) * ENGINE_PER_NM}
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
                  *, fixed_aberrations: dict[str, float] | None = None) -> SSBResult:
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
        fit = backend.fit_sample(trials=trials, tilt_limit_mrad=tilt_limit_mrad, seed=seed, verbose=verbose,
                                 polish_starts=0 if refinement is None else 3, polish=refinement or "nelder-mead",
                                 compare_standard=self._compare_standard, start=start, **fixed)
        aberrations = aberrations_from_engine(fit)
        sample_tilt_mrad = (float(fit["tilt_row_mrad"]), float(fit["tilt_col_mrad"]))
        depth_spread_nm = float(fit["thickness"]) / ENGINE_PER_NM
        diagnostic_started = time.perf_counter()
        phase, loss = self._phase(aberrations, sample_tilt_mrad=sample_tilt_mrad, depth_spread_nm=depth_spread_nm,
                                   phase_estimator="mean_phase")
        timings = dict(fit.get("timings", {}))
        timings["backend_preparation_s"] = backend_preparation_s
        timings["native_phase_diagnostic_s"] = time.perf_counter() - diagnostic_started
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
                "depth_spread_nm": values["thickness"] / ENGINE_PER_NM,
            })
        return SSBResult(object_wave=object_wave, trial_records=records, backend=self.backend, aberrations=aberrations, sample_tilt_mrad=sample_tilt_mrad,
                         timings=timings, optimizer=dict(fit.get("optimizer", {})), depth_spread_nm=depth_spread_nm, sample_agreement=float(fit["fit"]), tilt_fit_gain=None if fit["gain"] is None else float(fit["gain"]),
                         amplitude_estimated=False, phase_estimator="mean_phase",
                         scan_detector_rotation_deg=self.scan_detector_rotation_deg,
                         loss=None if loss is None else float(loss),
                         elapsed_s=time.perf_counter() - started, num_trials=trials, num_bright_field_pixels=self.num_bright_field_pixels,
                         refine_method=refinement, voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
                         object_sampling_A=self.scan_sampling_A)

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

        return self._backend().browser_state()

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
        retained by a large ``SSB.open`` session. A borrowed array or dataset
        is never released here.
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
