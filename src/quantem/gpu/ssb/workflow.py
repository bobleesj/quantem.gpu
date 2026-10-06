"""Backend-neutral scientist-facing SSB workflow.

This module owns the only public stateful SSB entry point. Backend modules own
device preparation and kernels, but they do not define a second user API.
"""

import math
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Literal, Self

import numpy as np

from quantem.gpu import detector, io
from quantem.gpu.device import resolve
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.io.representation import DataRepresentation
from quantem.gpu.ssb import series
from quantem.gpu.ssb.brightfield import crop_bright_field, disk_edge_radius
from quantem.gpu.ssb.contract import RefineMethod, SSBProtocol
from quantem.gpu.ssb.mps.backend import MpsSSBBackend
from quantem.gpu.ssb.mps.frames import MpsBfColumnFrames, find_bf_columns
from quantem.gpu.ssb.persistence import (
    SCHEMA,
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
    physical_rotation_deg,
    split_rotation,
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
    result_from_engine,
    search_ranges_to_engine,
    validate_aberrations,
)


def _in_notebook() -> bool:
    """True inside a Jupyter kernel, where fit() draws its histograms."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config


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
    return resolve(requested)


class SSB:
    """Single-sideband ptychography of one 4D-STEM acquisition, with the same science on CUDA and MPS.

    SSB recovers the object phase from the bright-field disk: each bright-field detector pixel sees the sample through
    a slightly different probe tilt, and once the probe aberrations are corrected for every pixel their phase images
    agree. ``find_aberrations`` fits C10, C12 and phi12 by minimising that disagreement (the phase variance over
    bright-field pixels); ``reconstruct`` and ``preview`` apply known aberrations. The full automatically detected
    bright-field disk, the exact phase-variance objective, float32 real and complex64 object storage are fixed.

    Parameters
    ----------
    data
        The dataset ``io.load`` returns (only its bright-field crop is decoded), or a NumPy, CuPy or Torch array of
        counts, 4D ``(scan_row, scan_col, det_row, det_col)`` or 3D ``(scan, det_row, det_col)`` with ``scan_shape``.
    backend : {"auto", "cuda", "mps", "webgpu"}
        Accelerator; there is no CPU fallback. WebGPU runs in the browser through the exported workflow, so it raises
        here and names the CLI that exports it.
    voltage_kV : float
        Accelerating voltage in kV.
    semiangle_mrad : float
        Probe convergence semiangle in mrad.
    scan_sampling_A : float or (float, float)
        Scan step in Angstrom, (row, col).
    scan_shape : (int, int), optional
        Scan shape of a 3D frame stack.
    det_sampling : float or (float, float), optional
        Detector sampling in mrad per pixel. When omitted, ``semiangle_mrad`` divided by the radius in pixels at which
        the bright-field disk of the full-detector mean pattern falls to half its plateau (``disk_edge_radius``).
    aberrations : dict, optional
        Starting C10, C12 (nm) and phi12 (radians).
    rotation_angle_deg : float
        Scan-detector rotation in degrees (any angle; stored below 180 with ``com_reversed``).
    com_reversed : bool
        Whether the centre-of-mass vectors point the other way (rotation + 180 degrees).
    bf_intensity_threshold : float
        Fraction of the mean-pattern maximum a bright-field pixel must exceed.
    bf_radius : float, optional
        Bright-field radius in detector pixels; the detected disk when omitted.
    source_path : str, optional
        The file the array came from, recorded with saved results.
    bf_center : (float, float), optional
        Bright-field centre (row, col) in detector pixels; detected when omitted.

    Examples
    --------
    >>> ssb = SSB(patterns, voltage_kV=300, semiangle_mrad=30,
    ...           scan_sampling_A=0.99, det_sampling=0.6081)
    >>> aberrations = ssb.find_aberrations()
    >>> result = ssb.reconstruct(aberrations)
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
        if self.backend == "webgpu":
            raise RuntimeError(
                "WebGPU SSB executes in the browser. Use `quantem showptycho "
                "--backend webgpu` so the CLI can export the same SSB plan and "
                "collect the shared result schema."
            )
        if isinstance(data, Dataset4dstemGPU):
            if data.ndim != 4:
                raise ValueError("SSB needs one 4D acquisition; select a dataset from the series first.")
            if data.representation is not DataRepresentation.DENSE:
                # Decode only the bright-field evidence into session-owned
                # tensors. The caller retains ownership of the acquisition.
                data, bf_center, bf_radius, calibration_radius = crop_bright_field(
                    data, self.backend, bf_intensity_threshold, bf_radius,
                    calibrate_detector=det_sampling is None, bf_center=bf_center,
                )
                if det_sampling is None:
                    det_sampling = semiangle_mrad / calibration_radius
            else:
                data = data.data
        # The backends read dense arrays, plus the exact BF columns SSB.open finds on MPS. An encoded
        # resident (``io.load(...).data``) has no bounded read of its own; only its dataset decodes the
        # bright-field crop, so it is refused here rather than at the first reconstruction.
        backend_array = cp.ndarray if self.backend == "cuda" else MpsBfColumnFrames
        if not isinstance(data, np.ndarray | backend_array):
            # Torch is imported only here, so NumPy and CuPy sessions run without it.
            import torch

            if not isinstance(data, torch.Tensor):
                raise TypeError(
                    f"SSB reads the dataset io.load returns or a NumPy, CuPy or Torch array; got "
                    f"{type(data).__name__}. Pass the loaded dataset itself, SSB(data, ...), not its "
                    "encoded storage data.data, or open the file with SSB.open(path, ...)."
                )
        if scan_shape is not None and data.ndim == 3 and not isinstance(data, backend_array):
            # a flat NumPy or Torch frame stack gets its scan axes as a view; CuPy stacks are reshaped by the
            # CUDA backend, and exact BF columns are stored flat
            rows, cols = int(scan_shape[0]), int(scan_shape[1])
            if rows * cols != int(data.shape[0]):
                raise ValueError(
                    f"scan_shape={scan_shape} describes {rows * cols} frames, but data "
                    f"contains {data.shape[0]}."
                )
            data = data.reshape(rows, cols, data.shape[1], data.shape[2])
        self._data = data
        self._scan_shape = scan_shape
        self.voltage_kV = float(voltage_kV)
        self.semiangle_mrad = float(semiangle_mrad)
        self.scan_sampling_A = scan_sampling_A
        self.det_sampling = det_sampling
        self._aberrations_explicit = aberrations is not None
        self.aberrations = validate_aberrations(aberrations)
        # public form: angle below 180 plus whether the CoM is reversed; the engines get the physical angle
        self.rotation_angle_deg, self.com_reversed = split_rotation(rotation_angle_deg, com_reversed)
        self.bf_intensity_threshold = float(bf_intensity_threshold)
        self.bf_radius = bf_radius
        # (row, col) detector pixels; set by SSB.open when it decodes only the bright-field crop of an encoded source
        self.bf_center = None if bf_center is None else (float(bf_center[0]), float(bf_center[1]))
        # latest find_aberrations(tilt=True): sample tilt (row, col) mrad, scan frame, and depth spread nm; None after a standard fit
        self.tilt_mrad: tuple[float, float] | None = None
        self.depth_spread_nm: float | None = None
        self.source_path = source_path
        self.calibration_path: str | None = None
        self.source_manifest_path: str | None = None
        self.source_storage_path = source_path
        self.source_kind: Literal["array", "detector", "bf_columns"] = "array"
        self.source_dtype = str(data.dtype)
        self.source_bytes = int(data.nbytes)
        # SSB.open replaces these two for exact BF columns, which record their own binning and provenance
        self.source_detector_bin = 1
        self.source_provenance: dict[str, object] | None = None
        self.source_load_seconds: float | None = None
        self._cuda_session = None
        self._mps_backend = None
        self._reconstruction: SSBResult | None = None
        self.best_loss = float("inf")
        self.trial_history: list[dict[str, object]] = []
        self._aberration_search: SSBResult | None = None

    # one implementation in ``series``, bound here so the public spelling is ``SSB.reconstruct_series``
    reconstruct_series = classmethod(series.reconstruct_series)

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
        verbose: bool = False,
    ) -> Self:
        """Open one lossless 4D-STEM source and prepare an SSB session.

        Exact BF-column storage is chosen automatically when it is available;
        otherwise detector counts are loaded with the explicitly requested
        storage dtype. Leave ``dtype=None`` for native detector precision.
        This storage choice never changes the float32/complex64 optimization
        precision or scientific objective.

        GPU acquisitions use ANS-encoded storage.
        """

        selected = _resolve_backend(backend)
        if selected == "webgpu":
            raise RuntimeError(
                "Browser WebGPU sources are opened by the exported SSB runtime."
            )
        bf_center = None
        source_detector_bin = 1
        # on MPS an exact BF-column export beside the source replaces decoding the detector
        frames = find_bf_columns(source, calibration, verbose=verbose) if selected == "mps" else None
        if frames is not None:
            data = frames
            source_kind = "bf_columns"
            source_storage_path = str(frames.source_path)
            source_dtype = str(frames.dtype)
            source_bytes = int(frames.nbytes)
            source_load_seconds = float(frames.load_seconds)
            source_detector_bin = frames.det_bin
            source_provenance = json_value(frames.source_provenance)
        else:
            load_started = time.perf_counter()
            with io.load(
                source,
                backend=selected,
                scan_shape=scan_shape,
                dtype=dtype,
                verbose=verbose,
            ) as loaded:
                data, bf_center, bf_radius, calibration_radius = crop_bright_field(
                    loaded, selected, bf_intensity_threshold, bf_radius,
                    calibrate_detector=det_sampling is None,
                )
                source_dtype = str(loaded.dtype)
                source_bytes = loaded.logical_bytes
                correction = loaded.metadata.get("hot_pixel_correction")
            if det_sampling is None:
                det_sampling = semiangle_mrad / calibration_radius
            source_provenance = None if correction is None else {"hot_pixel_correction": correction}
            source_kind = "detector"
            source_storage_path = str(source)
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

    def _prepare_cuda(self):
        """Construct the private CUDA implementation once."""

        if self._cuda_session is None:
            from quantem.gpu.ssb.cuda.backend import CudaSSBBackend

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
                    aberrations_to_engine(self.aberrations) if self._aberrations_explicit else None
                ),
                rotation_angle_deg=self.physical_rotation_deg,
            )
        return self._cuda_session

    def _result_signature(
        self,
        operation: Literal["find_aberrations", "reconstruct"],
        settings: dict[str, object],
    ) -> dict[str, object]:
        """Build the exact scientific identity of one SSB operation."""

        return json_value(
            {
                "schema": SCHEMA,
                "software": software_signature(),
                "operation": operation,
                "source": source_signature(
                    self.source_path, scan_shape=self._scan_shape, storage_path=self.source_storage_path,
                    calibration_path=self.calibration_path, manifest_path=self.source_manifest_path,
                ),
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
        # Only a backend that already holds prepared data is re-rotated; otherwise the change would trigger a load.
        backend_ready = self._cuda_session is not None or self._mps_backend is not None
        if result.physical_rotation_deg != self.physical_rotation_deg and backend_ready:
            # a reversed or reused result can sit on the other 180-degree branch; previews must use its rotation
            self.set_rotation(result.rotation_angle_deg, com_reversed=result.com_reversed)
        self.aberrations = dict(result.aberrations)
        self.tilt_mrad, self.depth_spread_nm = result.tilt_mrad, result.depth_spread_nm
        self._aberrations_explicit = True
        self.rotation_angle_deg, self.com_reversed = result.rotation_angle_deg, result.com_reversed
        self.best_loss = (
            float(result.loss) if result.loss is not None else float("inf")
        )
        self._reconstruction = result
        return result

    def _resolve_rotation_branch(self, result: SSBResult, *, refinement: RefineMethod, tilt: bool,
                                 tilt_limit_mrad: float, seed: int, verbose: bool) -> SSBResult:
        """Keep the CoM direction whose phase has bright atom columns (see ``find_aberrations(check_rotation=...)``).

        The CoM curl fixes the scan-detector rotation only up to 180 degrees; the other branch is the same angle with
        every CoM vector reversed. When fitted atom columns are dark, the CoM is
        reversed (the angle stays), with C10, C12 and tilt flipped in sign at
        the same astigmatism angle. The other branch is the conjugate object.
        The joint model's objective is invariant under this transformation.
        CUDA reuses the fitted parameters and reconstructs the other branch;
        the other fitting paths retain their existing refinement behavior.
        """
        sign = column_sign(result.phase)
        result.column_sign = sign
        result.initial_rotation_deg = result.physical_rotation_deg
        result.initial_column_sign = sign
        initial_phase = host_array(result.phase).copy()
        result.rotation_check_phases = np.stack((initial_phase, initial_phase))
        show = verbose and _in_notebook()
        angle = float(result.rotation_angle_deg)
        input_rotation = result.initial_rotation_deg
        if not math.isfinite(sign) or sign >= -COLUMN_SIGN_MIN:
            if verbose and (not math.isfinite(sign) or sign < COLUMN_SIGN_MIN):
                print(f"SSB: polarity score {sign:+.2f} is inconclusive; keeping the suggested rotation {input_rotation:.2f}°.")
            return result
        limit = draw_column_histogram(result.phase, f"Suggested rotation {input_rotation:.2f}° · polarity score {sign:+.2f}") if show else None
        reversed_now = not result.com_reversed
        start = {"C10": -result.aberrations["C10"], "C12": -result.aberrations["C12"], "phi12": result.aberrations["phi12"]}
        reuse_joint_fit = tilt and self.backend == "cuda"
        if verbose:
            print(f"SSB: negative phase polarity at the suggested rotation {input_rotation:.2f}° (score {sign:+.2f}).\n"
                  f"     Testing {(input_rotation + 180) % 360:.2f}° under the bright-column assumption."
                  + (" Reusing the joint fit; no second search." if reuse_joint_fit else " Refining the opposite branch."))
        self.set_rotation(angle, com_reversed=reversed_now)
        backend = self._backend_protocol
        if reuse_joint_fit:
            opposite = replace(
                result, aberrations=start, com_reversed=reversed_now,
                tilt_mrad=tuple(-value for value in result.tilt_mrad),
            )
            # Recompute the native phase and diagnostic at the selected branch.
            # Simply negating stored pixels would miss discrete Nyquist effects.
            reconstruction = self.reconstruct(
                opposite, phase_estimator=result.phase_estimator, force=True,
            )
            refit = replace(
                opposite, object_wave=reconstruction.object_wave,
                loss=reconstruction.loss, elapsed=reconstruction.elapsed,
            )
        elif tilt:
            refit = self._fit_tilt(result.n_trials or 200, refinement, tilt_limit_mrad, seed, False)
        else:
            refit = result_from_engine(backend.fit(
                aberrations=aberrations_to_engine(start), trials=0,
                refinement=refinement or "nelder-mead", search_ranges=None,
                refine_lock=None, seed=seed, verbose=False,
            ))
            refit.n_trials = result.n_trials
            refit.trial_records = []
        if not reuse_joint_fit:
            label_trials(refit, search=1)
            refit.trial_records = list(result.trial_records or ()) + list(refit.trial_records or ())
        for index, trial in enumerate(refit.trial_records):
            trial["trial"] = index
        refit.column_sign = column_sign(refit.phase)
        refit.rotation_flipped = True
        refit.initial_rotation_deg = result.initial_rotation_deg
        refit.initial_column_sign = sign
        refit.rotation_check_phases = np.stack((initial_phase, host_array(refit.phase)))
        if refit.elapsed is not None and result.elapsed is not None:
            refit.elapsed += result.elapsed   # First pass plus branch selection.
        if show:
            draw_column_histogram(refit.phase, f"SSB-selected rotation {refit.physical_rotation_deg:.2f}° · polarity score {refit.column_sign:+.2f}", limit=limit)
        if verbose:
            print(f"SSB: selected rotation {refit.physical_rotation_deg:.2f}°; C10 {refit.aberrations['C10']:+.2f} nm, "
                  f"polarity score {refit.column_sign:+.2f}. find_aberrations(check_rotation=False) keeps the input rotation.")
        return refit

    @property
    def _backend_protocol(self) -> SSBProtocol:
        """Return the sole strict backend implementation for this session."""

        if self.backend == "cuda":
            return self._prepare_cuda()
        if self._mps_backend is None:
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
                    aberrations_to_engine(self.aberrations) if self._aberrations_explicit else None
                ),
            )
        return self._mps_backend

    def _reconstruct_trials(self, parameters, trials, *, upsample, phase_estimator):
        """Replay selected records without replacing the current search or correction."""
        parameters = self._aberration_search if parameters is None else parameters
        if parameters is None:
            raise ValueError("Run ssb.find_aberrations() before reconstructing trials.")
        records = select_records(parameters.trial_records, trials)
        estimator = phase_estimator or ("phase_of_mean" if self.backend == "cuda" else "mean_phase")
        saved_rotation = self.rotation_angle_deg, self.com_reversed
        xp = cp if self.backend == "cuda" else np
        waves = []
        # a failed replay must not leave the session on a trial's rotation
        try:
            for record in records:
                self.set_rotation(record["rotation_angle_deg"], record["com_reversed"])
                values = record["params"]
                phase, _ = self._phase(
                    {"C10": values["C10_nm"], "C12": values["C12_nm"],
                     "phi12": math.radians(values["phi12_deg"])},
                    tilt_mrad=(record["tilt_row_mrad"], record["tilt_col_mrad"]),
                    depth_spread_nm=record["depth_spread_nm"],
                    upsampling_factor=upsample, phase_estimator=estimator, compute_loss=False,
                )
                # Native mean-phase kernels reuse a scratch buffer. Own each
                # wave before reconstructing the next trial on the same stream.
                waves.append(xp.exp(1j * phase))
        finally:
            self.set_rotation(*saved_rotation)
        wave = xp.stack(waves)
        sampling = np.asarray(self.scan_sampling_A) / upsample
        sampling = float(sampling) if sampling.ndim == 0 else tuple(sampling.tolist())
        return SSBResult(
            object_wave=wave, backend=self.backend, trial_records=records,
            voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
            scan_sampling_A=sampling, upsample=upsample, phase_estimator=estimator,
            amplitude_estimated=False, source_path=self.source_path,
            metadata={"trial_ids": [record["trial"] for record in records]},
        )

    def show_trials(self, *, best: int | None = None, last: int | None = None,
                    first: int | None = None, upsample: int = 1,
                    axsize: tuple[float, float] = (6, 6)):
        """Show selected attempts as phase images with a compact parameter table.

        Parameters
        ----------
        best, last, first
            Supply exactly one positive count. ``best`` selects the lowest
            search loss within the latest recorded search; ``last`` and ``first``
            select completed attempts in time order. Local refinement is not
            an Optuna trial and is displayed separately in the final report.
        upsample
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
                                  trials=[record["trial"] for record in records], upsample=upsample)
        kind = "Best" if best is not None else "Last" if last is not None else "First"
        return plot_trials(result, kind, axsize=axsize)

    def find_aberrations(
        self,
        *,
        tilt: bool = False,
        scan_region: tuple[int, int, int, int] | None = None,
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
        180 degrees), with C10, C12, and tilt reversed in sign. A CUDA joint fit
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

        ``scan_region=(row_start, row_stop, column_start, column_stop)``
        restricts fitting to that scan region, with exclusive stops. Detector
        geometry is estimated from the full input unless supplied explicitly.
        The returned phase and rotation diagnostics describe the region;
        ``ssb.reconstruct(result)`` applies its parameters to the full scan.
        Currently the region must be square with 128, 256, 512 or 1024 samples
        per side; it is never silently padded or cropped. The stored source
        must expose a dense four-dimensional scan array.

        Set ``save_to`` to reuse an exact prior result from ``SSB.open`` when the detector source,
        calibration, backend, physical parameters, and fit settings all match.
        Changed settings recompute automatically. Set ``force=True`` to recompute
        an otherwise matching result. Direct array inputs are saved but always
        recomputed: a source path does not identify an array's crop or mutations.
        ``tilt=True`` requires at least one trial; zero trials are supported only
        for the standard search, starting from the current session coefficients.
        """

        if tilt and (search_ranges is not None or refine_lock is not None):
            raise ValueError("search_ranges and refine_lock apply to the standard fit; tilt=True searches its own ranges.")
        if trials < 0:
            raise ValueError(f"trials must be non-negative, got {trials}.")
        if tilt and int(trials) == 0:
            raise ValueError(
                "tilt=True requires at least one trial; use trials=200, "
                "or reconstruct() to apply known parameters."
            )
        if refinement not in {"nelder-mead", None}:
            raise ValueError("refinement must be 'nelder-mead' or None.")
        paths = None
        signature = None
        if save_to is not None:
            paths = result_paths(save_to, "find_aberrations")
            signature = self._result_signature(
                "find_aberrations",
                {
                    "trials": int(trials),
                    "refinement": refinement,
                    "search_ranges": search_ranges,
                    "refine_lock": refine_lock,
                    "seed": int(seed),
                    "starting_aberrations": self.aberrations,
                    "starting_aberrations_explicit": self._aberrations_explicit,
                    **({"tilt": True, "tilt_limit_mrad": float(tilt_limit_mrad)} if tilt else {}),
                    "check_rotation": bool(check_rotation),
                    "scan_region": scan_region,
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
        if scan_region is not None:
            if len(scan_region) != 4 or any(type(value) is not int for value in scan_region):
                raise ValueError("Use scan_region=(row_start, row_stop, column_start, column_stop), in integer scan pixels.")
            if len(self._data.shape) != 4:
                raise NotImplementedError("Region fitting needs a dense 4D scan source. Open the original acquisition with io.load and pass it to SSB.")
            row_start, row_stop, column_start, column_stop = scan_region
            rows, columns = self._data.shape[:2]
            side = row_stop - row_start
            if not (0 <= row_start < row_stop <= rows and
                    0 <= column_start < column_stop <= columns and
                    side == column_stop - column_start and side in (128, 256, 512, 1024)):
                raise ValueError(f"scan_region={scan_region} must be inside {rows} × {columns} and square with 128, 256, 512 or 1024 scan pixels per side.")
            # The region keeps the full scan's bright-field disk and detector calibration.
            center, radius, det_sampling = self.bf_center, self.bf_radius, self.det_sampling
            if center is None or radius is None or det_sampling is None:
                mean_pattern = detector.mean(self._data)
                detected_center, detected_radius = detector.fit_probe(mean_pattern)
                center = detected_center if center is None else center
                radius = detected_radius if radius is None else radius
                if det_sampling is None:
                    det_sampling = self.semiangle_mrad / disk_edge_radius(mean_pattern)
            with SSB(
                self._data[row_start:row_stop, column_start:column_stop],
                backend=self.backend, voltage_kV=self.voltage_kV,
                semiangle_mrad=self.semiangle_mrad, scan_sampling_A=self.scan_sampling_A,
                det_sampling=det_sampling,
                aberrations=self.aberrations if self._aberrations_explicit else None,
                rotation_angle_deg=self.physical_rotation_deg,
                bf_center=center, bf_radius=radius,
                bf_intensity_threshold=self.bf_intensity_threshold,
            ) as regional:
                result = regional.find_aberrations(
                    tilt=tilt, trials=trials, refinement=refinement,
                    search_ranges=search_ranges, refine_lock=refine_lock,
                    tilt_limit_mrad=tilt_limit_mrad, check_rotation=check_rotation,
                    seed=seed, verbose=verbose,
                )
            result.fit_scan_region = tuple(scan_region)
            self._accept_result(result)
            self._reconstruction = None
            if paths is not None:
                result = self._save_result(result, paths=paths, signature=signature)
            self._aberration_search = result
            self.trial_history = list(result.trial_records or ())
            return result
        if tilt:
            result = self._accept_result(self._fit_tilt(int(trials), refinement, float(tilt_limit_mrad), int(seed), verbose))
        else:
            result = self._backend_protocol.fit(
                aberrations=(
                    aberrations_to_engine(self.aberrations)
                    if self._aberrations_explicit else None
                ),
                trials=int(trials),
                refinement=refinement,
                search_ranges=search_ranges_to_engine(search_ranges),
                refine_lock=refine_lock,
                seed=int(seed),
                verbose=verbose,
            )
            result = self._accept_result(result_from_engine(result))
        label_trials(result, search=0)
        if check_rotation:
            result = self._accept_result(self._resolve_rotation_branch(
                result, refinement=refinement, tilt=tilt, tilt_limit_mrad=float(tilt_limit_mrad), seed=int(seed),
                verbose=verbose))
        estimator = "phase_of_mean" if self.backend == "cuda" else "mean_phase"
        displayed = self.reconstruct(result, phase_estimator=estimator, compute_loss=False, force=True)
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

    def reconstruct(
        self,
        parameters: SSBResult | None = None,
        *,
        aberrations: dict[str, float] | None = None,
        trials: object = None,
        upsample: int | Literal["auto"] = 1,
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
            Partial coefficient overrides: C10/C12 in nm, phi12 in radians.
            Unspecified coefficients and sample geometry are preserved.
        trials
            Trial IDs from ``parameters.trials.index``. Reconstruct these
            attempts using their original rotation and sample geometry;
            return a phase stack without changing the active session.
        upsample
            Output factor: 1, 2, 3, 4, or 8. Scan positions and measured
            diffraction patterns are unchanged. Output pixel size decreases
            by this factor, preserving the physical field of view.
            ``"auto"`` chooses the smallest supported factor whose output
            Nyquist frequency covers the ideal circular-aperture SSB support,
            2 * semiangle / wavelength, on both scan axes. This is a grid
            criterion, not a measured resolution or a guarantee of useful
            signal at the cutoff. Upsampling currently requires CUDA.
        phase_estimator
            CUDA defaults to ``phase_of_mean`` at every output factor. MPS
            retains ``mean_phase``. The loss is the native-grid search
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
        >>> result = ssb.reconstruct(aberrations, upsample=4, save_to="results/4x")
        """
        if upsample == "auto":
            from quantem.gpu.optics.physics import ssb_upsampling_factor

            upsample = ssb_upsampling_factor(
                voltage_kV=self.voltage_kV,
                semiangle_mrad=self.semiangle_mrad,
                scan_sampling_A=self.scan_sampling_A,
            )
        if type(upsample) is not int or upsample not in (1, 2, 3, 4, 8):
            raise ValueError("upsample must be 'auto', 1, 2, 3, 4, or 8.")
        if phase_estimator not in (None, "mean_phase", "phase_of_mean", "complex_wave"):
            raise ValueError("phase_estimator must be 'mean_phase', 'phase_of_mean', or 'complex_wave'.")
        if parameters is not None and not isinstance(parameters, SSBResult):
            raise TypeError("Pass the result of find_aberrations(), or supply coefficients with aberrations={...}.")
        if trials is not None:
            if phase_estimator == "complex_wave":
                raise ValueError("Trial inspection returns phase images; choose mean_phase or phase_of_mean.")
            if aberrations is not None or save_to is not None:
                raise ValueError("Replay trials without coefficient overrides or save_to; save the returned result separately.")
            return self._reconstruct_trials(parameters, trials, upsample=upsample, phase_estimator=phase_estimator)
        correction = parameters
        if correction is not None:
            coefs = dict(correction.aberrations)
            tilt = correction.tilt_mrad
            depth = correction.depth_spread_nm
            if correction.physical_rotation_deg != self.physical_rotation_deg:
                self.set_rotation(correction.rotation_angle_deg, correction.com_reversed)
        else:
            coefs = dict(self.aberrations)
            tilt = self.tilt_mrad
            depth = self.depth_spread_nm
        coefs = validate_aberrations({**coefs, **(aberrations or {})})
        estimator = phase_estimator or ("phase_of_mean" if self.backend == "cuda" else "mean_phase")
        if estimator == "complex_wave" and (upsample != 1 or depth or any(tilt or ())):
            raise ValueError("complex_wave requires upsample=1 and no sample tilt/depth; choose a phase estimator instead.")
        settings = {
            "aberrations": coefs, "compute_loss": bool(compute_loss),
            "upsample": upsample, "phase_estimator": estimator,
            "tilt_mrad": tilt, "depth_spread_nm": depth,
        }
        paths = signature = None
        if save_to is not None:
            paths = result_paths(save_to, "reconstruct")
            signature = self._result_signature("reconstruct", settings)
            if not force and self.source_kind != "array":
                reused = load_result(paths=paths, signature=signature, backend=self.backend)
                if reused is not None:
                    if verbose:
                        print(f"Loading matching {upsample}x SSB result: {paths[0]}")
                    return self._accept_result(reused)

        if verbose:
            rows, cols = self.scan_shape
            print(f"Reconstructing {upsample}x: {rows * upsample} x {cols * upsample}; "
                  f"{estimator}; tilt {tilt or (0.0, 0.0)} mrad; "
                  f"depth spread {depth or 0.0:.3f} nm; loss at native 1x")
        cached = self._reconstruction
        if (not force and save_to is None and parameters is None and aberrations is None
                and cached is not None and cached.upsample == upsample
                and cached.phase_estimator == estimator
                and cached.physical_rotation_deg == self.physical_rotation_deg
                and cached.aberrations == coefs
                and cached.tilt_mrad == tilt
                and cached.depth_spread_nm == depth
                and (not compute_loss or cached.loss is not None)):
            return cached
        if estimator == "complex_wave":
            result = result_from_engine(self._backend_protocol.reconstruct_result(
                aberrations_to_engine(coefs), compute_loss=compute_loss,
            ))
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
            coefs, upsampling_factor=upsample, phase_estimator=estimator,
            tilt_mrad=tilt or (0.0, 0.0), depth_spread_nm=depth or 0.0,
            compute_loss=compute_loss,
        )
        wave = cp.exp(1j * phase) if self.backend == "cuda" else np.exp(1j * phase)
        sampling = np.asarray(self.scan_sampling_A) / upsample
        sampling = float(sampling) if sampling.ndim == 0 else tuple(sampling.tolist())
        result = SSBResult(
            object_wave=wave, backend=self.backend, aberrations=dict(coefs),
            tilt_mrad=tilt, depth_spread_nm=depth,
            rotation_angle_deg=self.rotation_angle_deg, com_reversed=self.com_reversed,
            loss=loss, elapsed=time.perf_counter() - started, num_bf=self.num_bf,
            voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad,
            scan_sampling_A=sampling, upsample=upsample,
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
        phase_estimator: Literal["mean_phase", "phase_of_mean"] | None = None,
    ) -> tuple[np.ndarray, float | None]:
        """Reconstruct a transient phase image for an interactive viewer.

        The returned NumPy array has the full selected output resolution. "Preview"
        means it is not saved and does not replace the fitted calibration or
        stored result; it does not imply a lower-quality reconstruction.

        ``aberrations`` C10 / C12 in nm, phi12 in rad. A positive ``depth_spread_nm`` switches to the thick-sample model:
        each bright-field pixel's correction is averaged over that depth, with the crystal leaning by ``tilt_mrad``
        (row, col; scan frame), as fitted by ``find_aberrations(tilt=True)``. With no depth spread the tilt has no effect and this is
        standard SSB. CUDA and MPS backends.

        ``upsampling_factor`` selects 1, 2, 3, 4, or 8 times finer output sampling.
        Values above one support C10/C12 with or without tilt/depth correction
        on CUDA. The diagnostic loss and aberration search remain on the native
        scan grid. Sampling never fits parameters or interpolates detector data.

        By default, CUDA C10/C12 reconstruction averages corrected complex
        waves before taking phase (``phase_of_mean``), with or without sample
        tilt, at every output factor. Explicit ``mean_phase`` retains the
        historical per-detector phase average for comparisons. Native MPS and
        higher-order paths retain their existing estimator; explicit wave
        averaging on unsupported paths raises an error. Quantitative phase
        amplitude is not established by stronger image contrast. Diagnostic
        loss still uses native-grid per-detector phase variance; fitting is
        unaffected.

        Examples
        --------
        Fit all correction parameters once, then change only output sampling:

        >>> fitted = ssb.find_aberrations(tilt=True)  # native-grid joint fit
        >>> phase, loss = ssb.preview(
        ...     dict(fitted), tilt_mrad=fitted.tilt_mrad,
        ...     depth_spread_nm=fitted.depth_spread_nm, upsampling_factor=4,
        ... )

        Compare the historical estimator explicitly, without changing the fit:

        >>> candidate, native_loss = ssb.preview(
        ...     dict(fitted), tilt_mrad=fitted.tilt_mrad,
        ...     depth_spread_nm=fitted.depth_spread_nm, upsampling_factor=4,
        ...     phase_estimator="mean_phase",
        ... )
        """

        phase, loss = self._phase(
            aberrations, compute_loss=compute_loss,
            upsampling_factor=upsampling_factor,
            higher_order_magnitudes=higher_order_magnitudes,
            higher_order_angles=higher_order_angles, context=context,
            tilt_mrad=tilt_mrad, depth_spread_nm=depth_spread_nm,
            phase_estimator=phase_estimator,
        )
        if self.backend == "cuda":
            phase = cp.asnumpy(phase)
        return np.asarray(phase, dtype=np.float32), loss

    def _phase(
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
        phase_estimator: Literal["mean_phase", "phase_of_mean"] | None = None,
    ) -> tuple[object, float | None]:
        """Compute the shared phase without exporting CUDA arrays to the host."""
        coefs = aberrations_to_engine(validate_aberrations(aberrations))
        if phase_estimator is None:
            # Keep unsupported native backends/higher-order paths available.
            # C10/C12 uses one estimator at every output factor on CUDA.
            higher_order_active = (higher_order_magnitudes is not None
                                   and np.any(np.asarray(higher_order_magnitudes)[2:]))
            phase_estimator = ("phase_of_mean" if self.backend == "cuda"
                               and not higher_order_active else "mean_phase")
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
                coefs = {**coefs, "C10": float(magnitudes[0]) * ENGINE_PER_NM,
                         "C12": float(magnitudes[1]) * ENGINE_PER_NM,
                         "phi12": float(angles[1])}
                higher_order_magnitudes = higher_order_angles = None
        scope = nullcontext() if context is None else context
        if upsampling_factor != 1 or phase_estimator == "phase_of_mean":
            if higher_order_magnitudes is not None:
                raise ValueError(
                    "Upsampling and wave averaging support C10/C12 with "
                    "tilt/depth; turn off higher-order magnitudes."
                )
            backend = self._backend_protocol
            with scope:
                return backend.preview_upsampled(
                    coefs, upsampling_factor=upsampling_factor, compute_loss=compute_loss,
                    tilt_mrad=tuple(float(value) for value in tilt_mrad),
                    thickness=float(depth_spread_nm) * ENGINE_PER_NM,
                    phase_estimator=phase_estimator,
                )
        if float(depth_spread_nm) > 0.0:
            sample = {"tilt_row_mrad": float(tilt_mrad[0]), "tilt_col_mrad": float(tilt_mrad[1]),
                      "thickness": float(depth_spread_nm) * ENGINE_PER_NM}
            if higher_order_magnitudes is not None:
                raise ValueError("The thick-sample preview does not combine with higher-order aberrations yet.")
            backend = self._backend_protocol
            with scope:
                return backend.preview_sample(coefs, sample, compute_loss=compute_loss)
        magnitudes = (
            None
            if higher_order_magnitudes is None
            # all 14 magnitudes (C10, C12 in slots 0-1, then C21..C56) are nm; the engines take Angstrom
            else np.asarray(higher_order_magnitudes, dtype=np.float32) * np.float32(ENGINE_PER_NM)
        )
        angles = (
            None
            if higher_order_angles is None
            else np.asarray(higher_order_angles, dtype=np.float32)
        )
        backend = self._backend_protocol
        with scope:
            return backend.preview(
                coefs,
                compute_loss=compute_loss,
                higher_order_magnitudes=magnitudes,
                higher_order_angles=angles,
            )

    @property
    def supports_tilt(self) -> bool:
        """True when this session's backend implements the thick-sample model (``find_aberrations(tilt=True)``, ``preview(tilt_mrad=...)``).

        Both Python backends, CUDA and MPS, implement it.
        """
        return self.backend in ("cuda", "mps")

    def _fit_tilt(self, trials: int, refinement: RefineMethod, tilt_limit_mrad: float, seed: int, verbose: bool) -> SSBResult:
        """Joint thick-sample fit (backend ``fit_sample``, engine units) -> an nm SSBResult at the fitted parameters."""
        backend = self._backend_protocol
        started = time.perf_counter()
        fit = backend.fit_sample(trials=trials, tilt_limit_mrad=tilt_limit_mrad, seed=seed, verbose=verbose,
                                 polish_starts=0 if refinement is None else 3)
        aberrations = aberrations_from_engine({key: float(fit[key]) for key in ("C10", "C12", "phi12")})
        tilt_mrad = (float(fit["tilt_row_mrad"]), float(fit["tilt_col_mrad"]))
        depth_spread_nm = float(fit["thickness"]) / ENGINE_PER_NM
        phase, loss = self._phase(aberrations, tilt_mrad=tilt_mrad, depth_spread_nm=depth_spread_nm,
                                   phase_estimator="mean_phase")
        # the thick-sample path recovers the phase only; the transmission amplitude is not estimated
        object_wave = cp.exp(1j * phase) if self.backend == "cuda" else np.exp(1j * np.asarray(phase))
        records = []
        for trial in fit["trial_records"]:
            values = trial["params"]
            records.append({
                "number": trial["number"], "loss": trial["loss"],
                "objective": "negative_thick_agreement", "band_inv_A": fit["band_inv_A"],
                "params": {"C10_nm": values["C10"] / ENGINE_PER_NM, "C12_nm": values["C12"] / ENGINE_PER_NM,
                           "phi12_deg": math.degrees(values["phi12"])},
                "tilt_row_mrad": values["tilt_row_mrad"], "tilt_col_mrad": values["tilt_col_mrad"],
                "depth_spread_nm": values["thickness"] / ENGINE_PER_NM,
            })
        return SSBResult(object_wave=object_wave, trial_records=records, backend=self.backend, aberrations=aberrations, tilt_mrad=tilt_mrad,
                         depth_spread_nm=depth_spread_nm, tilt_fit_gain=float(fit["gain"]), amplitude_estimated=False, phase_estimator="mean_phase",
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
        """Release the session's GPU state; data passed to ``SSB(data)`` stays the caller's.

        The session owns only what it made: the bright-field crop decoded from a
        loaded dataset (``SSB.open`` releases its acquisition as soon as the crop
        is read) and the backends' prepared Fourier data. A borrowed array or
        dataset is never released here.
        """

        self._data = None
        if self._cuda_session is not None:
            self._cuda_session.close()
            self._cuda_session = None
        if self._mps_backend is not None:
            backend = self._mps_backend
            self._mps_backend = None
            # The MPS backend's final allocator flush must run after the
            # workflow releases its own reference to the shared source.
            backend.close()

    def __enter__(self) -> Self:
        """Return this prepared SSB session."""

        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        """Release backend resources when leaving a context manager."""

        self.close()


__all__ = ["SSB"]
