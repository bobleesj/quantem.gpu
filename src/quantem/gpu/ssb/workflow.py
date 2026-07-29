"""Backend-neutral scientist-facing SSB workflow.

This module owns the only public stateful SSB entry point. Backend modules own
device preparation and kernels, but they do not define a second user API.
"""
from __future__ import annotations

import math
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

import numpy as np

from quantem.gpu.device import resolve

from .compute.protocol import SSBProtocol
from .results import SSBResult


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
        bf_intensity_threshold: float = 0.0,
        bf_radius: int | None = None,
        source_path: str | None = None,
    ) -> None:
        self.backend = _resolve_backend(backend)
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
        self.aberrations = _validate_aberrations(aberrations)
        self.rotation_angle_deg = float(rotation_angle_deg)
        self.bf_intensity_threshold = float(bf_intensity_threshold)
        self.bf_radius = bf_radius
        self.source_path = source_path
        self.source_storage_path = source_path
        self.source_kind: Literal["array", "detector", "bf_columns"] = "array"
        self.source_dtype = str(data.dtype)
        self.source_bytes = int(data.nbytes)
        self.source_load_seconds: float | None = None
        self._cuda_session = None
        self._mps_backend = None
        self._reconstruction: SSBResult | None = None
        self.best_loss = float("inf")
        self.trial_history: list[dict[str, object]] = []

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
        bf_intensity_threshold: float = 0.0,
        bf_radius: int | None = None,
        calibration: str | None = None,
        verbose: bool = False,
    ) -> "SSB":
        """Open one lossless 4D-STEM source and prepare an SSB session.

        Exact BF-column storage is chosen automatically when it is available;
        otherwise detector counts are loaded with the explicitly requested
        storage dtype. Leave ``dtype=None`` for native detector precision.
        This storage choice never changes the float32/complex64 optimization
        precision or scientific objective.
        """

        selected = _resolve_backend(backend)
        if selected == "webgpu":
            raise RuntimeError(
                "Browser WebGPU sources are opened by the exported SSB runtime."
            )
        data = None
        source_kind: Literal["detector", "bf_columns"]
        source_dtype: str
        source_bytes: int
        source_load_seconds: float
        source_storage_path: str
        if selected == "mps":
            from .compute.mps.engine import load_bf_columns_mps

            for candidate in _mps_brightfield_sources(source, calibration):
                try:
                    frames = load_bf_columns_mps(candidate, verbose=verbose)
                except (FileNotFoundError, ValueError):
                    continue
                data = frames
                source_kind = "bf_columns"
                source_storage_path = str(frames.source_path)
                source_dtype = str(frames.dtype)
                source_bytes = int(frames.nbytes)
                source_load_seconds = float(frames.load_seconds)
                break
        if data is None:
            from quantem.gpu.io.hdf5 import LoadResult, load

            load_started = time.perf_counter()
            loaded = load(
                source,
                backend=selected,
                det_bin=1,
                dtype=dtype,
                precompute_detector_sum=(
                    selected == "mps" and str(dtype).lower() in {"u8", "uint8"}
                ),
                verbose=verbose,
            )
            if not isinstance(loaded, LoadResult):
                raise TypeError(
                    "One SSB source must produce one LoadResult; "
                    f"got {type(loaded).__name__}."
                )
            data = loaded.data
            source_kind = "detector"
            source_storage_path = str(source)
            source_dtype = str(data.dtype)
            source_bytes = int(data.nbytes)
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
            bf_intensity_threshold=bf_intensity_threshold,
            bf_radius=bf_radius,
            source_path=str(source),
        )
        session.source_kind = source_kind
        session.source_storage_path = source_storage_path
        session.source_dtype = source_dtype
        session.source_bytes = source_bytes
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
        bf_intensity_threshold: float = 0.0,
        bf_radius: int | None = None,
        source_path: str | None = None,
    ) -> "SSB":
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
            bf_intensity_threshold=bf_intensity_threshold,
            bf_radius=bf_radius,
            source_path=source_path,
        )

    def _prepare_cuda(self):
        """Construct the private CUDA implementation once."""

        if self._cuda_session is None:
            from .compute.cuda.backend import CudaSSBBackend

            self._cuda_session = CudaSSBBackend(
                data=self._data,
                semiangle=self.semiangle_mrad,
                scan_sampling=self.scan_sampling_A,
                det_sampling=self.det_sampling,
                voltage_kV=self.voltage_kV,
                scan_shape=self._scan_shape,
                bf_intensity_threshold=self.bf_intensity_threshold,
                bf_radius=self.bf_radius,
                aberrations=self.aberrations,
                rotation_angle_deg=self.rotation_angle_deg,
            )
        return self._cuda_session

    @property
    def _backend_protocol(self) -> SSBProtocol:
        """Return the sole strict backend implementation for this session."""

        if self.backend == "cuda":
            backend = self._prepare_cuda()
        else:
            if self._mps_backend is None:
                from .compute.mps.backend import MpsSSBBackend

                self._mps_backend = MpsSSBBackend(
                    self._data,
                    voltage_kV=self.voltage_kV,
                    semiangle_mrad=self.semiangle_mrad,
                    scan_sampling=self.scan_sampling_A,
                    det_sampling=self.det_sampling,
                    bf_intensity_threshold=self.bf_intensity_threshold,
                    bf_center=None,
                    bf_radius=self.bf_radius,
                    rotation_angle_deg=self.rotation_angle_deg,
                    aberrations=self.aberrations,
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
        trials: int = 200,
        refinement: RefineMethod = "nelder-mead",
        search_ranges: dict[str, tuple[float, float] | float] | None = None,
        refine_lock: list[str] | None = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> SSBResult:
        """Optimize C10/C12/phi12 and return the final reconstruction."""

        if trials < 0:
            raise ValueError(f"trials must be non-negative, got {trials}.")
        if refinement not in {"nelder-mead", None}:
            raise ValueError("refinement must be 'nelder-mead' or None.")
        result = self._backend_protocol.fit(
            trials=int(trials),
            refinement=refinement,
            search_ranges=search_ranges,
            refine_lock=refine_lock,
            seed=int(seed),
            verbose=verbose,
        )
        result.source_path = self.source_path
        self.aberrations = dict(result.aberrations)
        self.rotation_angle_deg = float(result.rotation_angle_deg)
        self.best_loss = (
            float(result.loss) if result.loss is not None else float("inf")
        )
        self.trial_history = [dict(trial) for trial in result.optuna_trials or ()]
        self._reconstruction = result
        return result

    def reconstruct(
        self,
        aberrations: dict[str, float] | None = None,
    ) -> SSBResult:
        """Reconstruct the complex object wave at fixed aberrations."""

        if aberrations is None and self._reconstruction is not None:
            return self._reconstruction
        coefs = self.aberrations if aberrations is None else _validate_aberrations(aberrations)
        result = self._backend_protocol.reconstruct_result(coefs)
        result.source_path = self.source_path
        self.aberrations = dict(coefs)
        self._reconstruction = result
        return result

    def preview(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool = True,
        higher_order_magnitudes: np.ndarray | None = None,
        higher_order_angles: np.ndarray | None = None,
        context: AbstractContextManager | None = None,
    ) -> tuple[np.ndarray, float | None]:
        """Reconstruct a transient phase image for an interactive viewer."""

        coefs = _validate_aberrations(aberrations)
        if (higher_order_magnitudes is None) != (higher_order_angles is None):
            raise ValueError(
                "higher_order_magnitudes and higher_order_angles must be "
                "provided together."
            )
        magnitudes = (
            None
            if higher_order_magnitudes is None
            else np.asarray(higher_order_magnitudes, dtype=np.float32)
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

    def set_rotation(self, rotation_angle_deg: float) -> None:
        """Set scan-to-detector rotation and refresh backend geometry."""

        self.rotation_angle_deg = float(rotation_angle_deg)
        self._backend_protocol.cache_rotation(math.radians(self.rotation_angle_deg))

    def close(self) -> None:
        """Release backend-owned GPU state."""

        if self._cuda_session is not None:
            self._cuda_session.close()
            self._cuda_session = None
        if self._mps_backend is not None:
            self._mps_backend.close()
        self._mps_backend = None
        self._data = None

    def __enter__(self) -> "SSB":
        """Return this prepared SSB session."""

        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        """Release backend resources when leaving a context manager."""

        self.close()
__all__ = ["SSB"]
