"""Strict backend contract for single-sideband ptychography."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import AbstractContextManager
import pathlib
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from ..bf_selector import BrightfieldDisk
from ..results import SSBResult


@dataclass(frozen=True)
class SSBPrecision:
    """Numeric storage required from every SSB backend."""

    real_dtype: Literal["float32"] = "float32"
    complex_dtype: Literal["complex64"] = "complex64"


@dataclass(frozen=True)
class SSBExportState:
    """Exact compact state consumed by widget and browser integrations."""

    backend: Literal["cuda", "mps", "webgpu"]
    scan_shape: tuple[int, int]
    brightfield: BrightfieldDisk
    kx_bf: np.ndarray
    ky_bf: np.ndarray
    qx_1d: np.ndarray
    qy_1d: np.ndarray
    aperture_k: np.ndarray
    alpha_k2: np.ndarray
    cos2phi_k: np.ndarray
    sin2phi_k: np.ndarray
    wavelength_A: float
    semiangle_rad: float
    angular_sampling_rad: tuple[float, float]
    sampling_A: tuple[float, float]
    dc_value: complex
    precision: SSBPrecision = SSBPrecision()
    bf_source_path: pathlib.Path | None = None
    bf_source_dtype: np.dtype | None = None
    bf_source_max_value: int | None = None

    @property
    def num_bf(self) -> int:
        """Number of complete bright-field detector pixels."""

        return self.brightfield.size


@runtime_checkable
class SSBProtocol(Protocol):
    """Small backend contract implemented by CUDA and MPS.

    CUDA and MPS implement this protocol in Python. Browser WebGPU mirrors the
    same semantic contract in ``webgpu/protocol.ts`` because its execution is
    asynchronous and browser-resident. Unsupported work must fail explicitly;
    no implementation may fall back to CPU or change the BF evidence.
    """

    backend: Literal["cuda", "mps", "webgpu"]
    precision: SSBPrecision

    @property
    def scan_shape(self) -> tuple[int, int]: ...

    @property
    def detector_shape(self) -> tuple[int, int]: ...

    @property
    def num_bf(self) -> int: ...

    def fit(
        self,
        *,
        trials: int,
        refinement: str | None,
        search_ranges: dict[str, tuple[float, float] | float] | None,
        refine_lock: list[str] | None,
        seed: int,
        verbose: bool,
    ) -> SSBResult: ...

    def reconstruct_result(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool = True,
    ) -> SSBResult: ...

    def cache_rotation(self, rotation_rad: float) -> None: ...

    def preview(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool,
        higher_order_magnitudes: np.ndarray | None,
        higher_order_angles: np.ndarray | None,
    ) -> tuple[np.ndarray, float | None]: ...

    def preview_context(self, num_bf: int) -> AbstractContextManager | None: ...

    def browser_state(self) -> SSBExportState: ...

    def export_brightfield(
        self,
        data,
        path_stem: str | pathlib.Path,
    ) -> tuple[pathlib.Path, float] | None: ...

    def close(self) -> None: ...


__all__ = [
    "SSBPrecision",
    "SSBProtocol",
]
