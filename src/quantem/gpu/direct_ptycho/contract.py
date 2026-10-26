"""Strict backend contract for single-sideband ptychography."""

import pathlib
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from quantem.gpu.direct_ptycho.brightfield import BrightfieldDisk
from quantem.gpu.direct_ptycho.results import SSBResult

# The local refinement a low-order fit applied: the trust-region Newton on the exact derivatives ("newton"), or the
# exact-gradient quasi-Newton in (C10, C12, phi12) for a standard fit that locks a coefficient, bounds phi12 or holds a
# nonzero C12 ("bfgs").
RefineMethod = Literal["bfgs", "newton"] | None


def agreement_objective(objective) -> str:
    """``"agreement"``, what every fit maximizes: the least-squares agreement of the SSB model with the data.

    ``objective`` is retired: ``"agreement"`` prints one line; ``"phase_variance"`` (the earlier search) raises, since
    a silent agreement fit would land elsewhere.
    """
    if objective is None:
        return "agreement"
    if objective == "agreement":
        print("objective is retired; every fit maximizes the model agreement.")
        return "agreement"
    if objective == "phase_variance":
        raise ValueError("objective='phase_variance' is retired: every fit maximizes the agreement of the SSB model with "
                         "the data, which lands elsewhere. Drop objective=; the phase variance at the fit is still "
                         "result.loss.")
    raise ValueError(f"objective is retired: every fit maximizes the model agreement; got {objective!r}.")


def retired_higher_order(higher_order, *, standard: bool = False) -> None:
    """``higher_order`` is retired: the default fifth-order fit fits coma, trefoil, C30 and every term through fifth
    order on exact gradients, with the half-band check. A subset prints one line: ``standard`` (a plain primary call)
    says that call runs ``search="standard"`` instead; otherwise the primary fit holds the higher orders at 0."""
    if higher_order and standard:
        print("higher_order is retired; running search='standard', the fifth-order fit that fits coma, trefoil, C30 "
              "and every term through fifth order.")
    elif higher_order:
        print("higher_order is retired; the default fit (search='standard') fits every term through fifth order, and "
              "this primary fit holds them at 0.")
    return None


def auto_refinement(refinement) -> str | None:
    """``"auto"``, the refinement every fit uses, or None for none (``search="primary"`` only). Choosing the method is
    retired: another value prints one line and is ``"auto"``."""
    if refinement is None or refinement == "auto":
        return refinement
    print(f"refinement={refinement!r} is retired; using refinement='auto', the exact-derivative fit.")
    return "auto"


@dataclass(frozen=True)
class SSBPrecision:
    """Numeric storage required from every SSB backend."""

    real_dtype: Literal["float32"] = "float32"
    complex_dtype: Literal["complex64"] = "complex64"


@dataclass(frozen=True)
class SSBExportState:
    """Exact compact state consumed by widget and browser integrations."""

    backend: Literal["cuda", "mps"]
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
    # Present only when the original QEM still describes this session's exact counts.
    qem_source: dict[str, object] | None = None

    @property
    def num_bright_field_pixels(self) -> int:
        """Number of complete bright-field detector pixels."""

        return self.brightfield.size


class SSBProtocol(Protocol):
    """Backend contract implemented by CUDA and MPS: every call ``SSB`` makes.

    CUDA and MPS implement this protocol in Python. Browser WebGPU mirrors the
    same semantic contract in ``webgpu/protocol.ts`` because its execution is
    asynchronous and browser-resident. Unsupported work must fail explicitly;
    no implementation may fall back to CPU or change the BF evidence.
    ``preview_upsampled`` is every SSB image: the phase of the bright-field mean
    wave at the output factor, with the phase-variance loss on the native grid.
    """

    backend: Literal["cuda", "mps"]
    precision: SSBPrecision

    @property
    def scan_shape(self) -> tuple[int, int]: ...

    @property
    def detector_shape(self) -> tuple[int, int]: ...

    @property
    def num_bright_field_pixels(self) -> int: ...

    def fit(
        self,
        *,
        aberrations: dict[str, float] | None,
        trials: int,
        refinement: str | None,
        search_ranges: dict[str, tuple[float, float] | float] | None,
        refine_lock: list[str] | None,
        seed: int,
        verbose: bool,
        compute_loss: bool = True,
    ) -> SSBResult: ...

    def cache_rotation(self, scan_detector_rotation_rad: float) -> None: ...

    def preview_upsampled(
        self,
        aberrations: dict[str, float],
        *,
        upsampling_factor: int,
        compute_loss: bool,
        sample_tilt_mrad: tuple[float, float],
        depth_spread_A: float,
        higher_order_magnitudes: np.ndarray | None = None,
        higher_order_angles: np.ndarray | None = None,
    ) -> tuple[object, float | None]: ...

    def fit_sample(self, **options) -> dict[str, object]: ...

    def sample_agreement(self, parameters: np.ndarray, band_inv_A: tuple[float, float]) -> np.ndarray: ...

    def preview_context(self, num_bright_field_pixels: int) -> AbstractContextManager | None: ...

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
