"""Temporal joining helpers for drift-corrected SSB reconstructions.

The functions here deliberately sit above the single-frame SSB optimizer.  They
reuse calibrated SSB parameters, reconstruct each frame's complex object wave,
apply known scan-coordinate drift as a Fourier phase ramp, and average the
aligned object waves on the GPU.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cupy as cp

from quantem.gpu.ssb.results import SSBResult

PhaseReference = Literal["first", "none"]


@dataclass
class SSBTimeSeriesResult:
    """Shared-calibration SSB object-wave time series.

    This result keeps one object wave per time frame.  Use it when temporal
    variation must be preserved.  Use :func:`ssb_time_average` when the desired
    output is a single high-SNR object for initialization or static reporting.
    """

    object_waves: cp.ndarray
    aberrations: dict[str, float]
    rotation_angle_deg: float
    shifts: cp.ndarray
    weights: cp.ndarray
    phase_reference: PhaseReference = "first"
    elapsed: float | None = None
    voltage_kV: float | None = None
    semiangle_mrad: float | None = None
    scan_sampling_A: float | None = None
    mode: str = "shared_calibration_time_series"

    @property
    def phase(self) -> cp.ndarray:
        """Phase stack with shape ``(time, row, col)``."""

        return cp.angle(self.object_waves)

    @property
    def amplitude(self) -> cp.ndarray:
        """Amplitude stack with shape ``(time, row, col)``."""

        return cp.abs(self.object_waves)

    def time_average(self, weights: Sequence[float] | cp.ndarray | None = None) -> SSBResult:
        """Return a single complex average from the preserved time series."""

        result = join_object_waves(
            self.object_waves,
            shifts=None,
            weights=self.weights if weights is None else weights,
            phase_reference="none",
        )
        result.aberrations = dict(self.aberrations)
        result.rotation_angle_deg = self.rotation_angle_deg
        result.voltage_kV = self.voltage_kV
        result.semiangle_mrad = self.semiangle_mrad
        result.scan_sampling_A = self.scan_sampling_A
        result.join_mode = "shared_calibration_time_series_average"
        return result


def fourier_shift_2d(image: cp.ndarray, shift: Sequence[float]) -> cp.ndarray:
    """Apply a periodic subpixel image shift on the GPU.

    Parameters
    ----------
    image : cp.ndarray
        Two-dimensional real or complex image.
    shift : sequence of float
        ``(row, col)`` shift in pixels to apply to ``image``. Positive row
        shifts move content toward larger row indices, following the same
        Fourier phase-ramp convention as the parallax utilities.

    Returns
    -------
    cp.ndarray
        Shifted image. Complex inputs keep their dtype; real inputs return the
        real component in the original dtype.
    """

    arr = cp.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {arr.shape}")
    shift_row, shift_col = float(shift[0]), float(shift[1])
    n_row, n_col = int(arr.shape[0]), int(arr.shape[1])
    freq_row = cp.fft.fftfreq(n_row, d=1.0).astype(cp.float32).reshape(-1, 1)
    freq_col = cp.fft.fftfreq(n_col, d=1.0).astype(cp.float32).reshape(1, -1)
    phase = cp.exp(-2j * cp.pi * (freq_row * shift_row + freq_col * shift_col))
    shifted = cp.fft.ifft2(cp.fft.fft2(arr) * phase)
    if cp.iscomplexobj(arr):
        return shifted.astype(arr.dtype, copy=False)
    return shifted.real.astype(arr.dtype, copy=False)


def _as_object_wave_stack(object_waves: cp.ndarray | Sequence[cp.ndarray]) -> cp.ndarray:
    stack = cp.asarray(object_waves)
    if stack.ndim == 2:
        stack = stack[None, ...]
    elif stack.ndim != 3:
        try:
            stack = cp.stack([cp.asarray(w) for w in object_waves], axis=0)
        except TypeError as exc:
            raise ValueError(
                "object_waves must be a 2D object wave, a 3D stack, or a "
                "sequence of 2D object waves"
            ) from exc
    if stack.ndim != 3:
        raise ValueError(f"object_waves must resolve to (time, row, col), got {stack.shape}")
    if not cp.iscomplexobj(stack):
        stack = stack.astype(cp.complex64)
    return stack


def _normalize_shifts(shifts: Sequence[Sequence[float]] | cp.ndarray | None, n_time: int) -> cp.ndarray:
    if shifts is None:
        return cp.zeros((n_time, 2), dtype=cp.float32)
    shifts_cp = cp.asarray(shifts, dtype=cp.float32)
    if shifts_cp.shape != (n_time, 2):
        raise ValueError(
            f"shifts must have shape ({n_time}, 2), got {tuple(shifts_cp.shape)}"
        )
    return shifts_cp


def _normalize_weights(weights: Sequence[float] | cp.ndarray | None, n_time: int) -> cp.ndarray:
    if weights is None:
        return cp.full((n_time,), 1.0 / float(n_time), dtype=cp.float32)
    weights_cp = cp.asarray(weights, dtype=cp.float32)
    if weights_cp.shape != (n_time,):
        raise ValueError(
            f"weights must have shape ({n_time},), got {tuple(weights_cp.shape)}"
        )
    if bool(cp.any(weights_cp < 0).get()):
        raise ValueError("weights must be non-negative")
    weight_sum = float(weights_cp.sum().get())
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("weights must sum to a positive finite value")
    return weights_cp / weight_sum


def _phase_align_to_reference(wave: cp.ndarray, reference: cp.ndarray) -> cp.ndarray:
    """Remove the global complex phase offset relative to ``reference``."""

    overlap = cp.sum(wave * cp.conj(reference))
    angle = cp.angle(overlap)
    return wave * cp.exp(-1j * angle).astype(cp.complex64)


def _align_object_wave_stack(
    stack: cp.ndarray,
    shifts: cp.ndarray,
    phase_reference: PhaseReference,
) -> cp.ndarray:
    if phase_reference not in {"first", "none"}:
        raise ValueError("phase_reference must be 'first' or 'none'")

    n_time = int(stack.shape[0])
    aligned = cp.empty_like(stack)
    for i in range(n_time):
        shifted = fourier_shift_2d(stack[i], shifts[i])
        if i > 0 and phase_reference == "first":
            shifted = _phase_align_to_reference(shifted, aligned[0])
        aligned[i] = shifted
    return aligned


def join_object_waves(
    object_waves: cp.ndarray | Sequence[cp.ndarray],
    shifts: Sequence[Sequence[float]] | cp.ndarray | None = None,
    *,
    weights: Sequence[float] | cp.ndarray | None = None,
    phase_reference: PhaseReference = "first",
    return_aligned_stack: bool = False,
) -> SSBResult | tuple[SSBResult, cp.ndarray]:
    """Join a time series of SSB object waves after drift correction.

    Parameters
    ----------
    object_waves : cp.ndarray or sequence of cp.ndarray
        Complex SSB object waves, shaped ``(time, row, col)`` or a sequence of
        ``(row, col)`` arrays.
    shifts : array-like, optional
        ``(time, 2)`` row/col pixel shifts to apply to each frame so it lands
        in the shared specimen coordinate system. Use zeros if the stack is
        already drift-corrected.
    weights : array-like, optional
        Non-negative per-frame weights. Values are normalized internally.
    phase_reference : {"first", "none"}, default "first"
        ``"first"`` removes each aligned frame's global complex phase offset
        relative to frame 0 before averaging. This prevents arbitrary SSB phase
        constants from cancelling contrast. ``"none"`` performs a literal
        weighted complex average.
    return_aligned_stack : bool, default False
        If True, also return the aligned per-frame object-wave stack.

    Returns
    -------
    SSBResult or tuple[SSBResult, cp.ndarray]
        Joined object wave as an ``SSBResult``. If requested, the second return
        value is the aligned stack on GPU.
    """

    t0 = time.perf_counter()
    stack = _as_object_wave_stack(object_waves)
    n_time, n_row, n_col = (int(x) for x in stack.shape)
    shifts_cp = _normalize_shifts(shifts, n_time)
    weights_cp = _normalize_weights(weights, n_time)
    aligned = _align_object_wave_stack(stack, shifts_cp, phase_reference)

    joined = cp.tensordot(weights_cp, aligned, axes=(0, 0)).astype(cp.complex64)
    result = SSBResult(
        object_wave=joined,
        loss=None,
        elapsed=time.perf_counter() - t0,
        num_bf=None,
    )
    result.num_frames = n_time
    result.time_shifts = shifts_cp
    result.time_weights = weights_cp
    result.phase_reference = phase_reference
    result.join_mode = "drift_corrected_complex_mean"
    if return_aligned_stack:
        return result, aligned
    return result


def _reconstruct_objects_from_ssb_frames(
    ssb_frames: Sequence[object],
    *,
    aberrations: dict[str, float] | None,
    rotation_angle_deg: float | None,
) -> list[cp.ndarray]:
    objects: list[cp.ndarray] = []
    for ssb in ssb_frames:
        coefs = dict(getattr(ssb, "aberrations", {}) or {})
        if aberrations is not None:
            coefs.update(aberrations)
        rotation_rad = (
            math.radians(rotation_angle_deg)
            if rotation_angle_deg is not None
            else getattr(ssb, "_rotation_angle_rad", None)
        )
        reconstruct_object = getattr(ssb, "_reconstruct_object", None)
        if reconstruct_object is None:
            raise TypeError(
                "ssb_frames must contain objects with a _reconstruct_object method"
            )
        objects.append(
            reconstruct_object(
                coefs.get("C10"),
                coefs.get("C12"),
                coefs.get("phi12"),
                rotation_rad,
            )
        )
    return objects


def ssb_time_series(
    ssb_frames: Sequence[object],
    shifts: Sequence[Sequence[float]] | cp.ndarray | None = None,
    *,
    weights: Sequence[float] | cp.ndarray | None = None,
    aberrations: dict[str, float] | None = None,
    rotation_angle_deg: float | None = None,
    phase_reference: PhaseReference = "first",
) -> SSBTimeSeriesResult:
    """Reconstruct a shared-calibration SSB object-wave time series.

    Parameters
    ----------
    ssb_frames : sequence
        Sequence of prepared ``quantem.gpu.ssb.SSB`` objects, one per time
        frame. Each object should already hold its frame's ``G_qk`` on GPU.
    shifts : array-like, optional
        ``(time, 2)`` row/col shifts to apply to each object wave. Use zeros if
        the input frames are already sampled into drift-corrected coordinates.
    weights : array-like, optional
        Stored normalized frame weights. They are not used to merge the time
        series unless :meth:`SSBTimeSeriesResult.time_average` is called.
    aberrations : dict[str, float], optional
        Shared calibrated aberrations to apply to every frame.
    rotation_angle_deg : float, optional
        Shared scan-detector rotation override in degrees.
    phase_reference : {"first", "none"}, default "first"
        Global phase handling across time.

    Returns
    -------
    SSBTimeSeriesResult
        Aligned per-time object waves on GPU.
    """

    if not ssb_frames:
        raise ValueError("ssb_frames must contain at least one SSB frame")
    t0 = time.perf_counter()
    objects = _reconstruct_objects_from_ssb_frames(
        ssb_frames,
        aberrations=aberrations,
        rotation_angle_deg=rotation_angle_deg,
    )
    stack = cp.stack(objects, axis=0)
    n_time = int(stack.shape[0])
    shifts_cp = _normalize_shifts(shifts, n_time)
    weights_cp = _normalize_weights(weights, n_time)
    aligned = _align_object_wave_stack(stack, shifts_cp, phase_reference)

    first = ssb_frames[0]
    scan_sampling = getattr(first, "scan_sampling", None)
    if isinstance(scan_sampling, tuple):
        scan_sampling_A = float(scan_sampling[0])
    elif scan_sampling is not None:
        scan_sampling_A = float(scan_sampling)
    else:
        scan_sampling_A = None

    return SSBTimeSeriesResult(
        object_waves=aligned,
        aberrations=dict(aberrations or getattr(first, "aberrations", {}) or {}),
        rotation_angle_deg=(
            float(rotation_angle_deg)
            if rotation_angle_deg is not None
            else math.degrees(float(getattr(first, "_rotation_angle_rad", 0.0) or 0.0))
        ),
        shifts=shifts_cp,
        weights=weights_cp,
        phase_reference=phase_reference,
        elapsed=time.perf_counter() - t0,
        voltage_kV=getattr(first, "voltage_kV", None),
        semiangle_mrad=getattr(first, "semiangle_mrad", None),
        scan_sampling_A=scan_sampling_A,
    )


def ssb_time_average(
    ssb_frames: Sequence[object],
    shifts: Sequence[Sequence[float]] | cp.ndarray | None = None,
    *,
    weights: Sequence[float] | cp.ndarray | None = None,
    aberrations: dict[str, float] | None = None,
    rotation_angle_deg: float | None = None,
    phase_reference: PhaseReference = "first",
    return_aligned_stack: bool = False,
) -> SSBResult | tuple[SSBResult, cp.ndarray]:
    """Reconstruct and join a drift-corrected SSB time series.

    Parameters
    ----------
    ssb_frames : sequence
        Sequence of prepared ``quantem.gpu.ssb.SSB`` objects, one per time
        frame. Each object should already hold its frame's ``G_qk`` on GPU.
    shifts : array-like, optional
        ``(time, 2)`` row/col shifts to apply to the reconstructed object waves.
    weights : array-like, optional
        Per-frame weights, commonly dose or confidence weights.
    aberrations : dict[str, float], optional
        Shared calibrated aberrations to apply to every frame. Missing values
        fall back to each frame's stored aberrations.
    rotation_angle_deg : float, optional
        Shared scan-detector rotation override in degrees.
    phase_reference : {"first", "none"}, default "first"
        Global phase handling before averaging.
    return_aligned_stack : bool, default False
        If True, return the aligned per-frame object waves as well.

    Returns
    -------
    SSBResult or tuple[SSBResult, cp.ndarray]
        Joined SSB result on GPU.
    """

    if not ssb_frames:
        raise ValueError("ssb_frames must contain at least one SSB frame")
    objects = _reconstruct_objects_from_ssb_frames(
        ssb_frames,
        aberrations=aberrations,
        rotation_angle_deg=rotation_angle_deg,
    )

    joined = join_object_waves(
        cp.stack(objects, axis=0),
        shifts,
        weights=weights,
        phase_reference=phase_reference,
        return_aligned_stack=return_aligned_stack,
    )
    result = joined[0] if return_aligned_stack else joined
    result.aberrations = dict(aberrations or getattr(ssb_frames[0], "aberrations", {}) or {})
    if rotation_angle_deg is not None:
        result.rotation_angle_deg = float(rotation_angle_deg)
    else:
        result.rotation_angle_deg = math.degrees(
            float(getattr(ssb_frames[0], "_rotation_angle_rad", 0.0) or 0.0)
        )
    result.voltage_kV = getattr(ssb_frames[0], "voltage_kV", None)
    result.semiangle_mrad = getattr(ssb_frames[0], "semiangle_mrad", None)
    scan_sampling = getattr(ssb_frames[0], "scan_sampling", None)
    if isinstance(scan_sampling, tuple):
        result.scan_sampling_A = float(scan_sampling[0])
    elif scan_sampling is not None:
        result.scan_sampling_A = float(scan_sampling)
    result.join_mode = "ssb_time_average"
    return joined
