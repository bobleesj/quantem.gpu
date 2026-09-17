"""Independent double-precision SSB object and objective reference.

This module is a test-only oracle. It never imports the production SSB
implementation and production code never imports it. It restates the
documented SSB semantics directly:

- the full automatic bright-field disk: every detector pixel whose center is
  inside the calibrated radius, in detector row-major order;
- exact raw integer detector counts, with no binning, scaling, or clipping;
- the complete complex corrected spectrum, ``G(k) * conj(gamma) / |gamma|``,
  evaluated as a full-plane Fourier transform with no Hermitian or half-plane
  shortcut, so the signed Nyquist lines are treated exactly;
- the objective's documented DC policy: the corrected zero-frequency bin is
  the exact mean raw zero-frequency value over the logical BF set;
- the complex object is the mean of the per-BF inverse transforms over the
  logical BF set, and the loss is the BF phase variance averaged over pixels.

The default ``dtype=np.float64`` is the reference oracle. ``dtype=np.float32``
reproduces the identical formula in one straightforward single-precision
realization (SciPy's pocketfft, a different algorithm from MLX/Metal) and is
used only to measure the arithmetic noise floor the hardware backends are
compared against. Coordinates follow the public ``(row, col)`` convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as scipy_fft

__all__ = [
    "SSBReferenceResult",
    "brightfield_disk",
    "electron_wavelength_angstrom",
    "geometry_terms",
    "ssb_reference",
]

_MASS = 9.1093837139e-31
_CHARGE = 1.602176634e-19
_PLANCK = 6.62607015e-34
_SPEED = 299792458.0


def electron_wavelength_angstrom(voltage_kv: float) -> float:
    """Return the relativistic electron wavelength in angstroms."""

    energy = _CHARGE * float(voltage_kv) * 1e3
    return (
        _PLANCK
        / np.sqrt(2.0 * _MASS * energy * (1.0 + energy / (2.0 * _MASS * _SPEED**2)))
        * 1e10
    )


def brightfield_disk(
    detector_shape: tuple[int, int],
    center_row_col: tuple[float, float],
    radius_px: float,
    *,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full BF disk as detector row-major ``(rows, cols)``.

    Membership is the documented policy: a detector pixel belongs to the disk
    when its integer center lies within ``radius_px`` of the calibrated
    subpixel center, and the pixel passes the intensity mask when one is
    supplied. No pixel inside the disk is dropped for proximity to an edge.
    """

    rows_all, cols_all = np.meshgrid(
        np.arange(detector_shape[0], dtype=np.int64),
        np.arange(detector_shape[1], dtype=np.int64),
        indexing="ij",
    )
    if mask is None:
        keep = np.ones(detector_shape, dtype=bool)
    else:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != tuple(detector_shape):
            raise ValueError(
                f"mask shape {keep.shape} does not match detector {detector_shape}."
            )
    distance = np.hypot(
        rows_all - float(center_row_col[0]),
        cols_all - float(center_row_col[1]),
    )
    keep = keep & (distance <= float(radius_px))
    rows, cols = np.nonzero(keep)
    return rows.astype(np.int32), cols.astype(np.int32)


def geometry_terms(
    kx,
    ky,
    *,
    wavelength: float,
    semiangle_rad: float,
    ang_y_rad: float,
    ang_x_rad: float,
    dtype=np.float64,
):
    """Return ``(alpha^2, cos2phi, sin2phi, aperture)`` for probe coordinates.

    These are the documented probe terms at the supplied reciprocal-space
    coordinates: the aberration argument uses ``alpha = |k| * wavelength``,
    the two-fold angle terms use the direction of ``k``, and the aperture is
    the one-sampling-wide linear disk edge evaluated along that direction.
    """

    dtype = np.dtype(dtype).type
    dx = np.asarray(kx, dtype=dtype)
    dy = np.asarray(ky, dtype=dtype)
    dx2 = dx * dx
    dy2 = dy * dy
    r2 = dx2 + dy2
    r = np.sqrt(r2)
    alpha = r * dtype(wavelength)
    alpha2 = alpha * alpha
    inv_r2 = np.where(r2 > dtype(1e-30), dtype(1.0) / np.where(r2 > 0, r2, dtype(1.0)), dtype(0.0))
    cos2 = (dx2 - dy2) * inv_r2
    sin2 = dtype(2.0) * dx * dy * inv_r2
    denom_num2 = (dx * dtype(ang_y_rad)) ** 2 + (dy * dtype(ang_x_rad)) ** 2
    inv_r = np.where(r > dtype(1e-15), dtype(1.0) / np.where(r > 0, r, dtype(1.0)), dtype(0.0))
    denom = np.sqrt(denom_num2) * inv_r
    edge = np.where(
        denom > dtype(1e-15),
        (dtype(semiangle_rad) - alpha) / np.where(denom > 0, denom, dtype(1.0))
        + dtype(0.5),
        dtype(1.0),
    )
    return alpha2, cos2, sin2, np.clip(edge, dtype(0.0), dtype(1.0))


def _forward2(values, dtype):
    """Return the full-plane forward transform in the requested precision."""

    if np.dtype(dtype) == np.float32:
        return scipy_fft.fft2(values, axes=(-2, -1)).astype(np.complex64, copy=False)
    return np.fft.fft2(values, axes=(-2, -1))


def _inverse2(values, dtype):
    """Return the full-plane inverse transform in the requested precision."""

    if np.dtype(dtype) == np.float32:
        return scipy_fft.ifft2(values, axes=(-2, -1)).astype(np.complex64, copy=False)
    return np.fft.ifft2(values, axes=(-2, -1))


@dataclass(frozen=True)
class SSBReferenceResult:
    """Independent reference products for one fixed-aberration SSB request."""

    object_wave: np.ndarray
    loss: float
    mean_phase: np.ndarray
    num_bf: int
    active_bf: int
    inactive_bf: int
    dc_value: complex
    object_scale: float
    min_abs_object: float
    ill_conditioned_pixels: int

    def summary(self) -> dict[str, float]:
        """Return scalar diagnostics for reports and gates."""

        return {
            "num_bf": int(self.num_bf),
            "active_bf": int(self.active_bf),
            "inactive_bf": int(self.inactive_bf),
            "loss": float(self.loss),
            "object_scale": float(self.object_scale),
            "min_abs_object": float(self.min_abs_object),
            "ill_conditioned_pixels": int(self.ill_conditioned_pixels),
            "dc_real": float(np.real(self.dc_value)),
            "dc_imag": float(np.imag(self.dc_value)),
        }


def ssb_reference(
    counts,
    rows,
    cols,
    *,
    kx,
    ky,
    qx,
    qy,
    wavelength: float,
    semiangle_rad: float,
    det_sampling_rad: tuple[float, float],
    c10: float = 0.0,
    c12: float = 0.0,
    phi12: float = 0.0,
    rotation_angle_deg: float = 0.0,
    dc_value: complex | None = None,
    dtype=np.float64,
    chunk_bf: int = 64,
    ill_conditioned_ratio: float = 1e-3,
) -> SSBReferenceResult:
    """Evaluate the SSB object and objective for one exact BF selection.

    Parameters
    ----------
    counts : array-like
        ``(num_bf, scan_row, scan_col)`` exact integer detector counts for the
        selected bright-field pixels, in selection order.
    rows, cols : array-like
        Detector ``(row, col)`` of each selected BF pixel.
    kx, ky : array-like
        ``num_bf`` BF reciprocal-space coordinates in inverse angstroms.
    qx, qy : array-like
        Scan-grid reciprocal-space coordinates in inverse angstroms.
    wavelength : float
        Electron wavelength in angstroms.
    semiangle_rad : float
        Probe semi-convergence angle in radians.
    det_sampling_rad : tuple of float
        Detector angular sampling ``(row, col)`` in radians per pixel.
    c10, c12, phi12 : float
        Low-order aberrations in nanometers and radians.
    rotation_angle_deg : float
        Scan-detector rotation applied to ``kx``/``ky``.
    dc_value : complex, optional
        Corrected zero-frequency bin. Defaults to the exact mean raw
        zero-frequency value over the whole logical BF selection.
    dtype : numpy dtype
        ``float64`` for the reference oracle; ``float32`` for one
        straightforward single-precision realization of the same formula.

    Returns
    -------
    SSBReferenceResult
        Complex object, exact loss, mean phase image, and diagnostics.
    """

    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"dtype must be float32 or float64, got {dtype}.")
    complex_dtype = np.dtype(np.complex64 if dtype == np.float32 else np.complex128)
    scalar = dtype.type
    counts = np.asarray(counts)
    if counts.dtype.kind not in "iuf":
        raise TypeError(
            f"counts must contain numeric detector counts, got {counts.dtype}."
        )
    if counts.ndim != 3 or counts.shape[1] != counts.shape[2]:
        raise ValueError(
            "counts must be (num_bf, scan_row, scan_col) with a square scan, "
            f"got {counts.shape}."
        )
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    num_bf, side, _ = counts.shape
    if rows.size != num_bf or cols.size != num_bf:
        raise ValueError(
            f"BF rows/cols must have {num_bf} entries, got {rows.size}/{cols.size}."
        )
    kx = np.asarray(kx, dtype=np.float64).reshape(-1)
    ky = np.asarray(ky, dtype=np.float64).reshape(-1)
    if kx.size != num_bf or ky.size != num_bf:
        raise ValueError(
            f"kx/ky must have {num_bf} entries, got {kx.size}/{ky.size}."
        )
    qx = np.asarray(qx, dtype=np.float64).reshape(-1)
    qy = np.asarray(qy, dtype=np.float64).reshape(-1)
    if qx.size != side or qy.size != side:
        raise ValueError(
            f"qx/qy must each have {side} entries, got {qx.size}/{qy.size}."
        )

    ang_y_rad, ang_x_rad = (float(v) for v in det_sampling_rad)
    if rotation_angle_deg:
        angle = np.radians(-float(rotation_angle_deg))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        kx, ky = kx * cos_a + ky * sin_a, -kx * sin_a + ky * cos_a

    factor = np.pi / float(wavelength)
    cos2phi12 = np.cos(2.0 * float(phi12))
    sin2phi12 = np.sin(2.0 * float(phi12))
    c10 = float(c10)
    c12 = float(c12)

    qr = qx[:, None]
    qc = qy[None, :]

    raw_dc = counts.reshape(num_bf, -1).sum(axis=1, dtype=np.float64)
    dc = complex(np.mean(raw_dc)) if dc_value is None else complex(dc_value)

    # Static probe terms per BF pixel: p(k).
    alpha2_k, cos2_k, sin2_k, ap_k = geometry_terms(
        kx,
        ky,
        wavelength=wavelength,
        semiangle_rad=semiangle_rad,
        ang_y_rad=ang_y_rad,
        ang_x_rad=ang_x_rad,
        dtype=np.float64,
    )
    chi_k = factor * alpha2_k * (c12 * (cos2_k * cos2phi12 + sin2_k * sin2phi12) + c10)
    pk = ap_k * np.exp(-1j * chi_k)
    active_bf = int(np.count_nonzero(ap_k > 0.0))

    chunk = max(1, int(chunk_bf))
    object_sum = np.zeros((side, side), dtype=np.complex128)
    phase_sum = np.zeros((side, side), dtype=np.float64)
    phase_sumsq = np.zeros((side, side), dtype=np.float64)
    for start in range(0, num_bf, chunk):
        stop = min(start + chunk, num_bf)
        block = counts[start:stop].astype(dtype, copy=False)
        g = _forward2(block, dtype)
        kx_b = kx[start:stop][:, None, None]
        ky_b = ky[start:stop][:, None, None]
        alpha2_m, cos2_m, sin2_m, ap_m = geometry_terms(
            qr[None, :, :] - kx_b,
            qc[None, :, :] - ky_b,
            wavelength=wavelength,
            semiangle_rad=semiangle_rad,
            ang_y_rad=ang_y_rad,
            ang_x_rad=ang_x_rad,
            dtype=dtype,
        )
        alpha2_p, cos2_p, sin2_p, ap_p = geometry_terms(
            qr[None, :, :] + kx_b,
            qc[None, :, :] + ky_b,
            wavelength=wavelength,
            semiangle_rad=semiangle_rad,
            ang_y_rad=ang_y_rad,
            ang_x_rad=ang_x_rad,
            dtype=dtype,
        )
        chi_m = scalar(factor) * alpha2_m * (
            scalar(c12) * (cos2_m * scalar(cos2phi12) + sin2_m * scalar(sin2phi12))
            + scalar(c10)
        )
        chi_p = scalar(factor) * alpha2_p * (
            scalar(c12) * (cos2_p * scalar(cos2phi12) + sin2_p * scalar(sin2phi12))
            + scalar(c10)
        )
        imag_unit = complex_dtype.type(1j)
        pm = (ap_m * np.cos(chi_m)).astype(dtype) - imag_unit * (
            ap_m * np.sin(chi_m)
        ).astype(dtype)
        pp = (ap_p * np.cos(chi_p)).astype(dtype) - imag_unit * (
            ap_p * np.sin(chi_p)
        ).astype(dtype)
        pk_b = pk[start:stop].astype(complex_dtype)[:, None, None]
        gamma = pm * np.conjugate(pk_b) - np.conjugate(pp) * pk_b
        magnitude = np.abs(gamma)
        weight = np.conjugate(gamma) / np.maximum(magnitude, scalar(1e-8))
        corrected = g * weight
        corrected[:, 0, 0] = dc
        per_bf = _inverse2(corrected, dtype)
        object_sum += per_bf.sum(axis=0).astype(np.complex128)
        phase = np.arctan2(per_bf.imag, per_bf.real).astype(np.float64)
        phase_sum += phase.sum(axis=0)
        phase_sumsq += (phase * phase).sum(axis=0)

    object_wave = object_sum / num_bf
    mean_phase = phase_sum / num_bf
    loss = float(np.mean(phase_sumsq / num_bf - mean_phase * mean_phase))
    object_scale = float(np.max(np.abs(object_wave))) if object_wave.size else float("nan")
    min_abs = float(np.min(np.abs(object_wave))) if object_wave.size else float("nan")
    ill_conditioned = (
        int(np.count_nonzero(np.abs(object_wave) < ill_conditioned_ratio * object_scale))
        if object_scale > 0
        else 0
    )
    return SSBReferenceResult(
        object_wave=object_wave.astype(np.complex128, copy=False),
        loss=loss,
        mean_phase=mean_phase,
        num_bf=int(num_bf),
        active_bf=active_bf,
        inactive_bf=int(num_bf - active_bf),
        dc_value=dc,
        object_scale=object_scale,
        min_abs_object=min_abs,
        ill_conditioned_pixels=ill_conditioned,
    )
