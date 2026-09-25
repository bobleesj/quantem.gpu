"""Single-sideband (SSB) ptychography in plain torch: the readable reference for the fast backends.

Same formulation as the CUDA engine (``backends/cuda/engine.py`` + ``kernels/common.py``), written as ordinary tensor code so it
runs on any torch device (CUDA, MPS, CPU) and so the fused kernels have one exact reference to be tested against.

Data: ``G[k, q]`` = 2-D Fourier transform over scan positions of the intensity recorded by bright-field detector pixel k,
stored as the Hermitian half-plane of q (the bright-field images are real), with the q = 0 term replaced by ``dc_value``.

Standard SSB, per (q, k): with the probe P(v) = A(v) exp(-i chi(v)),
    gamma = P(q - k) conj P(k) - conj P(q + k) P(k)         (the two double-overlap "trotters")
    corrected_k(q) = G[k, q] conj(gamma / |gamma|)          (phase-only correction)
    phase = mean_k angle(ifft2 corrected_k)                 (per-pixel phase images, averaged)
    loss  = mean over the image of var_k angle(ifft2 corrected_k)
chi(v) = (pi / lambda) alpha^2 (C10 + C12 cos 2(phi_v - phi12)), alpha = lambda |v|; A is the soft aperture
clip((semiangle - alpha) / denom + 0.5, 0, 1), denom = |(cos phi_v ang_y, sin phi_v ang_x)|.

Thick sample (tilt theta, thickness t, C10 at mid-depth): the slice at depth z sees defocus C10 + z and is shifted by z theta;
averaging each trotter over depth multiplies it by a real weight
    w1 = sinc(rate1 t / 2), rate1 = -(pi/lambda)(alpha_{q-k}^2 - alpha_k^2) - 2 pi q.theta
    w2 = sinc(rate2 t / 2), rate2 = +(pi/lambda)(alpha_{q+k}^2 - alpha_k^2) - 2 pi q.theta
    gamma = w1 t1 - w2 t2     (t = 0: w = 1, standard SSB exactly)
Tilt fit objective (what finds the tilt; the phase-variance loss does not): least squares of G = Psi gamma,
    fit = sum_{q in band} |sum_k G conj(gamma)|^2 / sum_k |gamma|^2, evaluated on the half-plane (the -q term equals the q term).

Units follow the engine: C10, C12, thickness in Angstrom; phi12 rad; tilt mrad in the scan frame; q, k in 1/Angstrom. The public
``SSB`` session converts from nm.
"""
from __future__ import annotations

import math
from typing import Self

import numpy as np
import torch


class TorchSSB:
    """SSB on a prepared bright-field Fourier stack ``G`` (see module docstring)."""

    def __init__(self, G: torch.Tensor, *, kx: torch.Tensor, ky: torch.Tensor, qx: torch.Tensor, qy: torch.Tensor, nx: int,
                 wavelength: float, semiangle_rad: float, ang_y_rad: float, ang_x_rad: float, dc_value: complex,
                 chunk_bytes: int = 1 << 30):
        self.G, self.kx, self.ky, self.qx, self.qy = G, kx, ky, qx, qy
        self.num_bf, self.ny = G.shape[0], G.shape[1]
        self.nx = int(nx)
        self.half = G.shape[2] == self.nx // 2 + 1
        self.wavelength, self.semiangle, self.ang_y, self.ang_x = float(wavelength), float(semiangle_rad), float(ang_y_rad), float(ang_x_rad)
        self.factor = math.pi / self.wavelength
        self.dc_value = complex(dc_value)
        self.chunk_bytes = int(chunk_bytes)
        self.device = G.device

    @classmethod
    def from_ssb(cls, ssb, device: str | torch.device | None = None) -> Self:
        """The exact G and geometry of a prepared CUDA ``SSB`` session (same rotation), as torch tensors on ``device``."""
        backend = ssb._backend_protocol
        engine = backend._get_accelerator()
        engine.cache_rotation(backend._rotation_angle_rad)
        c = engine._cache
        import cupy as cp

        def tensor(array):
            out = torch.from_dlpack(cp.ascontiguousarray(array))
            return out if device is None else out.to(device)

        return cls(tensor(engine.G_qk), kx=tensor(c["kx_bf"]), ky=tensor(c["ky_bf"]), qx=tensor(c["qx_1d"]), qy=tensor(c["qy_1d"]),
                   nx=int(c["nx"]), wavelength=float(engine.wavelength), semiangle_rad=float(c["semiangle_rad"]),
                   ang_y_rad=float(c["ang_y_rad"]), ang_x_rad=float(c["ang_x_rad"]), dc_value=engine._dc_value_host)

    # --- the model ---------------------------------------------------------------------------------------------------------

    def _geometry(self, vx: torch.Tensor, vy: torch.Tensor):
        """alpha^2, cos 2phi, sin 2phi and the soft aperture of displacement v (same algebra as the CUDA compute_geometry)."""
        r2 = vx * vx + vy * vy
        alpha2 = r2 * self.wavelength * self.wavelength
        inv_r2 = torch.where(r2 > 1e-30, 1.0 / r2.clamp_min(1e-30), torch.zeros_like(r2))
        cos2 = (vx * vx - vy * vy) * inv_r2
        sin2 = 2.0 * vx * vy * inv_r2
        r = torch.sqrt(r2)
        inv_r = torch.where(r > 1e-15, 1.0 / r.clamp_min(1e-15), torch.zeros_like(r))
        denom = torch.sqrt((vx * self.ang_y) ** 2 + (vy * self.ang_x) ** 2) * inv_r
        edge = torch.where(denom > 1e-15, (self.semiangle - r * self.wavelength) / denom.clamp_min(1e-15) + 0.5, torch.ones_like(r))
        return alpha2, cos2, sin2, edge.clamp(0.0, 1.0)

    def _gamma(self, qx, qy, kx, ky, C10, C12, phi12, tilt_mrad=(0.0, 0.0), thickness=0.0) -> torch.Tensor:
        """Unnormalised gamma = w1 P(q-k) conj P(k) - w2 conj P(q+k) P(k) on broadcast (k, q) grids."""
        c2, s2 = math.cos(2.0 * phi12), math.sin(2.0 * phi12)

        def chi(alpha2, cos2, sin2):
            return self.factor * alpha2 * (C10 + C12 * (cos2 * c2 + sin2 * s2))

        a_k, cos_k, sin_k, ap_k = self._geometry(kx, ky)
        a_m, cos_m, sin_m, ap_m = self._geometry(qx - kx, qy - ky)
        a_p, cos_p, sin_p, ap_p = self._geometry(qx + kx, qy + ky)
        chi_k, chi_m, chi_p = chi(a_k, cos_k, sin_k), chi(a_m, cos_m, sin_m), chi(a_p, cos_p, sin_p)
        w1 = w2 = 1.0
        if thickness > 0.0:
            shift = 2.0 * math.pi * (qx * tilt_mrad[0] * 1e-3 + qy * tilt_mrad[1] * 1e-3)
            rate1 = -self.factor * (a_m - a_k) - shift
            rate2 = self.factor * (a_p - a_k) - shift
            w1 = torch.special.sinc(0.5 * rate1 * thickness / math.pi)     # torch sinc(x) = sin(pi x) / (pi x)
            w2 = torch.special.sinc(0.5 * rate2 * thickness / math.pi)
        t1 = (w1 * ap_m * ap_k) * torch.exp(-1j * (chi_m - chi_k))
        t2 = (w2 * ap_p * ap_k) * torch.exp(1j * (chi_p - chi_k))
        return t1 - t2

    def _full_plane(self, block: torch.Tensor) -> torch.Tensor:
        """Expand a half-plane block (b, ny, nx//2+1) to the full plane with G(-q) = conj G(q)."""
        if not self.half:
            return block
        ny, nx = self.ny, self.nx
        neg_rows = torch.as_tensor((-np.arange(ny)) % ny, device=self.device)
        neg_cols = torch.as_tensor((nx - np.arange(nx // 2 + 1, nx)) % nx, device=self.device)
        return torch.cat([block, torch.conj(block[:, neg_rows][:, :, neg_cols])], dim=2)

    def _chunk(self, per_pixel_bytes: int) -> int:
        return max(1, self.chunk_bytes // max(1, per_pixel_bytes))

    # --- public -----------------------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def reconstruct(self, C10: float, C12: float, phi12: float, tilt_mrad=(0.0, 0.0), thickness: float = 0.0,
                    compute_loss: bool = True) -> tuple[torch.Tensor, float | None]:
        """Mean phase over bright-field pixels and the phase-variance loss (standard SSB at thickness 0)."""
        ny, nx = self.ny, self.nx
        qx = self.qx.reshape(1, ny, 1); qy = self.qy.reshape(1, 1, nx)
        phase_sum = torch.zeros((ny, nx), dtype=torch.float32, device=self.device)
        phase_sumsq = torch.zeros((ny, nx), dtype=torch.float32, device=self.device)
        chunk = self._chunk(ny * nx * 8 * 4)
        for start in range(0, self.num_bf, chunk):
            stop = min(self.num_bf, start + chunk)
            full = self._full_plane(self.G[start:stop])
            gamma = self._gamma(qx, qy, self.kx[start:stop].reshape(-1, 1, 1), self.ky[start:stop].reshape(-1, 1, 1),
                                C10, C12, phi12, tilt_mrad, thickness)
            # gamma / |gamma|; where gamma vanishes (no overlap) the CUDA kernel scales by 1e8, which keeps it ~0 - same here
            mag2 = gamma.real ** 2 + gamma.imag ** 2
            unit = gamma * torch.where(mag2 > 1e-16, torch.rsqrt(mag2.clamp_min(1e-16)), torch.full_like(mag2, 1e8))
            corrected = full * torch.conj(unit)
            corrected[:, 0, 0] = self.dc_value
            angles = torch.angle(torch.fft.ifft2(corrected))
            phase_sum += angles.sum(0)
            if compute_loss:
                phase_sumsq += (angles * angles).sum(0)
        mean = phase_sum / self.num_bf
        loss = float((phase_sumsq / self.num_bf - mean * mean).mean()) if compute_loss else None
        return mean, loss

    @torch.inference_mode()
    def fit_power(self, C10: float, C12: float, phi12: float, tilt_mrad=(0.0, 0.0), thickness: float = 0.0,
                  band_inv_A: tuple[float, float] = (0.2, 0.9)) -> float:
        """Least-squares agreement of the (thick) SSB model with G over a q band, on the half-plane (see module docstring)."""
        ny, nx = self.ny, self.nx
        cols = self.G.shape[2]
        qx = self.qx.reshape(ny, 1).expand(ny, cols); qy = self.qy[:cols].reshape(1, cols).expand(ny, cols)
        q = torch.hypot(qx, qy)
        inside = (q > band_inv_A[0]) & (q < band_inv_A[1])
        rows_idx, cols_idx = torch.nonzero(inside, as_tuple=True)
        if self.half:
            self_mirror = (cols_idx == 0) | ((nx % 2 == 0) & (cols_idx == nx // 2))
            weight = torch.where(self_mirror, 1.0, 2.0)
        else:
            weight = torch.ones(rows_idx.shape, device=self.device)
        qx_b = self.qx[rows_idx].reshape(1, -1); qy_b = self.qy[cols_idx].reshape(1, -1)
        flat = rows_idx * cols + cols_idx
        g_flat = self.G.reshape(self.num_bf, -1)
        numerator = torch.zeros(flat.shape, dtype=torch.complex64, device=self.device)
        denominator = torch.zeros(flat.shape, dtype=torch.float32, device=self.device)
        chunk = self._chunk(int(flat.numel()) * 8 * 4)
        for start in range(0, self.num_bf, chunk):
            stop = min(self.num_bf, start + chunk)
            gamma = self._gamma(qx_b, qy_b, self.kx[start:stop].reshape(-1, 1), self.ky[start:stop].reshape(-1, 1),
                                C10, C12, phi12, tilt_mrad, thickness)
            numerator += (g_flat[start:stop][:, flat] * torch.conj(gamma)).sum(0)
            denominator += (gamma.real ** 2 + gamma.imag ** 2).sum(0)
        keep = denominator > 0
        return float((weight[keep] * numerator[keep].abs() ** 2 / denominator[keep]).sum())

    def fit_sample(self, *, band_inv_A: tuple[float, float] = (0.2, 0.9), **options) -> dict[str, object]:
        """Fit C10, C12, phi12, tilt and thickness (search shared with the CUDA / MPS backends: ``ssb._thick_fit``)."""
        from ._thick_fit import fit_sample_search

        return fit_sample_search(lambda c10, c12, phi12, tilt, t: self.fit_power(c10, c12, phi12, tilt, t, band_inv_A),
                                 band_inv_A=band_inv_A, **options)
