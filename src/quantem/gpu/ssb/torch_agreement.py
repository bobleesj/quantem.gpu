"""The SSB fit's model agreement in plain PyTorch: the ground truth for the exact-derivative kernels and searches.

For band frequencies q (0.2 < |q| < 0.9 1/A on the stored half plane, weight w_q = 2, or 1 on the self-mirrored
columns) and bright-field pixels k:

    S(z) = sum_q w_q |N_q|^2 / D_q,   N_q = sum_k G(q, k) conj(gamma),   D_q = sum_k |gamma|^2
    gamma = b1 exp(-i (chi(k - q) - chi(k))) - b2 exp(+i (chi(k + q) - chi(k)))
    b1 = A(k - q) A(k) sinc(t/2 (-(pi/lambda)(lambda^2 |k - q|^2 - lambda^2 |k|^2) - 2 pi q.theta))
    b2 = A(k + q) A(k) sinc(t/2 (+(pi/lambda)(lambda^2 |k + q|^2 - lambda^2 |k|^2) - 2 pi q.theta))

with the soft aperture A, the 14-term chi of ``higher_order.CHI_SOURCE`` and the 28 dimensionless coordinates z of
``fifth_order_parameters.Coordinates`` (z[0:3] C10 and the C12 Cartesian pair, z[3:5] the sample tilt, z[5] the depth
spread t, z[6:] the higher orders), without the float32 rounding of the packed coefficients. A term with D_q = 0 is 0.
These are the equations of the CUDA and MPS objectives (``cuda.broad_objective``, ``mps.fifth_order``) and of the
float64 reconstruction oracle (sections 1.3 to 1.7 of its notes): the same aperture, chi, depth weights and sidebands.

Everything is ordinary tensor arithmetic in real numbers (no custom kernels, no complex dtype), so the same code runs
on any torch device and dtype: float64 on the CPU is the reference, float32 on CUDA or MPS a check of float32
arithmetic. ``torch.autograd`` gives the gradient and Hessian. The bright-field pixels are summed in chunks;
``value_and_gradient`` applies the chain rule through the per-frequency sums (one pass for N and D, one backward pass
per chunk), so the full-size gradient needs memory for one chunk only.

``from_counts`` builds G from bright-field counts (the forward 2-D DFT of each pixel's scan image, in the reference
dtype); ``from_session`` reads a CUDA or MPS session's own G, k and band, so a kernel and the reference see the same
float32 inputs and differ only in arithmetic.
"""

import math

import numpy as np
import torch

from quantem.gpu.ssb.higher_order import ORDERS

_N = [n for n, _ in ORDERS]
_M = [m for _, m in ORDERS]
_POWER = [(n + 1 - m) // 2 for n, m in ORDERS]


def band_frequencies(shape, scan_sampling_A, band_inv_A=(0.2, 0.9)):
    """Stored half-plane band frequencies: (row index, column index, q_row, q_col, weight), by the CUDA float32 rule.

    q_row = fftfreq(rows, s_r) and q_col = fftfreq(cols, s_c)[: cols // 2 + 1] in float32, |q| = hypot in float32,
    0.2 < |q| < 0.9; weight 1 on column 0 and, for an even scan, column cols / 2 (their own mirror), else 2.
    """
    rows, cols = (int(v) for v in shape)
    step = np.broadcast_to(np.asarray(scan_sampling_A, dtype=np.float64), (2,))
    q_row = np.fft.fftfreq(rows, step[0]).astype(np.float32)
    q_col = np.fft.fftfreq(cols, step[1]).astype(np.float32)[: cols // 2 + 1]
    magnitude = np.hypot(q_row[:, None], q_col[None, :])
    r, c = np.nonzero((magnitude > band_inv_A[0]) & (magnitude < band_inv_A[1]))
    weight = np.where((c == 0) | ((cols % 2 == 0) & (c == cols // 2)), 1.0, 2.0)
    return r, c, q_row[r].astype(np.float64), q_col[c].astype(np.float64), weight


def fifth_from_low(z6):
    """The 28 coordinates of a low-order point z6 = (C10, C12 cos 2 phi12, C12 sin 2 phi12 [A], tilt row, tilt column
    [mrad], depth spread [A]) with every higher-order term 0."""
    z6 = torch.as_tensor(z6)
    scale = torch.tensor([100.0, 100.0, 100.0, 5.0, 5.0, 100.0], dtype=z6.dtype, device=z6.device)
    return torch.cat([z6 / scale, torch.zeros(22, dtype=z6.dtype, device=z6.device)])


def _sinc(x):
    """sin(x) / x, by its series below |x| = 1e-2 so that every derivative is exact at 0."""
    small = x.abs() < 1e-2
    safe = torch.where(small, torch.ones_like(x), x)
    x2 = x * x
    series = 1 - x2 / 6 * (1 - x2 / 20 * (1 - x2 / 42))
    return torch.where(small, series, torch.sin(safe) / safe)


class TorchAgreement:
    """S(z) on band evidence G(q, k) with autograd derivatives (see the module docstring).

    Parameters
    ----------
    g_real, g_imag : torch.Tensor
        G at the band frequencies, shape (bright-field pixels, band frequencies), in the reference dtype.
    k_row, k_col : torch.Tensor
        Rotated bright-field pixel frequencies (1/A), shape (pixels,).
    q_row, q_col, weight : torch.Tensor
        Band frequencies (1/A) and their half-plane weights, shape (band frequencies,).
    wavelength, semiangle_rad : float
        Electron wavelength (A) and convergence semiangle (rad).
    detector_sampling_rad : tuple of float
        Detector angular sampling (row, column) in rad, which sets the soft aperture edge.
    chunk : int
        Bright-field pixels per chunk.
    """

    def __init__(self, g_real, g_imag, k_row, k_col, q_row, q_col, weight, *, wavelength, semiangle_rad,
                 detector_sampling_rad, chunk=64):
        self.g_real, self.g_imag = g_real, g_imag
        self.k_row, self.k_col = k_row, k_col
        self.q_row, self.q_col, self.weight = q_row, q_col, weight
        self.dtype, self.device = g_real.dtype, g_real.device
        self.wavelength = float(wavelength)
        self.semiangle = float(semiangle_rad)
        self.sampling = tuple(float(v) for v in np.broadcast_to(detector_sampling_rad, (2,)))
        self.chunk = int(chunk)
        # z -> packed coefficients: scale of each of the 28 coordinates and its slot in the 28 packed floats
        slots, scales = [0, 2, 3], [50.0, 50.0, 50.0]
        for i, (n, m) in enumerate(ORDERS):
            if n < 2:
                continue
            scale = self.wavelength / (2 * math.pi * self.semiangle ** (n + 1))
            slots.append(2 * i)
            scales.append(scale)
            if m:
                slots.append(2 * i + 1)
                scales.append(scale)
        self._slots = torch.tensor(slots, device=self.device)
        self._optical = torch.tensor([0, 1, 2] + list(range(6, 28)), device=self.device)
        self._scales = torch.tensor(scales, dtype=self.dtype, device=self.device)

    @property
    def num_bright_field_pixels(self) -> int:
        return int(self.k_row.numel())

    # ----- constructors

    @classmethod
    def from_arrays(cls, g, k_row, k_col, q_row, q_col, weight, *, dtype=torch.float64, device="cpu", **options):
        """From host or device arrays; ``g`` complex (pixels, band frequencies)."""
        def tensor(values):
            return torch.as_tensor(np.asarray(values, dtype=np.float64)).to(device=device, dtype=dtype)

        g = np.asarray(g)
        return cls(tensor(g.real), tensor(g.imag), tensor(k_row), tensor(k_col), tensor(q_row), tensor(q_col),
                   tensor(weight), **options)

    @classmethod
    def from_counts(cls, counts, *, pixel_rows, pixel_cols, center_px, rotation_deg, wavelength, semiangle_rad,
                    scan_sampling_A, detector_sampling_rad, band_inv_A=(0.2, 0.9), dtype=torch.float64,
                    device="cpu", chunk=64, fft_chunk=256):
        """G from bright-field counts (pixels, scan rows, scan columns): the unnormalized forward DFT of each scan image.

        k_r0 = (p_r - c_r) dtheta_r / lambda, k_c0 = (p_c - c_c) dtheta_c / lambda, rotated passively by the
        scan-detector rotation: k_r = k_r0 cos a + k_c0 sin a, k_c = -k_r0 sin a + k_c0 cos a, a = -rotation.
        """
        counts_shape = tuple(np.shape(counts))
        r, c, q_row, q_col, weight = band_frequencies(counts_shape[1:], scan_sampling_A, band_inv_A)
        rows_t = torch.as_tensor(r, device=device)
        cols_t = torch.as_tensor(c, device=device)
        g_real = torch.empty((counts_shape[0], r.size), dtype=dtype, device=device)
        g_imag = torch.empty_like(g_real)
        for start in range(0, counts_shape[0], fft_chunk):
            block = torch.as_tensor(np.asarray(counts[start:start + fft_chunk])).to(device=device, dtype=dtype)
            spectra = torch.fft.rfft2(block)
            values = spectra[:, rows_t, cols_t]
            g_real[start:start + len(block)] = values.real
            g_imag[start:start + len(block)] = values.imag
            del spectra, values, block
        dth = np.broadcast_to(np.asarray(detector_sampling_rad, dtype=np.float64), (2,))
        kr0 = (np.asarray(pixel_rows, np.float64) - center_px[0]) * dth[0] / wavelength
        kc0 = (np.asarray(pixel_cols, np.float64) - center_px[1]) * dth[1] / wavelength
        a = -math.radians(rotation_deg)
        k_row = kr0 * math.cos(a) + kc0 * math.sin(a)
        k_col = -kr0 * math.sin(a) + kc0 * math.cos(a)

        def tensor(values):
            return torch.as_tensor(values).to(device=device, dtype=dtype)

        return cls(g_real, g_imag, tensor(k_row), tensor(k_col), tensor(q_row), tensor(q_col), tensor(weight),
                   wavelength=wavelength, semiangle_rad=semiangle_rad, detector_sampling_rad=tuple(dth), chunk=chunk)

    @classmethod
    def from_session(cls, ssb, *, band_inv_A=(0.2, 0.9), dtype=torch.float64, device="cpu", chunk=64):
        """The CUDA or MPS session's own float32 G, k, band and calibration (at its current rotation)."""
        backend = ssb._backend()
        if ssb.backend == "cuda":
            engine = backend._get_accelerator()
            cache = engine._cache
            _, _, flat, qx, qy, weight = engine.thick._prepare_fit_band(tuple(band_inv_A))
            g = engine.G_qk.reshape(engine.G_qk.shape[0], -1)[:, flat]
            host = lambda array: array.get()  # noqa: E731
            g, kx, ky = host(g), host(cache["kx_bf"]), host(cache["ky_bf"])
            qx, qy, weight = host(qx.ravel()), host(qy.ravel()), host(weight)
            wavelength, semiangle = float(engine.wavelength), float(cache["semiangle_rad"])
            sampling = (float(cache["ang_y_rad"]), float(cache["ang_x_rad"]))
        elif ssb.backend == "mps":
            from quantem.gpu.ssb.mps.thick_sample import _band

            backend.cache_rotation(math.radians(ssb.scan_detector_rotation_deg))
            p = backend._prepared
            flat, qx, qy, weight = _band(p, tuple(float(v) for v in band_inv_A))
            g = np.asarray(p.g_qk).reshape(int(p.g_qk.shape[0]), -1)[:, np.asarray(flat).astype(np.int64)]
            kx, ky, qx, qy = (np.asarray(v) for v in (p.kx, p.ky, qx, qy))
            wavelength, semiangle = float(p.wavelength), float(p.semiangle_rad)
            sampling = (float(p.ang_y_rad), float(p.ang_x_rad))
        else:
            raise ValueError(f"from_session reads CUDA or MPS sessions, not {ssb.backend!r}.")
        # the kernels' float32 values, exactly, in the reference dtype
        return cls.from_arrays(g, kx, ky, qx, qy, weight, dtype=dtype, device=device, wavelength=wavelength,
                               semiangle_rad=semiangle, detector_sampling_rad=sampling, chunk=chunk)

    def subsample(self, stride: int) -> "TorchAgreement":
        """Every ``stride``-th bright-field pixel (the BF subsample of the exact-gradient search)."""
        return TorchAgreement(self.g_real[::stride].contiguous(), self.g_imag[::stride].contiguous(),
                              self.k_row[::stride].contiguous(), self.k_col[::stride].contiguous(), self.q_row,
                              self.q_col, self.weight, wavelength=self.wavelength, semiangle_rad=self.semiangle,
                              detector_sampling_rad=self.sampling, chunk=self.chunk)

    def to(self, *, dtype=None, device=None) -> "TorchAgreement":
        convert = lambda t: t.to(dtype=dtype or self.dtype, device=device or self.device)  # noqa: E731
        return TorchAgreement(*(convert(t) for t in (self.g_real, self.g_imag, self.k_row, self.k_col, self.q_row,
                                                     self.q_col, self.weight)),
                              wavelength=self.wavelength, semiangle_rad=self.semiangle,
                              detector_sampling_rad=self.sampling, chunk=self.chunk)

    # ----- model

    def packed(self, z):
        """The 28 packed Cartesian coefficients C_nm / (n + 1) (cos, sin) in A of coordinates z."""
        out = torch.zeros(28, dtype=z.dtype, device=z.device)
        return out.index_put((self._slots,), z[self._optical] * self._scales)

    def chi(self, row, col, packed):
        """chi(p) of ``higher_order.CHI_SOURCE``: (2 pi / lambda) sum_i rho2^p_i (Re z^m_i a_i + Im z^m_i b_i)."""
        x, y = row * self.wavelength, col * self.wavelength
        r2 = x * x + y * y
        radial = [torch.ones_like(r2), r2, r2 * r2, r2 * r2 * r2]
        re, im = [torch.ones_like(x)], [torch.zeros_like(x)]
        for _ in range(6):
            re.append(re[-1] * x - im[-1] * y)
            im.append(im[-1] * x + re[-2] * y)
        total = torch.zeros_like(r2)
        for i in range(14):
            total = total + radial[_POWER[i]] * (re[_M[i]] * packed[2 * i] + im[_M[i]] * packed[2 * i + 1])
        return (2 * math.pi / self.wavelength) * total

    def aperture(self, row, col):
        """A(p) = clamp((alpha - lambda |p|) / denom(p) + 1/2, 0, 1), denom = |(p_r dtheta_r, p_c dtheta_c)| / |p|."""
        r2 = row * row + col * col
        r = torch.sqrt(r2)
        safe_r = torch.where(r > 1e-15, r, torch.ones_like(r))
        denom = torch.sqrt((row * self.sampling[0]) ** 2 + (col * self.sampling[1]) ** 2) / safe_r
        safe = torch.where(denom > 1e-15, denom, torch.ones_like(denom))
        edge = torch.where((r > 1e-15) & (denom > 1e-15), (self.semiangle - r * self.wavelength) / safe + 0.5,
                           torch.ones_like(r))
        return edge.clamp(0.0, 1.0)

    def _chunk_sums(self, z, start, stop):
        """N_q (real, imaginary) and D_q of bright-field pixels start..stop-1."""
        packed = self.packed(z)
        t = z[5] * 100
        theta_r, theta_c = z[3] * 5e-3, z[4] * 5e-3
        kr = self.k_row[start:stop, None]
        kc = self.k_col[start:stop, None]
        qr, qc = self.q_row[None], self.q_col[None]
        lam = self.wavelength
        a_k = self.aperture(kr, kc)
        chi_k = self.chi(kr, kc, packed)
        k2 = (kr * kr + kc * kc) * lam * lam
        shift = 2 * math.pi * (qr * theta_r + qc * theta_c)
        mr, mc = kr - qr, kc - qc
        pr, pc = kr + qr, kc + qc
        x1 = 0.5 * t * (-(math.pi / lam) * ((mr * mr + mc * mc) * lam * lam - k2) - shift)
        x2 = 0.5 * t * ((math.pi / lam) * ((pr * pr + pc * pc) * lam * lam - k2) - shift)
        b1 = self.aperture(mr, mc) * a_k * _sinc(x1)
        b2 = self.aperture(pr, pc) * a_k * _sinc(x2)
        phi1 = self.chi(mr, mc, packed) - chi_k
        phi2 = self.chi(pr, pc, packed) - chi_k
        gr = b1 * torch.cos(phi1) - b2 * torch.cos(phi2)
        gi = -b1 * torch.sin(phi1) - b2 * torch.sin(phi2)
        g_re, g_im = self.g_real[start:stop], self.g_imag[start:stop]
        return ((g_re * gr + g_im * gi).sum(0), (g_im * gr - g_re * gi).sum(0), (gr * gr + gi * gi).sum(0))

    def sums(self, z):
        """N_q (real, imaginary) and D_q over every bright-field pixel (differentiable)."""
        n_re = n_im = d = None
        for start in range(0, self.num_bright_field_pixels, self.chunk):
            stop = min(self.num_bright_field_pixels, start + self.chunk)
            a, b, c = self._chunk_sums(z, start, stop)
            n_re, n_im, d = (a, b, c) if n_re is None else (n_re + a, n_im + b, d + c)
        return n_re, n_im, d

    def combine(self, n_re, n_im, d):
        covered = d > 0
        safe = torch.where(covered, d, torch.ones_like(d))
        return torch.where(covered, self.weight * (n_re * n_re + n_im * n_im) / safe, torch.zeros_like(d)).sum()

    def _coordinates(self, z):
        return torch.as_tensor(np.asarray(z, dtype=np.float64) if not torch.is_tensor(z) else z).to(
            dtype=self.dtype, device=self.device)

    def value(self, z):
        """S(z) as a tensor; differentiable when z requires grad (the whole graph, for small problems)."""
        z = z if torch.is_tensor(z) and z.requires_grad else self._coordinates(z)
        return self.combine(*self.sums(z))

    def agreement(self, z) -> float:
        with torch.no_grad():
            return float(self.value(self._coordinates(z)))

    def value_and_gradient(self, z):
        """S(z) and dS/dz by the chain rule through N_q and D_q: memory of one chunk."""
        z = self._coordinates(z)
        with torch.no_grad():
            totals = self.sums(z)
        leaves = [t.clone().requires_grad_(True) for t in totals]
        value = self.combine(*leaves)
        adjoints = torch.autograd.grad(value, leaves)
        variable = z.clone().requires_grad_(True)
        gradient = torch.zeros_like(z)
        for start in range(0, self.num_bright_field_pixels, self.chunk):
            stop = min(self.num_bright_field_pixels, start + self.chunk)
            parts = self._chunk_sums(variable, start, stop)
            surrogate = sum((part * adjoint).sum() for part, adjoint in zip(parts, adjoints))
            gradient += torch.autograd.grad(surrogate, variable)[0]
        return float(value.detach()), gradient.detach().cpu().numpy().astype(np.float64)

    def hessian(self, z, coordinates=None):
        """d2S/dz2 on ``coordinates`` (indices into z, default all 28) by double autograd of the full graph."""
        z = self._coordinates(z)
        index = torch.arange(28, device=self.device) if coordinates is None else torch.as_tensor(
            list(coordinates), device=self.device)

        def restricted(values):
            full = z.index_put((index,), values)
            return self.value(full)

        hessian = torch.autograd.functional.hessian(restricted, z[index].clone())
        return hessian.detach().cpu().numpy().astype(np.float64)

    def gradient_differences(self, z, step=1e-4, coordinates=None):
        """Symmetrized central differences of exact gradients: a full-size Hessian with chunk memory."""
        z = np.asarray(z, dtype=np.float64)
        index = list(range(28)) if coordinates is None else list(coordinates)
        rows = []
        for i in index:
            plus, minus = z.copy(), z.copy()
            plus[i] += step
            minus[i] -= step
            rows.append((self.value_and_gradient(plus)[1] - self.value_and_gradient(minus)[1])[index] / (2 * step))
        rows = np.array(rows)
        return 0.5 * (rows + rows.T)
