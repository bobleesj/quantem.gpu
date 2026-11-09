"""Apply trained phase corrections through the current acquisition's normal equations."""

import math

import numpy as np
import torch
from torch.nn import functional as F

RADIAL_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.05, math.inf)


def _resample_axis(image: torch.Tensor, length: int, axis: int) -> torch.Tensor:
    """Fourier interpolate real counts; split the old Nyquist line on even grids."""
    old = image.shape[axis]
    if length == old:
        return image
    spectrum = torch.fft.rfft(image, dim=axis)
    if old % 2 == 0:
        spectrum.select(axis, old // 2).mul_(0.5)
    return (length / old) * torch.fft.irfft(spectrum, n=length, dim=axis)


def _readout(base, annular, radius, weights, kernels):
    """Linear combination of spectral, local convolution and annular image features."""
    sigma = base.std(correction=0).clamp_min(1e-10)
    spectrum = torch.fft.fft2(base)
    output = torch.zeros_like(spectrum)
    for weight, low, high in zip(weights[:6], RADIAL_EDGES, RADIAL_EDGES[1:]):
        output += weight * ((radius >= low) & (radius < high)) * spectrum

    normalized = F.pad((base / sigma)[None, None], (1, 1, 1, 1), mode="circular")
    convolution = torch.ops.aten._convolution(
        normalized, kernels, None, (1, 1), (0, 0), (1, 1), False, (0, 0), 1, False, False, True, False
    )[0]
    convolution[::2] = convolution[::2].tanh()
    convolution[1::2] = (convolution[1::2] - 0.5).relu()
    output += torch.fft.fft2(sigma * (weights[6:12, None, None] * convolution).sum(0))

    guide = annular.clamp_min(0).sqrt()
    guide = (guide - guide.mean()) / guide.std(correction=0).clamp_min(1e-9)
    guide = _resample_axis(_resample_axis(guide, base.shape[0], 0), base.shape[1], 1)
    for image, edges, coefficients in (
        (sigma * guide, (0.0, 0.5, 1.0, 1.5), weights[12:15]),
        (base * guide.tanh(), (0.0, 1.0, 1.5, 3.05), weights[15:18]),
    ):
        spectrum = torch.fft.fft2(image)
        for weight, low, high in zip(coefficients, edges, edges[1:]):
            output += weight * ((radius >= low) & (radius < high)) * spectrum
    return output


def readout_kernels() -> torch.Tensor:
    """Fixed zero-mean convolution basis used by schema version one."""
    rng = np.random.default_rng(621)
    kernels = rng.normal(size=(6, 3, 3))
    kernels -= kernels.mean(axis=(1, 2), keepdims=True)
    kernels /= np.linalg.norm(kernels, axis=(1, 2), keepdims=True)
    return torch.tensor(kernels[:, None], dtype=torch.float32).flip((-1, -2))


def _context(anchor, base):
    fields = []
    for image in (anchor, base):
        scaled = torch.fft.fft2(image).abs() / (math.sqrt(image.numel()) * image.std().clamp_min(1e-7))
        for window in (3, 9):
            padded = F.pad(scaled[None, None].square(), (window // 2,) * 4, mode="circular")
            fields.append(F.avg_pool2d(padded, window, stride=1)[0, 0].clamp_min(0).sqrt().log1p())
    return fields


def _scatter(spectrum, rows, cols, values):
    spectrum[rows, cols] = values
    spectrum[-rows, -cols] = values.conj()


def _direction_input(a, b, gram, rhs, prior, scale, qr, qc, radii, context, normalization):
    diagonal = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(0)
    response = (rhs - (gram @ a[..., None])[..., 0]) / (diagonal + 0.03 * scale * prior).clamp_min(1e-8)
    coherence = gram.abs() / (diagonal.sqrt()[..., None] * diagonal.sqrt()[:, None, :]).clamp_min(1e-8)
    features = [radii / 3.2, (qr.square() - qc.square()) / radii.square(), 2 * qr * qc / radii.square()]
    for k in (1, 2, 4, 8):
        angle = torch.pi * k * radii / 3.2
        features.extend((angle.sin(), angle.cos()))
    signal = diagonal / scale
    features.extend(
        (
            torch.log1p(a.abs() / normalization),
            torch.log1p(b.abs() / normalization),
            torch.log1p((a - b).abs() / normalization),
            (a * b.conj()).real / (a.abs() * b.abs()).clamp_min(1e-12),
            torch.log1p(10 * signal),
            torch.log1p(10 * signal / prior.clamp_min(1e-6)),
            *context,
            qr / radii,
            qc / radii,
        )
    )
    return torch.stack(features, -1), a, b, response, coherence, radii


def refine(prepared, base, annular, sampling, unet, heads, weights, kernels, *, noise_level=2**-.5):
    """Refine a single direct phase with measured counts, keeping all arrays on its device."""
    shape = tuple(base.shape)
    area = prepared.factor**2
    qr, qc = [torch.fft.fftfreq(n, d=s, device=base.device) for n, s in zip(shape, sampling)]
    radius = torch.hypot(qr[:, None], qc[None, :])
    sigma = base.std().clamp_min(1e-10)
    inputs = torch.stack((base / sigma, torch.full_like(base, noise_level)))[None]
    proposed = torch.fft.fft2(sigma * unet(inputs)[0, 0]) / area
    readout = _readout(base, annular, radius, weights, kernels) / area
    base_spectrum = torch.fft.fft2(base) / area
    anchor_spectrum = torch.zeros_like(base_spectrum)
    scale = torch.as_tensor(prepared.scale, device=base.device).float()
    for group in prepared.groups:
        rows, cols, gram, prior = [
            torch.from_dlpack(x) for x in (group.rows, group.cols, group.gram, group.prior)
        ]
        gram = gram.to(torch.complex64)
        d = 0.03 * scale * prior.float()
        matrix = gram + torch.diag_embed(2 * d)
        q = radius[rows, cols].clamp_min(1e-6)
        correction = torch.linalg.solve(matrix, (d * proposed[rows, cols])[..., None])[..., 0]
        low_penalty = 0.03 * scale * (0.15 / q) ** 4
        correction = (
            correction
            + torch.linalg.solve(
                matrix + torch.diag_embed(low_penalty), (d * readout[rows, cols])[..., None]
            )[..., 0]
        )
        _scatter(anchor_spectrum, rows, cols, base_spectrum[rows, cols] + correction)
    anchor = torch.fft.ifft2(anchor_spectrum).real * area
    context = _context(anchor, base)
    normalization = (base.std() * math.sqrt(base.numel()) / area).clamp_min(1e-5)
    refined = anchor_spectrum.clone()
    for group in prepared.groups:
        rows, cols, gram, rhs, prior = [
            torch.from_dlpack(x) for x in (group.rows, group.cols, group.gram, group.rhs, group.prior)
        ]
        for start in range(0, len(rows), 2048):
            block = slice(start, start + 2048)
            r, c = rows[block], cols[block]
            args = _direction_input(
                anchor_spectrum[r, c],
                base_spectrum[r, c],
                gram[block].to(torch.complex64),
                rhs[block].to(torch.complex64),
                prior[block].float(),
                scale,
                qr[r],
                qc[c],
                radius[r, c].clamp_min(1e-5),
                [v[r, c] for v in context],
                normalization,
            )
            delta = sum(head(*args) for head in heads) / len(heads)
            _scatter(refined, r, c, anchor_spectrum[r, c] + delta)
    return torch.fft.ifft2(refined).real * area
