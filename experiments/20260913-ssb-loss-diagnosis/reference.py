"""Independent double-precision full-plane reference and symmetry isolation."""

import json

import numpy as np

n = 512
index = np.arange(3 * n * n)
counts = ((index * 17 + index // 113 + 3) % 251).reshape(3, n, n)
q = np.where(np.arange(n) < n // 2, np.arange(n), np.arange(n) - n) * 0.001
row, col = q[:, None], q[None, :]
xx = 55 + 13 * np.cos(2 * 0.23)
xy = 13 * np.sin(2 * 0.23)
yy = 55 - 13 * np.cos(2 * 0.23)
factor = np.pi * 0.025
chi = factor * (xx * row**2 + 2 * xy * row * col + yy * col**2)


def aperture(row, col):
    return np.clip((0.02 - np.hypot(row, col) * 0.025) / 0.001 + 0.5, 0, 1)


def loss(objects):
    phases = np.angle(objects)
    return float(np.mean(np.mean(phases**2, axis=0) - np.mean(phases, axis=0)**2))


results = []
for remove_nyquist in (False, True):
    spectra = []
    for plane, kr, kc in zip(counts, [0.1, 0.2, 0.3], [0.2, -0.1, 0.1]):
        minus = aperture(row - kr, col - kc)
        plus = aperture(row + kr, col + kc)
        cross = 2 * factor * (xx * row * kr + xy * (row * kc + col * kr) + yy * col * kc)
        gamma = np.exp(1j * cross) * ((minus - plus) * np.cos(chi) - 1j * (minus + plus) * np.sin(chi))
        corrected = np.fft.fft2(plane) * gamma.conj() / np.maximum(np.abs(gamma), 1e-8)
        corrected[0, 0] = 0
        if remove_nyquist:
            corrected[n // 2, :] = 0
            corrected[:, n // 2] = 0
        spectra.append(corrected)
    spectra = np.array(spectra)
    # Reproduce the cached algorithm's half-plane + real-output projection,
    # independently of Metal's implementation and FFT arithmetic.
    first = np.fft.ifft(-1j * spectra[:, :, : n // 2 + 1], axis=1)
    mirrored = np.concatenate([first, first[:, :, 1:n // 2][:, :, ::-1].conj()], axis=2)
    projected = 1j * np.fft.ifft(mirrored, axis=2).real
    full = np.fft.ifft2(spectra)
    for dc in (0, 100_000_000):
        a = loss(full + dc / n**2)
        b = loss(projected + dc / n**2)
        results.append(dict(remove_nyquist=remove_nyquist, dc=dc, full_loss=a,
                            projected_loss=b, relative_difference=abs(a-b)/max(abs(a), 1e-30),
                            maximum_object_difference=float(np.max(np.abs(full-projected)))))
print(json.dumps(results, indent=2))
