"""Measure support-aware complex64 storage without altering the source data."""

import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
g = json.loads((root / "geometry.json").read_text())
n = 512
qr = np.asarray(g["qrow"], dtype=np.float64)[:, None]
qc = np.asarray(g["qcol"], dtype=np.float64)[None, :]
half_col = qc[:, :257]
radius = (g["semiangle"] + 0.5 * g["sampling"]) / g["wavelength"]
# Conservative margin for Float geometry and Metal fast-math boundaries.
bound = radius + 1e-4
angle = 158.88268568029937


def rotated(index, degrees):
    a = np.deg2rad(-degrees)
    kr, kc = g["kr"][index], g["kc"][index]
    return kr * np.cos(a) + kc * np.sin(a), kc * np.cos(a) - kr * np.sin(a)


def support(index, degrees=angle, universal=False):
    kr, kc = rotated(index, degrees)
    if universal:
        keep = np.hypot(qr, half_col) <= bound + np.hypot(kr, kc)
    else:
        keep = ((qr - kr)**2 + (half_col - kc)**2 <= bound**2) | (
            (qr + kr)**2 + (half_col + kc)**2 <= bound**2)
    # Preserve self-conjugate boundaries, DC, and exceptional Nyquist lines.
    keep[[0, 256], :] = True
    keep[:, [0, 256]] = True
    return keep


def interval_mask(mask):
    interior = mask[:, 1:256]
    start = interior.argmax(axis=1)
    end = 255 - interior[:, ::-1].argmax(axis=1)
    columns = np.arange(255)[None, :]
    expanded = mask.copy()
    expanded[:, 1:256] = ((columns >= start[:, None]) &
                         (columns < end[:, None]) & interior.any(axis=1)[:, None])
    return expanded


def corrected(half, index, degrees, c10):
    # Canonical full spectrum reconstructed from the same stored half-plane.
    full = np.empty((n, n), dtype=np.complex64)
    full[:, :257] = half
    full[:, 257:] = half[(-np.arange(n)) % n, 1:256][:, ::-1].conj()
    kr, kc = rotated(index, degrees)
    minus = np.clip((g["semiangle"] - np.hypot(qr-kr, qc-kc) * g["wavelength"]) / g["sampling"] + 0.5, 0, 1)
    plus = np.clip((g["semiangle"] - np.hypot(qr+kr, qc+kc) * g["wavelength"]) / g["sampling"] + 0.5, 0, 1)
    c12, phi = 42.96961, 0.293384
    xx, xy, yy = c10+c12*np.cos(2*phi), c12*np.sin(2*phi), c10-c12*np.cos(2*phi)
    f = np.pi * g["wavelength"]
    chi = f * (xx*qr**2 + 2*xy*qr*qc + yy*qc**2)
    cross = 2*f*(xx*qr*kr + xy*(qr*kc+qc*kr) + yy*qc*kc)
    gamma = np.exp(1j*cross) * ((minus-plus)*np.cos(chi) - 1j*(minus+plus)*np.sin(chi))
    result = full * gamma.conj() / np.maximum(np.abs(gamma), 1e-8)
    result[0, 0] = g["dc"]
    return result


totals = dict(fixed_values=0, universal_values=0, fixed_interval_values=0, universal_interval_values=0)
for index in range(len(g["pixels"])):
    for name, universal in (("fixed", False), ("universal", True)):
        mask = support(index, universal=universal)
        totals[name+"_values"] += int(mask.sum())
        totals[name+"_interval_values"] += int(interval_mask(mask).sum())
    if index % 2000 == 0:
        print(f"Support census {index}/{len(g['pixels'])}", file=sys.stderr, flush=True)

baseline = len(g["pixels"]) * n * 257 * 8
mask_metadata = len(g["pixels"]) * (n * 257 // 32) * 8
interval_metadata = len(g["pixels"]) * n * 8
sizes = {"dense_half_bytes": baseline}
for name in ("fixed", "universal"):
    sizes[name+"_bitmask_bytes"] = totals[name+"_values"]*8 + mask_metadata
    sizes[name+"_interval_bytes"] = totals[name+"_interval_values"]*8 + interval_metadata

counts = np.fromfile(root / "counts.u32", dtype=np.uint32).reshape(-1, n, n)
parity = []
payloads, edges, metadata, expected = [], [], [], []
payload_offset = 0
for index, count in zip(g["positions"], counts):
    half = np.fft.fft2(count).astype(np.complex64)[:, :257].copy()
    fixed = support(index)
    universal = support(index, universal=True)
    expanded = interval_mask(fixed)
    for row in range(n):
        columns = np.flatnonzero(expanded[row, 1:256]) + 1
        start = int(columns[0]) if columns.size else 1
        length = int(columns.size)
        metadata.append((payload_offset, start | (length << 16)))
        payloads.append(half[row, start:start+length])
        edges.extend((half[row, 0], half[row, 256]))
        payload_offset += length
    expected.append(np.where(expanded, half, 0).astype(np.complex64))
    stale = int(np.count_nonzero(support(index, angle+20) & ~fixed))
    for name, mask in (("fixed", fixed), ("universal", universal),
                       ("fixed_interval", interval_mask(fixed)),
                       ("universal_interval", interval_mask(universal))):
        packed = half[mask].copy()
        restored = np.zeros_like(half)
        restored[mask] = packed
        assert np.array_equal(restored[mask].view(np.uint64), half[mask].copy().view(np.uint64))
        maximum = 0.0
        for rotation in ([angle] if name.startswith("fixed") else [0, angle, angle+20, 270]):
            for c10 in (0, 55, 155.96977):
                before = corrected(half, index, rotation, c10)
                after = corrected(restored, index, rotation, c10)
                error = float(np.max(np.abs(before-after)))
                maximum = max(maximum, error)
                assert error == 0, (index, name, rotation, c10, error)
                assert np.array_equal(np.angle(np.fft.ifft2(before)), np.angle(np.fft.ifft2(after)))
        parity.append(dict(bf_index=index, layout=name, corrected_spectrum_max_error=maximum,
                           newly_needed_coefficients_after_20_degree_rotation=stale))
result = dict(bf_count=len(g["pixels"]), sizes=sizes, counts=totals,
              parity=parity, scope="all-BF memory census; eight real full-scan BF parity; CPU prototype only")
(root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
np.concatenate(payloads).astype(np.complex64).tofile(root / "payload.c64")
np.asarray(edges, dtype=np.complex64).tofile(root / "edges.c64")
np.asarray(metadata, dtype=np.uint32).tofile(root / "rows.u32")
np.asarray(expected, dtype=np.complex64).tofile(root / "expected.c64")
print(json.dumps(dict(sizes=sizes, parity_cases=len(parity), maximum_error=max(p["corrected_spectrum_max_error"] for p in parity)), indent=2))
