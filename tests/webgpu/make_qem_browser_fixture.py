"""Generate small disposable QEM acquisitions for the browser GPU parity gate."""

import json
import math
from pathlib import Path
import sys

import h5py
import hdf5plugin
import numpy as np

from quantem.gpu.formats.qem.reference import save_array


def generate(directory: Path) -> None:
    """Cover every integer stream mode, checkpoints, chunks, and a tail."""
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for dtype in (np.uint8, np.uint16):
        rng = np.random.default_rng(43)
        data = np.zeros((17, 33, 2, 3), dtype=dtype)
        columns = data.reshape(-1, 6)
        high = np.iinfo(dtype).max
        columns[:, 1] = high
        columns[:, 2] = rng.integers(0, int(high) + 1, len(columns), dtype=dtype)
        columns[:, 3] = rng.poisson(3, len(columns)).astype(dtype)
        columns[[0, 255, 256, 511, 512, 560], 4] = 77
        columns[:, 5] = rng.poisson(2, len(columns)).astype(dtype)
        columns[::43, 5] = high
        name = np.dtype(dtype).name + "-qem-modes"
        save_array(directory / (name + ".qem"), data, chunk_scans=512)
        data.tofile(directory / (name + ".bin"))
        records.append(dict(name=name, shape=list(data.shape), dtype=np.dtype(dtype).name, block_frames=512))
        # Two blocks in one chunk: once staged on the GPU, the second block starts
        # mid-word (byte 1425 of the uint8 payload).
        blocks = np.dtype(dtype).name + "-qem-blocks"
        save_array(directory / (blocks + ".qem"), data, chunk_scans=1024)
        data.tofile(directory / (blocks + ".bin"))
        records.append(dict(name=blocks, shape=list(data.shape), dtype=np.dtype(dtype).name, block_frames=512))
    zeros = np.zeros((17, 33, 2, 3), dtype=np.uint16)
    save_array(directory / "zero.qem", zeros, chunk_scans=512)
    zeros.tofile(directory / "zero.bin")
    records.append(dict(name="zero", shape=list(zeros.shape), dtype="uint16", block_frames=512))
    save_array(directory / "float.qem", np.ones((2, 3, 2, 3), dtype=np.float32))
    second = data.copy()
    second[..., 0, 0] = 3
    save_array(directory / "uint16-series-second.qem", second, chunk_scans=512)
    second.tofile(directory / "uint16-series-second.bin")
    save_array(directory / "shape-mismatch.qem", second[:-1], chunk_scans=512)
    valid = np.ones((2, 3), bool)
    valid[0, 1] = False
    save_array(directory / "validity-mismatch.qem", second, metadata={"valid_pixels": valid}, chunk_scans=512)
    save_array(
        directory / "saturated.qem",
        np.full((1, 1, 256, 256), 65535, dtype=np.uint16),
    )
    # rans-products.ts: two acquisitions of a 32 x 32 scan on a 3 x 4 detector whose last
    # pixel is masked and saturated, a 131072-position saturated scan (ROI sums above 2^32)
    # and a saturated 512-column detector (weighted moments above 2^32).
    scan = np.arange(1024)[:, None]
    pixel = np.arange(12)[None, :]
    valid = np.ones((3, 4), bool)
    valid[2, 3] = False
    for tilt in (0, 1):
        values = np.where(scan % 17 == 0, 0, (scan * (pixel + 1) + tilt * 7 + pixel) % 31)
        values[:, 0] = ((scan[:, 0] + tilt) % 2) * 7
        values[:, 11] = 65535
        products = values.astype(np.uint16).reshape(32, 32, 3, 4)
        save_array(directory / f"products-{tilt}.qem", products, metadata={"valid_pixels": valid}, chunk_scans=512)
        products.tofile(directory / f"products-{tilt}.bin")
    save_array(directory / "saturated-roi.qem", np.full((256, 512, 2, 2), 65535, dtype=np.uint16))
    save_array(directory / "saturated-columns.qem", np.full((16, 16, 1, 512), 65535, dtype=np.uint16))
    (directory / "cases.json").write_text(json.dumps(records))


def generate_h5(directory: Path) -> None:
    """Write Arina-layout bitshuffle-LZ4 acquisitions whose counts span all 16 bit planes.

    Each case is a master file with a pixel mask and two data files of one-frame chunks, plus
    the raw values as ``<name>.bin`` for exact host sums. Counts above 255, the dtype maximum
    on every pixel of one frame, and a masked hot pixel make a decoder that drops high bit
    planes, clips, or keeps the masked pixel fail. uint32 and float32 cover the 32-plane
    decoders. 96 x 96 frames are not a whole number of bitshuffle blocks, so each ends in a
    shorter block with its own plane stride.
    """
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, dtype, detector, frames in (
        ("h5-u16-64", np.uint16, (64, 64), 300),
        ("h5-u16-128", np.uint16, (128, 128), 120),
        ("h5-u8-64", np.uint8, (64, 64), 90),
        ("h5-u32-64", np.uint32, (64, 64), 60),
        ("h5-f32-64", np.float32, (64, 64), 60),
        ("h5-u16-96", np.uint16, (96, 96), 60),
        ("h5-u8-96", np.uint8, (96, 96), 60),
        ("h5-u32-96", np.uint32, (96, 96), 60),
        ("h5-f32-96", np.float32, (96, 96), 60),
    ):
        rng = np.random.default_rng(7)
        high = np.finfo(dtype).max if dtype == np.float32 else np.iinfo(dtype).max
        counts = rng.poisson(3, (frames, *detector)).astype(dtype)
        if dtype == np.float32:
            counts += rng.normal(0, 1, counts.shape).astype(dtype)
        counts[0] = high
        counts[1:, detector[0] // 2, detector[1] // 2] = rng.integers(0, min(int(high), 2**31 - 1) + 1, frames - 1)
        counts[1::7, 3, 4] = min(300, high)
        counts[2::11, -2, -3] = high
        bad = np.zeros(detector, np.uint32)
        bad[1, 2] = 1
        counts[:, 1, 2] = high
        with h5py.File(directory / f"{name}_master.h5", "w") as master:
            specific = master.create_group("entry/instrument/detector/detectorSpecific")
            specific["pixel_mask"] = bad
            specific["ntrigger"] = np.array([frames], np.uint32)
        split = frames // 2 + 1
        for index, part in enumerate((counts[:split], counts[split:]), start=1):
            with h5py.File(directory / f"{name}_data_{index:06d}.h5", "w") as data:
                data.create_dataset(
                    "entry/data/data", data=part, chunks=(1, *detector),
                    **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
                )
        counts.tofile(directory / f"{name}.bin")
        records.append(dict(name=name, dtype=np.dtype(dtype).name, frames=frames,
                            detector=list(detector), bad_pixels=[int(np.flatnonzero(bad)[0])]))
    (directory / "h5-cases.json").write_text(json.dumps(records))


def generate_ssb(directory: Path) -> None:
    """Write identical raw and QEM bright-field columns for browser SSB."""
    directory.mkdir(parents=True, exist_ok=True)
    size, radius = 128, 3
    row, col = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = row**2 + col**2 <= radius**2
    wavelength, angular_sampling = 0.019687, 0.0075
    rows = row[disk] * angular_sampling / wavelength
    cols = col[disk] * angular_sampling / wavelength
    counts = np.random.default_rng(47).poisson(5, (size, size, 1, len(rows))).astype(np.uint16)
    save_array(directory / "ssb columns.qem", counts, chunk_scans=1024)
    counts.reshape(size * size, -1).T.copy().tofile(directory / "ssb-columns.u16")
    # An off-centre crop on a rectangular full detector. A flagged BF pixel
    # remains measured evidence: source admission must not mask or replace it.
    detector_rows, detector_cols = row[disk] + radius, col[disk] + radius
    full_counts = np.random.default_rng(48).poisson(2, (size, size, 12, 13)).astype(np.uint16)
    full_counts[..., detector_rows + 1, detector_cols + 2] = counts[..., 0, :]
    valid = np.ones((12, 13), bool)
    valid[detector_rows[0] + 1, detector_cols[0] + 2] = False
    save_array(directory / "ssb-full.qem", full_counts,
               metadata={"valid_pixels": valid}, chunk_scans=1024)
    calibration = {
        "schema": "quantem.ssb-browser-calibration/3",
        "num_bright_field_pixels": len(rows), "g_shape": [len(rows), size, size],
        "wavelength_A": wavelength, "semiangle_mrad": (radius + 0.25) * angular_sampling * 1e3,
        "detector_sampling_mrad": [angular_sampling * 1e3] * 2,
        "dc_value": [float(counts.sum(dtype=np.uint64)) / len(rows), 0],
        "detector_shape": [7, 9],
        "bright_field_rows": detector_rows.tolist(), "bright_field_cols": detector_cols.tolist(),
        "bright_field_k_row": rows.tolist(), "bright_field_k_col": cols.tolist(),
        "scan_q_row": np.fft.fftfreq(size).tolist(), "scan_q_col": np.fft.fftfreq(size).tolist(),
        "aperture_k": [1] * len(rows), "alpha_k2": [0] * len(rows),
        "cos2phi_k": [1] * len(rows), "sin2phi_k": [0] * len(rows),
        "scan_detector_rotation_deg": 0,
        "precision": {"real_dtype": "float32", "complex_dtype": "complex64"},
    }
    (directory / "ssb-calibration.json").write_text(json.dumps(calibration))


# Krivanek (n, m) and the probe of the higher-order sign fixture: magnitudes in A, azimuths in degrees
HIGHER_ORDERS = {"C10": (1, 0), "C12": (1, 2), "C21": (2, 1), "C23": (2, 3), "C30": (3, 0)}
HIGHER_ORDER_TRUTH = {"C10": (-50.0, 0.0), "C12": (20.0, 30.0), "C21": (400.0, 60.0), "C23": (300.0, -20.0),
                      "C30": (3.0e4, 0.0)}
HIGHER_ORDER_TOLERANCE_RAD = 1e-5


def generate_ssb_higher_order(directory: Path) -> None:
    """Write a weak-phase SSB folder with coma, trefoil and C30, and its float64 reconstructions.

    An exact thin-object simulation: a periodic random phase (0.05 rad rms, |q| < 2.5 1/A) on a 6 A cell of
    64 x 64 pixels, the probe p(r - r0) at every pixel, and the far-field grid itself as the detector (3.3 mrad
    per pixel, 293 bright-field pixels). Two periods make the 128 x 128 scan of uint16 counts.

    The reference reconstructs the same counts in float64, written from the physics with chi in polar form:
    for a thin object G(q, k) ~ P(k - q) conj P(k) - P(k) conj P(k + q). The kernels once evaluated the first
    term at q - k, which reverses coma and trefoil on that sideband only; that form misses the reference here
    by about 0.03 rad. ssb-higher-order-sign.ts holds the browser engine to the k - q reference.
    """
    directory.mkdir(parents=True, exist_ok=True)
    volts = 300e3
    wavelength = 12.2643 / math.sqrt(volts * (1 + 0.978476e-6 * volts))
    semiangle, grid, cell, scan = 0.030, 64, 6.0, 128
    sampling = wavelength / cell                                    # rad per detector pixel

    def probe(kr, kc, terms):
        alpha = wavelength * np.hypot(kr, kc)
        azimuth = np.arctan2(kc, kr)
        chi = sum(magnitude * alpha ** (n + 1) / (n + 1) * np.cos(m * (azimuth - math.radians(degrees)))
                  for name, (magnitude, degrees) in terms.items() for n, m in [HIGHER_ORDERS[name]])
        # the engines' soft aperture, one detector pixel wide
        aperture = np.clip((semiangle - alpha) / sampling + 0.5, 0.0, 1.0)
        return aperture * np.exp(-2j * np.pi / wavelength * chi)

    k = np.fft.fftfreq(grid, cell / grid)
    kr, kc = np.meshgrid(k, k, indexing="ij")
    rng = np.random.default_rng(3)
    spectrum = (rng.standard_normal((grid, grid)) + 1j * rng.standard_normal((grid, grid))) * (np.hypot(kr, kc) < 2.5)
    spectrum[0, 0] = 0
    phase = np.fft.ifft2(spectrum).real
    phase *= 0.05 / phase.std()
    wave = np.fft.ifft2(probe(kr, kc, HIGHER_ORDER_TRUTH))
    bright = np.abs(probe(kr, kc, {})) > 0
    intensity = np.empty((grid, grid, int(bright.sum())))
    for row in range(grid):
        exit_waves = np.stack([np.roll(wave, (row, col), axis=(0, 1)) for col in range(grid)]) * np.exp(1j * phase)
        intensity[row] = (np.abs(np.fft.fft2(exit_waves)) ** 2)[:, bright]
    counts = np.tile(np.round(intensity * (60000 / intensity.max())).transpose(2, 0, 1), (1, 2, 2))
    counts = counts.astype(np.uint16)
    counts.tofile(directory / "bf_columns.u16")
    object_phase = np.tile(phase, (2, 2))

    bf = np.stack([kr[bright], kc[bright]], axis=1)
    k2 = (bf ** 2).sum(1)
    safe = np.where(k2 > 0, k2, 1.0)
    g = np.fft.fft2(counts.astype(np.float64), axes=(1, 2))      # e^{-2 pi i q r0} over the scan
    dc = float(g[:, 0, 0].real.mean())
    q = np.fft.fftfreq(scan, cell / grid)
    qr, qc = np.meshgrid(q, q, indexing="ij")
    detector_rows, detector_cols = ((index + grid // 2) % grid for index in np.nonzero(bright))
    calibration = {
        "schema": "quantem.ssb-browser-calibration/3",
        "num_bright_field_pixels": len(bf), "g_shape": [len(bf), scan, scan],
        "wavelength_A": wavelength, "semiangle_mrad": semiangle * 1e3, "detector_sampling_mrad": [sampling * 1e3] * 2,
        "dc_value": [dc, 0.0], "detector_shape": [grid, grid],
        "bright_field_rows": detector_rows.tolist(), "bright_field_cols": detector_cols.tolist(),
        "bright_field_k_row": bf[:, 0].tolist(), "bright_field_k_col": bf[:, 1].tolist(),
        "scan_q_row": q.tolist(), "scan_q_col": q.tolist(),
        "aperture_k": np.abs(probe(bf[:, 0], bf[:, 1], {})).tolist(),
        "alpha_k2": (wavelength ** 2 * k2).tolist(),
        "cos2phi_k": np.where(k2 > 0, (bf[:, 0] ** 2 - bf[:, 1] ** 2) / safe, 1.0).tolist(),
        "sin2phi_k": (2 * bf[:, 0] * bf[:, 1] / safe).tolist(), "scan_detector_rotation_deg": 0,
        "precision": {"real_dtype": "float32", "complex_dtype": "complex64"},
    }
    (directory / "cal.json").write_text(json.dumps(calibration))
    (directory / "source.json").write_text(json.dumps({
        "kind": "bf_columns", "url": "bf_columns.u16", "dtype": "uint16", "encoding": "uint16",
        "num_bright_field_pixels": len(bf), "plane": scan * scan, "scan_shape": [scan, scan],
        "bytes_per_bright_field_pixel": scan * scan * 2, "bits_per_value": 16,
    }))

    def reconstruct(terms, sign):
        """Both estimators: phase_of_mean = angle(ifft2(mean_k C_k)), mean_phase = mean_k angle(ifft2(C_k))."""
        total = np.zeros((scan, scan), complex)
        phases = np.zeros((scan, scan))
        for start in range(0, len(bf), 128):
            br, bc = (bf[start:start + 128, axis][:, None, None] for axis in (0, 1))
            first = probe(br - qr, bc - qc, terms) if sign == "k-q" else probe(qr - br, qc - bc, terms)
            centre = probe(br, bc, terms)
            gamma = first * np.conj(centre) - centre * np.conj(probe(br + qr, bc + qc, terms))
            size = np.abs(gamma)
            unit = np.where(size ** 2 > 1e-16, np.conj(gamma) / np.where(size > 0, size, 1), 0)
            corrected = g[start:start + 128] * unit
            corrected[:, 0, 0] = dc
            total += corrected.sum(0)
            phases += np.angle(np.fft.ifft2(corrected, axes=(1, 2))).sum(0)
        return {"phase_of_mean": np.angle(np.fft.ifft2(total / len(bf))), "mean_phase": phases / len(bf)}

    def correlation(image):
        a, b = image - image.mean(), object_phase - object_phase.mean()
        return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))

    truth = HIGHER_ORDER_TRUTH
    cases = {"truth": truth, "none": {name: truth[name] for name in ("C10", "C12")},
             "reversed": {**truth, "C21": (truth["C21"][0], truth["C21"][1] + 180.0),
                          "C23": (truth["C23"][0], truth["C23"][1] + 60.0)}}
    images = {(name, "k-q"): reconstruct(terms, "k-q") for name, terms in cases.items()}
    images["truth", "q-k"] = reconstruct(truth, "q-k")
    scores = {f"{name}|{sign}": correlation(image["phase_of_mean"]) for (name, sign), image in images.items()}
    separation = float(np.abs(images["truth", "k-q"]["phase_of_mean"] - images["truth", "q-k"]["phase_of_mean"]).max())
    # The sign must matter on this fixture, and the probe's own coefficients must reconstruct the object best.
    assert separation > 100 * HIGHER_ORDER_TOLERANCE_RAD, separation
    assert scores["truth|k-q"] == max(scores.values()), scores
    order = [[name, estimator, sign] for (name, sign) in images for estimator in ("phase_of_mean", "mean_phase")]
    reference = np.concatenate([images[name, sign][estimator].ravel() for name, estimator, sign in order])
    reference.astype(np.float32).tofile(directory / "reference.f32")

    def options(terms):
        higher = {}
        for name, (magnitude, degrees) in terms.items():
            if name not in ("C10", "C12"):
                higher[f"{name}_A"] = magnitude
                if HIGHER_ORDERS[name][1]:
                    higher[f"phi{name[1:]}_deg"] = degrees
        return {"C10_A": terms["C10"][0], "C12_A": terms["C12"][0], "phi12_deg": terms["C12"][1], "higherOrder": higher}

    (directory / "reference.json").write_text(json.dumps({
        "cases": {name: options(terms) for name, terms in cases.items()}, "order": order,
        "tolerance_rad": HIGHER_ORDER_TOLERANCE_RAD, "sign_separation_rad": separation, "object_correlation": scores,
    }))


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
    generate_h5(Path(sys.argv[1]))
    generate_ssb(Path(sys.argv[1]))
    generate_ssb_higher_order(Path(sys.argv[1]) / "ssb-higher-order")
