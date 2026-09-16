"""Export independent CuPy and production CUDA SSB references for native Metal.

Run on a CUDA host, then copy the output directory to the Metal test host::

    python tests/metal/export_ssb_scan_size_reference.py /tmp/ssb-reference

These are deterministic numerical controls, not real-acquisition benchmarks.
All pixels in the selected bright-field disk are retained at native scan size.
Existing references are never overwritten.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cupy as cp
import numpy as np

from quantem.gpu.ssb.backends.cuda.engine import SSBEngine


def _explicit_objects(engine, c10, c12, angle):
    """Independent full-plane complex-probe equation, using CuPy FFT."""
    cache = engine._cache
    qr = cache["qx_1d"][None, :, None]
    qc = cache["qy_1d"][None, None, :]
    kr = cache["kx_bf"][:, None, None]
    kc = cache["ky_bf"][:, None, None]

    def probe(row, col):
        radius2 = row * row + col * col
        radius = cp.sqrt(radius2)
        inverse = cp.where(radius2 > np.float32(1e-30), 1 / radius2, 0)
        cos2 = (row * row - col * col) * inverse
        sin2 = 2 * row * col * inverse
        denominator = cp.sqrt(
            (row * np.float32(cache["ang_y_rad"])) ** 2
            + (col * np.float32(cache["ang_x_rad"])) ** 2
        )
        denominator *= cp.where(radius > np.float32(1e-15), 1 / radius, 0)
        edge = cp.where(
            denominator > np.float32(1e-15),
            (
                np.float32(cache["semiangle_rad"])
                - radius * np.float32(cache["wavelength"])
            )
            / denominator
            + 0.5,
            1,
        )
        aperture = cp.clip(edge, 0, 1)
        alpha2 = (radius * np.float32(cache["wavelength"])) ** 2
        chi = (
            np.float32(engine._factor)
            * alpha2
            * (
                np.float32(c10)
                + np.float32(c12)
                * (
                    cos2 * np.float32(math.cos(2 * angle))
                    + sin2 * np.float32(math.sin(2 * angle))
                )
            )
        )
        return aperture * (cp.cos(chi) - 1j * cp.sin(chi))

    chi_k = (
        np.float32(engine._factor)
        * cache["alpha_k2_1d"]
        * (
            np.float32(c10)
            + np.float32(c12)
            * (
                cache["cos2phi_k_1d"] * np.float32(math.cos(2 * angle))
                + cache["sin2phi_k_1d"] * np.float32(math.sin(2 * angle))
            )
        )
    )
    pk = (cache["aperture_k_1d"] * (cp.cos(chi_k) - 1j * cp.sin(chi_k)))[:, None, None]
    gamma = probe(qr - kr, qc - kc) * pk.conj() - probe(qr + kr, qc + kc).conj() * pk
    corrected = engine.G_qk * gamma.conj() / cp.maximum(cp.abs(gamma), 1e-8)
    corrected[:, 0, 0] = engine._dc_value_host
    return cp.fft.ifft2(corrected)


def _export(root):
    root.mkdir(parents=True, exist_ok=False)
    rows, cols = np.indices((16, 16))
    # Avoid a rational center exactly perpendicular to many scan frequencies:
    # at C10=C12=0 those gamma values sit on the normalization singularity,
    # where independent float32 aperture roundoff can change their sign.
    center = (7.41327, 7.73113)
    disk = (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= 5.5**2
    rows, cols = rows[disk].astype(np.int32), cols[disk].astype(np.int32)
    cases = [(0.0, 0.0, 0.0, 0.0), (33.2, 17.3, 0.23, 0.0), (-75.0, 9.0, -0.41, 31.0)]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for size in (128, 256, 512):
        target = root / str(size)
        target.mkdir()
        index = np.arange(len(rows) * size * size, dtype=np.uint64)
        counts = ((index * 17 + index // 113 + 3) % 251).astype(np.uint8)
        counts.tofile(target / "counts.uint8")
        stack = cp.asarray(counts.reshape(len(rows), size, size), dtype=cp.float32)
        fourier = cp.fft.fft2(stack)
        q = cp.fft.fftfreq(size, d=0.5).astype(cp.float32)
        qr, qc = cp.meshgrid(q, q, indexing="ij")
        engine = SSBEngine(
            G_qk=fourier,
            bf_inds_row=cp.asarray(rows),
            bf_inds_col=cp.asarray(cols),
            bf_center=center,
            dc_value=complex(fourier[:, 0, 0].mean().get()),
            gpts=(16, 16),
            sampling=(1 / (16 * 0.19),) * 2,
            q_row=qr,
            q_col=qc,
            wavelength=0.0197,
            semiangle_cutoff=20.5865,
            angular_sampling=(3.743, 3.743),
        )
        engine.cache_rotation(0.0)
        cache = engine._cache
        fields = {
            "brightfieldKX": "kx_bf",
            "brightfieldKY": "ky_bf",
            "brightfieldAlphaSquared": "alpha_k2_1d",
            "brightfieldAperture": "aperture_k_1d",
            "brightfieldCos2Phi": "cos2phi_k_1d",
            "brightfieldSin2Phi": "sin2phi_k_1d",
            "qxByRow": "qx_1d",
            "qyByColumn": "qy_1d",
        }
        geometry = {
            key: cp.asnumpy(cache[value]).tolist() for key, value in fields.items()
        }
        geometry.update(
            wavelengthAngstroms=cache["wavelength"],
            semiangleRadians=cache["semiangle_rad"],
            angularSamplingYRadians=cache["ang_y_rad"],
            angularSamplingXRadians=cache["ang_x_rad"],
            dcValue=[engine._dc_value_host.real, engine._dc_value_host.imag],
            referenceRotationDegrees=0,
        )
        (target / "geometry.json").write_text(json.dumps(geometry))
        results = []
        for number, (c10, c12, angle, rotation) in enumerate(cases):
            engine.cache_rotation(math.radians(rotation))
            reference = _explicit_objects(engine, c10, c12, angle)
            obj = reference.mean(axis=0)
            phase = cp.angle(reference)
            loss = float(
                cp.mean(cp.mean(phase * phase, axis=0) - cp.mean(phase, axis=0) ** 2)
            )
            cuda_object = engine.reconstruct_object(c10, c12, angle)
            # Match native Metal's full-IFFT objective, not the legacy sparse
            # row-subsampled CUDA optimizer loss (a different quantity).
            _, cuda_loss = engine.reconstruct_with_loss(c10, c12, angle)
            relative = float(cp.linalg.norm(cuda_object - obj) / cp.linalg.norm(obj))
            print(
                "reference gate",
                size,
                number,
                relative,
                cuda_loss,
                loss,
                "max delta",
                float(cp.max(cp.abs(cuda_object - obj))),
                flush=True,
            )
            if relative >= 1e-4:
                legacy = engine._run_correction_pipeline_chunked(
                    c10, c12, angle, chunk_bf=1
                )
                print(
                    "diagnostic: legacy-vs-equation",
                    float(cp.linalg.norm(legacy - obj) / cp.linalg.norm(obj)),
                    "legacy-vs-fourier-sum",
                    float(cp.linalg.norm(legacy - cuda_object) / cp.linalg.norm(obj)),
                    flush=True,
                )
            np.testing.assert_allclose(cuda_loss, loss, rtol=1e-5, atol=1e-6)
            assert relative < 1e-4, relative
            cp.asnumpy(obj).astype(np.complex64).tofile(
                target / f"object-{number}.complex64"
            )
            results.append(
                dict(
                    c10=c10,
                    c12=c12,
                    phi12=angle,
                    rotation=rotation,
                    loss=loss,
                    cudaLoss=cuda_loss,
                    cudaObjectRelativeL2=relative,
                )
            )
            print(size, number, results[-1], flush=True)
        metadata = dict(
            size=size,
            brightfieldCount=len(rows),
            cases=results,
            cudaRevision=revision,
            cupyVersion=cp.__version__,
            countsSHA256=hashlib.sha256(counts.tobytes()).hexdigest(),
            filesSHA256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in target.iterdir()
            },
        )
        (target / "reference.json").write_text(json.dumps(metadata, indent=2))
        del engine, stack, fourier, reference, cuda_object
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    _export(parser.parse_args().output)
