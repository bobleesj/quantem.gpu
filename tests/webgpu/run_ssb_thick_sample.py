"""Cross-check the WebGPU thick-sample SSB correction (CPU mirror in ssb-thick-sample.ts) against the CUDA kernel.

Evaluates ``_thick_correct_kernel`` on random (q, k) points with CuPy, writes a JSON fixture, and runs the TS
reference with Node on the same points. Needs CuPy + a CUDA device and Node >= 23 (TypeScript type stripping).

    CUDA_VISIBLE_DEVICES=1 python tests/webgpu/run_ssb_thick_sample.py
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cupy as cp
import numpy as np

from quantem.gpu.ssb.backends.cuda.engine import _thick_correct_kernel


def run(thickness_A: float, tilt_mrad: tuple[float, float], num_points: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    wavelength = 0.019687  # 300 kV, Angstrom
    semiangle = 0.030
    params = dict(
        wavelength=wavelength, semiangle_rad=semiangle, ang_y_rad=0.5554e-3, ang_x_rad=0.5554e-3,
        C10=-26.0, C12=42.0, cos2phi12=math.cos(2 * math.radians(-63)), sin2phi12=math.sin(2 * math.radians(-63)),
        factor=math.pi / wavelength, thickness=thickness_A, tilt_row_rad=tilt_mrad[0] * 1e-3, tilt_col_rad=tilt_mrad[1] * 1e-3,
    )
    k_max = semiangle / wavelength
    radius = k_max * np.sqrt(rng.random(num_points)); angle = rng.random(num_points) * 2 * np.pi
    kx = (radius * np.cos(angle)).astype(np.float32); ky = (radius * np.sin(angle)).astype(np.float32)
    qx = rng.uniform(-2.0, 2.0, num_points).astype(np.float32); qy = rng.uniform(-2.0, 2.0, num_points).astype(np.float32)
    G = (rng.standard_normal(num_points) + 1j * rng.standard_normal(num_points)).astype(np.complex64)
    f32 = lambda v: cp.float32(v)
    cuda = _thick_correct_kernel(
        cp.asarray(G), cp.asarray(qx), cp.asarray(qy), cp.asarray(kx), cp.asarray(ky),
        f32(wavelength), f32(semiangle), f32(params["ang_y_rad"]), f32(params["ang_x_rad"]),
        f32(params["C10"]), f32(params["C12"]), f32(params["cos2phi12"]), f32(params["sin2phi12"]), f32(params["factor"]),
        f32(thickness_A), f32(params["tilt_row_rad"]), f32(params["tilt_col_rad"]),
    ).get()
    # float32 parameters as the GPU sees them, so both sides start from identical inputs
    params = {key: float(np.float32(value)) for key, value in params.items()}
    fixture = dict(
        params=params, G=[[float(v.real), float(v.imag)] for v in G],
        qx=qx.astype(float).tolist(), qy=qy.astype(float).tolist(), kx=kx.astype(float).tolist(), ky=ky.astype(float).tolist(),
        cuda=[[float(v.real), float(v.imag)] for v in cuda],
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(fixture, handle)
    script = Path(__file__).with_name("ssb-thick-sample.ts")
    out = subprocess.run(["node", str(script), handle.name], capture_output=True, text=True)
    Path(handle.name).unlink()
    print(out.stdout.strip() or out.stderr.strip())
    return json.loads(out.stdout) if out.stdout.strip() else {"ok": False}


if __name__ == "__main__":
    results = [run(0.0, (0.0, 0.0)), run(220.0, (-10.3, 4.7)), run(600.0, (20.0, -15.0), seed=1)]
    sys.exit(0 if all(r.get("ok") for r in results) else 1)
