"""Time the CUDA SSB, DPC and parallax hot paths of this checkout on real acquisitions.

A cleanup or refactor must neither slow these paths nor change a value. This prints, per acquisition and stage,
the first (cold, includes kernel compilation) and the median warm wall time, and the sha256 of every output array,
so two checkouts compared back to back show both. Each fit starts from a fresh session with a fixed seed.

Stages: ``session`` (SSB construction, bright-field crop and Fourier preparation), ``reconstruct`` (fixed probe,
without and with the loss), ``fit`` (``find_aberrations(search="primary", tilt=False)``, the standard model fit:
200 trials and the exact-derivative Newton), ``tilt`` (``find_aberrations()``, the default tilt-corrected fit
through fifth order), ``dpc`` (``dpc.run``) and ``parallax`` (``parallax.run`` with the aberration fit).

Compiled kernels are cached per process tree: give every checkout its own empty ``CUPY_CACHE_DIR`` (the script
makes a temporary one when it is unset), or one tree's cached binaries time the other.

Examples
--------
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_ssb.py scan_master.h5 --voltage-kV 300 --semiangle-mrad 30
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_ssb.py a_master.h5 b_master.h5 --stages session,reconstruct
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("session", "reconstruct", "fit", "tilt", "dpc", "parallax")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("masters", nargs="+", help="Acquisitions io.load reads (e.g. Arina *_master.h5).")
    parser.add_argument("--voltage-kV", type=float, default=300.0)
    parser.add_argument("--semiangle-mrad", type=float, default=30.0)
    parser.add_argument("--scan-sampling-A", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=5, help="Warm repeats of the fast stages.")
    parser.add_argument("--fit-repeats", type=int, default=3, help="Repeats of each fit (each is cold-session).")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", default=",".join(STAGES), help=f"Comma-separated subset of {', '.join(STAGES)}.")
    arguments = parser.parse_args()
    stages = arguments.stages.split(",")
    unknown = set(stages) - set(STAGES)
    if unknown:
        parser.error(f"unknown stages {sorted(unknown)}; choose from {', '.join(STAGES)}")
    with tempfile.TemporaryDirectory(prefix="benchmark-ssb-cupy-") as cache:
        os.environ.setdefault("CUPY_CACHE_DIR", cache)
        run(arguments, stages)


def run(arguments, stages) -> None:
    """Import this checkout (not an installed wheel) and time every stage of every acquisition."""
    sys.path.insert(0, str(ROOT / "src"))
    import quantem

    quantem.__path__ = [str(ROOT / "src/quantem"), *quantem.__path__]
    import cupy as cp
    import numpy as np
    import torch

    from quantem.gpu import SSB, dpc, io, parallax
    from quantem.gpu.device import release_cached_memory

    if not Path(quantem.gpu.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError(f"Imported {quantem.gpu.__file__}, not this checkout; refusing to time stale code.")
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print(f"checkout {ROOT}\nGPU {device['name'].decode()}, CuPy {cp.__version__}, torch {torch.__version__}, "
          f"CUPY_CACHE_DIR {os.environ['CUPY_CACHE_DIR']}")

    def synchronize():
        cp.cuda.Device().synchronize()
        torch.cuda.synchronize()

    def timed(function):
        synchronize()
        started = time.perf_counter()
        output = function()
        synchronize()
        return output, time.perf_counter() - started

    def digest(*values) -> str:
        """sha256 over the exact bytes of arrays and the repr of scalars, so any changed bit shows."""
        hasher = hashlib.sha256()
        for value in values:
            if isinstance(value, cp.ndarray | torch.Tensor):
                value = cp.asnumpy(value) if isinstance(value, cp.ndarray) else value.cpu().numpy()
            if isinstance(value, np.ndarray):
                hasher.update(str((value.dtype, value.shape)).encode())
                hasher.update(np.ascontiguousarray(value).tobytes())
            else:
                hasher.update(repr(value).encode())
        return hasher.hexdigest()[:16]

    def report(stage, times, hashes):
        """One line per stage; repeats whose outputs differ are all listed, since a result must not vary."""
        warm = times[1:] or times
        distinct = list(dict.fromkeys(hashes))
        outputs = distinct[0] if len(distinct) == 1 else f"NOT REPRODUCIBLE, {len(distinct)} distinct: " + " | ".join(distinct)
        print(f"  {stage:<22} first {times[0]:8.3f} s   median warm {statistics.median(warm):8.3f} s"
              f"   (n={len(times)})   {outputs}", flush=True)

    def session(data):
        return SSB(data, voltage_kV=arguments.voltage_kV, semiangle_mrad=arguments.semiangle_mrad,
                   scan_sampling_A=(arguments.scan_sampling_A, arguments.scan_sampling_A))

    def prepared(data):
        ssb = session(data)
        ssb.num_bright_field_pixels  # first read prepares the backend: bright-field crop, spectra and G
        return ssb

    for master in arguments.masters:
        data, seconds = timed(lambda: io.load(master, backend="cuda", verbose=False))
        print(f"{Path(master).name}: {tuple(data.shape)} {data.dtype}, encoded load {seconds:.2f} s")
        if "session" in stages:
            times, hashes = [], []
            for _ in range(arguments.repeats):
                ssb, seconds = timed(lambda: prepared(data))
                times.append(seconds)
                hashes.append(f"num_bf {ssb.num_bright_field_pixels}, detector_sampling_mrad {ssb.detector_sampling_mrad!r}")
                ssb.close()
            report("session", times, hashes)
        if "reconstruct" in stages:
            fixed = {"C10_nm": 1.47, "C12_nm": 0.5, "phi12_deg": math.degrees(0.3)}
            with prepared(data) as ssb:
                for compute_loss in (False, True):
                    times, hashes = [], []
                    for _ in range(arguments.repeats):
                        result, seconds = timed(lambda: ssb.reconstruct(aberrations=fixed, compute_loss=compute_loss))
                        times.append(seconds)
                        hashes.append(f"phase {digest(result.phase)} loss {result.loss!r}")
                    report(f"reconstruct loss={compute_loss}", times, hashes)
        for stage, options in (("fit", {"search": "primary", "tilt": False}), ("tilt", {})):
            if stage not in stages:
                continue
            times, hashes = [], []
            for _ in range(arguments.fit_repeats):
                with prepared(data) as ssb:
                    result, seconds = timed(lambda: ssb.find_aberrations(
                        **options, trials=arguments.trials, seed=arguments.seed, show=False, verbose=False))
                times.append(seconds)
                aberrations = {key: float(value) for key, value in result.aberrations.items()}
                hashes.append(f"phase {digest(result.phase)} loss {result.loss!r} aberrations "
                              f"{digest(aberrations, result.sample_tilt_mrad, result.depth_spread_nm, result.scan_detector_rotation_deg)}"
                              f" C10 {aberrations['C10_nm']:.6f} nm")
                release_cached_memory()
            report(stage, times, hashes)
        if "dpc" in stages:
            times, hashes = [], []
            for _ in range(arguments.repeats):
                result, seconds = timed(lambda: dpc.run(data))
                times.append(seconds)
                hashes.append(f"phase {digest(result.phase)} com {digest(result.com_row, result.com_col)} "
                              f"rotation {result.scan_detector_rotation_mod180_deg!r}")
            report("dpc", times, hashes)
        if "parallax" in stages:
            times, hashes = [], []
            for _ in range(arguments.repeats):
                result, seconds = timed(lambda: parallax.run(
                    data, bright_field_radius_px=20, upsampling_factor=2, fit_aberrations=True,
                    scan_sampling_A=(arguments.scan_sampling_A, arguments.scan_sampling_A), semiangle_mrad=arguments.semiangle_mrad))
                times.append(seconds)
                hashes.append(f"image {digest(result.image)} density {digest(result.density)} "
                              f"shifts {digest(np.asarray(result.shifts))} aberrations {result.aberrations} "
                              f"rotation {result.scan_detector_rotation_deg!r}")
            report("parallax", times, hashes)
        data.close()
        release_cached_memory()


if __name__ == "__main__":
    main()
