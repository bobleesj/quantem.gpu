# Testing and evidence

Run the smallest gate that can disprove the change first, then widen coverage.

## Portable Python

```bash
python -m pip install -e ".[dev,movie]"
npm ci
PYTHONPATH=src python -m pytest -q          # interactive tier, about a minute on one GPU
PYTHONPATH=src python -m pytest -q -m ""    # everything, including the slow SSB fits (about 12 minutes)
```

Plain `pytest` skips tests marked `slow` (SSB fits and long hardware checks,
each several seconds or more). Mark a new test `@pytest.mark.slow` when it
takes more than about 3 s. CI and every release run the full set.

`npm ci` installs the pinned JavaScript development tools from
`package-lock.json` (esbuild, TypeScript, jsfive, WebGPU types) into
`node_modules/`. The WebGPU contract tests bundle the TypeScript sources with
them and run the bundles in Node.js 22 or newer; they are skipped only when
Node.js is not installed.

Portable CI proves imports, public contracts, CPU references, source packaging,
and tests that do not require a physical accelerator. Backend skips are
reported; they are not parity evidence.

## Performance entry points

Each hot path has a repository-owned timing script. Run the before and after
checkouts back to back on the same device, data and calibration, and compare
both the timings and the output hashes the script prints.

| Operation | Entry point | Primary output |
|---|---|---|
| CUDA SSB, DPC and parallax | `scripts/benchmark_ssb.py` | First and median warm wall time per stage (session, reconstruct, fit, tilt fit, DPC, parallax) and the SHA-256 of every output array |
| Python CUDA/MPS/CPU HDF5 load | `scripts/benchmark_hdf5_load.py` | Run-level p50/p95/max, full-volume hash/dtype/shape parity, logical resident bytes, sampled accelerator/process peak, and release snapshots |
| CUDA/MPS screening | `scripts/benchmark_screening.py` | Separate build and prepared-reopen distributions, product hashes, stages, cache bytes, and memory |
| Python MPS SSB | `scripts/benchmark_ssb_mps_scaling.py` | Reconstruction, parity-pair, and complete phase-variance fit (`objective="phase_variance"`) JSONL |
| Native Swift/Metal indexed load | `metal-4dstem-indexed-load-benchmark` | Exact resident volume/products, stage timing, hashes, and Metal/process memory |
| Native Swift/Metal SSB | `metal-ssb-benchmark` | Reconstruction, phase/loss, calibration, and cache-policy report |

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_ssb.py scan_master.h5 --voltage-kV 300 --semiangle-mrad 30
```

CUDA compiles kernels on first use and caches them per process tree. Give each
checkout its own empty `CUPY_CACHE_DIR`, or one tree's cached binaries time the
other.

Raw logs and run folders stay in the private evidence archive, never in this
repository. The dated summary (question, setup, numbers, conclusion) goes under
`docs/` with the device, date and exact revision of every number.

## Native Swift and Metal

```bash
xcrun swift-format lint --strict --recursive \
  Package.swift \
  native/swift/Sources \
  native/swift/Tests \
  native/swift/Benchmarks
swift test
swift test -c release --filter MetalSSBKernelsTests
```

Opt-in real-source tests retain their fixture hashes and environment variables.
A simulator or compile-only result does not replace physical Metal execution.
The package-owned SSB benchmark accepts an exact metadata file, plane-major BF
source, independent phase reference, repetition count, cache policy, fit-trial
count, and fit-repetition count:

```bash
swift run -c release metal-ssb-benchmark \
  METADATA_JSON FULL_BF_U8 REFERENCE_PHASE_F32 7 full 200 3
```

Record preparation separately from warm reconstruction, exact loss, and the
complete fit. A finite cache budget is a different resource-policy row, not a
replacement for the complete-cache timing.

## SSB parity gates

```bash
bash scripts/check_metal_ssb.sh
bash scripts/check_ssb_parity.sh --build-metal
```

`check_ssb_parity.sh` checks the frozen GPU fitting trajectory, repeats it for
determinism and reports the MPS comparison. It does not run a CPU
reconstruction oracle or recapture a baseline; add `--full` only when a
512×512 fit is relevant. Reconstruction and saved-result checks are in
`check_metal_ssb.sh`. Set `QUANTEM_SSB_PARITY_SOURCE` to a local acquisition
when exporting a new fixture and `QUANTEM_SSB_PARITY_RUNS` to a disposable
working directory.

The frozen losses `0.14511984586715698`, `0.13808111846446991` and
`0.13864889740943909` never change to make a test pass. They are not a
sufficient gate for accumulation-order changes: `planesPerRange=4` diverges by
one float32 ULP at fixture index 8 while those three values still pass. Widen
the gate to the recorded 16-fixture set before accepting an accumulation
reordering.

Routine optimization compares a known-good GPU revision with its candidate on
identical inputs, calibration, precision, backend and hardware, and freezes the
object, phase, loss and fitting trajectory of the changed path. A bit-exact
claim needs a bit-exact comparison; cross-backend tolerance checks are
separate. The full-size CPU oracle runs only with `--cpu-oracle` and, under
pytest, `QUANTEM_SSB_CPU_ORACLE=1`.

## Documentation

```bash
python -m pip install -r docs/requirements.txt
python scripts/check_docs_links.py
python -m pytest -q tests/infrastructure/test_docs_site.py
jupyter-book build docs
python scripts/check_docs_links.py --html-root docs/_build/html
python scripts/check_docs_nav_toggle.py docs/_build/html
```

The documentation build is hardware-independent and never downloads private
data. Measured numbers are written into the pages as dated records.

## Backend parity

Follow [Cross-backend parity](../performance/parity.md). Re-run each unexpected
hardware mismatch twice on the same source/environment before deciding whether
it is deterministic drift or nondeterminism.

Never edit a frozen golden, expected hash, reference value, or tolerance merely
to make a new implementation pass.

A pull-request smoke test never becomes a physical-device performance claim.
