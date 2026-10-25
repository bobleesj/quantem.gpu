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
| Complete CUDA/MPS/native IO scale matrix | `scripts/benchmark_io_scale.py` | Three repetitions per acquisition, separate HDF5/encoding/write/QEM-reopen stages, full verification, and explicit unsupported cases |
| Routine CUDA/MPS/Swift IO regression check | `scripts/benchmark_io_scale.py --load-only` | First observation, five cold and five cached loads per format, frozen references, optional matched-baseline slowdown flag |
| IO benchmark history | `scripts/benchmark_io_history.py` or `--history-dir` | Immutable private reports, source/device/cache provenance, CSV timing history and Markdown index |
| QEM conversion and full source verification | `scripts/benchmark_qem.py` | Separate CUDA/MPS/native conversion stages, exact count coverage, reloads, and disposable output; native output independently verified with MPS |
| Python CUDA/MPS/CPU HDF5 load | `scripts/benchmark_hdf5_load.py` | Run-level p50/p95/max, full-volume hash/dtype/shape parity, logical resident bytes, sampled accelerator/process peak, and release snapshots |
| CUDA/MPS screening | `scripts/benchmark_screening.py` | Separate build and prepared-reopen distributions, product hashes, stages, cache bytes, and memory |
| Python MPS SSB | `scripts/benchmark_ssb_mps_scaling.py` | Reconstruction, parity-pair, and complete phase-variance fit (`objective="phase_variance"`) JSONL |
| Native Swift/Metal indexed load | `metal-4dstem-indexed-load-benchmark` | Exact resident volume/products, stage timing, hashes, and Metal/process memory |
| Native Swift/Metal SSB | `metal-ssb-fit-benchmark`, `metal-ssb-redraw-benchmark` | Joint-model fit stages and fitted parameters; SSB redraw latency |

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_ssb.py scan_master.h5 --voltage-kV 300 --semiangle-mrad 30
```

CUDA compiles kernels on first use and caches them per process tree. Give each
checkout its own empty `CUPY_CACHE_DIR`, or one tree's cached binaries time the
other.

### Routine IO loading check

Use the two selected full 512 × 512 × 192 × 192 acquisitions for routine loading
checks. Acquire the host's GPU/storage locks and inspect competing activity
before running. Set the following paths to private local inputs and evidence:

```bash
PYTHONPATH=src python scripts/benchmark_io_scale.py "$ARINA_ROOT" \
  --backend cuda --load-only --reps 5 \
  --selection-manifest "$IO_REFERENCES" --storage-id "$IO_STORAGE_ID" \
  --output-dir "$IO_RUN" --history-dir "$IO_HISTORY"
```

The private reference manifest extends `selected_acquisitions` with `shape`,
`dtype`, `indexed_product_sha256` and `qem_reference_header`. The last field is
an absolute path or a path relative to the manifest, pointing to an archived
QEM header from the previously qualified acquisition on that backend. The
product is `detector.prepare(data).masked_sum(mask)` with `mask[::2, ::2] = True`;
hash the returned array's raw bytes without casting. Preserve those known-good
references. Do not generate replacement hashes from a failing candidate.

The runner starts each acquisition with a disposable fresh frame index, records
its first HDF5 observation separately, then reuses the index and process memory
pools for repeated loads. Each dataset owner is released. It measures five
pairs of cold/cached HDF5 and QEM loads, synchronizing the GPU before stopping
each timer. Per-file advisory eviction, measured page residency and physical
read counters distinguish storage reads from cached reads. Only source files
for this acquisition and the disposable QEM are targeted; global caches are
not dropped. CUDA context initialization precedes timing; compiler caches are
uncontrolled, so first observations are not cold-process startup measurements.

QEM preparation uses `verify=False, source_hashes=False` and remains outside
all load timings. Each prepared QEM's encoded-body checksum table and layout
must match the archived header exactly. Public QEM loading authenticates its
body as usual. Shape, exposed working dtype and the frozen indexed detector product
are checked after every timed load. This is a regression check, not a rerun of
full original-count verification. All temporary QEM and index files are removed.

To compare against a reviewed prior `loading.json`, append:

```bash
--baseline "$IO_BASELINE" --regression-percent 20
```

The default 20% median threshold is a review trigger, not a statistical test.
Exit status 0 means correctness passed and, if supplied, the comparable
baseline stayed within the threshold. Status 1 means an execution/correctness
or source-stability failure; status 2 means slowdown review or an incomparable
baseline. Device identity, storage, fixtures, timing/allocator policies and
measured cache conditions must match. No baseline is automatically replaced.
Without `--baseline`, the performance status is `no_baseline`; a successful run
does not claim a performance pass. Review contention and rerun flagged cases.

Use `--backend mps` with its own frozen references for Python MPS. For native
Swift/Metal, build `swift build -c release --product metal-runtime-ans-benchmark`
and select `--backend native-metal --native-binary` followed by that executable.
Its manifest also supplies `sample_frames` and `sample_hashes_u32_le` from an
accepted native load: raw diffraction frames at zero, middle and last scan
positions. Native runs compare those hashes, shape and declared file dtype outside
loading time. HDF5 timing includes catalog/index preparation; QEM timing is
native snapshot loading. Each native trial starts a separate process; process
launch is recorded separately. This differs from Python process/allocator reuse.
Native QEM loading uses the same MPS-prepared, frozen-body-matched file, with
MPS preparation excluded. This does not measure native conversion speed.
The native manifest distinguishes `native_hdf5_dtype` from `native_qem_dtype`:
uint16 HDF5 can have a lossless uint8 QEM representation when all values fit.
These must come from the accepted native HDF5 record and archived QEM header,
respectively. This is existing lossless packing, not a numerical precision
change. Missing source pages are read before native warm trials to ensure a
fully cached starting state, including the small master file that a prepared
index may not need. Priming is excluded from loading time; physical-read
counters include catalog preparation.

Use the complete scale runner and full verification for backend/codec
qualification or releases. Hardware IO performance is not automatically
enforced by the ordinary CI test workflow.

### Keeping benchmark history

Keep one private history directory for the IO campaign. `--history-dir` registers
each completed loading report, including failures and review flags. Reports
are copied to content-addressed `records/` files; `index.json` stores their
source revision, physical device, storage, timing boundary, allocator policy,
correctness and performance status. `timings.csv` contains all measured cells,
and `README.md` is a compact index. Re-registering identical bytes is a no-op;
changed reports create new records and never overwrite old evidence. Concurrent
registrations are serialized. Import earlier loading reports with:

```bash
python scripts/benchmark_io_history.py "$OLD_RUN/loading.json" --out "$IO_HISTORY"
```

Choose a known-good baseline explicitly after review, keep its report immutable,
and pass it with `--baseline` on subsequent runs. New reports record the exact
baseline SHA-256 as well as its path. Registration order does not promote a
baseline. Keep each backend/device/storage/protocol combination separate and
retain diagnostic controls with their own allocator/cache definitions.

If another task is editing the checkout, benchmark an isolated frozen source
snapshot and retain its origin manifest and native binary hash. A checkout
change invalidates the run even when its timings and numerical checks pass;
do not silently waive that guard or replace the accepted baseline.

Run the small two-acquisition check after changes to loading, allocation,
decoding, encoding or storage scheduling. Recheck flagged medians under matched
quiet conditions before calling them regressions. Run complete qualification
before codec changes or releases. Publish only a reviewed dated summary in the
[results index](../performance/results.md); raw history stays in the private
archive. This is a manual workflow with automatic evidence registration, not a
scheduled job or an automatic hardware CI gate.

### Complete IO scale and conversion checks

For a selected acquisition suite, pass `--selection-manifest PRIVATE_SELECTION.json`
to `benchmark_io_scale.py`. Its `selected_acquisitions` entries contain `id`,
`filename`, and `source_master_sha256`; the runner checks each master before
starting and preserves those IDs. Add `--fresh-index` to give each Python
acquisition an empty frame-index cache for its first HDF5 observation. Use a
new output directory for a new measurement; existing reports are retained.
The current private training manifest selects five acquisitions with 192 × 192
detectors and excludes both 96 × 96 inputs from default/full-scale runs. Use
that current manifest for new runs; archived selections describe past evidence.
The QEM report's `verified_conversion_seconds` includes the complete producer
workflow and exhaustive verification before dedicated reload trials. Native
conversion includes native product checks followed by full MPS verification.
The QEM benchmark deliberately enables source hashing and full verification by
default, even though routine Python/CLI conversion defaults both to false.
To time routine CUDA or MPS conversion including saving, file-data fsync and
publication, pass `--no-verify --no-source-hashes --reloads 0`; its total is
`conversion_seconds`, without a `verified_conversion_seconds` claim. These
optional-check flags do not apply to the native runner. See the
[IO benchmark index](../performance/results.md#hdf5-and-qem-diskgpu-loading-2026-10-23)
for current results and the separate profiling handoff.
Each integer QEM report also accounts for payload, stream offsets, models,
spatial indexes, padding and envelope bytes. Subtracting index bytes is a size
analysis, not a supported index-free file or a measured index-free load.
The private report retains source-file sizes and SHA-256 identities when the
producer records them, allowing separately hosted input copies to be checked
before their timings are compared. An empty list means that producer did not
record this manifest; it is not evidence of matching source files.
The HDF5, QEM and scale runners reject Python modules imported from a different
checkout and record loaded module hashes outside the timing boundary. Run with
`PYTHONPATH=src` in the intended checkout; restart after changing source trees.
A declared source fingerprint alone does not establish which code was imported.
Use `scripts/benchmark_qem.py --profile-reload` with CUDA or MPS to collect one
additional instrumented load after the ordinary reload trials. It records
selected host calls, parallel worker read/hash intervals and available
process IO counters. Worker times overlap; their sum is not wall time, and
host call durations are not device-event timings.
Ordinary Python reload trials also record process IO before and after the
timed load: Linux uses `/proc/self/io`, and macOS uses `proc_pid_rusage`.
Use the physical-read delta to distinguish storage traffic from cached reads;
an initial cache-residency snapshot alone does not establish the load's cache
state. These counters describe the measured process, not an independently
launched native child. Native runners must report their own counters.
Add `--cold-reloads 3` to either QEM runner to request three additional reloads
after per-file eviction of its disposable QEM output. These observations live
in `cold_reload_runs`; ordinary reloads remain separate. Linux uses
`posix_fadvise(POSIX_FADV_DONTNEED)` and macOS uses a read-only shared mapping
with `msync(MS_SYNC | MS_INVALIDATE)`. Neither flushes the global cache.
The eviction request and residency probe occur outside load timing. Eviction
is advisory: call a trial cold only when its pre-load residency and measured
physical reads support that label. This option does not control HDF5 cache state.

For integer CUDA QEM scheduling changes, also run
`QEM_TEST_BACKEND=cuda python -m pytest -q tests/hardware/test_qem_cuda_loading.py`
on physical CUDA. It covers reordered authenticated blocks, exact decoded and
indexed detector products, staging reuse, corruption and recovery. Record file
cache residency and physical bytes read when interpreting large-file load
timings; a repeated load is not necessarily a cache hit. See the
[controlled loader comparison](../performance/io/reports/2026-10-23-cuda-qem-loader-fix.md).

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
Record Fourier preparation separately from the fit and from warm reconstruction.
A finite cache budget is a different resource-policy row, not a replacement for
the complete-cache timing.

## SSB parity gates

```bash
bash scripts/check_ssb_parity.sh
```

`check_ssb_parity.sh` checks that the native `SSBAgreementSearch` draws the
trials of the Python resident search (`scripts/ssb_agreement_gate.py`). It does
not run a CPU reconstruction oracle or recapture a baseline. Native
reconstruction, fit and saved-result checks are XCTest suites
(`MetalSSBKernelsTests`). Set `QUANTEM_SSB_PARITY_SOURCE` to a local acquisition
when exporting a new oracle fixture and `QUANTEM_SSB_PARITY_RUNS` to a
disposable working directory.

The native engine's non-model reconstruction and phase-variance search are
retired, with their harnesses (`check_metal_ssb.sh`, the frozen phase-variance
fit trajectory and its losses).

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

Follow [Cross-backend parity](../performance/parity/index.md). Re-run each unexpected
hardware mismatch twice on the same source/environment before deciding whether
it is deterministic drift or nondeterminism.

Never edit a frozen golden, expected hash, reference value, or tolerance merely
to make a new implementation pass.

A pull-request smoke test never becomes a physical-device performance claim.
