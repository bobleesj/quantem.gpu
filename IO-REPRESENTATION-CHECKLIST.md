# Count IO and resident detector integration

Owner: QuantEM.GPU count IO integration.
Starting revision: `8ca898e3a1a6af45b08807fbe7b985966098029f`.
Scope: CUDA, Python MPS, and native Swift/Metal.
Final Fourier fields and SSB changes are deferred to the next phase.

Consolidation started: **2026-09-06 18:34:14 UTC**. This checkpoint brings the
reviewed count-IO implementation and canonical names into one `main` entry point.
The original experiment history and raw logs remain preserved privately; public
evidence references retain their scientific results, hashes, and limitations.

Validation recorded: **2026-09-06 18:38:09 UTC**, MacBook Pro M5 Max 128 GB,
macOS 26.4. The complete Python suite passed **939 tests** with **109 skips**;
Swift executed **186 tests**, including **9 skips**, with no failures. Browser
TypeScript compilation, strict recursive Swift formatting, profile/backend
registries, the isolated wheel build, and the strict documentation build passed.
These are package checks, not a new full-volume or application-speed claim.

The first full Python run retained four failures: schema rejection was correct,
but its diagnostic omitted the field name expected by existing browser tests.
The diagnostic now names the schema; no admission rule or numerical tolerance
was weakened. An initial non-isolated wheel build lacked `hatchling`; the
standard isolated build installed that build dependency and passed.

## Added requirements and delivery status

| Requirement | Working checkpoint | Remaining gate |
| --- | --- | --- |
| ANS on disk, packed on GPU | Shared file contract and direct CUDA/MPS ANS-to-packed primitive; physical small-file MPS parity | Native reader/conversion, physical CUDA, incremental file-shard streaming and real-volume timing |
| Dense, packed and ANS conversions | Explicit conversion API; ANS-to-packed preserves exact counts and source ownership | Reverse GPU conversions and ANS-to-dense GPU materialization |
| Save ANS without HDF5 | Transactional explicit CPU reference writer and exact CPU/MPS reopen tests | Accelerated saving from GPU residents and full-data size/throughput/memory qualification |

These are three tracked requirements, not three completed performance claims.
The reference writer is useful for qualification, but is not the final fast
production encoder. The current checkpoint does not satisfy all backend gates.

## Locked scientific and API contract

- `io.save(..., format="arina")` is the canonical Arina acquisition/layout
  selector. Its HDF5 master/data layout and bitshuffle/LZ4 compression are
  unchanged. Format aliases are removed, not deprecated. Compression remains
  a separate option. Other acquisition formats,
  such as EMPAD, must not be advertised until implemented and qualified.
- One `io.load` owner with `dense`, `packed`, and `ans` representation choices.
- `io.save(..., format="quantem", compression="ans")` selects the standalone
  container and its lossless compression separately. `format="ans"` is rejected;
  supported file bytes are unchanged.
  `compression="auto"` preserves the prior effective defaults: bitshuffle/LZ4
  for Arina and ANS for QuantEM. Loading detects decompression from the file;
  there is no `decompression=` selector. The load-time
  `representation` and `FourDSTEMData.to_representation(...)` choose resident
  layout. Do not use one option ambiguously for both disk and resident state.
- Disk encoding is independent of resident representation. An ANS file may be
  loaded as ANS, decoded directly into packed storage, or materialized densely.
- Only `dense`, `packed`, and `ans` are public representation names. Decoder
  schemas remain distinct. New receipts are v3; preserved receipts and raw
  evidence keep their original version and labels, with no in-place relabeling.
- Loading currently defaults to the source-native representation. Universal
  packed-by-default loading requires a qualified original-HDF5 conversion path;
  do not silently reject or transcode the ordinary HDF5 default prematurely.
- Exact native integer counts, full declared coverage and dtype; no hidden crop,
  binning, downcast, exclusion, clipping, approximation, or weakened parity.
- Original values at excluded detector pixels remain recoverable. Product mask
  policy is explicit and separate from storage.
- Random DP/pixel/ordered batch access preserves request ordering and duplicates.
  Report actual block decoding/IO; do not claim direct access after full decode.
- Stateful detector-mask updates must equal fresh exact reductions, including
  empty masks, changed masks, rebase, failures, and cleanup.
- No full dense intermediate for ANS-to-packed streaming. Bound chunk staging;
  report payload, tables, exceptions, indexes, scratch, steady and peak memory.
- No user-interface or app policy in this package. No push/release/merge or
  shared-service changes implied. One coordinator integrates reviewed commits.

## Implementation checklist

- [x] Preserve original CUDA/Apple dirty lineages on both hosts.
- [x] Establish shared clean baseline and independent small integer oracle.
- [x] Assign isolated CUDA, MPS, native Metal subagents; keep shared API here.
- [x] Freeze canonical names and reject removed format/representation aliases.
- [ ] Qualify automatic HDF5-to-packed loading before making packed the default.
- [x] Define one self-contained ANS envelope with integrity checks and optional
  external whole-file authentication (not HDF5).
- [x] Implement a bounded reference ANS producer/reader for exact qualification.
- [x] `io.save(..., format="quantem", compression="ans", backend="cpu")`
  with transactional output,
  exact uint8/uint16 values, native 4D geometry and metadata validation.
- [ ] Accelerated ANS saving from dense/packed/ANS GPU owners.
- [ ] `io.load(ANS file, representation="ans")` on CUDA, Python MPS, native Metal.
- [ ] `io.load(ANS file, representation="packed")` streamed without full dense array.
- [ ] `io.load(ANS file, representation="dense")` with exact admission and dtype.
- [ ] Public reversible resident conversion; no implicit CPU fallback.
- [ ] Dense/packed/ANS save/reopen and conversion round trips, including raw exclusions.
- [ ] Header-only `io.inspect` and folder `io.discover` support for `.ans` sources.
- [ ] Exact requested DP and ordered/duplicate batch extraction for every representation.
- [ ] Exact BF/ABF/ADF and arbitrary binary-mask update paths for every representation.
- [ ] Mean DP, total, and moments: implementation and missing cases explicit.
- [ ] Malformed streams, overflow, cancellation, failed update, stale-generation tests.
- [ ] Same small frozen reference cases on all three physical runtimes.
- [ ] Complete real-data 512 and 1024 validation where sources/devices are available.
- [ ] Repeated cold/prepared/resident timing, copy/IO bytes, and peak memory.
- [ ] Real-time gate per operation, size, representation and physical device.
- [ ] Update canonical backend/profile/benchmark registries only from qualified runs.
- [ ] Review, focused regression, exact local commit, and consumer handoff.

## Current implementation boundary

The new file is an integration-stage `quantem.gpu.count-ans.v1` envelope using
`block-column-rans-byte-v1`. It does not supersede retained research rANS/tANS
experiments or old packed wire profiles. Those lineages and failed trials remain
preserved. Compression is lossless but its ratio is data-dependent; the reference
producer can use declared literal streams and may expand small/incompressible
inputs. No new disk-ratio, large-data speed or real-time claim is made.

| Operation | Runtime | Implemented | Current qualification |
| --- | --- | --- | --- |
| Standalone `.ans` save/reopen | Explicit CPU reference | Yes | Exact bounded uint8/uint16 tests |
| `.ans` to ANS resident | CUDA | Yes | Host oracle and NVRTC; physical GPU pending |
| `.ans` to ANS resident | Python MPS | Yes | Physical small integer tests |
| ANS resident to direct packed | CUDA | Yes | Actual integer-kernel serial oracle; physical GPU pending |
| ANS resident to direct packed | Python MPS | Yes | Physical small integer tests |
| Generic ANS raw DP/mask source | Native Swift/Metal | Yes | Physical small integer tests on M5 Max 128 GB; 24 GB device pending |
| ANS file reader/transcode | Native Swift/Metal | No | Pending canonical envelope integration |
| Exact DP and binary detector masks | CUDA | Yes | Physical GPU pending |
| Exact DP and binary detector masks | Python MPS | Yes | Physical small integer tests |
| New ANS mean-DP and moment products | CUDA | No | Pending; explicit unsupported result |
| New ANS mean-DP and moment products | Python MPS | No | Pending; explicit unsupported result |
| Dense resident to ANS | CUDA | No | Pending |
| Dense resident to ANS | Python MPS | No | Pending |
| Packed resident to ANS | CUDA | No | Pending |
| Packed resident to ANS | Python MPS | No | Pending |
| ANS to dense GPU resident | CUDA | No | Pending |
| ANS to dense GPU resident | Python MPS | No | Pending |

Direct ANS-to-packed currently uploads the encoded ANS source and retains it
while constructing the final packed buffers. No full dense count tensor is
materialized, but **both compressed representations coexist at the conversion
peak**. Incremental disk-shard upload/transcode and physical peak-memory/profile
qualification remain pending. Python MPS prefix-scans only the word-length
index on CPU; decoded scientific counts do not cross that boundary.

The private GPU packed layout uses independent word-aligned scan-block/detector
streams: uint32 words, uint64 offsets, and widths from 0 through 16. Rare 65535
values remain exact. Logical uint8/uint16 dtype is separate from storage words.
Stored metadata retains original acquisition identity and calibration; container
identity is separate. CPU qualification checks the decoded logical hash; GPU
stream validation and optional externally trusted container SHA are reported
without pretending to have independently recomputed the logical GPU hash.

This checklist tracks delivery steps. Existing backend/profile/benchmark
registries remain the source of support and performance claims. A checked API
step is not physical or real-time qualification.

## Historical device gates at initial qualification

- M5 Max 128 GB MPS/native: GPU windows were serialized. Disk capacity fluctuated below
  the existing 18 GiB safety floor; tests paused and resumed only after rechecks.
  No large fixtures, fresh giant builds or disk cleanup were performed.
- CUDA workstation: existing service contexts on both GPUs and unrelated GPU0 jobs;
  CPU/reference work proceeds, physical CUDA timing waits for owned capacity.
- M5 24 GB device: prior app process exited, but its measurement window had not been
  explicitly released by the active owner. No app launch/quit or competing GPU
  work; 128 GB results never substitute for 24 GB physical acceptance.

## Retained evidence

Baseline preservation and verification:
`outputs/representation-integration-20260906/` under the task root.
Current phase runs: `outputs/io-representations-20260906/`.
Per-run manifests record code/patch hashes, input identities, physical runtime,
parameters, outputs, and terminal status before any claim is promoted.

## Reproduction entry points

From this worktree with its Python dependencies installed:

```sh
PYTHONPATH=src python -m pytest -q tests/contracts/io tests/parity/test_resident_integer_contract.py
PYTHONPATH=src python -m pytest -q tests/hardware/mps/test_ans_file_workflow.py tests/hardware/mps/test_mps_ans_counts.py
PYTHONPATH=src python -m pytest -q tests/hardware/cuda/test_cuda_ans_counts.py
```

Run hardware suites only with verified device ownership and adequate memory/disk
capacity. CUDA opt-in flags and device requirements are recorded with the CUDA
tests and `experiments/20260906-cuda-ans-counts/README.md`. Do not interpret skips
as passes. At the earlier integration checkpoint, the CPU suite passed 192 tests with one explicit
skip; the integrated M5 Max MPS checkpoint passed 41 tests, including the two
standalone-file workflows. These are test counts, not loading-speed measurements.

At that earlier checkpoint, native Swift/Metal source was qualified separately: three focused tests and
103 broader native IO tests executed with three explicit opt-in skips and no
failures. That integrated Swift package was byte-identical to the reviewed source
revision and was not rebuilt in a fresh directory then. The consolidation checks
above supersede that build limitation. A ten-test registry
suite verifies both retained historical and canonical manifest formats; no
support or performance cell was promoted by this evidence-format correction.

The new user-facing calls currently proven on MPS are:

```python
from quantem.gpu import detector, io

# Explicit CPU reference encoding today; GPU encoding remains a checklist gate.
io.save(
    "experiment.qgpu", native_counts,
    format="quantem", compression="ans", backend="cpu",
)

# Disk remains ANS, while the selected GPU representation is direct bitpacking.
with io.load("experiment.qgpu", backend="mps", representation="packed") as data:
    session = detector.prepare(data)
    pattern = session.frame(scan_row * data.shape[1] + scan_column)
    image = session.masked_sum_exact(binary_detector_mask)
```

`native_counts` is a four-dimensional NumPy uint8/uint16 array with explicit
scan-row, scan-column, detector-row, detector-column order. `binary_detector_mask`
has the detector shape and contains only zero or one. The returned DP preserves
the native count dtype, and the exact detector image is uint64. The full source
is never copied to the host by either detector call. Metadata/shape transforms,
reverse conversions and full-resolution performance gates remain as listed above.
