# Integrating count encodings and Fourier storage

This is the integration boundary for ongoing lossless-memory work, not a claim
that every research encoding is available through every backend. Keep existing
dense and packed workflows usable while qualifying one new implementation at a
time. The [representation API](../api/representations.md) describes current
calls; [backend readiness](../performance/backend-readiness.md) retains the
scientific and physical-device gates.

## Four responsibilities, not four public loaders

| Layer | Owner | Retained information |
|---|---|---|
| Acquisition | `io` | Original count identity, geometry, dtype, calibration, exclusions |
| Count representation | `io/backends/<runtime>` | Dense values or an authenticated lossless payload and all decoding tables |
| Detector query index | `detector/backends/<runtime>` | Optional exact summaries, checkpoints, or spatial indexes bound to one resident source |
| Fourier representation | `ssb/backends/<runtime>` and native SSB products | Final Fourier coefficients and all descriptors needed to consume them |

```text
authenticated source -> io.load -> dense, packed, or ANS counts
                                     |                  |
                              detector products      SSB preparation
                                                        |
                                                final Fourier fields
                                                        |
                                            aberration updates and search
```

`representation="dense"`, `"packed"`, and `"ans"` are the count choices for
the integration, with no legacy selector aliases. `"packed"` covers compact
count storage; authenticated profile metadata still distinguishes bitpacking,
block compression, and the ANS-derived layout. File layout and compression
are separate: `io.save(..., format="quantem", compression="ans")`
uses the standalone envelope; `format="arina"` retains the HDF5 layout.
Decompression is detected during loading, not selected by a second argument.
Do not add `load_rans`, `load_tans`, or a backend-specific
scientific workflow. Encoding profiles belong in authenticated source metadata
and private dispatch. Reject an unknown profile before dispatch; do not guess
a codec from a compression ratio or filename. ANS coding probabilities do not
approximate measured counts: the decoded integers must remain exact.

Fourier storage is a separate SSB choice. An advanced
`fourier_representation` option is a **design proposal, not a shipped argument**.
Do not overload the count `representation` argument to select it. The existing
source-native load behavior and dense Fourier defaults remain unchanged during
extraction. Compact counts do not imply equally compact Fourier coefficients.

## Current integration lanes

| Component | Current public boundary | Research lane | Next acceptance gate |
|---|---|---|---|
| Direct bit-packed counts | Profile-specific CUDA, Python MPS, native Metal, and WebGPU readers | Supported packed profiles | Repeat exact counts, exclusions, corruption, lifetime, and products. |
| Existing uint16/LZ4 packed counts | Documented CUDA, native Metal, and WebGPU paths; not a Python MPS packed profile | Retained alongside bit-packing | Preserve authentication and distinguish resident bytes from full-file staging peak. |
| Standalone exact count-ANS envelope | `io.save` with explicit CPU reference; `io.load` as ANS or packed on Python MPS/CUDA, dense on CPU | New `quantem.gpu.count-ans.v1` integration | Physical CUDA, native file reader, GPU saving/reverse conversions, complete real data, and peak-memory qualification. |
| Range ANS (rANS) counts | Not a general public `io.load` profile | CUDA and native Metal research consumers | Remove fixed geometry and research paths; freeze a portable stream/table contract and independent complete-count parity. |
| Table ANS (tANS) counts | Not a general public `io.load` profile | CUDA producer and consumer research | Verify inverse transitions, literal values, malformed streams, bounds, and exact products before packaging. |
| Exact detector indexes | Backend-specific facilities, not a uniform guarantee | Moving-mask, checkpoint, and spatial-sum candidates | Account for all dependencies; compare changed-mask outputs with the direct exact reference. |
| Compressed stationary final Fourier fields | No public cross-backend compressed-Fourier option | Opt-in native Swift/Metal research implementation | Extract the qualified consumer and lifecycle; preserve coefficient bits, object/loss parity, invalidation, and packing peak. |

Python MPS and native Swift/Metal are distinct consumer paths even when both
execute Metal kernels. A native test does not qualify the Python API. CUDA,
WebGPU, and Vulkan do not inherit compressed-Fourier support from Metal tests.
A bounded decode test is not proof of full-dataset residency or display speed.

## Extraction sequence

1. **Preserve first.** Record branch, full commit, staged/unstaged binary
   patches, untracked-file hashes, dependencies, and retained run manifests.
   Keep two verified copies before reconciling distinct research trees. A Git
   commit alone does not preserve an untracked decoder or codebook.
2. **Freeze one profile.** Capture byte order, model/table identities, source
   and working dtypes, ordering, literals, bounds, calibration, exclusions, and
   invalidation. Retain the NumPy/Torch reference before the accelerated port.
   Profile versioning belongs in metadata, not method names.
3. **Extract the smallest implementation.** IO owns count decoding, detector
   owns indexes, and SSB owns final Fourier storage. Native clients use package
   products. No copied app kernels or runtime imports from research directories.
4. **Test both representations.** Exact integer equality includes rare maximum
   values and excluded raw values. Packing final Fourier fields must recover
   their original IEEE words; downstream arithmetic retains its frozen
   full-precision comparison. Checkpoint/FFT regeneration is not directly
   consumable stationary final Fourier storage.
5. **Qualify one backend.** Test complete real acquisitions, changed parameters,
   teardown/cancellation, authentication failures, and dense/packed switching.
   Separate cold source, prepared creation/reopen, first usable, exact complete,
   and resident updates. Preserve failures and outliers.
6. **Hand off a pin.** Deliver a clean revision, public contract, fixture hashes,
   commands, results, limitations, and memory accounting. Consumers select
   application policy and perform their own headed acceptance.

Do not merge whole experimental directories or delete slower baselines because
a newer layout exists. Keep rejected experiments outside production imports,
with their reason and reproducible retained records.

## Memory accounting

Report source logical count bytes, working logical count bytes, and resident
count bytes separately. Resident counts include payloads, offsets, codebooks,
literal exceptions, and alignment. Separately report optional indexes, final
Fourier payload/descriptors, outputs, and scratch.

Measure peak accelerator allocation and peak process RSS alongside steady
resident bytes. Packing can retain both old and new Fourier buffers until
verification succeeds. Count shared codebooks once per physical owner, not once
per dataset and not zero times. On Apple hardware, logical Metal buffer lengths,
driver allocation, RSS, compressed memory, and swap differ; do not add
overlapping unified-memory counters. No fixed compression factor is guaranteed
for arbitrary data. Admission uses actual payload and dependency sizes and
fails explicitly if the exact plan cannot fit.

## Reproduce the readiness checks

Use the existing registries instead of maintaining another support matrix:

```bash
python scripts/backend_status.py check
python scripts/backend_status.py summary
python scripts/backend_status.py json --backend cuda
python scripts/run_tests.py tests/contracts/io tests/parity/test_resident_integer_contract.py -q
```

IO import checks cover both standalone use and coexistence with an installed
`quantem` parent. Imports performed by that parent are attributed separately.
CuPy installation does not guarantee a CUDA device; unavailable hardware checks
skip explicitly and cannot count as passes.

Small physical Apple checks:

```bash
python scripts/run_tests.py tests/hardware/mps/test_mps_compact_v3.py tests/hardware/mps/test_resident_integer_contract.py -q -rs
```

On an available, owned CUDA device:

```bash
python scripts/run_tests.py tests/hardware/cuda/test_compact_h5_cuda.py tests/hardware/cuda/test_resident_integer_contract.py -q -rs
```

These are bounded regressions, not cold-load or full-acquisition performance
measurements. Use the [profiling runbooks](../performance/coverage.md) for those
gates. Add qualified numbers to the existing benchmark registry only when
source, fixture, hardware, sample count, timing boundary, and memory definitions
are complete.
