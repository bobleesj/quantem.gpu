# Benchmark methodology

Performance is reported only after scientific parity. A faster result that
changes source coverage, detector sampling, precision, mask, objective, or
output meaning is a different experiment.

The [continuous profiling plan](continuous-profiling.md) defines which checks
belong on every pull request, which require specified physical hardware, and how a
qualified measurement is promoted to the dashboard. Its machine-readable
platform/module schedule is `benchmarks/profile_matrix.json`.

## Required labels

Every benchmark records:

- repository path, branch, HEAD, and dirty-diff hash;
- operating system, driver/runtime, device model, and dependency versions;
- source identity, byte hash, compressed bytes, decoded bytes, shape, and dtype;
- scan/detector region, scan/detector bin, mask, accumulation/output dtype;
- exact command, environment variables, seed, and cache-preparation state;
- output hashes and parity metric; and
- process memory, accelerator memory, total-card occupancy, pressure, and swap.

## Source and cache states

Report these separately:

| Label | Meaning |
|---|---|
| Cold source | First encounter after a documented storage-cache reset or reboot; process and kernel caches are also identified |
| Controlled uncached source pages | A platform-specific page-reuse control such as macOS `F_NOCACHE` is applied to every declared source descriptor; source audit, index, process, and destination state are still reported separately |
| Warm source | Raw source reopened with operating-system page cache or storage cache available |
| Warm process | Same process and reusable allocations/kernel compilation retained |
| Saved-result reopen | A derived cache or persisted resident payload is opened; not a raw-source load |

If true cache eviction was not controlled, use `first process` or `first
observed source` rather than `cold`.

Controlled source-page IO is not automatically arbitrary-source cold. For
example, a run may use `F_NOCACHE` on an immutable source while reusing a sealed
value-range audit, or it may build a new QH5 index while the source itself is
already qualified. Name each retained state instead of compressing them into
one “cold” label.

## End-to-end stages

User-facing load time is wall clock from the initiating action to the first
complete usable scientific product. Profile at least:

1. file open and metadata/discovery;
2. index lookup or construction;
3. read-span planning and host allocation;
4. storage read and header parsing;
5. decode/decompression;
6. dtype conversion, masking, and bin decision;
7. allocation and resource-pressure admission;
8. detector/scan reduction and layout conversion;
9. GPU transfer where memory is not unified;
10. first complete BF/ADF/DF/CoM/DPC product;
11. cache write/finalization; and
12. total wall time.

Overlapped GPU intervals are not summed and presented as wall time. On unified
memory, page-in and GPU access may be inseparable; report that fact instead of
inventing an upload stage.

## Statistics

Use a smoke run before any matrix. For accepted configurations, report enough
repetitions to provide p50, p95, and maximum wall time. Preserve the run-level
records so initialization outliers and cache effects remain visible.

Kernel microbenchmarks are useful for diagnosis but are not user-facing load
time. End-to-end application evidence remains required.

Timing regressions are evaluated only within an exact comparison key: protocol,
module/platform cell, source identity and plan, cache state, timing boundary,
device/runtime, precision, and source revision. A new key begins in report-only
mode. Do not introduce a blocking percentage threshold until at least five
accepted sessions establish the variance of that exact configuration.

## Memory

Compressed file size is not a fit estimate. Admission includes decoded output,
decoder scratch, reduction/layout buffers, products, allocator reserve, cache
population, and concurrent-service baseline.

Every load row separates:

- source, requested, working, accumulation, and resident/output dtype;
- resident payload bytes calculated from the recorded output shape and dtype;
- planner-estimated peak and its included/excluded allocations;
- measured process RSS/footprint and host peak;
- measured accelerator allocated/reserved peak and total-device occupancy; and
- memory pressure and swap where the platform exposes them.

Do not call a payload size “peak memory.” Do not call `uint8` lossless unless a
complete source-identity-bound audit records the corrected maximum and zero
values above 255. A saturating `uint8` run records its saturation count and is
labeled browse-only.

On Apple unified memory, record four separate layers:

1. **logical resident payload**, calculated exactly from output shape and dtype;
2. **Metal-driver allocation**, sampled during the timed interval and again
   after output release;
3. **process RSS/footprint**, which does not necessarily include every direct
   Metal allocation; and
4. **whole-system pressure and swap** before, during, and after the run.

`torch.mps.current_allocated_memory()` covers Torch-managed allocations. It may
remain zero for buffers created directly through Metal/PyObjC; in that case it
must not be presented as total accelerator memory. Record
`torch.mps.driver_allocated_memory()` or an equivalent Metal counter alongside
RSS, and name the counter precisely.

A repeated-load protocol must explicitly release every caller-owned direct
Metal output before the next repetition. Clearing a framework cache or deleting
the Python wrapper is not proof that a `newBuffer...` allocation was released.
Record the Metal-driver allocation after output release and fail the run if it
grows across repetitions without an intentional cache explanation.

On CUDA, record process allocated/reserved VRAM and total-card occupancy before,
during, and after the run. On WebGPU, browser-process RSS is a useful host
signal but is not a complete GPU-device allocation measurement; an 8 GB gate
also requires whole-system pressure/swap and the physical device run.

## Minimum-device memory gates

Minimum-device support is a complete-pipeline claim, not a payload comparison:

- **CUDA floor:** 6 GiB of dedicated VRAM. Count process allocation and reserve,
  decoder/reduction scratch, products, staging, and other card occupants.
- **WebGPU floor:** 8 GB of total physical laptop RAM. Count the operating
  system, browser, JavaScript heap, staging, GPU buffers, presentation, memory
  pressure, and swap.
- **Apple native/MPS floor:** 8 GB of unified RAM when that row is claimed.
  Record the same whole-process and system-pressure signals as WebGPU.

Gate vocabulary is fail-closed:

| Status | Meaning |
|---|---|
| **✓** | Complete headed or native physical-device run at or below the floor, with parity and peak-memory evidence |
| **Pending** | Payload is a plausible candidate, but complete physical peak or parity evidence is missing |
| **No** | Payload alone exceeds the floor, or the complete run exceeds it |
| **Test** | A capped larger device or software adapter passed as a pre-check; physical floor signoff is still missing |

Do not convert **Pending** or **Test** to ✓ from a larger device, a calculated
payload, a kernel-only microbenchmark, or a prepared-source reopen.

## Interaction sidecars and scientific resolution

A detector-binned interaction sidecar is a distinct scientific sampling plan,
even when it is built automatically after a native-detector load. Its speed may
be reported only with the sidecar detector bin and output meaning. It cannot be
used as parity evidence for native-detector CoM, DPC, iDPC, a diffraction
pattern, or another resolution-sensitive product.

If the public API promises native resolution, parity must exercise the
full-resolution reducer. If a client chooses the interaction sidecar, metadata
and UI provenance must identify its detector bin; the application must not
present the result as native resolution.

## Acceptance

An optimization is retained only when:

- strict parity passes on the unchanged fixture and parameters;
- the intended physical device shows a reproducible wall-time or memory win;
- tails and responsiveness do not regress;
- provenance remains complete; and
- the implementation does not create a backend-specific public API.

A faster sidecar, prepared index, saved result, cropped scan, or detector-binned
representation never inherits the acceptance state of the native source plan.

Rejected experiments stay in the [optimization ledger](../maintainer/backend-optimization-matrix.md).
Every scheduled or diagnostic run also keeps a machine-readable manifest and a
terminal row in the
[`experiments/RUNS.md`](https://github.com/bobleesj/quantem.gpu/blob/main/experiments/RUNS.md)
registry.
