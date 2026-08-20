# Benchmarks and parity

This section is the numerical record for `quantem.gpu`. It keeps current
results easy to find while preserving the detailed history needed to understand
which optimizations were accepted, rejected, or remain provisional.

## Where to find the numbers

| Evidence | Use it for |
|---|---|
| [Documentation landing page](../intro.md) | Package contract and routes for API, kernel, and application developers; intentionally contains no benchmark tables |
| [Implementation dashboard](../dashboard.md) | The single friendly current overview of platform, bin, dtype, cache state, memory, parity, device, and test date |
| {ref}`Minimum-device gates <minimum-device-memory-gates>` | CUDA 6 GiB VRAM and WebGPU 8 GB laptop acceptance status; runtime details on [CUDA](../platforms/cuda.md) and [WebGPU](../platforms/webgpu.md) |
| [Verified benchmark results](results.md) | Metrics with date, revision, device, shape/dtype, cache state, load plan, benchmark definition, calibration, and parity provenance |
| [Revision and change ledger](changes.md) | Latest documentation changes, implementation baselines, and why a retained number changed or remained historical |
| [Backend coverage](../backends.md) | Capability and implementation-source status without duplicated timing tables |
| [Benchmark methodology](methodology.md) | Required timing stages, cold/warm definitions, memory reporting, and acceptance rules |
| [Continuous profiling](continuous-profiling.md) | PR smoke, weekly physical profiles, manual signoff, comparison keys, run registry, and regression decisions |
| [Cross-backend parity](parity.md) | Exact integer contracts, floating metrics, fixtures, and hardware gates |
| [Optimization ledger](../maintainer/backend-optimization-matrix.md) | Accepted and rejected IO, kernel, display, and browser experiments |
| [Load acceptance evidence](../maintainer/backend-4dstem-load-checklist.md) | Real-data load/decode/product gates and backend signoff |
| [SSB performance evidence](../maintainer/ssb-performance.md) | Shape-by-backend SSB kernel history, memory, and exactness |
| [Native Swift/Metal SSB migration](../maintainer/native-metal-ssb-migration.md) | iOS source lineage, package API, exact cache policies, real-reference parity, and benchmark fingerprints |
| [Native Metal loader postmortem](../maintainer/native-metal-hdf5-postmortem.md) | Pass-graph, redundant-work failure, and prevention checklist |
| [M2 Air Metal evidence](../maintainer/m2-air-lz4-match-unroll-2026-08-18.md) | Physical low-memory Apple load/decode profiling and retained kernel evidence |
| [WebGPU memory history](../maintainer/history/webgpu-gqk-memory-2026-07.md) | Archived browser G(q,k) layout and memory experiments |

The machine-readable fingerprints for retained numerical pages are available as
{download}`evidence_manifest.json <evidence_manifest.json>`. The manifest makes
an evidence edit explicit: changing a number requires updating its fingerprint
and rerunning the documentation guard.

The machine-readable execution schedule lives in
[`benchmarks/profile_matrix.json`](https://github.com/bobleesj/quantem.gpu/blob/main/benchmarks/profile_matrix.json),
and completed, failed, refuted, or superseded runs remain indexed in
[`experiments/RUNS.md`](https://github.com/bobleesj/quantem.gpu/blob/main/experiments/RUNS.md).

## Reading a result

A complete row answers all of these questions:

1. Which exact source and source revision ran?
2. What were the scan/detector shape, dtype, crop, bin, mask, and precision?
3. Which backend, device, driver/runtime, and kernel revision ran?
4. Was storage cold, page-cache warm, process warm, or a saved-result reopen?
5. What did file open, read, decode, reduction, upload, synchronization, first
   usable product, and total wall time cost?
6. What were peak process memory, accelerator allocation/reserve, total-device
   occupancy, and swap/pressure?
7. Which frozen output or reference proves scientific parity?

If one of these facts is missing, the number may still be useful diagnostically
but is not a release or migration signoff.

## Current versus historical evidence

The dashboard summarizes the current retained result; the results page owns its
complete provenance. Maintainer ledgers keep chronological experiments—including
regressions—so future work does not repeat failed kernel layouts or confuse an
older record with the production path.

Dated pages with headings such as “Question,” “TODO,” or “Next” are collected
under [Historical experiments](../maintainer/history/index.md). Their wording is
preserved as experiment provenance and does not define current commitments.

Valid older numbers are not copied through current overview pages. They remain
once in the owning historical ledger with exact revision and protocol context.
Measurements that failed parity or repeatability remain named as rejected
experiments, never as current timing rows.
