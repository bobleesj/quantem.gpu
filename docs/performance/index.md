# Benchmarks and parity

This section is the numerical record for `quantem.gpu`. It keeps current
results easy to find while preserving the detailed history needed to understand
which optimizations were accepted, rejected, or remain provisional.

## Where to find the numbers

| Evidence | Use it for |
|---|---|
| [Documentation landing overview](../intro.md) | Friendly module-first entry point with current representative devices and timings |
| [Implementation dashboard](../dashboard.md) | Dense atomic matrix of platform, bin, dtype, cache state, memory, parity, device, and test date |
| [Verified benchmark results](results.md) | Metrics with date, revision, device, shape/dtype, cache state, load plan, benchmark definition, calibration, and parity provenance |
| [Revision and change ledger](changes.md) | Latest documentation changes, implementation baselines, and why a retained number changed or remained historical |
| [Current verified results](../backends.md) | Capability status and the public-safe timing/parity summary across CUDA, MPS/Metal, Swift, and WebGPU |
| [Benchmark methodology](methodology.md) | Required timing stages, cold/warm definitions, memory reporting, and acceptance rules |
| [Cross-backend parity](parity.md) | Exact integer contracts, floating metrics, fixtures, and hardware gates |
| [Optimization ledger](../maintainer/backend-optimization-matrix.md) | Accepted and rejected IO, kernel, display, and browser experiments |
| [Load acceptance evidence](../maintainer/backend-4dstem-load-checklist.md) | Real-data load/decode/product gates and backend signoff |
| [SSB performance evidence](../maintainer/ssb-performance.md) | Shape-by-backend SSB kernel history, memory, and exactness |
| [Native Metal loader postmortem](../maintainer/native-metal-hdf5-postmortem.md) | Pass-graph, redundant-work failure, and prevention checklist |
| [M2 Air Metal evidence](../maintainer/m2-air-lz4-match-unroll-2026-08-18.md) | Physical low-memory Apple load/decode profiling and retained kernel evidence |
| [WebGPU memory history](../maintainer/history/webgpu-gqk-memory-2026-07.md) | Archived browser G(q,k) layout and memory experiments |

The machine-readable fingerprints for retained numerical pages are available as
{download}`evidence_manifest.json <evidence_manifest.json>`. The manifest makes
an evidence edit explicit: changing a number requires updating its fingerprint
and rerunning the documentation guard.

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

The summary page contains the current retained result. Maintainer ledgers keep
chronological experiments—including regressions—so future work does not repeat
failed kernel layouts or confuse an older record with the production path.

Dated pages with headings such as “Question,” “TODO,” or “Next” are collected
under [Historical experiments](../maintainer/history/index.md). Their wording is
preserved as experiment provenance and does not define current commitments.

Numbers are never deleted merely because a faster result appears. They are
superseded with exact revision and protocol context.
