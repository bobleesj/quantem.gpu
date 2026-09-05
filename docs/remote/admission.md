# GPU admission and residency

QuantEM.GPU Remote places complete resident datasets on individual CUDA GPUs.
It does not combine device memory to make one dataset fit.

## Capacity model

For each configured device, admission accounts for:

- the service cache budget (currently 80% of device memory);
- exact resident dataset bytes;
- peak load bytes, including decoder and conversion scratch;
- a CUDA load headroom reserve;
- active entries that cannot be evicted; and
- current physical free memory.

The capabilities response exposes both `available_peak_bytes` and
`available_resident_bytes`. A client must satisfy both dimensions:

```text
requested_peak_bytes    <= available_peak_bytes
requested_resident_bytes <= available_resident_bytes
```

Peak capacity may include reclaimable cache entries; resident capacity protects
active datasets. Neither value is a guarantee against concurrent allocations,
so the service's final load response remains authoritative.

## Placement and eviction

One dataset stays wholly on one selected GPU. Multiple GPUs increase the
number of datasets that may remain resident concurrently. Candidate selection
prefers a device that satisfies both capacity dimensions, then applies bounded
cache eviction when permitted. Active or reserved entries are not evicted.

If no device can satisfy the requested plan, the service returns a capacity
error. It does not split one volume across devices, crop scan positions, bin
detector pixels, change dtype, or fall back to CPU.

## Provenance and observability

Record the selected device, requested and admitted resident/peak bytes, cache
budget, active resident bytes, evictable bytes, physical free bytes, requested
shape/dtype, crop/bin plan, and response status. These values are admission
evidence, not a substitute for measured peak VRAM.

The LZ4 packed loader stages the complete encoded file on the selected GPU
during construction. Its admission estimate includes those file bytes plus
resident storage, conservative per-shard scratch, and headroom. The direct
bitpacked profile does not use that full-file staging allocation. A reported
five-gigabyte resident source is therefore not a five-gigabyte peak-load or
total-process memory claim.

For a compact entry, `/api/browse/residency` also exposes separately measured
metadata, whole-file integrity, source read, host validation, GPU upload,
NVRTC compile, GPU validation/decode, decoded-integrity, total-load, and private
CUDA-pool byte fields. A zero phase means that phase was not part of the
selected Lossless Pack Format v1 encoding profile; it is not an unmeasured
timing. Client decode, transport,
display upload, first presentation, and A-B-A switch latency remain client-side
measurements and must not be inferred from server load time.

Integrity verification, transfer, and prepared-moment construction can overlap.
Do not sum their phase durations to estimate wall time. Kernel warmup happens
at service initialization for configured packed sources and is outside an
already-running service's load request. Application launch, SSH setup, source
preparation, server residency, first presentation, and interactive switching
must each retain their own measurement boundary.

Physical acceptance also samples process allocation/reserve and total-card
occupancy while loading and computing. A memory-only regression can preserve
all numerical outputs, so value parity alone is insufficient.

## Admission tests

Tests cover free-memory-unavailable fallbacks, active entries, multi-GPU
placement, exact boundary requests, eviction, and over-budget rejection. The
advertised capacity calculation and the authoritative placement decision must
use the same estimator so they cannot drift.
