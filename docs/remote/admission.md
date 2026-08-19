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

Physical acceptance also samples process allocation/reserve and total-card
occupancy while loading and computing. A memory-only regression can preserve
all numerical outputs, so value parity alone is insufficient.

## Admission tests

Tests cover free-memory-unavailable fallbacks, active entries, multi-GPU
placement, exact boundary requests, eviction, and over-budget rejection. The
advertised capacity calculation and the authoritative placement decision must
use the same estimator so they cannot drift.
