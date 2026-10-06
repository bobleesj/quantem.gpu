# GPU admission and residency

QuantEM.GPU Remote places complete resident acquisitions on individual CUDA
GPUs. It does not combine device memory to make one acquisition fit.

## Capacity model

Each acquisition is held in the lossless encoded form `io.load` returns. One
resident serves every bin and crop plan, so admission sizes the acquisition,
not the plan. For each configured device, admission accounts for:

- the service cache budget (currently 80% of device memory);
- the expected encoded resident bytes: the smaller of the stored file bytes
  and the dense working bytes, replaced by the measured size once loaded;
- the loader's staging of decoded frames (128 MiB plus 64 bytes per detector
  pixel for each of at least 512 scan positions);
- a CUDA load headroom reserve;
- active entries that cannot be evicted; and
- current physical free memory.

On a 256 x 256 x 192 x 192 Arina acquisition stored as 0.84 GB of
bitshuffle-LZ4 files, the encoded resident measured 0.61 GB, against 4.8 GB
for the dense uint16 working array.

The capabilities response exposes both `available_peak_bytes` and
`available_resident_bytes`. A client must satisfy both dimensions:

```text
requested_peak_bytes    <= available_peak_bytes
requested_resident_bytes <= available_resident_bytes
```

Peak capacity may include reclaimable cache entries; resident capacity protects
active acquisitions. Neither value is a guarantee against concurrent
allocations, so the service's final load response remains authoritative. An
out-of-memory error during loading evicts another entry and retries on the
same GPU.

## Placement and eviction

One acquisition stays wholly on one selected GPU. Multiple GPUs increase the
number of acquisitions that may remain resident concurrently. Candidate
selection prefers a device that satisfies both capacity dimensions, then
applies bounded cache eviction when permitted. Active or reserved entries are
not evicted. Eviction closes the detector session and the loaded acquisition,
which returns the encoded storage to the device even if another reference to
the acquisition remains.

If no device can admit the acquisition, the service returns a capacity error.
It does not split one volume across devices, crop scan positions, bin detector
pixels, change dtype, or fall back to CPU.

## Provenance and observability

Record the selected device, requested and admitted resident/peak bytes, cache
budget, active resident bytes, evictable bytes, physical free bytes, requested
shape/dtype, crop/bin plan, and response status. These values are admission
evidence, not a substitute for measured peak VRAM.

The capabilities response reports the measured resident bytes per device.
Client decode, transport, display upload, first presentation, and switch
latency remain client-side measurements and must not be inferred from server
load time.

## Admission tests

Tests cover free-memory-unavailable fallbacks, active entries, multi-GPU
placement, exact boundary requests, eviction, encoded size estimates, release
on eviction, and over-budget rejection. The advertised capacity calculation
and the authoritative placement decision use the same estimator so they cannot
drift.
