# Trusted local Metal loading

The Swift loader verifies payload checksums by default. A native app whose
operator explicitly trusts local source files can omit the expensive payload
SHA scans without changing the lossless representation or source coverage:

```swift
let source = try MetalCompactH5Loader.load(
  sourceURL: url,
  device: device,
  authenticationPolicy: .boundedConcurrent,
  maximumAdditionalBytes: availableBytes,
  verifyChecksums: false
)
```

This is an explicit exception to checksum-authenticated QGIX admission, not a
claim that checksums passed. It applies to decoded v1 payloads, direct v3
payloads, native-cache payloads/descriptors, and prepared detector/DPC products.
It does not skip file-range, metadata, dtype, width, descriptor-coverage,
decoder-status, or allocation-budget checks. Small metadata identity hashes
remain. Undetected payload changes can produce incorrect scientific values.

`loadMetrics.checksumsVerified` is false and SHA check counters/timings are zero
when verification is skipped. Keep this status with any load receipt; source
identity fields describe the manifest and do not prove the loaded bytes passed
verification. Native-cache saving still verifies resident payloads against the
source digests before publication.

Use the default verified load for initial qualification, explicit audits, and
corruption-detection tests. Numerical parity tests compare both policies on the
same complete input. Runtime display does not run an independent reference
calculation. A trusted open and a verified open must be labeled separately in
performance comparisons.

Bounded concurrent scheduling overlaps at most three shards. Its phase timings
sum overlapping work; use `shardPipelineMilliseconds` for elapsed pipeline time.
The Metal LZ4 decoder initializes its own disjoint output chunks, avoiding a CPU
clear of the full decoded staging volume. Native cache reuse remains optional;
this policy neither creates a cache nor modifies the source HDF5.
