# Shared Metal image statistics

`MetalDisplayKernels` owns the colormap tables, normalization shaders, and
histogram/range kernels. `MetalImageRuntime` owns percentile and display-limit
math and the reusable statistics API. Neither imports a UI framework.

## Reusable outputs

Create a `MetalDisplayStatistics` once for the device. Each image needs its
existing scalar buffer, 8 bytes for minimum/maximum, and 1024 bytes for the
256-bin UInt32 histogram. Output buffers can be shared or private Metal storage.

```swift
let statistics = try MetalDisplayStatistics(device: device, commandQueue: queue)
let image = MetalStatisticsRequest(
    values: values, rows: rows, columns: columns, scalarType: .float32,
    scale: .linear, valueRange: range, histogram: histogram)
let command = queue.makeCommandBuffer()!
try statistics.encode([image], into: command)
// Encode dependent work here; do not read the outputs yet.
command.commit()
```

`encode` does not commit, wait, allocate data buffers, or read back image pixels.
The caller owns command completion and output lifetime. Ranges and histograms
must be disjoint from every input and other output. Use buffers on the same
device with normal hazard tracking; do not recycle them while in flight.
If encoding throws, discard the command rather than submitting partial work.
Different shapes and scalar types may share one batch. `updateRange: false`
reuses an already-completed range when only the display scale changes.

Use `analyzeUInt32`, `analyzeUInt32Batch`, or `analyzeFloat32` for a short,
synchronous call that allocates its own outputs and returns completed scalar
limits and histogram bins. These convenience methods use the same encoder.

## Numerical contract (v1)

- UInt32 samples are exact, including `UInt32.max`; it is not a missing value.
- Float statistics exclude NaN and infinities. An all-nonfinite image has zero
  range and an empty histogram. Negative finite values remain valid.
- A constant finite image occupies bin 128. Otherwise the normalized value
  maps to `min(255, floor(value * 256))`; the maximum occupies the last bin.
- Log display uses the existing signed-log convention. Histogram construction
  uses Float32 shader arithmetic; display labels use Double arithmetic.
- `MetalPercentileRange` uses fractions: `0.05...0.95` means 5...95 percent.
  Percentile windows preserve the existing rank `floor((N-1)*p)`, N-1
  bin-center display coordinates, and minimum display-window width. The
  full-percentile policy explicitly returns the full display window.
- `MetalFloatDisplayMapping`, `MetalRawContrastMapping`, and `MetalDisplayLimits`
  own conversion and representable display limits. Scientific samples never
  change when display limits or colormaps change.

The app owns which images share a policy, persisted preferences, labels,
gestures, visibility, cache eviction, and newest-request-only publication.
CPU work over a 256-bin summary is not a full-image CPU fallback.

## Migration and verification

`MetalFloat32Statistics.valueRange` now contains ordinary Float minimum/maximum
values. It replaces the implementation-specific `orderedValueRange`; no alias
is retained. The ordered-bit reduction representation stays inside encoding.

Run `bash scripts/check_metal_display.sh` on an Apple GPU. The standalone
consumer imports only shared packages and checks mixed shapes, integer/float
extremes, nonfinite samples, exact small-array histogram oracles, buffer reuse,
percentile conversion, and all colormap tables. Run native app presentation
gates separately; kernel correctness does not certify UI frame rate.
