# Native SSB count inputs and saved runs

The native `MetalSSBEngine.prepare(brightfield:countType:)` accepts plane-major
`[logical BF, 512, 512]` unsigned counts at uint8, uint16, or uint32 width.
Counts are converted directly to float32 on Metal before complex64 FFTs. There
is no clipping, integer narrowing, binning, or scan crop. Float32 arithmetic
still rounds integers above its exact representability limit; uint32 source
support is not a float64 reconstruction claim.

```swift
try engine.prepare(brightfield: columns, countType: .uint16)
let fit = try engine.optimize(start: initial, globalTrials: 200, seed: 42)
```

The existing optimizer is a seeded global search followed by Nelder-Mead,
not 200 gradient-descent iterations. It optimizes C10, C12 and phi12 using the
existing exact phase-variance objective. Higher-order native reconstruction and
original-file/packed-resident BF extraction are separate integration work;
this API does not claim they are implemented.

## Saved numerical result

`MetalSSBSavedRun` records both complex64 images, calibrated geometry,
aberrations, rotation, fit history, seed, source identity, backend revision,
historical reconstruction timings and provenance. The binary property-list
format is versioned and contains SHA-256 checksums for the two image payloads.
Writes are atomic. Loading validates the requested source identity and payloads.

```swift
let saved = try MetalSSBSavedRun(
    result: reconstruction, sourceIdentity: sourceSHA256,
    backendRevision: revision, geometry: geometry, aberrations: fitted,
    rotationDegrees: rotation, optimization: fit, seed: 42)
try saved.save(to: resultURL)
let previous = try MetalSSBSavedRun.load(
    from: resultURL, matchingSourceIdentity: sourceSHA256)
let images = try previous.reconstruction(device: device)
```

Loading restores the saved calibration; it does not silently apply current UI
overrides to an old result. No optimizer or raw-source decode runs during image
restoration. Historical timings must not be presented as reopening timings.
The app owns filenames, retention, source fingerprint construction, and whether
to restore a saved run or start a new one. User-valued saved runs belong in
Application Support or an explicitly exported file, not an evictable data cache.

## Evidence and remaining gates

`bash scripts/check_metal_ssb.sh` runs without XCTest on an Apple GPU. On Apple
M5 it verified identical reconstructions/loss for equal uint8/uint16/uint32
counts, values beyond narrower widths, full-cache versus streamed reconstruction
relative L2 of 1.03e-6 or less (existing 1e-4 limit), and a 200-trial synthetic
fit followed by 82 refinement evaluations. Saved images and fit history reload
exactly; another acquisition's identity is rejected.

The scaling fixture deliberately has zero DC. It is not valid evidence for a
phase-variance fit: attempting to fit it produced a nonfinite objective twice.
The fitting fixture uses positive DC; no objective or frozen parity tolerance
was changed to bypass that failure.

These are small synthetic BF fixtures with full 512×512 scan dimensions, not
real full-BF acquisition timing. The native UI, real-source calibration,
higher-order aberrations, 24 GB memory admission and complete 10–20 second
workflow still need end-to-end qualification.
