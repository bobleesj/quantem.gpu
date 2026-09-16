# Compressed resident regional mean diffraction

`MetalRuntimeANSResidentSource.meanDiffractionPattern(rows:columns:shape:)`
returns a detector-sized `Float32` mean and exact `UInt64` numerator, directly
from compressed resident counts. This complements point diffraction; it does
not change the encoded data or the SSB detector-column reduction.

## Contract, version 1

- Bounds are half-open `(row, column)` ranges in the native scan.
- Rectangle membership includes every position within both ranges.
- Circle bounds are square. Pixel centers on or inside the circle are included
  with equal weights; the divisor is the actual selected count, not box area.
- All detector pixels retain their original counts, including those excluded
  from virtual detectors. A one-position region equals raw point diffraction.
- Integer sums are exact. Division uses double precision followed by one
  Float32 rounding. No scan/detector crop, binning or subsampling is implicit.
- Omitted ranges select the full scan. Invalid regions fail before dispatch.
- Queries and release must be serialized by the resident owner.
- Only intersecting chunks are decoded. No dense 4D array is allocated; the
  reduction output is eight bytes per detector pixel. Full-scan results are not
  cached by this API.

## Verification

`bash scripts/check_metal_region_mean.sh` generates small uint8 and uint16
fixtures and tests rectangles, odd/even circles, one-point and boundary regions,
repeatability, and full-scan sums across compression chunks. The independent
oracle directly reads original array counts and checks every detector pixel.

For real data use `bash scripts/check_metal_region_mean.sh original.dm4 copy.qem`.
Older compressed copies can also be supplied. Integer sums and final Float32
means must match exactly; timings are reported separately, not used to relax
parity. `QGPU_TEST_BUILD_DIR` can point to an already-built package consumer's
release directory to test its exact linked backend objects and resources.
