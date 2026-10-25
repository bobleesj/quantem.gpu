# Original-HDF5 median fill and its DPC moments

A viewer may load an original indexed acquisition with
`MetalCompactH5Loader.load(..., hotPixelPolicy: .median)`, which replaces each
pixel of the stored detector mask by the median of its valid 3 × 3 neighbors.
Two defects made that resident differ from what Python `io.load` computes from
the filled counts.

## Rounding

`h5hot_pixel_median_qh5idx` and `NativeHotPixelCorrection.median3x3` took the
floored mean of the two middle neighbors of an even neighborhood. Python
`fill_flagged` rounds that mean half to even. On a real 512 × 512 acquisition
one flagged bright-field pixel held 4.6 % fewer counts than Python's fill: one
count less at 22 % of the scan positions. Both now round half to even.

## DPC moments

The default uint16 load fuses each decode window's exact DPC moments into the
unshuffle, which stores 0 at masked pixels before the median fill runs. A window
whose slices are all at least 2048 frames therefore left the flagged pixels out
of its moments, while a window with a shorter slice, one that straddles a data
file boundary, measured its moments after the fill and included them. The
viewer's CoM, DPC and iDPC images banded at that window period. uint32 loads fuse
every window, so they left the flagged pixels out everywhere.

`h5hot_pixel_dpc_qh5idx` now runs after the fill in every fused window and adds
the filled values to that window's moments. Every frame's moments are those of
the corrected counts, in fused and unfused windows alike; `.preserve` loads are
unchanged.

## Not changed

- `native_median3x3` (encoded source and corrected resident view) keeps the
  floored mean that its frozen counts record.
- A flagged pixel with no valid 3 × 3 neighbor becomes 0 natively. Python takes
  the median of the smallest larger window that holds one.

## Verification

`tests/hardware/metal/test_original_hot_pixel_median.py` writes a synthetic
two-file acquisition (one fused window, one straddling window; uint16 and
uint32) and compares a release-built `.median` load with `fill_flagged`: the
flagged pixels' counts in every frame, per-pixel totals and every frame's exact
moments. A `.preserve` load keeps the stored markers in all three. With either
fix removed, the test fails. `Native4DSTEMIOTests` pins the rounding of the CPU
correction and checks the Metal fill and moment correction against it.
