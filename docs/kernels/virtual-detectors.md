# BF, DF, and ADF reductions

A virtual detector reduces each diffraction pattern with a detector-space mask
$M(k_r,k_c)$:

```text
4D counts → detector geometry/mask → widened masked reduction
          → scan-shaped BF, DF, or ADF product → provenance
```

$$
V_M[R_r,R_c]
=\sum_{k_r,k_c}M[k_r,k_c]I[R_r,R_c,k_r,k_c].
$$

The result keeps the full scan shape. Array order remains
`(row, column) ≡ (r, c)` in both scan and detector space.

## Products

- **BF** selects the central bright-field disk.
- **ADF** selects an annulus between inner and outer detector radii.
- **DF** selects detector signal outside the bright-field disk or inside an
  explicitly supplied dark-field mask.
- **Mean diffraction** reduces scan coordinates instead:

  $$
  \bar I[k_r,k_c]
  =\frac{1}{N_R}\sum_{R_r,R_c}I[R_r,R_c,k_r,k_c].
  $$

```python
from quantem.gpu import detector, io

data = io.load("scan_master.h5")

bright = detector.bf(data)
annular = detector.adf(data)
dark = detector.df(data)

# Override physical detector choices when needed.
bright = detector.bf(data, radius=45)  # detector pixels
annular = detector.adf(data, inner=60, outer=85, unit="px")
```

Detector radii use detector-space calibration. A value in pixels must not be
reported as mrad without calibration.

The first BF/ADF/DF call fits the disk automatically and stores its geometry
with the encoded acquisition. Subsequent calls reuse the fit. Explicit
`center=(row, column)` and `radius` override only that call, without replacing
the automatic fit. Reopening an acquisition gets a fresh fit. Array selections
and mutable array inputs are fitted independently, so changes to measurements
or detector coordinates cannot reuse an acquisition's old geometry.

For diagnostics, `detector.fit_probe(detector.mean(data))` returns the fitted
center and radius. It estimates disk geometry from a thresholded mean
pattern, not probe phase or aberrations. The cached geometry is private runtime
state; it does not modify acquisition metadata or enter saved QEM files.

## Coordinate, shape, dtype, unit, and provenance contract

| Item | Contract |
|---|---|
| input | `I[scan_row, scan_column, detector_row, detector_column]` |
| mask | `M[detector_row, detector_column]`, boolean or declared dimensionless weight |
| BF/DF/ADF output | `(scan_row, scan_column)` |
| mean diffraction output | `(detector_row, detector_column)` |
| arithmetic | widen before integer accumulation; never wrap silently |
| units | detector radius in px or calibrated mrad; reduced intensity in the declared count/weight convention |

Provenance records source identity and loaded geometry, source/accumulation/
output dtypes, detector bin/crop, mask geometry or checksum, calibration and
units, backend/device, and package revision. A detector-binned source produces
a binned-detector product and cannot be presented as native detector sampling.

## Optimization model

The dominant operation is an embarrassingly parallel reduction over detector
pixels for every scan position. Efficient implementations:

- prepare and cache mask indices once;
- keep source counts and masks accelerator-resident;
- use widened integer accumulators with explicit overflow gates;
- combine BF/ADF/DF, total intensity, or detector moments in one source pass
  when their exact arithmetic permits it;
- use `total - complement` only when that identity is exact for the mask and
  dtype; and
- return scan-shaped products without downloading the detector volume.

Mask density determines topology. Sparse selected-index gathers, dense tiled
reductions, and fused multi-product kernels can each be correct; benchmark the
actual mask and source layout rather than assuming one universal winner.

## Source map and gates

| Layer | Source |
|---|---|
| Public contract and geometry | `src/quantem/gpu/detector` |
| CUDA reductions | `src/quantem/gpu/detector/compute/cuda` |
| Python MPS/Metal reductions | `src/quantem/gpu/detector/compute/mps` |
| WebGPU reductions | `src/quantem/gpu/detector/compute/webgpu` |
| Native Metal reductions | `src/quantem/gpu/swift/Sources/Metal4DSTEMKernels` |

Parity fixtures include asymmetric masks, rectangular scan/detector shapes,
odd edge bins, all-zero masks, and values near accumulation limits. Integer
products are byte-exact where the declared accumulation dtype is the same.
