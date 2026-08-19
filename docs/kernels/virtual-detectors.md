# BF, DF, and ADF reductions

A virtual detector reduces each diffraction pattern with a detector-space mask
$M(q_y,q_x)$:

$$
V_M[r_y,r_x]
=\sum_{q_y,q_x}M[q_y,q_x]I[r_y,r_x,q_y,q_x].
$$

The result keeps the full scan shape. Array order remains
`(row, column) ≡ (y, x)` in both scan and detector space.

## Products

- **BF** selects the central bright-field disk.
- **ADF** selects an annulus between inner and outer detector radii.
- **DF** selects detector signal outside the bright-field disk or inside an
  explicitly supplied dark-field mask.
- **Mean diffraction** reduces scan coordinates instead:

  $$
  \bar I[q_y,q_x]
  =\frac{1}{N_r}\sum_{r_y,r_x}I[r_y,r_x,q_y,q_x].
  $$

```python
from quantem.gpu import detector, io

loaded = io.load("scan_master.h5", backend="auto", det_bin=1)

bright = detector.bf(loaded.data)
annular = detector.adf(loaded.data, inner=40, outer=90, unit="px")
dark = detector.df(loaded.data)
mean_dp = detector.mean_dp(loaded.data)
```

Detector radii use detector-space calibration. A value in pixels must not be
reported as mrad without calibration.

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
