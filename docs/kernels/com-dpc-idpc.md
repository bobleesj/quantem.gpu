# CoM, DPC, and iDPC

Center of mass converts each diffraction pattern into a detector-space vector.
For optional detector mask $M[q_r,q_c]$, first compute

```text
4D counts → fused intensity/row-moment/column-moment reduction
          → CoM row/column fields → center and rotation selection
          → Fourier integration → scan-shaped iDPC phase
```

$$
S[R_r,R_c]=\sum_{q_r,q_c}M[q_r,q_c]I[R_r,R_c,q_r,q_c].
$$

Then

$$
\mu_r[R_r,R_c]
=\frac{\sum_{q_r,q_c}q_rM[q_r,q_c]I[R_r,R_c,q_r,q_c]}
{S[R_r,R_c]},
$$

$$
\mu_c[R_r,R_c]
=\frac{\sum_{q_r,q_c}q_cM[q_r,q_c]I[R_r,R_c,q_r,q_c]}
{S[R_r,R_c]}.
$$

`com_row` is $\mu_r$ and `com_col` is $\mu_c$:

$$
(\text{row},\text{column})\equiv(r,c).
$$

In plain terms, `(row, column)` is `(r, c)`.

This explicit naming is required at Python, Swift, CUDA, Metal, and WebGPU
boundaries; a backend may not swap components to match launch coordinates.

## Coordinate, shape, dtype, unit, and provenance contract

The detector input is
`I[scan_row, scan_column, detector_row, detector_column]`. `com_row`,
`com_col`, their aligned components, and `phase` are float32 arrays with shape
`(scan_row, scan_column)`. Detector coordinates are expressed in detector
pixels unless calibrated units are explicitly provided. Zero-intensity or
fully masked frames produce finite zero moments rather than division-by-zero
values.

Provenance records source identity and loaded geometry, detector bin/crop,
source and moment dtypes, detector mask/checksum, detector and scan calibration,
rotation search configuration, selected `rotation_deg`, `use_transpose`,
backend/device, and package revision. The phase convention and FFT
normalization are part of the result contract, not display choices.

## From CoM to DPC and iDPC

The CoM field is centered and rotated into a DPC field
$\mathbf g=(g_r,g_c)$. When automatic rotation is requested, the chosen angle
minimizes the configured curl criterion on the scan-shaped vector field.

Integrated DPC reconstructs a scalar phase-like field in Fourier space. With
scan frequency $\mathbf k=(k_r,k_c)$, a standard least-squares integration is

$$
\hat\phi(\mathbf k)
=\frac{-i\,[k_r\hat g_r(\mathbf k)+k_c\hat g_c(\mathbf k)]}
{k_r^2+k_c^2+\epsilon},
$$

with the zero-frequency value and normalization fixed by the shared contract.
The inverse two-dimensional FFT returns $\phi[R_r,R_c]$.

```python
from quantem.gpu import dpc, io

loaded = io.load("scan_master.h5", backend="auto", det_bin=1)
result = dpc.run(loaded.data)

print(result.com_row.shape, result.com_col.shape)
print(result.rotation_deg, result.use_transpose)
```

## Optimization model

CoM should not require three full detector-volume traversals. A fused moment
kernel accumulates $S$, $\sum q_rI$, and $\sum q_cI$ in one pass, with masks
and bad-pixel treatment applied in the same order as the reference. The large
source remains accelerator-resident; only the small scan-shaped moment fields
continue to rotation and FFT integration.

Rotation search operates on those scan-shaped fields. Batched analytic
curl/divergence moments avoid allocating a full rotated field for every
candidate angle. iDPC keeps both vector components and FFT intermediates on the
same device until the final result is requested.

## Source map and gates

| Layer | Source |
|---|---|
| Public workflow and result | `src/quantem/gpu/dpc` |
| CUDA CoM/DPC | `src/quantem/gpu/dpc/compute/cuda` and detector CUDA moment kernels |
| Python MPS/Metal | `src/quantem/gpu/dpc/compute/mps` |
| WebGPU | `src/quantem/gpu/dpc/compute/webgpu` |
| Native Metal and FFT | `Metal4DSTEMKernels` and `MetalImageFFT` |

Parity reports compare `com_row`, `com_col`, centered/rotated DPC components,
`rotation_deg`, transpose convention, and iDPC phase. They include asymmetric
detector patterns and rectangular scans specifically to catch row/column swaps.
