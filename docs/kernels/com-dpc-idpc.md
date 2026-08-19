# CoM, DPC, and iDPC

Center of mass converts each diffraction pattern into a detector-space vector.
For optional detector mask $M[q_y,q_x]$, first compute

$$
S[r_y,r_x]=\sum_{q_y,q_x}M[q_y,q_x]I[r_y,r_x,q_y,q_x].
$$

Then

$$
c_y[r_y,r_x]
=\frac{\sum_{q_y,q_x}q_yM[q_y,q_x]I[r_y,r_x,q_y,q_x]}
{S[r_y,r_x]},
$$

$$
c_x[r_y,r_x]
=\frac{\sum_{q_y,q_x}q_xM[q_y,q_x]I[r_y,r_x,q_y,q_x]}
{S[r_y,r_x]}.
$$

`com_row` is $c_y$ and `com_col` is $c_x$:

$$
(\text{row},\text{column})\equiv(y,x).
$$

In plain terms, `(row, column)` is `(y, x)`.

This explicit naming is required at Python, Swift, CUDA, Metal, and WebGPU
boundaries; a backend may not swap components to match launch coordinates.

## From CoM to DPC and iDPC

The CoM field is centered and rotated into a DPC field
$\mathbf g=(g_y,g_x)$. When automatic rotation is requested, the chosen angle
minimizes the configured curl criterion on the scan-shaped vector field.

Integrated DPC reconstructs a scalar phase-like field in Fourier space. With
scan frequency $\mathbf k=(k_y,k_x)$, a standard least-squares integration is

$$
\hat\phi(\mathbf k)
=\frac{-i\,[k_y\hat g_y(\mathbf k)+k_x\hat g_x(\mathbf k)]}
{k_y^2+k_x^2+\epsilon},
$$

with the zero-frequency value and normalization fixed by the shared contract.
The inverse two-dimensional FFT returns $\phi[r_y,r_x]$.

```python
from quantem.gpu import dpc, io

loaded = io.load("scan_master.h5", backend="auto", det_bin=1)
result = dpc.run(loaded.data)

print(result.com_row.shape, result.com_col.shape)
print(result.rotation_deg, result.use_transpose)
```

## Optimization model

CoM should not require three full detector-volume traversals. A fused moment
kernel accumulates $S$, $\sum q_yI$, and $\sum q_xI$ in one pass, with masks
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
