# What do the main 4D-STEM products measure?

A 4D-STEM dataset records a diffraction pattern at every probe position. Before
choosing an algorithm, it helps to see how the four axes become an image, a
vector field, or a reconstructed phase.

## What are the four coordinates?

Let

$$
I(\mathbf r,\mathbf q)
= I(r_y,r_x,q_y,q_x)
$$

be the measured detector counts. The package uses `(row, column)` order:

- $\mathbf r=(r_y,r_x)$ is the **scan coordinate** in real space; and
- $\mathbf q=(q_y,q_x)$ is the **detector coordinate** in diffraction space.

One scan position $\mathbf r$ therefore selects one two-dimensional diffraction
pattern $I(\mathbf r,\mathbf q)$. A full scan preserves every $\mathbf r$; no
real-space crop is introduced as a performance shortcut.

## What are the notation and units?

The equations use the following quantities consistently:

| Symbol | Meaning | Shape or units |
|---|---|---|
| $I(\mathbf r,\mathbf q)$ | measured detector signal | scalar detector counts at one scan and detector coordinate |
| $\mathbf r=(r_y,r_x)$ | scan coordinate in `(row, col)` order | scan pixels, or calibrated length such as nm |
| $\mathbf q=(q_y,q_x)$ | detector coordinate in `(row, col)` order | detector pixels, reciprocal length, or calibrated angle such as mrad |
| $M(\mathbf q)$ | detector-selection weight | dimensionless scalar mask |
| $N_r$ | number of included scan positions | dimensionless integer |
| $\mathbf k$ | spatial frequency conjugate to $\mathbf r$ | inverse scan pixels, or inverse calibrated length |

The selected unit and calibration are part of result provenance. An uncalibrated
detector index is not reported as mrad, and an uncalibrated scan index is not
reported as nm. Detector counts remain counts unless an explicitly documented
normalization changes them.

## How does a detector region become an image?

Choose a detector mask $M(\mathbf q)$. Summing the selected detector pixels at
each scan position gives a virtual image:

$$
V_M(\mathbf r)
= \sum_{\mathbf q} M(\mathbf q)I(\mathbf r,\mathbf q).
$$

The mask determines the contrast:

| Product | Detector selection | What it emphasizes |
|---|---|---|
| BF | central disk | intensity remaining near the transmitted beam |
| DF | off-axis or outside-BF region | intensity scattered away from the central disk |
| ADF | annulus $q_{\min}\leq |\mathbf q-\mathbf q_0|<q_{\max}$ | scattering within a chosen angular range |

The mean diffraction pattern answers the complementary question—what detector
signal is typical across the scan:

$$
\overline I(\mathbf q)
= \frac{1}{N_r}\sum_{\mathbf r}I(\mathbf r,\mathbf q).
$$

## Where do CoM and DPC come from?

The center of mass measures the intensity-weighted displacement of each
diffraction pattern from a reference center $\mathbf q_0$:

$$
\mathbf c(\mathbf r)
=
\frac{\sum_{\mathbf q}(\mathbf q-\mathbf q_0)
M(\mathbf q)I(\mathbf r,\mathbf q)}
{\sum_{\mathbf q}M(\mathbf q)I(\mathbf r,\mathbf q)}.
$$

`com_row` is the $q_y$ component and `com_col` is the $q_x$ component. After
detector-to-scan rotation, centering, and physical calibration, this vector
field becomes the DPC signal $\mathbf g(\mathbf r)$. Under the appropriate
thin-sample and weak-phase assumptions, $\mathbf g$ is proportional to a
projected phase gradient or electric-field signal; it is not automatically a
direct field measurement for every specimen.

## What does iDPC add?

iDPC asks for the scalar field whose gradient best matches the measured DPC
vectors:

$$
\phi_{\mathrm{iDPC}}
= \underset{\phi}{\operatorname{argmin}}
\;\|\nabla\phi-\mathbf g\|_2^2.
$$

With the Fourier derivative convention
$\mathcal F\{\nabla\phi\}=i\mathbf k\widehat\phi$, the nonzero spatial
frequencies are integrated as

$$
\widehat\phi(\mathbf k)
= \frac{-i\,\mathbf k\!\cdot\!\widehat{\mathbf g}(\mathbf k)}
{|\mathbf k|^2},
\qquad \mathbf k\neq 0.
$$

The zero frequency is a chosen phase offset. Rotation, transpose convention,
scan calibration, masking, and regularization are therefore part of the result
provenance.

## How is SSB different?

Single-sideband ptychography keeps the coherent interference information that a
detector sum or first moment discards. First Fourier transform the scan axes:

$$
G(\mathbf k,\mathbf q)
= \mathcal F_{\mathbf r\rightarrow\mathbf k}
\{I(\mathbf r,\mathbf q)\}.
$$

For each scan frequency $\mathbf k$, the probe aperture and its shifted copy
overlap over a detector region $\Omega_+(\mathbf k)$. SSB selects one sideband
and combines that overlap with the probe/aberration weighting $W$:

$$
O(\mathbf k)
\propto
\sum_{\mathbf q\in\Omega_+(\mathbf k)}
W(\mathbf k,\mathbf q)G(\mathbf k,\mathbf q),
\qquad
o(\mathbf r)=\mathcal F^{-1}\{O(\mathbf k)\}.
$$

The complex image $o(\mathbf r)$ provides amplitude and phase. Its validity
depends on scan sampling, aperture geometry, aberration calibration, detector
selection, and the reconstruction objective.

## Why do these operations fit a GPU?

| Stage | Independent outputs | Reduction or transform |
|---|---|---|
| BF/DF/ADF | scan positions $\mathbf r$ | sum selected detector pixels $\mathbf q$ |
| CoM | scan positions $\mathbf r$ | detector intensity and first moments |
| DPC/iDPC | scan-frequency pixels $\mathbf k$ | rotate, differentiate, and integrate a small vector field |
| SSB | scan frequencies $\mathbf k$ | FFT scan axes and combine detector overlap pixels |

The large detector reductions are parallel across scan positions, while FFTs
and overlap sums are parallel across frequencies. `quantem.gpu` implements
these reusable operations on CUDA, MPS/Metal, Swift/Metal, and WebGPU while
preserving the same coordinates, masks, precision, and provenance.

Continue with [BF/DF/ADF](bf_df_adf.md), [CoM/DPC/iDPC](dpc.md), or
[SSB](ssb.md) for the corresponding public workflow.
