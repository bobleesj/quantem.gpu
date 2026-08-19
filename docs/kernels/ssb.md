# Single-sideband ptychography

Single-sideband (SSB) ptychography uses scan-frequency interference from a
4D-STEM acquisition to reconstruct a complex transmission function. The full
default fit is:

```text
4D counts → bright-field selection → scan FFT → aberration correction
          → per-BF phase maps → phase-variance loss
          → 200 TPE trials → Nelder-Mead refinement
          → best aberrations → final complex object wave
```

The `200` is the default number of global Optuna TPE candidates. Each candidate
is scored with the same full active-bright-field phase-variance objective. The
best candidate then seeds a local Nelder-Mead refinement. The workflow selects
the minimum recorded loss; it does not average the 200 parameter sets.

## 1. Data and bright-field evidence

The input convention is

$$
I[R_r,R_c,q_r,q_c],
\qquad (\text{row},\text{column})\equiv(r,c),
$$

with real-space probe/scan coordinate $\mathbf R=(R_r,R_c)$ and detector
scattering coordinate $\mathbf q=(q_r,q_c)$.

The mean diffraction pattern is

$$
\bar I[\mathbf q]
=\frac{1}{N_R}\sum_{\mathbf R}I[\mathbf R,\mathbf q].
$$

The calibrated bright-field disk defines a set $\mathcal B$ of $B$ detector
coordinates. The default uses every active coordinate in $\mathcal B$; it does
not silently subsample this evidence for fitting.

## 2. Scan Fourier transform

For each selected bright-field coordinate $\mathbf q_b$, transform over the
two scan axes:

$$
G_b[\mathbf k]
=\mathcal F_{\mathbf R\rightarrow\mathbf k}
\{I[\mathbf R,\mathbf q_b]\},
\qquad \mathbf k=(k_r,k_c).
$$

The prepared $G_b[\mathbf k]$ columns, bright-field indices, aperture geometry,
and FFT plans remain resident and are reused across optimizer candidates.

Here $\mathbf k$ indexes **scan frequency**, while $b$ indexes one selected
bright-field detector coordinate. The inverse FFT converts each corrected
$\mathbf k$ plane back to probe/scan position $\mathbf R$ before the variance is
formed. The fit therefore does not search for a minimum variance "among
$\mathbf k$"; it measures phase variance across $b$, averages that variance
over $\mathbf R$, and minimizes the resulting scalar over candidate
$\boldsymbol\theta$.

## 3. Candidate aberration correction

For candidate parameters
$\boldsymbol\theta=(C_{10},C_{12},\phi_{12})$, the hot path evaluates

$$
\chi_b(\boldsymbol\theta)
=\frac{2\pi}{\lambda}\,\alpha_b^2
\left[C_{10}+C_{12}\cos 2(\phi_b-\phi_{12})\right],
$$

and forms the complex probe factor

$$
P_b(\boldsymbol\theta)
=A_b\exp\!\left[-i\chi_b(\boldsymbol\theta)\right].
$$

Here $\lambda$ is the electron wavelength, $\alpha_b$ and $\phi_b$ are the
calibrated polar coordinates of bright-field sample $b$, and $A_b$ is its soft
aperture weight. The corrected per-bright-field object contribution is

$$
O_b[\mathbf R;\boldsymbol\theta]
=\mathcal F^{-1}_{\mathbf k\rightarrow\mathbf R}
\{G_b[\mathbf k]P_b(\boldsymbol\theta)\}.
$$

## 4. Exact phase-variance objective

Let

$$
\varphi_b[\mathbf R;\boldsymbol\theta]
=\arg O_b[\mathbf R;\boldsymbol\theta],
\qquad
\bar\varphi[\mathbf R;\boldsymbol\theta]
=\frac{1}{B}\sum_{b\in\mathcal B}\varphi_b[\mathbf R;\boldsymbol\theta].
$$

The variance at one scan position and the scalar fit loss are

$$
V[\mathbf R;\boldsymbol\theta]
=\frac{1}{B}\sum_{b\in\mathcal B}\varphi_b^2
-\bar\varphi^2,
\qquad
L(\boldsymbol\theta)
=\frac{1}{N_R}\sum_{\mathbf R}V[\mathbf R;\boldsymbol\theta].
$$

This is the two-stage mean the implementation computes: moments across all
active bright-field contributions, followed by the mean spatial variance over
the scan. These are scientific reduction axes, not averages over optimizer
trials. The best candidate is

$$
\hat{\boldsymbol\theta}
=\operatorname*{arg\,min}_{\boldsymbol\theta}L(\boldsymbol\theta).
$$

## 5. Default optimization and final result

The default fit evaluates 200 seeded TPE candidates, chooses the lowest-loss
candidate, and refines it with Nelder-Mead. In compact form,

$$
\left\{\boldsymbol\theta_j,L_j\right\}_{j=1}^{200}
\xrightarrow{\operatorname*{arg\,min}_j L_j}
\boldsymbol\theta_{\mathrm{TPE}}
\xrightarrow{\mathrm{Nelder\text{-}Mead}
}
\hat{\boldsymbol\theta}.
$$

Thus the "second step" after the 200 trials is a local refinement beginning at
the best trial, not another average. With the final parameters, the
complex transmission function is

$$
O[\mathbf R]
=\frac{1}{B}\sum_{b\in\mathcal B}
O_b[\mathbf R;\hat{\boldsymbol\theta}].
$$

The public result stores this complex64 object wave. Its displayed phase is
$\arg O$ and its amplitude is $|O|$. The optimizer's $\bar\varphi$ is part of
the variance objective; it is not substituted for the final complex-object
phase.

## Reference array expressions

These short expressions explain the shared mathematics; they are not a second
production backend. NumPy notation is:

```python
phase_b = np.angle(np.fft.ifft2(corrected_b_k, axes=(-2, -1)))
mean_phase = phase_b.mean(axis=0)
variance = np.square(phase_b).mean(axis=0) - np.square(mean_phase)
loss = variance.mean()
object_wave = np.fft.ifft2(corrected_b_k.mean(axis=0), axes=(-2, -1))
```

The equivalent PyTorch expression is:

```python
phase_b = torch.angle(torch.fft.ifft2(corrected_b_k, dim=(-2, -1)))
mean_phase = phase_b.mean(dim=0)
variance = phase_b.square().mean(dim=0) - mean_phase.square()
loss = variance.mean()
object_wave = torch.fft.ifft2(corrected_b_k.mean(dim=0), dim=(-2, -1))
```

CUDA, MPS, and WebGPU may fuse and chunk these operations, but parity is judged
against the same definitions and full selected bright-field evidence.

## Public workflow

```python
from quantem.gpu import SSB

workflow = SSB.open(
    "scan_master.h5",
    backend="mps",
    voltage_kV=300,
    semiangle_mrad=21.4,
    scan_sampling_A=0.5,
)
result = workflow.fit(save_to="results/ssb")
```

For known aberrations, reconstruct without fitting:

```python
result = workflow.reconstruct(
    {"C10": 12.5, "C12": 3.0, "phi12": 0.25},
    save_to="results/fixed-ssb",
)
```

## Optimization model

SSB performance is governed by data preparation, FFT layout, active
bright-field count, phase evaluation, and optimizer trial scheduling. Reusable
optimizations include:

- keeping prepared bright-field columns and $G(\mathbf q,\mathbf k)$ on device;
- using backend-qualified FFT layouts without changing normalization;
- fusing phase/object/loss work when the same intermediates are consumed;
- batching aberration trials without duplicating the prepared source;
- reusing twiddles, aperture geometry, masks, and compiled pipelines; and
- separating first preparation, warm evaluation, optimization, and saved-result
  reopen in benchmarks.

An approximate preview is not calibration evidence. A fitted result is reused
only when source identity, detector selection, calibration, backend, physical
parameters, precision, and optimizer settings match.

## Coordinate and unit checks

- scan sampling is ordered `(row, column) ≡ (r, c)` and carries length units;
- detector angles are ordered $(q_r,q_c)$ and carry calibrated angle or
  reciprocal-length units;
- aberration coefficients and angles use the documented public units; and
- any transpose or Hermitian storage is private and reversed before producing
  the public result.

## Source map and gates

| Layer | Source |
|---|---|
| Public workflow/results | `src/quantem/gpu/ssb` |
| CUDA engine and optimizer | `src/quantem/gpu/ssb/compute/cuda` |
| Python MPS engine and optimizer | `src/quantem/gpu/ssb/compute/mps` |
| WebGPU kernels | `src/quantem/gpu/ssb/compute/webgpu` |

Parity uses the same source, bright-field selection, physical calibration,
aberrations, precision, and objective. Reports include complex-object or phase
error maps, full-BF loss, fitted parameters, preparation/evaluation/fit times,
active BF count, memory peak, and device/kernel revision.
