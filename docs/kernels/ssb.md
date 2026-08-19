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

The PyTorch blocks below form one progressive reference: run them from top to
bottom. Each equation introduces an operation, and the code immediately below
defines that operation before a later block calls it.

```python
from typing import NamedTuple

import torch

Tensor = torch.Tensor
```

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

The corresponding code starts with the complete counts and an explicitly
calibrated detector mask. It defines `selected_R_q` with shape
`(R_r, R_c, B)`:

```python
def select_bright_field(
    counts_R_q: Tensor,
    bf_mask_q: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the mean diffraction pattern and selected BF columns."""
    if counts_R_q.ndim != 4:
        raise ValueError("counts_R_q must have shape (R_r, R_c, q_r, q_c)")
    if bf_mask_q.shape != counts_R_q.shape[2:]:
        raise ValueError("bf_mask_q must match the detector shape (q_r, q_c)")

    # Average over scan row and scan column. Detector row and column remain.
    mean_q = counts_R_q.to(torch.float32).mean(dim=(0, 1))

    # A two-dimensional Boolean detector mask becomes one BF-sample axis B.
    selected_R_q = counts_R_q[:, :, bf_mask_q]  # (R_r, R_c, B)
    return mean_q, selected_R_q
```

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

Moving the already-defined bright-field axis first gives the mathematical
$G_b[\mathbf k]$ layout:

```python
def scan_fft(selected_R_q: Tensor) -> Tensor:
    """Transform the two scan axes and return shape (B, R_r, R_c)."""
    # Put the BF-sample axis first. The two scan axes are now last.
    scan_images_b_R = selected_R_q.permute(2, 0, 1)

    # fft2 transforms the last two axes by default: scan row and scan column.
    return torch.fft.fft2(scan_images_b_R)
```

Here $\mathbf k$ indexes **scan frequency**, while $b$ indexes one selected
bright-field detector coordinate. The inverse FFT converts each corrected
$\mathbf k$ plane back to probe/scan position $\mathbf R$ before the variance is
formed. The fit therefore does not search for a minimum variance "among
$\mathbf k$"; it measures phase variance across $b$, averages that variance
over $\mathbf R$, and minimizes the resulting scalar over candidate
$\boldsymbol\theta$.

## 3. Candidate aberration correction

For candidate parameters
$\boldsymbol\theta=(C_{10},C_{12},\phi_{12})$, define the probe transfer
function at detector-frequency coordinate $\mathbf u$ as

$$
P_{\boldsymbol\theta}(\mathbf u)
=A(\mathbf u)\exp[-i\chi_{\boldsymbol\theta}(\mathbf u)],
$$

with

$$
\chi_{\boldsymbol\theta}(\mathbf u)
=\frac{\pi}{\lambda}\,\alpha(\mathbf u)^2
\left[C_{10}+C_{12}\cos 2\left(\phi(\mathbf u)-\phi_{12}\right)\right].
$$

Here $\lambda$ is the electron wavelength, $\alpha(\mathbf u)$ and
$\phi(\mathbf u)$ are calibrated polar coordinates, and $A(\mathbf u)$ is the
soft aperture weight. For bright-field coordinate $\mathbf q_b$, the SSB
overlap is

$$
\Gamma_b(\mathbf k;\boldsymbol\theta)
=P_{\boldsymbol\theta}(\mathbf q_b-\mathbf k)
 P_{\boldsymbol\theta}^{*}(\mathbf q_b)
-P_{\boldsymbol\theta}^{*}(\mathbf q_b+\mathbf k)
 P_{\boldsymbol\theta}(\mathbf q_b).
$$

The phase-only correction and its real-space contribution are

$$
C_b(\mathbf k;\boldsymbol\theta)
=G_b(\mathbf k)
\frac{\Gamma_b^{*}(\mathbf k;\boldsymbol\theta)}
{\max\left(|\Gamma_b(\mathbf k;\boldsymbol\theta)|,\epsilon\right)},
\qquad
O_b(\mathbf R;\boldsymbol\theta)
=\mathcal F^{-1}_{\mathbf k\rightarrow\mathbf R}
\left\{C_b(\mathbf k;\boldsymbol\theta)\right\}.
$$

The implementation treats the DC term explicitly rather than allowing an
undefined phase at $|\Gamma|=0$.

One candidate correction translates directly to PyTorch. The geometry arrays
for $\mathbf q_b$, $\mathbf q_b-\mathbf k$, and
$\mathbf q_b+\mathbf k$ are calibration outputs prepared once. Their type and
shapes are defined before the correction function uses them:

```python
class SSBGeometry(NamedTuple):
    # Each tuple is (alpha_squared, azimuth, aperture).
    q: tuple[Tensor, Tensor, Tensor]          # each tensor: (B,)
    q_minus_k: tuple[Tensor, Tensor, Tensor]  # each tensor: (B, R_r, R_c)
    q_plus_k: tuple[Tensor, Tensor, Tensor]   # each tensor: (B, R_r, R_c)


def probe(
    alpha2: Tensor,
    azimuth: Tensor,
    aperture: Tensor,
    theta: tuple[Tensor, Tensor, Tensor],
    wavelength: Tensor,
) -> Tensor:
    c10, c12, phi12 = theta
    chi = (
        (torch.pi / wavelength)
        * alpha2
        * (c10 + c12 * torch.cos(2 * (azimuth - phi12)))
    )
    return aperture * torch.exp(-1j * chi)


def corrected_object(
    g_b_k: Tensor,
    geometry: SSBGeometry,
    theta: tuple[Tensor, Tensor, Tensor],
    wavelength: Tensor,
    dc_value: Tensor,
) -> Tensor:
    """Return O_b(R; theta) with shape (B, R_r, R_c)."""
    p_q = probe(*geometry.q, theta, wavelength)[:, None, None]
    p_minus = probe(*geometry.q_minus_k, theta, wavelength)
    p_plus = probe(*geometry.q_plus_k, theta, wavelength)

    gamma_b_k = p_minus * p_q.conj() - p_plus.conj() * p_q
    unit_gamma = gamma_b_k / gamma_b_k.abs().clamp_min(1e-8)
    corrected_b_k = g_b_k * unit_gamma.conj()

    # The overlap phase is undefined at zero scan frequency, so retain the
    # explicitly prepared DC value instead of dividing by zero.
    corrected_b_k[:, 0, 0] = dc_value

    # ifft2 again uses the last two axes, converting (k_r, k_c) to (R_r, R_c).
    return torch.fft.ifft2(corrected_b_k)
```

Here `geometry.q` contains `(alpha2, azimuth, aperture)` arrays with shape
`(B,)`; `q_minus_k` and `q_plus_k` contain the broadcast geometry with shape
`(B, R_r, R_c)`. The prepared `g_b_k` and geometry stay resident. A candidate
changes only `theta`, probe phases, the normalized overlap, and the inverse
transform.

## 4. Exact phase-variance objective

Let

$$
\phi_b[\mathbf R;\boldsymbol\theta]
=\arg O_b[\mathbf R;\boldsymbol\theta],
\qquad
\bar\phi[\mathbf R;\boldsymbol\theta]
=\frac{1}{B}\sum_{b\in\mathcal B}\phi_b[\mathbf R;\boldsymbol\theta].
$$

The variance at one scan position and the scalar fit loss are

$$
V[\mathbf R;\boldsymbol\theta]
=\frac{1}{B}\sum_{b\in\mathcal B}\phi_b^2
-\bar\phi^2,
\qquad
L(\boldsymbol\theta)
=\frac{1}{N_R}\sum_{\mathbf R}V[\mathbf R;\boldsymbol\theta].
$$

The reduction axes are equally explicit in PyTorch:

```python
def phase_variance_loss(object_b_R: Tensor) -> Tensor:
    """Return the scalar phase-variance loss L(theta)."""
    phi_b_R = torch.angle(object_b_R)       # (B, R_r, R_c)
    phi_R = phi_b_R.mean(dim=0)             # mean over bright-field samples
    variance_R = (
        phi_b_R.square().mean(dim=0)
        - phi_R.square()
    )
    return variance_R.mean()                 # mean over scan positions
```

All quantities needed to evaluate one optimizer candidate are now defined, so
the complete candidate function is short:

```python
def evaluate_candidate(
    selected_R_q: Tensor,
    geometry: SSBGeometry,
    theta: tuple[Tensor, Tensor, Tensor],
    wavelength: Tensor,
    dc_value: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return scalar loss and O_b(R; theta) for one candidate."""
    g_b_k = scan_fft(selected_R_q)
    object_b_R = corrected_object(
        g_b_k,
        geometry,
        theta,
        wavelength,
        dc_value,
    )
    return phase_variance_loss(object_b_R), object_b_R
```

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

TPE proposes candidates sequentially, but the selection rule itself is simply:

```python
def best_tpe_candidate(
    theta_candidates: Tensor,
    candidate_losses: Tensor,
) -> Tensor:
    """Select the parameter vector with minimum loss; never average trials."""
    if theta_candidates.shape != (200, 3):
        raise ValueError("theta_candidates must have shape (200, 3)")
    if candidate_losses.shape != (200,):
        raise ValueError("candidate_losses must have shape (200,)")
    best_trial_index = torch.argmin(candidate_losses)
    return theta_candidates[best_trial_index]
```

The returned tensor is $\boldsymbol\theta_{\mathrm{TPE}}$ and seeds the local
Nelder-Mead refinement. There is no mean over the 200 candidates. The mean
operations are only over bright-field samples and scan positions inside
`phase_variance_loss`.

Thus the "second step" after the 200 trials is a local refinement beginning at
the best trial, not another average. With the final parameters, the
complex transmission function is

$$
O[\mathbf R]
=\frac{1}{B}\sum_{b\in\mathcal B}
O_b[\mathbf R;\hat{\boldsymbol\theta}].
$$

The public result stores this complex64 object wave. Its displayed phase is
$\phi=\arg O$ and its amplitude is $|O|$. The optimizer's $\bar\phi$ is part of
the variance objective; it is not substituted for the final complex-object
phase.

The final reduction is also ordinary array code:

```python
def final_object(object_b_R: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return complex object, amplitude, and ordinary phase phi."""
    object_R = object_b_R.mean(dim=0)  # (R_r, R_c), complex64
    amplitude_R = torch.abs(object_R)
    phi_R = torch.angle(object_R)
    return object_R, amplitude_R, phi_R
```

## Why the production kernels are more elaborate

The PyTorch expressions are the readable array specification, not a maintained
production runtime. CUDA, MPS, and WebGPU may fuse candidate correction, phase
moments, and loss accumulation; chunk the $B$ axis; reuse FFT plans and
buffers; and avoid materializing every $O_b[\mathbf R]$. Parity is still judged
against the same equations, reduction axes, dtype, normalization, and complete
selected bright-field evidence.

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
