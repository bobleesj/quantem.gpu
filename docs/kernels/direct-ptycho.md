# Direct ptychography

Single-sideband (SSB) ptychography uses scan-frequency interference from a
4D-STEM acquisition to reconstruct a complex transmission function. This page
derives the reconstruction for given aberrations and the phase-variance
diagnostic that every backend reports as `result.loss`:

```text
4D counts → bright-field selection → scan FFT → aberration correction
          → per-BF corrected waves → phase of the mean wave (SSB phase)
                                   → phase-variance diagnostic (result.loss)
```

The aberrations are fitted with the SSB model instead:
`find_aberrations()` maximizes the least-squares agreement of the model with G
over 200 seeded TPE trials, fits the sample tilt and depth spread, and refines
on exact derivatives of the agreement (a trust-region Newton on the exact
gradient and Hessian at low order, a bounded quasi-Newton on exact gradients
through fifth order); `search="primary", tilt=False` holds the tilt and depth
spread at 0 ({ref}`fit methods <ssb-fit-methods>`). The fit selects the best
candidate and refines it; it never averages parameter sets. The phase-variance
search (`objective="phase_variance"`, 200 TPE trials and Nelder–Mead) that this
page once derived is retired.

The PyTorch blocks below form one progressive reference: run them from top to
bottom. Each equation introduces an operation, and the code immediately below
defines that operation before a later block calls it.

```{admonition} Executable reference, not copied production kernels
:class: note
These are real PyTorch function definitions, not pseudocode. They execute when
given the tensors and calibrated geometry defined on this page, and they express
the same equations and reduction axes used for parity. They are not the source
of the optimized CUDA, Python MPS, native Swift/Metal, or WebGPU runtimes; those
implementations may fuse or stream the same work and are listed in **Source map
and gates** below.
```

```python
from typing import NamedTuple

import torch

Tensor = torch.Tensor
```

## 1. Data and bright-field evidence

The input convention is

$$
I[R_r,R_c,k_r,k_c],
\qquad (\text{row},\text{column})\equiv(r,c),
$$

with real-space probe/scan coordinate $\mathbf R=(R_r,R_c)$ and detector
scattering coordinate $\mathbf k=(k_r,k_c)$.

The mean diffraction pattern is

$$
\bar I[\mathbf k]
=\frac{1}{N_R}\sum_{\mathbf R}I[\mathbf R,\mathbf k].
$$

The calibrated bright-field disk defines a set $\mathcal B$ of $B$ detector
coordinates. The default uses every active coordinate in $\mathcal B$; it does
not silently subsample this evidence for fitting.

The corresponding code starts with the complete counts and an explicitly
calibrated detector mask. It defines `selected_R_k` with shape
`(R_r, R_c, B)`:

```python
def select_bright_field(
    counts_R_k: Tensor,
    bf_mask_k: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the mean diffraction pattern and selected BF columns."""
    if counts_R_k.ndim != 4:
        raise ValueError("counts_R_k must have shape (R_r, R_c, k_r, k_c)")
    if bf_mask_k.shape != counts_R_k.shape[2:]:
        raise ValueError("bf_mask_k must match the detector shape (k_r, k_c)")

    # Average over scan row and scan column. Detector row and column remain.
    mean_k = counts_R_k.to(torch.float32).mean(dim=(0, 1))

    # A two-dimensional Boolean detector mask becomes one BF-sample axis B.
    selected_R_k = counts_R_k[:, :, bf_mask_k]  # (R_r, R_c, B)
    return mean_k, selected_R_k
```

## 2. Scan Fourier transform

For each selected bright-field coordinate $\mathbf k_b$, transform over the
two scan axes:

$$
G_b[\boldsymbol{\nu}]
=\mathcal F_{\mathbf R\rightarrow\boldsymbol{\nu}}
\{I[\mathbf R,\mathbf k_b]\},
\qquad \boldsymbol{\nu}=(\nu_r,\nu_c).
$$

The prepared $G_b[\boldsymbol{\nu}]$ columns, bright-field indices, aperture geometry,
and FFT plans remain resident and are reused across optimizer candidates.

Moving the already-defined bright-field axis first gives the mathematical
$G_b[\boldsymbol{\nu}]$ layout:

```python
def scan_fft(selected_R_k: Tensor) -> Tensor:
    """Transform the two scan axes and return shape (B, R_r, R_c)."""
    # Put the BF-sample axis first. The two scan axes are now last.
    scan_images_b_R = selected_R_k.permute(2, 0, 1)

    # fft2 transforms the last two axes by default: scan row and scan column.
    return torch.fft.fft2(scan_images_b_R)
```

Here $\boldsymbol{\nu}$ indexes **scan frequency**, while $b$ indexes one selected
bright-field detector coordinate. The inverse FFT converts each corrected
$\boldsymbol{\nu}$ plane back to probe/scan position $\mathbf R$ before the variance is
formed. The diagnostic is therefore not a variance "among
$\boldsymbol{\nu}$"; it measures phase variance across $b$ and averages that
variance over $\mathbf R$ for the given parameters $\boldsymbol\theta$.

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
soft aperture weight. For bright-field coordinate $\mathbf k_b$, the SSB
overlap is

$$
\Gamma_b(\boldsymbol{\nu};\boldsymbol\theta)
=P_{\boldsymbol\theta}(\mathbf k_b-\boldsymbol{\nu})
 P_{\boldsymbol\theta}^{*}(\mathbf k_b)
-P_{\boldsymbol\theta}^{*}(\mathbf k_b+\boldsymbol{\nu})
 P_{\boldsymbol\theta}(\mathbf k_b).
$$

The phase-only correction and its real-space contribution are

$$
C_b(\boldsymbol{\nu};\boldsymbol\theta)
=G_b(\boldsymbol{\nu})
\frac{\Gamma_b^{*}(\boldsymbol{\nu};\boldsymbol\theta)}
{\max\left(|\Gamma_b(\boldsymbol{\nu};\boldsymbol\theta)|,\epsilon\right)},
\qquad
O_b(\mathbf R;\boldsymbol\theta)
=\mathcal F^{-1}_{\boldsymbol{\nu}\rightarrow\mathbf R}
\left\{C_b(\boldsymbol{\nu};\boldsymbol\theta)\right\}.
$$

The implementation treats the DC term explicitly rather than allowing an
undefined phase at $|\Gamma|=0$.

One candidate correction translates directly to PyTorch. The geometry arrays
for $\mathbf k_b$, $\mathbf k_b-\boldsymbol{\nu}$, and
$\mathbf k_b+\boldsymbol{\nu}$ are calibration outputs prepared once. Their type and
shapes are defined before the correction function uses them:

```python
class SSBGeometry(NamedTuple):
    # Each tuple is (alpha_squared, azimuth, aperture).
    k: tuple[Tensor, Tensor, Tensor]             # each tensor: (B,)
    k_minus_nu: tuple[Tensor, Tensor, Tensor]    # each tensor: (B, R_r, R_c)
    k_plus_nu: tuple[Tensor, Tensor, Tensor]     # each tensor: (B, R_r, R_c)


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
    g_b_nu: Tensor,
    geometry: SSBGeometry,
    theta: tuple[Tensor, Tensor, Tensor],
    wavelength: Tensor,
    dc_value: Tensor,
) -> Tensor:
    """Return O_b(R; theta) with shape (B, R_r, R_c)."""
    p_k = probe(*geometry.k, theta, wavelength)[:, None, None]
    p_minus = probe(*geometry.k_minus_nu, theta, wavelength)
    p_plus = probe(*geometry.k_plus_nu, theta, wavelength)

    gamma_b_nu = p_minus * p_k.conj() - p_plus.conj() * p_k
    unit_gamma = gamma_b_nu / gamma_b_nu.abs().clamp_min(1e-8)
    corrected_b_nu = g_b_nu * unit_gamma.conj()

    # The overlap phase is undefined at zero scan frequency, so retain the
    # explicitly prepared DC value instead of dividing by zero.
    corrected_b_nu[:, 0, 0] = dc_value

    # ifft2 again uses the last two axes, converting (\nu_r, \nu_c) to (R_r, R_c).
    return torch.fft.ifft2(corrected_b_nu)
```

Here `geometry.k` contains `(alpha2, azimuth, aperture)` arrays with shape
`(B,)`; `k_minus_nu` and `k_plus_nu` contain the broadcast geometry with shape
`(B, R_r, R_c)`. The prepared `g_b_nu` and geometry stay resident. A candidate
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

The variance at one scan position and the scalar diagnostic loss are

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

All quantities needed to evaluate one parameter set are now defined, so the
complete evaluation is short:

```python
def evaluate_candidate(
    selected_R_k: Tensor,
    geometry: SSBGeometry,
    theta: tuple[Tensor, Tensor, Tensor],
    wavelength: Tensor,
    dc_value: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return scalar loss and O_b(R; theta) for one candidate."""
    g_b_nu = scan_fft(selected_R_k)
    object_b_R = corrected_object(
        g_b_nu,
        geometry,
        theta,
        wavelength,
        dc_value,
    )
    return phase_variance_loss(object_b_R), object_b_R
```

This is the two-stage mean the implementation computes: moments across all
active bright-field contributions, followed by the mean spatial variance over
the scan. These are scientific reduction axes, not averages over fit trials.

## 5. Final result

With the fitted parameters $\hat{\boldsymbol\theta}$, the complex
transmission function is

$$
O[\mathbf R]
=\frac{1}{B}\sum_{b\in\mathcal B}
O_b[\mathbf R;\hat{\boldsymbol\theta}].
$$

SSB keeps its phase, $\phi=\arg O$ (`phase_of_mean`), on CUDA and MPS, and the
public result stores a phase-only wave with unit amplitude, recorded as
`amplitude_estimated=False`. Amplitude is the regularized `method="complex"`'s.
The diagnostic's $\bar\phi$, the mean of the per-contribution phases
$\arg O_b$, is part of the variance; it is not substituted for the final
phase. (The `mean_phase` estimator, which displayed $\bar\phi$, and the
`complex_wave` estimator, which stored $O$ itself at 1x, were retired on
2026-10-25.)

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
from quantem.gpu import DirectPtycho

workflow = DirectPtycho.open(
    "scan_master.h5",
    backend="mps",
    voltage_kV=300,
    semiangle_mrad=21.4,
    scan_sampling_A=(0.5, 0.5),
)
result = workflow.find_aberrations(save_dir="results/ssb")
```

For known aberrations, reconstruct without fitting:

```python
result = workflow.run(
    aberrations={"C10_nm": 12.5, "C12_nm": 3.0, "phi12_deg": 15.0},
    save_dir="results/fixed-ssb",
)
```

## Optimization model

Direct Ptycho performance is governed by data preparation, FFT layout, active
bright-field count, phase evaluation, and optimizer trial scheduling. Reusable
optimizations include:

- keeping prepared bright-field columns and $G(\mathbf k,\boldsymbol{\nu})$ on device;
- using backend-qualified FFT layouts without changing normalization;
- fusing phase/object/loss work when the same intermediates are consumed;
- batching aberration trials without duplicating the prepared source;
- reusing twiddles, aperture geometry, masks, and compiled pipelines; and
- separating first preparation, warm evaluation, optimization, and saved-result
  reopen in benchmarks.

An approximate preview is not calibration evidence. A fitted result is reused
only when source identity, detector selection, calibration, backend, physical
parameters, precision, and optimizer settings match.

### Exact-derivative refinement

The fit's refinements use exact derivatives of the agreement on CUDA and MPS
(`cuda/exact_derivatives.py`, `mps/exact_derivatives.py`). The low-order fit
climbs with a trust-region Newton on the exact gradient and Hessian
(`_exact_newton.py`); the fifth-order fit with a bounded trust-region
quasi-Newton on exact gradients (`_gradient_refinement.py`), started from a
Hessian of every 64th bright-field pixel, first on every 8th pixel and then
converged on all of them. Per-(q, k) terms are float32; sums over
bright-field chunks and band frequencies are float64, and steps are accepted on
the full objective. A broad fit refines from the standard start and from the 8
best distinct TPE candidates and keeps the highest agreement.

[Exact-derivative SSB fit](../performance/ssb/reports/2026-10-25-ssb-exact-derivative-fit.md)
records the speed, the precision against a float64 autograd reference, and the
comparison with the earlier finite-difference refinement.

## Coordinate and unit checks

- scan sampling is ordered `(row, column) ≡ (r, c)` and carries length units;
- detector angles are ordered $(k_r,k_c)$ and carry calibrated angle or
  reciprocal-length units;
- aberration coefficients and angles use the documented public units; and
- any transpose or Hermitian storage is private and reversed before producing
  the public result.

## Source map and gates

| Layer | Source |
|---|---|
| Public workflow/results | `src/quantem/gpu/direct_ptycho` |
| CUDA engine and optimizer | `src/quantem/gpu/direct_ptycho/cuda` |
| Python MPS engine and optimizer | `src/quantem/gpu/direct_ptycho/mps` |
| Native Swift/Metal preparation, joint model, fit and direct methods | `native/swift/Sources/MetalSSBKernels` |
| WebGPU kernels | `src/quantem/gpu/direct_ptycho/webgpu` |

Parity uses the same source, bright-field selection, physical calibration,
aberrations, precision, and objective. Reports include complex-object or phase
error maps, full-BF loss, fitted parameters, preparation/evaluation/fit times,
active BF count, memory peak, and device/kernel revision.

The native Swift/Metal implementation supports 128×128, 256×256 and 512×512
scans with plane-major `uint8`, `uint16` or `uint32` BF columns. Its
complete-cache and bounded-memory streaming modes retain the same logical BF
normalization; 1024×1024 remains an explicit native gap.
