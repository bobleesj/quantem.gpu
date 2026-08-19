# Single-sideband ptychography

Single-sideband (SSB) ptychography uses the scan-frequency information in a
4D-STEM acquisition to reconstruct a complex object. The input convention is

$$
I[r_y,r_x,q_y,q_x],
\qquad (\text{row},\text{column})\equiv(y,x),
$$

with scan coordinate $\mathbf r=(r_y,r_x)$ and detector coordinate
$\mathbf q=(q_y,q_x)$.

After a two-dimensional FFT over scan coordinates, the data are represented
schematically as $G(\mathbf q,\mathbf k)$, where
$\mathbf k=(k_y,k_x)$ is scan spatial frequency. Bright-field detector
positions whose aperture-overlap terms transfer information at $\mathbf k$
contribute to the object estimate. The exact aperture, aberration phase,
normalization, and loss are defined by the shared SSB contract.

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

- scan sampling is ordered `(row, column) ≡ (y, x)` and carries length units;
- detector angles are ordered $(q_y,q_x)$ and carry calibrated angle or
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
