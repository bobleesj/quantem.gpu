# Device, optics, screening, and parallax API

This page completes the Python API map for public namespaces that do not need a
full operation-specific reference page. All coordinates exposed to users are
`(row, column)`.

## Device selection

| Call | Purpose | Failure behavior |
|---|---|---|
| `device.profile(device=None)` | notebook-friendly environment and selected-device summary | automatic selection may report CPU for diagnostics |
| `device.detect()` | choose a native CUDA or MPS accelerator | raises when neither GPU runtime is available |
| `device.resolve(name="auto")` | validate `cuda`, `mps`, or explicit browser `webgpu` | raises on an unavailable or unknown runtime |

CPU reported by `profile()` is diagnostic. Scientific GPU calls do not silently
turn that diagnostic result into a CPU execution path.

## Electron optics

`quantem.gpu.optics` exports the maintained physics and aberration helpers:

- `wavelength_A_from_kV`, `wavelength_m_from_kV`, and
  `keV_to_wavelength_nm`;
- `convergence_angle_to_k_max`;
- `chi_polar` and `chi_cartesian`; and
- `fit_aberrations`, `compute_shifts_from_aberrations`, and
  `AberrationFitter`.

Public voltage arguments are in kV/keV as named, convergence semi-angle is in
mrad, wavelength output uses the unit named by the function, and reciprocal
cutoff is in Å⁻¹. Aberration coefficient ordering is defined by
`ABERRATION_INDICES`; applications must persist that ordering and the beam
energy with fitted values.

## Screening products

```python
from quantem.gpu import screening

products = screening.prepare(
    "scan_master.h5",
    backend="auto",
    scan_shape=(512, 512),
    memory_budget_gb=6.0,
)
```

`screening.prepare` builds or reopens derived BF/DF/CoM/rotation products and
returns `ScreeningResult`. Its public controls include `scan_shape`, optional
rotation, cache/refresh policy, explicit memory budget, chunk rows, sampling
seed/count, rotation search steps, and dtype. A cache reopen is not a raw HDF5
load. The raw source remains the scientific evidence source, and metadata must
retain the source fingerprint, parameters, timing, and memory plan.

## Parallax reconstruction

```python
from quantem.gpu import parallax

result = parallax.run(
    data,
    scan_shape=(512, 512),
    backend="cuda",
    center=(96, 96),
    voltage_kV=200,
)
```

`parallax.run` returns `ParallaxResult`. The maintained optimized
implementation is CUDA-only; any other selected backend raises explicitly.
The detector center is `(row, column)`. Persist scan shape, center, BF and
sampling radii, accelerating voltage, scan sampling, upsampling factor,
aberration-fit choice, backend, and package revision with the result.

## Integration boundary

QuantEM.GPU owns these typed calculations, validation, backend selection, and
scientific provenance. A consuming application owns presentation, user-facing
policy, cache scheduling, and lifecycle. Private compute modules are not a
substitute for these public entry points.
