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

(screening-products)=
## Screening products

```python
from quantem.gpu import screening

products = screening.prepare(
    "scan_master.h5",
    backend="auto",
    scan_shape=(512, 512),
    memory_budget_gb=4.0,
)

print(products.metadata["memory"])
```

`screening.prepare` builds or reopens derived detector and DPC products and
returns `ScreeningResult`. Its public controls include `scan_shape`, optional
rotation, cache/refresh policy, explicit memory budget, chunk rows, sampling
seed/count, rotation search steps, and dtype. A cache reopen is not a raw HDF5
load. The raw source remains the scientific evidence source, and metadata must
retain the source fingerprint, parameters, timing, and memory plan.

`memory_budget_gb=4.0` requests a bounded streaming working set; it does not
promise that a complete unbinned 4D stack will fit in 4 GiB. For a native
`512x512x192x192 uint16` source, the current planner selects 56 scan rows per
chunk: **1.97 GiB** of raw counts across 10 sequential chunks. A 6.0 budget
selects 85 rows: **2.99 GiB** across 7 chunks. Decoder scratch, allocator
reserve, other processes, and the final measured peak remain separate gates.
The historical API name says `gb`, while the planner uses binary GiB bytes.

Every current preparation path publishes mean DP, BF, DF, CoM, rotation, and
iDPC. The default exact `uint16` CUDA stream also publishes
`total_intensity`, `annular_bright_field`, and `annular_dark_field` as exact
`uint64` count maps. Those three optional fields are `None` for older caches
and preparation paths that do not yet expose the fused exact statistics; a
caller must check them explicitly rather than infer support from the selected
backend. New caches contain either all three exact maps or none, and malformed
partial or non-`uint64` sets fail closed. No automatic scan crop is allowed,
and a client-selected detector bin must remain explicit in provenance.

`screening.prepare` is currently a Python CUDA/MPS API. Native Swift/Metal and
WebGPU expose reusable detector and DPC operations, but they do not implement
this prepared-product cache contract. Applications must not label an
independently assembled native or browser product set as a
`ScreeningResult`.

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
