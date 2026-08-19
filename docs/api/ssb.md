# SSB API

`quantem.gpu.SSB` is the public single-sideband ptychography contract for CUDA,
MPS, and WebGPU workflows. Python `SSB.open()` executes CUDA or MPS; the browser
WebGPU runtime mirrors the same result and precision contract asynchronously.
Backend engines, kernel launch geometry, FFT layouts, and optimizer batching
remain implementation details.

## Inputs and outputs

`SSB.open()` accepts a supported detector source. `SSB.from_array()` accepts an
existing backend-resident detector array. Both require electron voltage,
convergence semiangle, and scan sampling unless those values are available
from trusted source metadata.

`fit()` returns one `SSBResult`. Its primary field is the complex64
`object_wave` with shape `(scan_row, scan_column)`. `phase` and `amplitude` are
derived as `angle(object_wave)` and `abs(object_wave)`. The result also records
the backend, fitted aberrations, rotation, loss, trial/refinement counts,
bright-field geometry, timings, reuse state, and provenance metadata.

## Shapes, coordinates, dtypes, and units

- detector input: `I[scan_row, scan_column, detector_row, detector_column]`;
- complex result: `object_wave[scan_row, scan_column]`, complex64;
- `bf_center`: `(detector_row, detector_column)`;
- `scan_sampling_A`: `(row, column)` when anisotropic, in Å;
- `C10` and `C12`: nm; and
- `phi12`: radians.

The default fit evaluates 200 seeded TPE candidates with the exact full active
bright-field phase-variance objective, chooses the minimum loss, and performs
Nelder–Mead refinement. It does not average optimizer candidates.

## Errors and unsupported requests

- Missing voltage, semiangle, or scan sampling raises rather than inventing
  calibration.
- An unsupported backend or scientific request fails explicitly; SSB never
  falls back silently to CPU.
- `trials` must be non-negative and `refinement` is `"nelder-mead"` or `None`.
- When an in-memory array uses saved-result reuse, provide `source_path` so a
  different same-shaped array cannot match the original source accidentally.

## Provenance and exact reuse

`save_to` writes the complex object and a readable signature containing source
identity, source and output shapes/dtypes, calibration, physical parameters,
optimizer settings/history, bright-field geometry, loss, timings, backend,
package source identity, and Git revision. An exact signature match reopens the
saved result. Any scientific mismatch recomputes instead of reusing stale
output. Inspect `result.reused`, `result.saved_path`, and `result.metadata`.

## Minimal fit

```python
from quantem.gpu import SSB

with SSB.open(
    "scan_master.h5",
    backend="mps",
    voltage_kV=300,
    semiangle_mrad=30,
    scan_sampling_A=(0.264, 0.264),
) as workflow:
    result = workflow.fit(save_to="results/ssb")
```

Use `reconstruct()` when aberrations are known and no optimizer should run:

```python
result = workflow.reconstruct(
    {"C10": 12.5, "C12": 3.0, "phi12": 0.25},
    save_to="results/fixed-ssb",
)
```

`preview()` accepts the same complete aberration mapping and returns a
transient phase array plus an optional exact loss. It does not create a second
public result type.

## Series reconstruction

`SSB.reconstruct_series()` discovers numbered acquisitions in one directory
and returns one `SSBSeriesResult`. Inclusive `first_frame` and `last_frame`
values select acquisition identifiers, not positional array indices. Set
`probe_reference_frame` to reuse one fitted probe; omit it to fit each
acquisition independently. The result retains the phase, bright-field and
dark-field stacks, frame/dataset identifiers, source/results directories,
backend request, optimizer settings, and per-acquisition records.

## Integration boundary

QuantEM.GPU owns SSB preparation, the exact objective, optimization,
reconstruction, typed results, persistence signatures, and backend parity. A
consuming application owns controls, progress presentation, cache scheduling,
and visualization. It must present approximate previews as previews and must
not promote them to exact calibration evidence.

See [Single-sideband ptychography](../kernels/ssb.md) for the complete
mathematics and [SSB performance evidence](../maintainer/ssb-performance.md)
for dated, revision- and device-qualified measurements. Performance numbers do
not live in this API contract because a new benchmark must not silently change
API semantics.
