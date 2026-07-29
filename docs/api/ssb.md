# SSB API

quantem.gpu.SSB is the single public SSB compute entry point for CUDA, MPS,
and browser WebGPU workflows. Backend-specific engines and fit helpers are
implementation details.

~~~python
from quantem.gpu import SSB

workflow = SSB.open(
    "scan_master.h5",
    backend="mps",
    voltage_kV=300,
    semiangle_mrad=30,
    scan_sampling_A=(0.264, 0.264),
)
result = workflow.fit(trials=200, refinement="nelder-mead")
~~~

The default fit always uses the automatically detected full bright field and
the exact float32 phase-variance objective. `result` is the single
`SSBResult`: it contains optimizer metadata, `object_wave`, and the derived
`phase` and `amplitude`. A backend that cannot honor the request fails
explicitly.

Use `reconstruct()` when aberrations are already known and no optimizer should
run:

~~~python
fixed = workflow.reconstruct(
    {"C10": 12.5, "C12": 3.0, "phi12": 0.25},
)
~~~

Interactive controls call `preview()` with the same complete aberration
mapping. It returns a transient phase array and, when requested, its exact
loss; it does not create a second public result type:

~~~python
phase, loss = workflow.preview(
    {"C10": 12.5, "C12": 3.0, "phi12": 0.25},
    compute_loss=True,
)
~~~

`SSB.open()` loads one source, while `SSB.from_array()` accepts an existing
backend-resident detector array. Both require `voltage_kV`, `semiangle_mrad`,
and `scan_sampling_A`. Use the context-manager form for long-running or
repeated workflows so GPU buffers are released deterministically.

See [the maintainer architecture](../maintainer/ssb-architecture.md) for the
backend boundary, CLI contract, and parity gates.

The canonical anonymized real-data MPS regression case is
`reference-512-full-bf-v1`, shape `512x512x192x192`, with all
`8,937` automatically detected BF pixels, seed 42, 200 exact trials, and
Nelder-Mead. The retained MPS implementation reaches a quiet best of
`23.058 s` fit / `24.16 s` process wall; a five-run thermal soak measured
`24.289-25.221 s` fit (median `24.528 s`). Every run returned the exact values
and complete trial hash recorded in
[the performance contract](../maintainer/ssb-performance.md).

Signoff expectations:

- Use real data, not only synthetic controls.
- Compare CUDA and MPS on the same BF-pixel selection.
- Include images and difference maps, not only scalar tables.
- Do not use object mode or any other fast review mode for exact phase/loss
  reference-agreement claims.
- Keep temporal/joint SSB experiments separate until the improvement metric is
  clear.
