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
result = workflow.fit(save_to="results/ssb")
~~~

The default fit always uses the automatically detected full bright field and
the exact float32 phase-variance objective. `result` is the single
`SSBResult`: it contains optimizer metadata, `object_wave`, and the derived
`phase` and `amplitude`. A backend that cannot honor the request fails
explicitly.

`save_to` provides automatic saved-result reuse. The object wave is stored in
an NPZ file and its readable JSON companion records source and calibration
identity, physical parameters, optimizer settings and history, bright-field
geometry, timings, loss, package source identity, and Git provenance. An exact
match is loaded without fitting again. Any scientific mismatch recomputes and
replaces the saved result. `result.reused`, `result.saved_path`, and
`result.metadata` expose that state directly.

Use `reconstruct()` when aberrations are already known and no optimizer should
run:

~~~python
fixed_probe_result = workflow.reconstruct(
    {"C10": 12.5, "C12": 3.0, "phi12": 0.25},
    save_to="results/fixed-ssb",
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
When `SSB.from_array()` uses `save_to`, provide `source_path` so matching never
mistakes a different same-shaped detector array for the original source.

For a numbered QuantEM screening series, keep folder discovery, exact result
reuse, reconstruction, labels, and review inside the same SSB API:

~~~python
series = SSB.reconstruct_series(
    raw_directory,
    first_frame=52,
    last_frame=91,
    probe_reference_frame=57,
)

series.show()
series.metrics()
series.metadata()
~~~

The API discovers regular QuantEM Live products under
`<raw_directory>/quantem/screen` and reuses matching acquisitions. Missing
products run automatically and are saved there. `first_frame`, `last_frame`,
and `probe_reference_frame` use the trailing acquisition identifier in each
dataset name; the endpoints are inclusive and do not need to be consecutive.
Omit `probe_reference_frame` to fit every acquisition independently.
The result contains one unambiguous phase stack at native scan resolution. Its
`bright_field` and `dark_field` companion stacks are also retained. `show()`
opens on the SSB phase and keeps those two views hidden but available through
the standard Show3D panel controls. Its
metrics distinguish signature-checked saved results from newly computed
acquisitions. Historical arrays without a complete signature are recomputed.
Its metadata records the requested backend,
optimizer settings, sources, inclusive first/last frame, reference frame, and
reuse counts. The complete per-acquisition identifiers remain available through
`series.metrics()`, `series.frames`, and `series.datasets`.
With the default progress display, the workflow reports each acquisition while
the exact operation decides whether to reuse its saved result or reconstruct it.

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
