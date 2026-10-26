# Reuse Fourier preparation in four-case SSB review

> **Historical:** describes the code as of 2026-10-22; its fits refined with the finite-difference Newton, since replaced by exact derivatives, so its fit timings are not current. See the [current fit methods](../../../api/ssb.md).

The rotation/defocus comparison now prepares one temporary full-field session,
then reuses its source Fourier data for all four cases. Changing the rotation
refreshes the geometry. Each signed-defocus search still starts from zero
aberrations and has its own 200 trials plus Newton refinement. The caller's
rotation, calibration and fitted optics are preserved.

## Matched CUDA evidence

The private experiment group `1022-rotation-reuse` compares frozen source
`db305309` with the candidate on a physical Blackwell Max-Q GPU. All valid
bright-field pixels, detector dimensions, precision and estimators are retained.
Fitting uses native scan sampling; displayed reconstruction uses Auto.

Median wall time for the complete four-case tilt/depth comparison on 512-square
scans, excluding acquisition loading:

| Acquisition | BF pixels | Auto | Before | Reuse |
| --- | ---: | ---: | ---: | ---: |
| A | 8,807 | 3x | 4.551 s | 4.190 s |
| B | 8,930 | 4x | 7.493 s | 7.140 s |
| C | 101 | 1x | 0.249 s | 0.218 s |

Three matched repetitions provide initial timing evidence, not robust p95
latency. Acquisitions A and B retain 192-square detectors; C retains a
12-square detector and already has finer scan sampling than its aperture needs.
Do not compare these detector workloads as equivalent.

## Numerical acceptance

All three acquisitions passed at 256- and 512-square scan sizes, with ordinary
and tilt/depth-corrected SSB tested separately. Complete native and Auto phase
arrays and fitted coefficient records matched the frozen implementation
exactly. A subsequent run through the changed public API also matched every
frozen phase array and fit record. A contract test verifies independent starts,
unchanged trial budgets, Auto output and preservation of the caller's state.

This change reduces redundant preparation; it does not omit any search or
establish the physical rotation. The result remains unresolved without an
independent focus or rotation constraint. MPS hardware acceptance remains
separate from this CUDA study. No release or default estimator change is made.

## Consuming notebook acceptance

The changed API also passed in a real Jupyter/ShowPtycho workflow on acquisitions
A and B at full 512-square scans. The notebook loaded the acquisition, ran the
200-trial primary fit with Newton and tilt correction, displayed Auto phase and
FFT, and responded to a pointer-driven C10 adjustment. The subsequent four-case
comparison matched all eight frozen native/Auto phase arrays per acquisition,
left the parent session unchanged, and rendered one comparison figure. The
browser reported an NVIDIA Blackwell adapter; sessions were closed after testing.
This is functional workflow acceptance, not an interactive frame-rate benchmark.
The private evidence group is `1022-consumer-notebook-v6`.

## Additional acquisition coverage

A fourth acquisition with 9,298 valid BF pixels and Auto 3x output also
passed complete native/Auto phase and fit-record equality at both scan sizes,
with ordinary and tilt/depth SSB tested separately. Six matched comparison
sessions per variant used the same Blackwell workstation GPU. These sessions
reused one loaded acquisition and are not fresh-process loading measurements.
The private evidence group is `1022-rotation-reuse`.
