---
name: gpu-benchmark
description: Benchmark or compare quantem.gpu performance on physical CUDA or MPS, preserve scientific parity, record reviewed results, and qualify installed wheels. Use for timing claims, performance regressions, optimization validation, or release benchmark checks.
---

# GPU benchmarks

Keep the benchmark implementation and reviewed results in this repository.
Use the shipped runner before writing another timing script. This skill does
not authorize publishing, tagging, replacing baselines or stopping other jobs.

## Choose the existing measurement

- SSB joint fitting and regional correction: `quantem-gpu benchmark-ssb`.
  Read [the command and timing boundaries](../../../docs/performance/ssb/joint-fit.md).
- Other kernels, acquisition IO or native paths: read
  [the test guide](../../../docs/developer/testing.md) and select the existing
  runner for that operation and backend. Extend it for a missing case.
- Current reviewed numbers live in
  [the results index](../../../docs/performance/results.md); changes to their
  interpretation go in [the change ledger](../../../docs/performance/changes.md).
  Preserve dated results instead of relabeling them as current.

## Establish a matched comparison

Record the checkout and dirty state, imported source fingerprint or wheel,
dependency versions, physical GPU, input dimensions/dtype, calibration and
measurement boundary. Freeze the known-good GPU output before changing compute.
Use the same acquisition, calibration, precision and workload for the candidate.

For SSB, the default interaction is one seed, 200 trials plus Newton. The
full-field estimated rotation tests two branches. Regions jointly fit optics,
tilt and depth; reconstruction uses the full scan before cropping the view.
Benchmark repetitions estimate timing variation and do not change the number of
searches in an interaction. Retain every input position and bright-field pixel.

Keep scientific search and refinement on the selected GPU. CPU orchestration,
hashing and comparisons are permitted. Do not replace the GPU oracle with a
full-size CPU reconstruction. Do not reduce data, precision, iterations or
tolerances to obtain a faster number.

## Measure and verify

1. Check GPU activity and memory headroom. Respect the host's compute lock.
   Preserve other users' jobs. If no quiet slot is available, label the run
   contended or defer the timing claim; an allocation failure is not a result.
2. Use synchronized timing and separate load, preparation, fitting,
   reconstruction, transfer and browser paint. Report first observations
   separately from repeated warm median/p90. Do not call a first observation
   cold-disk unless caches were actually controlled.
3. Compare scientific output against the frozen, matched GPU reference and
   retain existing tolerances. Check fit parameters and reconstructed phase,
   not only loss or an image that looks plausible. Repeated self-consistency
   alone is not correctness against the previous implementation.
4. Inspect the produced artifact and any errors. Missing hardware/data,
   skipped tests and CPU/software-renderer fallback do not qualify a backend.
   MPS evidence is not CUDA evidence; kernel timing is not notebook image rate.

## Record the result once

Write raw JSON, logs, profiler traces and private images outside this public
repository, in the private evidence archive. Add a dated Markdown summary under
the owning `docs/performance/<operation>/reports/` folder with the question, exact source
fingerprint, hardware, workload, first/warm results, parity outcome, limitations and private experiment ID.
Link it from the operation report index and the results index. Keep one owner
for a cross-operation report and link to it from other relevant topics. Never publish private acquisition names, home
paths, host names or non-public images. Public gold data can be used for figures.

For a changed implementation, state the before/after values and the isolated
bottleneck. An unchanged source with different runtime conditions is a new
measurement, not an optimization. Do not silently replace a failing baseline.

## Release use

Build a wheel and install it into a new environment, unset `PYTHONPATH`, then
run the SSB command with `--require-installed` separately on MPS and CUDA.
The release workflow gates publication on these hardware checks. The widget
release additionally runs its installed-wheel browser test from fresh Jupyter
kernels: draw, move, resize, keep regions independent, save and restore.

No registered hardware runner means release qualification is incomplete.
Keep the candidate wheel and report the missing gate. Benchmark success does
not bypass the repository's publication and privacy checks.
