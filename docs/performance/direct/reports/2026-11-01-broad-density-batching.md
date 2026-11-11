# Exact broad-search density batching, 2026-11-01

## Question and change

The default fifth-order fit and `search="broad"` have different workloads. Standard uses 200 initial candidates and fifth-order refinement. Broad adds 400 higher-order candidates and refinement from the standard start plus eight distinct candidates. The current wide bounds also retain half-band consistency fits and the existing fallback when inconsistent.

Profiling identified repeated independent marginal-density calculations in the broad Optuna sampler. `_tpe_independent._batch_log_pdf` evaluates those marginals together, retaining each mixture's component order, float64 reductions, RNG sequence, candidate ranking and sequential trial feedback. It is enabled only by the existing Optuna 4.9.0 fixed-space sampler gate; other versions retain stock Optuna. This reduces host overhead. It does not turn the hybrid controller into a GPU-only optimizer.

A separate fused CUDA gradient-setup candidate preserved exact arrays but did not demonstrate a useful whole-fit improvement; it was not integrated.

## Matched measurements

Two private 512×512 scans with 192×192 detectors, existing calibration and all selected BF pixels. NVIDIA RTX PRO 6000 Blackwell Workstation Edition; synchronized wall times, one process, existing compute lock, same stored source/calibration and restored initial optics. No candidate, start, refinement, consistency check or output was removed. The mode uses its default 200 initial plus 400 exploration trials. Observations alternate before/candidate/candidate/before. Loading, UI and persistence are outside these fit times; first observation is not cold disk. CPU regression checks overlapped part of the second acquisition's sequence, so the small timing sample is not a quiet-host performance guarantee.

| Acquisition | Before first / warm (s) | Candidate observations (s) |
|---|---:|---:|
| A | 12.607 / 12.102 | 10.760 / 10.411 |
| B | 15.939 / 15.698 | 14.444 / 14.668 |

An exact 400-trial proposal replay, excluding GPU objective work, took 3.04–3.11 s before and 1.97–2.12 s after. Every proposed coordinate was identical for both acquisitions. These replay numbers isolate host candidate overhead, not scientific fit latency. Default standard fitting does not use the changed broad exploration path and receives no claimed speedup.

## Numerical and memory checks

- Existing adaptive 400-trial parity tests compare every proposal and score against stock Optuna for seeds 17, 42 and 87; quantile tail/version-gate tests also pass.
- Full CUDA comparisons retain identical primary records, all broad proposals/scores, refinement trajectories (excluding their wall-clock column), start selection, consistency outcomes, fitted coordinates and phase-array hashes. Candidate tests use the frozen existing GPU result, not a CPU reconstruction oracle.
- No CUDA buffer representation or allocation is changed. Batching uses transient host arrays across independent marginals, so it is not a claim of identical peak CPU allocation. The density arrays are released after each proposal; final per-parameter scores are small.
- Live additionally records queue, preparation, fit stage and mode metadata. Browser render timing remains a separate boundary.

The shipped `benchmark-direct` qualification runner currently rejects the existing hybrid fit via its `host_numerical_search` contract; it was not reported as passing. The matched broad API/profile harness in private experiment `direct-fit-speed-20261101` covers this operation. No installed-wheel, MPS, native application or release qualification is claimed. Native MPS is unchanged and untested by this CUDA experiment.

Raw profiles, source fingerprints, numerical outputs and browser evidence are retained in the private experiment. Existing broad results are preserved. No publication or release accompanies this local change.
