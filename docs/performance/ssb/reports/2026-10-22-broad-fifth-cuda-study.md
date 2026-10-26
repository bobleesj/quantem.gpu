# CUDA broad-fifth research workflow

> **Historical:** describes the code as of 2026-10-22; its fits refined with the finite-difference Newton, since replaced by exact derivatives, so its fit timings are not current. See the [current fit methods](../../../api/direct_ptycho.md).

This private-model experiment reduced full 512-square workflow time by
54–65% while preserving the complete tested phase arrays and search records.
It does **not** change the public higher-order estimator or its defaults.

## Workload and boundary

Each paired fresh process used the same full detector data, native fitting,
200 primary trials, 400 broad-fifth TPE trials, and a 3,000-evaluation Newton
cap. Output used Auto sampling. All measurements below used the same physical
Blackwell Max-Q GPU, with six repetitions per variant and scan size.

| Acquisition | Valid BF pixels | Auto | Before p50 | Candidate p50 / p95 / max |
| --- | ---: | ---: | ---: | ---: |
| A | 8,807 | 3x | 27.847 s | 12.404 / 13.853 / 13.984 s |
| B | 101 | 1x | 8.788 s | 3.115 / 3.345 / 3.402 s |
| C | 8,930 | 4x | 40.859 s | 18.660 / 18.968 / 19.023 s |
| D | 9,298 | 3x | 29.187 s | 13.103 / 14.224 / 14.536 s |

Workflow time includes loading, preparation, both fitting stages,
reconstruction, host transfer and buffered output. It excludes imports,
evidence bookkeeping and process cleanup. Candidate process-wall medians were
17.171, 7.658, 23.140 and 17.661 seconds respectively. Raw per-stage timings,
process boundaries and memory observations are retained in experiment group
`1022-freshfinal`. Filesystem caches and background host activity were
uncontrolled; six samples give preliminary p95 estimates.

## Changes and parity

The research evaluator reuses detector-side optical calculations, shares
coefficients within each block, and groups eight independent candidates per
saturated-grid launch. The sampler batches independent inverse-CDF work,
reuses the existing compiled central log-CDF, and ranks KDE candidates on CUDA
with an original-calculation fallback near ties. Scientific objective evaluation
remains on CUDA. Candidate generation and the float64 Newton controller remain
hybrid CPU/GPU work.

All 48 paired workflows across four acquisitions at 256 and 512 scan sizes
matched complete phase arrays, proposals, coarse scores, Newton trajectories,
stopping states and calibration. Twelve additional full-field searches at
three other seeds also matched frozen references. KDE density values are not
claimed bitwise identical to the CPU library; selected proposals and complete
scientific outputs were identical on the tested paths.

The combined prototype also passed an actual headed Jupyter/ShowPtycho check
on acquisitions A and C, including Auto phase/FFT display and a pointer-driven
C10 adjustment. This is functional acceptance, not an FPS benchmark.
Evidence groups: `1022-consumer-broad-final` and `1022-seedfinal`.

## Limits and integration status

The frozen research source started from `db305309`; the public four-case reuse
branch was at `6bb36fa0` during qualification. Prototype dependencies and
runners were snapshotted before execution in the private archive.

The historical research objective has a different numerical contract from the
public optional higher-order estimator. These improvements must not silently
replace that estimator. Public integration needs a separate numerical review
and regression qualification. The empirical reference is not physical ground
truth, and unchanged results do not establish a global minimum.

Rejected changes include full geometry caching, pointwise score fusion,
two-candidate grouping and smaller compiled trial arrays when they changed
objective rounding. Single-candidate specialization gave inconsistent workflow
gains. The shared-FFT mixed-compiler-cache discrepancy remains held. No
theoretical performance minimum, deployment or release is claimed.
