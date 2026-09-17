# Strict float32/complex64 SSB parity gate

## Question

At byte-identical inputs, do the accelerated SSB paths stay inside a strict
float32/complex64 tolerance of an independent double-precision evaluation of
the same objective - and if any pair disagrees, which implementation is wrong?

## Result, numbers first

| gate (worst gated setting) | 128x128 real crop | 512x512 full acquisition |
| --- | --- | --- |
| MPS vs double oracle, object relL2 | 2.102e-06 passes (bound 4.378e-06) | **4.317e-06 FAILS (bound 2.196e-06)** |
| MPS vs native Metal, object relL2 | 6.096e-07 passes | **4.179e-06 FAILS (bound 2.196e-06)** |
| MPS vs double oracle, loss rel | **FAILS at C10 = 155.97: 2.875e-07 against 2.240e-07**; worst passing 4.599e-07 against 1.032e-06 | 1.619e-07 passes |
| MPS vs double oracle, phase max (rad) | 4.421e-06 passes | **1.104e-05 FAILS (bound 6.532e-06)** |
| Metal cached vs double oracle, object relL2 | 2.116e-06 passes | 5.618e-07 passes (best of all) |
| Metal cached vs streamed, loss rel | 8.665e-08 passes | 7.600e-08 passes |
| Metal cached vs streamed, object relL2 | 5.030e-07 passes | 3.731e-06 passes (bound 4.583e-06) |
| native fit trajectory vs its pin | matches (digest `74cc7772...`, 2 processes) | not pinned |
| batched vs sequential fit, optimum delta | (-3.18, +7.32, +0.611) nm/rad | (-13.82, -13.83, -2.16e-04) nm/rad |

Verdict: **the native Metal path clears the strict float32 gate everywhere; the
MPS path does not, at 512x512 on the object/phase metrics and at one recorded
128x128 setting on the loss metric**, deterministically and reproducibly. The
fit is deterministic and pinned, and enabling the batched pair draw changes the
final optimum on both artifacts.

## Command

```bash
scripts/check_ssb_parity.sh              # fast objective gate: real 128x128 ARINA crops
scripts/check_ssb_parity.sh --full       # adds the full 512x512 acquisition
scripts/check_ssb_parity.sh --build-metal   # rebuild the native harness first
scripts/check_ssb_parity.sh --metal-only    # gate the native Metal pairs alone

scripts/check_ssb_fit_trajectory.sh          # fast fit gate: trajectory + batched draw
scripts/check_ssb_fit_trajectory.sh --full   # adds the full 512x512 acquisition

python scripts/ssb_parity_summary.py parity-runs/gate-fast.json

# pytest against the recorded report (fast, no GPU work; finding F10)
QUANTEM_SSB_PARITY_REPORT=parity-runs/gate-fast.json PYTHONPATH=src \
  ~/miniforge3/bin/python3.12 -m pytest tests/parity/test_ssb_strict_parity.py -q

# pytest that re-measures; needs a previous `check_ssb_parity.sh` run for the
# recorded-C10 case, otherwise one test fails on the missing case (F10)
~/miniforge3/bin/python3.12 -m pytest tests/parity/test_ssb_strict_parity.py -q
```

The two gates are independent: the first proves one loss is right, the second
proves the search that consumes it is pinned and that batching is a different
search. Both exit non-zero on any failure and never rewrite a number.

The unmodified objective gate is red on this artifact because the **MPS**
finding above is real, so its exit status is not the native Metal acceptance
signal. `--metal-only` measures and gates the `metal:*` and
`metal cached vs streamed` pairs alone - identical bounds, identical oracle,
MPS simply not measured - which makes the exit status of a Metal-only change
mean what its author needs it to mean: 0 if the native path kept its
precision, non-zero if it lost it.

The pytest suite follows the same split at 512x512.
`QUANTEM_SSB_PARITY_FULL=1` measures the native Metal pairs alone and asserts
them, and `test_full_512_mps_finding_stays_open` re-applies the unchanged bounds
to the recorded full report. That second test passes while F1 is open and fails
loudly the day MPS stops failing, so the finding cannot be retired by quietly
deleting a test or raising a bound.

Every GPU command runs under the shared `gpurun` lock
(`GPU_RUN_LABEL=parity ~/perf-lab/ssb-audit/gpurun <cmd>`), so concurrent agents
cannot corrupt each other's measurements. Generated data goes to
`/path/to/local/perf-lab/ssb-audit/parity-runs/` and never into the repository.

## What is compared

One exact BF-column artifact per case (real ARINA counts, `uint16`, one byte
range per value, `detector_bin=1`, no crop of the detector, no approximation)
is measured three ways at the same declared geometry and aberration settings:

| Pair | Path |
| --- | --- |
| `mps vs oracle` | public `quantem.gpu.SSB` MPS path vs the float64 oracle |
| `metal:* vs oracle` | native `MetalSSBKernels` via the standalone harness vs the oracle |
| `mps vs metal:*` | MPS vs Metal directly, cached / hybrid / streamed topologies |
| `metal cached vs streamed` | the topology pair the retired endpoint defect split |

`tests/parity/ssb_double_reference.py` is the oracle: an explicit
double-precision evaluation of the documented objective (full-plane forward FFT
of the exact counts, `G * conj(gamma) / max(|gamma|, 1e-8)`, DC bin replaced by
the exact mean raw zero-frequency value, inverse FFT, object = mean over BF of
the per-BF planes, loss = mean over scan pixels of `sum(phi^2)/N - (sum(phi)/N)^2`).
It imports nothing from `quantem.gpu` and is never used by production code.
Re-evaluating that same formula in float32 with SciPy pocketfft gives the
measured single-precision arithmetic floor of the identical mathematics.

## Inputs

- Source: `arina-fixture-b_master.h5`,
  `512x512` scan, native `192x192` detector, `uint16`.
- Bright field: automatic full disk, center `(94.88451385498047,
  96.35952758789062)`, radius `53.35992814757164 px` -> 8,938 geometric pixels
  minus the one hardware dead pixel inside it (`(78, 74)`) = **8,937** logical
  pixels, **2,464** of them aperture-active.
- Detector sampling `1.0909090909090908 mrad/px`, rotation
  `158.88268568029937 deg`, `300 kV`, `30 mrad`, `0.264 A/px`.

The export proves its own inputs: `verify_counts_against_hdf5` re-reads the
acquisition with h5py alone (no `quantem.gpu` loader, no hot-pixel correction,
`hot_pixel_correction="none"`, no dtype conversion) and compares every selected
value at every scan position.

## Gates

Each metric must clear **both** an absolute bound and `2x` the measured
single-precision floor of the identical formula. The multiple is 2 because no
accelerated float32 backend can be required to beat a straightforward float32
evaluation of the same objective, and a backend far above that floor is losing
precision.

| Metric | Absolute | Derivation |
| --- | ---: | --- |
| `object_relative_l2` | 1e-5 | `eps32 = 5.96e-8`; the object is a normalized sum of N partially cancelling float32 transforms, so a faithful evaluation lands in the 1e-6 decade |
| `object_max_abs_error_relative` | 1e-5 | same |
| `phase_max_error_radians` (object phase) | 1e-4 | worst case implied by the object bound at a conservative 0.1 amplitude floor: `1e-5 / 0.1`; every gated setting measures a minimum core amplitude of 0.876, and the `2x` floor term (6.5e-6 to 1.2e-5 rad) is the effective bound |
| `loss_relative_error` | 1e-5 | two orders below the retired half-plane defect (1.06e-3) and 5x below the 5e-5 gate that the cached/streamed disagreement broke |
| `loss_absolute_error` | 1e-6 | worst case for a variance of unit-circle phases in float32: `4*pi*eps32 = 7.5e-7` |

The gated aberration settings are the frozen optimum
`(73.18188621458395, 14.020962948808993, 0.4700365259977606)` and a generic
`(-42.5, 27.3, 1.13)`. The frozen optimizer start `(0, 50, 0)` is retained and
reported but **not gated**: its aberration phase is a pure quadratic form in
`k`, so the normalization denominator `gamma` has a zero curve through the
discrete scan grid. The measured single-precision floor there is 1.5e-2
relative L2, so no implementation can reproduce the double-precision objective
at that setting and measuring it as a precision gate would be meaningless.

## Measured results

All numbers below are one process run each on the Apple M5 under the shared GPU
lock. `parity-runs/summary-fast.txt` and `parity-runs/summary-512.txt` are the
generated tables; `parity-runs/gate-fast.json` and `parity-runs/gate-512-run2.json`
are the machine-readable reports.

### 128x128 real ARINA crop (fast gate, 8,937 BF / 2,464 aperture-active)

`arina-128-full-disk`, setting #0 = the frozen optimum
`(73.18188621458395, 14.020962948808993, 0.4700365259977606)`:

| pair | object relL2 | object maxrel | object phase max (rad) | loss rel | loss abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent float32 floor | 5.4623e-06 | 6.3338e-06 | 5.4433e-06 | 1.6556e-07 | 1.4261e-08 |
| metal:cached vs oracle | 1.8640e-06 | 4.5940e-06 | 4.8980e-06 | 1.4399e-07 | 1.2403e-08 |
| metal:hybrid vs oracle | 1.8640e-06 | 4.5940e-06 | 4.8980e-06 | 1.4399e-07 | 1.2403e-08 |
| metal:streamed vs oracle | 1.8972e-06 | 4.7729e-06 | 5.1513e-06 | 1.4399e-07 | 1.2403e-08 |
| metal cached vs streamed | 2.3216e-07 | 5.9527e-07 | 5.9258e-07 | 0.0 | 0.0 |
| **mps vs oracle** | **1.8349e-06** | 4.1410e-06 | 4.3560e-06 | 5.7496e-08 | 4.9527e-09 |
| mps vs metal:cached | 2.8997e-07 | 8.1812e-07 | 8.0565e-07 | 8.6493e-08 | 7.4506e-09 |

`arina-128-inner-disk` (1,812 BF, every one aperture-active), setting #0: floor
2.1890e-06, `metal:cached` 2.1161e-06, `mps` 2.0883e-06, `mps vs metal:cached`
3.5673e-07; the Metal cached/hybrid objects are bit-identical and the
`mps vs metal:cached` loss relative error is exactly 0.

`arina-128-recorded-c10` (the three settings of the 2026-09-13 loss diagnosis),
setting #2 (C10 = 155.96977 nm): floor object relL2 5.4456e-06, metal:cached
1.8465e-06, **mps 1.8209e-06** (better than the floor) but the MPS *loss* is
2.87459e-07 relative / 2.4969e-08 absolute against a `2x` floor bound of
2.24e-07 / 1.94569e-08 — see finding F4.

### 512x512 full acquisition (slow gate)

`arina-512-full-disk`, setting #0 = frozen optimum:

| pair | object relL2 | object maxrel | object phase max (rad) | loss rel | loss abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent float32 floor | 1.0981e-06 | 2.9128e-06 | 3.2660e-06 | 1.6216e-07 | 1.5906e-08 |
| metal:cached vs oracle | 5.6182e-07 | 1.9645e-06 | 2.1948e-06 | 9.9633e-09 | 9.7727e-10 |
| metal:streamed vs oracle | 1.8049e-06 | 4.0080e-06 | 4.3258e-06 | 6.5995e-08 | 6.4733e-09 |
| metal cached vs streamed | 1.8287e-06 | 2.9195e-06 | 3.2028e-06 | 7.5958e-08 | 7.4506e-09 |
| **mps vs oracle** | **4.3173e-06** | **9.8074e-06** | **1.1038e-05** | 1.6188e-07 | 1.5878e-08 |
| **mps vs metal:cached** | **4.1794e-06** | **9.1601e-06** | **1.0289e-05** | 1.5192e-07 | 1.4901e-08 |
| mps vs metal:streamed | 5.1540e-06 | 1.1074e-05 | 1.2162e-05 | 2.2788e-07 | 2.2352e-08 |

Setting #1 `(-42.5, 27.3, 1.13)`: floor 2.2913e-06, metal:cached 5.4122e-07,
mps 3.8205e-06, mps vs metal:cached 3.8810e-06; the one metric over its bound
there is `mps object_max_abs_error_relative` 1.0321e-05 against the 1e-5 absolute
bound (3.2% over).

Oracle losses: 512 setting #0 `0.098087540782`, #1 `0.098032100735`, start
`(0, 50, 0)` `0.097208093258` (diagnostic, not gated).

### Reproducibility

The 512x512 gate was run twice in separate processes
(`gate-512-run1.json`, `gate-512-run2.json`) and the fast gate three times
(`gate-fast.json`, `gate-fast-run1.json`, `gate-fast.json` after the
`loss_absolute_error` gate was added). **Every scientific value is bit-identical
across runs; only the recorded wall-clock timings differ.** The failures below
are therefore deterministic biases, not measurement noise: per the parity-test
skill's diagnosis checklist this is "bit-equal runs, off pin", i.e. a property
of the implementation, not of the host.

A fourth fast-gate run was measured after every commit of this record, on a
contended GPU (`parity-runs/scratch/gate-fast-confirm.json`). A structural
comparison against the recorded `gate-fast.json` finds 56 differing leaves and
**every one of them is a timing field** (`mps/object_seconds`,
`mps/preview_seconds`, and each variant's `gpuSeconds`/`wallSeconds`); no
scientific value moved. That run also reproduced the recorded verdict exactly:
`FAIL: parity gate`, exit 1, with the two `arina-128-recorded-c10#2` MPS loss
checks as the only failures - the same two rows the recorded report flags.

## Fit trajectory: the fit, not just the objective

A gate on one objective evaluation does not protect the fit. `SSBOptimizer.run`
draws `min(2, globalTrials - trial)` candidates from the *same* history whenever
an `evaluateBatch` closure is supplied, so the pair draw is a different search
from drawing trial `t + 1` after telling trial `t`. `MetalSSBEngine.optimize`
passes no `evaluateBatch`, so today's production fit is the sequential one.

`tests/metal/ssb_fit_trajectory.swift` records, on the exact 128x128 artifact and
seed 42: the sequential trajectory (initial loss + 200 trials + Nelder-Mead), the
pair draw through the same single-candidate evaluation, the production `optimize`
entry point, objective-purity probes, and an index-by-index comparison. The
sequential trajectory of the 220 evaluations (float32 point, float64 loss and
stage per record) is frozen as `tests/parity/fixtures/ssb_fit_trajectory_128.json`
together with the measured batched divergence, and
`scripts/check_ssb_fit_trajectory.sh` runs the harness twice (two processes),
runs the Python reference on the same artifact, and exits non-zero on any pin
mismatch. When the tree exposes the native batch objective
(`MetalSSBEngine.phaseVarianceBatch`, `-D SSB_HAS_BATCH_OBJECTIVE`), the gate
also drives the pair draw through it and checks its documented per-candidate
bit-identity claim against single evaluations, so "bit-identical per candidate"
is measured rather than assumed.

| run | C10 (nm) | C12 (nm) | phi12 (rad) | loss | evaluations |
| --- | ---: | ---: | ---: | ---: | ---: |
| sequential (production) | 12.842510790748161 | 0.0 | 1.1460220634838638 | 0.08559209108352661 | 220 (200 + 18) |
| batched pair draw | 9.663990384938845 | 7.323468400946136 | 1.7569089257461865 | 0.08557865023612976 | 248 (200 + 47) |

Divergence: 208 of 220 trials have a different float32 point, 206 differ in loss
at the same index, the maximum per-trial loss difference is 6.0676e-03 (7.1% of
the loss), the best-so-far differs on 208 trials, the final optimum moves by
`(-3.1785 nm, +7.3235 nm, +0.6109 rad)` and the final loss moves by
`-1.3441e-05` (relative 1.57e-04). Candidates that both draws evaluated have
**bit-identical** losses (0 mismatches), so this is purely the draw: same
numbers, different trajectory.

At 512x512 the same comparison gives 250 of 262 trials different, maximum
per-trial loss difference 3.1540e-03, optimum moved by
`(-13.8249 nm, -13.8321 nm, -2.1611e-04 rad)`, final loss
`0.09605839103 -> 0.09447139502` (`-1.5870e-03`, relative 1.65%).

`productionOptimizeAlwaysMatchesSequential` is true on both artifacts, and the
objective is a pure function of the float32 point: repeated evaluation, a fresh
engine, and the same float32 triple reached from a neighbouring double all agree
bitwise (0 mismatches, 0 ULP).

### Running the fit gate against a batching change

`scripts/check_ssb_fit_trajectory.sh` is the acceptance test for any change
that touches the native search or enables `evaluateBatch`. On a tree that
exposes `MetalSSBEngine.phaseVarianceBatch` the script compiles the harness with
`-D SSB_HAS_BATCH_OBJECTIVE` automatically and runs the pair draw twice: once
through the plain single-candidate closure (isolating the sampling effect) and
once through the native batch objective. It then reports, for every candidate
the batch objective evaluated, whether the batched loss is bit-identical to the
single-candidate loss, and whether running a batch perturbs later single
evaluations. A production change that starts drawing pairs is caught by the
sequential trajectory pin (`74cc7772...`); a batch objective that is not
bit-identical per candidate is caught by the identity counters; and a
non-deterministic fit is caught by the two-process digest comparison.

### Does the unbatched path match the frozen/QuantEM reference?

The repository's own frozen reference is
`tests/parity/fixtures/ssb_reference_512_mps.json`
(`expected.loss = 0.04469207674264908`, `trial_trace_sha256 = cb2b35ca...`).
It cannot be reproduced here: on this ARINA acquisition the same aberration
triple has oracle loss `0.098087540782`, and the documented BF companion maximum
count is 53 against 1913 measured here, so Reference-512 was measured on a
different source file (finding F8). Nothing in the repository verifies that hash:
`rg trial_trace_sha256` finds it only in the fixture and the prose. It is a
documentary pin with no executable check.

The executable QuantEM/Python reference in this tree is the public MPS fit. It
defaults to `optuna_batch_size: int = 2`, i.e. the **same-history pair draw**:
`Study.ask()` registers a trial in state RUNNING, Optuna 4.6.0's `TPESampler`
builds its history from `COMPLETE`/`PRUNED` trials only
(`samplers/_tpe/sampler.py:447,531`), so both asks of one step see the same
completed history - exactly the semantics `SSBOptimizer.run` implements for
`batchCount = 2`. The native production path is sequential, so the native
engine and its own reference disagree about the sampling contract. Measured on
the same 128x128 artifact and seed (`parity-runs/fit-reference-128.json`):

| reference run | best optuna loss | C10 (nm) | C12 (nm) | phi12 (deg) | loss after refinement |
| --- | ---: | ---: | ---: | ---: | ---: |
| `optuna_batch_size=2` (default) | 0.08505632728338242 | 17.05306550293158 | 17.340889360905464 | -65.50146188892997 | 0.0848577618598938 |
| `optuna_batch_size=1` | 0.08557009696960449 | -8.665518253322524 | 6.248562683755667 | -1.10898839103687 | 0.08556715399026871 |

The two implementations do not share a sampler (Optuna's `TPESampler` against
the native deterministic one), so no trajectory matches index by index; what is
comparable is the direction. In all three measured pairs the batched draw reaches
a lower loss than the sequential draw: native 128x128 `-1.34e-05`, native
512x512 `-1.59e-03`, Python 128x128 `-7.10e-04`. The two native loss values
(0.08559209 sequential, 0.08557865 batched) both sit within 2.9e-05 of the
Python reference results, so all four searches are in the same valley, but the
*points* differ (C12 = 0 nm sequential against 6.2-17.3 nm elsewhere): the
astigmatism direction is flat at 128x128, so its value is decided by the search
order, not by the data.

## F1 attribution: which line loses the precision

F2b left the excess somewhere inside the summed spectrum: the per-BF weighting
(the ``gamma / |gamma|`` unit-complex construction and the probe term) or the
complex64 accumulation over 8,937 terms. That is now bisected, one lever per
process, on the failing 512x512 setting #0 (C10 = 73.181886 nm,
C12 = 14.020963 nm, phi12 = 0.470037 rad). Nothing below changes production
arithmetic; every variant is a scratch probe under
``parity-runs/scratch/19_f1_variants.py`` with its own report in
``parity-runs/scratch/f1/``. The oracle and the float32 floor are the same
cached artifacts the gate uses (floor relL2 1.0981e-06, 2x bound 2.196e-06).

| variant (one lever changed) | object relL2 | x floor | phase max (rad) |
| --- | ---: | ---: | ---: |
| production (base) | 4.3173e-06 | 3.93 | 1.1038e-05 |
| object trig spelled `metal::sincos` instead of `metal::fast::sincos` | 4.3173e-06 | 3.93 | 1.1038e-05 |
| accumulation chunk 64 (140 groups) | 4.3151e-06 | 3.93 | 1.1049e-05 |
| accumulation chunk 512 (18 groups) | 4.3351e-06 | 3.95 | 1.1146e-05 |
| Kahan-compensated accumulation (fast mode) | 4.3173e-06 | 3.93 | 1.1038e-05 |
| Kahan-compensated accumulation (safe mode) | 5.8261e-06 | 5.31 | 1.3795e-05 |
| safe mode, uncompensated (control) | 5.8218e-06 | 5.30 | 1.3802e-05 |
| forward BF transform recomputed by SciPy float32 | 4.3213e-06 | 3.94 | 1.1103e-05 |
| forward BF transform recomputed by SciPy float64 then rounded | 4.3204e-06 | 3.93 | 1.1111e-05 |
| probe term `pk` computed in-kernel from raw kx/ky (Metal-equivalent) | 3.8184e-06 | 3.48 | 1.1083e-05 |

Readings, in the order the levers were asked for:

1. **The accumulation is not the line.** Re-grouping the complex64 sum from
   70 groups (production 128) to 140 and 18 groups moves the metric by <= 0.4%.
   Making the accumulation *exact* for the same float32 terms - Kahan
   compensation, which the compiler cannot remove in `math_mode: "safe"` -
   moves it by 0.07% (5.8261e-06 against the 5.8218e-06 uncompensated control);
   in `math_mode: "fast"` the compensated and uncompensated kernels are
   bit-identical (4.3173e-06), because fast-math reassociation removes the
   compensation. So neither the order/depth nor the precision of the
   8,937-term sum is the excess. Compensating it would buy nothing at any cost.
2. **The trig spelling is not the line, and neither is `math_mode`.** Over
   18,302,976 real chi samples of this setting (`|chi| <= 167.09 rad`,
   `parity-runs/scratch/f1/intrinsic-sincos-s0.json`), `metal::fast::sincos`
   and `metal::sincos` are **bitwise identical** (max |delta| = 0.0), so the
   `fast::` spelling in `engine.py:3919/3924` is a naming difference, not a
   precision difference - which is also why the earlier `math_mode` probe found
   a bit-identical object. Switching the whole kernel to `math_mode: "safe"`
   makes it *worse* (5.82e-06): fast mode is contributing fused multiply-adds,
   not shortcuts. What does differ from the reference is the trig
   *implementation*: over the same samples Metal's `sincos` is 3.19e-08 RMS
   (1.34e-07 max) from float64 where numpy float32 `cos`/`sin` is 1.70e-08 RMS
   (6.88e-08 max) - about 1.9x the reference's per-term trig noise.
3. **The FFT realization is not the line** - including the forward transform,
   which F2b had left untested. Replacing the MLX forward `rfft2` with SciPy's
   in float32 leaves the object error unchanged (4.3213e-06), and replacing it
   with an exactly-rounded float64 forward transform leaves it unchanged too
   (4.3204e-06). The error is created entirely downstream of the forward
   transform.
4. **The per-BF weighting construction is the line.** Re-building the probe
   term `pk` in the kernel from the raw `kx/ky` with the same float32 formula
   Metal uses, at identical accumulation, is the only lever that moves the
   object materially: 4.3173e-06 -> 3.8184e-06 relL2 (-11.6%), i.e. 15% of the
   3.219e-06 excess over the floor, at no measured cost (the variant skips the
   host `pk` build). The residual lives in the same per-(BF, pixel) float32
   weight chain the reference also evaluates, and it is diffuse rather than
   structural: the error field is uniform over the object (median 1.26e-05,
   max 3.96e-05, ratio 3.1; no concentration on borders, the Hermitian seam or
   the aperture rim), which is the signature of summed per-term rounding.

What that means for a fix: `double` is not available in Metal shaders on this
GPU (`'double' is not supported in Metal`, refused by the compiler), so a
higher-precision term chain has to be emulated; the cheap, measured step is the
`pk` construction above, worth 15% of the excess. There is no single line whose
repair is expected to reach the floor: the remaining 85% is the per-term float32
rounding of the weighting chain as a whole.

Fit impact of that candidate fix, measured rather than argued: **none**. The
fit never builds the object kernel - `_reconstruct_prepared_batch_exact_loss`
calls `_reconstruct_prepared(..., compute_loss=True, compute_object=False)` -
so an object-kernel change cannot enter the search. Checked directly anyway with
`parity-runs/scratch/21_f1_fit_variant.py --pk-inline`, which installs the same
`pk` rewrite inside a full 200-trial reference fit on the 128x128 artifact
(seed 42, batch 1): the object kernel really was rewritten
(`ssb_object_fourier_sum_dyn_fast_sincos_b128_n2464_logical8937_sparse0_128_128_g65`,
one match), and the trajectory digest is identical with and without the patch,
`227cb1f4f8fe0553a10e198d0f2148414ea6767907d08837deccf61043b1e424`, with 0 of
200 trial losses differing, the same best trial (184), the same optimum
(-8.665518253322524, 6.248562683755667, -1.10898839103687) and the same loss
(0.08557009696960449). The same-session unpatched control reproduces the
recorded reference trajectory bit for bit. The fix is a rendering fix: it
cannot move a fit optimum, and therefore cannot be validated through the fit
gate - only through the object/phase metrics of the objective gate.

### Is the MPS loss passing by luck?

No - the loss is a structurally different computation from the object, and it
is sound at the float32-representation level:

- The loss is computed by the phase-moment kernels (`_row_ifft*`,
  `_corrected*`, `ssb_phase_cols*`), never by the object-sum kernel. The object
  excess above does not enter it.
- One float32 ulp of a loss near 0.098 is 7.45e-09. The gate records
  `mps vs metal:cached` loss agreement of 0.0 to 1.49e-08 absolute
  (0 to 2.0 ulp) on all 9 gated 128x128 and 512x512 rows, and MPS against the
  double oracle is 1.5878e-08 (2.1 ulp) at 512x512 setting #0 against the
  float32 floor's own 1.5906e-08 (2.1 ulp). Two independent implementations
  agreeing to 1-2 ulp of the representable value, at every tested setting, is
  not a lucky pass: the loss path returns the correctly rounded float32
  objective to a couple of ulp.
- The one exception is the recorded 128x128 C10 = 155.96977 nm setting, where
  MPS is 2.8746e-07 relative (3.0 ulp, 2.57x the floor, 1.28x the 2x bound).
  The gate is right to flag it: the loss path is not uniformly floor-equal.
- Mechanism for the object/loss asymmetry: the object is a coherent sum of
  8,937 complex terms per pixel whose cancellation amplifies per-term rounding;
  the loss is a per-pixel variance of per-BF phases, averaged over the BF set,
  where the same per-term rounding averages down instead of amplifying.

Fit impact, measured rather than argued: the loss differences the search acts
on are 4 orders larger than the loss-path error. Changing only the *draw*
(batched against sequential, the same objective) moves the final loss by
1.34e-05 relative 1.57e-04 at 128x128 and 1.59e-03 at 512x512, while the
implementation difference is <= 3.0 ulp (~2.2e-08 absolute). The one place the
two scales meet is the Nelder-Mead tail: the last recorded improvements are
1, 6, 7, 17 and 24 ulp, so the *final few digits* of a reported optimum are at
the objective's float32 resolution and can differ between implementations. The
optimum's physical values and the loss are not.

## Trial-budget sensitivity at the production protocol (evidence only)

How much of the "200 TPE trials + Nelder-Mead" budget is load-bearing? Measured
on the native Metal path over the real 512x512 acquisition (8,937 logical /
2,464 aperture-active BF), seed 42, start `(0, 50, 0)`, **with no change to the
objective, the evaluator, the summation order or any search arithmetic**:
`tests/metal/ssb_fit_budget.swift` runs the production `SSBOptimizer.run` and
`MetalSSBEngine.phaseVariance` and varies only `globalTrials`, plus one
refinement-only arm (`nmWarm`) that starts at the recorded 200-trial optimum.
Every arm runs twice (forward and reverse pass, ABBA pairing for wall time)
under one `gpurun` lock, `GPU_RUN_LABEL=meitner-budget`.

| arm | TPE trials | objective evals | NM evals | best loss | gain vs initial | wall (fwd / rev) | load (fwd) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tpe25 | 25 | 93 | 63 | 0.097227625548839569 | 3.465e-06 | 6.969 s / 6.709 s | 4.92 -> 4.68 |
| tpe50 | 50 | 112 | 58 | 0.096058391034603119 | 1.173e-03 | 8.310 s / 8.339 s | 4.68 -> 5.56 |
| tpe100 | 100 | 162 | 58 | 0.096058391034603119 | 1.173e-03 | 12.587 s / 12.381 s | 5.56 -> 4.84 |
| tpe200 | 200 | 262 | 58 | 0.096058391034603119 | 1.173e-03 | 20.438 s / 20.107 s | 4.84 -> 4.84 |
| nmWarm (NM only, from the recorded optimum) | 0 | 145 | 140 | 0.094526410102844238 | 2.705e-03 | 11.073 s / 11.005 s | 4.84 -> 6.77 |

Machine load over the run: 4.92 at start, 5.33 at end (per-arm pairs above; the
machine was not idle, so quote the load with every time). The three `tpe50`,
`tpe100` and `tpe200` arms reach the *same* optimum to the last bit
(52.871296844577195, 52.878826420320394, 0.0021567655543952434).
`parity-runs/scratch/22_budget_report.py` regenerates this table.

Controls, all measured: the `tpe200` trace is **bit-identical to the recorded
production fit** in `parity-runs/fit-arina-512-full-disk.json` - all 262 trials,
same stages, points, losses, refinement count - in both passes, so the harness
is the production path and nothing drifted; forward and reverse traces are
bit-identical to each other for every arm, so the wall-time pairing is not
hiding a state leak; and the TPE prefixes nest exactly (the first 25/50/100
trials of `tpe200` are the `tpe25`/`tpe50`/`tpe100` trials).

What each block buys, best-so-far inside the 200-trial trace:

| block | best-so-far after the block | gain in block | gain per 25 trials |
| --- | ---: | ---: | ---: |
| 0 -> 25 | 0.097231090068817139 | 0.0 | 0.0 |
| 25 -> 50 | 0.097178995609283447 | 5.209e-05 | 2.084e-06 |
| 50 -> 200 (6 blocks) | 0.097178995609283447 | 0.0 | 0.0 |

Reading, in the order the protocol asks:

1. **TPE converges before trial 50 on this case.** The last improvement in the
   whole 200-trial phase arrives inside the second block; trials 50-200 buy
   exactly 0.0, so 150 of the 200 trials are pure cost here (12.1 s of the
   20.4 s).
2. **25 trials is not enough.** Its TPE best (0.09806397557258606) and its NM
   landing point (0.097227625548839569) are 1.17e-03 worse than the 50-trial
   result; the search has not found the valley yet.
3. **Nelder-Mead, not the trial count, is the binding constraint.** NM buys
   1.121e-03 from the 50-trial TPE best, and a *second* NM started from the
   recorded 200-trial optimum buys another 1.532e-03 - more than the entire
   from-scratch search bought - landing at 0.094526410102844238 in 145 evals
   (11.0 s). The recorded batched draw reaches 0.09447139501571655 in 338
   evals, i.e. the same deeper optimum. Both are consistent with the sequential
   NM stopping early on its own convergence test
   (`coordinateSpread < 0.1 && lossSpread < 3e-6`, `SSBOptimizer.swift`) in a
   shallow spot: stopping NM is what costs precision here, not the 200-trial
   budget.
4. **Timing context.** The user-quoted `228 evaluations / 61.97 s at load 1.86`
   belongs to a different harness (`experiments/20260916-metal-ssb-fit-acceleration`:
   201 trial-stage evaluations in 54.18 s plus 27 refinement evaluations), i.e.
   ~0.27 s/eval. This harness measures the production path at 259 objective
   evaluations (262 recorded trials) in 20.11-20.44 s at load 4.8-5.3, ~0.078
   s/eval, and the recorded `fit-arina-512-full-disk.json` artifact says
   20.331 s. The two are not the same measurement; times are volatile, the
   losses and optimums above are not, and the 61.97 s should be quoted with its
   harness and load, never as a machine constant.
5. **Caveats.** One case, one seed, one start point. TPE's candidate draw is
   seed-dependent and this acquisition's landscape is unusually benign at
   50 trials; "50 is enough" is a statement about this acquisition, not a
   guarantee for others. What generalises is the shape: the trial phase
   saturates, the NM phase is where the remaining loss is, and the recorded
   optimum is NM-termination-limited.

## Findings, root causes and open items

- **F1 (gate fails, MPS, deterministic).** At 512x512 the MPS object is
  4.317e-06 relL2 against the double oracle and 4.179e-06 against native
  Metal, where the independent float32 floor of the identical formula is
  1.098e-06; the `2x` floor bound is 2.196e-06. Same story on two further
  metrics and on the `mps vs metal` pairs. Metal passes every metric at both
  sizes (`metal:cached vs oracle` 5.618e-07 at 512x512, the tightest of all).
  Metal is therefore the reference-quality implementation and the excess belongs
  to MPS.
- **F2 (root cause of F1: not the fused-kernel math mode).** The 13 fused MLX
  kernels are compiled with `math_mode: "fast"`. Re-running the production
  object with `math_mode: "accurate"` under renamed kernels gives a
  **bit-identical** object (`relL2 = 0.0` against the default pipeline,
  `parity-runs/strict-parity/arina-512-full-disk/math-mode-probe-0.json`), so
  the compile option is not the cause.
- **F2b (root cause of F1: not the MLX transform realization).** Feeding MLX
  the oracle's own float64 spectrum and comparing the inverse transform gives
  `relL2 = 7.59e-08` at 512x512 against SciPy complex64's `6.40e-08`; the MLX
  forward `rfft2` plus exact Hermitian completion is `3.46e-07` against SciPy
  complex64's `1.26e-07` on 16 exact count planes. Given the *same* float32
  spectrum, MLX and SciPy produce the same object (2.0080e-06 against
  2.0075e-06). MLX's FFT is therefore as accurate as the reference library, and
  the MPS excess comes from the summed spectrum itself - the per-BF weighting
  and the complex64 accumulation over 8,937 terms - not from any transform.
  Line-level attribution inside that sum is **unresolved**: MPS sits at 1.7x
  (512 setting #1) to 3.9x (512 setting #0) and 2.6x (128 recorded C10
  155.96977) the same-formula float32 floor, while native Metal sits at 0.5x,
  and no single identified line reproduces the difference.
- **F3 (not a cause: the comparison itself).** Image-space and Fourier-space
  accumulations of the same objective agree to 1.35e-15 relL2; block-accumulated
  complex64 costs 8.6e-08 at 128x128 and 2.0e-06 at 512x512. So the oracle and
  the backends evaluate the same objective, and the production MPS error is
  larger than the same-formula float32 emulation at 512x512.
- **F4 (gate fails, MPS, the 2026-09-13 settings).** At C10 = 155.96977 nm the
  MPS loss is 2.8746e-07 relative against a 2.24e-07 `2x`-floor bound
  (1.3x over) and 2.4969e-08 absolute against 1.9457e-08. Its object error at
  the same setting is *better* than the float32 floor (1.8209e-06 against
  5.4456e-06), so the loss metric is the sensitive one there. Deterministic
  across runs.
- **F5 (recorded cached-vs-streamed defect is closed).** The 2026-09-13
  diagnosis recorded 4.43e-05 / 3.46e-05 / 5.79e-05 relative disagreement
  between the cached and streamed objective on a synthetic 512x512 fixture,
  the last exceeding the 5e-5 gate of the time. On the real artifact today the
  same three settings (`arina-128-recorded-c10`, C10 = 0 / 55 / 155.96977) agree
  to 0.0 / 8.665e-08 / 0.0 relative with object relL2 <= 2.14e-07 and absolute
  loss <= 7.45e-09, and across all nine gated cached-vs-streamed rows of both
  recorded reports the worst is 8.665e-08 relative and 3.731e-06 object relL2
  (512x512 setting #1); every row also agrees with the double oracle. The wrong
  path was the cached, half-plane-projected one; the streamed full-plane path
  was right; `4e0f8ab` (Nyquist correction) is the fix. What is stale in
  `experiments/20260913-ssb-loss-diagnosis/README.md` is its *status*: the
  record still presents the defect and its "account exactly for the exceptional
  endpoint contribution" recommendation as outstanding, and it never notes the
  fix that landed; its diagnosis, numbers and fixture are not stale.
- **F6 (BF policy).** The documented 8,937 logical pixels are the 8,938
  geometric disk pixels at center `(94.88451385498047, 96.35952758789062)` and
  radius `53.35992814757164` minus exactly one hardware dead pixel `(78, 74)`.
  The detector declares four dead pixels `(27,135) (78,74) (113,14) (156,13)`;
  only `(78,74)` is inside the disk. 2,464 pixels are aperture-active.
- **F7 (inputs are exact).** `verify_counts_against_hdf5` re-reads the
  acquisition with h5py alone (`hot_pixel_correction="none"`, no loader, no
  dtype conversion) and compares every selected value: 512x512 8,937 BF =
  2,342,780,928 values and 128x128 = 146,423,808 values, **0 mismatches**. The
  export also fixes a real bug: the chunked decode path used scan *rows* where
  flat scan positions are needed.
- **F8 (open, documentary).** Reference-512's own numbers cannot be reproduced
  from this acquisition: documented BF companion maximum count 53 against 1913
  measured here (1.05% of columns exceed 255, so production would never narrow
  to uint8), and `expected.loss` 0.04469207674264908 against 0.098087540782 for
  the same aberration triple. Its `trial_trace_sha256` is not checked anywhere
  in the tree. The reference's source specimen is not in the repository, so this
  stays unresolved rather than "fixed". **Recommendation (not applied here, and
  do not edit the fixture from this experiment):** either re-anchor it -
  regenerate the fixture from an artifact that ships with the tree and add an
  executable check of `trial_trace_sha256`, so the pin is verified instead of
  prose - or delete it. A "frozen" reference that nothing compares against and
  whose numbers contradict the only acquisition present is a trap: it reads as
  authority, it can never fail, and the next agent has no way to tell it from a
  live pin.
- **F9 (fit-trajectory trap, verified).** See the section above: batching moves
  the final optimum by 3.2 nm / 7.3 nm / 0.61 rad at 128x128 and by
  13.8 nm / 13.8 nm / 2.2e-04 rad at 512x512, and the loss by 1.57e-04 /
  1.65e-02 relative, while every shared candidate keeps a bit-identical loss.
  The production path is the sequential one, which is *not* the Python
  reference's default (`optuna_batch_size = 2`). Reproduce with
  `scripts/check_ssb_fit_trajectory.sh`.
- **F10 (harness, open, not a precision finding).** `pytest
  tests/parity/test_ssb_strict_parity.py -q` with no environment fails one test,
  `test_recorded_c10_settings_no_longer_disagree`, with `case
  arina-128-recorded-c10 is missing from the report`. Cause: the pytest fixtures
  measure `FAST_CASES` (the two 128x128 cases) while the recorded-C10 case
  exists only in the report `scripts/check_ssb_parity.sh` writes, so the gate
  must have run once for the suite to be green. Workaround: set
  `QUANTEM_SSB_PARITY_REPORT=parity-runs/gate-fast.json` (14 passed, 1 skipped,
  1.4 s, no GPU). Recommendation: make the fixture measure the missing case
  itself, so the documented one-line command is self-contained; the exit status
  of the suite is not the Metal acceptance signal until then.
