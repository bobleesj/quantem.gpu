# CUDA SSB refinement batching

> **Historical:** describes the code as of 2026-10-15; the Nelder–Mead joint refinement it batches and the phase-variance search are retired. See the [current fit methods](../../../api/direct_ptycho.md).

The joint SSB search now evaluates the independent Nelder-Mead starts together.
It keeps the existing initial trial batches, model, precision and optimizer
settings. The public `SSB.find_aberrations` and reconstruction API is unchanged.

## Matched full-acquisition comparison

The baseline is `81b9a7db`. Both acquisitions retain the complete
512 × 512 × 192 × 192 uint16 input, with 8,807 and 8,930 selected bright-field
pixels. Measurements used a reserved RTX PRO 6000 Blackwell Max-Q at its
300 W power limit. Each version ran three times in
baseline, candidate, candidate, baseline, baseline, candidate order.
The shared numerical environment was Python 3.14.6, CuPy 14.2.0 and
SciPy 1.18.0.

Both versions used seed 42, 200 reference trials, 200 joint trials and three
Nelder-Mead starts capped at 200 iterations each, with `xatol=0.05` and
`fatol=1e-6`. The benchmark sets that cap identically on both sides; the
package's existing 400-iteration default is unchanged.

Mean prepared joint search and save:

| Input | Sequential | Batched | Less time |
| --- | ---: | ---: | ---: |
| A | 5.93 s | 3.93 s | 33.8% |
| B | 10.16 s | 6.56 s | 35.4% |

These are means of three synchronized observations. The respective ranges
were 5.77–6.20 versus 3.82–4.05 seconds, and 10.00–10.27 versus
6.51–6.63 seconds. The timer covers joint search, native reconstruction,
rotation checking and saving. It excludes source loading and preparation.
The model searches C10, C12 and its angle, two specimen-tilt components and
depth spread; these measurements do not cover higher-order aberrations.

Automatic final reconstruction, native loss and saving added 0.38 seconds
for A at 3× sampling and 0.54 seconds for B at 4× sampling. Including that
output stage gives prepared totals of 4.31 and 7.10 seconds. Source loading
was about one second and preparation about 0.20 seconds in one observation
per dataset, with uncontrolled filesystem cache state. These are separate
measurements, not a cold end-to-end latency distribution.

## Why this saves time

The three SciPy solvers retain their own state and submit one candidate each.
The calling thread alone owns device evaluation. A shared model evaluation
reuses data and geometry across those candidates, then returns each score to
its original solver. Objective calls fell from 963 to 359 on A and from
966 to 361 on B. The total numbers of evaluated parameter vectors remained
1,313 and 1,316, including the 400 initial trial vectors.

## Numerical verification

Every trial record, refinement candidate, refinement score, final coefficient,
tilt, depth, diagnostic loss, gain and native/fine phase array matched the
frozen baseline bit for bit in all twelve final public-API runs.

The first candidate changed some scores despite identical pre-sum model
contributions. CuPy selected different reduction behavior for rows at different
memory offsets. Copying each refinement row into aligned storage before its
singleton reduction restored the original results. The initial Optuna batches
keep their existing reduction policy. This preserves both float32 arithmetic
and each scalar search trajectory rather than relaxing a tolerance.

Fourteen focused tests passed on the physical CUDA device, covering known
tilted and untilted simulations, the existing Newton and previous-scan paths,
zero-depth equivalence, the Torch comparison, partial BF tiles, repeated
objectives, independent-batch equality and scheduler behavior.

Earlier timings that overlapped another GPU job are excluded. The direct
batching candidate and the unsuccessful summation-only candidate remain in
the private evidence record. No input was cropped, binned or downcast. No
bright-field support or search effort was reduced. The initial diagnostic
rounding issue was corrected before retaining the implementation.

## Implementation and regression checks

| Responsibility | Source |
| --- | --- |
| Independent SciPy searches and batched scheduling | `src/quantem/gpu/direct_ptycho/batch_refinement.py` |
| Joint model search orchestration | `src/quantem/gpu/direct_ptycho/thick_sample_fit.py` |
| CUDA model scoring and singleton reduction | `src/quantem/gpu/direct_ptycho/cuda/thick.py` |
| Enabling the CUDA refinement batch | `src/quantem/gpu/direct_ptycho/cuda/optimizer.py` |

From a checkout with the CUDA and development dependencies installed, run:

```bash
PYTHONPATH=src python -m pytest -q -m "" \
  tests/contracts/test_ssb_batch_refinement.py \
  tests/contracts/test_ssb_phase_variance_fit.py \
  tests/hardware/cuda/test_ssb_thick_sample.py \
  tests/hardware/cuda/test_ssb_independent_refinement.py
```

The synthetic fixtures and scheduler checks are public and reproducible.
The measured acquisition timings require the retained private inputs.
Exact matching is established for the tested device and numerical stack;
a different device or library version requires its own parity check.

Evidence identifier: `20261015-cuda-batched-refinement`. Raw acquisitions,
logs, manifests, arrays, traces and visual comparisons remain in the private
evidence archive. This work does not qualify another accelerator, native
application integration or a published package release.
