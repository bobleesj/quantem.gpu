# 20260916-timeline-fit — GPU-timestamp verification of the 8937-BF Metal fit floor

## Question

An earlier verdict held that the 8937-BF Metal fit (267.1 ms/eval) is within
2–5% of the achievable floor, derived from streaming controls and a pass-rate
model. That reasoning can be circular: a model can fold host-side overhead into
"the rate". This experiment re-tests the verdict with a method that cannot be
circular — `MTLCommandBuffer` GPU timestamps recorded for every one of the
228 objective evaluations of a real fit.

Falsifiable claim under test: *the fit's wall time is GPU execution; no gap,
synchronization stall, or host-side category above 5% is hidden inside the
quoted per-evaluation rate.*

## Method

- Measurement-only instrumentation, no change to the loss math:
  - `MetalSSBTimeline.swift` records, per command buffer, the monotonic wall
    time before `commit()` and after `waitUntilCompleted()`, plus
    `gpuStartTime`, `gpuEndTime`, `kernelStartTime`, `kernelEndTime`; and, per
    evaluation, the wall span, the summed GPU seconds, and the command-buffer
    records in submission order.
  - `MetalSSBKernels.swift` gains one optional `timelineRecorder` property plus
    hooks in `commitAndWait` and `phaseVariance`. With no recorder installed the
    path is unchanged (one optional-chaining check per commit).
- `probe-timeline.swift` reproduces the probe5 fit exactly: same input,
  calibration, BF selection (8937 terms), rotation, start point, seed 42,
  200 TPE trials plus Nelder–Mead.
- Two arms, same executable and output directory (catalog index reused), run
  under the shared `gpurun` GPU lock:
  1. timeline on (`SSB_TIMELINE=1`) — writes `timeline.jsonl`
  2. timeline off (`SSB_TIMELINE=0`) — paired control
- `analyze_timeline.py` computes every category from *differences only*, so
  monotonic wall seconds never mix with GPU-timestamp seconds.

## Precision and instrumentation overhead

| arm | fit | loss | best (C10, C12, phi12) |
|---|---|---|---|
| timeline on | 62.184 s | 0.13769799470901489 | 6.6036715919750746, 0.098487623142316849, 1.1341200767509283 |
| timeline off | 62.349 s | 0.13769799470901489 | 6.6036715919750746, 0.098487623142316849, 1.1341200767509283 |

Loss and optimum are bit-identical between arms; the instrumentation costs
−0.26% (i.e. below run-to-run noise, the "on" arm being marginally faster).
The frozen loss and optimum used by the SSB gates reproduce exactly.

## Results — 228 evaluations, ARINA 512x512x192x192 uint16, 8937 BF

Fit span (first evaluation start to last evaluation end) 61.403 s; fit wall
62.184 s (0.78 s pre-roll before the first evaluation).

| category | seconds | % of fit span |
|---|---|---|
| GPU busy, total | 61.023 | 99.4% |
| — GPU busy, cache-accumulate command buffer | 60.899 | 99.2% |
| — GPU busy, clear/tables command buffer | 0.124 | 0.2% |
| — GPU busy, streamed tail | 0.000 | 0.0% |
| GPU idle inside evaluations | 0.357 | 0.6% |
| — of which CPU encoding of the cache command buffer | 0.215 | 0.4% |
| — of which commit/wait minus GPU execution | 0.102 | 0.2% |
| — of which CPU reduction tail | 0.039 | 0.1% |
| optimizer between evaluations (TPE/NM bookkeeping) | 0.023 | 0.04% |
| unaccounted | 0.000 | 0.0% |

Gap distributions:

- intra-evaluation, between the two command buffers: n=228, p50 1.154 ms,
  p95 1.218 ms, max 1.663 ms
- inter-evaluation GPU idle: n=227, p50 0.509 ms, p95 0.613 ms, max 0.789 ms

Per evaluation: wall p50 275 ms / p95 281 ms / max 294 ms; GPU p50 274 ms /
p95 279 ms / max 293 ms; exactly 2 command buffers every time.

No category larger than 5% exists outside GPU execution of the loss; the
largest non-GPU category is 0.4%.

## What the timestamps cannot show

`kernelStartTime`/`kernelEndTime` returned ~0 on this device (0.022 s summed
against 61.0 s of GPU busy), so a timestamp-based row-pass/column-pass split is
**not** established here. The source of the intra-evaluation ~275 ms is a
single fused command buffer; separating its kernels requires `MTLCounterSample
Buffer` encoder sampling, which was not implemented because no category
exceeded the 5% stop threshold. Occupancy/stall reasons are likewise not
observable from timestamps.

## Verdict

The timeline corroborates the floor verdict; it does not expose a hidden
category. 99.4% of the fit span is GPU-executing time, the optimizer's own
bookkeeping is 0.04%, and all host-side work inside evaluations (encoding,
synchronization, the 262144-pixel reduction) sums to 0.6%. The quoted per-eval
rate is therefore not diluted by host overhead: the remaining lever is moving
the same bytes faster or moving fewer bytes, not removing host work. This is
consistent with the earlier streaming-model result (2-pass model 266 ms versus
267.1 ms measured at load 1.86; 274 ms p50 here at load 2.7–5.0, i.e. ~2.6%
load sensitivity, both inside the quoted 2–5% band).

## Evidence

Retained under `perf-lab/ssb-audit/metal-runs/timeline-fit/`:

- `timeline.jsonl` — 228 evaluation records with per-command-buffer wall and GPU
  timestamps (sha256 `7aa4ac453ed20b6a153727aa6ebe5bf13030d245e298421352e76bf08f96a23e`, 124935 B)
- `summary-timeline.json` (sha256 `f7d4c3201745cb7d86ad87c8461305dd7b717ebebafa035129b9df0a5a121b7e`, 512 B)
- `summary-plain.json` (sha256 `9cd1235e618733b4ef18884b865c020749838b49badb4ad9728b61ba2096bcc5`, 511 B)
- analysis `perf-lab/ssb-audit/timeline-runs/timeline-analysis.json`

## Limitations

- One dataset (ARINA logic-pmos master, aggregate hash
  `57cb94904ffcf7a5b2470174b6e3b9d819a80b96e565b69772ce4fdf0a5cf95b`), one host
  load band (2.7–5.0), one session; absolute p50 will move with host load.
- No per-kernel split and no occupancy counters (see above).
- `gpuStartTime/gpuEndTime` bound command-buffer execution on the GPU; they do
  not distinguish useful progress from memory stalls.
