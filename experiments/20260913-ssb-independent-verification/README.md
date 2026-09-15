# Independent full-aperture SSB verification

The current-source reproduction uses the existing original-file probe with
all 8,937 BF terms, a full 512×512 scan and 192×192 detector. Input master
SHA-256 matches the September 11 experiment. This is GPU objective timing,
not UI frame rate. No scan cropping, binning, or precision reduction.

## Initial reproduction

- Objective: mean 263.779 ms, p50 263.224 ms, p95 266.350 ms.
- Object redraw: mean 76.698 ms, p50 76.361 ms, p95 78.848 ms.
- Original-file load: 2.793 s; SSB preparation: 3.719 s.
- 200 trials plus refinement: 58.866 s, 34 refinement evaluations.
- Final loss: 0.136068195104599; best parameters exactly match the recorded run.
- Sampled Metal allocation: 12,573,016,064 bytes (not a peak-memory measurement).
- The historical 222.7 ms / 55.17 s performance was not reproduced in this run.
  No adjacent pre-optimization baseline was run, so this does not establish
  either a regression or the earlier percentage speedup.

Raw reports and complex object arrays are retained locally under
`local-evidence://ssb-independent-20260913/run1/`. The report SHA-256 is
`622991e63594df99707105bd4cd0e2e7e9ef575f09b73673ab594302650c8697`.
Initial executable SHA-256:
`ce79b40bb6e9e4237b42bd34f9529f563d67f77a0ac0863ac3dd6024e9ba9740`.

## Cleanup and gates

The cleaned-build repeat measured objective mean/p50/p95 235.664 / 235.745 /
237.101 ms, object mean/p50/p95 74.575 / 74.237 / 77.630 ms, and a 55.891 s
fit. Load was 2.610 s and preparation 2.922 s. Allocation was unchanged.
Three complex object arrays have relative L2 difference 0.0, and all sampled
loss values have absolute difference 0.0 against the initial run. Final fit
parameters, loss and 34 refinement evaluations also match exactly.
This is a before/after cleanup comparison, not a reproduction of the original
pre-blocking baseline. Do not attribute the timing variation to dead-code removal.

Cleaned report: `local-evidence://ssb-independent-20260913/cleaned/report.json`,
SHA-256 `6ea099a70ca9334be830491095919f27d56d1b5d4b2987834d5465a6c042ae3c`.
Cleaned executable SHA-256:
`4d5dbfe3ebe1ffa4df8b6de8b699ac606acd159ed2110094733c61866d6103f6`.
Cleaned tracked-source diff SHA-256:
`ff9cf0d4a30b0b1fd714f2fb0f66f3c9e189bdcdb5d1638401dd8daeffbee58f`.

The old probe build script globbed stale object files, causing undefined
symbol errors. It now uses SwiftPM's active product object lists. The unused
half-plane loss-correction pipeline, kernel, and unreachable boolean branch
were removed. Cached loss still uses the fused blocked intermediate; streamed
loss retains its full-plane implementation. A one-use DC tuple wrapper was
inlined. No public API was removed. The historical pipelined prototype patch
was preserved as experiment evidence; it is not a production caller.

The standalone test now checks cached/streamed loss as well as object output,
using the existing XCTest relative tolerance of 5e-5. It fails reproducibly:
cached loss 0.17328005, streamed loss 0.1730958, relative error 0.0010632381
(0.1063%). The tolerance was not changed. The previous 18-pass suite did not
cover this comparison. This failure blocks scientific parity signoff; its
cause is not established by this cleanup review.

The current worktree also contains pre-existing streamed command batching and
unrelated ANS changes, all preserved. Nothing is committed or published.
