# Coordinated ANS submission and split-packet reduction

The original update already applies the signed entering/leaving detector mask.
The existing task group overlaps synchronous calls, but does not coordinate
their commit boundary. Compare it with preparing every source's command before
committing all seven and waiting for completion. This is coordinated submission,
not one fused seven-source dispatch.

Separately, split the selected detector pixels across 2/4/8 packet owners.
Each owner keeps the same small threadgroup accumulator and atomically adds
its final 512-scan contribution to the existing output. Exact modular UInt32
addition preserves signed deltas; there is no additional decoded image volume.
The variant trades more parallel work for additional output atomics.

Use one resident load per executable, with A/B/A trials and the frozen 140
independently checked reference-map hashes. New host code requires one
benchmark-process restart; switches between prepared variants do not reload.
No UI, image atlas, or production default promotion is part of this experiment.

## Completed comparisons

Twenty configurations ran on one resident load, including five- and seven-cycle
confirmation arms. All 11,760 full-map hashes matched the frozen independent
140-map reference; in-process full-array parity also passed. Hashing and loading
are outside the all-seven synchronized API timer. First cycles are excluded.

Coordinated submission did not beat the existing concurrent calls. Raw ADF
center-1 measured 25.673 / 25.984 / 21.991 ms in control/batch/control order.
Packet splits of 2, 4, and 8 likewise did not establish a consistent gain.
The partial-store/index combination improved some cases but regressed others.
None qualifies as a new default or a general fourfold acceleration.

Rechecking the existing exact zero-versus-previous base selection gave a
repeatable benefit for one large transition, without an ADF image atlas:

| Ordered transition | Control before (ms) | Choose base (ms) | Control after (ms) |
|---|---:|---:|---:|
| ADF center-1 | 18.583 | 20.532 | 21.522 |
| ADF center-8 | 42.072 | 40.775 | 44.589 |
| ADF center-20 | 73.132 | 81.538 | 71.344 |
| ADF radius-1 | 84.215 | 20.896 | 83.379 |
| BF radius-1 | 42.754 | 8.457 | 43.010 |

These are seven-cycle arms 18/19/20, not new implementation of base selection.
The radius-1 label is **not a pure one-pixel resize**: it follows the off-center
center-20 mask and returns to the detector center. This is a large combined
center/radius transition. Do not claim fourfold faster ordinary dragging.
Center-20 regresses in the repeat. Small ADF drags remain above the 8.33-ms
budget, and no displayed-FPS result is claimed.

Initial resident storage was 11,877,814,048 bytes, including 2,603,403,760 bytes
of the existing exact regional index (unused by raw arms). Partial trials grew
retained scratch by 58,720,256 bytes; subsequent arms stayed at 11,936,534,304.
Metal current allocation was 11,942,641,664 bytes afterward. These are not
peak process memory measurements. Batch and split variants added no decoded
4D storage. No GPU occupancy/cache-counter or theoretical-floor claim is made.

The small mixed-data split probe passed 2/4/8-way signed reductions, including
65,535 values, sparse/raw/entropy streams and partial lane groups. Its other
malformed-stream tests passed as well. It supplements the seven-source checks.

Read-only batch review found no blocking lifetime or ordering issue: locks use
pointer-address order and every submitted command is drained after failure.
Cancellation remains synchronous; a prepare failure can update diagnostic
metadata or grow scratch without changing the last completed detector image.
No failure-injection or UI test of this new batch API was performed here.
