# Exact regional mean diffraction: low-memory decoder improvement

Target: Circle and Rectangle mean diffraction at 120 presented frames/s
(8.33 ms/frame), retaining every detector pixel and exact raw-count sums.
The target is **not met** by this change.

## Implementation and measured bottleneck

The original kernel decoded all 512 positions in each intersecting compressed
block and computed 64-bit row/column division plus circle geometry for every
position and every detector pixel. The candidate computes scan membership once
on the host (one byte per scan position), skips zero streams, directly sums
sparse events, and uses a specialized entropy loop with safe UInt32 block
subtotals accumulated into UInt64. It still validates the complete decoded
streams. There is no detector crop/bin, approximation, hot-pixel exclusion, or
decoded-volume cache. Selection masks are 10,000 / 44,100 bytes for the two scans.

GPU decode remains the dominant cost, not the roughly 1–2 ms array conversion
and readback overhead. These are synchronized backend query times, not native
drag/presentation FPS, and not a hardware-counter/occupancy profile or a
theoretical lower bound. An additional app change avoids calculating the mean
histogram twice; its benefit is not included in this kernel timing table.

## Results

Apple M5, 24 GB; uint8 acquisitions with 864×864 detector pixels. Two warmups
and 20 measured queries per shape/size, offsets 10–14. Times include independent
new means, not cached images. Original-file oracle comparisons are outside the
timed query loop. OS file-page state is uncontrolled; these are resident queries,
not cold file-load measurements.

| Scan / selection | Original p50 ms | Candidate p50 / p95 ms |
|---|---:|---:|
| 100×100 / rectangle 6×6 | 163.49 | 24.92 / 26.78 |
| 100×100 / circle diameter 6 | 164.62 | 25.09 / 26.98 |
| 100×100 / rectangle 12×12 | 255.95 | 36.75 / 49.09 |
| 100×100 / circle diameter 12 | 256.75 | 36.97 / 50.14 |
| 100×100 / rectangle 26×26 | 524.37 | 72.92 / 74.04 |
| 100×100 / circle diameter 26 | 524.27 | 72.38 / 74.08 |
| 210×210 / rectangle 26×26 | not measured | 122.18 / 133.47 |
| 210×210 / circle diameter 26 | not measured | 121.58 / 131.40 |

Both original K3 files pass exact UInt64 sums and correctly rounded Float32
means against direct original-file counts for every detector pixel in the tested
regions (point, rectangle, circle, edge, 12×12 and 26×26). Synthetic uint8/uint16
tests include zero, constant 65535, literal and one/two-event sparse streams,
and a final partial 512-position block. These are regional checks, not an audit
of every possible region in the complete acquisition. Release reports zero
resident bytes. Device allocation at the end of the standalone runs was about
1.35 / 2.61 GB; this is not peak application memory.

A final control rerun using the original kernel resource with the same final
benchmark executable returned 246.82 / 242.84 ms for 12-wide rectangle/circle,
and 503.18 / 507.97 ms for 26-wide rectangle/circle. The speedup persists against
both baseline runs. The control keeps the candidate wrapper's tiny mask setup
but the original kernel ignores that buffer.

## Native acceptance (local development build)

App commit `f910948b2be39b34164c02e2eee591a61725aed8`, backend commit
`43f6105e5dc24ec72c4e28ef1b563e01de983c0a`; installed 0.0.15 build uses both the
new binary and updated packaged Metal source. Its existing Developer ID
signature was refreshed and verified; no new release or notarization.

- PASS: zero → one → zero exact target processes (native test PID 43359).
- PASS: actual native Circle drag, Rectangle drag and corner resize on full K3.
- PASS: mean-DP Linear/Log switching and subsequent region movement.
- PASS: switch to reduced K3, Circle drag and corner resize; region bounds and
  sample counts update with rendered DP and histogram, not just the outline.
- PASS: return to Point, original center-frame hash `7935619298768162159`.
- NOT CLAIMED: 120 FPS, latency inferred from automation call durations, or
  exhaustive rapid-switch/render-frame stress acceptance.

Screenshots retained at `local-evidence://live4dstem-region-performance.apS52D/`
(`full-circle.png`, `full-rectangle-resized.png`, `reduced-circle-resized.png`).
The predecessor app bundle is retained alongside them for rollback. Source files
were not converted or changed. The app still builds against the local backend
override; a remote dependency update/push is outside this local optimization.

## Rejected probes and next bottleneck

1. Membership and sparse handling alone improved about 5×; specializing the
   entropy loop improved the candidate further to approximately 6.6–7.2×.
2. On-demand entropy restart checkpoints every 16 positions were tested with a
   128 MiB cap. A single dense 864×864 block exceeded that admission cap on the
   real K3 data. All real queries fell back to direct decode (zero retained cache
   bytes), so there was no justified reuse benefit. The unqualified checkpoint
   code was removed, not shipped or silently given a larger memory budget.
3. Reusable restart state needs a more selective layout; alternatively, exact
   incremental region sums can avoid unchanged interior blocks on some drags.
   Neither is claimed implemented here. Large-region full-resolution 120 FPS
   still requires a substantial structural reduction in entropy work.

Run `QGPU_REGION_BENCHMARK=1 bash scripts/check_metal_region_mean.sh ORIGINAL
COMPRESSED_COPY` after a release build; `QGPU_TEST_BUILD_DIR` can select the
app's release dependency objects. Source identity, raw samples and controls are
retained alongside this record. No push, release or notarization is implied.
