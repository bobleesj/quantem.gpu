# WebGPU fused frame-cooperative uint16 + exact clip8 decode

## Question

Can the browser bslz4 decode deliver (a) lossless native uint16 and (b) exact
clip-u8 at speeds near the audited low8 path, so high-count acquisitions (tilt
series with per-pixel counts to ~1900) load correctly without binning and
without the count-audit flag?

## Setup

- Data: one real tilt-series acquisition, 512x512 scan, 192x192 uint16 detector,
  262,144 frames, ~1.9 GB compressed bslz4 across 3 data files, good-pixel max
  1715-1954 per tilt (uint8 wraps or clips this data; low8 wraps).
- Machine: Apple GPU (metal-3) laptop, headed Chrome via CDP, local files
  granted through the widget file input (read-worker path).
- Parity oracle: `checksumFrames([0, 131072, 262143])` (bad pixels zeroed)
  against h5py ground truth computed independently on the acquisition box.
- Timing: `window.__loadprof.totalMs`, 3 reps per config, fresh Chrome profile
  per config after early sweeps showed cross-run contamination.

## Results

| Path | Variant | Median load | Parity |
|---|---|---|---|
| exact u8, OLD default | `fused-clip-u8` (per-block dispatch) | 3396 ms | exact (clip) |
| exact u8, NEW default | `fused-frame-coop-clip8-wg64` | **1199 ms** | exact (clip), bit-exact vs clipped truth |
| lossless uint16, NEW | `fused-frame-coop-u16-wg64` | **1252-1497 ms** | bit-exact vs raw truth incl. max 1715 |
| audited low8 (wraps >255) | `fused-frame-coop-low8-...` | ~1269-1391 ms (other dataset) | lossy by design |
| CUDA warm reference | | 450 ms | |

Decoded stacks: 9.7 GB (u8) / 19.3 GB (u16). The uint16 kernel decodes the FULL
LZ4 stream (low8 stops halfway at the plane boundary) yet lands at low8-class
time: the LZ4 token loop was not the binding constraint at these sizes.

Key structural finding: the old exact-u8 slowness was DISPATCH SHAPE, not
arithmetic. One workgroup per bitshuffle block (`fused-clip-u8`) vs one
workgroup per frame looping blocks (`frame-coop`) is a 2.8x difference on the
same data. The frame-coop clip8 kernel now runs by default for uint16 sources
(no globals, no audit); low8 remains opt-in for audited counting data.

## Swept and rejected (fresh Chrome per config, 3 reps)

| Knob | Result | Verdict |
|---|---|---|
| `__BSLZ4_FRAME_WG` 16 / 32 / 64 / 128 (u16) | 4045 / 1298 / 1252 / 2525 ms | wg64 marginally beats wg32; 16 and 128 are much worse |
| `__QT_H5_LOCAL_GROUP` 2 / 4 / 8 (clip8) | 1453 / 1382 / 1199 ms | keep group 8; smaller groups do NOT deepen the pipeline usefully |
| `__QT_H5_DECODE_BATCH` 8 | 1284 ms | no win over default |
| `__BSLZ4_UPLOAD_WRITEBUFFER` | ~2700+ ms | keep staging pipeline |

Early sweep rounds without Chrome restarts showed 2-4x inflated numbers on
later configs (VRAM accumulation across 19.3 GB decodes). Any future browser
decode benchmark must restart the browser between configs.

## Remaining gap to CUDA (450 ms)

Stage split at best config: GPU decode wait ~530-650 ms, compressed upload
~400-500 ms (partially overlapped), read workers overlapped. Effective decode
write throughput ~21 GB/s vs ~400+ GB/s device bandwidth: the kernel is
latency/occupancy-bound in the serial LZ4 token loop (2 workgroup barriers per
token), not bandwidth-bound. Knobs are exhausted; closing the remaining ~2.6x
needs kernel redesign: subgroup-cooperative token parse, cross-block software
pipelining (decode block N+1 tokens while unshuffling block N), or fpw>1 frame
batching per workgroup.

## MPS end-to-end (same 4 tilts, no binning)

Fresh venv on the laptop from freshly built wheels (widget rc30 + gpu rc5),
zero-copy Metal chunk path: per-tilt load 1.26-1.63 s for 19.3 GB native uint16
(11-16 GB/s), 3 chunks per tilt, `free()` between tilts, peak RSS 22.1 GB,
counts preserved (max 1913/1913/1873/1954). PASS. (The 18-35 s BF-VI in that
harness is a numpy CPU sum in the test script, not the MPS VI kernels - do not
read it as widget interaction speed.)
