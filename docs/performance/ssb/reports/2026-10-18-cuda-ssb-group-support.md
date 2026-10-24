# Exact group support in CUDA SSB

## Question and change

Can fine-grid reconstruction avoid optical calculations whose contribution
is exactly zero? Each BF group now caches its coordinate bounds on the GPU.
If neither shifted soft aperture reaches that box, the entire group contributes
zero at that frequency. An outward float32 margin protects the aperture edge.
All BF pixels, nonzero corrections, summation order and precision are retained.
The native-grid and fitting paths keep their existing kernels.

## Measured result

Two full `512 × 512 × 192 × 192` acquisitions used Auto sampling, all calibrated
BF pixels, and their fitted tilt and depth. On an RTX PRO 6000 Blackwell Max-Q,
ten changed-parameter application calls after the first call gave:

| Output | C10 reconstruction median, before → after | Full response median, before → after | After response p95 / max |
|---|---:|---:|---:|
| 1536² | 56.3 → 48.8 ms | 63.6 → 57.1 ms | 57.9 / 57.9 ms |
| 2048² | 97.8 → 84.0 ms | 119.6 → 110.7 ms | 116.2 / 116.7 ms |

Tilt updates showed the same reconstruction benefit. These are resident
reconstruction and local HTTP response timings, not loading, fitting or
displayed FPS. Other GPU activity can increase latency.

Nsight Compute measured 88.9% compute throughput and 14.6% DRAM throughput
in a representative pre-change correction launch, with no register spilling.
The principal remaining cost is optical arithmetic. This measurement does not
establish a theoretical minimum or 60 FPS.

## Validation and rejected alternatives

- All 132 complete application phase arrays were bit-identical to frozen
  pre-change outputs, covering native/Auto and ordinary/tilt-corrected modes.
- 57 CUDA tests passed, including unpruned GPU controls after rotation changes,
  higher-order optics, native diagnostic loss and engine cleanup.
- Frozen fit and native Metal workflow gates passed without updating pins.
- Larger CUDA BF groups saved reduction traffic but slowed the optical step.
- A new Metal support-mask cache gave little joint-reconstruction benefit
  relative to its added memory. The existing Metal kernel was retained.

Private timing distributions, profiler data and application checks are archived
under experiment `ssb-kernel-limits-20261018`.
