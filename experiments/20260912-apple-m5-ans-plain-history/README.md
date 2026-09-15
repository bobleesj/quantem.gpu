# Exact bounded detector-result reuse

Seven full 512×512×192×192 uint16 acquisitions remain ANS resident throughout 15 arms. This is a compute experiment, not a displayed-FPS or loading benchmark. No cropping, binning, clipping, decoded 4D duplicate, or precomputed ADF atlas is used.

## Results

One previous output image and its exact mask are retained per acquisition. Immediate A/B backtracking swaps the previous image into use instead of recomputing it. The additional GPU allocation is 7 MiB total; host masks add approximately 258 kB. Resident allocation is 11,885,154,080 bytes.

| Work | All-seven synchronized median |
| --- | ---: |
| Fresh misses within the alternating trajectory, arm 10 | 21.827 ms |
| Exact backtracking hits, arm 10 | 0.289 ms |
| Repeated backtracking confirmation, arm 15 | 0.290 ms |
| Backtracking confirmation p95, arm 15 | 0.402 ms |

There are 84 warm all-seven hits in each of arms 10 and 15; arm 10 also has 12 warm fresh updates. Cycle zero is excluded. Hits are deliberately separate from misses: this does **not** establish 120 FPS for new detector regions. The ordinary 20-mask trajectory remains around 21–22 ms for the one-pixel large-ADF center move.

Single-writer plain shared accumulation (function constant 12) did not provide a reproducible fresh-compute improvement. Indexed ADF center-one controls/candidate/control measured 20.725 / 22.147 / 21.016 ms. Keep it disabled. Smaller coordinated batches and register reduction were also unsuccessful in the preceding reduction-bounds experiment.

## Correctness and scope

9,520 full-map hashes match 140 frozen reference maps exactly, including repeated masks. The benchmark also checks repeated outputs before replacing prior results. The GPU adversarial probe passes mixed dense/sparse streams, signed cancellation including 65535, packet splitting, and malformed entropy cases.

History is experimental and opt-in with `QGPU_PAIRED_RUNTIME_HISTORY=1`. No installed application or release default was changed. Runtime toggling and ordinary/coordinated return visits passed. Failure rollback was reviewed, but GPU fault injection was not performed. Coordinated batch publication is not all-or-nothing; do not claim transactional batch semantics.

## Reproduction and interpretation

`manifest.json` pins the tested executable, shader and resident-source hashes because the checkout is dirty. `resident-loop.jsonl` contains every arm; `summary.json` summarizes fresh trajectories; `reuse-summary.json` separates complete seven-source hits from computed updates. Keep the full trajectory order: a radius-named mask following an off-center mask is not a pure radius adjustment.

The extra allocation is small, not proof of zero memory pressure. The machine had existing swap usage. No measured cache-miss, occupancy, or complete bandwidth-floor result is available; timestamp counters alone cannot establish those properties.
