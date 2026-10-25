# Does the live registration need its notch from more than frame 0? (2026-10-24)

## Question

Live placement (`ssb/stream.py`, `_lattice`) finds its Bragg spots and its notch (every other spot, cut from the aperiodic part) on frame 0's power spectrum alone. One frame's spectrum is speckled: frame 0 of frames 151-191 gives 2860 notch peaks where the series mean gives 80, and the notch keeps 61 % of the coarse band (periods 3 to 50 Å) instead of 99.7 %. Does that hurt placement, and would a notch from the running mean power of the frames so far (refreshed every frame) do better, especially at low dose?

## Setup

- frames 151-191, 40 SSB phase frames, 0.1243 Å a pixel (`$QUANTEM_DRIFT_BASELINE/series_phases.npz`). Reference: the frozen pairwise `drift()` (`baseline.npz`, `series_drift_displacement`).
- Live chain emulated with the package's own `_start`, `_guess`, `_place` and `CONFIDENCE`, `REACH_A`; only the lattice (spots, notch) changes:
  - frame0: from frame 0 (current code),
  - running: from the mean power of frames 0..f-1, recomputed before each frame,
  - series: from the mean power of all 40 frames (what `drift()` uses; an oracle a stream does not have).
- Lower dose emulated by white noise added to every frame, sigma = k times that frame's phase std (seed 7). Ducky phase stacks were not found on the test machine, so no Ducky run.
- GPU 0, script `notch.py` and `pair.py` in this session's scratchpad.

## Results

Live placement against the pairwise reference, max |error| over 40 frames (Å), frames off by more than 1 Å:

| noise k | pairwise drift() | frame0 | running | series |
|---|---|---|---|---|
| 0 | 0 (is the reference) | 0.087, 0 | 0.236, 0 | 0.235, 0 |
| 0.1 | 0.002, 0 | 0.101, 0 | 0.237, 0 | 0.235, 0 |
| 0.25 | 0.003, 0 | 0.102, 0 | 0.236, 0 | 0.233, 0 |
| 0.5 | 0.007, 0 | 0.099, 0 | 0.238, 0 | 0.231, 0 |
| 1 | 0.014, 0 | 54.8, 29 | 62.3, 28 | 62.3, 28 |
| 2 | 3.3, 32 | 66.7, 36 | 135.7, 35 | 135.7, 35 |

- All 40 frames stay confident through k = 0.5 in every variant (lowest confidence 0.040 to 0.052).
- The error is the chain's random walk: frame0 and series agree to 0.01 Å over frames 0-19 and separate over frames 28-39 (0.087 vs 0.235 Å at frame 39, 0.7 vs 1.9 px). Neither takes a lattice image (3.2 Å).
- The running notch converges to the series one (73-80 peaks) after a few frames and then behaves like it. Its refresh costs 1.4-1.6 ms a frame (12.8 ms at k = 0, first frames' many peaks).

## Conclusion

No change. The frame-0 notch does not hurt: it places the series closer to the pairwise drift than the series-mean notch (0.09 vs 0.24 Å over 40 frames) and survives the same noise. A running-mean notch costs time and lands on the series result. Where the live chain fails (k = 1, 29 of 40 frames wrong) the pairwise drift still holds to 0.014 Å with the series-mean notch, so the low-dose weak link is registering each frame against one earlier frame, not the notch. `SSBStream.align()` (pairwise) remains the answer for low-dose series; a live reference with more dose than one frame is the place to look if live placement must work there.

## Rejected

- Running-mean notch refreshed each frame: converges to the series notch, which is 2.7x further from the pairwise drift live, plus 1.5 ms a frame.
- Taking the series notch for live placement: not available to a stream, and no better (above).
