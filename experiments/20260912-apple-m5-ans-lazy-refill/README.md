# Demand-driven ANS refill: no qualified improvement

Following the checksum ablation, FC15 tests the decoder itself. It refills the
bit reservoir only when the next table entry requires more bits, replacing the
ordinary eager threshold of 32 available bits. Escape handling and all bounds,
state and terminal checks remain unchanged. No extra resident storage is added.
The default stays off.

Seven full uint16 `(512,512,192,192)` sources, identical masks, no crop/bin.
All-seven synchronized image return is measured, excluding loading, validation
and UI. First cycles are warmup; two raw and four indexed repeats remain per
mask. Each cycle starts at zero. Masks describe absolute positions, so the
actual delta depends on the preceding mask. History and trusted-table paths
are disabled throughout. Only FC15 changes within each off/on/off sequence.

| Path / transition | Off A1 ms | On B ms | Off A2 ms |
|---|---:|---:|---:|
| Raw ADF center-1 | 25.477 | 21.668 | 20.658 |
| Raw ADF center-8 | 190.313 | 187.682 | 187.945 |
| Raw ADF center-20 | 257.298 | 282.844 | 276.652 |
| Indexed ADF center-1 | 20.646 | 21.247 | 21.300 |
| Indexed ADF center-8 | 46.356 | 46.259 | 45.925 |
| Indexed ADF center-20 | 73.921 | 77.823 | 76.522 |
| Indexed ADF radius-1 | 86.941 | 90.940 | 88.442 |

The apparent raw center-1 gain disappears against A2. Indexed large movements
are slower. This is not a general optimization and is not promoted.

3360 complete image hashes match 140 independently frozen references exactly.
The adversarial probe passes signed high-count sums including 65535 and rejects
truncated, trailing and nonzero-padding entropy streams. Resident allocation
is 11,877,814,048 bytes for every arm, including the unchanged regional index;
history buffers were not prepared. This is not peak process memory.

The decoder remains the target, but merely delaying refills is insufficient.
Next hypotheses should target dependent table transitions or representation
traversal without adding a dense count copy. No hardware cache/occupancy
counters were collected, so neither a bandwidth bottleneck nor a theoretical
floor has been established. No app, release or public default was changed.
