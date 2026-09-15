# Seven-source stage timing

Seven distinct full uint16 `(512, 512, 192, 192)` acquisitions on Apple M5
24 GB. No count changes, cropping, binning or algorithm changes. All-seven
synchronized backend return is timed; loading, hash validation and UI are not.
The first cycle is warmup, leaving two raw and four indexed samples per mask.
Each cycle starts at a zero mask. Transition names describe absolute masks,
not displacement from the immediately preceding mask.

| Indexed transition | Wall ms | GPU command union ms | CPU preparation union ms | Residual encoder union ms | Index encoder union ms |
|---|---:|---:|---:|---:|---:|
| ADF center-1 | 21.006 | 19.311 | 1.922 | 18.150 | 7.399 |
| ADF center-8 | 45.949 | 44.203 | 1.909 | 29.960 | 22.303 |
| ADF center-20 | 75.891 | 74.028 | 2.081 | 51.698 | 44.225 |
| ADF radius-1 | 86.666 | 84.946 | 1.989 | 47.036 | 53.413 |

These intervals overlap across command buffers. Do not add columns. Encoder
duration includes its execution interval under contention, not hardware busy
cycles. Residual is fused ANS decode plus accumulation, not decode alone.
Only timestamp counters are available: occupancy, cache misses and achieved
memory bandwidth remain unmeasured.

## Controls and correctness

Indexed profile off/on/off center-1: 21.710 / 21.006 / 21.442 ms;
center-20: 72.629 / 75.891 / 74.933 ms. Raw center-1:
25.321 / 25.980 / 25.020 ms. Sampling does not produce a consistent large
change, but small gains cannot be established by these short diagnostic runs.
All seven sources have valid timestamps in these reported cases. Full-map
validation checked 4060 hashes against 140 independently frozen reference maps.

The shared plan cache is disabled in this configuration. CPU preparation is
about 2 ms and host output copies about 0.2-0.4 ms; reducing those alone cannot
bring 21 ms to the 8.33 ms target. Large movements exercise both indexed queries
and the fused residual kernel. Decode versus reduction still needs an ablation.

Resident bytes: 11,885,154,080, including the existing 2,603,403,760-byte index
and prior-result buffer. These remain allocated in raw mode but are unused when
disabled. No installed app changed; these are not on-screen FPS measurements.

## Large ADF: one complete 74.565 ms timeline

Sequence 5, cycle 1, `adf-center-20` in results.jsonl is a concrete example,
not a sum of medians. The preceding mask is `adf-center-8`; their centers are
(5,8) and (12,20), so this is a (7,12)-pixel translation at fixed radii48/94.
All seven sources report6269 changed detector pixels, reduced by the index to
371 selected fields and1067 residual detector pixels per source. Logical work:
1,957,953,536 residual count positions and680,787,968 indexed field values.
These are logical positions, not measured bytes or instruction counts; sparse
and constant modes can avoid some count-by-count work.

Partitioning the GPU timestamp span into non-overlapping categories gives:

| Category | ms |
|---|---:|
| Index encoder intervals only |20.117|
| Residual encoder intervals only |22.086|
| Both encoder types overlap across datasets |30.634|
| Neither sampled encoder within GPU command span |0.156|
| Outside first-to-last GPU command span |1.573|
| Total all-seven return |74.565|

Encoder intervals indicate timestamp coverage, not measured ALU activity or
GPU occupancy. The final outside-span value is the wall-time remainder, not
the sum of CPU work. CPU/GPU work can overlap.

Relative to first source request entry:

| Source | Submitted ms | Index interval ms | Residual interval ms |
|---|---:|---|---|
|1|1.452|1.498-7.752|18.684-25.345|
|2|1.301|1.392-6.071|7.765-16.498|
|3|1.726|7.765-18.649|32.714-42.621|
|4|1.623|18.685-32.310|32.714-48.172|
|5|1.983|25.346-32.704|48.186-55.099|
|6|2.055|48.186-56.674|56.688-65.016|
|7|1.738|56.688-67.687|67.758-74.384|

Submission is concurrent but execution is not seven independent full-speed
GPUs. Source7 waits roughly55ms after commit before its first encoder starts;
other sources also have between-encoder scheduling gaps. These waits overlap
other GPU work and must not be counted again as additive overhead. This shows
partial serialization/competition, not a measured cause such as register
spilling, bandwidth saturation or an occupancy limit.

Four warm repetitions measured67.877-77.446ms. One repetition had no overlap
between sampled index and residual stages, illustrating scheduling variability.
The raw positive-mask checksum ablation is separate evidence: it suggests most
residual-path cost remains without image accumulation, but it does not isolate
this exact signed indexed residual stage. Next isolation should compare each
stage alone and together on the same residual plans; increasing submission
concurrency by itself is not the missing implementation.
