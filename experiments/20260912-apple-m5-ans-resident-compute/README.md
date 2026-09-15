# Resident-only seven-source ANS computation

Target: seven exact UInt32 detector images from full uint16 512×512×192×192
acquisitions in 8.33 ms. Keep the seven residents alive between matched trials.
No UI testing, no saved ADF image atlas, no dense duplicate 4D data.

Controls and candidates must use identical masks and validity. Hashing and full
array comparisons occur outside timed intervals. Report raw/incremental ANS
separately from existing polar-index-assisted computation. Use A/B/A repeats;
do not treat a reciprocal compute time as visible FPS.

## Floor accounting

The seven UInt32 outputs contain 7,340,032 bytes. An exact incremental update
must preserve the previous sum or reconstruct it; a full read/write of those
outputs alone is 14,680,064 bytes. This is only an output-traffic bound, not a
measured hardware latency floor. Compressed input traffic depends on the actual
selected streams and their modes. Entropy streams have 256 dependent pair
decodes per 512 scan positions, so aggregate bandwidth alone cannot predict
runtime. Occupancy, instruction dependencies, sparse-event contention, partial
scratch traffic and submission overlap still need measurements.

## Completed resident sweep

Thirteen arms ran on one load of the same seven residents: 900 complete
seven-source updates, 6,300 full-map hashes matching the 140 independently
checked reference maps from the preceding experiment. Every cycle also passed
full-array equality. Loading and hashes are outside the timed query interval.
The first cycle of each arm is excluded below.

The last matched raw-ANS A/B/A repeat (five cycles per arm) measured:

| Transition | Control before (ms) | Partial stores (ms) | Control after (ms) |
|---|---:|---:|---:|
| ADF center +1 | 23.997 | 21.946 | 24.822 |
| ADF center +8 | 196.440 | 183.572 | 186.816 |
| ADF center +20 | 261.364 | 251.220 | 258.315 |
| ABF center +1 | 17.018 | 14.174 | 16.400 |

These labels identify an ordered mask sequence, not isolated movements from
the same base for every row. All arms use the identical sequence. This shows
modest case-dependent gains, not fourfold improvement. Four streams per thread
regressed badly; one did not reliably improve on two. The regional-index arms
remain separately labeled in `summary.jsonl`; they are not raw-decode results.
No UI or presented-FPS conclusion follows from these compute measurements.

The new FC8 partial-store pipeline is opt-in. It replaces unique-owner dense
atomic adds with stores, initializes partials on the GPU, and synchronizes
before sparse atomic updates. Exact outputs are unchanged. Larger partial
requests fall back to packet-owner reduction above 128 MiB per source.
Partial scratch grew aggregate retained allocations from 11,877,814,048 to
12,641,177,376 bytes, then stayed constant across later arms. This includes
2,603,403,760 bytes of existing regional indices, present but unused in raw
arms. No 4D duplicate or precomputed ADF image atlas was created.

The fourfold hypothesis is refuted for these tested variants. The machine's
theoretical minimum is **not established**: no occupancy or cache-miss counters
were measured. The next investigation should target the dependent ANS decode
loop and collective reduction, rather than assume memory bandwidth is the
only limit.

## Resident protocol

Launch the series benchmark with `QGPU_ANS_RESIDENT_LOOP=1` and send JSON lines:

```json
{"command":"run","arm":"A1","mode":"raw","cycles":3}
{"command":"run","arm":"candidate","mode":"raw","kernel":"partials","partial_stores":true,"cycles":3}
{"command":"run","arm":"A2","mode":"raw","kernel":"packet-owner2","partial_stores":false,"cycles":3}
```

Prepare FC8 at load with `QGPU_PAIRED_RUNTIME_PREPARE_PARTIAL_STORES=1`.
Optional indexed comparisons require `QGPU_PAIRED_RUNTIME_POLAR_INDEX=1` at
load. Explicit `{"command":"quit"}` releases the process. Between commands
the process waits without GPU computation and retains the seven acquisitions.

No default promotion, UI edits, commits, or pushes were performed in this round.
