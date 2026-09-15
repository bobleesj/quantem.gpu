# Phased independent table reads

FC18 preloads the two existing readers' table entries before advancing either
reader. No new resident buffers; ordinary default remains off.

20 measured trials per arm plus one warmup, seven full uint16 sources and
the identical signed ADF transition. All 294 source transitions passed exact
stage parity. Signed high-count and malformed-stream fixture checks pass too.

| Arm | Residual mean / median (ms) | Combined mean / median (ms) | Combined p95 (ms) |
|---|---:|---:|---:|
| Original |44.55 /45.86|67.32 /67.75|71.68|
| Phased |43.11 /44.03|66.27 /66.66|72.60|

Residual paired ratio: 0.945 (first half 0.950, second half 0.931).
Combined paired ratio: 1.003 (first half 1.010, second half 0.996).
The small residual improvement merits follow-up, but no combined speedup
qualified. Identical index code also varied; GPU scheduling/state variation
cannot be separated fully by this experiment. No hardware stall counters were
measured. Timing is all-seven backend return, not UI presentations.

Retained source plus index plus diagnostic scratch: 11,885,234,636 bytes.
Scratch reuse eliminates repeated scratch-buffer allocations. Driver/current
allocation snapshots are in raw responses; these are not process peak memory.

Reproduce with the preceding campaign's `run.py --phased-readers` and the
same executable/folder/cache/output arguments. No app default was changed.
