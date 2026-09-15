# Retained-resident decoder campaign

No production speedup qualified in this first batch. The harness now retains
one exact query plan and its scratch buffers per source, avoiding repeated
allocation. This is a diagnostic improvement, not a new application default.

Seven distinct full uint16 sources, each 512x512x192x192, same exact signed large
ADF center transition and unchanged exclusions/index. All counts retained.
One warmup per arm and 20 measured repetitions per arm, shuffled within rounds
with seed 1709. All 735 source transitions passed full-array stage parity:
index+residual=combined=ordinary indexed target-minus-previous modulo UInt32.
This checks against the existing indexed implementation, not a fresh independent
full-volume count oracle. Small signed high-count/malformed fixtures also pass.

| Arm | Residual mean / median (ms) | Combined mean / median (ms) | Combined p95 (ms) |
|---|---:|---:|---:|
| Allocate each call |44.25 /45.39|65.99 /66.25|72.30|
| Reuse scratch, original decoder |42.72 /43.04|64.67 /64.83|70.99|
| Reuse + branchless pop |44.09 /44.93|66.85 /66.43|74.44|
| Reuse + refill16 |46.06 /46.46|66.04 /68.19|73.19|
| Reuse + refill24 |44.86 /46.23|65.74 /67.36|71.06|

Timers cover all-seven backend return including preparation, submission,
synchronization and output copies. Not displayed FPS, not loading time.
Per-source GPU timestamps, all raw samples and means/medians/tails are retained.
Device occupancy, register counts, cache misses and physical traffic were not
measured. Interleaved index controls vary too; do not attribute all differences
to decoder code. The allocation/reuse combined paired ratio is 0.9995, so no
reliable combined gain can be assigned to scratch reuse from these samples.

Scratch totals 7,420,588 bytes across seven sources. Reused trials allocate zero
new scratch Metal buffers; they still create command/encoder objects and host
readback arrays. Resident accounting is 11,885,234,636 bytes including scratch,
versus 11,877,814,048 before diagnostic scratch. It is not process peak memory.
Allocation-control trials retain the cached scratch and allocate another
temporary query set: they are an allocation-cost probe, not a peak-memory
equivalence test. There is no dense 4D duplicate or new lookup table.

See [HYPOTHESES.md](HYPOTHESES.md) for 20 distinct approaches and their updated
statuses. Entries do not mean 20 new implementations were tested. Phased table
reads, refill thresholds and pair unrolling have already been measured; no app,
installed release, push or merge changed.

Run with `python3 run.py --exe <benchmark> --folder <seven-source-folder>
--cache <existing-encoded-cache> --out <fresh-output-directory>`.
Then `python3 summarize.py <output-directory>/trials.jsonl`.
