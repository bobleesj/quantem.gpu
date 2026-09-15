# Limit simultaneous seven-source detector updates

Status: planned. The benchmark now measures bounded in-flight source updates
on the ordinary, unbatched path; earlier `bounded_concurrency` values were
ignored when `batch=false`.

## Hypothesis

With all seven exact uint16 ANS residents already loaded, limiting independent
GPU submissions to four at a time may lower the all-seven large-ADF wall time
by reducing GPU scheduling or working-set contention, without increasing the
resident-memory ceiling. This is only a hypothesis: the existing batch API
was slower, but it is a different submission path and does not answer whether
bounded concurrency within the ordinary per-source path helps.

## Frozen workload and gates

- Apple M5, 24 GB unified memory; seven distinct full
  `(512, 512, 192, 192)` uint16 acquisitions from `~/data/maped-seven-tilts`.
- Exact `adf-center-8` then `adf-center-20`; 20 measured cycles per arm after
  one warmup. `scan512` polar query and trusted decode table stay enabled for
  all arms. Only in-flight update concurrency changes: 7 / 4 / 7 (A/B/A).
- Require full-map hashes to match the A1 baseline every cycle, exact resident
  source identities, resident bytes no greater than 11,877,814,048 B, sampled
  Metal allocation no greater than 11,883,921,408 B, and complete release.
- Report all-seven backend wall p50/p95 and the per-source observations
  separately. These measurements exclude loading and UI rendering; they are
  not presentation FPS or cold-I/O timing.

## Run

```sh
swift build -c release --disable-sandbox \
  --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-update-concurrency/run.py \
  --exe .build/arm64-apple-macosx/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-update-concurrency-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-update-concurrency/results
```

The executable, shader, source identities, arm configuration, exact map
hashes, resident allocation and teardown are retained in the manifest and raw
records. One GPU workload at a time.

## Decision rule

Keep concurrency four only if it beats both bracket controls beyond their
spread, has no p95/per-source critical regression, preserves exact maps, and
does not exceed the existing memory ceiling. Otherwise mark the hypothesis
refuted and leave concurrency seven as the measured best path.
