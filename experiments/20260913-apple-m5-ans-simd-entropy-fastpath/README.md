# SIMD32 entropy decoder fast-path A/B/A

Status: completed; the SIMD fast-path speed hypothesis was refuted. The initial
harness-failure artifacts remain preserved under `results/`.

## Question

Does the opt-in SIMD32 entropy-only branch reduce the unprofiled all-seven
indexed ADF center-8 → center-20 update wall time while preserving every full
detector map and the seven-source resident contract?

## Fixed protocol

- One benchmark process loads seven distinct, original full `(512, 512, 192,
  192)` `uint16` sources. No crop, clipping, binning, or count modification.
- `QGPU_PAIRED_RUNTIME_PREPARE_SIMD_ENTROPY_FAST_PATH=1` is set before the
  process starts, and the ready response must confirm that the candidate
  pipeline was prepared. A/B/A toggles only the runtime
  `simd_entropy_fast_path` command field.
- Inherited `QGPU_ANS_OPT_*` settings are cleared. The ready response and each
  arm configuration must report `polar_query_variant=packet-groups`; the
  scan512 A/B/A flag must be false and all seven scan512 pipelines unprepared.
- Arms run in order `A1=false`, `B=true`, `A2=false`; each request uses indexed
  `packet-owner2`, ordinary unbatched submission, bounded concurrency 7, and
  one warmup plus 20 measured cycles for both `adf-center-8` and
  `adf-center-20`.
- Every arm response must report full-map parity. The runner checks every
  cycle/mask/source hash across all three arms, stable resident bytes, seven
  distinct source identities, exact arm configuration, and final release of
  all residents. The same immutable resident objects stay alive in the single
  process from ready through A2.
- Timing is unprofiled. The manifest records all cycle samples, including the
  excluded warmup, plus measured-cycle all-seven wall p50 and nearest-rank p95
  overall and by mask. Device allocation is recorded for ready and every arm.

## Run

Build the benchmark using the project's normal release workflow, then run:

```sh
python3 experiments/20260913-apple-m5-ans-simd-entropy-fastpath/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-simd-entropy-fastpath-cache \
  --out experiments/20260913-apple-m5-ans-simd-entropy-fastpath/results-retry
```

The cache is metadata/index preparation only. Source load is outside the timing
boundary. The runner fingerprints the executable, benchmark, resident source,
kernel API, shader, and its own source before launch. It preserves raw JSONL,
exact arm samples, allocation and latency summaries, release evidence, and any
failure status in the manifest and result artifacts. A failed attempt is kept
with `status: failed` and structured failure details.

## Result

The corrected `results-retry/` run completed A1/B/A2. Exact full-map parity and
per-cycle hashes passed in every arm. Persistent residency stayed at
11,877,814,048 bytes and Metal allocation stayed at 11,883,921,408 bytes; all
seven residents were released.

For the `adf-center-8` → `adf-center-20` update, all-seven unprofiled timings
were:

| Arm | p50 | p95 |
|---|---:|---:|
| A1 | 70.16 ms | 72.36 ms |
| B, SIMD entropy path | 70.94 ms | 76.65 ms |
| A2 | 70.19 ms | 75.53 ms |

The controls agree closely and B is slower, so the speed hypothesis is
refuted; the candidate remains default-off.

The original `results/` attempt completed A1 and B with exact parity and matching
map hashes, then the Python harness stopped on a comparison between a
mask-to-source hash map and a per-cycle hash map. That harness-only failure is
retained in the manifest and `results/failure.json`; it did not indicate a
kernel mismatch. The fix tracks the two hash references separately, and the
completed retry wrote to the separate `results-retry/` directory.
