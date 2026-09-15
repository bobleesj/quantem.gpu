# Polar query scan512 A/B/A

Status: scan512 speedup reproduced in a separate warmup-controlled A/B/A run;
kernel remains opt-in pending broader transition and application-path checks.

## Question

Does the prepared scan512 polar-query path change the unprofiled all-seven
indexed `adf-center-8` → `adf-center-20` update latency while preserving exact
detector maps and the seven-source resident contract?

## Fixed protocol

- One process loads seven distinct original full `(512, 512, 192, 192)`
  `uint16` sources. No crop, clipping, binning, or count modification.
- The process starts with `QGPU_ANS_RESIDENT_LOOP=1` and
  `QGPU_ANS_OPT_POLAR_QUERY_SCAN512=1`; all seven scan512 pipelines must be
  prepared before any timed command. The ready response must report the
  experiment flag enabled and the current query variant as `packet-groups`.
  The runner forces all inherited `QGPU_ANS_OPT_*` flags off except the
  scan512 experiment flag, and explicitly enables the polar index.
- Unprofiled resident-loop requests run A1 `packet-groups`, B `scan512`, then A2
  `packet-groups`. Each arm uses indexed `packet-owner2`, unbatched submission,
  bounded concurrency 7, and 20 measured cycles for `adf-center-8` followed by
  `adf-center-20`. The separate warmup-controlled repeat uses one unmeasured
  cycle per arm before its 20 measured cycles.
- Every response must pass full-map parity. The runner checks each
  cycle/mask/source hash across arms, seven distinct ready-time source IDs,
  unchanged resident bytes and Metal allocation, and explicit release of all
  seven residents. The same resident objects remain alive in this one process
  throughout A1/B/A2; their ready-time IDs are attached to each arm record.
- Center-20 timings are collected after center-8 within every cycle. The
  summary reports measured all-seven wall p50 and nearest-rank p95 by mask and
  arm, with particular attention to `adf-center-20`.

## Run

Build the benchmark using the project's normal release workflow, then run:

```sh
python3 experiments/20260913-apple-m5-ans-polar-scan512/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-polar-scan512-cache \
  --out experiments/20260913-apple-m5-ans-polar-scan512/results
```

The cache is metadata/index preparation only; source loading is outside the
timing boundary. The runner stores every stdout JSON line, stderr, the ready
record, exact arm samples and summaries. Failed attempts retain their partial
records, structured failure, and release evidence in the manifest.

The initial screening run used these measured JSON-line commands, in order:

```json
{"op":"run","arm":"A1","mode":"indexed","kernel":"packet-owner2","batch":false,"bounded_concurrency":7,"profile":false,"cycles":20,"mask_names":["adf-center-8","adf-center-20"],"polar_query_variant":"packet-groups"}
{"op":"run","arm":"candidate","mode":"indexed","kernel":"packet-owner2","batch":false,"bounded_concurrency":7,"profile":false,"cycles":20,"mask_names":["adf-center-8","adf-center-20"],"polar_query_variant":"scan512"}
{"op":"run","arm":"A2","mode":"indexed","kernel":"packet-owner2","batch":false,"bounded_concurrency":7,"profile":false,"cycles":20,"mask_names":["adf-center-8","adf-center-20"],"polar_query_variant":"packet-groups"}
```

## Initial screening result

The first full A/B/A run had no per-arm warmup. For `adf-center-20`, p50/p95
were 70.22/73.59 ms for A1, 61.02/62.72 ms for scan512, and 71.00/76.03 ms
for A2: a 13.6% p50 reduction versus the mean of the controls. Exact full-map
hashes matched for all seven sources and both masks. Resident bytes stayed at
11,877,814,048 and Metal allocation stayed at 11,883,921,408 bytes; all seven
residents were released. The first A1 `adf-center-8` sample was 433.50 ms and
is preserved as a first-use outlier, not blended into the center-20 result.

The separate warmup-controlled replication at
`../20260913-apple-m5-ans-polar-scan512-warmup-retry/` measured A1 70.22/73.01,
scan512 60.58/66.62, and A2 71.71/75.52 ms. This reproduces a 14.6% p50
reduction versus the bracketed controls. Both runs preserved exact full-map
hashes for all seven sources and both masks, unchanged resident bytes
(11,877,814,048) and Metal allocation (11,883,921,408 bytes), and successful
release of all seven residents. The initial screen's 433.50 ms first-use
`adf-center-8` sample remains preserved as an outlier; the controlled
replication used one warmup per arm.

This is a repeatable but bounded win: scan512 changes the index/query stage,
not the dependent residual tANS decoding, and the large ADF update remains
about 60.6 ms p50, above the 20–30 ms target. The variant remains opt-in while
its effect is checked against other exact masks and scheduling choices. Neither
run measures UI frame rate.
