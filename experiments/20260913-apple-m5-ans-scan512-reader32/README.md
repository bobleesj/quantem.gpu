# Scan512 with the 32-bit reverse bit reader

Status: planned. This isolates the 32-bit reverse-reader specialization while
holding the previously qualified scan512 query path fixed.

## Question

Does the 32-bit reservoir implementation improve exact all-seven large-ADF
updates with scan512 enabled, while preserving full maps and the established
resident/Metal allocation ceilings?

## Fixed protocol

- Apple M5, seven distinct full `(512, 512, 192, 192)` `uint16` acquisitions.
- `scan512` query path in every arm; indexed `packet-owner2`, two streams/lane,
  no batching, concurrency seven, and all other reader/index options off.
- A1 reader32 off, B reader32 on, A2 reader32 off; one warmup plus 20 measured
  cycles per arm, each cycle `adf-center-8` then `adf-center-20`.
- Exact full-map hashes for every source/mask/cycle, stable source identities,
  resident bytes at or below 11,877,814,048, Metal bytes at or below
  11,883,921,408, and explicit release of all seven sources.
- The reader32 pipeline is prepared before loading. Its activation adds no
  resident data by design; sampled Metal allocation is still a hard gate.
- Timing includes planning, submission, GPU wait, and readback. It excludes
  loading and UI presentation; it is not an FPS measurement.

## Rationale

The dominant measured component is dependent entropy decoding. This candidate
uses a two-word, 32-bit reservoir rather than the 64-bit reservoir for the same
reverse bitstream. It does not shorten the tANS state dependency chain, so the
test is a low-risk memory/ALU-path screen, not a presumed breakthrough. If it
does not produce a reproducible gain, prioritize changing the codec's
independent work topology instead of stacking more small reader variants.

## Run

```sh
swift build -c release --disable-sandbox --product metal-paired-runtime-tans-series-benchmark
python3 experiments/20260913-apple-m5-ans-scan512-reader32/run.py \
  --exe .build/release/metal-paired-runtime-tans-series-benchmark \
  --folder ~/data/maped-seven-tilts \
  --cache /tmp/ans-scan512-reader32-cache-20260913-1 \
  --out experiments/20260913-apple-m5-ans-scan512-reader32/results
```
