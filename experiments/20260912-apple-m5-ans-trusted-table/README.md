# One-time transition validation

Hypothesis: remove a state-range comparison from each decoded pair by proving
the transition bound once for the actual immutable table. Function constant 13
is opt-in and disabled by default.

The private GPU table is read back once per resident and compared exactly with
the deterministic internal table. Every entry also satisfies `bits <= 10` and
`base + 2^bits <= 1024`. The private resident initializer accepts only the
internal builder product; no caller is given a mutable table handle. Payload
length, escape, initial state, terminal state and bit-exhaustion checks remain.

This adds no persistent GPU allocation. Validation uses a transient 128 KiB
staging buffer plus a host copy, and a shared 128 KiB expected table. This work
is initialization overhead, not included in detector-update timings.

## Measurement contract

Seven full native uint16 acquisitions, each 512×512×192×192, remain resident.
No history hits, clipping, binning or cropping. Raw and regional-index-assisted
paths are reported separately. The latter uses the existing 2,603,403,760-byte
regional index; the raw path leaves it unused. Total resident bytes remain
11,877,814,048 in both arms because both pipelines are prepared in one process.

Twelve arms test off/on/off for raw and indexed paths under both coordinated
batch submission and independent concurrent submission. The first six arms
used the harness's default `batch=true`; the second six explicitly use
`batch=false`. Comparisons must stay within matched triplets. Full maps are
checked against frozen references outside the timing interval. No UI or
presented-FPS claim follows from these backend measurements.

The first host build found an unsupported `Zip2Sequence.firstIndex` call in an
error-reporting branch; it was corrected to an array-index search before the
successful release build and GPU tests. Do not count parse-only checks as a
successful build.

## Results

15 arms and 8,820 complete detector-map comparisons passed against 140 frozen
reference maps. Three additional raw concurrent arms repeated the promising
result with five cycles each. Warm medians exclude each first cycle.

| Concurrent update | Off before (ms) | On (ms) | Off after (ms) |
| --- | ---: | ---: | ---: |
| Raw ADF center +1, first | 25.292 | 22.466 | 25.059 |
| Raw ADF center +1, repeat | 25.297 | 22.453 | 24.482 |
| Raw ADF center +8, repeat | 189.748 | 177.984 | 190.585 |
| Raw ADF base, repeat | 472.116 | 449.382 | 468.813 |
| Indexed ADF center +1 | 21.604 | 21.096 | 21.083 |
| Indexed ADF center +8 | 46.074 | 45.932 | 45.191 |

The raw small move improved approximately 8-11% on repetition; the indexed
small move did not beat both controls. This is a narrow kernel gain, not a
universal speedup or 120 FPS. Keep the specialization opt-in until default
integration and native UI gates pass. Resident allocation stayed unchanged.

The GPU adversarial probe also passed high uint16 signed cancellation, mixed
dense/sparse paths, truncation, nonzero tail padding and trailing bytes. No
hardware occupancy or cache-counter evidence was collected.
