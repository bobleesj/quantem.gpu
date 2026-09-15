# Conservative exact history-base selection

This follows `20260912-apple-m5-ans-history-base`. The indexed candidate must
reduce residual pixels or indexed fields without increasing either. This
prevents choosing many more indexed fields to save only a few decoded pixels.
Raw history-base selection is unchanged. All options remain experimental and
disabled by default; the installed application was not modified.

## Matched results

Seven full uint16 512×512×192×192 acquisitions, exact paired ANS residents.
All-seven synchronized backend timings, excluding parity and UI presentation.
Every cycle contains five A→B→C sequences, with A=`adf-base`,
B=`adf-center-8`, C=`adf-center-1`. C is a new mask relative to both saved
images. Cycle zero is excluded; each reported C median has ten warm all-seven
updates (70 source images). There are **zero exact-history hits** among these
C samples. The benchmark resets to zero only between cycles, not between the
five sequences; do not use the first A as proof of continuous-drag speed.

| Fresh B→C computation | Off before (ms) | On (ms) | Off after (ms) |
| --- | ---: | ---: | ---: |
| Raw ANS history-base selection | 195.099 | 25.450 | 184.782 |
| Regional-index history-base selection | 44.828 | 21.337 | 44.964 |

Raw candidate p95 was 26.929 ms; indexed candidate p95 was 22.854 ms.
The raw path needs only 568 history-relative detector pixels rather than
4465 current-relative pixels for the first source, explaining the reduced
work without changing the target image. This is specific to reversals, not a
7× acceleration of every fresh update.

With history-base selection enabled, adding FC13 measured on/off/on medians
23.589 / 25.172 / 23.966 ms for the same raw C transition. No new resident
allocation was added by either of these two changes.

## Full-trajectory check

The 20-mask BF/ABF/ADF center/radius sequence was also tested off/on/off with
five cycles per arm. For the formerly bad ADF center+20 transition, the
candidate now keeps the same current-base plan (371 fields, 1067 residual
pixels for source zero). Times were 74.727 / 75.644 / 72.863 ms. The bad GPU
tradeoff is removed, but extra planning overhead is not proven eliminated.
The large ADF radius transition improved 86.335 / 46.435 / 88.202 ms.
Small monotonic ADF center+1 remained 20.631 / 21.257 / 20.967 ms.
Thus neither sustained 120 FPS nor a universal no-regression claim passes.

## Parity, memory and remaining gates

All 5,075 full-map hashes match 140 frozen reference maps exactly; repeated
masks are also compared as complete arrays. No clipping, binning, cropping,
decoded 4D duplicate or precomputed image atlas was introduced.

Resident bytes remain 11,885,154,080 and current Metal allocation remains
11,891,261,440. These totals include the existing 2,603,403,760-byte regional
index, unused in raw mode, and the earlier 7 MiB history allocation. The
nearer-base policy adds no third buffer. System-wide swap was 5557.19 MiB at
the post-run check; stable resident bytes do not establish zero memory
pressure or a process peak measurement.

Failure rollback was reviewed, but GPU fault injection and native UI tests
are still pending. Multi-resident publication is not transactional on failure.
The tested executable and source hashes are in `manifest.json`.

## Next measured question

Before adding more metadata caching, instrument the single-entry plan cache's
hit/miss counts and CPU time. Current/history planning can alternate entries
while its lock serializes construction. A larger cache is not automatically
useful because source validity masks differ. Measure that cost first; keep
the low-level raw decoder improvement separate from reuse-based gains.
