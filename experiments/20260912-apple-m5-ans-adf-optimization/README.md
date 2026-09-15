# Seven-source compact ANS detector optimization

Target: 120 distinct, complete seven-tile ADF presentations/s on Apple M5
24 GB. Full 512×512×192×192 uint16 acquisitions, exact compact paired tANS,
no crop/bin, no dense resident duplicate. Display statistics must describe
the presented image, including during detector and contrast motion.

Baseline evidence: native run `ans-frame-statistics-20260912` measured
large ADF center 7.1/s (reduction median 158.70 ms), radius 16.1/s
(55.81 ms), selected DP 118.3–120/s. All seven images passed independent
range and histogram parity in `ans-live-visual-20260912`. This does not
certify all possible interactions or 120 Hz histogram updates.

Plan: profile exact changed-pixel reductions, test one structural kernel
change at a time against identical masks, retain failed experiments,
then rerun native presentation without diagnostic overhead. Kernel
throughput alone does not pass the visible frame-rate target.

Status: active; no release or performance qualification yet.

## First sparse-split probe

`sparse-aba-unmatched-mask.jsonl` retains 1,260 full-map comparisons for
A/B/A with sparse split disabled/enabled/disabled. These comparisons passed,
but the following independent runtime-rANS check failed on `bf-base`.
Inspection found different mask policies: the runtime-rANS API automatically
excludes marked invalid pixels, while the paired API consumes its mask as
given (the native app filters it first). The harness must apply the same
per-source validity mask before drawing conclusions. Do not count this run
as independent parity. The failed stderr is retained.

Preliminary speed result: sparse splitting alone did not materially improve
broad translated ADF. No default changed. Corrected mask-parity run follows.

## Dense compaction and plain scratch: rejected for performance

The corrected plain-scratch A/B/C/A run completed 1,680 exact full-map
comparisons and 140 independent runtime-rANS full-map checks (20 masks ×
seven sources). All passed. The independent checker uses a per-source
autorelease pool; an earlier run exhausted its allocation budget after five
sources, and is retained as `compaction-abca.stderr` rather than reported
as a complete pass.

Seven-source median return times in `plain-abca.jsonl`:

| Absolute mask | Control A1 | Sparse split B | Dense/plain C | Control A2 |
|---|---:|---:|---:|---:|
| ADF center +1 | 24.35 ms | 22.42 ms | 28.82 ms | 23.88 ms |
| ADF center +8 | 173.85 ms | 190.00 ms | 217.46 ms | 181.96 ms |
| ADF center +20 | 266.13 ms | 269.91 ms | 307.19 ms | 262.77 ms |

These are backend returns, not visible FPS. Successive masks define each
delta; the center +8 sample is not a fresh eight-pixel displacement from
the original center. Nearly all ADF streams are entropy coded, so removing
sparse-stream overhead does not address the dominant work. None of these
variants is promoted to the default.

## Multi-pair lookahead: exact but slower

`macro-abcda.jsonl` completed 2,100 exact full-map comparisons and 140
independent full-map checks. The four-bit lookahead table adds 4.125 MiB
per resident and can emit up to three pairs per lookup. Despite reducing
dependent lookups on favorable probability models, it was slower on the
actual device: ADF center +1 was 54.28 ms versus 25.61/23.07 ms controls;
center +8 was 411.96 ms versus 189.09/186.81 ms; center +20 was 605.34 ms
versus 263.75/267.41 ms. No macro default is enabled. Register pressure,
table-cache behavior, and branch overhead are hypotheses, not measured
hardware-counter diagnoses.

The targeted GPU probe additionally checks complete signed outputs with
zeros, sparse events, constants, literals, entropy escapes, and partial-byte
tails. Truncated entropy, invalid tail padding, trailing entropy, and
duplicate sparse positions are rejected. An arbitrary altered initial
state is not necessarily malformed and is not used as a rejection oracle.

## Cooperative packet ownership: rejected

`cooperative-aea.jsonl` retains the A/E/A comparison and independent reference
checks. Four SIMD groups share each packet instead of owning four separate
packets. The scratch allocation stays 8 KiB per threadgroup and no resident
count copy is added. The candidate is slower despite the larger dispatch
grid; it remains disabled. The synthetic probe also validates the changed
ownership and malformed-stream path. More dispatched groups alone is not
evidence of better occupancy or faster execution.

## Exact compressed region summaries: measured improvement

The `polar64-aga.jsonl` run passed 1,260 full-map comparisons and 140
independent full-map checks. It adds 848,207,812 bytes of exact packed region
sums across the seven residents, not a dense count duplicate. Building the
index took 282–309 ms per acquisition. Source/decoder rows remain exact
uint16; the index uses UInt32 sums with lossless integer packing.

Matched seven-source API-return medians (not visible FPS):

| Transition label | A1 control | Indexed G | A2 control |
|---|---:|---:|---:|
| ADF center +1 | 26.21 ms | 23.43 ms | 25.51 ms |
| ADF center +8 | 190.69 ms | 67.42 ms | 190.12 ms |
| ADF center +20 | 264.66 ms | 87.50 ms | 265.03 ms |
| ADF radius +8 | 117.92 ms | 36.70 ms | 127.64 ms |

`polar16-radial1-aga.jsonl` uses finer 16-pixel leaves and one-pixel radial
bands. It passed the same full and independent checks. Its index is
2,603,403,760 bytes across seven residents (about 11.91 GB total resident
allocation including experiment tables). The two large-center medians
improved to 43.20 and 74.84 ms, but center +1 remained 20.77 ms. Index
construction increased to 756–881 ms per acquisition. This is an explicit
memory/load-time tradeoff, not a free speedup, and remains opt-in.

`polar16-half-adaptive-aga.jsonl` tests the subsequent opt-in adaptive layout:
16-pixel leaves with half-pixel radial bands where selected by the prototype.
Five A/G/A cycles produced 2,100 maps that matched the frozen A1 maps exactly;
all 140 independent runtime-rANS checks also passed. Stderr is empty. Matched
seven-source API-return medians were:

| Transition label | A1 control | Indexed G | A2 control |
|---|---:|---:|---:|
| ADF center +1 | 24.89 ms | 15.27 ms | 24.86 ms |
| ADF center +8 | 189.38 ms | 53.18 ms | 185.12 ms |
| ADF center +20 | 264.87 ms | 104.39 ms | 264.90 ms |
| ADF radius +1 | 468.19 ms | 111.16 ms | 465.39 ms |
| ADF radius +8 | 123.53 ms | 11.15 ms | 124.49 ms |
| ADF radius +20 | 117.58 ms | 8.43 ms | 122.39 ms |

The exact packed indexes occupy 2,545,100,860 bytes across the residents:
339,678,320; 342,215,832; 364,233,200; 367,841,316; 383,424,028;
366,204,292; and 381,503,872 bytes. Total series resident storage reported by
the benchmark is 11,849,788,780 bytes, while Metal reported 13,809,876,992
currently allocated bytes at the beginning of the experiment. Per-acquisition
index builds were 713.651, 734.284, 803.471, 930.920, 747.331, 894.014, and
765.162 ms (713.651–930.920 ms). These memory and construction costs are part
of the result, and the variant remains opt-in.

The half/adaptive run improves the measured center +1 backend return relative
to both matched controls, but its 15.27 ms median remains above an 8.33 ms
backend budget. Radius +20 is close to that budget rather than conclusively
below it. These are five-cycle medians from one diagnostic benchmark, not a
native presentation measurement or evidence of visible frame rate.

The transition labels retain the sequential-mask caveat above. Neither run
demonstrates 120 visible frames/s. Native presentation remains a separate gate.

## Native mask-policy diagnosis

The first native polar64 run passed geometric masks directly to every paired
resident. Preset products had already excluded each acquisition's marked bad
pixels, but the custom masks had not. The low-level paired API intentionally
preserves its raw mask, and its polar index is used only when both old and new
masks are zero at invalid pixels. The index was therefore resident but bypassed.

Applying each source's marked-pixel exclusions to its custom mask increased the
two-repeat, no-debug native ADF center presentation rate from 6.5 to 14.3 per
input second and resize from 13.4 to 21.7. This is evidence that the index is
now used and materially helps the native path; it remains far below the target.

The identifier-free diagnostic summary is retained in
`native-polar64-mask-policy-summary.json`. For accepted ADF-center generations,
30 of 31 used indexed fields on all seven sources; one used the exact direct
fallback. Resize used indexed fields on all sources for 23 of 45 generations
and the exact direct fallback for 22. The center samples reported medians of
4,997 changed pixels, 72 indexed fields,
and 1,455 residual pixels per source. Median seven-source backend-group wall
time was 59.94 ms, while the full reduction stage was 60.42 ms. Resize medians
were 1,236 changed pixels, 9 fields, 964 residuals, 37.835 ms backend wall, and
38.32 ms full reduction. The backend group therefore accounts for nearly the
whole measured reduction stage; post-update publication is not the next large
bottleneck in this run.

The seven sources use independent queues and overlap. Summing their reported
GPU durations would overstate elapsed GPU time. The per-generation maximum
source duration is only a lower bound: its center median was 17.721 ms and
resize median 12.307 ms, versus backend wall medians of 59.94 and 37.835 ms.
The gap may contain incomplete queue overlap, CPU planning and buffer upload,
command waiting, and seven 512×512 result copies. Absolute GPU start/end
intervals and per-source update wall times were not recorded, so their shares
are not yet identified. Backend wall correlates strongly with residual count
in this sample (0.957 center, 0.987 resize), but correlation is not a hardware
counter diagnosis.

## Wider adaptive partials: measured rejection

`ans-adf-opt-polar16-half-partial32-aga.jsonl` tested extending the exact
adaptive-partials scratch cap from 8 to 32 MiB per source. Three A/G/A cycles
produced 1,260 full-map comparisons, all exact against the matched A1 maps;
all 140 independent runtime-rANS checks also passed. Stderr is empty. The
result file SHA-256 is
`fcf2801088c436665cfadfa502ee7dcd823aa8f6fe0f1877800ab87869f8e52d`;
the empty stderr SHA-256 is
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Matched seven-source API-return medians were:

| Transition label | A1 control | Indexed G, 32 MiB cap | A2 control | Earlier indexed G, 8 MiB cap |
|---|---:|---:|---:|---:|
| ADF center +1 | 23.23 ms | 15.08 ms | 22.57 ms | 15.27 ms |
| ADF center +8 | 188.52 ms | 61.84 ms | 187.25 ms | 53.18 ms |
| ADF center +20 | 269.35 ms | 124.02 ms | 262.96 ms | 104.39 ms |
| ADF radius +20 | 124.78 ms | 9.16 ms | 115.83 ms | 8.43 ms |

The wider cap is rejected and the default remains unchanged. In particular,
the intended larger-residual cases regressed by 8.66 ms for center +8 and
19.63 ms for center +20 relative to the earlier 8 MiB-cap run. That comparison
is directional rather than an isolated kernel A/B: intervening bounds and
malformed-input hardening changed the compiled binary between the two runs.

## Scratchless polar-index construction: exact exploratory pass

`ans-adf-opt-scratchless-polar16-aga.jsonl` is a one-cycle A/G/A exploration
of constructing the exact 16-pixel polar index without retaining the large
unpacked field scratch. All 420 full detector maps matched A1 exactly, and all
140 independent runtime-rANS reference maps passed. Stderr is empty. The exact
packed indexes occupy 2,603,403,760 bytes across seven residents, and reported
series resident storage is 11,908,091,680 bytes.

The scratchless construction removes 1,207,959,548 bytes of construction
scratch. It does not add a dense count volume or change the exact UInt32 sum
contract. The measured seven-source load was 26.616701 seconds, but this single
exploratory run did not control filesystem or operating-system cache state and
is not a cold-load claim or a load-speed comparison.

`ans-scratchless-encoder-parity.log` independently records exact encoder parity
for 67 streams, 389 scans, and 16,309 bytes. `ans-adf-resident-concurrency.log`
records passing same-resident serialization, polar-only update, and release-race
coverage. These focused probes and the one-cycle map pass establish correctness
evidence only; native fine-index validation and performance remain separate.

## Subsequent native index sizing

The retained anonymized native summary also compares mask-fixed configurations.
“All-seven rate” means distinct generations presented by every one of the seven
visible tiles during the trajectory; “per input second” divides that count by
the input interval. It is not a single-tile or accepted-request rate.

| Native configuration | ADF center all seven | Center per input second | ADF resize all seven | Resize per input second |
|---|---:|---:|---:|---:|
| No-index control | 6.7/s | 6.7/s | 14.0/s | 14.1/s |
| Scratchless polar32 | 14.1/s | 14.0/s | 63.6/s | 63.4/s |
| Windowed-pool polar16, visible | 29.9/s | 30.0/s | 49.0/s | 49.0/s |

The valid polar32 run reached comparison readiness in 18.78 s after a 3.58 s
catalog step and reported 10,723,749,864 comparison-resident bytes. The visible
polar16 run reached comparison readiness in 17.04 s after a 3.03 s catalog step
and reported 11,885,154,080 comparison-resident bytes. These are individual
native observations, not controlled cold-load comparisons.

An earlier windowed-pool polar16 run completed loading but its display was
occluded by another application. Its presentation measurements are invalid and
are not included in the table. No screenshots or private source paths from any
native run are retained here.

The windowed autorelease fix permits the 16-pixel index to build with the
default exact encoder by releasing transient construction objects between
windows. Scratchless encoding remains a separately retained opt-in experiment;
it was not required for this seven-resident polar16 configuration. Every valid
ADF configuration above fails the strict 120 complete-seven-tile-generations/s
gate. The current best center and resize results occur in different index
configurations, so neither is a complete solution.

## Final configuration comparison

`ans-adf-opt-final-windowpool-aga.jsonl` completed one A/G/A cycle with exit
status zero: all 420 full maps matched A1 exactly and all 140 independent
runtime-rANS references passed. Its SHA-256 is
`25799a9944a4f5f98ac87a2824c61ffd2050f6af1f674178f49a7a8f95648268`;
stderr is empty.

| Separate native configuration | ADF center all seven | Center/input s | ADF resize all seven | Resize/input s | 120 gate |
|---|---:|---:|---:|---:|---|
| Windowed-pool polar16 | 29.9/s | 30.0/s | 49.0/s | 49.0/s | Failed |
| Windowed-pool radial-half16 | 20.4/s | 20.2/s | 66.3/s | 65.9/s | Failed |

These are separate executions and configurations. The 29.9/s center and 66.3/s
resize results must not be combined as a synthetic best configuration.
`native-final-configurations.json` retains only anonymized aggregates; no raw
native summary, screenshot, dataset identity, or private path is copied.

The 120 complete-seven-tile-generations/s hypothesis is refuted. The exact
polar index remains disabled by default. A separate final native functional run
is recorded below without changing this performance conclusion.

The later shared-plan-cache native probe reached 27.0/s center and 48.7/s
resize, with one incomplete center generation, and failed the strict gate. It
did not improve on the matched polar16 run. Its CPU concurrency probe passed 10
fixtures and 160 calls, which supports correctness of the cache key and locking
only; no performance gain is claimed.

The separate anonymized final functional journey is retained in
`native-final-functional-summary.json`. It exited zero with no reported
problems, loaded all seven residents, traversed all sources forward and backward,
settled rapid navigation to the requested source, and passed BF/ABF/ADF/iDPC,
FFT, scan/custom-detector motion, contrast, and seven-tile comparison journeys.
All 15 navigation receipts were received and applied once with no retries. This
is functional evidence, not an additional timing or 120 Hz claim.

## Historical 66-acquisition apple-m5-max-128gb reference

The detailed bounded port note is [reference-atlas-port.md](reference-atlas-port.md).

A clean apple-m5-max-128gb application reference at revision
`48503d7fe3aa63bc42691941dc38f600df094e43` pins the clean quantem.gpu backend
revision `fbd1c87264668b4feb64bf975de9a19d2bd784a8`. Its exact index is a different
memory/performance point: it stores an atlas of 489 complete ADF output fields
for 66 acquisitions on a 2-pixel center lattice within 25 pixels, then chooses
the nearest stored mask/base and applies an exact signed residual. The 489
fields are packed two-dimensional outputs, not 489 duplicate 4D count volumes.
The implementation is in `MetalTANSResidentSeries.swift` (index construction
and query selection), `TANSExactTileIndex.swift`, and the exact-index kernels in
`tans.metal`; the application atlas lifecycle is in
`EntropySeriesController.swift`.

One retained publication run reported 106.36 published/s (105.28 steady) from
267 publications, but only 259 presentations, a 16.667 ms p95 presentation
interval, and a failed 120-publication/s gate. Repeats measured 50.45 and 52.22
steady published/s. Thus the isolated roughly 100 published/s observation is
not evidence of reproducible 100 presented FPS. That system used a 128 GB M5
Max and roughly 92.93 GB of resident state, whereas this experiment targets a
24 GB system and seven sources; no throughput or memory claim is extrapolated
between the two architectures.

## First exact reference-base port

`ans-adf-opt-choosebase-aga.jsonl` records the first local port of choosing
between the zero and previous exact bases. Its one A/G/A cycle completed with
all 420 full maps exact and all 140 independent runtime-rANS references passing;
stderr is empty. The focused CPU choose-base probe passed 45 differing cases
plus an unchanged-zero case. The shared-plan-cache probe separately passed 10
fixtures and 160 concurrent calls, and resident concurrency coverage passed.
The final choose-base concurrency log specifically covers same-resident
serialization, a polar-only update, and a release race with update-before-release.

The corresponding opt-in native polar16 observation reached 30.6 complete
seven-tile ADF center presentations/s and 50.7/s for resize (30.5 and 50.4 per
input second), with one incomplete generation in each trajectory. It failed
the strict 120/s gate. This is one observation and does not establish an
improvement over the matched polar16 result; reference-base selection remains
off by default. `native-choosebase-summary.json` retains only anonymized
aggregates. No raw native screenshot or private source path is retained.
