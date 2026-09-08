# Packed-resident optimization on Apple M5

This is an unpublished candidate investigation, not a release signoff. The
seven-acquisition resident allocation remains **14,752,175,312 bytes**. Selected
diffraction reaches the display's 120 Hz cadence, but the measured wide-detector
trajectories do not all sustain 120 complete seven-image updates per second.
The four-to-five-second fully usable seven-acquisition loading target is also
not met.

## Scientific and measurement boundaries

Each of the seven distinct acquisitions contains a complete
`(512, 512, 192, 192)` array in `(scan row, scan column, detector row, detector
column)` order. Sources are bitshuffle/LZ4 HDF5 with uint16 counts, including
65535. No count is clipped or excluded; neither scan nor detector is binned or
cropped. The resident is lossless blockwise bit-plane packing, not an ANS
representation or a saved two-dimensional image.

Loading reads original compressed counts and reconstructs the complete resident.
Saved layout metadata and exact small DPC sums can avoid repeated preparation;
they do not substitute for loading the four-dimensional data. OS page state is
uncontrolled, so none of these measurements proves cold disk performance.
Fresh layout preparation and metadata-assisted reopening are separate cases.

The full experimental lineage, rejected hypotheses, source patches, executable
and Metal-resource hashes are in the
[experiment registry](https://github.com/bobleesj/quantem.gpu/blob/main/experiments/RUNS.md).
The retained candidate records are available in the local `experiments/` tree;
the remote registry is not updated until publication is authorized.
The backend base is `a0aaa52be1fb7b66406f8624527383178509e1f7`; each run names its
uncommitted patch. Native integration evidence uses application base
`689c4b71b7651e1ff175d19cfae2c6c8f472091a` and separately identified app binaries.
An edited source tree is never evidence that an older binary exercised it.

## Qualified loading changes

The decoder recognizes a bounded terminal zero run only after validating the
LZ4 match, complete decoded length, final literal token, and all remaining
literal bytes. It writes through the next 16-byte boundary and records the
proven-zero suffix in an existing scratch allocation. The packing kernel uses
that proof when consuming four neighboring source words per lane. Other blocks
still take the complete decoder path. This removes redundant scratch writes,
not scientific data or validation.

The selected configuration combines 32-thread decode and packing dispatches,
four-column bit-plane transposition, exact direct DPC preparation, and the
terminal-zero path when saved DPC sums are available. First-time preparation
retains the complete bounded scratch decode. No additional resident count
buffer is introduced.

The following controlled backend experiment reconstructed all seven sources in
four cycles per arm and checked 5,040 complete detector-map hashes across the
three arms. The statistic excludes the first cycle, releases residents between
loads, and measures indexed-source open through complete resident return.

| Configuration | Statistic | Time (s) | Device tested | Date tested |
|---|---|---:|---|---|
| Terminal-zero disabled, first control | Reopen median | 1.14724 | Apple M5, 24 GB | 2026-09-08 |
| Terminal-zero enabled | Reopen median | 1.03488 | Apple M5, 24 GB | 2026-09-08 |
| Terminal-zero disabled, second control | Reopen median | 1.14219 | Apple M5, 24 GB | 2026-09-08 |

This is a roughly ten-percent improvement in that controlled reopen comparison,
not a claim that every file loads in less than one second. A separate ordinary
build audit started with an empty packing-plan directory: its seven first loads
took 2.24–2.42 seconds; the seven metadata-assisted reconstructions took
0.987–1.025 seconds. These are descriptive observations, not a controlled
comparison: source-page state was uncontrolled and light CPU documentation work
ran during the excluded full-count verification intervals.

## Exactness and recovery

The completed terminal-zero full-volume audit authenticated
**135,291,469,824 counts**: every count in all seven acquisitions, twice. It also
checked exact UInt64 DPC sums, detector sums, and Float32 mean diffraction. Its
1,075.284 seconds of verification work is excluded from loading time. The
in-process detector-oracle flag remains false because that invocation received
the full-count oracle; 210 full detector hashes were checked separately against
the frozen independent reference. Neither raw flag nor reference was rewritten.

A second full-volume audit of the ordinary build passed the same count and image
checks on fresh-plan creation and reopening. It authenticated another
135,291,469,824 counts, excluding 1,077.043 seconds of verification from load
timings. All fourteen loads reported zero fallback; the largest post-release
device allocation was 4,915,200 bytes.

The current-default regression suite passes **38 tests**. Coverage includes
uint8 and genuine high-count uint16 sources, saved-plan reconstruction, malformed
streams, cancellation, tight budgets, source changes, and preservation of
unrelated files. The permanent zero-tail fixture checks 2,702 exact cases and
458 malformed cases with poisoned scratch storage.

Failed qualification runs remain in the registry. Initial tests assumed
an isolated decode stage and miscounted the newly fused initial DPC preparation.
A later test expected the old staging fallback at a budget that the optimized
path now fits. The final test preserves that legacy fallback and separately
requires the optimized path to fit the identical budget. Scientific hashes,
count references, and memory limits were not relaxed.

## Native integration evidence

These measurements use the ordinary, untuned candidate binary identified in
`20260908-apple-m5-native-ordinary-controls`, not a packaged release. All seven
tiles were visible, virtual-image scaling was linear, and selected-DP scaling
was logarithmic. FFT began hidden.

| Operation | Measurement | Value | Device tested | Date tested |
|---|---|---:|---|---|
| First acquisition | Load request to actual first presentation (s) | 1.24959 | Apple M5, 24 GB, 120 Hz | 2026-09-08 |
| Additional six acquisitions | Compare request to resident-ready state (s) | 6.02 | Apple M5, 24 GB, 120 Hz | 2026-09-08 |
| Large ABF center drag | Complete seven-tile presentations/s | 112.6 | Apple M5, 24 GB, 120 Hz | 2026-09-08 |
| Large ABF resize | Complete seven-tile presentations/s | 115.0 | Apple M5, 24 GB, 120 Hz | 2026-09-08 |
| Large ADF center drag | Complete seven-tile presentations/s | 101.1 | Apple M5, 24 GB, 120 Hz | 2026-09-08 |
| Large ADF resize | Complete seven-tile presentations/s | 112.0 | Apple M5, 24 GB, 120 Hz | 2026-09-08 |
| Selected diffraction drag | Steady distinct presentations/s | 120.0 | Apple M5, 24 GB, 120 Hz | 2026-09-08 |

Resident-ready timing does not include every possible first-use interaction
cost. Missing first-submission timing is recorded as missing, never zero.
Selected-DP throughput including initial input delay is lower than its steady
rate. Full-scan average diffraction was presented and verified as a static
mean; after returning to selected diffraction the measured rate was 118.3/s,
with a maximum 25 ms gap. The 120 Hz everywhere gate therefore remains unmet.
This also does not prove 120 newly computed full-scan averages per second.

The folder journey visited all seven acquisitions, issued 36 rapid selections,
checked latest-selection-wins behavior, forced an actual reload, and replaced
folders in both directions. A failed open retained the prior DP, virtual image,
and FFT with an actionable failure message instead of a stale loading banner.
Colormap and contrast changes preserved the underlying image hashes, and light
and dark appearance were checked in actual screenshots. Forced reload produced
a new resident generation and actual presentation after 1.14842 seconds.
During progressive loading, 356 distinct selected-DP presentations occurred
before all seven residents were ready, with a maximum observed gap of 16.67 ms.
That is evidence of interaction during loading, not a 120 Hz guarantee then.

Earlier native runs synchronously polled state during gestures and omitted a
required diagnostic build define. Their measured times remain historical, but
their originally inferred kernel settings and FPS comparisons are invalid.
Current trajectory observation reads timing logs without forcing state flushes
during a gesture and checks the kernel settings actually reported by the run.

## Rejected hypotheses and remaining work

### Later release-preflight repeat

`20260908-apple-m5-release-native-recheck` records a freshly rebuilt ordinary
candidate (executable SHA-256
`0e105919e7c90ac222292daed5b3c73b778b5ff60947eb92c71a98fd0d0f73e8`).
Seven-source controls, 36 rapid selections, failed-input retention and folder
replacement passed. All seven resident allocations remained 14,752,175,312 bytes.
Large ABF center/resize measured 113.3/116.7 complete seven-tile presentations/s;
large ADF center/resize measured 103.9/114.7. Selected DP reached 120 steady
presentations/s, but seven-image ADF center had a maximum 58.33 ms interval.
These are repeat observations, not a controlled speedup over the earlier run.

The separate six-file journey failed once: requesting dataset 3 left dataset 2
active until a 120-second timeout. An unchanged repeat passed every file,
returns, products, FFT and drags. Notification delivery versus application
navigation remains unresolved; the failed record is retained and blocks release.
The successful repeat also lacked its first-presentation timing at the initial
sample. Neither a passing repeat nor missing timing is rewritten as full acceptance.
The app changelog has an Unreleased entry; neither it nor these new records has
been published. This investigation has not replaced the installed v0.0.8 ZIP.
The current-source numerical repeat is retained as
`20260908-apple-m5-release-parity-recheck`: 38 tests passed in 289.13 seconds,
with no skips, reference changes or budget relaxation.

### Kernel hypotheses

Larger vector groups, sorted LZ4 blocks, whole-block cooperative expansion,
zero-initialized scratch, lazy plane writes, fused block decode/packing, and
compiled-library reuse did not consistently improve their controlled
comparisons. Their evidence is retained; losing prototypes are not promoted.

Detector compiler threadgroup limits also failed to beat both controls despite
1,680 exact complete images and unchanged memory. The initial decoder compiler
constraint gain did not replicate in the decode/packing combination repeat.
Linear width grouping had mixed trajectory results, and wide-only partitioning
did not beat both controls. Their terminal records have been reconciled from
retained raw artifacts; none is promoted.
Only timestamp counters are exposed on this device, so no hardware occupancy
or bandwidth-counter measurement is claimed. Kernel GPU milliseconds are not
presented FPS, and a 120 Hz screen cannot show more than 120 distinct frames per
second; faster computation would provide headroom.

Before release: repeat numerical and native gates after any further kernel
change, pin the reviewed backend revision, and preserve the current installed
app until its replacement passes the release workflow. The installed application
and colleague ZIP have not been replaced by this investigation.

## Local source freeze

The implementation and parity fixtures are committed as `8d631dc`. A fresh
release-mode run of `tests/hardware/metal/test_original_packing.py` passed all
38 tests in 270.14 seconds on the same Apple M5, with no skips. This is test
suite duration, not load latency. No scientific reference or tolerance changed.
The only subsequent fixture edit removed trailing blank lines.

Three retained pytest logs contained absolute Python and checkout paths. Their
repository copies replace only those two paths with role placeholders. Each
manifest records the original log hash, redaction scope, and new artifact hash;
the original logs remain private. Failed outcomes remain failed. These edits
do not recapture or alter measured scientific results.

The app subsequently passed its fixed public-control journey and seven-image
hook journey after packed-only cleanup; its own validation note records exact
binary identity and the remaining frame-rate limitations. That does not turn
this backend branch into a published release. Publish the reviewed backend
only with authorization, then pin and resolve that exact SHA in the app and
repeat the release qualification. Do not use an editable dependency or relabel
the historical qualification revision as the current build revision.
