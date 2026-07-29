# SSB performance notes

This page records the native CUDA, MPS, and WebGPU SSB performance contracts
so future work starts from measured behavior, not from memory.

## Canonical anonymized Reference-512 MPS regression contract, 2026-07-27

The machine-readable source of truth is
`tests/fixtures/ssb_reference_512_mps.json`. Use the production ShowPtycho CLI
for the scientific run; do not substitute a temporary Python harness.
The specimen, source filename, filesystem path, and content hashes are kept in
an operator-local private mapping outside Git. The public identifier is
`reference-512-full-bf-v1`.

All MPS timings in this contract were measured on a reference MacBook Pro
(`Mac17,2`) with an Apple M5, 10-core integrated GPU, and 24 GB unified memory.
Results from other Apple Silicon systems require separate benchmark records.

The source is native `512x512x192x192` uint16. Automatic full bright field
selects `8,937` logical
pixels, of which `2,464` have nonzero probe aperture. The exact uint8 BF-column
companion is 2,342,780,928 bytes and has maximum count `53`.
The narrowed storage is integer-exact; optimization remains float32/complex64.

The fixed optimizer contract is seed `42`, start `C10=0 nm`, `C12=50 nm`,
`phi12=0 rad`, 200 Optuna trials, exact full-BF phase-variance objective,
Nelder-Mead refinement, and final independent reconstruction. It returns:

- `C10 = 73.18188621458395 nm`
- `C12 = 14.020962948808993 nm`
- `phi12 = 0.4700365259977606 rad`
- `loss = 0.04469207674264908`

The automatic geometry is center `(row, col) =
(94.88451385498047, 96.35952758789062)` and radius
`53.35992814757164 px`. Any change to the BF indices, these scientific values,
the objective, or the precision contract is a correctness regression.

The pre-optimization prepared-source baseline had a `46.75 s` median
wall-to-HTML and `37.9370 s` median fit. The retained packed-column topology
reached a quiet best of `23.058 s` fit / `24.16 s` complete CLI wall. A later
clean current-tree signoff measured `24.065 s` fit /
`25.26 s` process wall, broken down as `0.905 s` preparation, `0.185 s`
initial loss, `19.547 s` Optuna, `3.226 s` refinement, `0.051 s` final object,
and `0.144 s` final phase/loss. All 231 records matched the canonical JSON hash
`cb2b35cacaabd01ac473df91875383dc5f633dbb53bf5ebb27627c80fd0168c1`.

Five complete retained-tree thermal repeats measured
`24.289/24.528/24.406/24.648/25.221 s` fit (median `24.528 s`). Every repeat
returned the same 231-record hash, 30 physical refinement calls, aberrations,
and final loss. Treat wall time as a distribution: on this Mac, a prepared
process wall near `24-26 s` is current and sustained fit above `27 s` warrants
checking browser GPU residency, swap, and thermal pressure. The still-unmet
long-term target is below `20 s` without changing BF evidence, precision,
trial count, refinement, or objective.

The final community-handoff run on 2026-07-28 measured `24.4223 s` fit /
`25.56 s` process wall, with `0.9026 s` preparation, `0.2026 s` initial loss,
`19.8293 s` Optuna, `3.2798 s` refinement, `0.0511 s` final object, and
`0.1477 s` final phase/loss. Peak process footprint was `10,823,322,048` bytes
and the process recorded no swap. Its complete trial list was semantically
identical to the canonical reference and reproduced the `cb2b35ca...` indented
JSON hash.

The exported full-field WebGPU UI prepared the 8,937-pixel reducer in
`1.2911 s` on the hardware `apple metal-3` adapter. Initial phase-only reconstruction
was `62 ms` GPU / `63 ms` UI. Sixteen committed C10/C12/phi12/rotation updates,
which include the exact full-BF loss, measured `257.5 ms` GPU p50 (`261 ms`
p95) and `261 ms` UI p50 (`264.5 ms` p95). FFT-visible redraw stayed at `254 ms`
GPU / `258 ms` UI, and phase-sign flip did not rerun reconstruction.

### What made Reference-512 faster

The retained result is an architectural reduction in repeated work, not a
smaller scientific problem. Relative to the prepared-source baseline, complete
CLI wall time fell from a `46.75 s` median to a `24.16 s` quiet best (`1.94x`),
while fit time fell from `37.937 s` to `23.058 s` (`1.65x`). The benchmark still
uses all `8,937` logical BF pixels, float32/complex64 arithmetic, 200 seed-42
Optuna trials, Nelder-Mead, and an independent final reconstruction.

| Retained change | Why it is faster | Scientific invariant |
| --- | --- | --- |
| Exact detector-major `uint8` BF columns | Reads the required 2.34 GB integer evidence instead of retaining the roughly 9 GiB decoded detector stack. | Counts are exactly narrowed only because the measured maximum is 53; no derived float cache or detector binning is used. |
| One contiguous BF-major Hermitian `G_qk` layout | Pays the layout conversion once instead of implicitly repacking slices in every loss evaluation. | Complex64 values and Hermitian reconstruction are unchanged bit-for-bit. |
| Analytic zero-transfer and inactive-aperture elimination | Skips FFT/multiply work whose contribution is provably zero. | All logical BF pixels remain in the normalization and output reduction. |
| Fused two-candidate row evaluation | Shares detector geometry, aperture terms, and `G_qk` reads across the pair Optuna already requests. | Each candidate keeps its own float32 correction, FFT, phase, and loss. |
| Radix-8 eight-column Metal consumer with pack-aware dispatch | Improves column locality and evaluates adjacent sparse row packs with fewer dispatches. | Original reduction boundaries are accumulated in their original order. |
| Bounded 288-plane allocation class | Reuses one stable Metal allocation shape and avoids unified-memory allocator pressure. | Column consumers read only valid planes; unwritten padding never enters a reduction. |

The current profile is memory-traffic limited: the retained exact topology
moves about `3.43 TB` at an effective `120-125 GB/s`. Reaching the still-unmet
sub-20-second target requires eliminating the global row-IFFT write/read or an
equivalent exact decomposition; changing trials, BF evidence, precision, or the
objective is explicitly not an acceptable speedup.

### Cross-backend and native-size validation notes

The CUDA automatic-probe path now computes the mean diffraction pattern,
mean-plus-standard-deviation probe fit, and full-disk crop on CUDA. It no
longer calls the public host-returning detector reduction. On the reference
512 acquisition, a raw CUDA run selected the identical `8,937` pixels and
center `(94.88451385498047, 96.35952758789062)`. Its float32 endpoint loss was
`0.04469253495335579`; the canonical MPS bit-contract remains
`0.04469207674264908`. Another agent was using CUDA during this validation
window, so CUDA wall, load, write, and fit timings are deliberately excluded
from performance signoff. The run is correctness/geometry evidence only.

No native 128 or 256 scientist acquisition is available on this Mac. The
full-grid synthetic gate therefore uses CUDA to select and export one exact
293-pixel automatic BF companion, then gives that identical companion to MPS.
The current canonical CLI measurements are:

| Scan | MPS wall to HTML | MPS fit | Shared BF pixels |
| --- | ---: | ---: | ---: |
| 128x128 | `2.89 s` | `0.9420 s` | `293` |
| 256x256 | `3.93 s` | `1.9426 s` | `293` |

These are full native scan grids, but synthetic kernel/workflow checks rather
than real-data claims. Old 203/199-pixel companions were rejected because they
encoded the former intensity-threshold selection and cannot prove automatic
full-BF parity. Fit endpoints alone are also not a backend-parity criterion on
multi-basin synthetic objectives; the fixed-parameter reference suites passed
36/1 (MPS) and 14/23 (CUDA) pass/skip splits on their respective hosts.

One provenance trap is now explicit: `--calibration none` also sets
scan-detector rotation to zero. The reference-512 benchmark must use its
private calibration input containing rotation `158.88268568029937°` and the
fixed optimizer start `(0, 50, 0)`. A zero-rotation run was rejected rather
than recorded as a regression.

## Automatic full-probe MPS optimizer, 2026-07-26

The production MPS auto-BF path now applies the disk returned by the shared
`quantem.gpu.detector.auto_probe` fit when `bf_radius=None`. Previously it
computed an automatic radius but did not use that radius to filter positive
detector pixels. The GPU mean-DP reduction remains the expensive detector
operation; fitting the center/radius from the resulting `192x192` image is a
small host reduction.

The private Reference-512 `512x512x192x192` uint16 source, loaded losslessly as uint8 after
verified count narrowing, produced center `(94.8845, 96.3595)` and radius
`53.3599 px`. With `bf_intensity_threshold=0.0`, the complete positive-count
automatic disk contains `8,937` BF pixels. The SSB probe aperture is nonzero
for `2,464`; the other `6,473` analytically zero entries remain in the exact
normalization and are handled by the existing inactive-work elimination. No
BF pixel is removed from the automatic selection.

The reference system's 24 GB unified memory cannot safely retain the 9.0 GiB decoded scan,
roughly 9.4 GB Hermitian `G_qk`, and paired row scratch simultaneously. The
measured workflow therefore computes the mean DP and auto fit on the GPU-loaded
source, extracts 48 auto-disk edge columns missing from the older companion,
releases the complete scan, then prepares all 8,937 exact raw BF columns from
the lossless detector-major companion plus those extracted columns.

| Stage | First full run | Warm repeat |
| --- | ---: | ---: |
| GPU load, auto probe, exact BF staging, `G_qk` | `15.583 s` | `14.981 s` |
| Initial exact loss | `1.039 s` | `0.313 s` |
| Optuna, 200 exact candidates | `24.817 s` | `24.359 s` |
| Nelder-Mead | `4.466 s`, 30 physical calls | `4.445 s`, 30 physical calls |
| Final object + exact phase/loss | `0.290 s` | `0.249 s` |
| Fit critical path | `30.612 s` | `29.366 s` |
| Detection/load through final result | `46.199 s` | `44.352 s` |

Both runs returned bit-identical parameters
(`C10=73.1818862146`, `C12=14.0209629488`,
`phi12=0.4700365260`), final loss `0.04469207674264908`, phase statistics,
and amplitude statistics. The warm fit is `1.86x` faster than the older
`54.53 s` warmed full-BF checkpoint, although that older run used a manually
pinned 8,826-pixel disk rather than the new 8,937-pixel automatic policy; the
ratio is therefore a closely related workflow comparison, not a pure
same-selection kernel speedup.

### Full-field row-occupancy follow-up

The exact two-candidate 512 row kernel now processes four rows per
threadgroup instead of two. This raises the group from 128 threads and 16 KiB
of shared row storage to 256 threads and 32 KiB, the hardware limit on the
measured Mac, without changing arithmetic or reduction order. Full-field warm
pair calls improved from roughly `0.241-0.251 s` to `0.235-0.240 s`; all loss
bits and the canonical optimizer result remained unchanged.

The independent 200-trial signoff returned the same parameters, loss, phase,
and amplitude statistics. It measured `26.801 s` Optuna, `5.952 s`
Nelder-Mead, `33.882 s` fit, and `40.831 s` load-to-result wall time in a
pressure-affected run. The first exact call included `0.931 s` of compilation,
so this run is parity evidence rather than a new warmed-record claim.

Rejected exact experiments in this follow-up were removed: paired positive and
negative Fourier rows changed active-plane loss by about `2e-8` and was slower;
a conservative zero-aperture pre-cull was bit-identical but neutral; four-column
phase groups regressed warm pairs to about `0.25-0.27 s`; and eight row groups
exceeded the 32 KiB Metal threadgroup-memory limit. Collapsing logical BF
chunks improved pair calls by only about 4% and changed float32 reduction
association, so the canonical 512-logical-BF boundaries remain intact.
Three-row groups tied four rows within timing noise. Specializing the column
kernel for compact all-active storage unexpectedly regressed canonical pairs
to about `0.267-0.280 s`. Hoisting the small probe vector once per evaluation
also regressed the 18-boundary automatic field, even after changing it to
contiguous BF-major storage; MLX slice/view materialization outweighed the
saved launches. Both specializations were removed completely.

### Automatic-probe detector-sum reduction

The lossless-uint8 MPS automatic-probe path now reduces the resident scan in
two exact integer stages. A frame-block kernel reads four adjacent detector
pixels at once and writes bounded 1,024-frame partial sums; a second kernel
merges those partials before the existing per-chunk atomic accumulation. This
replaces one long serial frame loop per detector pixel with enough independent
work to fill the GPU while keeping the same uint32-safe integer sum.

On one already-loaded real `512x512x192x192` scan, alternating calls measured
`0.0875-0.0888 s` warm for the new reducer versus `0.1440-0.1469 s` for the
old kernel. Every detector-sum integer matched exactly. End-to-end automatic
probe timing reached `0.276-0.277 s` on warm fresh-process setup runs; complete
setup remained `4.11-4.15 s` because HDF5 decode and `G_qk` FFT dominate.
The focused MPS/SSB gate passed (`41 passed, 2 skipped`). Vectorizing the old
long-loop kernel directly was slower (`0.47-0.65 s` probe stage) and was
removed. Frame blocks of 512, 1,024, and 2,048 were exact; 1,024 retained the
best warm balance.

The subsequent setup audit separated the one-dispatch active gather from the
FFT. With all 2,464 active planes gathered together, Metal column gathering
takes only `0.058-0.066 s`; MLX `rfft2` takes `1.15-1.55 s` and is the active
preparation critical path. The complete safe setup is about `4.1-4.4 s`, of
which `1.75-1.88 s` is the lossless HDF5 decode, `0.29-0.34 s` is exact probe
detection, and `0.11-0.46 s` is full Python/Metal lifetime collection before
the memory-intensive optimizer.

Several setup shortcuts were rejected and removed. Releasing the source scan
before the FFT was neutral in alternating runs. Omitting full collection made
setup appear about `0.25 s` faster but left enough pressure to slow a canonical
fit to `41.76 s`; generation-0 and generation-1 collection had the same
failure. MLX row-then-column FFT, full complex FFT, dual-real packing,
MPSGraph FFT, and dummy prewarming all lost to native `rfft2`. A raw BF-column
companion produced a measurably different automatic center/radius and cannot
replace the complete scan for exact probe detection. Finally, precomputing the
aperture support on the host selected the same 2,464 entries but saved only
about `20-35 ms`, less than FFT variance, so the duplicate numerical path was
removed. These results put the safe setup floor near three seconds on this
machine without a derived complex cache: roughly `1.5 s` decode, `0.09 s`
probe reduction, `0.9 s` FFT, and required geometry/lifetime overhead.

Parallel HDF5 chunk-plan discovery was also rejected. In isolation, two
workers reduced the 27-file metadata scan from `0.469 s` to `0.116 s`, but the
production loader already discovers later file layouts while earlier GPU
decodes are in flight. Front-loading those plans removed that overlap;
alternating fresh-process loads were neutral to slightly slower. Keep plan
discovery demand-driven unless a replacement preserves decode overlap.

A fresh canonical signoff with the retained reducer reached `4.008 s` setup:
`1.747 s` load, `0.301 s` automatic probe, `0.058 s` active gather,
`1.335 s` MLX FFT, and `0.100 s` source release/collection. The complete run
returned the canonical parameters, loss, phase statistics, and amplitude
statistics bit-for-bit. Its fit took `44.814 s` (`32.614 s` Optuna and
`9.193 s` refinement) while the Mac was deeply swapped, so `48.825 s` total is
pressure-state parity evidence, not a replacement for the faster retained
kernel signoff. An immediate second run showed the pressure explicitly
(`9.017 s` load and `6.602 s` probe) while still returning every canonical
scientific value exactly.

### Stateful cache-blocked row/column feasibility gate

A cache-blocked Metal prototype split each original 512-logical-BF boundary
into smaller row-IFFT microtiles, immediately consumed each tile with the
column IFFT, and carried per-pixel phase/phase-square state between tiles. The
state load/store preserves the original sequential BF addition order; sampled
paired losses were bit-identical to the retained kernel. No BF evidence,
precision, FFT arithmetic, or optimizer input changed.

The M5 did not expose a useful cache-resident speed tier. Warm paired calls
were about `0.301 s` for the retained boundary, `0.309 s` at 128 BF,
`0.311 s` at 64-96 BF, and `0.489 s` at four BF. Four BF reduces paired row
scratch to about 16 MiB, but requires more than 1,200 producer/consumer
dispatches across the automatic field. Deferring synchronization reduced only
a few milliseconds. The `~0.19 s` four-BF penalty is already as large as the
entire possible launch-overhead recovery, while the smaller scratch produced
no compensating bandwidth gain. A raw-command-buffer port therefore cannot
plausibly halve this topology on the measured hardware; it could at best
recover the added dispatch cost. The stateful implementation and environment
switches were removed completely rather than shipping a slower experimental
path.

## User-facing target

Microscopists need to steer aberration controls while viewing the same
full-BF reconstruction used for the final result. For native full-resolution
live display, the current targets are:

- `512x512`: at least 30 FPS (`<=33.3 ms/redraw`).
- `1024x1024`: at least 10 FPS (`<=100 ms/redraw`).

Do not claim these numbers from detector binning, scan cropping, fewer BF
pixels, or saved complex64 caches. Those are separate preview/export choices.

## Final exact MPS optimizer handoff, 2026-07-26

The accepted MPS path keeps all 2,826 radius-30 BF pixels, float32/complex64
kernel inputs, 200 seed-42 Optuna candidates, Nelder-Mead refinement, and final
independent reconstruction. No crop, bin, candidate reduction, saved result,
or approximate objective is used.

| Exact workflow stage | Original profile | Final committed tree | Improvement |
| --- | ---: | ---: | ---: |
| Optuna, 200 candidates | `34.485 s` | `21.823 s` | `1.58x` |
| Nelder-Mead refinement | `12.522 s`, 77 calls | `6.516 s`, 53 physical calls | `1.92x` |
| Comparable fit critical path | `49.392 s` | `28.658 s` | `1.72x` (`42.0%` lower) |

The final best is `C10=73.6415583633`, `C12=14.8295713639`, and
`phi12=0.4843161759`, with exact/final loss `0.11136186867952347`.
Parameters, loss, phase statistics, and amplitude statistics are bit-identical
across the final endurance runs. Sustained pair/scalar medians are
`0.21316/0.12253 s` with no late thermal slowdown.

Ten independent committed-tree reproductions at the end of the seven-hour pass
again returned those values exactly. Comparable fit min/median/max was
`28.788/28.924/29.999 s`; after the first memory-pressure outlier, the nine
warm fits narrowed to `28.788/28.915/28.953 s` with `0.060 s` standard
deviation. Optuna ranged from `21.889-22.840 s`, 53-call refinement from
`6.530-6.797 s`, and preparation from `0.955-2.260 s`. These are validation
repeats rather than a new best; the authoritative minimum above remains the
isolated `28.658 s` run.

The final focused MPS/SSB/IO gate passed ten consecutive times (`350 passed,
30 skipped` in aggregate). The repository-wide gate remained `143 passed,
70 skipped` plus one unchanged `main` failure in a stale WebGPU source-string
assertion; neither that test nor its TypeScript source is modified by this
branch.

A later 30-minute footprint audit ran under a slower machine state. The clean
retained tree took `32.317 s` for the fit path (`24.737 s` Optuna and `7.250 s`
refinement), but returned the exact canonical parameters, final loss, phase
statistics, and amplitude statistics above. A simultaneous 1 GiB native Metal
compute-copy probe measured min/median/max `95.38/111.77/130.02 GB/s`, below
the earlier `116.7 GB/s` sustained median. This is recorded as pressure-state
parity evidence, not a replacement performance signoff. The repository gate
again returned `143 passed, 70 skipped` plus the same unrelated stale WebGPU
source-string assertion.

Terminal retained-tree thermal soaks in that state remained stable and exact.
Three hundred pair calls totaled `65.726 s`, with min/median/p95/max
`0.21255/0.21868/0.22482/0.23042 s`; 300 scalar calls totaled `38.038 s`, with
`0.12060/0.12647/0.13113/0.13594 s`. Every reported loss was unchanged. These
medians are above the quiet signoff in the same direction as the lower copy
bandwidth, with no late correctness or allocator failure. A subsequent full
fit measured `30.400 s` and returned every canonical value exactly; a final
200-pair sample had `0.21998 s` median and unchanged losses. The focused gate
passed (`25 passed, 3 skipped`).

The accepted topology moves about `3.43 TB`. Native Metal compute-copy probes
measure `116.7-121.6 GB/s` sustained median and about `134 GB/s` best, while
the real fused kernels consume an effective `120.0-124.5 GB/s`. Even the best
observed bandwidth puts the current exact traffic above `25.6 s`; under 20 s
requires a new exact decomposition that eliminates the global row-IFFT
write/read, not another optimizer or launch tweak.

The requested `500 GB/s` is meaningful as a current-work-equivalent target.
Removing only the row write/read leaves one `5.157 GB` active `G_qk` read for
each of 100 pair and 55 scalar/final calls, about `0.799 TB` instead of
`3.43 TB`. That is a `4.29x` traffic reduction; at the established
`116.7 GB/s` sustained physical median it represents about `500.6 GB/s` of
current-topology work. It also gives a `6.85 s` memory-only floor before the
replacement transform arithmetic. Thus `500 GB/s` requires intermediate
elimination specifically; it is not a physical DRAM bandwidth setting.

A native Metal copy working-set sweep found no hidden physical 500-GB/s cache
tier. Median bandwidth at `4/16/64/256/1024 MiB` per buffer was
`141.6/129.5/131.0/127.3/112.8 GB/s`; the noisy 4-MiB peak was `297.4 GB/s`.
One active complex plane is `5.157 GB`, so shrinking BF chunks cannot place
the exact intermediate in such a cache regime. The 500-GB/s goal must remain
an effective-byte metric.

MLX allocation counters quantified the exact chunk tradeoff. With `2.771 GiB`
of prepared evidence active, peak memory was `4.775/4.275/3.775 GiB` for
`512/384/256` BF. A complete 256-BF fit retained the canonical parameters,
53 physical refinement calls, final loss, and phase/amplitude statistics, but
its `29.794 s` pressure-state fit did not beat the quiet 512-BF signoff.
Same-process alternating calls settled the speed question: 512-BF median was
`0.21716 s` versus `0.22969 s` for 256 BF, and all 24 pair-loss bit patterns
matched. Thus 256 is a valid measured low-memory tradeoff for this dataset,
not a faster default; 384 also changed one sampled loss by one float32 ulp.

## MPS radix-8 exact-loss checkpoint, 2026-07-25

The reference system is a 24 GB Apple M5 MacBook Pro with 10 GPU cores. The benchmark used
the private Reference-512 `512x512x192x192` dataset and its lossless uint8 BF-column
companion. The radius-30 policy selected 2,826 BF pixels; no scan crop,
detector binning, BF subsampling, reduced precision, or objective change was
used.

The accepted MPS path replaces both 512-point radix-4 FFT stages in the exact
optimizer with 64-thread radix-8 Metal kernels. Each transform now uses three
radix-8 stages instead of four radix-4 stages plus a final radix-2 stage. The
column kernel drops from six threadgroup barriers per BF to three, and the row
kernel reduces both barrier count and thread count while retaining float32
complex evidence and phase/loss accumulation.

| Quantity | Previous warmed | Radix-8 warmed | Speedup |
| --- | ---: | ---: | ---: |
| Exact loss, one candidate | `0.302-0.307 s` | `0.152-0.161 s` | `1.9-2.0x` |
| Exact loss, two candidates | `0.407-0.416 s` | `0.308-0.313 s` | `1.3x` total, `6.4 eval/s` |
| 10-iteration Nelder-Mead, 22 calls | `6.417 s` | `3.279 s` | `1.96x` |

The before/after Nelder-Mead run returned identical fitted values
(`C10=73.7906607`, `C12=14.6919939`, `phi12=0.48484473`) and identical
reported loss (`0.1113653481`). Direct real-data loss differed from the
radix-4 path by at most about `7.5e-9`, consistent with float32 association.
The focused MPS reference suite passed (`20 passed, 2 skipped`).

Rejected experiments in the same benchmark session included fast `atan2`, exact
zero-support masks, fused scalar reduction, algebraic geometry rewrites,
larger row threadgroups, reordered/coalesced column thread layouts, MLX's
built-in column FFT, and fused HDF5 decode/unshuffle variants. Each either
regressed or stayed within timing noise and was reverted.

### MPS fused candidate-pair checkpoint, 2026-07-26

The first accepted follow-up fuses the two exact candidates used by the
Optuna batch into one row/gamma threadgroup. Detector geometry, aperture
weights, and the Hermitian `G_qk` fetch are shared; each candidate retains its
own float32 aberration phase, gamma correction, radix-8 row FFT, column FFT,
and loss reduction. This changes scheduling only: BF evidence, objective,
precision, trial count, seed, and optimizer decisions are unchanged.

| Quantity | Radix-8 baseline | Fused pair v1 | Change |
| --- | ---: | ---: | ---: |
| Exact loss, two candidates, warmed | `0.308-0.313 s` | `0.283-0.287 s` typical | `~8%` lower wall time |
| Optuna, 200 exact candidates | `34.485 s` | `29.386 s` | `1.17x` |
| Nelder-Mead, 77 exact calls | `12.522 s` | `12.374 s` | unchanged single-candidate path |
| Full measured workflow | `49.392 s` | `44.235 s` | `1.12x` |

The complete run reproduced the same best parameters. Final loss was exactly
`0.11136213690042496`; final phase mean/std and amplitude mean/std agreed at
the recorded float32 values. The focused MPS/CUDA reference suite again
passed (`20 passed, 2 skipped`). The remaining under-20-second gap is now
mostly the single-candidate Nelder-Mead path and the still-sequential column
FFT/reduction work; the pair fusion alone is not sufficient.

Rejected follow-ups during this pass were a SIMD-shuffle/double-buffered
column FFT (about 5% slower from register pressure), a CUDA-style polynomial
`atan2` approximation (neutral for one candidate and slower for two), and an
algebraically factored gamma/single-aperture branch (neutral to slower). All
were reverted rather than retained as speculative complexity.

### MPS wide radix-8 column checkpoint, 2026-07-26

The next accepted version raises the radix-8 column kernel from two to eight
columns per threadgroup: 512 threads and 32 KiB of threadgroup storage, the
measured useful edge on the reference M5. It also routes final 512 exact phase/loss
through the same radix-8 column implementation instead of the remaining
radix-4 scalar kernel. The FFT, `atan2`, BF accumulation, and float32 outputs
are unchanged mathematically; only the radix association and scheduling
differ.

| Quantity | Fused pair v1 | Wide column v2 | Change from radix-8 baseline |
| --- | ---: | ---: | ---: |
| Exact loss, two candidates, warmed | `0.283-0.287 s` | `0.262-0.278 s` | `~14%` lower typical wall time |
| Optuna, 200 exact candidates | `29.386 s` | `26.883 s` | `1.28x` |
| Nelder-Mead, 77 exact calls | `12.374 s` | `11.623 s` | `1.08x` |
| Final exact phase/loss | `1.268 s` noisy run | `0.270 s` | radix-8 path, `4.7x` in these runs |
| Full measured workflow | `44.235 s` | `39.964 s` | `1.24x` versus `49.392 s` baseline |

The 200-trial seed-42 run returned the identical best parameters. Phase mean
changed by `1.4e-9` and phase standard deviation by `1.5e-8`, consistent with
float32 FFT association; amplitude statistics were unchanged at the recorded
float32 values. The focused reference suite passed (`20 passed, 2 skipped`).

Rejected measurements for this version: pairing both candidates inside one
256-thread column threadgroup regressed to `0.305-0.321 s`; removing the six
per-chunk synchronizations regressed warmed calls to `0.276-0.290 s` and made
the first two calls exceed `1.6 s`; chunks from `128` through all `2826` BF
did not break the same compute floor. Early end-to-end `K_BF` timings were too
noisy to select a new default, so the follow-up below isolates the column
stage. The component timings localize roughly half of each paired call to
row/gamma and half to column FFT/phase accumulation.

### MPS full-chunk column accumulation checkpoint, 2026-07-26

With eight columns resident per threadgroup, the earlier `K_BF=32` grouping
was no longer optimal. A direct column-only sweep from `24` through `2826` BF
showed that keeping each complete input chunk in the threadgroup removes
partial phase-image writes and the follow-up group reduction. The 512 default
therefore uses `K_BF=4096`, which covers every supported exact-loss chunk in
one group while preserving every per-BF FFT and phase sample.

| Quantity | Wide column v2 | Full-chunk column v3 | Change from radix-8 baseline |
| --- | ---: | ---: | ---: |
| Exact loss, two candidates, warmed | `0.262-0.278 s` | `0.247-0.257 s` | `~19%` lower typical wall time |
| Optuna, 200 exact candidates | `26.883 s` | `25.423 s` | `1.36x` |
| Nelder-Mead, 77 exact calls | `11.623 s` | `11.229 s` | `1.11x` |
| Full measured workflow | `39.964 s` | `38.178 s` | `1.29x` versus `49.392 s` baseline |

The full run again returned the identical best parameters and restored the
recorded final loss `0.11136213690042496`. No BF samples, trials, precision,
or reconstruction outputs were removed. The remaining time is almost wholly
the exact row/gamma and column FFT/`atan2` work; Optuna orchestration and final
reconstruction together are now well below one second.

### MPS zero-aperture elimination checkpoint, 2026-07-26

The real radius-30 selection contains 367 of 2,826 BF pixels whose aperture
amplitude is identically zero for every candidate. Their corrected spectrum
contains only the positive DC term, so their per-BF phase and phase-squared
contributions are exactly zero. Version 4 keeps all 2,826 BF pixels in the
normalization but writes the analytic DC row and skips their column FFT/phase
work. This is sparse zero elimination, not BF subsampling.

| Quantity | Full-chunk column v3 | Zero-aperture v4 | Change from radix-8 baseline |
| --- | ---: | ---: | ---: |
| Exact loss, two candidates, warmed | `0.247-0.257 s` | `0.230-0.236 s` | `~25%` lower typical wall time |
| Exact loss, one candidate, warmed | `~0.146 s` workflow average | `0.132-0.137 s` | `~9%` |
| Optuna, 200 exact candidates | `25.423 s` | `23.735 s` | `1.45x` |
| Nelder-Mead, 77 exact calls | `11.229 s` | `10.286 s` | `1.21x` |
| Full measured workflow | `38.178 s` | `36.155 s` | `1.37x` versus `49.392 s` baseline |

The run returned the identical best parameters. Final loss differed by
`1.49e-8`, within float32 FFT/reduction association, and phase/amplitude
statistics retained float32 parity. The focused MPS/CUDA suite passed
(`20 passed, 2 skipped`).

Additional rejected hypotheses since v3: speculative reflection plus inside-
contraction Nelder-Mead preserved decisions but increased refinement to
`13.40 s` by evaluating 104 candidates instead of 77; pre-folded aberration
coefficients were faster in an isolated row probe but neutral end to end;
explicit fast `atan2`, vector `sincos`, aperture interior/exterior branches,
and moving the final column barrier were neutral; treating all BF transforms
as real was invalid outside the central/zero-aperture subset. Each code path
was reverted.

### Optional MPS four-candidate fusion, 2026-07-26

The row kernel can also share geometry and `G_qk` loads across four exact
candidates while staying at 32 KiB threadgroup memory by processing two rows
per group. Warm batch-4 calls improved from `0.529-0.545 s` to
`0.450-0.458 s` (`~16%`). Batch-2 behavior and the default optimizer policy
are unchanged.

This is deliberately not counted as a no-loss workflow improvement. With
Optuna batch 4, the 200-candidate stage took `23.737 s`, essentially tied with
v4 batch 2 after orchestration, and the changed TPE ask/tell trajectory ended
at loss `0.1113640293`, about `1.9e-6` worse than the batch-2 result. The
four-candidate kernel remains useful for callers that already choose that
batch, but batch 2 remains the scientific default.

The generic batch path also had a latent launch mismatch above four
candidates: its shader reserves two rows per threadgroup, while the launcher
requested four and caused half the rows to return before writing. Batch 8 then
reported invalid losses near `0.05` instead of the exact `0.1114` band. The
launcher now uses the shader's two-row contract for every batch above one;
real-data batch-8 losses return to exact agreement and the focused MPS suite
passes (`23 passed, 2 skipped`). This is a precision fix, not a default speed
win: warmed batch 8 is about `0.953-0.964 s` (`119-121 ms/candidate`), slower
per candidate than the fused batch-2 optimizer path.

### MPS zero-transfer q-space checkpoint, 2026-07-26

After the paired geometry work, it became profitable to branch on pixels where
both shifted apertures are exactly zero. Those pixels have identically zero
transfer for every candidate, so the row kernel writes exact complex zero and
skips the Hermitian `G_qk` fetch, aberration `sincos`, gamma normalization, and
complex multiply. All FFTs, BF pixels, normalization, and float32 outputs are
retained.

Alternating 25-call real-data runs measured a typical paired call near
`0.228-0.231 s`, versus `0.232-0.235 s` without the branch. The complete
seed-42 workflow measured `34.992 s`: `23.405 s` for 200 Optuna candidates,
`9.963 s` for the same 77-call Nelder-Mead refinement, and the identical best
parameters and final loss `0.11136215180158615`. This is a small but robust
`~1.8%` end-to-end win over the v5 batch-2 repeat (`35.621 s`).

### MPS paired-row occupancy checkpoint, 2026-07-26

The fused pair originally processed four rows in a 32 KiB threadgroup, which
limited each GPU core to one resident row group. Processing two rows in a
16 KiB group allows two independent groups to reside together and hide
geometry/`sincos` latency. Total pixels, row FFTs, and candidate arithmetic are
unchanged. A one-row/8 KiB variant added scheduling overhead and was slower.

Warm paired calls moved from the `~0.228-0.231 s` v6 band to typically
`0.223-0.227 s`. The full workflow measured `34.466 s`, including `22.998 s`
for Optuna and `9.944 s` for the same 77-call refinement. Best parameters,
loss `0.11136215180158615`, phase statistics, and amplitude statistics were
identical to v6.

### MPS dead inactive-write elimination checkpoint, 2026-07-26

The zero-aperture path already marks its 367 inactive BF entries for the
column kernel, which checks that mask before reading the row-IFFT scratch.
Version 8 therefore stops writing the otherwise-unused analytic DC/zero
planes. All 2,826 BF entries remain in the normalization, and the active BF
arithmetic and reduction order are unchanged.

Warm paired calls moved from `~0.223-0.227 s` to `~0.213-0.218 s`. The full
seed-42 workflow measured `33.558 s`: `21.864 s` for 200 exact Optuna
candidates, `9.490 s` for the same 77-call Nelder-Mead refinement, `0.045 s`
for the object reconstruction, and `0.242 s` for the final exact phase pass.
It returned the identical parameters, loss `0.11136215180158615`, phase
statistics, and amplitude statistics as v7. The focused suite passed
(`22 passed, 3 skipped`).

Two topology probes were rejected before this version. Running the paired
candidate FFTs concurrently in a 3-D threadgroup preserved exact losses but
stayed at `0.224-0.233 s`; the extra threadgroup depth exchanged occupancy for
concurrency without a net win. A transposed row-IFFT intermediate reduced the
column stage from about `0.109-0.113 s` to `0.094-0.098 s`, but scatter writes
raised the row stage from about `0.118 s` to `0.139-0.151 s` even with a
four-row shared-memory store tile. The full call regressed to `0.232-0.249 s`,
so both probes were reverted. A future transposed design must eliminate the
intermediate or use a genuinely coalesced tile without lowering row-kernel
occupancy.

Further v8 follow-ups also failed the acceptance gate. A 2.96 GB float32
aperture cache preserved the displayed losses but regressed warm pairs to
`0.249-0.256 s`, showing that recomputation is cheaper than the added unified-
memory traffic. Vector `atan2`, reciprocal-square-root gamma normalization,
removing exact-grid bounds checks, and an isotropic-detector aperture
specialization were neutral; reciprocal square root also moved one benchmark
loss by about `2e-8`, so it was rejected on both speed and parity grounds.
A local quadratic model fit to the 200 exact Optuna observations reduced
Nelder-Mead from 77 to 72 calls, but converged at loss `0.1113962233` instead
of `0.1113621518`. It was rejected: fewer exact calls do not count when the
optimization result loses precision.
Two independent single-candidate MLX streams preserved both exact losses but
took about `0.233-0.240 s`, slower than the fused pair. Pipelining independent
BF chunks across two streams took `0.223-0.235 s`; three streams improved that
to `0.220-0.225 s` but still did not beat the committed `0.213-0.218 s` path.
The M5 is already saturated by the fused kernels, so extra command queues are
not a substitute for reducing row/column arithmetic or intermediate traffic.
An exact quadratic gamma factorization reused the q-only probe phase and
replaced two shifted-phase `sincos` calls with one bilinear-phase call, but the
extra q-probe load/arithmetic regressed pairs to `0.217-0.226 s` and moved one
loss by `2e-8`; it was removed. An exact empty-row aperture bound identified
34,784 of 1,446,912 `(BF,row)` FFTs as zero, but the 2.404% sparsity was too
small: the full workflow stayed `33.545 s`, statistically identical to v8.
Full-plane `G_qk` expansion regressed to `0.224-0.260 s` while doubling
resident evidence to 5.93 GB. Same-stream `mx.async_eval` chunk submission
regressed to `0.219-0.225 s` from extra in-flight memory. All were reverted.
Reordering the fused pair so both candidates shared each radix-8 stage barrier
was exact but also neutral: row time stayed `0.107-0.111 s` and paired calls
stayed `0.218-0.223 s`. The compiler/scheduler already hides that independent
candidate synchronization; removing source-level barriers does not remove the
row/column arithmetic or intermediate-memory floor.
The clean committed rerun measured `32.757 s` end to end: `21.887 s` for the
same 200 Optuna candidates, `9.421 s` for the same 77-call refinement, and
`0.319 s` for final object plus phase. Parameters, loss, phase, and amplitude
statistics remained identical to v8. Physically compacting the 2,459 active BF
entries while retaining the original 2,826-BF normalization moved one test
loss by only one float32 ulp, but stayed at `0.215-0.224 s` per paired call.
The existing early returns already make the 367 inactive dispatch slots nearly
free, so active-array packing was reverted.

### MPS exact-loss memory roofline, 2026-07-26

A 1 GiB float32 Metal copy on the reference system sustained `112-130 GB/s` after warm-up.
The 2,459 analytically active BF entries occupy `5.157 GB` per complex64
512-plane stack. The current exact single candidate moves approximately three
such plane-equivalents through global memory (`G_qk` read, row-IFFT write,
column-IFFT read), or `15.471 GB`; a fused pair shares `G_qk` but writes and
reads two row stacks, or `25.784 GB`. The copy roof predicts `119-138 ms` for a
single and `198-230 ms` for a pair. Measurements are about `123 ms` and
`218-219 ms`, respectively, so the hot path is already at the reference system's sustained
unified-memory roof once inactive work is excluded.

For 100 paired Optuna calls plus 78 single calls (initial and 77-call
Nelder-Mead), the present topology moves about `3.785 TB`. Finishing that
traffic in 20 seconds requires roughly `189 GB/s`, above the measured M5 copy
roof before FFT arithmetic, shared-memory traffic, or final reconstruction is
counted. This does not relax the exact/full-BF target: it narrows the required
breakthrough to eliminating the global row-IFFT write/read, most likely with a
tiled or persistent 2-D FFT. More sampler scheduling, active packing, or scalar
geometry changes cannot produce the requested 2-3x on this topology.

After the wider simplex, flat-tail cutoff, and float32-input cache, the current
workflow issues 100 paired Optuna calls, one initial scalar call, 53 physical
refinement calls, and one final phase call. Their row topology moves about
`3.43 TB`, giving a `26.4 s` memory-only floor at the best measured `130 GB/s`
copy rate. The measured fit critical path is `28.757 s`, within about 9% of
that optimistic floor even before accounting for FFT/shared-memory arithmetic
and final object work. Reaching 20 seconds with this same traffic would require
about `171 GB/s`; the remaining target therefore still requires eliminating a
global row-intermediate pass rather than another launch or optimizer constant.

A final native Metal compute-copy probe tightened that estimate without host
wall-clock ambiguity. A 512 MiB source and destination moved 1.074 GB per
dispatch through a `uint4` compute kernel. Across 30 steady calls, command-
buffer GPU timestamps measured min/median/max bandwidth of
`99.0/121.6/133.5 GB/s`; median host wait exceeded median GPU duration by only
about `0.15 ms`. At the measured median bandwidth, the same `3.43 TB` topology
predicts a `28.21 s` traffic floor, within `1.6%` of the final `28.66 s` fit
path. Even the fastest single copy observation predicts `25.69 s`. This makes
the under-20 requirement an intermediate-traffic problem, not an unmeasured
Python, command-queue, or ordinary shader-tuning gap.

Repeating the same compute probe with 1 GiB source and destination buffers
(2.147 GB traffic per dispatch) ruled out a working-set/cache artifact. Sixteen
steady GPU timestamps gave min/median/max `103.8/116.7/134.0 GB/s`. The peak is
unchanged and the larger sustained median is slightly lower; even its best
device state puts `3.43 TB` above `25.6 s` before FFT arithmetic. The roofline
conclusion is therefore stable across both 512 MiB and 1 GiB buffers.

The queried Apple M5 device limit is `32,768` bytes of threadgroup memory. One
exact `512x512` complex64 plane is `2,097,152` bytes, exactly 64 times larger.
Metal exposes barriers within a threadgroup/simdgroup but no grid-wide barrier
inside this compute dispatch, so a conventional persistent 2-D FFT cannot keep
the full row intermediate on chip and then synchronize all rows for columns.
Any topology that removes the global pass must use a new exact decomposition
or accept recomputation; increasing a threadgroup tile cannot bridge this gap.

A persistent one-threadgroup-per-BF formulation does not evade that bound. One
group could loop over all 512 rows, fence its global row writes, and then run
the column FFTs without a grid-wide barrier. It cannot, however, preserve the
accepted column kernel's all-BF register accumulation: a BF-owned group must
emit one `512x512` phase plane per BF for a later ordered reduction, adding
roughly another full-field write/read intermediate, or serialize all 64 column
tiles and BF terms inside one group. The former increases the multi-terabyte
traffic floor and the latter underutilizes the ten-core GPU. A viable fused
topology must keep the BF-major phase reduction local as well as remove the row
intermediate; merely moving the synchronization boundary into one persistent
group is not enough.

The reference system reports a 10-core Apple M5 with Metal 4 support. The installed command-
line developer tools do not provide `xctrace`, the offline `metal` compiler, or
`metallib`, so this run cannot add trustworthy Metal occupancy/cache counters
or inspect an offline shader binary. Timing starts after MLX kernel compilation,
and the sustained kernel-versus-copy roofline is therefore the available
hardware gate. Counter-free source tweaks are not accepted unless repeated
end-to-end timing and exact parity demonstrate a win.

Performance measurements were taken on macOS `26.0.1` with 24 GB unified
memory, AC power connected, battery charged, and Low Power Mode disabled on
both AC and battery policies. This base-M5 system exposes no separate High
Power Mode entry, so the sustained roofline was not measured under an
accidental low-power policy.

Querying the Metal device directly does not recover the missing profiler data:
`device.counterSets()` exposes only the `timestamp` set with one
`GPUTimestamp` counter. No occupancy, cache, bandwidth, or instruction counter
is available to this process, and MLX does not expose its internal command
buffer for timestamp sampling around these kernels. The sustained call soaks
and external copy roofline therefore remain the reproducible profiling record.
MLX's higher-level `metal.start_capture()` route was also checked, but it fails
with `Capture layer is not inserted`; installing the full Metal capture layer
is required before the reference system can emit an Instruments `.gputrace`.

Lossless compression of the row intermediate was checked on real 32-BF
complex64 chunks. Analytically inactive chunks are all zero, but production
already skips their writes and column reads. Representative active chunks
compressed only `1.08-1.23x` with zlib level 1. The three mantissa-dominated
byte lanes carried roughly `7.5-8.0` bits/byte; only the exponent/sign lane was
low entropy (`2.8-3.0` bits/byte). An exact pack/unpack pass cannot repay its
extra reads, writes, and arithmetic with at most about 8-19% active-byte
reduction, so lossless row compression is not a viable under-20 topology.

The traffic budget makes that codec threshold explicit. About `2.63 TB` of
the accepted `3.43 TB` workflow is the complex64 row-intermediate write/read.
Even a zero-overhead format that perfectly halves every float's high byte
would encode one complex value in seven rather than eight bytes, saving only
about `0.33 TB`; the resulting `3.10 TB` still has a `23.1 s` floor at the
best observed `134 GB/s`. Reaching 20 seconds requires removing at least
`0.75 TB` (`28.5%`) of row traffic at that best bandwidth, or about `1.10 TB`
(`41.7%`) at the sustained `116.7 GB/s` median, before counting codec work.
Any future exact row codec therefore needs an effective representation below
about `5.7` bytes/complex in the optimistic case and `4.7` bytes/complex at
the sustained median; a seven-byte sign/exponent packing path cannot qualify.

Private Metal texture storage was also rejected as a hidden compression/cache
tier. A raw Metal feasibility probe moved the same 256 MiB complex surface
through buffer-to-buffer-to-buffer and buffer-to-private-RG32Float-texture-to-
buffer paths. High-entropy bits measured `117.9 GB/s` for buffers and
`120.5 GB/s` for texture (`2.1%`); a row-like distribution with redundant
sign/exponent bytes but random mantissas measured `122.2 GB/s` for buffers and
`121.5 GB/s` for texture. The latter is slightly slower, proving that Apple
texture tiling/compression does not provide the missing exact-intermediate
bandwidth on this M5.

Fully private Metal buffers likewise expose no faster storage tier. A matched
256 MiB two-copy compute probe measured `117.8 GB/s` with shared source/output
endpoints and a private intermediate, versus `117.2 GB/s` when source,
intermediate, and output were all `storageModePrivate`. A raw backend cannot
reach the required `171 GB/s` merely by changing MLX-visible allocation mode.

Sparse row-producer packing is the first accepted launch/allocation reduction
from this roofline pass. The automatic full-BF aperture leaves 12 nonempty logical
512-BF reduction boundaries in the private Reference-512 case. Adjacent
boundaries are now allowed to share one row-IFFT allocation up to 300 packed
BF planes, while the column kernel still consumes and adds every original
boundary separately and in the original order. This reduces the measured row
producer count from 12 to 10 without changing the float32 reduction
association. Two complete 200-trial plus Nelder-Mead runs measured `32.229 s`
and `32.164 s` for the fit and `38.05 s` and `37.86 s` CLI wall time. The prior
accepted best was `34.898 s` fit / `40.48 s` wall. Both new runs returned the
same 231 records, `C10=73.18188621458395`, `C12=14.020962948808993`,
`phi12=0.4700365259977606`, and loss `0.04469207674264908`. The 20-trial
production gate retained trial hash
`1bc4474983a3f37fb162b592969983c645150f4118ab1729e8cedaf3297615a2`.
Direct row-major and tiled-storage subrange tests additionally require array-
exact column sums and squared sums.

The row pack is deliberately capped at 300. A 400-plane cap raised peak
footprint from about `19.42 GB` to `20.00 GB` and did not improve repeated
20-trial timing. A 512-plane cap regressed Optuna from about `3.27 s` to
`6.75 s`. Those variants were removed; do not infer that fewer row dispatches
are faster once the unified-memory allocation crosses this device's pressure
cliff.

Apple's native MPSGraph multi-axis complex FFT was rejected as a replacement
transform. A raw `512x512` inverse-FFT probe peaked near `14.3k` planes/s for
batches of 64 through 256. The retained fused candidate-pair evaluator already
processes about `19.2k` planes/s while also generating the correction and
accumulating phase/loss. MPSGraph is therefore slower before accounting for
the extra custom-kernel handoffs or its different float32 FFT association. The
standalone probe was removed without entering the production tree.

Compiling all ten sparse row packs as one MLX graph was also rejected. It let
MLX plan one graph without the per-pack evaluation boundaries, but changed the
canonical 20-trial hash from `1bc447...` to `8248ce...`, raised the measured
peak footprint slightly, and regressed Optuna from `2.940 s` to `3.154 s`.
Evaluation boundaries are part of the accepted float32 association and cannot
be removed by a whole-graph compile. The cache, flag, and graph refactor were
deleted in full.

BF-major `G_qk` materialization is the largest accepted improvement in this
pass. MLX's Hermitian RFFT result for the real 2,464-plane evidence had physical
byte strides `(4096, 8, 10092544)` for logical shape `(2464, 512, 257)`, so the
repeated row kernels paid an implicit layout conversion for each BF slice. The
prepared backend now calls `mx.contiguous()` once after the RFFT and releases
the former layout through the normal post-preparation cache clear. This is a
bit-preserving storage-order change, not another evidence allocation.

Three 20-trial gates measured Optuna at `2.357`, `2.313`, and `2.321 s`, versus
`2.940 s` immediately before the change; the production gate measured
`2.386 s`. Every run retained canonical trial hash
`1bc4474983a3f37fb162b592969983c645150f4118ab1729e8cedaf3297615a2`.
Two complete 200-trial plus Nelder-Mead signoffs then measured `25.711` and
`25.903 s` fit time and `30.99` and `31.40 s` CLI wall. Their Optuna stages
were `20.605/20.684 s`, refinement `3.860/3.945 s`, and peak footprints
`18.66/18.52 GB`. Both returned all 231 records, 30 physical refinement calls,
the canonical aberrations, and loss `0.04469207674264908` exactly. The prior
row-packed version measured `32.164/32.229 s` fit and `37.86/38.05 s` wall.

The canonical CLI then removed unrelated eager widget imports in
`quantem.widget` commit `b80aa449`. Starting `python -m quantem.widget.cli`
fell from `2.02 s` and `417 MB` RSS to `1.29 s` and `344 MB`; public names are
still resolved from one static module/symbol table. A matched full-data
20-trial A/B measured `9.80 s` before versus `8.77 s` after, with identical
trial records. The clean committed 200-trial plus Nelder-Mead signoff measured
`31.12 s` CLI wall, `25.819 s` fit, `20.691 s` Optuna, `3.868 s` refinement,
and `1.993 s` source load. It retained all 231 records, 30 physical refinement
calls, `C10=73.18188621458395`, `C12=14.020962948808993`,
`phi12=0.4700365259977606`, and loss `0.04469207674264908` exactly. This is a
fixed control-plane saving; it does not change or relabel the MPS kernel time.

The MPS backend now retains the optimizer's already-prepared GPU state instead
of rebuilding the identical full-BF selection for the first interactive
preview. Two canonical 20-trial gates measured `8.29/8.34 s` CLI wall,
`3.820/3.716 s` fit, and `2.456/2.376 s` Optuna, with every trial record
unchanged. The corresponding complete 200-trial plus Nelder-Mead signoff
measured `30.35 s` wall, `25.615 s` fit, `20.556 s` Optuna, `3.823 s`
refinement, and `2.116 s` source load. It returned all 231 records, 30 physical
refinement calls, the canonical aberrations, and loss
`0.04469207674264908` exactly. This is private backend lifecycle state: the
public `SSBResult` and backend-neutral API remain unchanged, while the first
widget reconstruction no longer pays a second `0.786 s` preparation pass.

The same private completion handoff now retains the optimizer's exact final
float32 phase and loss for the initial widget preview. It does not derive phase
from `SSBResult.object_wave`, add a result type, or reuse the value after an
aberration or rotation change. The canonical 20-trial gate fell to `7.84 s`
wall with all 21 records bit-for-bit identical. A clean committed 200-trial
plus Nelder-Mead signoff then measured `30.13 s` wall, `25.663 s` fit,
`20.536 s` Optuna, `3.861 s` refinement, and `1.926 s` source load. All 231
records, 30 physical refinement calls, final aberrations, and loss remained
exact; the recorded `quantem.gpu` and `quantem.widget` worktrees were clean.

The core `quantem` package root then changed its five existing public module
exports to explicit lazy bindings in commit `c7f93f40`. CLI help fell from
about `1.29 s` and `344 MB` to `0.05 s` and `23 MB` because parsing a
ShowPtycho command no longer imports Torch, SciPy, Matplotlib, datasets, and
unrelated ptychography implementations. Core tests passed (`250 passed, 2
skipped`), and the public module names remain available. Widget commit
`8e73b6a8` added that core checkout to the fit artifact's software provenance.

The canonical 20-trial full-data gate measured `7.36 s` wall with all 21
records unchanged. Two provenance-bearing 200-trial plus Nelder-Mead repeats
measured `31.06 s` and `29.18 s`; the first was localized to a transient
`5.007 s` refinement. The accepted repeat measured `25.574 s` fit, `20.594 s`
Optuna, `3.727 s` refinement, and `2.058 s` source load. It reproduced all 231
records, 30 physical refinement calls, final aberrations, and loss exactly.
The artifact records clean `quantem c7f93f40`, `quantem.gpu 4e6fdaa`, and
`quantem.widget 8e73b6a8` commits. This is a control-plane improvement; the
remaining exact fit still has the same row-intermediate traffic floor.

The exact BF companion already stores the float32 DC value used by its
exporting SSB state. MPS now validates and reuses that scalar instead of
rereading all 2.343 GB / 8,937 logical columns solely to reconstruct the same
small detector-sum product. Old companions without `dc_value` retain the exact
Metal sum path, and an explicit public mean-DP request computes it lazily. The
optimizer still reads every one of the 2,464 aperture-active columns and builds
the identical complex64 `G_qk`; no scientific evidence or Fourier cache is
substituted.

The canonical 20-trial gate fell from `7.36 s` to `5.14 s` wall. Source open
became metadata-only, active-column read plus `G_qk` preparation measured
`1.080 s`, and maximum RSS fell from about `5.28 GB` to `3.59 GB`. All 21
records remained bit-for-bit identical. The clean committed 200-trial plus
Nelder-Mead signoff then measured **`26.45 s` wall**, `25.220 s` fit,
`20.360 s` Optuna, `3.528 s` refinement, and `0.906 s` preparation. It
reproduced all 231 records, 30 physical refinement calls, final aberrations,
and loss exactly; all three package worktrees were clean. Loading is no longer
the remaining under-20 blocker: the exact repeated row intermediate dominates.

The exact raw-Metal command-buffer prototype was retested after this layout
fix and removed. It retained the canonical hash, but Optuna measured `2.355 s`
against the `2.313-2.321 s` retained MLX range, peak footprint rose to
`20.09 GB`, and first-use shader compilation increased complete gate wall.
Once the evidence is BF-major, raw submission does not remove enough work to
justify a second dispatcher.

A cooperative contiguous-load variant of the retained eight-column consumer
was exact but substantially slower. Its 512 threads first staged every
column tile through threadgroup memory, synchronized, and then ran the same
radix-8 column transform. All focused tests passed and the real 21-record gate
remained bit-for-bit identical, but Optuna regressed from `2.317 s` to
`4.968 s`; the initial and final exact passes took `0.704/0.449 s`. Complete
gate wall rose from `5.14 s` to `8.86 s`. The extra threadgroup write/read and
barrier dominate any coalescing benefit on this M5, so the prototype was
removed and the original direct eight-column loads remain.

Serially evaluating both candidates inside one 512 column threadgroup was
also exact but slower. The variant reused the same 32 KiB column scratch and
kept each candidate's radix and reduction order unchanged, halving the number
of column groups. The canonical gate hash remained unchanged, but Optuna
regressed from `2.285 s` after restoration to `2.610 s`; the added source loop
also pushed initial/final scalar passes to `0.653/0.615 s`. The restored direct
candidate grid returned those passes to `0.221/0.240 s` and completed the gate
in `5.33 s`. Metal does not specialize the extra control flow cheaply enough,
so the prototype was removed and candidates remain independent column groups.

The exact safe-inside aperture predicate retained by CUDA was ported to the
512 MPS row producer and rejected. It proves the soft aperture is exactly one
before skipping its denominator square root/divide, while leaving the old
expression on every edge point. Focused references and all 21 real gate
records were bit-exact, but the extra predicate and two live radial values
crossed the Metal row kernel's occupancy/compiler cliff: Optuna rose to
`4.987 s`, initial/final exact passes to `0.745/1.011 s`, and complete wall to
`9.02 s`. The branch was removed. A control-flow optimization accepted by CUDA
must not be carried into this register-limited MPS kernel without measurement.

Narrowing the first 512 column radix exchange from a threadgroup barrier to a
SIMD-group barrier was exact but slower. That exchange depends only on groups
of eight lanes contained within one 32-lane Apple SIMD group; the later
exchange still retained its full barrier. Focused tests and the canonical hash
passed, but Optuna regressed from `2.285 s` to `2.421 s`, and the changed scalar
specialization took `0.652/0.610 s` for its initial/final passes. The full
barrier was restored. A narrower valid synchronization scope is not a faster
instruction/schedule for this shared-memory kernel on the measured M5.

A lossless candidate-delta row codec was rejected before kernel implementation.
On the first real Optuna pair, XORing the second candidate's float32 row bits
against the first left `80.67%` of components requiring the high byte; an
escape-coded 24-bit payload would reduce the two-candidate row traffic by only
`0.85%` before tags, packing, or decode work. Smaller exact samples across
early/middle/late Optuna and final refinement pairs produced estimated row-only
savings of `0.84%`, `8.25%`, `7.30%`, `9.75%`, and `10.07%`. Since only the
second candidate is encodable and full-field Optuna pairs dominate wall time,
even the converged case cannot approach the required 2x traffic cut. No codec,
escape buffer, environment switch, or production branch was added.

A complete q/-q lossless row codec was then implemented, measured, and removed.
The existing 32 KiB producer paired positive and negative q rows without wider
scratch, stored rows 0:256 as complex64, and represented each negative row by
per-thread ordered-float deltas in int8, int16, or exact complex64 fallback
tiers. The column kernel reconstructed every original float bit before the
unchanged radix-8 FFT. The codec matched the retained fused-pair loss bit for
bit in focused 1/2/4/8-candidate tests and preserved the canonical 21-record
hash, but tier selection, bit packing, and decode caused a severe compiler/
register regression: Optuna took `9.938 s` versus `2.285 s`, complete gate wall
rose to `15.02 s`, and peak footprint increased from about `18.51` to
`18.93 GB`. Exact variable-rate compression is not free bandwidth on this M5;
the 360-line producer/decoder and temporary test hook were deleted in full.

MLX allocator controls exposed a memory/speed trade rather than a faster path.
The default cache limit is the runtime memory limit (1.5 times Metal's
recommended working set). Capping it at 1 GiB reduced complete-gate peak
footprint from about `18.5` to `10.6 GB`, and an 8 GiB cap measured `13.6 GB`,
with identical hashes. Both forced row-buffer reclamation and regressed Optuna
from `2.285 s` to `3.714/3.690 s`, so no global cache policy was retained.

A fixed 300-plane output stride then let all ten sparse packs reuse one MLX
allocation without a cache cap. It also reached `10.64 GB` peak and retained
the exact hash, but the noncompact candidate stride hurt locality: Optuna took
`2.886 s`, initial/final scalar specializations took `0.710/0.705 s`, and the
gate wall was `6.41 s`. Compact per-pack candidate adjacency is worth more than
single-shape allocation reuse; the storage-stride prototype was removed.

A three-class 64-plane storage bucket is accepted as the narrower allocation
solution. The ten real sparse packs round to 192/256/320 planes, adding 352
unwritten planes in total (`14.3%`) instead of fixed-300's 536 (`21.8%`).
Column consumers still read only each original compact boundary, and all
focused references plus the canonical hash are exact. After one-time
compilation, two 20-trial gates measured `2.045/2.021 s` Optuna and
`4.55/4.49 s` complete wall versus the matched compact gate's `2.213/5.00 s`;
peak footprint fell from about `18.52` to `12.41 GB`. The preceding 32-plane
version used four allocation classes, added only 160 unwritten planes, and
measured `2.058/2.070 s` Optuna with `13.21-13.22 GB` peak. A 16-plane neighbor
used six allocation classes and reduced padding further, but its repeated gate
regressed Optuna to `2.099 s` and raised peak footprint to `15.29 GB`; it was
removed. On this allocator, fewer reusable shapes outweigh modest unwritten
tail capacity. The first new-source gate can still pay pipeline compilation,
so provenance must distinguish first-ever specialization cost from the
sustained scientific path.

The next coarser 128-plane bucket reduced allocation classes from three to two
and lowered peak footprint again from `12.41` to `11.87 GB`, but its two warmed
Optuna gates measured `2.055/2.063 s` versus 64-plane's `2.045/2.021 s`.
The extra 256/384-plane tail traffic loses about one percent on the speed-first
contract, so 128 was removed and 64 remains the measured local optimum.

A nonuniform two-class follow-up is accepted. It keeps 256- and 320-plane
allocations and rounds only the final 176-plane pack up to 256, adding 64 more
unwritten planes than 64-plane rounding while eliminating its 192-plane cache
class. Three exact gates measured `2.036/2.049/2.020 s` Optuna, in the retained
performance band, while peak fell from about `12.41` to `11.60-11.61 GB`.
The first full 200-trial plus Nelder-Mead signoff measured `24.644 s` fit /
`25.80 s` wall, including `20.020 s` Optuna and `3.340 s` refinement, with all
231 records, 30 physical refinement calls, fitted values, loss, and canonical
`f9748622...` hash exact. A bounded helper owns this private allocation policy;
it is not a caller-facing tuning option.
The clean committed signoff at GPU `f1773be` measured `24.661 s` fit /
`25.83 s` wall: `0.906 s` preparation, `0.186 s` initial loss, `20.045 s`
Optuna, `3.325 s` refinement, `0.050 s` final object, and `0.143 s` final
phase/loss. It reproduced the canonical 8,937-plane full-BF selection, all 231
records, 30 refinement evaluations, fitted values, and final loss exactly;
peak footprint was `11.607 GB`.

The real pack histogram then exposed a tighter aligned class: all ten packs
top out at 285 planes, so 288 replaces 320 for those packs while 320 remains a
bounded fallback for other full-BF geometries up to the 300-plane cap. Three
exact gates measured `2.013/2.004/2.038 s` Optuna and `3.324/3.310/3.368 s`
fit, with all 21 losses bit-for-bit identical. Peak footprint fell again to
`11.469-11.474 GB`. The policy therefore retains 256/288/320 as its small
private class set; Reference-512 exercises only 256 and 288.
Two clean full signoffs at GPU `3cfbb67` remained exact and measured
`24.784/25.345 s` fit (`25.93/26.48 s` wall), the slower second run tracking a
system-wide timing drift. An immediate short A/B in that state measured the
old 320 class at `2.0867 s` Optuna / `3.391 s` fit and the 288 class at
`2.0852 s` / `3.385 s`; it is a speed-neutral memory reduction, not a claimed
wall-time breakthrough.

A single 288-plane real-data class is the next accepted allocator checkpoint.
It adds 160 unwritten tail planes across the five smaller packs but eliminates
their separate 256-plane cache class; 320 remains only as a bounded fallback
for other geometries. After one specialization warm-up, exact gates measured
`2.012/2.008 s` Optuna and peak fell from about `11.47` to `11.239-11.240 GB`.
The first full 200-trial plus Nelder-Mead run measured `24.785 s` fit /
`25.94 s` wall: `20.144 s` Optuna, `3.328 s` refinement, and `0.915 s`
preparation. All 231 records, 30 physical calls, fitted values, and final loss
remained bit-for-bit identical.
The clean committed signoff at GPU `c5d5316` measured `24.803 s` fit /
`25.98 s` wall: `20.183 s` Optuna, `3.317 s` refinement, `0.919 s`
preparation, `0.051 s` final object, and `0.143 s` final phase/loss. It retained
the complete exact trace and lowered observed peak again to `11.142 GB`.
The exact 285-plane real maximum was tested and removed. Its warmed Optuna gate
measured `2.021 s` versus 288's `2.008/2.012 s`, while observed peak was
effectively identical (`11.237` versus `11.239 GB`). Allocator granularity
erases the nominal three-plane saving; the 32-plane-aligned 288 class remains.

A 32-thread paired-row FFT was tested by assigning each thread two of the
unchanged radix-8 work items. The focused kernel cases and all 21 canonical
real-data gate records remained bit-for-bit exact, but the warmed Optuna stage
regressed to `2.114 s` versus the retained 64-thread kernel's
`2.008/2.012 s`. Fewer resident threads did not offset the loss of latency
hiding, so the prototype was removed completely.

A BF-major candidate-interleaved row handoff placed the two fused candidate
planes adjacently for each BF instead of using candidate-major storage. The
focused 512 cases and canonical 21-record trace were exact, but two warmed
Optuna stages measured `2.034/2.044 s`, above the retained `2.008/2.012 s`.
The producer's shorter address jump slightly worsens the candidate-separated
column walk, so the private layout change was removed.

A bijective XOR swizzle of the row FFT's 512-entry threadgroup scratch mapped
logical indices across the 32 shared-memory banks without changing any stored
complex64 value or radix operation. Focused cases and the complete canonical
gate trace were exact, but warmed Optuna measured `2.024/2.029 s` and scalar
passes also moved slightly slower than the retained path. The added address
arithmetic costs more than any bank-conflict reduction, so the swizzle was
removed.

Suppressing the final threadgroup write-back for tiled row output was exact
but also removed. The tiled handoff already writes the final radix values from
registers, so the source-level shared stores appeared dead; two warmed gates
nonetheless measured `2.028/2.032 s` Optuna versus the retained
`2.008/2.012 s`. Metal was already eliminating or scheduling those stores
better, and the extra generator branch had no production value.

Embedding the exact 512 complex64 twiddle bit patterns as a Metal constant
table removed the row kernel's twiddle-buffer reads without approximating any
value. The canonical trace stayed exact, but the warmed Optuna stage measured
`2.029 s`. The 4 KiB input table is already effectively cache-resident, while
the embedded table enlarges the shader, so the header generator and constant
path were removed.

The same exact-bit table is accepted for the radix-8 column consumer. Unlike
the row producer, each column threadgroup reuses the twiddles inside every BF
iteration, so Metal constant space removes repeated device-buffer addressing.
Three warm 20-trial gates measured `1.961/1.964/1.970 s` Optuna versus the
retained `2.008/2.012 s` band, and every canonical trial bit stayed unchanged.
Two complete 200-trial plus Nelder-Mead signoffs at GPU `e09eb2d` measured
`24.357/24.465 s` fit and `25.49/25.62 s` wall. Their Optuna stages were
`19.775/19.859 s`, refinement `3.286/3.314 s`, and preparation about
`0.899 s`. Both reproduced all 231 records, 30 physical calls, fitted values,
final loss, and the canonical full-trace hash exactly. The focused backend
suite passed (`65 passed, 7 skipped`), and peak footprint remained about
`11.24 GB`.

Removing the column kernel's now-unread twiddle argument was exact but slower.
Two warm gates measured `1.999/1.992 s` Optuna versus the accepted constant
table's `1.961/1.964/1.970 s`. The unused binding still changes Metal's
generated function signature and favorable schedule, so it remains deliberately
present; the wrapper-cleanup prototype was removed.

Replacing the raw `uint2` table plus `as_type<float2>` with exact hexadecimal
`float2` constants was also exact but slower (`2.043 s` warm Optuna). Metal's
raw-bit constant representation is preferable despite the visible bitcast, so
the typed-table prototype was removed.

Removing the redundant `& 511` from the constant-table index kept every live
index and canonical loss exact, but warmed Optuna regressed to
`1.980/1.990 s`. The explicit power-of-two mask enables a better Metal address
schedule and remains in the accepted lookup.

Pack-aware column dispatch is the next accepted checkpoint. When adjacent
original 512-logical-BF boundaries already share one row-IFFT allocation, one
column grid now evaluates those boundaries together while writing a separate
phase image and squared-phase partial for each. The outer loop adds those
partials in the original boundary order, so this removes dispatches without
merging a BF reduction or changing float32 association. Focused synthetic
batch-2/4/8 checks exercise this path directly.

Two warmed 20-trial gates measured `1.927/1.900 s` Optuna. Two complete
200-trial plus Nelder-Mead signoffs at GPU `74d19d5` measured
`23.066/23.058 s` fit and `24.15/24.16 s` wall: Optuna was
`18.681/18.686 s`, refinement `3.134/3.103 s`, preparation
`0.845/0.873 s`, and final phase/loss `0.141/0.138 s`. Both reproduced all
231 records, 30 physical calls, fitted values, final loss, and the canonical
full-trace hash exactly. Peak footprint fell from the preceding `11.24 GB`
band to `10.97/10.96 GB`, and the focused backend suite passed
(`65 passed, 7 skipped`). This is about `1.3 s` faster than the constant-table
checkpoint and `1.74 s` faster than the prior clean one-288-class fit, but it
still does not meet the under-20-second wall target.

A later clean-tree signoff after all rejected follow-ups reproduced the same
231-record hash `cb2b35ca...`, 30 physical refinement calls, fitted values,
and loss exactly. Under active Screen Sharing/system load it measured
`24.205 s` fit / `25.33 s` wall: `0.869 s` preparation, `19.667 s` Optuna,
`3.271 s` refinement, `0.054 s` final object, and `0.140 s` final phase/loss.
This is regression evidence rather than a new speed record; the quiet retained
best remains `23.058 s` fit / `24.16 s` wall. The complete local suite passed
(`182 passed, 73 skipped`), and the changed MPS files are Ruff-clean.

Extending the same pack-aware dispatch to scalar initial/refinement calls was
exact but neutral. Two full runs measured refinement at `3.095/3.148 s` and
walls at `24.13/24.20 s`, versus the retained paired-only signoff's
`3.134/3.103 s` and `24.15/24.16 s`. The median refinement difference was
only about 3 ms, so batch one retains the simpler established boundary path.

Routing every paired pack through the pack-aware column API, including packs
that contain only one original reduction boundary, was also exact but slower.
After specialization warm-up, three candidate gates measured
`1.878/1.884/1.888 s` Optuna; an immediate three-run restoration of the
multi-boundary-only selector measured `1.871/1.872/1.879 s`. All six runs
reproduced the canonical 21-record trace bit-for-bit. The nominally uniform
Python route does not reduce Metal work and adds a small wrapper cost, so the
one-line selector change was removed.

Raising the sparse row-pack cap from 300 to 320 planes was exact but did not
remove useful work. Three gates measured `1.883/1.887/1.908 s` Optuna versus
an immediately preceding retained three-run band of `1.871/1.872/1.879 s`;
all canonical records remained bit-for-bit identical. The real greedy pack
boundaries do not gain a profitable merge within those extra 20 planes, so
production retains the lower 300-plane bound and its smaller worst-case
allocation.

Larger 448- and 576-plane packs were re-evaluated only after pack-aware column
dispatch could emit each original 512-logical-BF partial separately. Both
retained the exact canonical trace, but neither improved sustained pair work:
their warm Optuna gates measured `1.887 s` and `1.886 s`, respectively, in the
same `1.87-1.89 s` band as the 300-plane path. The 448 class raised peak to
`11.54 GB`; the 576 class raised it to `15.84 GB` and pushed the final scalar
phase pass above `1.1 s` under allocator pressure. Dispatch consolidation
still cannot repay the wider live row intermediate, so both prototypes were
removed.

Reducing every packed range's squared-phase tile through one range-preserving
MLX graph was bit-for-bit exact but not faster. Three warm Optuna gates were
`1.883/1.887/1.894 s`, slightly above the immediately measured retained band.
The shared reduction expression saves Python graph nodes but does not remove
the dominant row/column traffic, so the established per-boundary reduction was
restored.

Submitting all ten scalar row packs through one bounded graph instead of the
retained two waves of five was also rejected after complete signoff. Two exact
200-trial plus Nelder-Mead runs reproduced the full trace and measured
`3.115/3.128 s` refinement, indistinguishable from the retained
`3.103/3.134 s` signoffs. A lower-contention one-trial CLI A/B found only a
roughly 6 ms median gain for the initial scalar pass and 2 ms for final phase,
while the independent trial itself stayed at `106-107 ms`. Peak did not rise,
but the complete refinement did not improve, so production retains the
simpler five-pack synchronization bound.

Folding the prior phase image into each single-boundary column shader removed
the follow-up elementwise phase addition for eight of the ten real sparse
packs. All four canonical gates remained bit-for-bit exact, but three warmed
Optuna stages measured `1.880/1.881/1.885 s`, not better than the retained
`1.871/1.872/1.879 s` matched band. The extra shader input and registers trade
one tiny image dispatch for no sustained gain against the row-intermediate
floor, so the added signature and branch were removed completely.

Advising macOS to discard the 2.34 GB u8 companion's mapped file pages after
evidence preparation was exact but had no measurable effect. Three gates ran
at `1.881/1.882/1.895 s` Optuna and `11.239-11.241 GB` peak, the same retained
bands. The active optimizer footprint is MLX evidence and row scratch; the
source mapping is already nonresident enough after preparation. The platform
hook and extra public source method were removed rather than exposing a no-op
policy.

Narrowing only the paired row FFT's first radix exchange to a SIMDgroup barrier
was valid and bit-for-bit exact, because each eight-lane exchange stays inside
one Apple SIMDgroup. It was nevertheless slower: after first-use compilation,
two gates measured `1.986/1.994 s` Optuna versus the retained `1.88 s` band.
As with the earlier column result, the narrower synchronization instruction
produces a worse Metal schedule on this M5, so the full threadgroup barrier was
restored.

Hoisting the row shader's immutable seven-float geometry block from each sparse
pack to once per exact evaluation was exact but neutral. Three gates measured
`1.995/1.998/1.999 s` Optuna versus an immediate restored baseline of
`1.993/2.000/2.013 s`; scalar passes also overlapped. MLX's tiny-array handling
is not material beside the Metal kernels, so the override parameter and extra
graph state were removed.

Caching the BF-active byte mask in prepared state instead of rebuilding it from
`p(k)` on every exact call was also exact but neutral. Three Optuna gates were
`1.981/1.997/1.998 s`, overlapping the immediate `1.993-2.013 s` band, while
preparation rose from roughly `0.85` to `0.90-0.93 s`. The tiny mask dispatch
is hidden beside the row pipeline; retaining extra prepared state would only
move cost into setup, so the cache field was removed.

Packing only the two candidates' already-corrected row FFT values into
`float4` lanes was rejected on the first real gate. Focused synthetic cases
passed, but canonical trials 2, 3, and 8 moved by one float32 ULP and Optuna
regressed to `2.600 s`. Even with scalar geometry, `sincos`, gamma, and output
stores, Metal reschedules vector radix arithmetic differently from two scalar
`float2` FFTs. The complete 100-line vector path was deleted under the exact
trace contract.

Reducing each column's 64 squared-phase thread partials inside the existing
Metal threadgroup cut that output by 64x but was rejected on the first real
gate. The canonical trace moved by one ULP on trials 2, 4, and 19, and Optuna
regressed to `2.322 s`; six additional shared-memory barriers outweighed the
smaller output. The MLX reduction tree and its exact float32 association were
restored in full.

Adding a second inert column-kernel buffer binding was exact but did not extend
the compiler-scheduling benefit of the deliberately retained twiddle binding.
Two warm gates measured `1.990/2.006 s` Optuna, overlapping the current loaded
baseline, while the new signature paid another first-use compile. Only the
single previously accepted unused binding remains; the extra schedule hint was
removed.

Disabling MLX's row-contiguous input guarantee on the column kernel was safe
for the current tiled row output and exact mask/twiddle slices, but it was not
faster. Three gates measured `1.991/1.991/1.992 s` Optuna; immediately restoring
the default produced `1.966/1.983/1.986 s`. The wrapper guarantee is retained;
removing a no-copy check does not improve the device pipeline and slightly
changes its scheduling.

Materializing all twelve paired phase partials pack-by-pack, then applying
their additions through one final ordered MLX graph, was exact and raised peak
by less than measurement noise. It did not reduce wall time: three Optuna gates
measured `1.979/1.987/1.993 s`, versus the immediately matched retained median
of `1.983 s`. MLX already schedules the small elementwise additions efficiently
beside each column dispatch, so the deferred list and extra ownership were
removed.

A SIMDgroup asynchronous-copy prototype attempted to stage each contiguous
32 KiB, eight-column row-intermediate tile into threadgroup memory before the
unchanged radix-8 column FFT. It was rejected at the compile gate: the Metal
environment exposed by MLX 0.32 does not declare `simdgroup_event`, so neither
its `async_copy` operation nor its wait primitive is available to custom Metal
kernels. The complete source experiment was removed before timing. Do not retry
this topology through `mx.fast.metal_kernel` unless MLX first exposes the Metal
SIMDgroup event API; scalar cooperative copies would add a second load/store
pass and are a different experiment.

Asking clang to unroll the column kernel's ordered BF loop by two preserved
the complete canonical 21-record hash. After the first-use specialization,
two gates measured `1.974/1.979 s` Optuna, identical to the immediately
preceding retained `1.979 s`; the cold specialization also raised the scalar
initial/final passes to `0.660/0.616 s`. The pragma was removed because Metal
already schedules the loop effectively and no warm workflow gain separated
from noise.

Marking the compact source's zero-active-BF column branch `[[unlikely]]` was
also exact but neutral. Warm Optuna gates measured `1.975/1.976 s`, again
indistinguishable from the retained `1.979 s`, after a cold specialization
with slower scalar passes. The hint was removed: it did not reduce the
row-intermediate traffic or establish a device-side gain.

Submitting independent sparse packs through two persistent worker-local MPS
streams preserved every canonical trial bit and consumed the returned GPU
partials in the original boundary order. It did not overlap useful work:
three Optuna gates measured `2.001/2.006/2.003 s` versus the retained
`1.979 s`, while peak footprint rose from about `10.9` to `11.6 GB`. The row
producer and column consumer contend for the same unified-memory bandwidth,
so the stream state, executor, worker, and selector were removed completely.

A local `restrict`-qualified view of the column kernel's complex64 row input
compiled and preserved the canonical trace, but warm gates at `1.966/1.979 s`
did not separate from the retained `1.979 s` control or its established timing
band. The generated Metal signature already gives clang sufficient alias
information; the extra pointer and changed source specialization were removed.

Reinterpreting the column input as `float2` and issuing explicit 64-bit loads
also preserved every canonical trial bit. Warm Optuna measured
`1.978/1.981 s`, no better than the retained complex64 member accesses. Metal
already coalesces those real/imag reads, so the reinterpretation and less
descriptive load expressions were removed.

The clean retained-tree full signoff after these compiler and stream probes
used all 8,937 logical / 2,464 active full-BF terms, 200 seed-42 Optuna trials,
and unchanged Nelder-Mead. It measured `24.065 s` fit / `25.26 s` process wall:
`0.905 s` preparation, `0.185 s` initial exact loss, `19.547 s` Optuna,
`3.226 s` refinement, `0.051 s` final object, and `0.144 s` final phase/loss.
All 231 records matched the reference file byte-for-byte after canonical JSON
formatting (SHA-256 `cb2b35cacaabd01ac473df91875383dc5f633dbb53bf5ebb27627c80fd0168c1`),
with 30 physical refinement calls, the same fitted aberrations, and final loss
`0.04469207674264908`. This loaded-system validation does not replace the
quiet retained best of `23.058 s` fit / `24.16 s` complete CLI wall.

Explicit column-tile prefetches were exact but not retained. One lane per
cache line requested the next BF's complete contiguous 32 KiB tile while the
current tile ran its FFT and `atan2`; warm Optuna measured `1.976/1.973 s`,
too close to the `1.979 s` control to establish a gain. Moving the request two
BF planes ahead regressed both warm gates to `1.986 s`. Metal's normal loads
already saturate this streaming path, so all prefetch instructions and address
logic were removed.

The barrier between consecutive BF transforms was safely narrowed from a
threadgroup-memory fence to an execution-only `mem_none` barrier: it still
prevented the next BF from overwriting shared FFT values before every thread
finished reading them, and focused plus canonical parity stayed exact. It was
nevertheless slower, with warm Optuna at `2.072/2.023 s` and slower scalar
passes. The full memory-fence form was restored because it produces a better
Metal schedule on this M5.

The Metal 4.1 acquire-release overload for `threadgroup_barrier` was also
checked as a stricter alternative to `mem_none`, but MLX 0.32's custom-kernel
language surface does not declare `memory_order_acq_rel`. The focused compile
gate failed before timing, and all three overload calls were removed. This can
only be revisited after MLX exposes the Metal 4.1 memory-order API.

The installed `mlx` and `mlx-metal` 0.32.0 packages were verified as the latest
published releases. Official MLX main at `973e27f8` still accepts only
`math_mode` as a custom-kernel compile option and selects Metal language 4.0 on
macOS 26; it therefore cannot opt this process into the Metal 4.1 barrier API.
There is no released or main-branch MLX upgrade path for this optimization yet.

A five-run retained-tree thermal soak then exercised the complete canonical
200-trial plus Nelder-Mead CLI five times, not a reduced kernel harness. Fit
elapsed values were `24.289/24.528/24.406/24.648/25.221 s` (median
`24.528 s`); Optuna was `19.748/19.889/19.757/19.866/20.401 s`, and
preparation stayed `0.882-0.936 s`. Every run reproduced the same 231-record
`cb2b35ca...` hash, 30 physical refinement calls, fitted aberrations, and loss
`0.04469207674264908`. The final run's mild slowdown tracks sustained thermal
or active-desktop load; there was no late allocator or scientific failure.

Hoisting both candidates' complex probe values outside the row producer's
eight-column loop preserved the canonical trace but increased their live range.
Two warm Optuna gates measured `2.000/1.998 s`, slower than the retained
`1.965 s` gate. The extra cached value crosses a small register/scheduling cost;
the generator branch and cached declarations were removed completely.

Reusing only the already-loaded first candidate probe avoided the second live
value but was also rejected. Focused synthetic parity passed, while the full
canonical gate moved trials 2 and 8 by one float32 ULP and warm Optuna measured
`2.001/1.970 s`, no stable gain. The conditional select changed Metal's gamma
schedule; the original uniform probe expression was restored.

Removing the paired row producer's zero-probe early return after compact
preparation was exact on all canonical trials, because compact storage contains
only active probe planes. It was still slower: warm Optuna measured
`1.999/1.988 s` versus the retained `1.965 s` gate, with much slower cold scalar
specializations. The uniform guard remains as a beneficial Metal schedule cue.

Marking that same retained row guard `[[unlikely]]` preserved every canonical
trial bit but regressed warm Optuna to `1.998/2.000 s`. The compiler hint did
not reduce work and perturbed the favorable branch schedule, so it was removed.

One unified 288-plane paired-row shader was tested in place of the ten
actual-pack-length specializations. It read the already-computed full probe
vector through a runtime pack offset and launched only the real BF count. The
first new-source gate improved to `5.08 s` wall, but warm Optuna measured
`2.020/2.064 s`. The required full gate reproduced all 231 records, 30
physical refinement calls, fitted values, and final loss exactly, yet measured
`24.838 s` fit / `26.02 s` wall versus the retained clean signoff's
`24.803/25.98 s`. A one-time compilation saving did not justify a runtime
offset path with no sustained scientific-workflow gain, so it was removed.

The 400-plane pack cap was retested after allocation bucketing because it can
reduce the real producer count from ten to nine while retaining each logical
column-reduction boundary. It remained exact, but the new 448-plane storage
class raised peak footprint from `12.41` to `13.95 GB`; the warmed gate took
`2.062 s` Optuna and `0.165 s` final phase versus pack-300's `2.021-2.045 s`
and `0.154-0.158 s`. The cap was restored to 300.
An exact 400-plane class removed the old 48-plane padding and was still
rejected: its warmed gate measured `2.085 s` Optuna and `0.190 s` final phase,
while peak remained high at `13.609 GB` versus the current `11.47 GB` band.
The ninth row producer does not repay the larger terminal allocation.

The clean committed 64-plane 200-trial plus Nelder-Mead signoff at GPU commit
`9311ea2` measured **`25.99 s` complete wall** and `24.817 s` fit: `20.061 s`
Optuna, `3.427 s` refinement, `0.913 s` preparation, `0.051 s` final object,
and `0.143 s` final phase/loss. Peak footprint was `12.407 GB`. The preceding
promotion gate measured `25.94 s` wall / `24.777 s` fit. Both reproduced all
231 records, 30 physical refinement evaluations, canonical aberrations, final
loss `0.04469207674264908`, and full trial hash
`f974862233ca80464d9bc1dae2474818224a7f9afed9ebbe3456f0a843da1b04`
exactly. The preceding clean 32-plane signoff was `26.04 s` wall / `24.867 s`
fit at GPU commit `5366489`; the preceding exact-DC best was `26.45 s` wall /
`25.220 s` fit with about `18.40 GB` peak. Bucketing is a small fit improvement
plus a material working-set reduction, not the still-required under-20
topology breakthrough.

Setting MLX's process-local wired-memory limit to 16 GiB was exact but neutral.
Two wired gates measured `2.367/2.216 s` Optuna; the immediately matched
unwired gate measured `2.213 s`. Peak footprint remained about `18.5 GB` and
the better complete walls tracked preparation variance, not the fit. The reference system
had no swap activity in any run, so wiring already-resident unified buffers
does not raise the sustained row-intermediate bandwidth. The setting was
removed, and the system-wide `iogpu.wired_limit_mb` was never changed.

Compact preparation stores only probe-aperture-active BF planes, but removing
the therefore-uniform `active_bf` guard from the column kernel was not bitwise
neutral. The warmed gate was performance-neutral (`2.045 s` Optuna), while
trials 3 and 19 moved by one float32 ULP because the changed source let Metal
reschedule surrounding arithmetic. The specialized signature, mask elision,
and source branch were removed; an analytically redundant branch is still part
of this compiler's accepted exact schedule.

A conservative row-kernel support precheck was exact but neutral. It tested
both shifted squared radii against `semiangle + max(detector_sampling)` before
the existing geometry, safely leaving all potentially nonzero aperture pixels
on their unchanged arithmetic path. Sampling the canonical compact geometry
showed that only about `24%` of q pixels could skip. The warmed gate measured
`2.040 s` Optuna versus the retained `2.021-2.045 s` range: duplicated distance
work cancels the saved square roots and aperture arithmetic. The precheck was
removed rather than burdening the hot kernel with an unproductive branch.

A temporary synchronized profile of the accepted 64-plane exact pair measured
`0.156/0.158 s` in correction plus row IFFT and `0.096/0.098 s` in column IFFT,
`atan2`, and local loss output. Synchronization inflates the pair above its
normal overlapped `~0.20 s`, but consistently assigns about 61% of staged time
to the producer and 39% to the consumer. The environment hook and forced
barriers were removed after measurement.

Scheduling `q` and `-q` rows inside the same four-row producer group preserved
every canonical trial bit while leaving each row's correction and FFT
arithmetic untouched. It nevertheless regressed warmed Optuna to `2.199 s`:
any Hermitian evidence-cache reuse was outweighed by scattering the four row
stores across distant locations in the tiled intermediate. The row permutation
was removed; q-pair arithmetic needs a different coalesced handoff first.

Depth-two row-pack submission was retested after 64-plane bucketing. It kept
the full canonical hash and two complete 200-trial plus Nelder-Mead runs took
`24.746/24.695 s` fit, versus depth one's `24.777/24.817 s`. The `0.076 s`
median gain (`0.3%`) raised peak footprint from `12.41` to `14.82 GB`; short
pair gates were also slightly slower. That is not a useful MacBook trade or an
under-20 topology, so one-pack synchronization remains.

Depth two was the first accepted scalar-only pipeline. Initial-loss gates improved
from `0.219-0.220 s` to `0.206-0.207 s`, while paired Optuna keeps one-pack
synchronization and its original working set. Two full 200-trial plus
Nelder-Mead runs measured `24.667/24.680 s` fit and `25.84/25.85 s` wall;
refinement took `3.360/3.369 s`. Both reproduced all 231 records, 30 physical
refinement calls, fitted values, final loss, and canonical `f9748622...` hash
exactly. Peak footprint remained `12.407 GB`. This saves about `0.12 s` versus
the preceding `24.777/24.817 s` signoffs without widening paired state.
The clean committed signoff at GPU `b67cd06` measured `24.739 s` fit /
`25.90 s` wall with the same exact trace and footprint.

The scalar-only depth sweep then advanced through three and four to the natural
two-wave depth of five for the ten real row packs. Depth-five initial-loss gates
measured `0.1857/0.1859 s`, versus depth two's `0.2057/0.2072 s` and the old
depth-one `0.219-0.220 s`. Paired Optuna remains synchronized at every pack.
Two depth-five full runs measured `24.682/24.624 s` fit and `25.86/25.83 s`
wall, with refinement at `3.354/3.332 s`; both retained the exact 231 records,
30 physical refinement calls, final result, and `f9748622...` hash. Process
peak remained about `12.41 GB`. Depth five replaces depth two without adding
a public policy or widening the dominant paired working set.
The clean current-tree signoff after rejected follow-ups measured `24.703 s`
fit / `25.87 s` wall, including `20.062 s` Optuna, `3.356 s` refinement, and
`0.901 s` preparation, with the same exact trace and `12.413 GB` peak.

Applying the three 64-plane storage buckets to scalar calls as well was exact
but rejected. The warmed scalar pass improved only from about `0.186` to
`0.180 s`, while keeping separate scalar bucket caches alongside paired caches
raised process peak from `12.41` to `16.15 GB` and regressed complete-gate
latency. Scalar outputs remain compact; storage bucketing stays pair-only.

Stage-retiring those scalar buckets was tested after the tighter pair classes
and removed. Explicitly clearing the scalar allocator cache before Optuna and
the pair cache before refinement kept all 21 losses exact and cut gate peak to
`10.641 GB`, but the bucketed scalar specialization plus retirement raised the
gate fit to `3.993 s`; initial loss alone took `0.715 s`. Cache teardown and
fresh page allocation cost much more than the few warmed scalar milliseconds
saved, so production retains compact scalar buffers and no new stage barriers.

A scalar-only 400-plane pack cap was also exact but slower. It reduced the
producer count from ten to nine, yet warmed initial loss took `0.189 s` versus
pack-300's `0.186 s`, first-use latency worsened, and peak rose by about
`0.46 GB`. Both scalar and paired calls retain the 300-plane pack cap.

The q/-q row schedule was then given a fully coalesced handoff: the producer
stored partner rows adjacently, and the column kernel applied the inverse row
map while loading the tiled intermediate. Every canonical trial bit remained
exact, but warmed Optuna regressed to `2.119 s`. Per-load inverse mapping cost
more than Hermitian evidence-cache reuse saved. The alternate layout, flag,
and address expressions were removed in full.

Transposing the paired producer's thread coordinates from 64 FFT lanes by four
rows to four rows by 64 lanes made each SIMDgroup's tiled stores contiguous as
a `4x8` block without changing any output address or arithmetic. The canonical
hash stayed exact, but warmed Optuna regressed to `2.139 s`; FFT-lane scheduling
cost more than store coalescing saved. The original `64x4` coordinates remain.

Replacing the column FFT's first 8-by-8 threadgroup-memory transpose with an
in-register SIMD shuffle transpose was rejected and removed. It preserved the
radix expressions and produced a scientifically equivalent result, but trial 2
moved by one float32 ULP (`0.0465815365` to `0.0465815403`), violating the exact
trace contract. Its cold gate also compiled to `2.406 s` Optuna with roughly
`0.65/0.61 s` scalar passes. The accepted shared-memory transpose and barrier
remain part of the exact Metal schedule.

Skipping candidate `sincos` when either already-computed shifted aperture was
exactly zero was also removed. The added branches changed trials 3 and 8 by one
float32 ULP, regressed Optuna to `2.509 s`, and pushed cold scalar passes to
`0.617/0.688 s`. Multiplication by the exact zero aperture remains in the
branch-free correction schedule; on this compiler, apparently dead transcendental
work is cheaper and more reproducible than the extra control flow.

The 128/256 shared-candidate column topology was then ported to the tiled 512
pair as a four-column/two-candidate, 32 KiB threadgroup. All five focused 512
row/batch reference cases passed, but the canonical real-data gate exposed the
compiler-level difference: trials 2, 3, and 8 moved by one float32 ULP, and
Optuna regressed to `2.947 s` from the retained `2.313-2.386 s` range. Sharing
barriers across candidate column FFTs increased register pressure and did not
preserve the production kernel's exact scheduling. The complete paired column
kernel and selector were removed; 512 keeps candidate-separated column groups.

An exact integer BF-column detector-sum kernel was also rejected. It replaced
the float32-safe scan slices with 256-plane dispatches and 65,536-sample
uint32 segments; synthetic u8 and worst-case u16 sums matched uint64 references
exactly, including totals above 32 bits. The canonical source-load stages were
`1.90 s` and `1.76 s`, however, overlapping the retained `1.72-1.99 s` range.
Reading/page-faulting the 2.34 GB companion is the load floor on this path, not
the number of reduction dispatches. The integer kernel, tests, and selector
were removed instead of keeping a neutral second reducer.

The 396-plane row pack was retested after BF-major evidence reduced memory
pressure. It retained the canonical hash but measured `2.399/2.341 s` Optuna,
not better than the retained `2.313-2.386 s` range, and did not lower peak
footprint. Production therefore remains at the simpler 300-plane cap.

Scheduling Hermitian mirror-row pairs adjacently was exact but slower. The row
grid visited logical rows as `0, 1, 511, 2, 510, ..., 256` and wrote each FFT
back to its original tiled location, attempting to reuse `G_qk(r)` for
`G_qk(512-r)` while cached. The warmed Optuna gate measured `2.481 s`, above
the retained range, and the first new specialization took `5.273 s` Optuna.
Scattered row stores and mapping/compiler cost outweighed mirror reuse, so the
variant was removed.

The 16-column row-intermediate tile was also retested after the evidence-layout
change. Its two warmed Optuna stages were `2.318/2.325 s`, effectively the same
median as the retained eight-column samples, while its first specialization
was slower at `2.875 s`. With no stable gain, production keeps the smaller
eight-column tile.

The separately evaluated starting point cannot profitably be folded into the
first two Optuna candidates. On one prepared real dataset, interleaved warm
scalar, pair, and triple exact calls measured `0.125-0.133 s`, `0.214-0.219 s`,
and `0.370-0.378 s`, respectively. The existing scalar plus pair costs about
`0.34 s`; one triple costs about `0.37 s` even before adding special-case
optimizer orchestration. The candidate losses were identical across call
shapes. This exact-sequence-preserving grouping was rejected as slower.

### MPS wider exact-simplex checkpoint, 2026-07-26

The exact Nelder-Mead initializer previously used 5% of each current parameter
as its axis step. After TPE landed near `C12=10.20` and `phi12=0.387`, that
made the C12/phi axes only `0.51 nm` and `0.019 rad`; the simplex spent exact
calls expanding before it could resolve the lower basin. The exact-objective
policy now retains the 5% scale but floors C12 at `2.00 nm` and phi12 at
`0.04 rad`, with axis steps rounded to two decimals. Sparse/reference
refinement is unchanged.

Two complete seed-42 runs measured `31.931 s` and `31.994 s` end to end.
Nelder-Mead fell from 77 calls / `9.421 s` to 69 calls / `8.506-8.539 s`.
Unlike a shortcut, this found a **lower** exact loss: `0.11136186867952347`
versus v8's `0.11136215180158615`. Both runs returned identical parameters
(`C10=73.64155836`, `C12=14.82957136`, `phi12=0.48431618`) and the final exact
reconstruction independently returned the same improved loss. Final phase and
amplitude statistics were identical across the two runs. The focused suite
passed (`37 passed, 3 skipped`).

The convergence trace showed that this exact minimum first appeared at call
51; the remaining calls only contracted the simplex. Raising the exact-only
function-spread tolerance from `1e-6` to `3e-6` stops at 55 calls while
returning bit-identical parameters, exact loss, phase statistics, and amplitude
statistics. Two full runs measured `30.217 s` and `30.729 s` total (the latter
included a noisier `1.656 s` setup), with refinement at `6.747-6.758 s`.
The boundary is measured rather than arbitrary: `fatol=4e-6` and `1e-5` both
stopped at 42 calls and worsened the loss to `0.1113685369`, so they were
rejected. The retained `3e-6` boundary removes only the verified flat tail.

Follow-up optimizer probes were rejected against the improved loss, not merely
the old timing. Nearby simplex steps could stop in 55-64 calls, but returned
losses from `0.1113619208` to `0.1113632321`, all above the retained
`0.1113618687`. Sequential batch-1 TPE reduced refinement to 62 calls but raised
Optuna to `24.720 s`, total to `33.668 s`, and loss to `0.1113625318`.
Materializing and retaining the mean phase from every refinement call, intended
to reuse the winning phase and skip the final `~0.24 s` pass, caused the 24 GB
process to be killed before report generation in two full runs. MLX/NumPy phase
views stay pinned too long for that cache policy, so the safe final exact phase
pass remains. A second version returned the mean phase as a lazy MLX expression
and copied only improving candidates; it was also killed twice before report
generation because the expression retained its large reduction graph. Both
ownership strategies were reverted.

The inactive-write optimization also exposed a phase-only corner: the legacy
512 radix-4 sum kernel had no active-BF mask and therefore read the deliberately
unwritten inactive rows, producing an all-NaN phase when `compute_loss=False`.
The 512 phase-only route now uses the same masked radix-8 kernel and reduction
order as exact phase+loss, discarding only the unused scalar square sum. On the
real best fit, phase-only and phase+loss arrays are bit-identical
(`changed=0`, `max_abs=0`) and warm passes are both about `0.224 s`; the focused
MPS suite passes (`24 passed, 2 skipped`).

A true sum-only specialization of that masked radix-8 kernel removed the
square accumulation and square-reduction tile. It did not improve warmed
phase-only timing (both variants remained about `0.223 s`) and changed 346
phase pixels, with maximum absolute change `8.94e-8`, because the altered
Metal expression schedule changed floating-point rounding. It was reverted:
there is no measured speed benefit to justify losing bit-identical output.

Larger exact-loss BF chunks also failed at full-workflow scale. With the same
200 seed-42 candidates and 55-call refinement, `chunk_bf=640` took `36.670 s`
total (`21.972 s` Optuna, `12.232 s` refinement) and changed the final scalar
loss by `1.49e-8` through a different chunk-reduction order. `chunk_bf=1024`
returned the retained optimum and loss but took `40.553 s` total (`24.582 s`
Optuna, `13.668 s` refinement) under unified-memory pressure. Both were
rejected; the validated `512`-BF workflow remains faster and bit-identical.

A final-tree footprint sweep revisited smaller pair chunks after the radix-8,
inactive-write, and cache changes. Alternating warm calls put `384` BF near
`0.221 s`, versus roughly `0.226-0.230 s` for `512`, while reducing peak paired
row scratch from about 2.15 to 1.61 GB. However, both `384` and `448` changed
one benchmark loss by about `1e-8` because they move the float32 chunk
accumulation boundaries. They were rejected under the bit-identical optimizer
contract; a smaller allocation is not a no-precision-loss win when it changes
the objective samples.

Scalar row-IFFT occupancy was then reduced from four independent rows per
threadgroup to two. This halves scalar row-kernel threadgroup scratch from
`16 KiB` to `8 KiB` without changing the paired Optuna topology. Matched
40-call runs moved the exact single-candidate median from roughly `0.128 s` to
`0.125 s`; every call retained loss `0.11143244`. An 80-call two-row soak
measured min/median/mean/p95/max
`0.12123/0.12519/0.12542/0.12842/0.13053 s`. Two complete 200-candidate plus
Nelder-Mead pressure-state gates retained the exact optimum and scalar loss,
but took `32.582 s` and `30.718 s` end to end because the unchanged pair stage
ran above its established quiet-machine band. They validate scientific parity,
not a new end-to-end signoff. More importantly, the focused reference suite
found that this topology changes the batch-one result for very small BF chunks
(`num_bf=4`, `chunk_bf=2`): all three batch-versus-single fixtures failed.
The two-row change was therefore reverted despite its real-data benchmark win.

The `4 KiB` one-row scalar lower bound was also tested. Its 80-call
min/median/mean/p95/max was
`0.11943/0.12501/0.12526/0.13000/0.13238 s`: only `0.18 ms` lower at the median
than two rows and worse at p95. It was rejected as noise-scale. The retained
four-row scalar schedule remains the smallest reference-safe version from this
sweep. Neither smaller schedule reduces the dominant global row-IFFT
write/read traffic.

Inactive BF compaction was also constrained to each original 512-BF reduction
chunk so active ordering and chunk accumulation boundaries remained fixed.
The six row allocations fell from `[512, 512, 512, 512, 512, 266]` BF to
`[398, 469, 480, 475, 462, 175]`, and both paired losses were bit-identical.
However, retaining the packed `G_qk` evidence alongside the source increased
unified-memory pressure. In that alternating process, warmed compact calls
took about `0.460-0.492 s` versus `0.301-0.337 s` for the already-pressured
baseline. This was rejected: allocation shape alone does not lower the active
global traffic already enforced by inactive-write and inactive-read skips.

A stateful split-chunk column design was gated before implementation. It would
carry per-pixel phase and phase-square registers between 256-BF launches so
the original 512-BF float32 addition order could be retained while halving
peak row scratch. The prerequisite raw 512/256 reverse-order sweep measured
paired medians of `0.23482/0.22417 s` and then `0.22167/0.22816 s`; after
warm-up the smaller scratch was slower. Because this design retains all active
row bytes and adds state traffic plus launches, it was rejected before adding
a second complex column kernel.

The missing three-row paired occupancy point was measured with `24 KiB` of
threadgroup scratch. Forty warm calls gave min/median/mean/p95/max
`0.21971/0.22413/0.22437/0.22988/0.23160 s`; restoring the retained two-row,
`16 KiB` schedule gave
`0.21498/0.22120/0.22119/0.22550/0.23296 s`. Displayed pair losses were
identical. Three rows was rejected: it prevents two row groups from residing
together and regresses sustained calls by about `1.3%`.

Reducing the column group from eight to seven resident columns lowered
threadgroup scratch from `32 KiB` to `28 KiB` but destroyed the power-of-two
512-column tiling/alignment benefit. Forty warm paired calls measured
min/median/mean/p95/max
`0.29703/0.30811/0.31073/0.34097/0.35951 s`, with unchanged displayed losses.
It was reverted immediately; smaller non-power-of-two column groups cannot
turn reduced threadgroup allocation into useful bandwidth.

An exact-zero transfer census ruled out a pruned sparse row FFT. Across all
`1,446,912` `(BF,row)` transforms, nonzero input density was `73.291%`; the
row support-count p5/p25/p50/p75/p95 was `168/319/390/461/512` columns.
Only `34,843` rows (`2.408%`) were completely empty, which is the case the
production kernel already skips. With nearly three quarters of inputs live
and half the rows at 390 or more columns, sparse radix bookkeeping would add
divergence without eliminating the dense row intermediate.

Widening Optuna to three exact candidates per ask/tell group was worse even
with the retained `512`-BF chunk. The complete seed-42 run took `43.232 s`
(`26.105 s` Optuna plus 67 calls / `15.629 s` refinement) and converged to the
higher independently reconstructed loss `0.1113636643`. The retained batch-2
sequence takes 55 refinement calls and reaches `0.1113618687`; batch 3 was
therefore rejected for both wall time and scientific result quality.

The retained 55-request Nelder-Mead trace contains two pairs of distinct
float64 simplex coordinates that produce identical float32 `C10`, `C12`,
`cos(2*phi12)`, and `sin(2*phi12)` values at the Metal boundary (requests 37
and 44 repeat prior GPU inputs). Exact refinement memoizes those actual shader
inputs. The full
seed-42 validation issued 53 GPU evaluations for the same 55 simplex requests
and returned bit-identical best parameters, exact/final loss, phase statistics,
and amplitude statistics. This removes redundant exact work rather than
approximating it; the focused suite passes (`25 passed, 2 skipped`).

Nelder-Mead was also tested in Cartesian twofold-astigmatism coordinates
`(C12*cos(2*phi12), C12*sin(2*phi12))`, which evaluate the same physical
aberration but avoid polar-coordinate distortion. A bounded coefficient-step
sweep (`1.0`, `1.5`, `2.0`, and `2.5`) took 45-70 exact calls. The fastest
45-call path stopped at loss `0.1113634929`, and all variants ended between
`0.1113626212` and `0.1113657802`, above the retained `0.1113618687`. The
Cartesian refinement was rejected; its lower call count does not preserve the
best exact scientific result.

The accepted trace has 13 reflection, two expansion, 12 contraction, and one
shrink decisions. Pairing only naturally independent work (two initial
vertices and the two uncached shrink vertices) preserved every decision and
the exact final result without speculative calls. Alternating refinement runs
measured `6.13-6.20 s` with those two pair launches versus `6.19-6.21 s`
sequentially. The roughly `0.04 s` average difference is noise-scale and does
not justify a second evaluation topology inside Nelder-Mead, so natural
refinement batching was rejected.

The scalar exact optimizer evaluator was then found to dispatch through the
general reconstruction wrapper even though it needs only a loss. Alternating
real-data passes measured that wrapper at `0.223-0.226 s` per candidate versus
`0.122-0.127 s` through the already reference-checked fused exact batch kernel
with batch size one; both returned exactly `0.11136186867952347` at the best
fit. Scalar exact evaluations now use that loss-only fused dispatch. Final
phase reconstruction remains on the full reconstruction path.

Two complete 200-trial validations measured `28.943 s` and `28.900 s` for the
fit critical path. Refinement took `6.535 s` and `6.551 s` for 53 physical GPU
calls / 55 unchanged Nelder-Mead requests. End-to-end totals were `29.891 s`
with `0.947 s` setup and `30.568 s` with a noisy `1.667 s` setup. Both returned
bit-identical best parameters, exact/final loss, phase statistics, and
amplitude statistics. The focused suite passes (`25 passed, 2 skipped`).

The remaining difference between the general single-candidate wrapper and the
batch-one path was an MLX view materialization: indexing `[0]` from the Metal
row output copied a `512 MiB` `(512, 512, 512)` complex64 chunk. Keeping the
`(1, chunk, 512, 512)` output and removing the unit batch axis with `reshape`
is a shape-only view. Alternating one-chunk row passes fell from `28-39 ms` to
`10-13 ms`; the column stage remained `8.5-9.9 ms`, and full public exact
passes now match batch one at `0.122-0.128 s` with identical loss.

The complete 200-trial signoff measured `28.757 s` on the fit critical path
and `29.721 s` end to end. Final independent phase/loss reconstruction fell
from `0.320-0.322 s` to `0.123 s`. Best parameters, exact/final loss, phase
statistics, and amplitude statistics are bit-identical, and the focused suite
passes (`25 passed, 2 skipped`).

A final clean-source repeat after the IO work measured `28.690 s` on the same
fit critical path: `0.150 s` initial exact loss, `21.878 s` for 200 Optuna
candidates, `6.494 s` for 53 physical GPU calls / 55 Nelder-Mead requests,
`0.045 s` object reconstruction, and `0.123 s` final exact phase/loss. Total
wall time was `30.174 s` because setup took a noisier `1.483 s`. Best parameters
(`C10=73.6415583633`, `C12=14.8295713639`, `phi12=0.4843161759`), exact/final
loss `0.11136186867952347`, phase mean/std, and amplitude mean/std were all
bit-identical to the accepted signoff.

| Exact workflow checkpoint | Original profile | Final signoff | Improvement |
| --- | ---: | ---: | ---: |
| Optuna, 200 candidates | `34.485 s` | `21.878 s` | `1.58x` |
| Nelder-Mead refinement | `12.522 s` / 77 calls | `6.494 s` / 53 physical calls | `1.93x` |
| Comparable fit path | `49.392 s` | `28.690 s` | `1.72x` (`41.9%` lower) |

The remaining `8.690 s` to the requested 20-second target is larger than all
non-kernel work combined. On the current exact topology it requires about
`171 GB/s`, versus the reference system's measured `112-130 GB/s` sustained copy band. A
future under-20 result therefore needs an exact decomposition that removes the
global row-IFFT write/read, not another optimizer scheduling constant.

A sustained 100-call exact-pair soak on one prepared dataset measured
min/median/p95/max `0.2106/0.2149/0.2191/0.2227 s`, totaling `21.520 s` for the
same number of GPU calls used by the 200-candidate Optuna stage. Both losses
were bit-identical across all 101 calls including warm-up. The full Optuna
stage's `21.878 s` is therefore only about `0.36 s` above sustained kernel time;
TPE ask/tell and Python scheduling are not a large remaining target.

The pair traffic estimate also predicts the measured kernel directly:
`25.784 GB / 0.2149 s = 120.0 GB/s`. The scalar estimate gives
`15.471 GB / 0.1242 s = 124.5 GB/s`. Both effective rates sit inside the
independently measured native compute-copy band above. The row/G_qk accounting
is therefore not just an aggregate workflow estimate; it explains each fused
dispatch at the hardware bandwidth limit.

A longer 300-pair thermal soak then ran the same exact dispatch continuously
for `64.032 s`. Min/median/p95/max were
`0.20895/0.21316/0.21745/0.23499 s`; early/middle/late 100-call medians were
`0.21339/0.21244/0.21374 s`. The late median is only `0.16%` above the early
median, and both losses were bit-identical across all 301 calls including
warm-up. Sustained thermal behavior does not open or erase meaningful headroom.

The corresponding 53-call scalar soak measured
min/median/p95/max `0.1208/0.1242/0.1269/0.1299 s` and `6.594 s` total, again
with bit-identical loss on every call. The real 53-call refinement took
`6.494 s`; candidate-dependent activity and measurement noise are larger than
any Nelder-Mead orchestration cost. Both dominant optimizer stages are at their
kernel traffic floor.

The matching 300-call scalar thermal soak totaled `36.778 s`, with
min/median/p95/max `0.11828/0.12253/0.12575/0.12805 s`. Its
early/middle/late 100-call medians were `0.12252/0.12210/0.12285 s`, only
`0.27%` late-over-early, and every loss was bit-identical. The real 53-call
refinement is therefore much shorter than the validated stable scalar window.

Three additional back-to-back, fresh-process 200-candidate plus Nelder-Mead
endurance runs measured fit critical paths of `28.787`, `28.520`, and
`28.549 s`. Their Optuna stages were `21.949`, `21.711`, and `21.729 s`; all
three refinements used the same 53 physical calls and took `6.516`, `6.494`,
and `6.505 s`. Every run returned bit-identical best parameters, scalar loss,
phase mean/std, and amplitude mean/std. The accepted fused kernels therefore
show neither progressive thermal slowdown nor result drift after the extended
profiling session; `28.520 s` is the fastest complete fit-path observation,
while `28.690 s` remains the conservative clean-source signoff used above.

The final committed-tree repeat after restoring bounded native-IO buffer
lifetime measured `28.658 s` for the fit critical path: `21.823 s` Optuna,
`6.516 s` for the same 53 physical refinement calls, `0.042 s` object, and
`0.125 s` final phase. Its best parameters, loss, phase statistics, and
amplitude statistics were again bit-identical. Setup was a separately reported
`1.632 s`; the IO lifetime correction does not touch or regress the SSB path.

The one-time BF-column FFT setup chunk was also rechecked at
`256/512/1024/1413/2826` BF. Device-side comparison found the 256- and 512-BF
`G_qk` arrays and DC value bit-identical, but alternating fresh-process medians
favored the retained 256-BF setup at about `0.921 s` versus `0.979 s` for 512;
one all-2826 setup took `2.242 s`. Fewer setup launches do not repay the larger
FFT working set, so 256 remains the benchmark setup rather than moving noisy
setup time into the hot-path result.

Smaller `128`-BF setup chunks looked about `0.08 s` faster on the pre-extracted
BF-column companion, but the production full MPS `ChunkedFrames.columns` path
reversed the result. After a cold first run, 256-BF preparation measured
`2.251-2.261 s` versus `2.307-2.333 s` for 128 BF at the same radius and shape.
Extra full-stack column gathers outweigh the smaller FFT working set, so the
public 24 GB default also remains 256.

Two attempts to decouple full-stack column gathering from 256-BF FFT staging
were slower. Gathering all 2,826 selected columns once created a 741 MB
transposed uint8 matrix; keeping its wide stride took `3.763 s` total, while a
one-time contiguous copy still took `3.686 s`. A bounded 512-column gather
feeding two 256-BF FFT chunks took `2.857 s` (`0.913 s` gather plus `1.944 s`
conversion/FFT). Wider column strides and copies cost more than the saved
full-stack walks, so gather and FFT staging remain aligned at 256 BF.

The public column-gather worker count was swept at `4/6/8/10/14` on the same
path. No lower count beat the one-worker-per-output-chunk default; a direct
14-versus-8 interleave under a noisier system state averaged about `2.74 s`
versus `2.87 s` total preparation. The scattered unified-memory reads benefit
from chunk-level latency hiding even though the M5 has 10 CPU cores, so the
existing cap/default remains unchanged.

Filling `ChunkedFrames.columns` directly into a float32 scan-major gather was
also exact in representation and removed the later uint8-to-float32 copy, but
preparation regressed to `2.742 s` versus the retained `~2.25-2.26 s` band.
Quadrupling the scattered gather writes costs more than the compact sequential
conversion pass, so detector columns remain uint8 until FFT staging.

Using NumPy's `take(..., out=destination)` to remove each per-chunk temporary
also looked simpler but regressed controlled public preparation. Four fresh
processes gave median `2.585 s` with direct output versus `2.520 s` for the
restored temporary-plus-assignment path in the same system state. NumPy's
direct-output implementation does not beat the optimized temporary copy here.

### Public MPS load-to-fit signoff, 2026-07-26

The final public-API gate loaded the complete lossless-u8 MPS stack, selected
the same 2,826 radius-30 BF pixels, prepared `G_qk`, ran 200 exact Optuna
candidates plus Nelder-Mead, and independently reconstructed the final result.
The initial public profile measured `1.747 s` load plus `38.833 s` fit,
`40.580 s` total. Its fit split was about `3.80 s` BF selection, `2.91 s`
preparation, `31.10 s` exact evaluations, and `0.18 s` final object/phase work.

Preparation left unused FFT/gather temporaries in MLX's buffer cache beside
the 9.66 GB input and 2.98 GB prepared `G_qk`. Calling `mx.clear_cache()` once
after preparation preserves all active arrays but releases those temporaries
before paired row intermediates are allocated. The committed-tree repeat
measured `1.726 s` load plus `37.530 s` fit, `39.257 s` total; exact evaluations
fell to `30.132 s`. Best parameters, loss, phase statistics, and amplitude
statistics remained identical. The focused MPS suite passes and asserts that
the public fit clears setup cache exactly once.

Explicitly clearing the reusable IO decoder cache additionally reduced one
preparation from `3.09` to `2.21 s`, but did not improve exact evaluations and
would globally discard scratch intended for subsequent loads. It remains an
optional caller memory policy rather than a hidden Metal-fit side effect.

A proposed `threshold=0` geometric BF shortcut was not exact: the explicit
radius contains 2,827 detector pixels, while the positive-intensity rule
correctly excludes one masked bad pixel and selects 2,826. Skipping the BF sum
would change normalization, so the required `~3.8 s` selection remains in the
honest full-stack public total.

A follow-up considered dispatching the Metal mean-DP reduction only for the
2,827 pixels inside the explicit radius. That can be exact for pixel inclusion
only when the center, radius, detector sampling, and zero-threshold policy are
all fixed by the caller: it still has to sum every selected column to discover
the masked zero. It cannot transparently replace the public reduction because
the same full mean diffraction pattern supplies auto-centering and the detected
radius used to infer detector sampling. Adding an opt-in fixed-geometry mode
would change the API contract and would not shorten this benchmark's exact
optimizer critical path, so no restricted-reduction kernel was retained.

The zero-copy row fix prompted a fresh larger-chunk check for only the one-time
final phase pass. `chunk_bf=1024` measured `0.127-0.129 s`, versus
`0.123-0.129 s` for the retained `512`, and changed 228,461 phase pixels
(`max_abs=5.36e-7`, `mean_abs=3.32e-8`) through different float32 reduction
grouping. It was rejected as neither faster nor bit-identical.

Launching the independent final object and exact phase reconstructions on two
worker-owned MLX GPU streams preserved both arrays bit-for-bit and returned the
same loss, but took `0.168-0.173 s` versus `0.165-0.170 s` sequentially. The
final kernels already saturate the device, so stream creation/scheduling adds
overhead rather than hiding the object tail. Final-stage concurrency was not
adopted.

Removing the six explicit evaluations of the small per-chunk probe vector
`pk` kept paired losses identical but did not reduce full-call latency. Warm
paired calls without the synchronization were `0.210-0.223 s`; the restored
path was `0.211-0.221 s` and slightly better by median. Keeping `pk` lazy only
extends the MLX graph retained until the large row/column evaluation, so the
explicit small-vector materialization remains.

Replacing the remaining singleton-axis indexing on 1 MiB phase reductions
with reshape-only views preserved exact loss and passed the focused suite, but
full public calls remained `0.123-0.126 s`, indistinguishable from the indexed
path. Unlike the former 512 MiB row slice, MLX does not charge a material copy
cost here, so the simpler reductions were restored.

### Automatic whole-probe sparse-storage checkpoint, 2026-07-26

The automatic full-probe benchmark detects center
`(94.8845139, 96.3595276)` and radius `53.3599281` on the GPU-derived mean
diffraction pattern. It selects all `8,937` positive pixels inside that disk.
Only `2,464` of those pixels have nonzero support in the physical probe
aperture; the other `6,473` remain in the exact phase mean, variance
normalization, and DC value but contribute analytic zero phase.

The accepted sparse-storage path stores and transforms only those `2,464`
nonzero `G_qk` planes while retaining the original logical 512-BF reduction
boundaries. The final object kernel maps logical BF indices to packed storage,
so adding the analytic-zero pixels occurs in the same order as the dense path.
On the real 512 field, best parameters, exact/final loss, every reported phase
statistic, and every reported amplitude statistic were bit-identical to the
dense automatic benchmark.

The already computed mean diffraction pattern also supplies the selected
columns' FFT DC exactly when the frame count is a power of two and every
recovered sum is within float32's exact-integer range. The real u8 benchmark
passed a direct raw-column bit check: both paths produced complex64 bits
`[1241858495, 0]`. Other datasets fall back to reading all selected columns for
DC rather than assuming this round trip is exact.

| Automatic 8,937-BF checkpoint | Dense repeat | Sparse `G_qk` | Sparse + exact mean-DP DC |
| --- | ---: | ---: | ---: |
| Setup | `14.981 s` | `12.855 s` | `8.186 s` |
| Fit critical path | `29.366 s` | `29.509 s` | `32.415 s` noisy first run |
| Total | `44.352 s` | `42.366 s` | `40.601 s` |

The setup win is `6.795 s` (`45.4%`) versus the dense automatic repeat and
reduces resident prepared evidence from about `9.4 GiB` to about `2.4 GiB`.
The DC-reuse run's optimizer was above the established thermal band, so its
total understates the setup architecture win. The exact fit kernels and their
global row intermediate remain the dominant target.

The next version prototyped a 257-row intermediate from the mathematical
anti-Hermitian relation `C(-q, k) = -conj(C(q, k))`. It reduced Optuna to
`16.301 s`, refinement to `2.675 s`, and final exact phase to `0.086 s`; the
full automatic workflow took `29.841 s` including a first-use `1.288 s` kernel
compile and `9.433 s` setup. This confirms that eliminating half the corrected
rows is the right scale of architectural change.

It was not accepted. The optimum parameters and amplitude statistics stayed
identical, but loss changed from `0.04469207674` to `0.04469696432`, phase mean
changed by `2.40e-7`, and phase standard deviation changed by `2.85e-6`.
Isolation showed that the detector Nyquist column has no positive-frequency
partner and produces the largest violation; after correcting that column,
fast-sincos and float32 evaluation asymmetry still leave row residuals. The
prototype remains a measured topology lesson, not a no-precision-loss result.

A subsequent setup profile found that lossless-u8 `MetalRawBackend.mean_dp()`
was incorrectly routed through the host uint64 chunk fallback even though the
resident Metal virtual-image implementation already provides an exact u8
detector-sum kernel. Restricting the host fallback to uint32 moved mean-DP plus
automatic probe fitting from `3.846 s` to `0.350-0.392 s`. A complete exact
run then measured `4.133 s` setup, `25.502 s` Optuna, `4.680 s` refinement,
and `34.755 s` total. Automatic probe geometry, all `8,937` selected pixels,
best parameters, exact/final loss, and phase/amplitude statistics remained
identical.

The accepted setup now consists of about `1.79 s` HDF5 load, `0.35-0.39 s`
GPU mean-DP/probe fit, `1.20-1.45 s` active evidence gather/FFT, and
`0.45-0.48 s` metadata/missing-column/release overhead. Probe detection is no
longer the setup bottleneck; file load is the largest component.

The backend policy was then tightened: a selected GPU backend must execute an
operation on that backend or raise a corrective `NotImplementedError`; it must
never silently run a CPU implementation. MPS uint32 detector sum and frame-max
now raise until overflow-safe native Metal kernels exist, and the Numba
row-prefix production route was removed. Lossless u8/u16 detector sums remain
native Metal.

The honest automatic SSB harness also removed its pretransposed CPU/memmap
`bf_columns.u8` companion. It now runs raw HDF5 load, mean DP, probe fit,
active-column gather, FFT evidence preparation, optimization, and final
reconstruction through the MPS path. A tiled Metal gather reads scan-major
columns and writes BF-major float32 evidence. The first complete all-GPU run
measured `10.690 s` setup, `23.861 s` Optuna, `4.642 s` refinement, and
`39.655 s` total. Every scientific result remained identical. The setup
regression localizes to the Metal-buffer-to-MLX FFT handoff (`7.517 s` active
preparation); no CPU companion will be restored. The next setup breakthrough
must fuse or zero-copy that GPU handoff.

The first follow-up batched all `2,464` aperture-active columns into one exact
handoff. That reduced active preparation to `2.79-2.99 s` and complete setup
to `6.02-6.44 s`; the actual tiled Metal gather was only `0.13-0.19 s`.
The rest was MLX's 2,464-plane real FFT (`1.83-2.03 s`) plus framework import
and geometry. This corrected the earlier shorthand: the seven-second stage
was not a seven-second gather.

The next accepted version allocates the BF-major float32 destination in MLX,
wraps that same unified-memory allocation as a no-copy `MTLBuffer`, and has the
existing Metal gather write into it directly. Metal completion remains the
only synchronization before MLX consumes the array. There is no host compute
path and no second evidence allocation. The old allocation behavior remains
for callers that do not provide a destination.

| Exact all-GPU setup stage | Separate Metal/MLX allocation | Direct MLX allocation |
| --- | ---: | ---: |
| HDF5 load | `1.887 s` | `1.827 s` |
| GPU mean DP + automatic probe | `0.365 s` | `0.380 s` |
| Metal gather | `0.251 s` | `0.058 s` |
| MLX Hermitian FFT | `2.066 s` | `1.687 s` |
| Complete active evidence preparation | `3.059 s` | `2.103 s` |
| Complete setup | `5.480 s` | `4.515 s` |

The complete 200-trial seed-42 Optuna run plus unchanged Nelder-Mead returned
bit-identical automatic center/radius, all `8,937` logical BF pixels, optimum
parameters, final loss `0.04469207674264908`, and phase/amplitude statistics.
Its fit path was a pressure-state outlier (`41.534 s`) and is parity evidence,
not a replacement optimizer signoff. HDF5 load is now about `40%` of honest
setup, up from about `18%` in the first all-GPU profile. Reaching the requested
`80%` still requires replacing or fusing the fixed-size FFT and overlapping
the mean-DP reduction; it cannot come from another gather launch constant.

Two row/column FFT replacements were then measured and rejected. A universal
MPSGraph implementation wrote directly between MLX-owned buffers and executed
a real-to-Hermitian row FFT followed by a complex column FFT. It preserved
complex64 numerical agreement on rectangular inputs, but the real 2,464-plane
512 setup took `2.079 s` for the transform versus `1.570-1.687 s` for MLX and
regressed complete setup to `5.421 s`.

A lower-level prototype fused scan-major u8 column reads with four parallel
radix-8 row FFTs, then ran a dedicated radix-8 column pass. The contiguous
9-GiB Metal load itself remained fast (`1.785 s`), but direct strided detector
reads lost the tiled gather's coalescing: row and column stages took
`1.897 s + 1.358 s`, versus `1.695 s` for the retained gather plus MLX FFT in
the same process. It also moved the canonical loss by `3.7e-9`. The prototype
was deleted. Any future gather/row fusion must retain a tiled
scan-major-to-BF-major transpose; simply reading one detector column per FFT
threadgroup is the wrong memory topology.

The same audit removed 178 lines of dead production code: the never-dispatched
non-tiled gather shaders and pipeline, obsolete bin-2 compatibility helpers,
and two uncalled obsolete reconstruction helpers. Reference-only radix
kernels remain because parity tests intentionally exercise them.

The real automatic workflow also exposed a pathological public wide-batch
route. A four-candidate Metal threadgroup raised the 200-trial Optuna stage to
`60.914 s`. Tiling public batches larger than two through the exact fused-pair
kernel reduced the same batch-4 trial stage to `33.772 s` (`1.80x`) and kept
its optimum, loss, phase statistics, and amplitude statistics identical. The
batch-4 TPE trajectory is not the canonical seed-42 batch-2 trajectory because
four trials are asked before feedback; this is an infrastructure repair for
wide-batch callers, not a replacement canonical optimizer signoff. A parity
test now pins candidate batches `2/4/8` and asserts that no physical 512 kernel
receives more than two candidates.

The canonical pair profile was then synchronized at the row/column boundary.
After warm-up, twelve sparse logical slices totaled about `0.190 s` in dynamic
correction + row IFFT and `0.131 s` in column IFFT + phase/loss; forced stage
synchronization made the complete pair `0.330 s`. Increasing the requested
logical chunk from `512` to `1024/2048/8937` did not improve the warm call
(`~0.293-0.305 s` versus `~0.297-0.299 s`) and changed two sampled losses by
`+1/-1` float32 ULP. Larger logical reductions were rejected. A future launch
consolidation must emit the same twelve partial sums and add them in the same
order; changing the reduction association is not an exact speedup.

## Current CUDA/MPS checkpoint, 2026-07-17

This checkpoint excludes WebGPU by request. It uses full active BF evidence:
no scan crop, detector binning, BF subsampling, preview/settle split, or saved
derived float32/complex64 cache.

Microscopist workflow, real 512 field:

```text
source: held-out HDF5 master file
shape: (512, 512, 192, 192), uint16, 19.33 GB
BF policy: threshold=0.0, bf_radius=53, full active BF
active BF: 8827
resident G_qk: Hermitian complex64, (8827, 512, 257), 9.29 GB
```

CUDA GPU1 live-control timing, sustained real-data run:

| Quantity | Mean | p50 | p95 | FPS | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Phase-only | `31.10 ms` | `31.13 ms` | `31.20 ms` | `32.2` | Passes 30 FPS |
| Phase+loss | `31.27 ms` | `31.27 ms` | `31.37 ms` | `32.0` | Passes 30 FPS |

CUDA GPU1 calibration path, same data and BF policy:

| Stage | Wall time | Notes |
| --- | ---: | --- |
| HDF5 load | `1.19 s` | CUDA load of full `(512,512,192,192)` uint16 block |
| SSB construction / Gqk | `0.27 s` | Hermitian `G_qk`, `9.29 GB` |
| `optimize(n_trials=200)` | `7.39 s` | Full BF, no `bf_subsample`, latest rerun used `8793` active BF |
| `refine()` | `1.14 s` | Nelder-Mead, `36` evaluations |
| Final `result()` | `0.046 s` | Final loss `0.0464598909` |
| Total through final result | `9.91 s` | End-to-end CUDA/Python compute path |

CUDA synthetic matrix on GPU1, Hermitian `G_qk`, `8809` BF:

| Scan | Object mean / FPS | Phase mean / FPS | Phase+loss mean / FPS |
| --- | ---: | ---: | ---: |
| `128x128` | `4.85 ms / 206.1` | `8.30 ms / 120.5` | `8.35 ms / 119.7` |
| `256x256` | `2.20 ms / 454.0` | `20.94 ms / 47.8` | `20.99 ms / 47.7` |
| `512x512` | `8.59 ms / 116.3` | `27.24 ms / 36.7` | `27.33 ms / 36.6` |
| `1024x1024` | `41.79 ms / 23.9` | `195.54 ms / 5.11` | `197.71 ms / 5.06` |

The latest CUDA 1024 exact phase/loss path uses a split-512 row and column IFFT
topology over transposed scratch. It keeps the same full active BF evidence,
Hermitian complex64 `G_qk`, float32 phase/loss arithmetic, scan size, and
objective definition. On the synthetic full active BF-style benchmark, this
moved phase+loss from `382.24 ms` (`2.62 FPS`) to `197.71 ms` (`5.06 FPS`)
while keeping the CuPy memory-pool footprint about `45.9 GB`. This is a real
exact-kernel speedup, but it still fails the 10 FPS and 30 FPS targets.

Nsight Compute on the current CUDA 1024 exact phase/loss path shows the row and
column FFT kernels are scheduler/shared-memory limited, not disk or DRAM
bandwidth limited:

| Kernel | Eligible warps/scheduler | No eligible cycles | Achieved occupancy | Memory throughput | Main stall |
| --- | ---: | ---: | ---: | ---: | --- |
| `ifft1024_rows_fused_pk_split512_t64_packed` | `0.36` | `74.9%` | `31.6%` | `520 GB/s` | `123` regs/thread, MIO/short scoreboard, shared-memory and memory-pipe pressure |
| `ifft1024_cols_accumulate_split512_t64` | `0.70` | `58.8%` | `32.8%` | `888 GB/s` | `121` regs/thread, MIO/short scoreboard, high memory-pipe use |

The accepted split-512 topology is the first larger CUDA 1024 phase/loss
breakthrough in this sequence. The next breakthrough must reduce the
register-heavy row/gamma stage or the amount of exact per-BF phase work; chunk
size and BF-group retuning were measured and stayed flat.

MPS 512 timing on an Apple Silicon reference host, 2026-07-21. These rows intentionally separate BF
policies because exact phase/loss scales with the active BF count:

| BF policy | Quantity | Mean | p50 | p95 | FPS | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| radius `30` px, `2824` BF | Object redraw | `10.86 ms` | `11.28 ms` | `11.67 ms` | `92.1` | Real-time object-wave steering |
| radius `30` px, `2824` BF | Phase-only | `76.67 ms` | `76.88 ms` | `78.98 ms` | `13.0` | Reviewable, not 30 FPS |
| radius `30` px, `2824` BF | Phase+loss | `76.28 ms` | `76.52 ms` | `77.41 ms` | `13.1` | Reviewable, not CUDA-like |
| full active BF mask, `13137` BF | Object redraw | `55.20 ms` | `58.34 ms` | `61.43 ms` | `18.1` | Usable but not 30 FPS |
| full active BF mask, `13137` BF | Phase-only | `481.15 ms` | `476.32 ms` | `509.44 ms` | `2.1` | Too slow for live exact steering |
| full active BF mask, `13137` BF | Phase+loss | `528.90 ms` | `537.58 ms` | `557.51 ms` | `1.9` | Needs a deeper topology change |

Reference checks for this checkpoint:

- CUDA: `tests/test_ssb_cuda_128.py` + `tests/test_ssb_batch_optuna.py`,
  `29 passed` on GPU1.
- MPS: `tests/test_ssb_mps_cuda_reference.py`, `21 passed` on the Apple
  Silicon reference host with the matching real reference fixture; the full
  package suite also passed
  (`105 passed, 63 skipped`).
  This includes the fused 128/256/512/1024 column phase/loss helpers and the
  fused 128/256/1024 dynamic row-IFFT helpers.

MPS scalar-loss reduction is scientifically valid, but it does not solve the
large-BF wall time: avoiding a full phase-squared image write leaves the
row/column IFFT work dominant. The default exact phase/loss chunk is now
`4096` BF on 96 GB-class Macs, `1024` BF on 64 GB-class Macs, and `512` BF on
smaller Macs. The 96 GB Mac setting is a small warmed steady-state win, but it
is still not a real-time full-active-BF exact phase/loss breakthrough. The next
MPS breakthrough needs a different exact row/column FFT topology, not another
scalar-loss or chunk-size tweak.

Latest MPS exact-loss chunk repeats on the same real `512x512` field moved the
high-memory default from `3072` to `4096` BF. The earlier `3072` repeat
measured `165.47 ms` mean / `166.96 ms` p95, while a later single-candidate
repeat favored the larger chunks: `2048` BF measured `248.21 ms` mean /
`257.40 ms` p95 and `4096` BF measured `248.74 ms` mean / `251.28 ms` p95,
versus `3072` BF at `265.31 ms` mean / `311.79 ms` p95 in that same run.

## Exact object redraw path

`SSB.result()` displays the complex object wave and then exposes its phase.
For that object path, the inverse FFT can move outside the BF average:

```text
mean_bf(ifft2(corrected_bf)) == ifft2(mean_bf(corrected_bf))
```

The CUDA object redraw path now uses this identity for large native scans:

1. Compute the same per-BF `pk` correction as the full custom SSB path.
2. Sum corrected Fourier-domain terms in BF groups on the GPU.
3. Reduce those group sums to one corrected Fourier image.
4. Run one `ifft2` for the final object.

This changes the memory/computation topology only. It does not change the BF
selection, scan size, precision type, or object definition.

The path is intentionally separate from the phase-variance optimizer path.
`reconstruct()` and `reconstruct_with_loss()` average per-BF phase and still
need their own optimized kernels because `angle(mean(object))` is not the same
operation as `mean(angle(object))`.

## CUDA Hermitian G_qk storage

The CUDA object redraw path now uses a lower-memory resident `G_qk` layout by
default:

```python
ssb = SSB(...)                       # default: gqk_storage="herm"
```

`gqk_storage="herm"` stores only scan-frequency columns `0..N/2`. The CUDA
object Fourier-sum kernels mirror-conjugate the missing half-plane directly
when they fetch `G(q, k)`. The CUDA phase/loss row kernels also fetch the
Hermitian half-plane directly through `ld_gqk_maybe_herm`; the engine no
longer materializes transient full-plane `G_qk` chunks for those paths. This
relies on the exact Hermitian symmetry of the FFT of real virtual-BF images.
The object-wave definition, phase/loss definition, and BF selection are
unchanged; only the resident storage layout and fetch path change.

Persistent `gqk_storage="full"` has been removed from the public SSB runtime
path. Full-plane `G_qk` appears only in low-level reference checks as a canonical
expansion from the Hermitian half-plane; it is not a user-facing mode.

Resident `G_qk` memory becomes:

```text
full: num_bf * N * N         * sizeof(complex64)
herm: num_bf * N * (N/2 + 1) * sizeof(complex64)
```

For a microscopist this is useful when the goal is to fit aberrations and steer
the final object view without spending the persistent VRAM budget on redundant
Fourier columns. Phase mean, phase variance/loss, `optimize()`, `refine()`,
`grid_search()`, defocus sweeps, and higher-order aberration paths now keep the
same resident Hermitian storage and fetch missing columns on demand.

Implementation status from the 2026-07-17 pass:

- CUDA object kernels `128/256/512/1024` use Hermitian half-plane `G_qk` in the
  public SSB path; full-plane references are constructed only inside tests.
- CUDA phase/loss paths `128/256/512/1024` preserve Hermitian resident storage
  and fetch the missing half-plane directly inside the row kernels; no
  transient full-plane `G_qk` chunk is built for the current phase/loss paths.
- CUDA `512x512` phase-only redraw has a sum-only column accumulator so
  `reconstruct()` does not compute phase variance when `reconstruct_with_loss()`
  is not requested. The measured speed did not improve materially, which shows
  the remaining floor is FFT topology rather than the removed `sumsq` writes.
- `_extract_gqk(...)` builds the half-plane directly after the BF-stack FFT,
  avoiding a persistent full `G_qk` allocation.
- Reference checks validate default Hermitian end-to-end `SSB(...).result()` against
  explicit canonical full storage. A raw `cp.fft.fft2` redundant half-plane can
  differ from exact conjugate symmetry at the expected fp32 arithmetic-noise
  floor, so full storage is canonicalized from the half-plane rather than using
  that redundant noise as a separate reference.
- MPS fixed-preview and sparse-fit prepared paths now store the same Hermitian
  half-plane. The cached sparse objective Metal kernel fetches missing columns
  by mirror-conjugate symmetry; non-cached MLX paths expand only per chunk.

Synthetic storage benchmark on GPU1, `8809` BF pixels, object-redraw mode:

| Scan | Storage | Resident `G_qk` | Mean | p95 | FPS |
| --- | --- | ---: | ---: | ---: | ---: |
| `512x512` | full | `18.47 GB` | `15.65 ms` | `16.70 ms` | `63.9` |
| `512x512` | herm | `9.27 GB` | `14.96 ms` | `15.07 ms` | `66.8` |
| `1024x1024` | full | `73.90 GB` | `70.85 ms` | `74.41 ms` | `14.1` |
| `1024x1024` | herm | `37.02 GB` | `66.08 ms` | `67.99 ms` | `15.1` |

Interpretation: this pass is a memory-topology win with no observed object-
redraw penalty. It is not a 30 FPS breakthrough for `1024x1024`; reaching that
target on one GPU still needs a deeper FFT/reduction topology change.

Public constructor-to-result smoke profile on GPU1, synthetic
`(256, 256, 20, 20)` uint16 data with `47` BF pixels:

| Storage | Resident `G_qk` | Warm `result()` mean | Object agreement vs full |
| --- | ---: | ---: | ---: |
| default `herm` | `12.42 MB` | `0.20 ms` | `p99.9 abs = 0.0` |
| explicit `full` | `24.64 MB` | `0.50 ms` | reference |

This is an end-to-end API check (`SSB(...) -> result()`), not only a raw kernel
probe. The exact-zero reference agreement here comes from canonicalizing both storage modes
from the same Hermitian half-plane.

## Historical measured baseline

This subsection is retained as the pre-Hermitian-only CUDA benchmark history.
The later "Hermitian-only and MPS matrix follow-up" section is the current
source of truth for public runtime storage, MPS status, and post-removal test
results.

Hardware: RTX PRO 6000 Blackwell-class CUDA workstation.

Input: synthetic complex64 `G_qk`, fitted 192-pixel detector BF disk radius
`53 px`, `8809` BF pixels, native scan size, no crop, no binning.

Benchmark: `SSBEngine.reconstruct_object(C10, C12, phi12)` after cache warmup.

| Scan | Mean | p50 | p95 | FPS | VRAM pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| `128x128` | `0.80 ms` | `0.80 ms` | `0.81 ms` | `1247.9` | `2.3 GB` |
| `256x256` | `3.97 ms` | `3.37 ms` | `5.55 ms` | `251.6` | `9.4 GB` |
| `512x512` | `12.29 ms` | `12.30 ms` | `12.53 ms` | `81.4` | `19.1 GB` |
| `1024x1024` | `56.22 ms` | `55.54 ms` | `62.20 ms` | `17.8` | `76.2 GB` |

This meets the current live-object target for both sizes. Treat it as a kernel
microbenchmark, not as complete scientist-workflow signoff. Real-data HDF5
load, hot-pixel filtering, BF-mask setup, browser/widget interaction, and
display readback still need end-to-end checks for every public workflow.

The same run measured the existing phase-mean and phase+loss paths. Those
paths are different scientific quantities and remain the next optimization
target:

| Mode | `128x128` | `256x256` | `512x512` | `1024x1024` |
| --- | ---: | ---: | ---: | ---: |
| Object redraw | `0.80 ms / 1247.9 FPS` | `3.97 ms / 251.6 FPS` | `12.29 ms / 81.4 FPS` | `56.22 ms / 17.8 FPS` |
| Phase redraw | `10.05 ms / 99.5 FPS` | `22.49 ms / 44.5 FPS` | `72.39 ms / 13.8 FPS` | `342.71 ms / 2.9 FPS` |
| Phase+loss | `9.17 ms / 109.1 FPS` | `19.55 ms / 51.2 FPS` | `69.48 ms / 14.4 FPS` | `326.71 ms / 3.1 FPS` |

### Direct Hermitian phase/loss follow-up

The 2026-07-17 follow-up removed the transient full-plane `G_qk` chunk from
the CUDA phase/loss row kernels and routed `512x512` phase-only redraw through
the existing radix-8 column topology used by the variance kernel. Direct
Hermitian fetch alone was not a 2x breakthrough because `G_qk` fetch is not the
dominant cost; the radix-8 column path is the first real phase redraw speedup
from this pass.

Focused CUDA reference agreement after the direct-fetch change:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q tests/test_ssb_cuda_128.py

24 passed
```

Synthetic `512x512`, `8809` BF timing on GPU1:

| Mode | Storage | Mean | p50 | FPS |
| --- | --- | ---: | ---: | ---: |
| Object redraw | herm | `16.49 ms` | `16.82 ms` | `60.6` |
| Phase redraw | full | `86.35 ms` | `87.34 ms` | `11.6` |
| Phase redraw | herm direct fetch only | `86.66 ms` | `87.20 ms` | `11.5` |
| Phase redraw | herm + radix-8 column | `73.24 ms` | `73.25 ms` | `13.7` |
| Phase+loss | full | `89.98 ms` | `90.60 ms` | `11.1` |
| Phase+loss | herm + radix-8 column | `73.57 ms` | `73.59 ms` | `13.6` |

GPU event profile for the `512x512`, `8809` BF Hermitian phase redraw:

| Component | Total |
| --- | ---: |
| `pk` update | `0.35 ms` |
| Row gamma + row IFFT | `36.91 ms` |
| Column IFFT + phase accumulation | `37.75 ms` |
| Partial-sum reduction | `0.80 ms` |
| Profiled GPU total | `75.81 ms` |

After the radix-8 column route, the same component probe measured:

| Component | Total |
| --- | ---: |
| `pk` update | `0.33 ms` |
| Row gamma + row IFFT | `37.59 ms` |
| Radix-8 column IFFT + phase accumulation | `24.90 ms` |
| Partial-sum reduction | `0.81 ms` |
| Profiled GPU total | `63.63 ms` |

Interpretation: direct Hermitian fetch is the right storage architecture, but
live exact phase redraw is still limited by doing all per-BF row/column IFFTs.
The radix-8 column path reduces the column phase accumulation cost enough for a
single-GPU `1.22x` phase-redraw speedup (`89 ms -> 73 ms`) and brings
phase+loss to the same range, but it is still not the `30 FPS` target.
cuFFT was checked as a topology baseline for a `1024`-BF `512x512` chunk:
`ifft2` alone took `8.91 ms` and `angle().sum(axis=0)` another `2.93 ms`,
where the custom row/column/phase path takes about `8.7 ms` for the same
chunk. A naive cuFFT replacement is therefore not the next breakthrough.

### 512 exact phase/loss GPU1 push

The 2026-07-17 GPU1 optimization pass targeted the exact full-BF
`512x512`, `8809` BF phase/loss path directly. The target was `30 FPS`, or
`33.3 ms` per exact redraw. No detector binning, scan cropping, BF reduction,
preview path, persistent derived float/complex cache, or multi-GPU work was
counted as a win.

Accepted kernel changes:

- Added a `64`-thread radix-8 row/gamma kernel for the `C10/C12/phi12`
  phase/loss hot path. This replaced the older `128`-thread radix-4 row kernel
  for that path.
- Changed the row staging layout to `[bf, col, row]`, so the column
  phase/loss kernel reads coalesced memory. This intentionally trades more
  expensive row writes for a much cheaper column pass.
- Updated the batch variance row staging layout to match the transposed column
  reader, preserving reference agreement for batched optimizer candidates.
- Tested larger `512x512` column phase/loss BF groups as an intermediate
  partial-plane optimization, but the durable direct-accumulate path keeps the
  fixed 32-BF variance grouping. The row-variance kernel is specialized for
  32 BF pixels per group; changing only the wrapper group count under-counts
  BF evidence and is not valid.
- Relaxed the two 512 radix-8 hot kernels from `__launch_bounds__(64, 10)` to
  `__launch_bounds__(64, 8)`, which gave a small scheduling win without reference agreement
  changes.
- Added a 512 direct-accumulate path where the column phase/loss kernel
  atomically accumulates into the final phase planes. This removes the
  per-chunk partial-plane reduction launches; atomic cost is lower than the
  removed launch/reduction overhead at this size.
- Added an exact safe-inside aperture branch to `compute_geometry()`. For the
  real-data-like `512`, `8809` BF benchmark, both shifted apertures are exactly
  `1.0` for `99.9995%` of points, so the row kernel now avoids the soft-edge
  aperture `sqrt/div` path except at the edge while preserving reference agreement.
- Enabled CUDA fast math for these RawModules, switched global-load cache
  policy from `dlcm=cg` to `dlcm=ca`, changed the 512 column phase/loss loop
  from unroll `8` to unroll `2`, and retuned the 512 row/gamma launch bound to
  `__launch_bounds__(64, 10)` with the column kernel staying at
  `__launch_bounds__(64, 8)`.
- Replaced the hot 512 column `atan2f` calls with a degree-6 polynomial
  `atan2` helper. Full CUDA reference checks still pass; this is a small
  column-side win after `--use_fast_math`, not the main breakthrough.

Steady-state synthetic `512x512`, `8809` BF timing on GPU1:

| Mode | Before this pass | After radix-8 row | After transposed staging | After 64-BF groups | FPS after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase redraw | `70.57 ms` | `58.10 ms` | `53.46 ms` | `52.27 ms` | `19.1` |
| Phase+loss | `69.26 ms` | `58.09 ms` | `52.98 ms` | `52.36 ms` | `19.1` |

Component timing for phase redraw after the accepted changes:

| Component | Total |
| --- | ---: |
| `pk` update | `0.02 ms` |
| Row gamma + row IFFT + transposed write | `33.79 ms` |
| Column IFFT + phase accumulation | `14.40 ms` |
| Partial-sum reduction | `0.75 ms` |
| Profiled GPU total | `48.99 ms` |

Longer worker measurements are slightly slower than the isolated component
probe because they include the full `reconstruct()` / `reconstruct_with_loss()`
loop overhead and steady GPU clocks:

```text
phase: mean 53.46 ms, p50 53.74 ms, p95 53.91 ms, 18.7 FPS
loss:  mean 52.98 ms, p50 53.29 ms, p95 53.43 ms, 18.9 FPS
with 64-BF column groups:
phase: mean 52.63 ms, p50 53.12 ms, p95 53.31 ms, 19.0 FPS
loss:  mean 52.83 ms, p50 53.42 ms, p95 53.61 ms, 18.9 FPS
with 64-BF groups and relaxed launch bounds:
phase: mean 52.45 ms, p50 52.97 ms, p95 53.09 ms, 19.1 FPS
loss:  mean 52.67 ms, p50 53.31 ms, p95 53.44 ms, 19.0 FPS
with direct accumulation:
phase: mean 52.27 ms, p50 52.78 ms, p95 52.90 ms, 19.1 FPS
loss:  mean 52.36 ms, p50 52.97 ms, p95 53.11 ms, 19.1 FPS
with aperture shortcut, fast math/cache retune, column unroll 2, row launch
bound 10, and polynomial atan:
phase: mean 45.18 ms, p50 45.33 ms, p95 45.47 ms, 22.1 FPS
loss:  mean 45.11 ms, p50 45.32 ms, p95 45.42 ms, 22.2 FPS
```

Final component timing for the accepted 2026-07-17 incremental pass:

| Component | p50 total |
| --- | ---: |
| `pk` update | `0.015 ms` |
| Row gamma + row IFFT + transposed write | `29.99 ms` |
| Column IFFT + phase/loss accumulation | `13.71 ms` |
| Partial/direct final accumulation overhead | `0.59 ms` |
| Profiled GPU total | `44.39 ms` |

### 512 paired-BF phase/loss follow-up

The next GPU1 pass paired bright-field pixels at `+k` and `-k` for the
`512x512` C10/C12 exact phase/loss path. For the even C10/C12 probe,
`P(-k) == P(+k)` under the paired BF map, so the row kernel can share the
shifted probe geometry, `sincos`, and gamma normalization for the pair while
still applying each BF pixel's own `G_qk` evidence. This preserves the same
BF disk, scan size, and phase/loss definition; no preview path, binning,
cropping, saved `g_bf` cache, or multi-GPU work is counted.

Focused reference agreement during this pass:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q \
  tests/test_ssb_cuda_128.py -k 'engine_matches_explicit or phase_loss'

6 passed, 18 deselected
```

Sustained synthetic `512x512`, `8809` BF timing on GPU1 after the paired path:

| Mode | Mean | p50 | p95 | FPS |
| --- | ---: | ---: | ---: | ---: |
| Phase redraw | `43.30 ms` | `43.76 ms` | `43.87 ms` | `23.1` |
| Phase+loss | `43.30 ms` | `43.83 ms` | `43.93 ms` | `23.1` |

The first few samples in a run can report `37-38 ms`, but sustained samples
settle around `43-44 ms`. During the same run, `nvidia-smi dmon` showed GPU1
at `100%` SM, `83-85%` memory activity, about `280 W`, and about `1550 MHz`
graphics clock under the `300 W` board power limit. An unprivileged attempt
to raise GPU1 to its reported `325 W` max was rejected by the driver. GPU0 was
also checked as a single-GPU side-by-side check but was slower under current machine
load (`p50 ~47.8 ms`), so GPU1 remains the cleaner benchmark device.

Current component split for the paired full-storage path:

| Component | p50 total |
| --- | ---: |
| `pk` update | `0.015 ms` |
| Row gamma + row IFFT + transposed write | `28.55 ms` |
| Column IFFT + phase/loss accumulation | `13.69 ms` |
| Singleton/final overhead | `0.18 ms` |
| Profiled GPU total | `42.45 ms` |

Follow-up on GPU1 kept two exact-path micro-improvements for the `512x512`,
`8809` BF C10/C12 phase/loss path:

- The paired row kernel now processes four rows per block, reducing row-kernel
  block scheduling pressure while preserving the same `+k/-k` arithmetic.
- The C10/C12 chunked core can use a memory-aware transient staging buffer.
  On the 96 GB GPU1 test run it staged the full `8809 x 512 x 512` complex64
  intermediate in VRAM (`~37.5 GB` CuPy pool including source `G_qk`) and
  reduced row/column launch count. This is not a saved cache and does not
  change BF selection, scan size, precision, or phase/loss definition.
- The paired C10/C12 row helper now evaluates the quadratic phase directly
  from `r^2`, `dx^2 - dy^2`, and `2dxdy` instead of forming
  `cos(2phi)`/`sin(2phi)` through a division. This is restricted to the
  `512x512` paired C10/C12 hot path; broader polar/aberration paths still use
  the shared geometry helper.
- The paired row kernel now stages each 4-row `+k/-k` output tile in shared
  memory and writes row-contiguous groups to the transposed intermediate. This
  keeps the column reader's fast layout while reducing excessive global store
  sectors from the row stage.

Sustained timing after those changes:

| Mode | Mean | p50 | p95 | FPS |
| --- | ---: | ---: | ---: | ---: |
| Phase redraw | `39.26 ms` | `39.40 ms` | `39.46 ms` | `25.5` |
| Phase+loss | `39.28 ms` | `39.44 ms` | `39.54 ms` | `25.5` |

This is a small checkpoint, not the `30 FPS` target. The exact path still
misses the `33.3 ms` frame budget by about `6.1 ms` p50.

Component split for the full-staging four-row path:

| Component | p50 total |
| --- | ---: |
| `pk` update | `0.012 ms` |
| Row gamma + row IFFT + coalesced transposed write | `25.43 ms` |
| Column IFFT + phase/loss accumulation | `13.17 ms` |
| Singleton/final overhead | `0.11 ms` |
| Profiled GPU total | `~38.7 ms` |

### 512 real subpixel-BF dual path

The real held-out `512x512` central field has a subpixel fitted BF center. Under
the exact integer detector-pixel mirror test, that means there are no usable
`+k/-k` pairs:

```text
source: held-out HDF5 master file
shape: (512, 512, 192, 192), uint16, 19.33 GB
BF policy: bf_radius=53, threshold=0.0
active BF: 8822
exact symmetry pairs: 0
```

For that scientist workflow, the accepted exact optimization is an arbitrary
dual-BF row kernel. It pairs the remaining singleton BF pixels two at a time
for launch/staging efficiency, but computes each BF pixel's own `kx/ky`, probe
correction, `G_qk` fetch, inverse FFT, phase, and loss contribution. It is not
a symmetry approximation and does not reduce the BF disk, scan size, detector
sampling, or precision.

Focused reference agreement now includes this condition directly:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q tests/test_ssb_cuda_128.py

25 passed
```

The new regression test constructs a `512x512` subpixel-BF center with zero
exact symmetry pairs, asserts that the dual path is used, and checks
`reconstruct_with_loss()` against an explicit chunked CuPy reference.

Real held-out GPU1 timing after the dual-BF path, plus the Hermitian-specialized
dual row fetch:

| Mode | Storage | Active BF | Mean | p50 | p95 | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Phase+loss | herm | `8822` | `35.41 ms` | `35.40 ms` | `35.64 ms` | `28.2` |

Follow-up scalar-loss checkpoint: the exact C10/C12 loss path now keeps the
mean phase image as before but accumulates the phase-squared term into one
scalar for the no-pair dual path. This matches the optimizer objective,
because the loss only needs the global mean of `phase^2`; it avoids writing and
clearing a full per-pixel variance plane for the hot real held-out condition.
Focused CUDA reference agreement still passes:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  python -m pytest -q tests/test_ssb_cuda_128.py

25 passed
```

Sustained real held-out GPU1 timing after the scalar-loss path:

| Mode | Storage | Active BF | Mean | p50 | p95 | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Phase+loss | herm | `8822` | `34.97 ms` | `34.95 ms` | `35.47 ms` | `28.6` |

This is a valid incremental win, but not a signoff. The exact full-BF 512
phase/loss path still misses the `30 FPS` frame budget (`33.3 ms`) by about
`1.6-1.7 ms` on GPU1.

Component timing for the same no-pair condition:

| Component | p50 total |
| --- | ---: |
| `pk` update | `0.012 ms` |
| Dual row gamma + row IFFT + transposed write | `~21.3 ms` |
| Column IFFT + phase/loss accumulation | `~13.5 ms` |
| Final mean/loss bookkeeping | `<0.2 ms` |

This is real progress for the microscopist: the exact full-BF 512 phase/loss
view is now around `28 FPS` on the real central held-out dataset field. It is still a
fail against the declared `30 FPS` target because the frame budget is
`33.3 ms`, leaving a sustained `~2.1 ms` gap.

Hardware note from the same 2026-07-17 push: sustained GPU1 runs sit at the
`300 W` power limit with `100%` SM utilization, memory utilization around
`80-83%`, and SM clocks around `1.41-1.45 GHz`. A one-off side-by-side check on GPU0
(`600 W` limit, but shared with display and another process) crossed the target
at p50 (`32.98 ms`) while mean/p95 were noisy (`35.67 ms` mean,
`47.51 ms` p95). Treat GPU0 as evidence that the current kernel is near the
target on a higher-power card, not as a clean signoff. The GPU1 goal still
needs a code-side `~2-3 ms` sustained reduction or a deliberate power-limit
change by an operator.

Rejected follow-ups from this subpixel-BF pass:

| Candidate | Result | Decision |
| --- | --- | --- |
| Full-plane resident `G_qk` instead of Hermitian fetch | Real held-out p50 regressed to about `37.8 ms` and doubled resident `G_qk` from `9.29 GB` to `18.50 GB`. | Rejected. |
| Dual row launch bound relaxed from `__launch_bounds__(256, 4)` to `256,2` | Reference agreement passed, but real held-out timing regressed to about `35.5 ms` p50 in short runs. | Reverted. |
| Dual row blocks reduced from 4 rows/block to 2 rows/block | Reference agreement passed, but real held-out timing regressed to about `36.3 ms` p50. | Reverted. |
| Precomputed row/q/k term helper for the dual row kernel | Reference agreement passed, but register pressure made real timing worse (`~36.6 ms` p50). | Reverted. |
| Wrapper-only `_colvar_group` change from 32 to 64/128 | Initially looked faster, but it was invalid because `ifft512_rows_var_radix8_t64` hard-codes 32 BF pixels per group. A dynamic-k attempt hit illegal memory access at full BF. | Reverted; do not repeat without a separate reference-checked fixed-size kernel. |
| Column launch-bound tightening (`64,10`, `64,9`, `64,12`) | Full reference agreement passed for the tested bounds. `64,10` produced no sustained real-data win (`36.17 ms` p50 before the Hermitian branch), and `64,12` regressed column p50 to about `14.4 ms`. | Rejected: register pressure/spills outweighed the occupancy attempt. |
| Hermitian row offset precompute | Full reference agreement passed, but sustained real-data timing stayed about `35.43 ms` p50 versus `35.40 ms` for the simpler Hermitian branch. | Rejected: extra live offsets did not beat the compiler's simpler inline path. |
| Reusing loaded `qy` values across A/B dual gamma calls | Full reference agreement passed, but component p50 did not improve and added register-pressure risk. | Rejected: the compiler/read-only cache already handles this cheaply enough. |
| Runtime shared-memory carveout `100` on the no-pair row/column kernels | Scratch component timing left row p50 about `21.3 ms` and combined dual p50 about `34.7 ms`, not a sustained breakthrough. | Rejected: larger carveout did not turn the shared-memory occupancy warning into real FPS. |
| One-BF/four-row index row kernel | Existing reference agreement still passed, but forcing all BF pixels through the experimental topology gave row p50 about `24.2 ms` and row+column p50 about `37.6 ms`. | Rejected: doubling the independent BF blocks did not compensate for lost dual-BF staging efficiency. |
| Closed-form radix-8 source indices instead of `octal_reverse_512(tid*8+s)` | Full focused CUDA reference agreement passed. A 240-step real held-out run briefly measured `34.90 ms` p50, but a 600-step sustained run settled at `35.53 ms` p50, matching or slightly regressing the accepted `35.50 ms` baseline. | Rejected: not a robust wall-time win for a microscopist dragging controls. |
| Exact inside-aperture phase-identity branch for normalized gamma | Full focused CUDA reference agreement passed, but real held-out loss regressed to `37.43 ms` p50 (`26.7 FPS`). | Rejected: the added branch, `chi_k` load, and changed special-function mix cost more than the removed normalization. |
| Degree-5 column `atan2` polynomial | Full focused CUDA reference agreement passed and a 240-step run measured `35.22 ms` p50, but the 600-step sustained run regressed to `35.79 ms` p50. | Rejected: removing one FMA was not a durable column-side speedup. |
| Column loop unroll reduced from `2` to `1` | Full focused CUDA reference agreement passed. A 240-step run stayed near baseline (`35.53 ms` p50), and the 600-step sustained run regressed to `36.27 ms` p50. | Rejected: lower unroll did not overcome the register/L1TEX floor. |
| Column launch bound relaxed from `__launch_bounds__(64, 8)` to `64,4` | Full focused CUDA reference agreement passed, but the 240-step real held-out run regressed to `35.79 ms` p50. | Rejected: giving the compiler more register freedom did not beat the occupancy loss. |
| Module `--maxrregcount` reduced from `96` to `80` | Full focused CUDA reference agreement passed and a 240-step run improved to `35.12 ms` p50, but the 600-step sustained run settled at `35.72 ms` p50. | Rejected: lower register cap was not a durable sustained win. |
| Column phase/loss group reduced from `32` BF to `16` BF | Full focused CUDA reference agreement passed. A 240-step run stayed near baseline (`35.52 ms` p50), and the 600-step sustained run regressed to `36.25 ms` p50. | Rejected: extra scheduler parallelism did not offset the doubled group/atomic overhead. |
| Fixed `64`-BF column phase/loss kernel after scalar-loss path | Full focused CUDA reference agreement passed. A 240-step real held-out run improved to `34.62 ms` mean/p50, but the 600-step sustained run regressed to `35.25 ms` mean and `35.28 ms` p50. | Rejected: halving phase-sum atomics was not durable under sustained GPU1 power/occupancy behavior. |
| In-place `phase_sum` normalization after scalar-loss path | Full focused CUDA reference agreement passed. A 240-step run had `34.82 ms` p50 but worse mean/p95, and the 600-step sustained run regressed to `35.61 ms` mean and `35.62 ms` p50. | Rejected: allocation avoidance in finalization did not beat the existing CuPy expression path under sustained timing. |
| Reusing `_sum_buffer` for phase accumulation after scalar-loss path | Full focused CUDA reference agreement passed. A 240-step run had similar p50 but worse mean/p95, and the 600-step sustained run regressed to `35.63 ms` mean and `35.64 ms` p50. | Rejected: CuPy's fresh zeroed plane path is more stable than reusing and filling the internal accumulation buffer here. |
| Dedicated scalar-loss column kernel without per-pixel `sumsq0..7` accumulators | Full focused CUDA reference agreement passed, but the 240-step real held-out run regressed to `35.23 ms` p50. NCU showed registers dropped only from `115` to `108`, theoretical occupancy stayed `33.3%`, and L1TEX/eligible-warp stalls were unchanged. | Rejected: removing per-pixel sumsq registers was not enough to change the column kernel's occupancy class or latency floor. |
| Dual row `float4` row-IFFT packing for arbitrary no-pair BF pixels | Full focused CUDA reference agreement passed after isolating the probe to the real held-out no-pair dual row kernel. The 240-step real held-out run regressed to `36.18 ms` mean and `36.17 ms` p50 (`27.6 FPS`) versus the accepted scalar-loss baseline of about `34.97 ms` mean / `34.95 ms` p50. | Rejected: packing the A/B row FFTs into `float4` reduced no useful sustained wall time; the extra register/instruction pressure outweighed fewer shared-memory slots and helper calls. |
| Sequential-index mode for the arbitrary no-pair dual row kernel | The real held-out full-BF cache has `8822` sequential singleton BF pixels and dual pairs `[0,1], [2,3], ...`, so a guarded mode replaced `pair_a/pair_b` loads with affine `idx_a = base + 2*pair`. Full focused CUDA reference agreement passed, but the 240-step real held-out run regressed to `35.31 ms` mean and `35.31 ms` p50. | Rejected: pair-index gathers are not a meaningful part of the remaining row/column floor. |
| Row-aware singleton pairing for the arbitrary no-pair dual row kernel | Reordered singleton BF pairs to maximize same-detector-row pairs (`4403/4411` versus `4351/4411`) while keeping every BF pixel and no tail. Full focused CUDA reference agreement passed, but the 240-step real held-out run regressed to `35.24 ms` mean and `35.26 ms` p50. | Rejected: pairing locality alone does not remove enough row-kernel work without a different same-`ky` kernel. |
| Same-`ky` dual gamma helper inside the arbitrary no-pair row kernel | Added an exact branch for pairs with the same detector row so A/B share `dy_m` and `dy_p` while still computing each BF's own `kx`, `pk`, `G_qk`, IFFT, phase, and loss. Full focused CUDA reference agreement passed, but the 240-step real held-out run regressed to `35.40 ms` mean and `35.40 ms` p50. | Rejected: the extra branch/code pressure outweighed the small duplicated-geometry savings. |
| Warp-shuffle scalar-loss block reduction in the column kernel | Replaced the scalar `phase^2` shared-memory tree reduction with warp shuffles plus one shared handoff. Full focused CUDA reference agreement passed, but the 240-step real held-out run regressed to `35.46 ms` mean and `35.47 ms` p50. | Rejected: the block reduction barriers are not the dominant column cost; the register/L1TEX FFT path remains the floor. |
| Dual row transposed copy-out changed from `float2` stores to adjacent-row `float4` stores | Full focused CUDA reference agreement passed and a 240-step run measured `35.34 ms` p50, but the 600-step sustained run regressed to `36.06 ms` p50. | Rejected: fewer store instructions did not improve the sustained power-capped row path. |
| Dual row Hermitian path forced at compile time by replacing the `gqk_cols == 257` branch with `true` | Exploratory Hermitian-only timing regressed to `35.79 ms` p50 in a 240-step real held-out run. | Rejected: a separate Hermitian-only row kernel is not justified by this branch-cost probe. |

Nsight Compute on the accepted dual row kernel:

- Grid `(1, 128, 4404)`, block `(64, 4, 1)`.
- `64` registers/thread and `32.77 KB` static shared memory/block.
- `50.0%` theoretical occupancy, `49.2%` achieved occupancy.
- About `1.06` eligible warps/scheduler, with `~52%` cycles having no
  eligible warp.
- About `972 GB/s` memory throughput, `57.5%` memory busy, `64.6%` L2 hit
  rate, and `37.4%` L1/TEX hit rate.

Interpretation: the no-pair real-data path is no longer dominated by HDF5
load, BF selection, or Python. The remaining floor is the row/column FFT
topology: row is shared-memory/scheduler limited, and column still spends
about `13-14 ms` doing exact per-BF phase/loss accumulation. The next
breakthrough should target one of these structural costs, not another storage
flag.

Nsight Compute on the four-row paired row kernel with
`__launch_bounds__(256, 3)`:

- `77` registers/thread, no local/shared spills.
- `50.0%` theoretical occupancy, `48.7%` achieved occupancy.
- `0.80` eligible warps/scheduler and `61.7%` cycles with no eligible warp.
- `993 GB/s` memory throughput, `57.9%` memory busy, `65.1%` L2 hit rate.
- Main stall: still MIO/shared-memory pressure, but the coalesced tile write
  roughly halves warp cycles per issued instruction (`27.9 -> 14.5`) versus
  the prior direct transposed stores.

Nsight Compute on the column phase/loss kernel:

- `115` registers/thread, no local/shared spills.
- `33.3%` theoretical occupancy, `33.1%` achieved occupancy.
- `0.57` eligible warps/scheduler and `68.4%` cycles with no eligible warp.
- `846 GB/s` memory throughput but low L2 hit rate, with L1TEX scoreboard
  stalls. Launch-bound forcing to reduce registers regressed timing.

Follow-up full-plane profiling on GPU1, matching the synthetic FPS benchmark
storage mode, confirmed the same limit after the coalesced row-store commit:

- Paired row/gamma kernel: `77` registers/thread, no spills, `50.0%`
  theoretical occupancy, `49.2%` achieved occupancy, `0.57` eligible
  warps/scheduler, about `1.09 TB/s` memory throughput, and source counters
  reported about `50%` excessive shared-memory wavefronts.
- Column phase/loss kernel: `115` registers/thread, no spills, `33.3%`
  theoretical occupancy, `33.1%` achieved occupancy, `0.57` eligible
  warps/scheduler, about `847 GB/s` memory throughput, near-zero L1 hit rate,
  and source counters reported L1TEX scoreboard stalls plus about `33%`
  excessive shared-memory wavefronts.

Hardware note from the same continuation: GPU1 was power-capped at `300 W`.
During a sustained 512 loss benchmark it held `100%` GPU utilization with
throttle reason `0x4` and SM clocks around `1.59-1.61 GHz`. The driver reports
`325 W` as the max power limit, but raising GPU1 to `325 W` failed with
insufficient permissions. This is real clock headroom, but the available
`300 -> 325 W` increase is too small to explain the full `39.5 -> 33.3 ms`
target gap by itself.

Nsight Compute on the paired row kernel:

- `94` registers/thread, no local/shared spills.
- `41.7%` theoretical occupancy, `38.9%` achieved occupancy.
- `0.37` eligible warps/scheduler and `79%` cycles with no eligible warp.
- `577 GB/s` memory throughput, `91%` memory busy, `85.8%` L2 hit rate.
- The primary warning is still uncoalesced global traffic: about `63%`
  excessive global sectors, plus about `33%` excessive shared wavefronts.

Interpretation: the paired BF symmetry is a real exact-path improvement, but
it does not reach the `33.3 ms` / `30 FPS` target. The remaining bottleneck is
the same topology issue: the row stage writes a full transposed complex
intermediate so the column stage can read coalesced data. Reaching 30 FPS
requires a deeper row/column topology change that reduces this intermediate
traffic or coalesces both sides without changing the per-BF phase/loss math.

Final Nsight samples on a `1024`-BF chunk still show the structural limit:

- Row/gamma kernel: `93` registers/thread, no spills, `38.2%` achieved
  occupancy, `0.53` eligible warps/scheduler, high memory-pipe/MIO pressure,
  and about `550 GB/s` memory throughput. The row kernel is still the largest
  single cost.
- Column phase/loss kernel: `115` registers/thread, no spills, `32.1%`
  achieved occupancy, `0.59` eligible warps/scheduler, L1TEX scoreboard
  stalls, and about `845 GB/s` memory throughput.
- A scratch gamma-bypass lower-bound experiment, which is not scientifically
  valid and was reverted, still landed around `36 ms` component time. That
  means gamma-only shortcuts cannot reach the `33.3 ms` budget by themselves;
  the row/column staging and column phase accumulation topology also has to
  change.

Rejected candidates from the same pass:

| Candidate | Result | Decision |
| --- | --- | --- |
| Phase-only radix-8 column variant | Reference agreement passed, but timing stayed around `69.5 ms` before the row/transposed breakthrough. | Reverted. |
| Resident aperture-pair cache | Reference agreement passed, but full-BF frame time regressed to about `178 ms` because the row kernel streamed two huge aperture arrays every redraw. | Reverted. |
| Shared-memory padding in row/column radix-8 kernels | Reference agreement passed, but short-run timing was neutral (`~50.4 ms`) and component timing did not improve. | Reverted. |
| Legacy radix-4 column accumulator | Column pass measured about `43 ms`, versus about `14 ms` for the transposed radix-8 column path. | Rejected. |
| Batch throughput for optimizer candidates | Batch sizes `4/8/16` stayed near `19.2` exact eval/s (`~52 ms/eval`). | Not a 30 FPS breakthrough. |
| Replacing `sqrtf`/division geometry with `rsqrtf` geometry in the exact C10/C12 helper | Focused reference agreement failed the 1024 explicit-reference gate (`p99.9` phase error `4.35e-4` versus `3e-4`). | Reverted. |
| Lowering global CUDA `--maxrregcount` from `96` to `64`/`48` | Component timings were only noise-level better, and sequential full-loop timing stayed around `53 ms`. | Reverted. |
| Row-major row output followed by an out-of-place tiled GPU transpose | Reference agreement passed, but `512` phase redraw regressed to `71.4 ms` (`14 FPS`) because the added full-stack transpose outweighed the row-store savings. | Reverted. |
| 512 phase-only skip of `sumsq` accumulation | Reference agreement passed, but phase-only timing barely moved and loss did not improve. | Reverted as extra complexity without a target-path win. |
| 512 column `__launch_bounds__(64, 10)` under the final compiler settings | Reference agreement passed, but steady p50/p95 regressed relative to column launch bound `8`. | Reverted. |
| 512 row `__launch_bounds__(64, 12)` under the final compiler settings | Reference agreement passed, but steady p50/p95 regressed relative to row launch bound `10`. | Reverted. |
| Computing 8 rows per block and writing transposed tiles directly | Reference agreement passed, but `512` phase redraw regressed to `60.1 ms` (`16.6 FPS`) from lower occupancy/shared-memory cost. | Reverted. |
| Computing 4 rows per block and writing transposed tiles directly | Reference agreement passed, but `512` phase redraw still regressed to `55.7 ms` (`18.0 FPS`). | Reverted. |
| Computing 2 rows per block and writing transposed tiles directly | Reference agreement passed, but `512` phase redraw regressed to `54.5 ms` (`18.4 FPS`). | Reverted. |
| Skipping `sincos` when shifted apertures are exactly zero | Reference agreement passed, but `512` phase redraw stayed around `53.1 ms`; branch/control-flow cost offset the skipped work. | Reverted. |
| Raising the exact phase/loss chunk cap from `2 GB` to `4 GB` | Reference agreement passed, but phase/loss timing stayed around `52.7-52.9 ms` while using more transient memory. | Reverted. |
| Raising the column phase/loss BF group from `64` to `128` | Reference agreement passed, but phase/loss timing regressed slightly to `52.8-53.0 ms`. | Reverted. |
| Relaxing 512 radix-8 launch bounds further from `8` to `6` blocks | Reference agreement passed, but phase timing regressed to `52.6 ms` from the `52.45 ms` launch-bounds-8 result. | Reverted. |
| Raising the direct-accumulate BF group from `64` to `128` | Reference agreement passed, but phase timing regressed to `52.47 ms` from the `52.27 ms` direct 64-BF result. | Reverted. |
| Paired row FFT helper applying `+k` and `-k` simultaneously | Reference agreement passed, but sustained p50 regressed to `43.1 ms`; higher register pressure erased the saved barrier/twiddle work. | Reverted. |
| Exact phase/loss staging chunk raised from `2 GB` to `4 GB` with paired rows | Reference agreement passed, but timing stayed around `42.9 ms` while using more transient VRAM. | Reverted. |
| Global CUDA `--maxrregcount=80` with paired rows | Reference agreement passed, but p50 stayed around `42.7 ms`; this broad compile knob was not worth the risk. | Reverted. |
| Paired row launch bound `__launch_bounds__(64, 12)` | Reference agreement passed, but p50 regressed to about `42.9 ms`; `64,10` remains better. | Reverted. |
| Column BF group `128` after the paired row change | Reference agreement passed, but sustained p50 stayed around `43.9 ms` and mean worsened slightly. | Reverted. |
| Column BF group `16` after the paired row change | Reference agreement passed, but sustained p50 regressed to about `43.8 ms`; it added more atomic/group overhead. | Reverted. |
| Column BF group `48` after the paired row change | Reference agreement passed, but sustained p50 regressed to about `43.9 ms`; `32` was the best measured group in this pass. | Reverted. |
| Column BF group `64` after the coalesced paired-row write | Reference agreement passed, but exact loss p50 regressed to `39.56 ms`; halving group/atomic count reduced wavefront parallelism too much. | Reverted. |
| Partial-plane reduction instead of direct atomics for paired chunks | Scratch component timing was slower (`pair_col` p50 about `13.3 ms` plus reduction) than direct accumulation. | Rejected. |
| Two-lane column phase/loss block `(64,2,1)` | Reference agreement passed, but sustained p50 stayed about `43.8 ms`; extra shared-memory reduction and lower occupancy offset the halved BF-loop iterations. | Reverted. |
| Direct full-plane `G_qk` loads in the paired row kernel | Full-storage timing regressed to p50 `44.0 ms`; the Hermitian-capable helper branch is not the row bottleneck. | Reverted. |
| `16x16` tiled intermediate layout balancing row writes and column reads | Reference agreement passed, but p50 regressed to `50.5 ms`; improved row-store locality was outweighed by worse column-load locality. | Reverted. |
| Eight rows per paired row block using dynamic 64 KB shared memory | Reference agreement passed, but p50 regressed to `46.0 ms`; lower occupancy/shared-memory pressure outweighed lower block count. | Reverted. |
| Packed `float4` helper transforming the `+k/-k` row FFTs together | Reference agreement passed, but p50 regressed to `43.7 ms`; extra register and shuffle pressure outweighed saved barriers. | Reverted. |
| Column launch bound `__launch_bounds__(64, 12)` | Reference agreement passed, but p50 regressed to `44.25 ms`; forcing more blocks over-constrained the compiler. | Reverted. |
| Column launch bound `__launch_bounds__(64, 10)` | Reference agreement passed, but p50 regressed to `43.21 ms`; the original `64,8` launch bound remains best for the column kernel. | Reverted. |
| Paired row launch bound relaxed from `__launch_bounds__(256, 3)` to `256,2` | Reference agreement passed, but p50 regressed to `44.08 ms`; lower occupancy outweighed any extra compiler freedom. | Reverted. |
| Paired row blocks reduced from 4 rows/block to 2 rows/block | Reference agreement passed, but p50 regressed to `43.11 ms`; lower shared memory per block did not hide the row-stage stalls. | Reverted. |
| Paired row blocks reduced to 1 row/block | Reference agreement passed, but p50 regressed to `43.35 ms`; more independent blocks added overhead without enough latency hiding. | Reverted. |
| Bit-reversed transient row layout plus contiguous column loads | Reference agreement passed after fixing the direct/partial address mode, but the split-kernel version stayed around p50 `42.8 ms` and the branch version raised column registers from `115` to `127` for only a noise-level win. | Reverted. |
| Contiguous `G_qk` row-load microscope for a hypothetical column-permuted storage layout | Synthetic constant-`G_qk` throughput probe regressed to p50 `47.8 ms`; row `G_qk` column order is not the current breakthrough. | Reverted. |
| CUDA cache policy `-Xptxas=-dlcm=cg` instead of `ca` | Exact loss p50 regressed to `42.83 ms`; cache-all remains better for the current row/column mix. | Reverted. |
| Column no-prefetch register-reduction variant | Reference agreement passed and registers dropped from `115` to `101`, but exact loss p50 stayed about `39.42 ms` and L1TEX scoreboard stalls worsened. | Reverted. |
| Degree-5 `atan2` polynomial in the phase accumulator | Full CUDA reference agreement passed, but A/B timing was noise-level (`~0.04-0.09 ms` p50) and not worth a precision-sensitive change. | Reverted. |
| Negative `use_partial` phase-only branch to skip dummy `sumsq` writes | Small CUDA reference agreement passed, but the full `8809` BF phase benchmark hit `CUDA_ERROR_ILLEGAL_ADDRESS`, reproducing the earlier unsafe branch failure mode. | Reverted. |
| Column read-only `ld_float2` loads for the transposed intermediate | Full CUDA reference agreement passed, but exact loss p50 regressed to `39.48 ms`; the plain global-load path remains better for the streaming intermediate. | Reverted. |
| Global CUDA `--maxrregcount=80` after the coalesced paired-row write | Full CUDA reference agreement passed, but exact loss p50 stayed around `39.29 ms`; occupancy pressure is not solved by this broad cap. | Reverted. |
| Global CUDA `--maxrregcount=128` after the coalesced paired-row write | Full CUDA reference agreement passed, but exact loss p50 stayed around `39.46 ms`; extra compiler freedom did not reduce the dependency floor. | Reverted. |
| Runtime preferred shared-memory carveout on paired row/column kernels | Scratch timings regressed to about `41-47 ms` p50 for carveout values `0-50`; the default driver carveout remained best. | Rejected. |
| Row-level aperture-fast microscope for the synthetic geometry | Only rows `250..262` can touch the soft aperture edge, but hard-coding aperture=1 elsewhere still measured about `39.47 ms` p50; the skipped branch is not the row bottleneck. | Reverted. |

GPU1 was saturated during the long run (`100%` SM at the `300 W` power cap,
about `66%` memory controller). The remaining exact-path bottleneck is not
data loading or React/browser rendering. It is the row/column IFFT topology:
the column pass is now much cheaper, but the row pass pays for exact gamma
math, row IFFT, and strided transposed stores. The next single-GPU breakthrough
needs a topology that gives both coalesced row writes and coalesced column
reads, or fuses/tile-transposes the two dimensions without changing the exact
per-BF phase/loss definition.

Do not assume the half-plane source storage permits a half-complex inverse FFT
for exact phase/loss. A direct symmetry probe on GPU1 showed the source
`fft2(real)` plane was Hermitian to `3e-7` relative error, but after the SSB
`q-k`/`q+k` phase/aperture correction the corrected Fourier plane had relative
Hermitian error about `2.0`. The corrected per-BF image is complex, so a
real-output half-complex IFFT would be mathematically wrong for exact phase
mean/loss.

Rejected broad exact-path experiment: replacing the standard C10/C12 polar
phase calculation with an algebraically equivalent Cartesian polynomial across
the general fixed-size kernels reduced some row-stage math on paper, but
changed float32 rounding enough to fail the current 1024 explicit-reference
gate (`p99.9` phase error `3.89e-4` versus `3e-4`). Do not generalize that
shortcut without an explicit reference/tolerance decision. The narrower
512-only C10/C12 hot-path helper used by the accepted paired/dual kernels is
covered by the focused CUDA reference agreement suite above.

Synthetic `1024x1024`, `1382` BF Hermitian timing on GPU1:

| Mode | Mean | p50 | FPS |
| --- | ---: | ---: | ---: |
| Object redraw | `9.42 ms` | `9.42 ms` | `106.1` |
| Phase redraw | `63.63 ms` | `63.35 ms` | `15.7` |
| Phase+loss | `71.25 ms` | `65.95 ms` | `14.0` |

The practical next breakthrough is not another `G_qk` layout flag. Keep the
roadmap single-GPU: redesign the row/column FFT topology with less
shared-memory/barrier pressure, reduce repeated row-stage math, or use a
clearly labeled preview/settle workflow when the user is willing to inspect
object phase during drag and exact mean phase on release.

## 12-cell backend tracking matrix

Track native SSB live-redraw work as a 12-cell backend matrix: three GPU
backends by four native scan sizes. Each cell must record implementation
status, reference-check status, and the best measured performance before it is treated
as a supported scientist workflow.

| Backend / size | `128x128` | `256x256` | `512x512` | `1024x1024` |
| --- | --- | --- | --- | --- |
| CUDA object redraw | Implemented. High-BF exact fallback mean `4.81 ms`, p95 `5.08 ms`, `208.1 FPS`. | Implemented. Mean `1.74 ms`, p95 `1.78 ms`, `575.2 FPS`. | Implemented. Mean `6.97 ms`, p95 `7.30 ms`, `143.5 FPS`. | Implemented. Older full-BF mean `56.22 ms`, p95 `62.20 ms`, `17.8 FPS`; current pass measured `3000` BF mean `12.41 ms`, p95 `12.60 ms`, `80.6 FPS`. |
| MPS Hermitian preview/free-fit | Implemented on a Mac MPS machine. Sparse `3.60 ms` / exact `4.02 ms` at `128` BF. | Implemented on a Mac MPS machine. Sparse `10.25 ms` / exact `10.68 ms` at `96` BF. | Implemented on a Mac MPS machine. Sparse `28.27 ms` / exact `33.26 ms` at `64` BF. | Implemented on a Mac MPS machine. Sparse `44.66 ms` / exact `50.39 ms` at `24` BF. |
| WebGPU phase/loss path | Real BF30 crop agreement passed against CUDA on NVIDIA WebGPU: phase max abs `1.42e-7`, FFT log-mag max abs `4.89e-6`, loss diff `2.37e-8`; warm full-BF WGSL `3.3 ms`, FFT `2.7 ms`. | Implemented in domain-owned WebGPU source bundled by `quantem.widget`. Synthetic browser reference agreement passed; real CUDA-reference agreement artifact still needed. | Implemented in domain-owned WebGPU source bundled by `quantem.widget`. Real 512 full-BF drive measured mean `31.4 ms` GPU and `41.8 ms` UI for C10 changes at `9070/9070` BF; real CUDA-reference agreement artifact still needed. | Implemented in domain-owned WebGPU source bundled by `quantem.widget`. Real 1024 BF-column load passes on Mac Chrome Metal. Full active-BF controls work but remain about `168-170 ms` UI/GPU, about `5.9 FPS`, below the 30 FPS target; real CUDA-reference agreement artifact still needed. |

Interpretation:

- CUDA is the only backend with all four object-redraw cells implemented and
  reference-checked against the previous per-BF IFFT object path. The current
  `1024x1024` full-BF synthetic rerun needs a quiet GPU because the Hermitian
  resident `G_qk` allocation is about `37 GB` before scratch.
- MPS now stores prepared `G_qk` as the same Hermitian half-plane and supports
  `128/256/512/1024` MLX/Metal preview/free-fit runs. Treat the MPS table as
  prepared-data MPS evidence, not as CUDA object Fourier-sum reference agreement or full-BF
  real-data signoff.
- WebGPU executes inside the browser bundle, but reusable SSB TypeScript/WGSL
  source now lives beside the SSB domain and is synced into
  `quantem.widget` before bundling. Use `quantem.gpu` native CUDA/MPS SSB
  outputs as the reference for browser agreement checks.
- Current WebGPU real-data signoff is strongest at `128x128` BF30. The
  `256/512/1024` cells are implementation/interaction evidence until a frozen
  native `quantem.gpu` reference artifact is generated for each size and the
  browser compares phase, FFT display, and loss at the same aberration state.

### WebGPU 1024 status from 2026-07-16

The browser kernel was extended to support `1024x1024` by using a 256-thread
WGSL topology with looped row/column load-store and looped butterflies. This
avoids relying on a 1024-thread workgroup, which common WebGPU limits reject.

Browser performance signoff must record the WebGPU adapter. SwiftShader,
llvmpipe, or any other software adapter is a CPU fallback. It can prove that an
HTML page opens, fetches data, and avoids crashes, but it is not valid evidence
for FPS, GPU latency, or end-to-end interactive performance. Re-run those tests
on Mac Chrome Metal, a working NVIDIA Chrome/WebGPU session, or the native
CUDA/MPS path before claiming responsiveness.

Headed Chrome/CDP evidence on NVIDIA Blackwell:

| Case | Data | Result |
| --- | --- | --- |
| Synthetic shape matrix | `128/256/512/1024`, 8 BF pixels | Browser reference agreement passed for phase and FFT log-magnitude at every size. |
| Real BF30 crop agreement | `128x128`, `2829` BF pixels | Phase max abs `1.42e-7`, FFT log-mag max abs `4.89e-6`, loss diff `2.37e-8` versus CUDA; warm full-BF WGSL `3.3 ms`, FFT `2.7 ms`, real NVIDIA WebGPU adapter. |
| Synthetic stress | `1024x1024`, 64 BF pixels | WGSL compute mean `8.2 ms`; page wall mean about `503 ms` because the standalone reference agreement/demo page repaints and checks too much on the CPU. |
| Real held-out full BF | `512x512`, `9070/9070` BF | C10 keyboard drive mean `31.4 ms` GPU, mean `41.8 ms` UI, about `23.9 FPS`; screenshot/report under `local UI report directory`. |

Local real `1024x1024` data target used for browser signoff:

```text
/path/to/local/1024_scan.h5
dataset: /entry/data/data
native shape: (1024, 1024, 192, 192) via flattened (1048576, 192, 192)
dtype: uint16 on disk; exact max count 12, so uint8 is lossless for browsing/load
wrapper: /path/to/local/1024_scan_master_wrapper.h5
```

### Evidence-selective WebGPU BF loading

The preferred 1024 browser source is no longer a persistent float32/complex64
`g_bf` cache. It is an exact detector-major BF-column companion:

```text
/path/to/local/1024_bf_columns.u4
layout: [bf, scan]
shape: [1805, 1048576]
encoding: uint4
size: 946.3 MB / 902.5 MiB
```

This companion stores raw detector counts only for the BF candidates, packed
losslessly because the real dataset's maximum count is `10`. It is not a
derived `g_bf` cache, does not reduce the scan, and does not bin the detector.
The browser range-reads only the active BF columns required by the current BF
policy:

| BF policy | Selected BF | Active aperture BF | Bytes fetched |
| --- | ---: | ---: | ---: |
| `BF=0.3` | `542/1805` | `379` | `198.7 MB` (`189.5 MiB`) |
| `BF=1.0` | `1805/1805` | `1382` | `724.6 MB` (`691.0 MiB`) |

Interpretation for the microscopist: `BF=1.0` is the full active-BF review
path, but it still should not fetch non-BF detector pixels or decode the whole
scan-major HDF5 file. `BF<1.0` is a selected-BF or preview policy and must be
reported as such when reporting speed.

Headed Mac Chrome Metal result after switching the real 1024 target to the
BF-column companion:

| Step | Result |
| --- | --- |
| Adapter gate | `apple metal-3`; `software=false`. This is valid browser WebGPU evidence. |
| BF-column dispatch fix | Changed the BF-column unpack dispatch from the old `16 x 16` shape to `32 x 8`, so `1024 x 1024` uses `32768` workgroups in X instead of exceeding Chrome's `65535` per-dimension WebGPU limit. |
| Default BF setup | `BF=0.3`, `542/1805` selected, `379` active, `0.199 GB` fetched, `558 ms` fetch, `157 ms` unpack, `40 ms` FFT, `757 ms` total. |
| Full active BF setup | `BF=0.99-1.0`, `1783-1805/1805` selected, `1382` active, `0.725 GB` fetched, `1668 ms` fetch, `640 ms` unpack, `216 ms` FFT, `2525 ms` total. |
| Full active BF controls | C10, C12, phi12, and scan rotation all updated the phase image; UI mean `169.5 ms`, GPU mean `167.8 ms`, about `5.9 FPS`. |
| Screenshots | `local screenshot: after-load`, `local screenshot: bf1-exact`, `local screenshot: controls-visible`. |

This is a real improvement in first-use loading and source layout, not a solved
redraw target. For full active BF at 1024, the browser still recomputes the
phase/loss path over `1382` active BF columns on each control change. The next
breakthrough must change that WebGPU math topology, for example by porting an
exact object Fourier-sum formulation or another reference-checked reduction that
keeps the same BF policy and precision. Do not align the CUDA object-redraw
`17.8 FPS` figure directly with this browser phase/loss path without naming the
different scientific quantity being timed.

### Browser WebGPU SSB checklist

Before a browser result is treated as a supported scientist workflow, record:

- [ ] Hardware adapter, with `software=false`; reject SwiftShader/llvmpipe FPS.
- [ ] Native phase size: `128`, `256`, `512`, or `1024`.
- [ ] Source mode: BF-column companion or compressed HDF5 fallback.
- [ ] Selected BF and active aperture BF; label preview BF separately from
  full active BF.
- [ ] First-use profile: bytes fetched, fetch, unpack/decode, FFT/reducer
  setup, and total time.
- [ ] Interaction profile for C10, C12, phi12, scan rotation, histogram,
  colormap, FFT, and flip where those controls are visible.
- [ ] Mean, p50, p95, and FPS-equivalent UI/GPU timing from repeated drives.
- [ ] Screenshot/report paths and console/WebGPU errors.
- [ ] Pass/fail against the declared target frame budget.

Headed Chrome result on a Linux GPU workstation after adding the range-index HDF5 source path:

| Step | Result |
| --- | --- |
| Initial compressed-source load | Reducer ready in `15.6 s` wall; profile `parse 103 ms`, `decode 5013 ms`, `gather 2286 ms`, `fft 65 ms`, total setup `13.05 s`. |
| Network shape | One small master fetch, one `16 MB` chunk-index fetch, then `206` byte-range reads for the `2.7 GB` compressed HDF5 data file. The previous single `200` full-file fetch failed in Chrome before WebGPU work started. |
| 0.3 BF interaction | `542/1805` selected BF, `379` active aperture BF. C10/C12/phi12/scan-rotation coordinate drives updated live, with UI readouts `148-200 ms` and GPU readouts `134-189 ms`. |
| Near-full BF setup | `1767/1805` selected BF, `1382` active aperture BF. HDF5 setup completed in `14.1 s` wall; profile total `13.11 s`. |
| Near-full BF interaction | C10 drive updated live at about `141 ms` GPU and `152 ms` UI. |

Interpretation for the microscopist: the full native 1024 field can now be
opened from the compressed HDF5 source without saving `g_bf.c64`, and the
controls do update the scientific image and FFT at 1024. It is not yet a
30 FPS steering experience. The next WebGPU work is reducing redraw latency
for the 1024 phase/loss path, not further reducing or binning the dataset.

This is the correct real-data target for WebGPU 1024 workflow testing. Do not
substitute a synthetic 1024 page for final signoff.

## Reference evidence

Focused CUDA reference checks live in `tests/test_ssb_cuda_128.py`.

The object Fourier-sum path is validated against the previous per-BF chunked
IFFT object path for `128x128`, `256x256`, `512x512`, and `1024x1024`:

- `p99.9(abs_err) < 5e-9`
- `p99.9(rel_err) < 1e-4`

The Hermitian `G_qk` object and phase/loss paths are validated against a
canonical full-plane reference for the same four scan sizes, and `_extract_gqk`
is tested to keep only the nonredundant columns. Default constructor-to-
`result()` reference agreement is tested against a test-only full-plane expansion of the
same half-plane, including the diagnostic loss. Focused CUDA check from
2026-07-17 on GPU1:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q \
  tests/test_ssb_cuda_128.py -k 'hermitian or phase_loss'

14 passed, 14 deselected
```

Full CUDA SSB test file from the same pass:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q tests/test_ssb_cuda_128.py

28 passed
```

MPS Hermitian half-plane checks from a Mac MPS machine:

```text
cd /path/to/quantem.gpu
PYTHONPATH=src python -m pytest -q \
  tests/test_ssb_mps_cuda_reference.py

2 passed, 2 skipped
```

The skipped MPS cases are optional real-data CUDA-reference checks that
require local `QUANTEM_GPU_SSB_MASTER` / `QUANTEM_GPU_SSB_REFERENCE_NPZ`
fixtures. The executed checks cover supported sparse row masks through
`1024x1024` and exact Hermitian half-plane expansion against `fft2`.

The `1024x1024` reconstruct-with-loss path is validated against an explicit CuPy
reference with a tolerance that allows rare `atan2` branch-cut pixels while
requiring the scalar objective and 99.9% of phase pixels to match.

Run before changing SSB kernels:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
pytest -q tests/test_ssb_cuda_128.py
```

## Repeat the native benchmark

Use the local Codex skill benchmark for synthetic kernel timing:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
python quantem-ssb-kernel-optimization/scripts/ssb_native_bench.py \
  --repo . \
  --sizes 128,256,512,1024 \
  --num-bf 8809 \
  --iters 4 \
  --mode object
```

Repeat with `--mode phase` and `--mode loss` for the full 12-run matrix. Do not
validate object-mode FPS to phase-mode optimizer FPS without saying which
scientific quantity is being drawn.

## Rejected 512 phase/loss probes

Follow-up GPU1 probes after the accepted `~45-47 ms` exact `512x512`, `8809`
BF path did not produce another accepted kernel change. Keep these as negative
evidence so the next pass starts at the remaining bottleneck instead of
repeating local minima.

| Probe | Result | Decision |
| --- | ---: | --- |
| Quadratic C10/C12 gamma fast branch inside the aperture | Reference agreement passed, but row p50 stayed about `30.6 ms`; full loss p50 was about `46.8 ms`. | Rejected: branch/scheduling pressure erased the special-math savings. |
| Precomputed `sign(sin(Q(q)))` plus BF row/column phase tables | Reference agreement passed, but full loss p50 worsened to about `48.3 ms`. | Rejected: extra table loads were worse than recomputing gamma. |
| Column BF group `64 -> 128` | Reference agreement passed, full loss p50 about `46.8 ms`. | Rejected: atomics/group count is not the dominant cost. |
| Adaptive/full 8809-BF staging chunk | Component total improved slightly, but engine p50 improved only about `0.1 ms` while staging memory rose to about `37 GB`. | Rejected as a default: too much memory for negligible wall-time gain. |
| Coalesced radix-8 row output plus legacy column reader | Row p50 dropped to about `23.8 ms`, but column p50 rose to about `22.3 ms`; total about `46.8 ms`. | Rejected: legacy column path gives back the row-store win. |
| Coalesced radix-8 row output plus radix-8 normal-layout column reader | Column p50 rose to about `28.3 ms`; total about `52.8 ms`. | Rejected: strided column reads dominate. |
| Coalesced radix-8 row output plus tiled explicit transpose | Reference agreement passed, but full loss p50 worsened to about `66 ms`. | Rejected: explicit full transpose costs far more than the row-store savings. |
| Degree-2 column `atan2` polynomial | Focused reference agreement passed, full loss p50 improved only about `0.1 ms`. | Rejected: too little speedup for a rougher scientific approximation. |
| Fixed 64-BF variant of `ifft512_rows_var_radix8_t64` | Full CUDA reference agreement passed, but real held-out no-pair loss p50 regressed to `36.09 ms` from the `~35.97 ms` short-run baseline. | Rejected: halving the group count/atomics reduced useful parallelism enough to erase the savings. |
| Dual row launch bound relaxed from `__launch_bounds__(256, 4)` to `256,3` | Full CUDA reference agreement passed, but real held-out no-pair loss p50 regressed to `36.79 ms`. | Rejected: the compiler freedom did not overcome the row-stage scheduling/shared-memory floor. |
| No-pair dual partial-plane reduction instead of direct atomics | Scratch profile p50 moved from about `13.4 ms` direct column accumulation to about `14.0 ms` with partial reduction plus summing. | Rejected: extra global writes/reduction cost more than atomics for this path. |
| Same-row dual `kx` reuse branch | Full CUDA reference agreement passed and `4351/4411` real held-out dual pairs shared detector row, but real loss p50 stayed about `36.6 ms` and row p50 stayed about `21.8 ms`. | Rejected: saved scalar `dx` work was too small and added branch/register pressure. |
| One-shared-buffer arbitrary dual row IFFT | Focused CUDA reference agreement passed, but row p50 regressed from about `21.6 ms` to `24.2 ms`; sustained real held-out loss regressed to `38.78 ms` p50 (`25.8 FPS`). | Rejected: halving shared memory serialized the A/B row IFFTs and added barriers/stores, so occupancy pressure was not the dominant floor. |
| Phase partial planes plus scalar loss (`use_partial=3`) | Full CUDA reference agreement passed, but real held-out loss regressed to `35.46 ms` mean / `35.44 ms` p50 in a 240-step run. | Rejected: replacing phase atomics with partial writes plus a separate reduction costs more than the current in-kernel atomics. |
| Analytic `qy` index formula in the dual row kernel | Full CUDA reference agreement passed and a 240-step run was neutral (`35.03 ms` mean), but the 600-step sustained run regressed to `35.66 ms` mean / `35.66 ms` p50. | Rejected: `qy_1d` cached loads are not a durable row-stage bottleneck. |
| Double-zero aperture early return before `sincos` | Full CUDA reference agreement passed, but real held-out loss regressed to `35.28 ms` mean / `35.28 ms` p50. | Rejected: the exact branch and code pressure cost more than the skipped outside-aperture work for this dataset. |
| Column launch bounds relaxed from `__launch_bounds__(64, 8)` to `64,6` | Full CUDA reference agreement passed, but real held-out loss regressed to `35.60 ms` mean / `35.61 ms` p50. | Rejected: the column kernel still wants the current occupancy constraint. |
| Column staged-data loads through `ld_float2`/read-only path | Full CUDA reference agreement passed, but real held-out loss regressed to `35.69 ms` mean / `35.66 ms` p50. | Rejected: the staged row output does not benefit from the read-only cache path. |
| Column no-prefetch schedule | Full CUDA reference agreement passed, but real held-out loss regressed to `35.49 ms` mean / `35.48 ms` p50. | Rejected: lower register lifetime did not beat the lost global-load overlap. |
| PTX cache policy `-dlcm=ca -> -dlcm=cg` | Full CUDA reference agreement passed, but real held-out loss regressed to `35.76 ms` mean / `35.72 ms` p50. | Rejected: L1 caching is still the better default for this mixed row/column path. |
| Column BF group `32 -> 16` | Full CUDA reference agreement passed, but real held-out loss regressed to `35.92 ms` mean / `35.91 ms` p50. | Rejected: extra phase-atomic groups outweighed lower loop pressure. |
| Hermitian-only duplicate arbitrary-dual row kernel | Full CUDA reference agreement passed, but real held-out loss regressed to `35.80 ms` mean / `35.80 ms` p50. | Rejected: removing the runtime `gqk_cols` branch increased code footprint/register pressure enough to lose. |
| Two-stream row/column chunk overlap prototype | Exact BF chunks ran with two staging buffers, but best chunked p50 was still about `35.15 ms` for row+column work only. | Rejected as a breakthrough path: the kernels do not overlap enough on one GPU to close the budget. |
| 8-row arbitrary-dual row block with 64 KB dynamic shared memory | Small focused reference agreement passed, but the real full-BF held-out dataset launch hit `CUDA_ERROR_ILLEGAL_ADDRESS` inside the row kernel under `CUDA_LAUNCH_BLOCKING=1`. | Rejected: the larger row-store topology is unsafe on the target grid and must not be carried without a fresh memory-correct redesign. |
| 6-row arbitrary-dual row block with static shared memory | Focused CUDA reference agreement passed, but real held-out loss regressed to `37.21 ms` mean / `37.20 ms` p50 (`26.9 FPS`) in a 240-step run. | Rejected: the larger block lowered scheduling efficiency and did not recover enough row-stage throughput. |
| Global module register cap tightened from `96` to `80`/`72` | Focused CUDA reference agreement passed. Real held-out loss measured `35.32 ms` mean at `80` and `35.44 ms` mean at `72`. | Rejected: forcing lower register allocation did not overcome the column occupancy limiter and likely traded occupancy for spills/scheduling pressure. |

Accepted 2026-07-17 breakthrough: native `512x512` exact phase/loss now uses
64-BF staging chunks by default. This keeps the row-IFFT producer/consumer
working set small instead of writing and rereading one `18+ GB` intermediate.
The scientific contract is unchanged: same full active BF disk (`8822` BF on
the central held-out dataset field), same Hermitian `G_qk`, same per-BF phase/loss
arithmetic, no binning, no crop, and no preview/settle split. Focused CUDA
reference agreement passed (`25/25`).

Sustained real held-out GPU1 result after the default change:

| Mode | Steps | Mean | p50 | p95 | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase+loss | 600 | `32.49 ms` | `32.15 ms` | `33.31 ms` | `30.78 FPS` |
| Phase-only | 240 | `31.94 ms` | `31.93 ms` | `32.08 ms` | `31.31 FPS` |

Earlier full-buffer component timing was approximately `22.6 ms` row/gamma,
`12.3 ms` column/phase, and `<0.2 ms` finalization. The new chunking win is not
from changing gamma algebra or scalar loss; it is from making the producer/
consumer memory working set smaller. GPU1 remains a 300 W card and sustained
runs can still show `pviol=100%`, so the p95 margin around the 30 FPS frame
budget should be watched on other machines.

Current conclusion: the `512x512` exact full-BF phase/loss target is met on the
real central field. The next structural performance target is
`1024x1024`, where the same exact phase/loss path still needs a larger topology
change, likely fused/tiled row-column work or another exact formulation.

### 512 full-BF calibration timing follow-up

The same real `512x512` field was used to test the full calibration workflow,
not only live redraw. Source: held-out HDF5 master file loaded as full
scan/full detector `uint16`, `9070` active BF after the fitted
aperture, Hermitian `G_qk=(9070,512,257)` / `9.55 GB`.

The important workflow split is:

- Live interaction uses the exact full-IFFT `reconstruct_with_loss()` path.
- Default historical calibration uses sparse `variance_loss_batch()`, a
  different scalar objective.
- Forced exact calibration uses the exact full-IFFT path for optimizer and
  refiner objective evaluations.

Accepted calibration improvement:

| Stage | Before | After | Notes |
| --- | ---: | ---: | --- |
| Exact 200-trial optimize | `~6.7 s` | `~6.7 s` | Still bounded by `~32-33 ms` per exact full-BF candidate. |
| Exact Nelder-Mead refine | `~7.1 s / 200 evals` in the earlier forced path | `1.12 s / 34 evals` | Exact full-BF objective; default exact-fallback tolerances now stop before the invisible flat tail. |
| Load -> Gqk -> optimize -> refine -> widget | `17.05 s` | `11.32 s` | Full BF, no binning, no crop, no trial-count reduction. |

The new exact-refine default for this fallback path was chosen from a sweep:
`xatol=0.25`, `fatol=2e-6`, `max_iter=160`. Versus the longer exact
baseline, the selected policy gave loss delta about `4-6e-7` and phase deltas
around `5.7e-5 rad` mean / `3.9e-4 rad` p99.9 in the real-data probe. The
too-loose four-evaluation policies were rejected because they produced
`~0.0014 rad` mean and `~0.01 rad` max phase deltas.

Rejected calibration hypotheses:

| Hypothesis | Measurement | Decision |
| --- | --- | --- |
| Host-sync-free GPU scalar losses for exact fallback | Reference agreement passed (`2.2e-8` max scalar-loss delta), but batch-4 exact stayed about `130.6 ms` (`32.6 ms/candidate`). | Rejected and removed from production code: no measurable throughput win for the added complexity. |
| Concurrent exact candidates on separate CUDA streams | Four candidates were slower concurrently (`140.9 ms`) than sequentially (`129.7 ms`). | Rejected: kernels contend for the same shared-memory/scheduler resources. |
| Larger exact BF chunks for calibration | Real-data sweep showed `32/64` BF chunks at `~32.0-32.1 ms`; `96+` chunks regressed to `35-36 ms`. | Keep `64` as the default; larger chunks are not a calibration win. |
| Fewer exact Optuna trials | `150` trials + exact refine reached `6.72 s` optimize+refine with `0.00054 rad` p99.9 phase delta versus the 200-trial baseline. | Useful evidence for an opt-in fast-calibration mode, but do not present it as the full 200-trial default. |

Nsight on the exact 512 row kernels confirms the next kernel-level ceiling:
`ifft512_rows_fused_pk_dual_radix8_t64_packed` runs at about 50% theoretical
occupancy, limited by shared memory, with about `49%` no-eligible scheduler
cycles and MIO/short-scoreboard stalls. The column phase/loss accumulator
`ifft512_rows_var_radix8_t64` is register-limited, reaches about 20% achieved
occupancy in the small chunk launch, and also shows scheduler starvation. The
next kernel breakthrough therefore needs a different row/column topology or a
reference-checked exact multi-candidate formulation, not bigger chunks or streams.

### Hermitian-only and MPS matrix follow-up

The 2026-07-17 follow-up made Hermitian `G_qk` the only public runtime storage
mode. `SSB(..., gqk_storage="full")` now raises a corrective `ValueError`.
Full-plane `G_qk` remains available only as a test-only canonical expansion of
the Hermitian half-plane, so reference agreement references are stable without carrying
redundant FFT roundoff as a separate public mode.

CUDA validation after the removal:

```text
env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q \
  tests/test_ssb_cuda_128.py tests/test_ssb_batch_optuna.py

29 passed in 3.91s
```

The `128x128` high-BF object path currently uses the exact fused-IFFT fallback
when `num_bf > 1024`. The small-BF Fourier-sum kernel remains reference-checked,
but a high-BF synthetic stress probe left the CUDA context in an illegal-
address state. The fallback keeps the user path exact and fast at `128x128`
while that microkernel is investigated.

CUDA synthetic Hermitian timing from this pass:

| Scan | Mode | BF | Resident `G_qk` | Mean | p50 | p95 | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `128x128` | object, fused-IFFT fallback | `8809` | `586 MB` | `4.81 ms` | `4.70 ms` | `5.08 ms` | `208.1` |
| `256x256` | object | `8809` | `2.33 GB` | `1.74 ms` | `1.73 ms` | `1.78 ms` | `575.2` |
| `512x512` | object | `8809` | `9.27 GB` | `6.97 ms` | `6.82 ms` | `7.30 ms` | `143.5` |
| `1024x1024` | object | `8809` | `37.02 GB` | `41.79 ms` | `42.32 ms` | `43.93 ms` | `23.9` |
| `1024x1024` | phase-only, split-512 row/column | `8809` | `37.02 GB` | `195.54 ms` | `194.99 ms` | `200.84 ms` | `5.11` |
| `1024x1024` | phase+loss, split-512 row/column | `8809` | `37.02 GB` | `197.71 ms` | `199.00 ms` | `199.24 ms` | `5.06` |
| `128x128` | phase+loss | `8809` | `586 MB` | `7.86 ms` | `7.80 ms` | `7.99 ms` | `127.2` |
| `256x256` | phase+loss | `8809` | `2.33 GB` | `18.31 ms` | `18.20 ms` | `18.51 ms` | `54.6` |
| `512x512` | phase+loss | `8809` | `9.27 GB` | `26.98 ms` | `27.11 ms` | `27.12 ms` | `37.1` |

The `1024x1024` exact phase/loss path now uses a split-512 row and column IFFT:
each 1024 IFFT is decomposed into exact even/odd 512-point radix-8 transforms
plus a final radix-2 combine. The row kernel writes transposed scratch, and the
column kernel consumes that layout directly. This preserves the default math
and focused CUDA reference agreement while cutting the synthetic full-BF phase+loss time
from `382.24 ms` to `197.71 ms`.

The current `1024x1024` exact phase/loss default still uses a `1024` BF staging
chunk for high-BF runs. Chunk sweeps from `512` to `4096` BF measured about
`197-200 ms`; BF-group sweeps kept `32` BF as the most stable column
accumulation group. Those knobs reduce memory footprint or stabilize the run,
but the next FPS win must come from the register-heavy row/gamma path or a new
exact formulation.

MPS validation and timing were run on a Mac MPS machine with MLX `0.32.0`.
Prepared MPS `G_qk` now stores the same Hermitian half-plane. The cached sparse
objective Metal kernel fetches missing columns by mirror-conjugate symmetry,
and the non-cached MLX preview path expands only the active chunk.

Mac MPS test:

```text
PYTHONPATH=src python -m pytest -q \
  tests/test_ssb_mps_cuda_reference.py

2 passed, 2 skipped in 1.29s
```

Mac MPS synthetic prepared-data timing:

| Scan | BF | Resident `G_qk` | Prep | Sparse objective mean / p95 / FPS | Exact preview mean / p95 / FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `128x128` | `128` | `8.52 MB` | `0.163 s` | `3.60 ms` / `4.25 ms` / `278` | `4.02 ms` / `4.40 ms` / `248` |
| `256x256` | `96` | `25.36 MB` | `0.036 s` | `10.25 ms` / `11.08 ms` / `97.5` | `10.68 ms` / `11.36 ms` / `93.6` |
| `512x512` | `64` | `67.37 MB` | `0.067 s` | `28.27 ms` / `30.67 ms` / `35.4` | `33.26 ms` / `33.61 ms` / `30.1` |
| `1024x1024` | `24` | `100.86 MB` | `0.120 s` | `44.66 ms` / `48.79 ms` / `22.4` | `50.39 ms` / `55.98 ms` / `19.8` |
| `512x512` pushed | `96` | `101.06 MB` | `0.166 s` | `44.18 ms` / `45.90 ms` / `22.6` | `49.71 ms` / `50.20 ms` / `20.1` |
| `1024x1024` pushed | `30` | `126.07 MB` | `0.092 s` | `54.27 ms` / `55.75 ms` / `18.4` | `61.05 ms` / `61.58 ms` / `16.4` |

Interpretation for Mac users: MPS now supports the native size matrix for the
prepared SSB path and is interactive for moderate BF counts, especially at
`128/256/512`. This is not yet a full-BF `1024x1024` Mac signoff. Report the
BF count with every MPS FPS number because Mac unified memory and MLX FFT cost
scale directly with the number of active BF columns.

### MPS full-BF 512 follow-up

The next MPS pass targeted Mac no-crop/no-bin review on the real `512x512`
logic dataset:

```text
held-out HDF5 master file
native shape: (262144, 192, 192), uint16
BF policy: threshold=0.0, bf_radius=53
selected BF: 8827
Hermitian G_qk: (8827, 512, 257), 9.29 GB
```

Accepted MPS changes:

- Added batched `ChunkedFrames.columns(rows, cols)` so SSB preparation reads BF
  columns in groups instead of calling `column()` once per detector pixel.
- Changed the non-prefix MPS column gather to fill scan-major chunks with
  `np.take` from the flattened detector grid, then enabled threaded extraction
  over independent Metal chunks for large BF selections. This keeps the same
  raw detector evidence and only changes the host extraction topology.
- Added a fused dynamic Metal correction kernel for large-BF prepared paths.
  It computes the same C10/C12/phi12 correction and fetches missing Hermitian
  columns directly, avoiding expanded `G_qk` and large MLX geometry
  temporaries.
- The Metal compute backend reconstructs the exact BF-averaged object wave by
  summing in Fourier space before one inverse FFT. Exact fitting and optional
  loss evaluation continue to use the mean-of-per-BF phase-variance quantity.
- Precomputed the BF-only probe term `p(k)` once per live aberration setting
  and passed it into the object Fourier-sum Metal kernel. The kernel still
  computes the `q-k` and `q+k` terms per pixel, but it no longer recomputes the
  same BF-only probe phase for every output pixel.
- Split object-mode chunking into separate setup and redraw defaults:
  `QUANTEM_MPS_SSB_OBJECT_CHUNK_BF` controls first-use BF FFT setup, while
  `QUANTEM_MPS_SSB_OBJECT_REDRAW_CHUNK_BF` controls repeated live redraw. On
  the Mac MPS machine, setup is fastest at `1024` BF chunks, but synchronized live redraw is
  fastest with the object redraw default `128`; tying those together regressed
  interaction timing.
- Added `QUANTEM_MPS_SSB_OBJECT_THREADGROUP` for repeated object redraw and
  set the high-memory Mac default to a `16`-thread Metal threadgroup after launch-shape
  sweeps.
- Replaced separate object-kernel `sin`/`cos` calls with
  `metal::fast::sincos` for the two shifted aperture phases. Mac MPS reference agreement
  stayed green, and this was the decisive redraw speedup.
- Added a 512-only fused Metal column-IFFT + phase/loss accumulator for the
  prepared MPS exact mean-phase path. The route still uses the same full BF
  disk and the same float32 per-BF phase/loss definition, but it no longer
  materializes the full per-BF object chunk before summing phases.
- Added a 512-only fused dynamic correction + row-IFFT Metal kernel for the
  prepared MPS exact mean-phase path. This removes the MLX corrected-plane
  materialization and MLX row-IFFT from the large-BF loop while preserving the
  same Hermitian `G_qk`, BF disk, and float32 phase/loss definition.

Real Mac MPS before/after:

| Path | Before | After | FPS after | Status |
| --- | ---: | ---: | ---: | --- |
| Full no-bin MPS load | not retimed in same harness | `0.94 s` final run, earlier `1.05-1.85 s` | n/a | Pass. |
| BF select | not retimed in same harness | `0.28-0.33 s` | n/a | Pass. |
| Prepare Hermitian `G_qk` | `22.67 s` | `3.21 s` final run, earlier `2.91-3.04 s` | n/a | `~7x` faster than the old per-column setup. |
| Public object-mode preview after load | `24.84 s` | about `3.3-3.5 s` | n/a | `~7x` faster first review, no BF reduction. |
| Repeated object-wave redraw | `~211 ms` scratch loop; first optimized pass `50.6 ms` mean, p95 `65.0 ms` | mean `24.01 ms`, p50 `23.85 ms`, p95 `26.34 ms` | `41.65 FPS` | Exact object-wave path passes the 30 FPS target after warm-up. |
| Repeated exact phase+loss redraw | `710 ms`; pre-column-fusion best `~550-640 ms`; fused-column-only best `335.18 ms` mean | best `169.62 ms` mean, `169.68 ms` p50, `170.24 ms` p95 | `5.90 FPS` | Fused correction+row plus fused column is exact and about `2x` faster than the fused-column-only path, but still fails the 30 FPS live target. |

Latest setup split with `chunk_bf=1024`:

```text
load_s=1.05, bf_select_s=0.33
prepare_total_s=2.91
  gather_s=1.49
  cpu_to_mlx_stage_s=0.48
  rfft2_eval_s=0.54
  concat_s=0.30
selected_bf=8827
```

Final default-path object redraw timing:

```text
load_s=1.196, prepare_s=3.230
selected_bf=8827
Hermitian G_qk=(8827, 512, 257)
object redraw defaults: chunk_bf=128, threadgroup=16
best measured chunk_bf=256:
  mean_ms=21.82, p50_ms=21.88, p95_ms=23.16, fps=45.82
default chunk_bf=128:
  mean_ms=22.21, p50_ms=22.19, p95_ms=22.98, fps=45.03
shared SSB reconstruction after load:
  wall_s=3.436, loss=None
repeated shared object reconstruction after load:
  wall_s=3.257, loss=None
```

Final exact mean-phase/phase-loss redraw timing after fused correction+row and
fused column accumulation:

```text
load_s=1.196, prepare_s=3.230
selected_bf=8827
Hermitian G_qk=(8827, 512, 257)
phase-only best chunk_bf=1024:
  mean_ms=168.77, p50_ms=168.52, p95_ms=169.86, fps=5.93
phase+loss best warmed chunk_bf=3072:
  mean_ms=165.20, p50_ms=165.57, p95_ms=166.71, fps=6.05
component split at chunk_bf=1024:
  correction+row-IFFT mean=74.30 ms
  column-IFFT+phase/loss mean=93.76 ms
  accumulation mean=2.87 ms
exact mean-phase diagnostic after load:
  wall_s=4.281, loss=None
exact phase-variance loss after load:
  latest default chunk_bf=3072 wall_s=4.751, loss=0.050665274262428284
```

Current Apple Silicon MPS BF30 refresh, 2026-07-21:

```text
source: anonymized real 512 HDF5 master
shape: (512, 512, 192, 192), uint16
BF policy: bf_radius=30, threshold=0.0
selected BF: 2829
load_s=1.248, prepare_s=1.112
```

Warmed resident prepared-state timing, no full phase readback in the timing
loop:

| Quantity | Mean | p50 | p95 | FPS | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Object redraw | `10.33 ms` | `10.71 ms` | `11.24 ms` | `96.8` | Passes 30 FPS for object-wave steering. |
| Exact mean phase | `75.96 ms` | `75.84 ms` | `78.00 ms` | `13.2` | Better than the older full-BF high-radius result, but still below 30 FPS. |
| Exact phase+loss | `77.27 ms` | `77.18 ms` | `78.46 ms` | `12.9` | Usable for settle/refinement checks, not 30 FPS live dragging. |

Public API first-use timing on the same source, including preparation inside
each call, was `1.85 s` object, `3.35 s` exact phase, and `5.41 s` exact
phase+loss after a `0.92 s` full MPS HDF5 load. Treat those as first-review
latency, not warm interaction timing.

Follow-up on 2026-07-17 changed `_default_phase_loss_chunk_bf()` to return
`3072` on 96 GB-class Macs, `1024` on 64 GB-class Macs, and `512` otherwise.
A 2026-07-18 repeat moved the 96 GB-class default to `4096` BF after the
single-candidate sweep above. The public real
the Metal reconstruction with exact loss and `chunk_bf=16` completed with
`8827` BF and `512x512` phase/amplitude in `4.75 s` after a `0.97 s` full
MPS load on the earlier default. The prepared steady-state exact phase+loss
redraw is still only a few FPS, so do not present this as a solved 30 FPS MPS
phase/loss path.

Real full-BF reference agreement against the previous MLX row-IFFT + fused-column path:

```text
loss_old=0.021938618272542953
loss_new=0.021938620135188103
phase_max_abs=1.4901161193847656e-08
phase_mean_abs=1.894959078541092e-09
```

### MPS native full-BF matrix follow-up, 2026-07-18

This pass extended the exact fused Metal phase/loss topology beyond the earlier
512-only path. It keeps the same full active BF-style evidence, native scan
size, Hermitian complex64 `G_qk`, and float32 mean-phase/loss definition. No
scan crop, detector binning, BF subsampling, preview/settle split, or derived
float/complex cache is used.

Accepted MPS changes:

- Added fused 128/256 Metal column-IFFT + phase/loss accumulation.
- Added fused 128/256 Metal dynamic correction + row-IFFT.
- Added fused 1024 Metal dynamic correction + row-IFFT and column-IFFT +
  phase/loss accumulation.
- Fixed the top-level fused MPS loss accumulator so all fused exact sizes
  (`128/256/512/1024`) use scalar `phase^2` accumulation instead of
  accidentally broadcasting the scalar loss tile through an image-shaped
  accumulator.
- Changed the 128/256 MPS exact phase/loss default chunk to a large
  full-BF-capable chunk on 96 GB-class Macs. This removes avoidable loop and
  partial-reduction overhead without changing BF evidence.
- Changed the 1024 MPS exact phase/loss default chunk to `512` BF after the
  scalar-loss fix. Isolated sweeps show `512/768` are close; the default keeps
  the smaller, safer working set.
- Changed 128/256 phase-only column grouping to 8 columns per Metal
  threadgroup. Phase+loss keeps 4 columns because 8 columns regressed the loss
  path in repeated timing.
- Changed repeated MPS object redraw to use a `64`-threadgroup default for
  `512x512` and larger scans after a small launch-shape sweep.
- Added `QUANTEM_MPS_SSB_PHASE_COL_K_BF` as an internal tuning knob. Sweeps
  from `8` to `128` BF did not produce a durable 512 breakthrough, so the
  default remains `32`.

Mac MPS reference check:

```text
PYTHONPATH=src python -m pytest -q tests/test_ssb_mps_cuda_reference.py

20 passed, 2 skipped
```

MPS synthetic prepared full-BF-style matrix, Hermitian `G_qk`, `8809` BF.
The 512/1024 rows were refreshed on the Apple Silicon reference host from the
current source tree on 2026-07-21 after rerunning the MPS path from `origin/main`:

| Scan | Object mean / FPS | Phase mean / FPS | Phase+loss mean / FPS | Notes |
| --- | ---: | ---: | ---: | --- |
| `128x128` | `2.45 ms / 408.4` | `~8.0 ms / 122-126` | `~8.3 ms / 119-121` | Exact fused row/column path passes 30 FPS. |
| `256x256` | `8.62 ms / 116.1` | `32.75 ms / 30.5` | `~34-35 ms / 28.6-29.4` | Phase-only reaches 30 FPS; phase+loss remains just above the strict `33.3 ms` budget. |
| `512x512`, radius `30` px | `10.86 ms / 92.1` | `76.67 ms / 13.0` | `76.28 ms / 13.1` | Fixed-BF policy is reviewable for exact phase/loss but still not 30 FPS. |
| `512x512`, full active BF | `55.20 ms / 18.1` | `481.15 ms / 2.1` | `528.90 ms / 1.9` | Full active BF is correct but not interactive; object-wave steering is a different quantity. |
| `1024x1024` | `142.7 ms / 7.0` | not rerun separately at full BF | `669.1 ms / 1.5` | Full-BF-sized synthetic `G_qk` (`37.02 GB`) remains far from 10/30 FPS. A reduced-BF topology probe at `1382` BF measured `~104 ms`, showing BF-count scaling is the dominant cost. |

Before/after for the exact MPS phase/loss paths in the same synthetic prepared
full-BF-style harness:

| Scan | Quantity | Before | After | Speedup | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `128x128` | phase | `38.70 ms` | `~8.0 ms` | `~4.8x` | Passes 30 FPS. |
| `128x128` | phase+loss | `37.33 ms` | `~8.3 ms` | `~4.5x` | Passes 30 FPS. |
| `256x256` | phase | `144.98 ms` | `32.75 ms` | `4.4x` | Passes 30 FPS by mean; p95 remains close to budget. |
| `256x256` | phase+loss | `146.51 ms` | `~34-35 ms` | `~4.2x` | Near miss for 30 FPS. |
| `512x512`, radius `30` px | phase+loss, current prepared path | older generic path `~550-640 ms` | fresh Apple Silicon `origin/main` probe `76.28 ms` mean / `77.41 ms` p95 | `~7x` versus the older generic path for the radius-30 policy | Reviewable, still slower than CUDA. |
| `512x512`, full active BF | phase+loss, current prepared path | older generic path not directly comparable | fresh Apple Silicon `origin/main` probe `528.90 ms` mean / `557.51 ms` p95 | not recorded | Still fails 30 FPS and remains much slower than CUDA. |
| `1024x1024` | phase | `1994 ms` | not rerun separately at full BF | not recorded | Still fails 10/30 FPS. |
| `1024x1024` | phase+loss | `1984 ms` | fresh full-BF-sized source-tree probe `669 ms` | `~3.0x` | Still fails 10/30 FPS. |

Rejected or non-breakthrough MPS probes from this pass:

| Probe | Result | Decision |
| --- | --- | --- |
| 512 exact phase/loss chunk sweep down to `64` BF | Small CUDA-like chunks regressed (`~226 ms` at `64` BF). Larger chunks stayed best, and the latest single-candidate repeat favored `2048-4096` BF. | Keep the high-memory 512 default at `4096` BF. MPS benefits from fewer loop launches here. |
| 512 ShowPtycho adapter object-wave accumulation | The adapter must not ask `_reconstruct_prepared(..., compute_object=True)` for exact phase/loss interactions because object-wave output is a separate quantity. A fresh Apple Silicon `origin/main` probe measured radius-30 phase-only/phase+loss at `~76 ms`, while the full-active-BF policy measured `~481/529 ms`. | Keep `compute_object=False` for exact phase/loss, keep BF policy explicit, and do not claim a 512 exact-phase 30 FPS breakthrough. Object-wave steering remains the fast MPS interaction path. |
| 512 column BF grouping `8/16/32/64/128` | Best cases moved only `1-2 ms`; larger groups regressed. | Not a topology breakthrough; keep default `32`. |
| 1024 fused chunk sweep `128/256/512/768/1024` after scalar-loss fix | `512/768` were close and better than the earlier `256` cap in isolated runs, but order and thermal state moved the result by hundreds of ms. | Set 1024 default to `512` for a smaller safe working set; this remains far from interactive. |
| Object threadgroup sweep `8/16/32/64/128` | `64/128` modestly improved large-object redraw. | Keep `64` for `512+`; it is a small object-mode win, not a phase/loss breakthrough. |
| 128/256 in-kernel scalar-loss reduction | Reference agreement passed, but 256 phase+loss regressed to about `34.5 ms`. | Reverted. Smaller loss outputs did not beat the extra threadgroup barriers. |
| 128/256 8-column Metal grouping | Phase-only improved enough for 256 to reach about `32.75 ms`, but phase+loss regressed. | Use 8 columns only for phase-only; keep phase+loss at 4 columns. |

Interpretation for the microscopist: on an Apple GPU, exact full-BF
mean-phase/loss is now real-time at `128x128`, very close at `256x256`, still
review-only at `512x512`, and not live-interactive at `1024x1024`. The 1024
fused Metal path proves the generic MLX FFT route was a major bottleneck, but
the remaining wall time is still per-BF exact phase work. The next MPS
breakthrough needs a different exact 512/1024 row-column topology or a
scientifically equivalent reformulation, not BF reduction.

Rejected MPS probes from the same pass:

| Probe | Result | Decision |
| --- | --- | --- |
| Direct one-threadgroup-per-Fourier-pixel reduce | Reference agreement passed, but real full-BF timing was `~103-168 ms` depending on thread count because it destroyed useful parallelism. | Removed from production code. |
| Lazy aperture branch that moved astigmatism/trig work after support checks | Reference agreement passed, but real timing regressed because the extra branching hurt Metal occupancy. | Reverted. |
| Old redraw launch defaults `chunk_bf=48/64`, threadgroup `64/256` | Worked, but sustained real-data redraw stayed roughly `18-24 FPS`. | Replaced by `chunk_bf=128`, threadgroup `16`. |
| Phase-only sum kernel that skipped `sumsq` | Reference agreement passed, but phase-only still measured about `334-348 ms` because the inverse FFT and correction dominate. | Kept only where it is simple; do not expect it to be the MPS breakthrough. |
| Column BF grouping `k_bf=8/16/32/64/128/256` after fused row | Real full-BF column+accumulation stayed best around `k_bf=16/32` at `~92 ms`; larger groups reduced partial outputs but lost useful BF parallelism. | Keep `k_bf=32`; the next exact breakthrough needs a different row-column topology, not a grouping constant. |
| CUDA 1024 direct atomic phase/loss accumulation | Focused reference agreement passed, but `1024x1024`, `1382` BF loss regressed to `210.8 ms` mean before reverting; the known partial-sum path measured `62.0 ms` in the same condition. | Removed. Direct atomics are not the 1024 breakthrough. |
| CUDA 1024 column grouping `k_bf=8` | The isolated column kernel moved slightly, but full `8809` BF loss regressed from `406.5 ms` to `427.3 ms`. | Keep `k_bf=32`. |
| CUDA 1024 Cartesian row correction helper | Focused reference agreement failed the 1024 CuPy reference gate (`99.9%` phase error `3.26e-4` vs `3e-4`). | Reverted. Keep exact reference agreement. |
| CUDA 1024 launch bounds on row/column kernels | Focused reference agreement passed, but full `8809` BF loss regressed to `859 ms`. | Reverted. Occupancy pressure is not solved by forcing launch bounds. |
| CUDA 1024 polynomial `atan2` in column phase/loss | Focused reference agreement passed, but warmed full `8809` BF loss was noise-level (`389.0 ms` vs `389.5 ms` same-session baseline). | Rejected as a non-breakthrough. |
| CUDA 1024 specialized Hermitian `G_qk` fetch helper | Focused reference agreement passed, but default full `8809` BF timing did not improve (`380.84 ms` phase, `382.67 ms` loss). | Reverted; the generic fetch is not the bottleneck. |
| CUDA 1024 split-512 launch bound tightened from `64,8` to `64,12` | Focused reference agreement passed, but component timing regressed from about `15.2 ms` row / `6.9 ms` column to about `17.0 ms` row / `11.5 ms` column per 1024-BF chunk. | Reverted. The higher-register compiler choice is faster despite lower occupancy. |
| CUDA 1024 coalesced-load split row | Focused reference agreement passed, but warmed full `8809` BF loss stayed about `200-201 ms`, slower than the direct split row at about `197-199 ms`. | Removed. The extra shared gather and synchronization cost more than the improved source-load order. |

The object-mode reference agreement guard checks the fused object Fourier-sum kernel
against the looped corrected-object reference on a Mac MPS machine:

```text
PYTHONPATH=src python -m pytest -q \
  tests/test_ssb_mps_cuda_reference.py

5 passed, 2 skipped
```

Interpretation for the microscopist: on an Apple GPU, a full-BF `512x512`
field can now load no-bin data, build the BF evidence, and show an
exact object-wave phase review a few seconds after load, then steer the object
view above `40 FPS` after warm-up on a Mac MPS machine. The stricter exact
mean-phase/phase-variance loss view improved from about `550-640 ms` to about
`335 ms`, so it is better but still not usable for live full-BF steering on
MPS. The next real MPS breakthrough has to fuse the correction + row FFT side
of the exact phase/loss path or port the CUDA row/column topology more fully;
another BF-column gather or UI flag will not close the remaining budget.

### MPS exact objective default

The next MPS pass changed Metal fitting from the CUDA-shaped sparse-row
objective to the exact full active-BF phase-variance objective, then added a
candidate-batched exact-loss evaluator for the Optuna phase. The sparse
reference implementation was removed after this comparison; MPS now has one
fit objective, matching the final full-BF phase reconstruction.

Real MPS `512x512` HDF5 timing:

```text
native scan: 512x512
detector: 192x192
dtype: uint16
BF policy: threshold=0.0, bf_radius=53
selected BF: 8826
Hermitian G_qk: (8826, 512, 257), 9.29 GB
```

| Path | BF Policy | Mean / Wall Time | p50 | p95 | FPS / Rate | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| HDF5 load | full scan metadata-backed MPS load | `0.88-1.56 s` | not recorded | not recorded | n/a | Warm local SSD range |
| BF select | full active BF disk | `0.231-0.287 s` | not recorded | not recorded | n/a | Mean diffraction pattern path |
| Gqk construction | full active BF | `3.20-3.36 s` | not recorded | not recorded | n/a | Hermitian complex64, `9.29 GB` |
| Object redraw | full active BF | `22.46-30.50 ms` | `22.31-31.63 ms` | `23.66-35.48 ms` | `32.8-44.5 FPS` | Microscopist live steering path; machine thermal/load state moved the number |
| Exact loss candidate | radius `30` px BF | `76.28 ms` | `76.52 ms` | `77.41 ms` | `13.1 eval/s` | Single-candidate fused row-IFFT + scalar phase-loss on Apple Silicon `origin/main` |
| Exact loss candidate | full active BF | `528.90 ms` | `537.58 ms` | `557.51 ms` | `1.9 eval/s` | Same exact objective with many more BF pixels; not interactive |
| Exact loss batch, 2 candidates | full active BF | `320-428 ms` | `319-424 ms` | `326-452 ms` | `4.7-6.2 eval/s` | Same exact loss; batch-vs-single max abs error `0-3.7e-9` |
| Removed sparse loss | full active BF, sparse rows | `603.60 ms` | `598.09 ms` | `616.78 ms` | `1.66 eval/s` | Historical measurement; implementation removed |
| 200 trials + Nelder-Mead, previous exact default | full active BF | `61.44 s` | not recorded | not recorded | `274 evals` | Exact objective, final loss `0.051254` |
| 200 trials + Nelder-Mead, current direct first-use run | full active BF | `64.22 s` including `1.56 s` HDF5 load | not recorded | not recorded | `218 evals` | Exact objective, first-use Metal compile included, final loss `0.051252` |
| 200 trials + Nelder-Mead, current warmed-kernel run | full active BF | `54.53 s` fit after load | not recorded | not recorded | `218 evals` | Exact objective, same final loss `0.051252`; kernels/data path already warm |

Accepted MPS changes:

- Metal fitting has one full phase-variance objective, so Mac calibration cannot
  silently select a sparse-row objective.
- `optuna_batch_size=2` is the default for the exact MPS path because the real
  candidate sweep showed larger batches did not improve candidate throughput
  and increased memory pressure.
- The final phase pass now computes and reuses its exact scalar loss instead of
  running one redundant extra exact-loss evaluation.

Rejected MPS experiments:

- Generic exact batching with MLX `ifft2` measured only `~2.2-2.3` candidates/s
  at batch sizes `1`, `2`, and `4`, and therefore lost to the fused
  single-candidate exact kernel. Do not add generic exact batching unless the
  row-IFFT and phase-loss kernels are also fused over candidate batches.
- Transposed row-IFFT scratch for candidate batches preserved exact loss
  agreement (`max_abs=3.7e-9`) and sped up the column read, but slowed the row
  write enough that the full `200`-trial plus Nelder-Mead workflow regressed
  to about `70 s`. It was removed.

### MPS decode-side automatic-probe sum

The lossless uint8 MPS loader can optionally compute the exact detector sum
inside each fused decode command buffer. A second Metal command buffer merges
the independent source-chunk sums. The sum remains uint32 until the existing
mean-DP conversion, and the default loader remains unchanged unless
`precompute_detector_sum=True` is requested.

Canonical automatic full-field Reference-512 validation used the native
`512x512x192x192` acquisition, all `8937` logical BF pixels (`2464` active),
200 exact Optuna trials, unchanged Nelder-Mead refinement, float32/complex64,
and the normal row/column MPS objective:

| Quantity | Separate detector sum | Decode-side detector sum |
| --- | ---: | ---: |
| HDF5 load | `1.642 s` | `1.738 s` |
| Automatic probe mean/fit | `0.289 s` | `0.039 s` |
| Complete setup | `3.927 s` | `3.476 s` |
| Full 200-trial wall time | recent reference `46.17 s` | `43.67 s` |

The decode-side and independent Metal detector sums were elementwise equal
(`max_abs=0`). The full signoff reproduced the canonical center/radius, fitted
parameters, exact loss `0.04469207674264908`, and final phase/amplitude
statistics bit-for-bit. This is an accepted setup optimization, not a claim
that the remaining optimizer path has reached the under-20-second target; the
same run still spent `33.25 s` in Optuna and `6.10 s` in refinement.

Two follow-up evaluator shortcuts were rejected and removed:

- Increasing the packed-storage chunk above the established 512 logical-BF
  reduction boundary sometimes reduced isolated wall time, but changed losses
  materially for the second candidate. Packed storage must preserve the
  original logical reduction boundaries; chunk timing is not a valid speedup
  when it changes float32 accumulation order or results.
- The packed active set makes the column kernel's `active_bf` branch logically
  redundant. Compiling a branch-free specialization nevertheless changed the
  Metal instruction schedule and moved the refined loss to
  `0.0446920320391655`. It was removed because logical equivalence is not
  sufficient for the bit-parity contract.

The next accepted evaluator change moved the small candidate probe-vector
calculation outside the logical-BF loop. The exact row and column Metal kernels,
512-logical-BF boundaries, and accumulation order remain unchanged; each chunk
receives a view of one pre-evaluated packed probe vector. Two independent
200-trial signoffs measured `42.35 s` and `42.12 s` wall, with Optuna at
`32.09 s` and `31.67 s`, respectively. Both runs reproduced the canonical
parameters, loss, and phase/amplitude statistics bit-for-bit. A focused test
pins one probe-vector construction per fused candidate pair rather than one
construction per BF chunk.

Reusing one seven-float row-kernel scalar buffer across BF chunks was also
tested and removed. Three bit-identical 200-trial runs measured `44.26 s`,
`41.96 s`, and `42.50 s`; the `42.50 s` median did not beat the retained
packed-probe version's `42.24 s` two-run median. Tiny allocation changes must
not be kept on the strength of one favorable thermal-state sample.

Passing the complete precomputed two-candidate probe vector into every row
pack with compile-time stride/offsets avoided MLX's compact slice handoff and
kept all 21 canonical losses exact. After the new specializations warmed, its
Optuna gates measured `2.056/2.063 s`; an immediate restored compact-slice
control measured `2.069 s`. The matched 6 ms difference is noise-scale and did
not justify a wider row-kernel API or ten offset-specialized shaders, so the
full-buffer handoff was removed.

Computing the active-BF mask once from that full probe vector is retained. The
former row wrapper rebuilt the identical mask from each of ten pack slices;
the column kernel's mask branch and every Metal arithmetic expression remain
unchanged. Two exact gates measured `2.060/2.059 s` Optuna. A matched complete
200-trial plus Nelder-Mead A/B measured `25.110 s` fit with one mask versus
`25.168 s` with per-pack masks: `20.431` versus `20.452 s` Optuna and `3.387`
versus `3.413 s` refinement. All 231 records, 30 physical refinement calls,
fitted values, and final loss were identical; peak remained about `11.47 GB`.
The clean committed signoff at GPU `b9ef92d` measured `25.054 s` fit /
`26.21 s` wall, including `0.890 s` preparation, `20.388 s` Optuna, `3.380 s`
refinement, `0.051 s` final object, and `0.146 s` final phase/loss. It again
reproduced the complete exact trace and used `11.474 GB` peak.

Moving that mask into dataset-prepared state was exact but removed. Two gates
measured `2.067/2.075 s` Optuna versus `2.059/2.059 s` when deriving it once
from each already-hot candidate probe vector. The extra persistent buffer and
prepared-state field did not eliminate useful work; the per-evaluation mask is
apparently fused cheaply and remains the narrower ownership boundary.

Leaving the per-evaluation mask lazy was likewise exact but slower. Two gates
measured `2.063/2.078 s` Optuna versus the synchronized mask's
`2.059/2.060 s`. The explicit mask evaluation remains: on MLX it is a useful
small dependency boundary, not removable host overhead.

A four-pixel vectorized decode-side detector-sum reducer with a general scalar
tail was likewise removed. Vector load samples (`1.965/1.732/1.735 s`) did not
separate from scalar samples (`2.016/1.639 s`) or the retained scalar history
near `1.74 s`. One-quarter as many threads did not lower the disk/decode floor,
so the simpler arbitrary-shape scalar kernel remains production code.

### MPS anti-Hermitian half-spectrum investigation

The corrected non-DC SSB spectrum has an additional exact mathematical
structure that is not captured by the Hermitian `G_qk` storage alone. For a
real scan image, `G(-q) = conj(G(q))`, while the SSB correction obeys
`C(-q) = -conj(C(q))`. Consequently the corrected non-DC spectrum is
anti-Hermitian. Multiplication by `-i` turns it into a Hermitian spectrum that
can be represented by `512x257` complex values. The real DC coefficient must
remain separate.

The even-sized FFT Nyquist row and column are unpaired frequencies and do not
obey this relation in the sampled grid. They must not be dropped or averaged.
An exact implementation can retain them as two 512-value complex edge vectors;
their inverse transform is two one-dimensional FFTs plus separable sign
modulation. This adds about `0.4%` to the half-spectrum storage rather than
restoring the full plane.

Real Reference-512 validation used the automatic full field (8937 logical BF pixels,
2464 active), float32/complex64, and the canonical fitted aberrations. After
retaining the Nyquist edges exactly, the half-spectrum reconstruction differed
from the existing float32 phase by at most `8.35e-7 rad` (`3.87e-7 rad` p99)
for the sampled BF terms. The squared-phase sum was unchanged at float32
resolution. This is float32 FFT ordering roundoff, not a dtype or scientific
objective reduction.

An isolated 64-BF MLX FFT-stage experiment measured:

| FFT representation | Median | Notes |
| --- | ---: | --- |
| Full complex `ifft2`, `512x512` | `10.70 ms` | Existing full-plane representation |
| Half core: row `ifft` + column `irfft` | `5.97 ms` | `512x257`, DC and Nyquist edges separated |
| Two exact Nyquist edge `ifft`s | `0.21 ms` | Two 512-value complex vectors |

The isolated transform result is a `1.73x` reduction, but it is not yet an
end-to-end claim. Production adoption requires fusing correction generation
with the first strided half-width FFT, then validating all 200 exact trials,
unchanged Nelder-Mead refinement, fitted parameters, scalar loss, and final
reconstruction on the canonical benchmark.

An unfused half-width correction prototype was also rejected and removed. For
64 BF terms it reduced correction from `6.51 ms` to `4.70 ms`, but correction
plus the two generic FFT calls took `9.76 ms`; the extra intermediate dispatch
and buffer traffic consumed the transform saving. The required topology is a
Metal correction + strided-row-IFFT producer writing the `512x257` core
directly, followed by a fused real-column-IFFT + phase/loss consumer. Do not
restore the standalone half-correction buffer path.

A bounded two-boundary MLX graph was tested on the same exact paired 512
evaluator. It preserved all 21 records of the canonical 20-trial gate
bit-for-bit, but Optuna took `3.363 s` versus a three-run serialized median of
`3.295 s` (`3.329/3.295/3.274 s`). Deferring the synchronization alone does not
hide useful latency on this topology; it was removed before a 200-trial run.

An opt-in synchronized stage profile of a warm paired evaluation measured
`0.180 s` in fused correction plus row IFFT and `0.114 s` in column IFFT,
`atan2`, and loss reduction. The diagnostic synchronization and environment
branch were removed after measurement. The split shows that neither half alone
can deliver the required end-to-end reduction.

Two persistent MPS streams were then used to submit adjacent logical BF
boundaries concurrently, with completed partials added in the original order.
All 21 records of the canonical 20-trial gate remained bit-identical, but
Optuna regressed to `3.571 s` from the `3.295 s` baseline median. The 512
kernels contend for the same GPU resources rather than overlapping; the stream
pipeline was removed.

Holding the second candidate's eight corrected values per thread in registers
allowed the paired row kernel to reuse a 16 KiB shared row instead of reserving
32 KiB. Exact parity passed, but the long live range spilled: canonical
20-trial Optuna regressed to `7.578 s` from `3.295 s` despite a modest peak
memory reduction. The register-resident candidate was removed.

A three-stage SIMD XOR transpose replaced the first shared-memory FFT exchange
in the 512 column kernel. The randomized radix-8 reference passed, but real
canonical losses moved by one ULP and 20-trial Optuna regressed to `3.649 s`.
Twenty-four shuffle operations cost more than the eliminated barrier on this
GPU, so the SIMD variant was removed under the exact-bit contract.

Two adjacent sparse logical BF ranges were fused into one row/column launch,
while the column kernel emitted separate partials at the original boundary and
the evaluator added them in the original order. All canonical trial records
were bit-identical, but the larger scratch/chunk regressed 20-trial Optuna to
`6.406 s` from `3.295 s`. Boundary-preserving launch fusion was removed.

Three correction/FFT attempts to exploit the anti-Hermitian half were also
rejected and removed. A standalone half correction followed by exact expanded
row/column FFTs took `7.069 s` for the canonical 20-trial Optuna gate and moved
two intermediate losses by one ULP. Fusing paired-row correction into shared
memory while retaining every full row FFT improved that gate to `3.441 s`, but
still lost to a same-thermal exact baseline of `3.374 s` and retained the
one-ULP drift. Finally, a Metal producer emitted only rows `0:256` plus the
Nyquist row for a real column IFFT. Its contiguous-axis two-trial Optuna probe
fell to `2.959 s`, but the first candidate loss changed from
`0.047669749706983566` to `0.046805430203676224`; this is a scientific parity
failure, not an acceptable speedup. The real-FFT path was stopped before the
20-trial gate. Half-spectrum work must reproduce the exact sampled-grid edge
and float32 reduction semantics before performance is considered.

An edge-corrected follow-up proved that the missing Nyquist column was not the
only parity problem. The prototype retained rows 0 and 256, recomputed the
exact positive/negative Nyquist-column residual for rows 1:255, and applied
its alternating-sign contribution before the column IFFT. That reduced the
old first-candidate error substantially, but the canonical initial loss still
moved from `0.04743797704577446` to `0.04747070372104645`; another sampled
candidate moved by one float32 ULP. The separate edge kernel and mirrored-row
column loads also made the two-trial probe much slower. Independent full-grid
rounding residuals therefore remain scientifically relevant; the 281-line
prototype and its environment switch were removed in full.

A follow-up lossless-codec census compared each independently computed
negative-q row IFFT with `-conj(positive)` plus one alternating Nyquist-edge
term, using the real canonical CLI path. Typical BF chunks placed about `98%`
of real/imag components within a signed 8-bit ULP delta and more than `99.99%`
within signed 16-bit. That compactness was not uniform: several chunks for the
canonical initial candidate put only `66-94%` within int16, with sign/near-zero
outliers reaching very large ordered-bit distances. Exact storage would need
per-component escapes and full-float fallback plus paired-row synchronization.
Its best plausible saving is only a fraction of the negative-half traffic,
well short of the 2x topology reduction, so the diagnostic was removed and a
variable residual codec was not added to production.

Interleaving the two candidate radix-8 row stages so they shared one barrier
per stage was exact but not a measurable win. Two canonical 20-trial Optuna
runs took `3.324 s` and `3.371 s`; the immediately following restored baseline
took `3.351 s`. The source grew by 46 lines without separating from thermal
noise, so the simpler candidate-serial FFT schedule remains.

The paired 512 row-group storage sweep also retained four rows per group.
Reducing shared storage from 32 KiB/four rows to 16 KiB/two rows preserved all
trial bits but measured `3.365 s` for the 20-trial Optuna gate versus a
same-session `3.351 s` baseline. Eight KiB/one row increased launch pressure
catastrophically and took `21.090 s`. Both variants were removed; lower shared
memory does not imply higher throughput when it multiplies threadgroups.

Halving the paired 512 column consumer from eight columns/32 KiB to four
columns/16 KiB was also rejected. All canonical trial records remained exact,
but 20-trial Optuna regressed from `3.351 s` to `3.839 s`. The extra column
threadgroups outweighed any occupancy benefit, so the eight-column consumer
remains.

Submitting the two candidates as independent row threadgroups in one Metal
dispatch was exact but demonstrated why candidate fusion is essential. The
16 KiB independent groups duplicated all geometry and `G_qk` reads; canonical
20-trial Optuna increased from `3.351 s` to `21.936 s`. The independent
candidate launch was removed immediately.

Explicitly hoisting the row-invariant `q_row +/- k_row` values outside each
thread's eight-column loop was exact but not retained. Warm 20-trial Optuna
runs measured `3.312 s` and `3.324 s`, overlapping the retained
`3.295-3.351 s` range; the first new Metal specialization took `6.987 s`
because of compilation. The compiler already handles these uniform terms well
enough that the extra source variables do not establish a speedup.

The first accepted follow-up is a tiled paired-row intermediate. The
two-candidate row producer stores the unchanged complex64 row IFFT as
`[64-column block][row][column-within-block]`; row writes remain coalesced,
while the radix-8 column consumer reads digit-reversed rows from smaller cache
neighborhoods. An explicit `tiled_input` flag keeps the standalone row-major
column API and its tests unchanged. No FFT operation, BF boundary, phase
accumulation, or scalar reduction was reordered.

Three canonical 20-trial Optuna gates measured `3.299 s`, `3.200 s`, and
`3.192 s` (median `3.200 s`) versus an immediately restored `3.334 s`
baseline. Two complete 200-trial plus Nelder-Mead signoffs reproduced all 231
trial records, 30 physical refine calls, fitted aberrations, exact loss, and
final reconstruction bit-for-bit. The warmed full run improved fit time from
`37.880 s` to `36.095 s`, including Optuna from `30.412 s` to `28.861 s` and
refinement from `5.970 s` to `5.747 s`; CLI wall moved from `43.35 s` to
`42.31 s`. Peak footprint remained about `19.4 GB`. This is a measured
`~2.4%` wall / `~4.7%` fit improvement, not the under-20-second breakthrough.

Doubling the accepted tile from 64 to 128 columns preserved exact parity but
crossed a severe address/compiler cliff: the canonical 20-trial Optuna gate
rose from the accepted `~3.20 s` median to `22.111 s`. The 64-column block
matches the radix-8 row producer's 64 contiguous thread outputs; retain it
unless a different producer and consumer are designed together.

Halving the tile to 32 columns aligned each tile with one 32-thread Apple
SIMDgroup and preserved exact parity, but its warmed 20-trial Optuna time was
`3.213 s`, not better than the accepted 64-column median of `3.200 s`. The
extra split-SIMD address arithmetic was removed.

Combining the accepted 64-column tile with the previously rejected
`8 columns x 64 FFT lanes` coordinate transpose was also exact but slower.
Two 20-trial Optuna runs measured `3.535 s` and `3.309 s`, versus the accepted
`3.200 s` median. Cache-block locality did not reverse the transpose's shared
and reduction scheduling cost, so the original thread coordinates remain.

Doubling the single-candidate 512 row group from four to eight rows was exact
but slower. Warm one-candidate canonical probes took `0.1917-0.1929 s` for
Optuna and final reconstruction, while the immediately restored four-row
kernel took `0.1885-0.1887 s`. The 32 KiB group reduced occupancy without
removing work, so the eight-row specialization was removed.

The accepted 64-column intermediate layout now also covers exact
single-candidate optimizer evaluations, while the final reconstruction keeps
the standalone row-major contract. A warm one-candidate probe improved from
about `0.1885 s` to `0.1817 s`. The complete 200-trial plus Nelder-Mead run
reproduced all 231 records, 30 refinement calls, fitted aberrations, and final
loss exactly. Refinement improved from `5.747 s` to `5.639 s`, total fit from
`36.095 s` to `35.903 s`, and CLI wall from `42.31 s` to `41.79 s`. This is an
incremental accepted storage-order improvement; it does not change arithmetic.

Narrowing the optimizer intermediate from 64-column to 8-column cache blocks
is also accepted. The producer still writes eight-element coalesced segments,
while the column consumer's row stride falls from 64 to 8 complex values. Two
warmed 20-trial gates measured `3.153 s` and `3.111 s`, with every trial bit
unchanged. Two complete signoffs again reproduced all 231 records, 30
refinement calls, aberrations, and final loss exactly. They measured
`34.898/35.385 s` fit and `40.48/41.79 s` CLI wall, versus the preceding
single-candidate-tiled result of `35.903 s` fit and `41.79 s` wall. The first
signoff broke down as `27.971 s` Optuna and `5.464 s` refinement. Peak
footprint remained about `19.3 GB`; this improves cache locality rather than
the size of the full intermediate.

A 2026-07-28 clean-tree signoff after the rejected follow-ups again used the
2,342,780,928-byte u8 BF-column source, all 8,937 logical / 2,464 active full-BF
terms, 200 seed-42 Optuna candidates, and unchanged Nelder-Mead. It reproduced
all 231 records, 30 physical refinement calls, the canonical aberrations
(`73.1818862146`, `14.0209629488`, `0.4700365260`), and loss
`0.04469207674264908`. Source load was `1.940 s`; the fit was `36.132 s`, split
into `28.917 s` Optuna and `5.745 s` refinement. This is a clean regression
validation, not a new speed record: the accepted quiet-machine best remains
`34.898 s` fit / `40.48 s` complete CLI wall.

A 16-column follow-up was exact but slower. Its two warmed 20-trial gates
measured `3.169 s` and `3.196 s` (median `3.182 s`), versus `3.153 s` and
`3.111 s` (median `3.132 s`) for the retained 8-column layout. The 16-column
variant was removed.

A 4-column follow-up was also exact but regressed the warmed 20-trial gate to
`3.252 s`. Four-element producer write segments cost more than the tighter
column stride saved, so this variant was removed as well. Eight columns is the
measured local optimum across 4, 8, 16, 32, 64, and 128-column blocks.

Mapping SIMD lanes across the eight adjacent columns of the retained tile was
exact but slower. The coordinate-transposed 20-trial gates took `3.475 s` and
`3.295 s`, versus the retained `3.132 s` median. Contiguous input loads did not
offset the worse shared-memory FFT lane schedule, so the original 64 FFT-lane
by 8-column coordinates remain.

Separating each candidate's minus-shift gamma contribution from plus-shift
geometry shortened the nominal geometry live range, but introduced candidate
register arrays and changed instruction scheduling. The 20-trial gate
regressed to `7.073 s`, and its canonical trial hash changed. The refactor was
removed on both performance and exact-parity grounds.

Explicitly unrolling the compile-time two-candidate correction loop preserved
exact parity but did not improve the warmed gate (`3.137/3.142 s` versus the
retained `3.132 s` median). The compiler already specializes this loop, so the
pragma was removed.

A fused `q/-q` row-pair prototype reused swapped `+k/-k` geometry while
retaining the full four-row FFT and computing the DC and Nyquist row/column
independently. The focused batch-versus-single tests passed after the Nyquist
fix, but two of 21 canonical gate losses moved by one float32 ULP. The larger
partner live state also caused a severe Metal spill: 20-trial Optuna took
`22.861 s`. The prototype was removed. Any future symmetry kernel needs a
separate low-register producer and exact canonical rounding, not extra state
inside the already occupancy-limited paired row FFT.

Reinterpreting the tiled complex output as `float2` to issue explicit 64-bit
stores preserved exact parity but regressed the warmed 20-trial gate to
`3.165 s`. Metal already coalesces the component stores effectively, so the
pointer reinterpretation was removed.

The paired two-row/16 KiB row group was retested after adopting the 8-column
intermediate. It remained exact but measured `3.180 s` for the warmed 20-trial
gate, slower than the retained `3.132 s` median. Four rows remain the measured
producer optimum; halving shared memory does not repay twice as many groups.

The remaining three-row/24 KiB producer point was also retested with the
8-column intermediate. It retained the exact canonical trial hash, but its
non-power-of-two 192-thread scheduling crossed a severe runtime cliff: Optuna
took `6.907 s`, and the initial/final exact passes took `0.672/0.670 s`.
The 256-thread four-row producer remains the complete 1/2/3/4-row optimum.

Hoisting the BF-plane, row, and mirrored-row Hermitian `G_qk` base addresses
outside the eight-column loop was exact but slower (`3.170 s` warm). The four
additional live 64-bit values cost more registers than their repeated index
multiplies, so the original fetch expression remains.

Rewriting the Hermitian direct/mirror predicate from `col <= 256` to its
equivalent radix-lane form was exact but crossed the same Metal compiler
cliff. The canonical 21-record hash remained unchanged, while Optuna rose to
`6.934 s`, initial loss to `0.930 s`, and final phase/loss to `0.967 s`.
Metal's simpler column predicate is retained even though the lane form makes
seven of eight loop iterations visibly uniform at source level.

Generating separate full-plane and Hermitian-half Metal source removed the
compile-time-dead full-plane branch and live mirror flag, but was a still
larger exact regression. The 21-record hash stayed unchanged while Optuna
rose to `8.237 s`; initial and final exact passes took `0.923/0.932 s`.
Retain the original `constexpr GQK_COLS` branch and mirror flag: on this M5
they guide a much faster compiler schedule despite looking less specialized.

Explicitly unrolling the row producer's eight q-column lanes was rejected.
All 21 records in the real automatic full-BF 20-trial gate remained
bit-for-bit identical, but the expanded correction live state crossed a Metal
register/compiler cliff: Optuna rose from `3.801 s` to `6.864 s`, initial loss
from `0.362 s` to `0.962 s`, and final phase/loss from `0.387 s` to `1.000 s`.
The pragma was removed; the compiler-controlled loop is materially faster.

Submitting two 512-logical-BF chunks per bounded Metal graph instead of
synchronizing after each chunk was exact but rejected as noise-scale. Its two
20-trial Optuna stages measured `3.328 s` and `3.158 s`; the immediately
restored one-chunk baseline measured `3.208 s`. Final exact passes overlapped
at `0.312 s` for depth two and `0.304 s` for depth one. The matched difference
was only `1.6%`, with no stable peak-footprint reduction, so 512 retains one
row intermediate per synchronization boundary on the 24 GB target Mac.

A conservative pre-aperture radial bound skipped the complete geometry path
when both shifted apertures were certainly zero. The sampled canonical field
has about `25.3%` such q sites, and all trial bits remained exact, but the extra
live bound variables crossed a Metal register/scheduling cliff: 20-trial
Optuna rose from `3.351 s` to `21.674 s`, while the final single exact pass rose
to `1.152 s`. The branch was removed; avoiding square roots is not useful if
the guard spills the 512 paired row kernel.

Moving that conservative support decision into an MPS-built 81 MB bitmask did
not avoid the compiler cliff. A linear 32-bit layout was exact and added only
about `0.07 s` to preparation, but eight strided mask loads per row thread
regressed Optuna to `3.527 s`. Repacking the same bits as one coalesced byte
covering each thread's eight columns kept preparation below one second, yet
the additional kernel input/control path caused a larger spill: Optuna took
`22.099 s` and the final single exact pass took `2.265 s`. Both mask builders,
fields, and row branches were removed.

A cache-blocked split-radix handoff used the existing four-row/two-candidate
32 KiB row scratch to compute the even or odd half of the first column radix-8
before writing the intermediate. The column consumer completed the identical
radix-8 association, and all 21 canonical gate records were bit-exact. It was
nevertheless slower: 20-trial Optuna took `3.484 s` versus `3.351 s`. The
extra row barrier and four-row global scatter cost more than the removed
column arithmetic, so the 69-line split stage was removed.

Transposing the column threadgroup coordinates to put columns on the fastest
Metal thread dimension was exact but did not improve the full gate. The
`8 columns x 64 FFT lanes` layout measured `3.392 s` versus `3.351 s`; reducing
it to four columns/16 KiB regressed to `3.795 s`. More contiguous eight-value
segments did not outweigh the launch/shared/reduction behavior of the retained
`64 FFT lanes x 8 columns` layout, so both coordinate transposes were removed.

## Next performance work

### Native 128/256 exact MPS candidate scheduling

The exact 128 and 256 MPS kernels previously serialized every candidate in an
Optuna batch. A size-specific scheduler now submits the unchanged
single-candidate Metal kernels through two persistent worker-local MLX streams
and synchronizes only when both losses are requested. CPU threads perform
submission only; all reconstruction and loss arithmetic remains on MPS.

No native 128x128 or 256x256 Samsung/ARINA acquisition was available on the
measured Mac. These results are therefore kernel-microscope signoffs, not real
scientist-workflow claims. The prepared evidence used native complex64
Hermitian arrays at each scan size, with the real Reference-512 automatic geometry and
the same `8937` logical / `2464` aperture-active BF policy as the canonical 512
run. There was no scan crop, detector binning, BF reduction, or objective
change.

| Native size | Quantity | Serialized | Optimized MPS | Speedup |
| --- | --- | ---: | ---: | ---: |
| 128 | exact candidate-pair p50 | `27.71 ms` | `18.84 ms` | `1.47x` |
| 128 | 200-trial Optuna | `3.009 s` | `2.133 s` | `1.41x` |
| 128 | complete synthetic fit | `4.291 s` | `3.126 s` | `1.37x` |
| 256 | exact candidate-pair p50 | `85.59 ms` | `65.66 ms` | `1.30x` |
| 256 | 200-trial Optuna | `8.850 s` | `6.934 s` | `1.28x` |
| 256 | complete synthetic fit | `9.900 s` | `7.738 s` | `1.28x` |

Twenty randomized candidate pairs at each size matched the serialized loss
bit-for-bit. The full seed-42 runs also returned identical optimum parameters,
final loss, phase statistics, and amplitude statistics.

The 256 path subsequently gained a fused candidate-pair topology. Its row
kernel calculates candidate-invariant geometry and reads each complex64
`G_qk` value once, then evaluates both float32 aberration corrections and row
IFFTs in one threadgroup. A batched column kernel schedules both unchanged
column IFFTs and phase accumulations in one dispatch. Per-candidate reductions
retain the scalar path's exact shapes and association order. Host-side
`cos(2 phi)` and `sin(2 phi)` also deliberately use Python-float trigonometry
followed by float32 conversion, matching the scalar path bit-for-bit.

The fused 256 row stage alone was `1.44x` faster than two serialized rows, but
was rejected as a complete topology because serial column processing made the
exact pair `32%` slower than two streams. Pairing both row and column stages
produced the accepted result in the table. At 128 the complete fused topology
was only about `2%` faster than two streams, too close to noise to justify the
additional production path, so 128 retains persistent stream scheduling.

The later one-row fused-pair topology changed that 128 conclusion. Reusing
the 256 pair kernel with one row per group reduced the exact pair from
`19.81` to `18.84 ms` p50, passed 20 randomized pairs bit-for-bit, and lowered
the full fixed-seed fit from `3.255` to `3.126 s`. Native 128 batch pairs now
use fusion; persistent streams remain the fallback for batches wider than two.

The accepted 256 column consumer now places both candidates in separate
float2 threadgroup planes and runs their scalar radix stages inside one
threadgroup, sharing one barrier after each stage. This reduced the exact pair
from `70.82` to `66.59 ms` p50 and the fixed-seed fit from `8.248` to
`7.848 s`. Forty randomized pairs across 128/256 matched the prior scalar
losses bit-for-bit. A preliminary float4/vector-atan2 version was faster but
failed randomized exact parity by one ULP and was deleted. The scalar paired
column was slightly slower at 128, so only native 256 selects it.

The paired-column width sweep retained four columns per group. At 256, widths
one/two/four measured `79.24` / `69.29` / `67.52 ms` exact-pair p50; at 128,
one/two/four/eight measured `22.06` / `19.09` / `18.73` / `19.20 ms`.
All widths were bitwise identical, but none beat the accepted width four.
Non-power-of-two 256 widths were later checked on the complete fixed-seed fit:
three/five/six columns measured `9.270` / `9.711` / `9.001 s` versus about
`7.74 s` at four. Exact losses remained unchanged, so the regressions are
threadgroup occupancy/scheduling effects rather than scientific differences.

The fused row producer subsequently interleaved each scalar radix stage across
both candidate planes and issued one barrier after the pair instead of one per
candidate. This preserved all randomized loss bits and reduced the 256 pair
from `66.59` to `65.66 ms`, Optuna from `7.044` to `6.934 s`, and the full fit
from `7.848` to `7.738 s`. The 128 full fit measured `3.139 s` versus the
accepted `3.126 s`, a `0.4%` difference within observed run noise; no 128
speedup is claimed for this scheduling change.

The native pair path now submits two sparse logical-BF chunks before each MLX
evaluation barrier while retaining the exact sequential float32 phase and loss
additions. All 40 randomized 128/256 candidate pairs remained bit-for-bit
identical. Repeated full fits moved from about `3.137` to `3.017 s` at 128 and
from about `7.744` to `7.596 s` at 256. The bounded two-chunk graph raises peak
active allocation by `71 MiB` at 128 and `284 MiB` at 256 (`2.70` to `2.77 GiB`
and `3.38` to `3.67 GiB` total in the synthetic microscope) but returns to the
same steady active allocation after each pair. Do not extend this to an
unbounded deferred graph; the two-chunk cap is the measured speed/memory trade.

A bounded three-chunk follow-up superseded the two-chunk cap after repeated
signoff. Full-fit medians improved again to `2.892 s` at 128 and `7.469 s` at
256, with unchanged optimum, final reconstruction statistics, and randomized
loss bits. Peak totals are `2.84 GiB` and `3.94 GiB`, an additional `67 MiB`
and `269 MiB` over the two-chunk graph but still far below this Mac's `17.8 GiB`
recommended Metal working set. Production stops at three pending evidence that
a deeper graph justifies each additional full row-IFFT buffer.

Four-chunk submission provided that evidence and superseded the three-chunk
cap. Repeated fits reached about `2.862 s` at 128 and `7.40 s` at 256 while
retaining the same exact loss bits. Peak totals are `2.91 GiB` and `4.20 GiB`;
steady active allocation is still unchanged after the pair. Each deeper value
must continue to be tested as a complete fit because synchronization savings
flatten while every level retains another row-IFFT intermediate.

Six-chunk submission then reached the next evaluation-count breakpoint and
superseded four. Repeated fits measured about `2.834 s` at 128 and `7.338 s` at
256 with identical scientific outputs. Peak totals are `2.98 GiB` and
`4.47 GiB`, only `69 MiB` / `275 MiB` above depth four because the automatic
BF storage chunks have variable packed lengths. The six-chunk cap is explicit;
the graph is never allowed to retain the complete BF walk by default.

Depth nine was rejected as the synchronization/memory saturation point. It
regressed 128 to `2.862 s`; repeated 256 fits had a `7.321 s` median versus
about `7.338 s` at depth six, only `0.2%`, while peak allocation jumped from
`4.47` to `5.26 GiB`. The transient nine-chunk change was removed, and both
sizes retain the simpler six-chunk cap.

A conservative threadgroup-uniform zero-row aperture bound passed randomized
bit parity but was rejected as noise. The 256 fit measured `7.723 s` versus
the accepted `7.738 s` while pair p50 slightly regressed (`65.66` to
`66.39 ms`); too few rows were eliminated to justify the branch, so it was
removed.

Replacing the native 128/256 row kernel's compile-time BF chunk length with a
runtime bound and stride was also rejected. It removed per-length Metal source
variants and all 40 randomized candidate pairs retained bit-for-bit loss
parity, but warmed fixed-seed fits were unchanged within noise: `3.130 s`
versus `3.126 s` at 128 and `7.722 s` versus `7.738 s` at 256. The runtime
indexing was removed because source-cache simplification without a measured
wall-time gain does not justify production complexity.

Packing the two candidates into `float2` lanes in the row correction stage was
rejected before performance signoff. Although every source-level arithmetic
operation retained its scalar association, Metal's vector `fast::sincos`
changed both losses by one ULP on the first randomized 128 candidate pair
(`6.0763655e-10` / `6.0891486e-10` versus `6.0763660e-10` /
`6.0891490e-10`). The vector prototype was deleted under the exact-bit
contract; candidate-lane transcendental vectorization is not interchangeable
with the accepted scalar correction path.

A narrower follow-up retained all four scalar `fast::sincos` calls and packed
only the downstream gamma magnitude, normalization, and complex multiply. It
produced the same one-ULP loss mismatch on the first randomized pair, so even
non-transcendental candidate-lane SIMD is not exact-bit safe with this Metal
compiler. That prototype was also deleted.

A complete logical-BF chunk sweep retained 512 for both native sizes. For 128,
chunks 256/512/1024/2048 measured `6.975` / `3.111` / `3.877` / `3.362 s`;
for 256 they measured `11.458` / `7.841` / `8.447` / `8.032 s`. Larger chunks
did not amortize enough work to offset allocation/scheduling pressure, and at
256 the 256 and 1024 boundaries also changed the float32 loss association by
one ULP. Production keeps the exact 512 boundary used by the fixed benchmark.

Explicitly selecting `metal::fast::atan2` in the paired 256 column kernel was
bit-for-bit identical across 20 randomized pairs but was not a distinct speed
path under the existing fast-math compile option. Three-run fixed-fit medians
were `7.740 s` for explicit fast namespace versus `7.744 s` for the accepted
spelling, with overlapping run variance. The source-only change was removed.

The actual fused 128 row wrapper was re-swept at one/two/four/eight rows per
threadgroup after discovering that an earlier occupancy script patched the old
scalar wrapper. Exact-pair p50 values were `19.11` / `19.22` / `19.04` /
`18.92 ms`, all bitwise identical. Eight rows did not survive full-fit A/B:
three-run medians were `3.133 s` at eight versus `3.137 s` at the accepted one
row, with overlapping tails. Production therefore remains at one row/group.

The accepted 256 row pair uses one row per threadgroup (64 threads and 4 KiB
of shared row storage for two candidates), rather than four rows (256 threads
and 16 KiB). Across the complete symmetry-aware exact pair this reduced p50
from `73.69 ms` to `71.56 ms`, preserved all loss bits, and lowered the warmed
full fit from `8.397 s` to `8.248 s`. The first invocation of a newly compiled
Metal variant paid about `0.65 s` of one-time pipeline compilation; steady-state
and cold-compile timing must remain separate in future comparisons.

A 256 column threadgroup-width sweep was rejected. Eight columns per group
was about `3%` faster than four for one isolated 512-BF dispatch with identical
partial float32 outputs, but the real symmetry-aware storage walk uses variable
chunk sizes. On the complete fixed-seed run, width eight regressed Optuna from
`7.579 s` to `8.023 s` and the fit from `8.397 s` to `8.826 s`. Production
therefore retains four columns per group; do not select this topology from a
single uniform-chunk microbenchmark.

The analogous 128 single-candidate row-group sweep was also rejected. Under
the actual two-stream exact-pair scheduler, one, two, four, and eight rows per
group measured `20.40`, `20.01`, `20.22`, and `20.70 ms` p50 respectively.
The apparent `1%` p50 advantage at two rows came with a worse p95 (`23.36 ms`
versus `20.96 ms` at four rows), so the existing four-row topology remains.

An exact double-zero aperture branch copied from the 512 row topology was
rejected for native 128/256. It skipped both candidate `sincos` evaluations
and the `G_qk` read whenever both shifted apertures were zero, and focused
parity passed, but divergence outweighed the skipped work for this automatic
probe geometry. The fixed-seed 128 fit regressed from `3.250 s` to `3.286 s`;
the 256 fit regressed from `8.248 s` to `9.090 s`. The branch was removed.

A row-output transpose / contiguous-column-input topology was also rejected.
It preserved the exact loss bits and was neutral at 128 (`19.43` versus
`19.40 ms` p50), but strided row stores outweighed coalesced column reads at
256: the exact pair regressed from `71.55` to `76.37 ms` p50. The transient
row-IFFT remains row-major.

Removing apparently redundant final-stage FFT barriers was rejected. Each
thread consumes only its own final four values and focused parity passed, but
the changed Metal schedule regressed the fixed-seed 128 fit from `3.250` to
`3.275 s` and the 256 fit from `8.248` to `9.287 s`. The synchronization was
restored; source-level dependency reasoning alone was not a performance win.

A register-resident radix-4x4 prefix for the first two 256 FFT stages was
rejected. It improved pair p50 from `71.24` to `70.08 ms` and the warmed fit
from `8.248` to `8.120 s`, but randomized parity found a one-ULP float32 loss
difference at candidate pair 7 even though the fixed benchmark candidates
matched bit-for-bit. The implementation was removed under the exact-bit
contract. Its first uncached run also compiled every symmetry-chunk
specialization and cost `9.613 s`, including about `1.54 s` of one-time Metal
pipeline compilation.

Caching immutable MLX twiddle/scalar arrays was rejected as a speed change.
The fixed-seed fits remained within noise (`3.266 s` at 128 and `8.249 s` at
256 versus `3.250` / `8.248 s` accepted), so the cache and an accidental
cross-size 512 wrapper edit were both removed rather than retained as dead
complexity.

Problem: the live object redraw target is met for the real Mac MPS `512x512`
radius-30 object-wave workflow, and the fused column accumulator makes exact
radius-30 phase/loss reviewable at about `13 FPS`, but full-active-BF exact
phase/loss is still far below real-time on MPS.

Action: port or prototype the fused correction + row-FFT half of the CUDA
topology on MPS, then rerun the same before/after table with full BF.

Problem: the `512x512` exact phase/loss path now meets the `33.3 ms` / `30 FPS`
target on the real central held-out dataset field, but the p95 margin is small on the
300 W GPU1 power envelope.

Action: keep the 64-BF default chunking for 512, and re-run a 600-step
real-data signoff whenever row/column kernels, power settings, BF policy, or
driver/toolkit versions change.

Problem: the phase-variance optimizer path cannot inherit the object Fourier-
sum result because `mean(angle(object_bf))` is a different scientific quantity
from `angle(mean(object_bf))`.

Action: keep optimizing `reconstruct()` and `reconstruct_with_loss()` with
dedicated native variance kernels or an equivalent exact reformulation; do not
reuse the object-path claim for the optimizer objective.

Problem: `1024x1024` batched optimizer variance is still disabled.

Action: implement and reference-check a dedicated 1024 batch variance kernel before
enabling batch trials at that size.

Problem: MPS and WebGPU are implemented for native-size SSB review, but they
are not yet equivalent to CUDA full-BF real-data signoff.

Action: keep extending the 12-cell matrix with real-data MPS and WebGPU runs.
For MPS, measure the same BF policies used by scientists on a Mac MPS machine and
validate against CUDA-reference fixtures. For WebGPU, keep reusable WGSL/browser
kernel sources beside their scientific domains, bundle them through `quantem.widget`,
reference-check against native `quantem.gpu` outputs, and do not use
SwiftShader performance numbers.
