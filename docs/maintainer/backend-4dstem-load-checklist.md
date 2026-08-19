# Backend 4D-STEM Load/Decode Checklist

## Native remote browse ownership, 2026-08-15

- `quantem.gpu.remote` owns the versioned native-viewer protocol, CUDA-resident
  master cache, exact detector products, selected diffraction, and acquisition
  readiness polling.
- `quantem-gpu serve DATA_FOLDER --gpus auto --port P` binds to loopback only.
  `--gpu N` remains the explicit single-device form. SSH
  owns authentication and encryption; the HTTP service must never listen on a
  public interface by default.
- One resident 4D volume belongs to exactly one CUDA GPU. New datasets are
  assigned to the least-populated device that can admit the exact transition
  peak; cache hits stay on their original device. Per-device LRU eviction must
  never evict an entry on another GPU, and the active dataset is preserved when
  another device can admit the request.
- Capabilities report `browse_gpus`, maximum single-device admission budget,
  aggregate cache budget, and live per-device resident bytes. Do not use the
  aggregate budget to admit one volume: the viewer does not silently shard a
  dataset or change its arithmetic.
- The server must import no `quantem.live` modules. Dashboard, automated SSB,
  trial orchestration, and reports remain separate consumers.
- Crop and detector binning go directly into `quantem.gpu.io.load`; scan
  binning replaces the cropped resident volume and its decoded-source
  transition peak is admitted before CUDA allocation.
- Enter the configured CUDA device context inside every server worker thread.
  CuPy device selection is thread-local, so loading on one worker does not make
  later detector or transfer work on another worker safe automatically.
- Integer detector sums and selected diffraction frames use little-endian
  uint32 wire buffers. Float32 is reserved for scalar CoM/DPC products.
- Acceptance requires a clean environment containing `quantem.gpu[cuda,remote]`
  but no `quantem.live`, endpoint parity on real data, a bounded-memory failure
  test, warm interaction timing, and one packaged Live4DSTEM SSH workflow.

### Multi-GPU acceptance record, 2026-08-15

The real-CUDA gate used two 96 GB RTX PRO 6000 Blackwell GPUs and two independent
catalog entries for the same `128x128x48x48` uint16 S128 evidence. Automatic
placement retained one complete 75,497,472-byte volume on each GPU. No volume
was split and an existing dashboard allocation on GPU 0 was left untouched.

| Check | Result |
|---|---:|
| Cold BF + exact load, session A | 0.366 s |
| Cold BF + exact load, session B | 0.133 s |
| Warm BF cache hit | 1.39 ms |
| Warm selected diffraction, both GPUs | 1.14–1.46 ms |
| BF versus independent CPU reference | exact, max error 0 |
| Selected diffraction versus HDF5 reference | exact, max error 0 |

The clean-install gate installed the wheel with `[cuda,remote]` into an empty
environment with no system site packages. The 3.6 MB wheel pulled its CUDA
runtime/compiler dependencies through the CUDA extra, initialized both GPUs,
and launched through the packaged native app over an SSH alias. The app loaded
session A, switched to session B through its Dataset menu, and displayed
non-black BF, diffraction, product thumbnails, histograms, and metadata for
both. The full Python suite passed with `287 passed, 55 skipped`; the macOS
Swift suite passed 38 tests with the opt-in SSH test skipped, and that real
SSH/CUDA test then passed separately in automatic multi-GPU mode.

Handoff checklist for accelerated Show4DSTEM HDF5 load, decode, and product
paths across **CUDA**, **MPS**, and **WebGPU**. Pair this with
`backend-optimization-matrix.md`; that matrix is the measured evidence log and
this page is the pass/fail rollup.

Rule: do not count a capability as done if it changes the microscope evidence.
No hidden scan crop, detector bin, BF reduction, saved derived float cache, or
CPU fallback can masquerade as an accelerated backend. Parity is exact integer
match versus the CUDA reference on real data, except float CoM/DPC values where
the acceptance tolerance is `<=1e-5`.

For the pipeline-topology failure that caused a native macOS loader to decode a
full dataset twice and then serialize a 19.327 GB transpose, see the
[Native Metal HDF5 loader postmortem](native-metal-hdf5-postmortem.md). Its
pass-graph and memory-traffic checklist is required for future loader work.

## Exact crop/bin and arbitrary-scan gate, 2026-08-15

Crop/bin implementations must use ceiling output dimensions and preserve
partial row, column, and detector edge bins. CUDA reductions accumulate exact
integer counts in uint64 and reject values that cannot be transported as exact
uint32 display evidence; they must not round through float32.

Native iDPC must also accept cropped scan shapes that are not powers of two.
The retained Swift kernel uses the radix-2 path unchanged for power-of-two
dimensions and Bluestein FFT for arbitrary dimensions. The gate includes odd
and rectangular crops, exact selected-DP/BF/ABF/ADF agreement, and float32
DPC/iDPC tolerance.

Memory admission uses the exact output plan:

~~~text
ceil(crop rows / scan bin)
× ceil(crop columns / scan bin)
× ceil(detector rows / detector bin)
× ceil(detector columns / detector bin)
× (native bytes when unbinned, otherwise 4-byte uint32)
~~~

Do not estimate a binned plan from compressed file size, drop edge values, or
silently retain a sum in the source uint16 dtype. Detector sums widen to uint32
before their first output write. A backend claiming load-time scan binning must
also reduce into the replacement volume during decode; a post-load `bin()` path
must account for the temporary source-plus-replacement peak and must not be
described as streaming.

## Reference M5 MPS single-file checkpoint, 2026-07-25

The private Reference-512 `512x512x192x192` master was fully local on a 24 GB, 10-GPU-core
Apple M5. Native uint16 output is 19.33 GB; clipped uint8 browse output is
9.66 GB, and a full raw count audit found `max=53` with no values above 255, so
the uint8 result is lossless for this file.

The production MPS load now groups source-file chunks into approximately
1.5 GB Metal output buffers by default. This remains a zero-copy result; it
only changes output-buffer ownership from one buffer per HDF5 source file to
larger adjacent groups. It also enables the same grouping for uint8 output,
which still decodes through reusable native scratch before the existing Metal
clip kernel.

| Output | Per-source buffers | 1.5 GB grouped buffers | Result arrays |
| --- | ---: | ---: | ---: |
| uint8 cold single load | `2.310 s` | `1.737 s` | `27 -> 7` |
| uint8 warmed repeats | `1.792-1.808 s` | `1.727-1.736 s` | `27 -> 7` |
| uint16 cold single load | `2.333 s` | `1.972 s` | `27 -> 14` |

Fifty random uint8 samples and twenty random native uint16 samples matched
direct HDF5 decoding exactly. Stage profiling showed only about `0.5-0.65 s`
of local reads, overlapped with Metal work; GPU LZ4 plus unshuffle/clip remains
the dominant floor. A 3.6 GB grouping and fused decode/unshuffle prototypes
were slower or timing-neutral and remain rejected.

### Final safe reference-Mac handoff

The chronological checkpoints below include rejected probes. The authoritative
current-tree public-API results are:

| Output | Resident result | Fresh-process min/median/max | Warm cycles 3-10 median | Loaded / freed Metal bytes |
| --- | ---: | ---: | ---: | ---: |
| lossless uint8 | `9.66 GB`, 7 buffers | `1.680/1.775/1.821 s` | `1.493 s` | `12,349,669,376 / 2,685,992,960` |
| native uint16 | `19.33 GB`, 14 buffers | `1.915/1.960/2.285 s` | `1.642 s` | `21,276,016,640 / 1,948,663,808` |

Each loaded/freed byte pair repeated exactly for ten same-process cycles. The
uint8 result is byte-identical to the unfused scientific reference over all
9,663,676,416 output bytes; native uint16 retains the complete-stack digest
and direct-HDF5 masked sample parity reported below. Both paths use the safe
scope-wide native mask barrier; no result buffer remains allocated after
explicit `free()`.

A final alternating fresh-process check under approximately 10.9 GB of active
system swap returned the expected 7-buffer/9,663,676,416-byte uint8 and
14-buffer/19,327,352,832-byte uint16 layouts in all eight processes. The four
uint8 loads measured min/median/max `1.671/1.707/1.877 s`; the four uint16
loads measured `2.119/2.330/2.708 s`. This is a stressed-system lifetime and
layout confirmation, not a replacement for the controlled timings above.

The terminal full-content rerun also reproduced both authoritative digests.
Fused and unfused masked uint8 loads produced
`e15d361f9b917ffe10692391238c33f10cf20d44d5d58177f6748aedae398938`
at `1.680/1.706 s`. Two repetitions of the safe native uint16 path produced
`497f2b0fcdf29ac4851530b8db35049c70560f827e87b214a8482581d46b2337`
at `2.331/2.014 s`. The compared buffers were all 9,663,676,416 or
19,327,352,832 bytes as appropriate.

### Lossless-uint8 depth-3 follow-up, 2026-07-26

The 9.66 GB lossless-uint8 path now defaults to three in-flight Metal decode
buffers; native 19.33 GB uint16 stays at the conservative depth of two. On the
same real master, alternating fresh-process probes measured about `1.97 s` at
depth 3 versus `2.04-2.06 s` at depth 2. A separate three-cold-load harness
under higher system variance measured medians `2.437 s` versus `2.537 s`, the
same roughly 4% direction. Fifty random frame/pixel samples again matched
direct HDF5 values exactly.

The measured depth-3 split was about `0.35 s` read/parse, `1.35 s` waiting for
the three-slot GPU/input pipeline, and `0.07 s` final drain, with allocation
and setup making up the remainder. Depth 4 regressed to `2.16 s`; it exceeds
the three compressed-input slots and adds scratch pressure. Depth 3 is not the
uint16 default because the extra roughly 0.8 GB scratch allocation can push a
full native result beyond the recommended working set.

The same-session follow-up localized the current single-file floor. Lossless
uint8 measured `1.77-1.90 s` wall with about `0.36-0.58 s` total reads and
`2.44-2.83` summed GPU-seconds overlapped across command buffers. Native
uint16 measured `1.92-2.03 s` wall with about `0.48-0.61 s` reads and
`1.81-2.03` summed GPU-seconds. LZ4 occupancy values `4/8/16` remained within
noise; the retained uint8 default of `8` was best in the direct sweep. Output
grouping at `1.0 GB` alternated at `1.82-1.86 s`, versus `1.79-1.80 s` for the
retained `1.5 GB` grouping. These probes were rejected as non-wins; the next
load breakthrough must reduce decode/cast work or its memory traffic, not add
pipeline depth or smaller output allocations.
Bit-unshuffle occupancy was also swept at 8, 16, and 32 SIMD groups per
threadgroup for both output dtypes. Alternating 16/32 runs overlapped and
tracked filesystem/cache state rather than occupancy; neither dtype showed a
repeatable improvement over the existing 32-group kernel, so the experimental
compile knob was removed.

### Fused lossless-uint8 unshuffle checkpoint, 2026-07-26

The no-mask uint16-source browse path now combines bit-unshuffle and saturating
uint8 conversion in one Metal kernel. It writes `min(value, 255)` directly to
the compact 9.66 GB output, removes the separate clip dispatch, and avoids the
old three-slot roughly 2.4 GB uint16 output-scratch pool. Loads with a bad-pixel
mask retain the original uint16-unshuffle, mask, then clip ordering.

Alternating fresh decoder runs measured `1.518-1.527 s` fused versus
`1.775-1.994 s` unfused. GPU wait fell from about `1.34-1.38 s` to
`1.07-1.09 s`; summed GPU command time fell from `2.93-2.98 s` to
`2.52-2.56 s`. Public one-load cold comparison measured `2.074 s` fused versus
`2.519 s` unfused; later filesystem-warm public runs converged near
`2.04-2.11 s` because output allocation and cache state dominated. One
thousand random frame/pixel values matched direct HDF5 `min(raw,255)` exactly
(`mismatch=0`, `max_abs=0`), and the masked fallback produced exact zero at the
requested bad pixel. Focused IO tests passed (`55 passed, 5 skipped`).

A follow-up removed another path-independent reservation: the decompressor no
longer allocates the legacy batch-sized unshuffle result until `load()` or the
detector-binning path actually needs it. The fused chunked uint8 path never
reads that buffer, so this saves `0.737 GB` of Metal scratch for the real master
without changing a kernel or value. Fresh direct loads remained at the fused
decode floor (`1.539-1.795 s`, with the first run cold); the value of this
checkpoint is lower peak pressure rather than a claimed timing win.

A deeper LZ4+unshuffle fusion was also tested and rejected. The first exact
kernel decoded each 8 KiB block into threadgroup memory and performed all 128
uint16 unshuffle groups in its decoder SIMD group; a second exact version used
eight unshuffle SIMD groups per block and two blocks per 512-thread group. Both
matched 1,000 direct-HDF5 samples exactly, but both took about `2.086-2.105 s`
in the direct harness versus `1.518-1.527 s` for the retained two-stage fused-
uint8 path. Holding 8-17 KiB of decoded data per block sacrifices the LZ4
occupancy that is worth more than the eliminated global scratch round trip on
this M5. Do not retry this topology without asynchronous threadgroup copies or
a materially different block scheduler.

Metal resource hazard tracking was tested as another bounded loader-floor
probe. Marking all explicitly ordered shared buffers untracked produced a warm
lossless-uint8 wall time of `1.560 s` and `1.074 s` GPU wait, versus `1.549 s`
and `1.038 s` with the normal tracked buffers in the immediately preceding
run. It was reverted: command/resource hazard bookkeeping is not the limiting
cost, and removing it adds correctness responsibility without a speed win.

Using `MTLCPUCacheModeWriteCombined` only for the three compressed-input slots
was likewise neutral-to-worse. The warm lossless-uint8 load measured `1.579 s`
wall, `0.405 s` reads, and `1.063 s` GPU wait, versus the immediately preceding
normal-cache baseline of `1.549 s`, `0.416 s`, and `1.038 s`. Direct `preadv`
into shared Metal pages does not gain from write-combined CPU caching on this
machine, so normal cache mode remains.

Allocating the three GPU-only LZ4 intermediates with
`MTLResourceStorageModePrivate` also regressed the unified-memory pipeline. A
warm lossless-uint8 load took `1.668 s` wall with `1.142 s` GPU wait, versus
`1.549 s` and `1.038 s` with shared scratch. The LZ4-to-unshuffle handoff is
faster in ordinary shared memory on this Apple GPU; private scratch was
reverted.

`commandBufferWithUnretainedReferences()` was tested because all decode
resources are explicitly owned through completion. Across eight interleaved
fresh processes, excluding the first cold retained outlier, retained command
buffers averaged about `1.566 s` and unretained about `1.561 s`; GPU/read stage
metrics crossed in both directions. The roughly `5 ms` difference is
noise-scale and does not justify weaker resource-lifetime guarantees, so normal
retained command buffers remain.

Native uint16 output grouping was refreshed at `1.5`, `2.5`, `3.5`, `5.0`,
and `8.0 GB`. A one-way sweep initially appeared to favor larger groups
(`2.09-2.16 s` at `2.5-5.0 GB`), but the interleaved `1.5 GB` rerun reached
`1.988 s`, faster than repeated `5.0 GB` runs at `2.077-2.093 s` and consistent
with the historical `1.92-2.03 s` band. `8.0 GB` regressed to `2.320 s` from a
long output-buffer dependency. The apparent trend was cache-state confounding;
the retained `1.5 GB` grouping remains fastest.

The LZ4-to-unshuffle synchronization was narrowed from a buffer-scope barrier
to a resource-specific barrier on only the LZ4 intermediate. Eight interleaved
fresh-process u8 loads showed that, excluding the first cold scope-wide run,
the normal barrier averaged about `1.522 s` versus `1.527 s` targeted; targeted
GPU wait was also about `8.6 ms` higher. Metal already optimizes the scope-wide
barrier for this encoder, so the targeted form was reverted.

### Fused masked lossless-uint8 checkpoint, 2026-07-26

The public scientific load applies the detector bad-pixel mask by default. The
first fused uint8 kernel deliberately excluded that case, so a public load still
materialized the full 19.33 GB uint16 scratch stack, zeroed four detector pixels
per frame, and then clipped into the 9.66 GB uint8 result. The masked variant now
uses the detector index already owned by each unshuffle lane to apply the same
mask before writing `min(value, 255)` directly to uint8. The unmasked pipeline
remains separate and has no added mask read or branch.

Warm public `quantem.gpu.io.load(..., backend="mps", dtype="u8")` calls fell
from `1.795 s` to `1.544-1.556 s` on the real `512x512x192x192` master. The
result is still a seven-buffer, 9.66 GB zero-copy `MPSChunked4DSTEM`; public
metadata setup costs only `2-5 ms`. A direct-HDF5 gate checked 1,012 positions,
including every one of the four bad detector pixels in the first, middle, and
last frames: mismatches `0`, maximum absolute error `0`. This is the production
masked path, not a no-mask benchmark exception.

A stronger full-stack gate then hashed every byte of the 9,663,676,416-byte
fused result and the prior unfused unshuffle→mask→clip result. Both produced
the same BLAKE2b-256 digest
`e15d361f9b917ffe10692391238c33f10cf20d44d5d58177f6748aedae398938`.
The accepted speedup is therefore byte-for-byte identical over the entire real
dataset, not inferred from sampled pixels.

### Lazy compressed-buffer planning checkpoint, 2026-07-26

The public loader formerly called `chunk_iter` over all 27 HDF5 source files
before decoding solely to size its reusable compressed-input Metal buffers.
That serial first-load scan cost about `0.49 s`; the decoder then consumed the
same detailed plans. The buffer now uses a cheap safe upper bound from each
source file's byte size. A chunk payload and its on-disk span cannot exceed its
containing file, and the existing 150 MiB minimum still exceeds the largest
123.14 MB source file in this acquisition. Detailed plans are built just in
time by the normal read stage, where later files overlap earlier GPU commands.

Eight interleaved fresh-process public native-uint16 loads moved from
`2.54-2.75 s` with eager planning to `2.19-2.34 s` with lazy planning. Output
remains 19,327,352,832 bytes in 14 zero-copy buffers; dtype, decode, mask, and
chunk order are unchanged. A 1,012-position direct-HDF5 uint16 gate, including
all four bad pixels in first/middle/last frames, reported mismatches `0` and
maximum absolute error `0`. Public masked lossless-uint8 warm repeats remain at
the fused floor and measured `1.514-1.532 s` in the follow-up.

The public wrapper also no longer forces a full Python garbage collection after
explicitly transferring Metal-buffer ownership. That legacy call cost about
`50-70 ms`; warm native uint16 measured about `2.21 s` without it versus
`2.26 s` with it. A final ten-cycle 9.66 GB uint8 load/free stress with GC
disabled returned to the exact same `2,685,992,960`-byte cached-scratch baseline
after every free, while every loaded state was exactly `12,349,669,376` bytes.
Explicit `MPSChunked4DSTEM.free()` and decoder pool ownership—not cyclic GC—own
these Metal allocations.

That ten-cycle u8 gate also recorded public-API latency. After two warm-ups,
cycles 3-10 ranged from `1.485` to `1.539 s` with median `1.493 s`, and the
tenth load was `1.511 s`. Together with the native series below, the safe
same-process full-stack floors are therefore about `1.49 s` lossless uint8 and
`1.64 s` native uint16 on the reference M5 MacBook Pro.

A single background worker was tested to prebuild detailed HDF5 chunk plans
while prior files decoded. It reduced one cold outlier but regressed steady
fresh-process uint8 loads to `1.775-1.798 s`, versus `1.655-1.685 s` for serial
just-in-time planning. HDF5 metadata locking and read/parse contention delay
GPU feeding more than the overlap saves, so planning remains serial and lazy.

The final clean-source interleaved fresh-process signoff alternated both public
dtypes four times. Lossless uint8 measured `1.778-1.910 s` for 9.66 GB in seven
zero-copy buffers; native uint16 measured `1.984-2.471 s` for 19.33 GB in 14
buffers. The wider fresh-process band includes Metal compilation, allocation,
filesystem state, and HDF5 plan construction. Same-process warm uint8 remains
`1.514-1.532 s`; the fastest fresh native uint16 run reached `1.984 s`.

Large Metal-buffer reservation is not the remaining cold-start gap. Instrumented
public loads found only `0.60 ms` summed allocation call time for uint8 and
`0.72 ms` for uint16, despite reserving 9.66/19.33 GB of output plus scratch;
Metal commits those shared pages lazily as the kernels write them. Lazy output
allocation cannot recover the observed filesystem/plan/dispatch variance and
was not added.

### Native mask barrier lifetime checkpoint, 2026-07-26

Native uint16/uint32 unshuffle must finish before the sparse bad-pixel kernel
writes the same output. A Metal resource-specific barrier on only the output
buffer initially appeared `60-70 ms` faster than the scope-wide buffer barrier,
and its complete 19,327,352,832-byte output produced the identical BLAKE2b-256
digest
`497f2b0fcdf29ac4851530b8db35049c70560f827e87b214a8482581d46b2337`.
Extended lifetime testing found that PyObjC retained the resource-list argument
after command completion: three load/free cycles grew Metal allocation from
`21.28` to `40.60` to `59.93 GB`, exactly one native output per cycle. The
resource-specific barrier was therefore reverted. The scope-wide barrier
returns every 19.33 GB output on `free()` and remains the production path; a
small cold-load timing gain cannot trade away bounded ownership.
Wrapping each complete load/free cycle in an explicit PyObjC autorelease pool
did not help: the same `21.28 -> 40.60 -> 59.93 GB` growth remained after pool
drain and garbage collection. The retain is below ordinary Python autorelease
lifetime, so do not retry the targeted barrier without a different native
encoding API and the full repeated-allocation gate.

An eight-pair alternating fresh-process soak during the targeted-barrier probe
gave
uint8 min/median/max `1.668/1.810/2.005 s` and native uint16
`2.015/2.098/2.391 s`. Each run used the public API, the full
`512x512x192x192` stack, default mask correction, and a newly started Python
process. Its u16 numbers are rejected probe evidence rather than final handoff
figures; the uint8 path did not use that barrier.

A second eight-pair fresh-process soak before that barrier lifetime issue was
found
measured uint8 min/median/max `1.691/1.769/1.826 s` and native uint16
`1.871/2.115/2.534 s`. This confirms that mixed sustained MPS use did not
degrade decoder throughput within fresh processes, but its native timings used
the now-reverted targeted barrier. The u16 numbers are retained only as rejected
probe evidence; the uint8 path never used that barrier and remains valid.

The final eight-pair soak after restoring the safe scope-wide barrier measured
uint8 min/median/max `1.680/1.775/1.821 s` and native uint16
`1.915/1.960/2.285 s`. A separate ten-cycle native allocation gate loaded at
exactly `21,276,016,640` Metal bytes each time and returned to the identical
`1,948,663,808`-byte reusable-scratch baseline after every explicit `free()`.
These are the authoritative single-file timings and lifetime figures for the
final source.

The same ten-cycle native gate also recorded latency. After two cache-warming
loads, cycles 3-10 ranged from `1.528` to `1.718 s` with median `1.642 s`, and
the tenth load was `1.600 s`. There is no progressive slowdown: the safe
scope-wide barrier both releases all 19.33 GB outputs and reaches a lower
same-process decoder floor than the rejected resource-list path.

The same fusion was prototyped for native masked uint16 output, but it was not
a breakthrough. Eight interleaved fresh-process public loads averaged about
`2.575 s` fused versus `2.585 s` for the existing sparse correction, with both
variants winning individual runs. Reading the detector mask for every one of
the 9.66 billion uint16 output pixels merely trades against the barrier and four
bad-pixel writes per frame. The native masked kernel was reverted; uint16 keeps
the simpler sparse correction path.

A true four-deep uint8 pipeline was also tested with four independent
compressed-input/metadata slots and four LZ4 scratch buffers. Interleaved warm
runs measured about `1.604 s` at depth four versus `1.596 s` at the retained
depth three. The extra slot lowered summed command-buffer waiting by only about
`20 ms`, while input preparation rose about `35 ms` and temporary memory grew
by roughly `0.9 GB`. Three slots remain the best throughput/headroom point on
the 24 GB M5.

Doubling each LZ4 threadgroup input window from 256 to 512 bytes was exact but
slower. Warm interleaved masked-uint8 loads averaged about `1.602 s` at 512
bytes versus `1.585 s` at 256 bytes, with generally higher summed GPU command
time. Halving refill barriers does not repay loading twice as much cache data
or the occupancy/scheduling change, so the original 256-byte window and
128-byte prefetch distance remain.

Reducing only the refill threshold from 128 to 64 bytes was also neutral. Warm
interleaved loads averaged about `1.587 s` at 64 bytes versus `1.584 s` at 128
bytes, and GPU timing crossed in both directions. The extra boundary-path work
offsets any avoided refill, so the 128-byte threshold remains.

Moving native uint16 bad-pixel correction to the CPU after GPU completion was
much worse despite only four masked detector pixels. Interleaved public loads
took `4.61-5.40 s` with CPU masking versus `2.60-2.97 s` with the existing GPU
dispatch. Four strided writes across every frame touch hundreds of thousands of
distant unified-memory pages; sparse element count does not imply sparse page
traffic. The CPU path was removed.

Replacing the explicit pre-mask buffer barrier with a second compute encoder in
the same command buffer was timing-neutral. After excluding cold/noisy outliers,
both forms centered around `2.60-2.61 s` and won individual interleaved runs.
The explicit barrier remains because an encoder split adds lifecycle complexity
without a measured synchronization win.

## Legend

| Mark | Meaning |
| --- | --- |
| Done | Implemented and signed off with real-data parity plus performance evidence. |
| Partial | Implemented or source-present, but not signed off. |
| Gap | Not implemented; real backend gap. |
| NA | Not applicable to this backend. |

## Capability Matrix

| # | Capability | CUDA | MPS | WebGPU | Notes / gap |
| --- | --- | :---: | :---: | :---: | --- |
| 1 | **uint8 source load/decode** | Done | Done | Done | All three decode bitshuffle/LZ4 `uint8` browse evidence. WebGPU low8 is allowed only after count audit proves it is lossless after bad-pixel correction. |
| 1 | **uint16 source load/decode** | Done | Done | Done | CUDA/MPS preserve native integer evidence. WebGPU decodes `uint16` source and may pack browse output to lossless `uint8` only when audited; native `uint16` masked-sum remains source-supported. |
| 1 | **uint32 source load/decode** | Partial | Partial | Gap | CUDA/MPS have partial/source plumbing but no real detector signoff. WGSL has 32-bit bitshuffle pieces, but no real acquisition path or parity gate. |
| 2 | **Detector bin, min-memory streaming** | Done | Done | Done | CUDA and MPS can bin during load without materializing the full no-bin stack. WebGPU has explicit count-preserving `detBin` source support in the local-H5 loader; full-512 and true crop-256 `detBin=2/4/8` headed parity passed on a real NVIDIA WebGPU adapter with exact corrected-frame checksums, including native non-low8 `uint16` `detBin=2`. |
| 3 | **Scan-region crop load, true crop** | Done | Done | Done | CUDA/MPS crop during HDF5 load. WebGPU slices frame windows and prefilters data files before upload/decode. |
| 4 | **Product-first region, BF/DF/ADF without full stack** | Done | Done | Done | CUDA/MPS use backend kernels over resident/chunked data. WebGPU selected-block sidecars compute exact product evidence without the full decoded browse stack. |
| 5 | **128 scan load** | Done | Done | Done | CUDA/MPS crop equality gates pass; WebGPU headed stress passes on real hardware. |
| 5 | **256 scan load** | Done | Done | Done | WebGPU true crop and selected-block product gates are exact versus CUDA and faster than the previous warm CUDA crop baseline. |
| 5 | **512 scan load** | Done | Done | Done | CUDA warm steady load meets the target; MPS is near CUDA; WebGPU local-file block-index path is exact but still above the strict full-stack target. |
| 5 | **1024 scan load** | Done | Done | Partial | CUDA and MPS have true real-acquisition no-bin `1024x1024x192x192` reference loads with bit-exact selected-frame parity. WebGPU has true-acquisition product-first BF signoff, but full-stack no-bin browser scan load still needs signoff; repeat-stress gates do not count. |
| 6 | **Minimum memory footprint / chunking** | Done | Done | Partial | CUDA streams/binning into device arrays. MPS has chunk-backed unified memory. WebGPU product-first avoids the full stack, but full-browse still materializes the decoded cube. |
| 7 | **Speed comparable to CUDA** | Done | Done | Partial | CUDA is the reference. MPS is near CUDA for one full load. WebGPU product/crop paths are CUDA-like and full no-bin browser iDPC now clears 30 FPS by median; full-browse is still short of the strict CUDA-like target. |
| 8 | **Arina-style compressed HDF5 save, `uint16`** | Done | Done | NA | CUDA uses GPU Bitshuffle/LZ4 plus direct chunk writes. MPS now uses Metal Bitshuffle/LZ4 plus HDF5 `write_direct_chunk`; full no-bin `512x512x192x192` real-data save measured about `3.9-4.1 s` and decoded sample agreement was exact. CPU/HDF5 filter fallback is correct but about `30.4 s`. |
| 8 | **Arina-style compressed HDF5 save, `float32`** | Done | Done | NA | MPS stores lossless float32 Bitshuffle/LZ4 chunks; synthetic round-trip agreement is exact. Full no-bin real-data MPS save measured about `6.8 s`. |
| 8 | **Arina-style compressed HDF5 save, `uint8`** | Partial | Gap | NA | CPU/HDF5 can write explicit clipped/rounded `uint8` display exports. This is not scientific agreement when detector counts exceed 255; real MAPED samples reached about `1600-1750` counts. MPS `uint8` needs a tail-block writer before it can be backend-native for `192x192`. |
| 9 | **1-2 s full no-bin save target** | Partial | Gap | NA | MPS compression is GPU-side, but full `uint16` save remains about `4 s`. Remaining likely floor is Python/HDF5 `write_direct_chunk` overhead for 262,144 chunks plus writing about `1.2 GB`, not CPU compression. |

## Signed-Off Evidence Map

Every `Done` cell above must be represented here. Evidence rows are public-safe:
they summarize shape, parity, stage split, and footprint without raw file paths.

| Capability | Backends covered | Evidence anchor in matrix | Parity gate | Performance / footprint evidence |
| --- | --- | --- | --- | --- |
| uint8 source load/decode | CUDA, MPS, WebGPU | `HDF5 load/decompress`, `HDF5 local-file Show4DSTEM production path`, `WebGPU corrected-frame checksum gate` | Corrected-frame integer checksums match CUDA; count audit permits WebGPU low8 browse. | CUDA warm full load median `0.450 s` across 946 runs on an RTX PRO 6000 Blackwell GPU; MPS median `0.577 s`; WebGPU block-index median `0.772 s` across 946 real Chrome rows; decoded full browse stack is `9.7 GB`. |
| uint16 source load/decode | CUDA, MPS, WebGPU | `HDF5 load/decompress`, `MPS no-bin load + VI + CoM smoke`, WebGPU source contract | CUDA/MPS preserve integer load; WebGPU source supports `uint16` decode and product kernels. | Native `uint16` raw footprint for full `512` is `19.33 GB`; WebGPU browse may pack only audited low8 output. |
| Detector bin, min-memory streaming | CUDA, MPS, WebGPU | `Seven-panel BF/ADF/DF grid`, MPS fused bin rows, `WebGPU detector-bin local-file load` | Product parity is exact for CUDA detector-bin workflows; MPS fused bin path is source-backed and covered by chunk tests; WebGPU full-512 and true crop-256 `detBin=2/4/8` corrected-frame checksums are exact against the zero-bad-before-bin reference. | CUDA bin2 reduces resident full-stack bytes from `19.33 GB` to `4.83 GB`; MPS exposes fused bin sidecar paths; WebGPU full-512 count-audited low8 page profiles were `1.199/1.212/1.106 s` for `detBin=2/4/8` on NVIDIA Blackwell WebGPU. True crop-256 repeated medians were `0.774/0.755/0.733 s` with p95 `0.798/0.813/0.775 s` for `detBin=2/4/8`; native non-low8 `uint16` `detBin=2` was exact at `2.651 s`. |
| Scan-region crop load, true crop | CUDA, MPS, WebGPU | `Real HDF5 crop-first equality gate`, `HDF5 local-file scan-region full-stack path` | Crop-first arrays/checksums match the corresponding full-load slice. | WebGPU true `256x256` crop page profile median `0.338 s` over the 946-cycle soak, with reduced compressed decode/read volume. |
| Product-first region, BF/DF/ADF without full stack | CUDA, MPS, WebGPU | `BF virtual image`, `Dense DF virtual image`, `Real crop product agreement gate`, `Product-first BF selected-block sidecar` | BF/ADF/DF integer sums exact; CoM within `1e-5`; WebGPU product max/mean abs error `0` versus CUDA. | CUDA full `512` BF/ADF/DF kernels are millisecond-scale; WebGPU selected-block BF medians were `0.210 s` for true `256`, `0.378 s` for full `512`, `1.170 s` for `1024` repeat-stress, and `4.92 s` wall / `1.56 s` product stage for true real-acquisition `1024`, without materializing the full stack. |
| 128 scan load | CUDA, MPS, WebGPU | `Real HDF5 crop-first equality gate`, `Show4DSTEM WebGPU headed stress` | CUDA/MPS crop equality; WebGPU real-adapter headed smoke with product interaction. | WebGPU `128x128` headed stress shows warm BF in a few milliseconds and idle RAF at 60 FPS. |
| 256 scan load | CUDA, MPS, WebGPU | `HDF5 local-file scan-region full-stack path`, `Product-first BF selected-block sidecar` | WebGPU crop checksums and product parity are exact versus CUDA. | WebGPU selected-block true `256x256` crop page total median `0.210 s`, range `0.185-0.246 s`; full-stack crop page profile median `0.338 s`. |
| 512 scan load | CUDA, MPS, WebGPU | `HDF5 load/decompress`, `HDF5 local-file Show4DSTEM production path`, `Product-first BF selected-block sidecar` | CUDA/MPS full-load gates pass; WebGPU corrected-frame checksum parity passes on first/middle/last frames. | CUDA warm median `0.450 s`; refreshed MPS range `0.550-0.593 s`; WebGPU full-browse median `0.772 s` and product-first BF median `0.378 s`. |
| 1024 scan load | CUDA, MPS | `HDF5 load/decompress` | Real-acquisition no-bin selected corrected frames are bit-exact against direct HDF5. | `1024x1024x192x192 uint16`, `77.31 GB` resident stack; CUDA `4.704 s` on an RTX PRO 6000 Blackwell GPU, MPS chunk-backed `4.617 s` on Apple Metal. |
| Full no-bin DPC/iDPC | WebGPU | `DPC/CoM/iDPC` | Headed real-adapter load parity passed. Browser DPC row/col uses GPU-resident display buffers with validation readback; iDPC uses paired DPC buffers plus a dual-real FFT and matches the Python reference within float32 FFT tolerance. | Full `512x512x192x192` no-bin after FFT command batching: DPC row/col/iDPC display medians `14.9/13.2/13.2 ms`; recompute medians `13.7/19.3/22.7 ms`; DPC max abs error `7.63e-6`; iDPC mean abs error `4.70e-6`, max `3.05e-5`; idle RAF `60 FPS` on NVIDIA Blackwell WebGPU. Local-file timing reruns must use `--require-local-profile` to reject URL fallback. |
| Minimum memory footprint / chunking | CUDA, MPS | Memory footprint table, CUDA detector-bin rows, MPS chunk-backed rows | Chunked/bin paths preserve the requested explicit evidence policy. | CUDA and MPS avoid unnecessary full no-bin materialization when crop/bin is requested; WebGPU full-browse is not signed off for this row. |
| Speed comparable to CUDA | CUDA, MPS | `HDF5 load/decompress`, `Seven-master HDF5 load/decompress`, MPS load rows | Same evidence policy and dtype; no hidden crop/bin. | CUDA is reference; refreshed MPS one-load median is about `0.58 s`, within the current near-CUDA band. WebGPU remains `Partial` for full-browse. |
| Arina-style compressed HDF5 save, `uint16` | CUDA, MPS, CPU fallback | `Compressed HDF5 save` | MPS-resaved full no-bin real-data `uint16` matched the decoded reference on 200,000 random samples: `mismatch=0`, `max_abs=0`. | Full no-bin real-data MPS save: `~3.9-4.1 s`, output about `1.21 GB`; CPU/HDF5 filter fallback: `~30.4 s`; chunks are `(1, 192, 192)`, filter `32008`. |
| Arina-style compressed HDF5 save, `float32` | CUDA, MPS, CPU fallback | `Compressed HDF5 save` | MPS synthetic `float32` round trip is exact against source float32 values. | Full no-bin real-data MPS float32 save: `~6.8 s`, output about `1.69 GB`. |
| Arina-style compressed HDF5 save, `uint8` | CPU fallback | `Compressed HDF5 save` | CPU/HDF5 `uint8` output matches an explicit `round+clip[0,255]` display reference. | Full no-bin real-data CPU/HDF5 `uint8` display export measured `~19.3 s`, output about `1.10 GB`; not a scientific agreement format for this dataset. |
| 1-2 s full no-bin save target | None | `Compressed HDF5 save` | Not met. Do not mark Done until full decoded HDF5 agreement and wall time are both signed off on real data. | Current MPS best is about `4 s` for full `uint16`; next work should reduce direct-chunk call count/overhead or move chunk packing/writes lower than Python loops. |

## Current WebGPU Execution Queue

These are the current follow-up targets after the 946-cycle browser soak. Each
item must produce a JSON artifact with adapter, parity, stage split, and
footprint fields before it can update a `Done` cell.

| Priority | Target | Required result |
| ---: | --- | --- |
| 1 | Full-stack WebGPU `512x512x192x192` load/decode | Current median is `0.772 s`; next win must close the remaining gap to the strict `0.5 s` full-stack target while preserving checksum parity. |
| 2 | WebGPU `256x256` crop load/decode | Current median is `0.338 s`; preserve exact crop checksum and keep crop-first IO below CUDA warm crop timing. |
| 3 | WebGPU selected-block product for `256`, `512`, and true `1024` | Current medians are `0.210 s`, `0.378 s`, and `4.92 s` wall for true real-acquisition `1024`; preserve `max_abs=0` and keep the automatic route as the default. |
| 4 | WebGPU detector-bin load path | Full-512 plus true crop-256 `detBin=2/4/8` are signed off with repeated/p95 evidence. Keep presets explicit about binning and continue p95 refreshes after decoder or upload changes. |
| 5 | Browser iDPC optimization | WebGPU iDPC is implemented and signed off against the Python reference at float32 FFT tolerance. Median display/recompute now clears 30 FPS; next target is p95/outlier tightening and stricter max-error analysis if the Python reference moves to a float32 FFT baseline. |
| 6 | True `1024x1024x192x192` acquisition on WebGPU | Product-first BF selected-block true-acquisition signoff is done with exact parity. Full-stack no-bin browser browse/load still needs either enough free WebGPU VRAM for a true full-stack run or an explicit documented memory-policy rejection. |

## Acceptance Gate Per Capability

1. **Real data only.** Synthetic tests can guard source shape, but they do not
   make a `Done` cell.
2. **Exact parity.** Corrected-frame `sum/min/max/n` integer-exact versus CUDA;
   float CoM/DPC within `1e-5`. Record max/mean abs error for products.
3. **Adapter honesty.** WebGPU timing must log the adapter and reject software
   adapters such as SwiftShader.
4. **Stage split.** Report fetch/read, parse, pack, upload, decode/GPU wait,
   compute, readback, and display separately where the path has those stages.
5. **No hidden reduction.** If bin/crop/subsample/selected blocks are active,
   they are explicit in the API/report and reflected in the shape/footprint.
6. **Footprint stated.** Report resident bytes, compressed upload/read bytes,
   sidecar bytes, and peak transient when available.
7. **Default path named.** A profiling flag does not count as shipped unless
   the production default selects it automatically and tests cover the default.

## Rejected Or Bounded Hypotheses

Do not retry these without a new reason that changes the bottleneck:

- `decodeBatch` larger than the retained defaults. Some kernels ran faster, but
  total wall time worsened from fetch/upload pressure.
- Fetch window `16/24` and broad worker/group/batch sweeps. They increased
  contention or variance; the useful wins came from decoder/layout changes.
- Full-source `uint8` compressed sidecar as the main speed path. It kept about
  the same multi-GB compressed payload and did not beat the retained local-file
  route. This does **not** reject the retained audited low8 frame decoder.
- Compressed-payload low6 sidecar. It preserved parity in a count-audited run
  but was slower than the block-index path, so low6 code is not shipped.
- High-bitplane prefix trimming for native HDF5. The useful low8 prefix was
  still about `99.55%` of the native compressed bytes.
- `queue.writeBuffer`, chunked `writeBuffer`, combined staging buffers, packed
  shared-memory low8, zero-literal barrier skip, and packed decoded-output group
  buffers. Each preserved or partially preserved parity in a profiling run but
  regressed the full-stack profile.
- A subgroup-token full-stack low8 decoder. It reduced apparent GPU wait in a
  scratch shader, but failed the corrected-frame checksum gate with all-zero
  output, so it is not shipped.

## Privacy Rule

Public docs and JSON artifacts may contain anonymized labels such as
`master-1`, shape, dtype, backend, adapter, and timing. They must not contain raw local file paths or collaborator/project-specific dataset names.
