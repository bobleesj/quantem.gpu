# Reference-66 exact ADF atlas: bounded port note

## Source identity

This note records a read-only inspection of the reference implementation. The
application repository was clean at
`48503d7fe3aa63bc42691941dc38f600df094e43`; its `Package.swift` pins the
backend to `fbd1c87264668b4feb64bf975de9a19d2bd784a8`. The corresponding clean
backend checkout was on `metal-tans-exact-interactive` at that revision.

Relevant source paths are:

- application: `Sources/Live4DSTEM/EntropySeriesController.swift` and
  `Sources/Live4DSTEM/EntropyDecodePipeline.swift`;
- backend: `Metal4DSTEMStreamingIO/ExperimentalMetalEntropySeries.swift`,
  `Metal4DSTEMStreamingIO/MetalTANSResidentSeries.swift`,
  `Metal4DSTEMStreamingIO/TANSExactTileIndex.swift`, and
  `Metal4DSTEMKernels/Resources/tans.metal`.

## What the reference atlas is

The atlas contains 489 caller-selected annulus masks. Their centres form a
two-detector-pixel lattice within 25 detector pixels of the saved ADF centre.
Each field is the complete exact UInt32 512 by 512 detector image for every
retained acquisition, packed losslessly by `TANSExactTileIndex`. It is derived
by an unseeded exact query into detached temporary outputs and audited bit for
bit before publication. It is a collection of packed 2D sums, not a duplicate
4D count volume.

Fields are built nearest-first only while the pointer and worker are idle.
Input interrupts the loop after the current field, and later idle time resumes
it. The measured reference-66 run built all 489 fields in 43,976.116 ms. Its
last recorded atlas size was 12,423,809,632 bytes; late fields took about
81--98 ms apiece. Atlas fields were rebuilt during that run. The code supports
a digest-verified disk cache for the prerequisite tile index, but exposes no
corresponding atlas import/export cache.

For a query, the planner ranks atlas masks first by centroid, then exact
word-wise mask XOR, prices the best candidates with per-column decode costs,
and uses an atlas field only when that exact base plus its signed residual is
cheaper. The selected packed image initializes the output from zero; tANS then
adds only pixels where the requested mask differs. The backend source reports
12.7--16.7 ms GPU per reference-66 frame for 2.5--3.3 pixel steps, versus
36--54 ms without that base.

## Codec and provenance boundary

The reference source is not binary-compatible with the local paired-runtime
resident. It opens an authenticated prepared archive with codec contract
`source112-tans1024-pair-v1` and sparse contract
`position9-flag1-count8-rank256-v1`. The local resident is built directly from
original indexed HDF5 into its separate paired-runtime modes, offsets, payload,
and decoding tables. Both use 16,384-scan records, 512-scan streams, native
UInt16 counts, and 1024-state pair decoding, but their record components,
sparse representation, model metadata, and buffer ABI differ. Atlas code and
packed field data therefore cannot be copied directly; only the exact-base
design is portable.

The reference archive contract requires native `<u2`, no source mask applied,
and preserved hardware counts. SHA-256 authenticates the prepared archive and
atlas append audits its generated UInt32 field against the exact query output.
Those checks establish exact behavior relative to that authenticated archive;
they do not by themselves constitute an independent comparison to an original
HDF5 file. The interactive implementation is native Metal on an Apple M5 Max
with 128 GB unified memory. It does not use Python MPS in the query path.

## What the reported approximately 100 FPS meant

One controlled 2.5-second reference-66 lockstep ADF drag used 300 pointer
inputs at 120 Hz and a centre speed of 100 detector pixels per second. It
published 267 frames: 106.36 published frames/s and 105.28 frames/s after the
first second. Only 259 frames were presented; presentation spacing was 8.333
ms p50 and 16.667 ms p95, input-to-presented latency was 35.157 ms p50 and
43.515 ms p95, and the run's 120-FPS publication gate was false. A rerun
measured 50.45 steady published frames/s, and the subsequent run measured
52.22. The approximately 100 FPS observation is therefore one publication
throughput result, not reproducible 100-FPS presentation evidence.

## Smallest local experiment

Keep the local paired-runtime codec and existing polar index. Add a separate,
explicitly bounded atlas containing exact packed 2D outputs for one ADF annulus
at a small centre lattice:

1. Begin with 64 fields at two-pixel spacing, nearest-first around the saved
   ADF centre. Stop at the byte cap; do not require a complete lattice.
2. Build one field only while idle. Run an unseeded existing polar query into
   detached outputs, pack each acquisition's seven UInt32 images with the
   existing exact field packer, and retain the mask bytes and centroid.
3. For each request, compare previous-base, zero-base, and the few nearest
   atlas bases using the existing exact plan cost. Always form the residual by
   full mask subtraction; ranking must never affect counts.
4. In the same command, initialize from zero plus the chosen packed field and
   then execute the current polar/residual path. Preserve the existing failure
   reset and output-ring ownership rules.
5. Gate the experiment independently, report actual retained bytes and build
   time, and require full-map parity plus native input-to-presented evidence.

The reference run's 12,423,809,632 bytes divided by 489 fields and 66
acquisitions is about 376 KiB per acquisition-field. Multiplying that observed
average by seven gives about 2.57 MiB per seven-acquisition field, or about 164
MiB for 64 fields and 1.23 GiB for 489. These are planning estimates only:
packed width depends on each acquisition's actual image range, allocator
alignment is not proven linear, and the local field format differs. Admission
must use measured local bytes rather than those projections.
