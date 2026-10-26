# Speed round after rc16: where the time went and what changed - 2026-10-11

> **Historical:** describes the code as of 2026-10-11; the fit it times (200 trials and Nelder–Mead) is retired. See the [current fit methods](../api/ssb.md).

## Question

Which waits of a 4D-STEM session on CUDA can be shortened without changing a
value: the aberration fits, the SSB session build, the mean pattern and centre
of mass of an encoded acquisition, loading a master into the encoded resident,
scan-ROI reductions and dense CuPy detector sessions? Every path was profiled
before it was changed, and every change had to leave the output bits alone.

## Setup

- One NVIDIA RTX PRO 6000 Blackwell (96 GB). Other jobs held the GPU at 90 to
  100% utilization and the 20-core host at a load average of 12 to 36 for the
  whole round. Times here are therefore two to five times those of an idle GPU
  and are only comparable inside one row.
- Python 3.12, CuPy 14.2.0, PyTorch 2.14.1 (CUDA 13), Optuna 4.9.0, SciPy 1.18.1.
- Data: one Arina acquisition, 512 x 512 scan, 192 x 192 detector, uint16 (27
  files); 8405 bright-field pixels at 300 kV and 30 mrad. A second acquisition
  (256 x 256 scan) was used for the output digests only.
- Base: `d1a01af` (rc16 plus the viewer-reduction merge). The first three
  changes were measured against `b771520`, which has the same code on those
  paths.

### How the times were taken

A shared GPU breaks the usual methods, and three observations decided the ones
used here:

1. Two processes run one after another land in different load, and one tree
   can be slower in every round. Base and candidate therefore alternate inside
   one process wherever the code allows it: the base tree's module is loaded
   from its file under another name and both are called in turn on the same
   prepared data. Rows marked "in process" were taken that way.
2. CUDA events around one launch do count other processes' time slices: a
   kernel that outlasts the rest of its time slice waits for the others before
   it finishes. A 0.9 ms kernel measured 0.9 ms and a 2.6 ms kernel 7 to 9 ms
   in the same minute. Kernel variants are therefore compared by queueing 40
   launches between two events ("queued"): the queue never empties, the share
   of the GPU stays steady, and the per-launch time is work divided by share.
3. A synchronized stage timer adds a wait for a time slice at every stage
   boundary, so stage breakdowns show proportions only.

Each row gives the minimum and the median. "Same" means equal SHA-256 digests
of the output arrays, or exact array equality where stated.

## Fixed first: the decoder's upload stream could overwrite counts still being read

`decompress_prepared` uploads compressed chunks on a second CUDA stream as soon
as a decode has more than one batch (3641 or more 192 x 192 frames at the
default batch, every block of `io.load`). That stream did not wait for work
queued on the caller's stream, and the memory pool hands a freed block to the
next allocation at once. A caller that joined decoded batches, freed them and
decoded the next batch could therefore have the upload write into a block that
its queued join had yet to read.

Reproduced with real compressed frames: decode eight batches of 2048 frames,
join them, free them, decode 4096 more frames. Frames of the joined array that
differ from a reference decoded with waits, per attempt, starting each attempt
from an empty pool on a busy GPU:

| Tree | wrong frames in 12 attempts |
|---|---|
| base | 686, 686, 686, 686, 686, 0, 686, 0, 248, 686, 686, 686 |
| fixed | 0 in every attempt |

`tests/hardware/cuda/test_streamed_h5.py::test_decode_right_after_freeing_batches_with_queued_reads_keeps_their_counts`
(real master through `QUANTEM_SSB_ARINA_MASTER`) fails on the base tree in 3 of
3 runs and passes on the fixed tree in 3 of 3. `io.load` was not affected
before this round because it waited between blocks, but the running detector
total added here queues one more kernel after that wait, so the same hazard
would have reached `io.load` too.

Fix: one event recorded on the caller's stream, which the upload stream waits
on before its first copy (`device.cuda_runtime.start_after_queued_work`). The
same call now precedes the two other private upload streams that write into
pool memory, the `.qem` loader's arena and the paired-file reader. The paired
HDF5 loader already waits for the caller's stream before its streams start,
the DigitalMicrograph loader uploads on the caller's stream, `retain` waits
for the stream, and the encoders and the save path run on the caller's stream.

Cost on an idle GPU, with and without the wait alternating in one process:

| | without | with |
|---|---|---|
| decode of 16384 frames, 20 rounds, ms | 8.41 / 8.56 | 8.34 / 8.56 |
| `io.load` of the 512 x 512 scan, 6 rounds, ms | 409 / 441 | 403 / 422 |

## Where the time went, and what changed

### Aberration fit (`find_aberrations`, 200 trials and Nelder-Mead)

Profile (cProfile around the fit, CUDA events around one loss evaluation): 211
loss evaluations take 93% of the fit, Optuna's sampler 0.58 s (3% under load,
9% of a 6.3 s fit on an idle GPU), everything else under 2%. One evaluation is
142 chunks of 64 bright-field pixels, each a row kernel (aberration correction
and row IFFT) and a column kernel (column IFFT, phase, running sums); the host
needs 2 ms of the evaluation to queue them and then waits for the loss.

| Chunk (BF pixels) | row kernel | column kernel | both |
|---|---|---|---|
| 16 | 67.2 | 37.6 | 106.6 |
| 32 | 56.3 | 21.3 | 79.2 |
| 64 (current) | 55.1 | 13.8 | 60.3 |
| 128 | 55.2 | 34.4 | 88.3 |
| 256 | 55.4 | 34.3 | 93.4 |

(ms per loss evaluation, minimum of 5, events around every launch of one
evaluation.) The row kernel is 80% of an evaluation and does not depend on the
chunk; the column kernel is fastest at the current chunk. Each evaluation moves
8.8 GB of spectra and 2 x 17.6 GB of intermediate planes and evaluates two sine
and cosine pairs per spectrum sample, and the earlier record
(`ssb-performance.md`) lists more than twenty rejected micro-optimizations of
these two kernels. No change that keeps the loss bits was found, so the fit
is unchanged: same loss, same aberrations, same phase digest.

One thing a reader of the result should know: on this acquisition the
Nelder-Mead refinement returns after its four starting evaluations. The
starting simplex is 5% of each coefficient wide, the fitted defocus is 0.42 nm,
so the simplex (0.2 A) is already inside the 0.25 A stopping tolerance.

### Tilt fit (`find_aberrations(tilt=True)`): a faster polish that is not bit-identical, not merged

Profile: of 13.4 s, 9.5 s were the Nelder-Mead polish (three starts, 1540
objective evaluations, one kernel launch and one host round trip each), 1.8 s
the two Optuna searches (batches of 8), 1.9 s the final thick-sample phase.
The batched objective kernel computes the probe geometry of every (pixel,
frequency) pair once per launch and the model once per parameter set (queued,
per launch: 5.4 ms for one set, 9.7 ms for three, 19.2 ms for eight).

Tried: the polish as a generator that yields the points it needs, the three
starts advancing side by side so each round evaluates the points of all of
them in one launch. The simplex arithmetic is SciPy's bounded Nelder-Mead step
for step (a parity test checks every requested point and the result bit for
bit on nine cases covering reflection, expansion, both contractions, shrinks
and clipped steps; an independent run of 2100 cases agreed).

| Tilt search, in process | one launch per point | side by side | values |
|---|---|---|---|
| 512 x 512 scan, 3 rounds, s | 12.62 / 12.76 | 9.22 / 9.23 | fit dictionary identical (repr), every round |
| 256 x 256 scan, 2 rounds, s | 12.08 / 12.56 | 6.79 / 6.79 | NOT identical |

On the 256 x 256 scan the two differ: C10 -13.286893 against -13.288473 nm,
tilt row 0.75873 against 0.75864 mrad, thickness 9.4096 against 9.4060 nm, fit
objective lower by 1.3e-7 of its value, final loss 0.05541149 against
0.05541126. Each variant repeats bit for bit. The cause is in the batch kernel,
not the polish: the value of a parameter set depends on its position in the
batch. Positions 0 and 4 of a batch reproduce the single-set value on every
one of 64 parameter sets; positions 1 to 3 and 5 to 7 agree with each other
and differ from it in the last float32 bit for about half of the sets (the
loop over parameter sets is compiled four copies at a time, and the first copy
rounds like the one-set loop). The 512 x 512 scan shows no difference at any
position. The Optuna stages have always evaluated batches of eight, and the
polish single sets, so the two already use both roundings; side by side, the
polish moves to positions 1 and 2, its comparisons flip at the last bit, and
it ends 1.6e-3 nm away, inside its own stopping tolerance (5e-3 nm) but
outside the 1e-4 nm that "equally good" was defined as for this round.

The change is therefore reverted on this branch and kept as commit `669a701`
(branch `perf-gpu-tilt-batched`) for a decision. `ThickSample.fit_batch`
documents "same value as `fit` row by row"; making the kernel's rounding
independent of the position would make that true and let the polish share
launches, at the price of changing fitted values in their last digits on some
acquisitions once.

### SSB session build on an encoded acquisition

Profile (stage timers): the bright-field crop (mean pattern plus decode)
130 to 190 ms, bright-field selection 35 to 49 ms, spectra G(q, k) 404 to
507 ms. Inside the spectra: gather 58, transpose 125, cast to complex 68,
FFT 195, crop to the half-plane 65, copy into G 65 ms.

Changes:

- Spectra: one kernel writes each chunk's scan images straight into the
  staging buffer as complex64 (one thread per scan position walks the chunk's
  pixels, so the reads stay inside one pattern and neighbouring threads write
  neighbouring positions), cuFFT transforms the buffer in place, and the kept
  columns are copied into G once. cuFFT gives the same bits in place and for
  any batch size (checked on real data).
- Crop read: whole scan rows are one scan range, which the decoder writes
  cropped into its output; that output is now the result. Decoding 256 MiB
  blocks into scratch and copying them cost 45% of the read.
- Mean pattern: see the next section.

| | base | candidate | values |
|---|---|---|---|
| spectra of 8405 pixels, in process, 5 rounds, ms | 435 / 448 | 313 / 321 | G(q, k) identical (digest of all 8.8 GB), same DC |
| bright-field crop read, in process, 8 rounds, ms | 135 / 179 | 77 / 99 | identical counts |
| session build, 512 x 512 scan, processes alternating, 2 rounds, median of 5, s | 2.17, 1.84 | 0.68, 0.70 | `num_bf`, detector sampling same |
| session build, 256 x 256 scan, median of 3, s | 0.52 | 0.27 | same |
| `dpc.run`, 512 x 512 scan, median of 5, s | 0.63 | 0.37, 0.42 | phase, centre of mass and rotation digests same |

### Mean pattern and centre of mass of an encoded acquisition

Why both needed a full decode: a stream holds one detector pixel over 512 scan
positions, and the spatial index holds 8 x 8 and 32 x 32 tile sums per scan
position. A masked sum is answered by tiles plus the few pixels on the mask
edge. The mean pattern needs every pixel summed over the scan, and the centre
of mass needs first moments at pixel resolution: ramps of row and column
indices have no constant tile, so every stream was decoded, twice (once per
moment), and the mean decoded every chunk to dense counts, masked them and
summed them (76 GB of memory traffic for 19 GB of counts).

Changes:

- The encoder's statistics pass reads every count anyway; it now adds them, so
  `append` keeps the exact uint64 detector total (one array of the detector's
  size per source, folded in only when the chunk is held). A source whose
  chunks come from a file sums its streams once on first request, in a kernel
  where each thread adds one stream in a register and nothing is written out.
- One kernel computes both moments from one decode of each stream.
- The index's 32 x 32 tiles are summed from the sixteen 8 x 8 tiles instead of
  from the counts a second time.

| | base | candidate | values |
|---|---|---|---|
| mean pattern after `io.load`, in process, ms | 468 / 527 | 6.6 / 6.8 | identical total; session digest same |
| mean pattern, chunks from a file, first request, ms | 468 / 527 | 207 / 253 | identical |
| centre of mass, in process, 6 rounds, ms | 508 / 525 | 221 / 266 | identical, with and without a mask |
| `append` of 8192 scans, in process, 20 rounds, ms | 27.8 / 41.6 | 25.6 / 33.2 | retained bytes same |
| `append` of 512 scans, in process, 30 rounds, ms | 21.9 / 29.0 | 22.0 / 29.1 | retained bytes same |

### Loading a master into the encoded resident

Profile (stage timers, 16 blocks of 16384 scans): host read and header walk
280 to 290 ms, GPU decode 580 to 595 ms, GPU encode and index 770 to 785 ms.
The loop read a block, then decoded and encoded it, so the GPU idled during
every read and the host during every encode.

Changes: one block is read ahead on a worker thread while the GPU works on the
current one (the worker enters the load's device so the page-locked staging
buffer belongs to that context; a block read ahead of a failed decode is
released), and the bitshuffle kernels write straight into the decoded frames
instead of into a second scratch buffer that was then copied.

| | base | candidate | values |
|---|---|---|---|
| load with the staging batch pinned to 8192 scans, in process, 8 rounds, ms | 1875 / 1966 | 1367 / 1809 | same layout digest; candidate faster in 8 of 8 rounds, median difference 200 ms |
| decode of 16384 frames, in process, 12 rounds, ms | 34.5 / 37.5 | 32.9 / 35.9 | identical counts |

The staging batch follows free GPU memory, so the number of chunks (and with it
the digest of the retained bytes) changed between runs on the shared GPU on
both trees; the counts are the same in every layout.

### Saving

`.qem` save wrote its body to a temporary file, read it back to hash it and
wrote it again. It now writes the body once, hashes each array in passing on a
second thread while the next array leaves the GPU, and copies through two
reused page-locked buffers; the Arina HDF5 writer stages its chunk bytes in
reused page-locked buffers. Measured here on an idle GPU, base and candidate
processes alternating, minimum / median of 3 saves each, page cache as found:

| | base | candidate | bytes |
|---|---|---|---|
| `.qem` save of the 512 x 512 scan (2.46 GB), s | 6.50 / 10.02, 5.29 / 12.72 | 2.32 / 4.04, 1.72 / 2.08 | body identical (SHA-256 of every byte after the header, per-block hashes, chunk table) |
| Arina HDF5 save of 65536 patterns (0.92 GB, 3 files), s | 0.54 / 0.54, 2.57 / 2.99 | 0.49 / 0.54, 0.58 / 0.59 | all files identical (SHA-256) |

Two saves of the same acquisition never have the same header on either tree:
it records the load's timings. Between the trees one more header value
differs, `physical_resident_bytes`, which is larger by 8 bytes per detector
pixel: the detector total the resident now keeps. The second base round of the
HDF5 save shows the disk, not the code.

### Scan-ROI reductions of an encoded acquisition

Every 512-scan block holding a selected position was decoded to dense counts,
the selected patterns gathered and summed, with one host wait per block. One
kernel now decodes each stream of such a block as far as its last selected
position and adds the selected counts (times how often a position is listed)
in a register.

| | base | candidate | values |
|---|---|---|---|
| mean of a 32 x 32 scan square, in process, 8 rounds, ms | 266 / 284 | 18.6 / 27.2 | mean, sum and maximum identical |
| mean of a circle of 45213 positions, ms | 2307 / 2634 | 129 / 139 | identical |

### Dense CuPy detector sessions

Since the stale-totals cache was removed, a mask covering more than half of
the detector sums its own pixels. Four kernels were compared on 65536 frames
(queued, ms per launch at the same load):

| Mask (share of the detector) | index gather (current) | byte mask | runs | runs, 32-bit lanes |
|---|---|---|---|---|
| all (100%) | 16.9 | 16.8 | 16.8 | 17.1 |
| bright field (23%) | 4.4 | 7.6 | 6.1 | 6.1 |
| annulus (64%) | 12.1 | 13.0 | 14.0 | 14.0 |
| dark field (77%) | 13.7 | 14.2 | 14.2 | 13.5 |

The time follows the bytes read whatever the kernel does: these reductions are
bound by memory bandwidth, and the current kernel is as fast as any. The old
speed came from reading only the complement and subtracting it from kept
totals, which is only right while the array does not change. That is now an
explicit promise: `detector.prepare(array, immutable=True)` keeps each
pattern's total (masks covering more than half of the detector read only what
they leave out) and the full centre of mass. The default reads the counts on
every call, and `test_cuda_session_over_a_filling_buffer_reads_current_counts`
still passes.

| 65536 frames, in process, 60 rounds, ms | default | `immutable=True` | values |
|---|---|---|---|
| bright-field image | 6.9 / 9.8 | 5.5 / 9.9 | same |
| annular image | 13.5 / 20.7 | 8.2 / 10.5 | same |
| dark-field image | 17.1 / 21.6 | 7.7 / 10.1 | same |
| centre of mass, repeated | 14.9 / 22.4 | 0.02 / 0.03 | same |

### Import and first-use compile time

`python -X importtime`: `import quantem.gpu` and `import quantem.gpu.io` each
take 1.16 s, of which PyTorch is 0.84 s and Numba (the HDF5 header walk) 0.12 s.
Kernel modules compile on first use and CuPy keeps them in its on-disk cache:

| Module | empty cache, ms | cached, ms |
|---|---|---|
| streamed counts codec | 471 | 13 |
| paired counts codec | 932 | 12 |
| HDF5 bitshuffle and LZ4 | 265 | 9 |
| dense detector | 457 | 8 |
| SSB FFT 256 / 512 / 1024 | 209 / 381 / 700 | 12 / 12 / 13 |
| thick-sample fit batch | 89 | 10 |

Nothing was changed here. Importing Numba where the header walk is first used
would take 0.12 s off sessions that never read an HDF5 master.

## The repository benchmarks, rc16 against the final tree

`tests/benchmarks/data_paths.py` (median of 7 after a warm-up) and
`scripts/benchmark_ssb.py` (median of the warm repeats), rc16 (`b771520`) then
the final tree, two rounds, on the 512 x 512 scan, started while the GPU was
idle. Every digest is equal between the trees in both rounds, on this scan and
on the 256 x 256 one, including the digest of the retained encoded bytes.

| ms | rc16, rounds 1 and 2 | final, rounds 1 and 2 |
|---|---|---|
| `io.load` to the encoded resident | 536, 531 | 371, 396 |
| prepared decode, 65536 frames | 131, 97 | 88, 89 |
| `append` of 4096 / 512 scans | 3.17 / 0.66, 2.53 / 0.60 | 2.10 / 0.62, 2.12 / 0.62 |
| encoded mean pattern | 254, 112 | 0.08, 0.08 |
| encoded centre of mass | 115, 195 | 50.0, 50.2 |
| encoded BF / ADF / DF image | 11.2 / 3.7 / 2.7, 3.7 / 3.6 / 2.7 | 3.5 / 3.5 / 2.6, 3.6 / 3.5 / 2.7 |
| encoded exact BF image | 0.76, 2.37 | 0.60, 0.67 |
| encoded scan-ROI mean (32 x 32) | 11.6, 85.2 | 1.34, 1.38 |
| dense mean / BF / ADF / DF / centre of mass | 4.7 / 10.1 / 10.0 / 10.4 / 7.9, 4.4 / 3.1 / 3.1 / 3.4 / 3.1 | 4.4 / 3.1 / 3.1 / 3.4 / 3.1 in both |
| dense, `immutable=True`: BF / ADF / DF / centre of mass | - | 2.4 / 2.4 / 1.8 / 0.01 in both |
| paired mean / BF / DF / centre of mass / scan-ROI mean | 0.08 / 2.2 / 6.2 / 29.2 / 18.1, 0.10 / 2.3 / 2.3 / 16.7 / 7.6 | 0.07 / 1.7 / 1.6 / 15.5 / 5.7, 0.07 / 1.9 / 1.6 / 15.9 / 5.8 |
| SSB reconstruct, without / with loss | 16 / 42, 16 / 44 | 16 / 44, 17 / 45 |
| `find_aberrations`, s | 8.03, 5.95 | 5.95, 6.00 |
| `find_aberrations(tilt=True)`, s | 4.52, 3.86 | 3.90, 3.92 |
| `parallax.run` with the aberration fit | 330, 333 | 214, 214 |
| SSB session build | 449, 283 | 306, 308 (first build 168, 241) |
| `dpc.run` | 113, 112 | 203, 130 |

Two rows need a caution. The GPU did not stay idle: another job came and went
during the runs, and with two processes each gets half of the GPU.

- Session build. Ten builds in a row, timed one by one after the table was
  taken (GPU shared at the time): rc16 640 ms for most builds with single
  builds between 363 and 551; final 389, 154 to 167, then 300. With stage
  timers the final tree's unshared build is 137 ms (crop read 37, selection 7,
  spectra 91) and its shared builds 292 ms (62, 23, 200): every stage doubles
  together, which is the other process, not a stage. The 283 ms of rc16 and the
  137 to 167 ms of the final tree are the builds that ran alone; they were not
  taken in the same minute.
- `dpc.run`. The table has the final tree slower although its centre of mass,
  nearly all of the call on the GPU, takes half as long in the same runs (50
  against 115 to 195 ms). Three later rounds, trees alternating on the shared
  GPU, median of 7: `dpc.run` 369, 372, 400 ms for rc16 against 141, 211,
  270 ms; `center_of_mass` alone 229, 191, 229 against 118, 59, 118 ms. No
  cause for the table's two values was found; this row should be taken again on
  an idle GPU.

The dense rows of rc16's first round were disturbed the same way; its second
round is the one to read. On an idle GPU a dense dark-field image takes 3.4 ms
on rc16 too: the 7 ms of the rc16 record was a kernel outlasting its time
slice, not the kernel.

Under load (the state of most of this round; every digest equal there too):
session build 2166, 1843 ms against 676, 703 ms; `io.load` 2777, 2348 against
1822, 2093 ms; encoded mean 892, 754 against 11, 9 ms; encoded centre of mass
819, 605 against 373, 335 ms; scan-ROI mean 477, 385 against 33, 31 ms.

## Metal

The Metal LZ4 decoders now bound their reads and writes as the CUDA decoder
does; the behaviour of damaged blocks before and after, and the cost on an
Apple silicon laptop, are in
[the Metal record](2026-10-11-metal-lz4-bounds.md). The released-row fix below
was run there too (33 passed, 3 skipped across the counts, float and resident
suites with the Metal backend).

## Two fixes found on the way

- The mean pattern and the detector total cover every scan position. After
  `release_rows_before` they summed the rows still held and divided by the full
  scan count, on CUDA and Metal, for counts and floats (and the paired layout
  returned the total from before the release). They now raise, as a read of a
  freed row and a scan-ROI reduction over one already do
  (`test_mean_pattern_of_released_rows_raises` in the counts and the float
  suites fail on the base tree with "DID NOT RAISE").
- A partially arrived source raises for the same requests instead of
  returning the mean of the scans so far divided by the full count.

## Conclusion

- No output bit changed on any merged path: every digest of
  `scripts/benchmark_ssb.py` and `tests/benchmarks/data_paths.py` is equal
  between rc16 and the final tree on both acquisitions, as are the in-process
  checks above. The digest of the retained encoded bytes follows the staging
  batch, which follows free GPU memory: under load one base run of five loaded
  a different layout with the same counts.
- A decode right after freed batches could return wrong frames in them; that
  is fixed and tested.
- Neither fit is faster. The standard fit's time is the loss kernel's, which
  is at the floor of its design. The tilt fit's polish can share launches (27%
  to 45% of the fit), but the batch kernel rounds differently by batch
  position, so that change waits for a decision.
- A session build no longer decodes the acquisition for its mean pattern,
  reads the bright-field crop in half the time and builds the spectra 28%
  faster.
- On an idle GPU: mean pattern after a load 112 to 254 ms to 0.08 ms; centre
  of mass 115 to 195 ms to 50 ms (one decode pass instead of two); a 32 x 32
  scan-ROI mean 12 to 85 ms to 1.3 ms; `io.load` 0.53 s to 0.37 to 0.40 s;
  `.qem` save of 2.46 GB from 5.3 to 6.5 s at best to 1.7 to 2.3 s.
- Dense sessions cannot be faster than reading the selected counts; the
  complement shortcut is available to callers that declare the array immutable.

## Rejected

- Loss kernel chunk sizes other than 64 bright-field pixels (table above).
- Remembering loss values of repeated aberration sets: 2 of 211 evaluations
  repeat (the best trial is the first simplex vertex, and the result's loss is
  the last vertex), under 1% of the fit.
- Evaluating several Nelder-Mead candidates of one polish speculatively
  (reflection, expansion, both contractions per launch): the batch kernel costs
  2.1 ms per parameter set beside 3.3 ms per launch, so four sets cost more
  than the one or two a step uses.
- Overlapping Optuna's sampling with GPU work: a batch's suggestions need the
  previous batch's losses.
- A real-to-complex FFT or two one-dimensional FFTs for the spectra (half the
  transform work): neither gives cuFFT's two-dimensional complex bits (largest
  difference 1.6e-2 on spectra of order 1e5), so every loss would change in its
  last bits.
- Smaller staging chunks for the spectra (64 to 512 pixels, to stay in cache):
  no consistent gain over the 2 GB chunk in six alternating rounds.
- Dense masked-sum kernels that read contiguously with a byte mask, read runs
  of selected pixels, or add in 32-bit lanes (table above): bandwidth-bound,
  none faster than the index gather.
- Answering the centre of mass from the spatial index: a ramp over an 8 x 8
  tile is not a tile sum, whatever offset is taken out of it.
- Ranking trees by whole processes run one after another on the shared GPU:
  the same tree moved by 40% between rounds.

## Not done

- The scaled-precision source still decodes one pattern per selected scan
  position for a scan-ROI reduction (`resident/cuda/precision.py`); its float64
  running total fixes the order of the additions, so a batched version needs
  its own bit-level check.
- The spatial index holds no first moments, so the encoded centre of mass
  still decodes every stream once. Per-scan moments kept while encoding (24
  bytes per scan position) would answer the unmasked centre of mass without a
  decode; that is a design decision about what a resident keeps.
- The paired layout keeps its two-query centre of mass and block-wise scan-ROI
  reductions; its residual decoder adds one weighting per pass.
