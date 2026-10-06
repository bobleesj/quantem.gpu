# quantem.gpu restructure: what was removed, moved and fixed

Date: 2026-10-05. Branch `gpu-refactor`, from 3a8abad6.

## Question

Can quantem.gpu be organized into layered, plainly named modules, with no legacy code, while every
number that survives stays bit-identical and MAPED stays under 16 GB?

Owner's direction (2026-10-04): no `from __future__`, Python 3.11 to 3.14, no file names starting
with an underscore, no legacy ("this is a very new code so it's meant to change a lot"), a good level
of modularity; callers in quantem.widget and quantem.live are updated rather than kept compatible.

## Setup

- CUDA: RTX PRO 6000 Blackwell (Linux), torch 2.13 + CUDA 13.0, CuPy 14.2 (cuda12x),
  Python 3.14 and Python 3.11.
- MPS: Apple M5 Max, Python 3.14, full suite per wave.
- Every wave: full CUDA suite on both interpreters and the MPS suite compared per test (junit XML),
  MAPED's three test files with real seven-tilt data, MAPED peak memory, annotation checker for
  Python 3.11 evaluation, import-layer checker.

## Size

| | 3a8abad6 | after |
|---|---|---|
| Python lines in the package | 89,556 | 48,121 |
| files with an underscore name | 42 | 0 |
| `from __future__` | 68 | 0 |
| import cycles between modules | 6 (closed by imports inside functions) | 0 |
| wheel | 5.4 MB (swift, vulkan, libhdf5.a inside) | 1.45 MB |

## Layout

Imports go downward only: `device` < `formats` (GPU-free files) < `resident` (data held on the GPU
and its exact reductions) < `io` (load, save, discover, inspect, read) < `detector`, `geometry` <
`dpc`, `parallax` < `screening` < `ssb` < `remote` < `cli`; `optics`, `display`, `movie` need only
`device`. Backend code sits in `cuda/`, `mps/`, `webgpu/` under the package that owns the science;
kernel and shader files sit beside the Python that compiles them. `swift/` and `vulkan/` moved to
`native/` (out of the wheel); `android/` and the Direct3D projects were deleted.

## Removed (nothing reached it)

`io.load` on a GPU returns encoded residents only (`StreamedCounts`, `MPSStreamedCounts`,
`FloatANSResident`), so these were unreachable or used only by test fakes:

- the packed QGPUH5/QGIX/uint4 family and the PACKED representation;
- the test-only block-rANS format and the seven-tilt rANS artifacts;
- the 66-acquisition prepared series and the source112 archive (Python, CUDA, TypeScript);
- dense, cropped, sparse, random, sharded and u8/u4 GPU load routes and their options;
- the MPS dense loader, lazy multi-dataset series and `MultiChunkedFrames`;
- the remote SSB and MAPED services (no client called them; every compute route failed on real
  data); the browse service, `/api/ssb/saved-results` and `serve` stay for Live4DSTEM;
- compatibility shims, aliases, dead optics functions, environment switches, `SourceIntegrity`.

## Fixed

| What | Before | After |
|---|---|---|
| `io.load` on CUDA under GPU contention (`retain()` copied pooled arrays without waiting; added in 3a8abad6) | wrong counts in non-final chunks: 26/40 test runs at 3a8abad6 | 0/40; 160 probe loads clean |
| Browse `scan_bin`/`det_bin`/crop (Live4DSTEM) | HTTP 500 | exact, equal to dense-then-bin |
| `serve` without `--implementation-revision` (Live4DSTEM-linux) | refused | starts |
| `screening.prepare(backend="mps")` (quantem.live) | NotImplementedError on every cache miss | works; integer products equal CUDA |
| `dpc.center_of_mass` on MPS | NotImplementedError | exact, byte-equal to CUDA |
| `detector.prepare([...])` on MPS | ImportError cupy | works |
| `SSB(loaded.data)` then `close()` | released the caller's acquisition | refused at construction; borrowed data never released |
| `parallax.run` / `geometry.rotate_scan` on `io.load` output | TypeError | work, bit-identical to the dense path |
| MPS virtual-image results | aliased a reused buffer | owned by the caller |
| Metal buffers per query (100 rounds, 100 images) | +85 MB | flat |
| Screening (256² and 512² scans) | 0.97 s, 2.91 s | 0.63 s, 1.56 s |
| Show4DSTEM view construction (bounded mean) | 3.2 s CUDA, 6.4 s MPS | 0.13 s, 0.14 s |

## Numbers

Every surviving path was compared before and after: frozen tests, SSB fingerprints (72 CUDA, 57 MPS
products at 128-512 scans), detector products (400, CUDA, NumPy, Torch, MPS), screening and browse
products on real acquisitions (bytes), encoded load checksums on real data, MAPED frozen shifts and
merge sums (19 tests). MAPED seven-tilt `run()` peak with row release: 11.3-11.5 GiB (unchanged).

## Rejected ideas

- Compatibility shims for moved private modules: rejected; quantem.widget, quantem.live and denoise
  were updated instead (widget and live committed on `gpu-refactor-callers` branches; denoise left
  uncommitted for review).
- Reproducing the `retain` race in-process with a busy CUDA stream: never reproduced; it needs a
  second process loading the GPU (a second process running large matrix products while loading).
- One `ArrayBackend`/`TorchBackend` for NumPy and Torch: NumPy input lost float64 accumulation, so
  the NumPy path keeps its own float64 backend.

## Fixed after the restructure

The same day, three parallel fix branches (merged into `gpu-refactor`) fixed every pre-existing bug
the restructure had reported. Numbers of correct paths stayed bit-identical; each fix below changes
its numbers on purpose and has a test that failed before.

| What | Before | After |
|---|---|---|
| SSB automatic `det_sampling` (no `det_sampling` given) | `2 * semiangle / R` with three radius rules: 1.0909 (CUDA), 1.1287 (MPS array), 2.3077 (MPS dataset) mrad per pixel on an Arina acquisition calibrated at 0.554 | `semiangle / R` with one half-plateau disk radius on CUDA and MPS: 0.5524; abTEM known answer 0.546875, new 0.5486 (+0.32 %) |
| CUDA SSB result loss before a fit | batched estimate, 0.0566 against exact 0.1405 (256 scan) | always the exact objective |
| MPS SSB fit | re-prepared its own evidence from the source | uses the session's prepared state |
| MPS SSB export of a NumPy session | `AttributeError` | exports; read-back reconstructs within 6e-10 rad |
| CUDA float virtual images of saturated frames | uint32 accumulator wrapped (65534.0 for a sum of 4,295,032,830) | exact uint64 sum, then float32 (4.295033e9) |
| Metal detector sums | int32 atomics | uint64 accumulation |
| CUDA weighted sums and centre of mass | 32 lanes of weight times count in int32 (detectors wider than 1024 could overflow) | exact weight digits, as MPS |
| MPS selected-frame sums and maxima | ignored flagged detector pixels | zero them, as CUDA does |
| `masked_sum_exact(output="native")` on precision or float sources | float image | exact counts, or refused for float sources |
| Parallax upsampling (`upsampling_factor=2`) | summed spectrum tiled: every odd row and column 0 | each image tiled and shifted on the finer grid, as QuantEM's `DirectPtychography.reconstruct`; equals a float64 oracle to 2.4e-7 |
| Parallax shifts | applied twice (image matched "shifted twice" to 6e-3) | applied once (float64 single-shift oracle to 2.3e-7) |
| Parallax repeatability | float32 atomics: shifts differed between identical runs (up to 1.6e-4 px) | fixed-order sums: bit-identical |
| CUDA NVENC movie under GPU load | 5 of 40 runs encoded a frame four frames early | 0 of 40; encoder copies on the kernels' stream |
| Metal movie labels | 16 pt (CUDA and CPU 24 pt) | one size on every writer |
| Paired direct I/O without `cuda.bindings` | unaligned staging, 6 tests failed | page-locked through CuPy; `io.load` of a 256 x 256 Arina scan in that environment 3.0-3.9 s to 0.2-0.5 s |
| WebGPU contract tests | errored (no esbuild) | `npm ci` installs the pinned JS tools; tests run |
| CUDA `io.load` of a bitshuffle/LZ4 `uint8` master | wrong counts without an error (16,319 of 16,384 values differ); MPS refused the file | one-byte unshuffle on CUDA and MPS: equal to h5py for full, partial and multi-block frames; `uint16`/`uint32` decode byte-identical |
| Five MPS tests failing since 3a8abad6 | stale expectations, a missing `pandas` on the test Mac, a temp-path assumption | pass |
| Docs naming removed features, `.ans` containers, compatibility shims | present | removed or rewritten; the `.ans` container leaves no trace outside dated records |

**Correction.** Every SSB reconstruction made with automatic detector sampling before this fix
used about twice the physical sampling: on the Arina acquisition only 2,460 of the 8,878 selected
bright-field pixels kept a nonzero aperture weight, and the fitted defocus moved from 1.71 nm to
4.16 nm once corrected. Re-run such fits, or pass the calibrated `det_sampling` explicitly. Details:
`docs/maintainer/2026-10-05-ssb-detector-sampling.md`.

## Standards and anonymization pass

The restructure held `src/` to the standards; `tests/` and `scripts/` still carried 64
`from __future__ import annotations` lines and two underscore file names. The future imports are
gone (an annotation checker run on Python 3.11 finds 0 problems in 209 files), and
`scripts/_benchmark_support.py` and `scripts/_webgpu_cdp.py` are `benchmark_support.py` and
`webgpu_cdp.py`. Scripts inside the dated experiment records, now kept in the private evidence
archive, keep their imports as recorded.

Retained experiment records named computers by private host nicknames and one dataset by its
device description. They now say `maca`, `macb`, `macc` and `cudahost` (single words, so the registry's
public ids such as `macbook-pro-m5-max-128gb-webgpu-resident-dpc-row` come out unchanged) and
`arina-device-master`. Both registry validators pass. The four fingerprinted scripts changed only
by the import line and the rename (syntax trees otherwise identical), and
`docs/performance/evidence_manifest.json` records the new fingerprints. The docs guard against
host nicknames compares sha256 digests, so the test no longer spells out the names it refuses.
Every image in the tree (10 documentation figures, 8 notebook outputs) shows the public gold
dataset; the HDF5, EMD, NumPy and `.qem` fixtures are synthetic.

## Open for the owner

Decisions:

1. MAPED merges made with quantem.gpu between 3a8abad6 and the `retain` fix while the GPU was shared
   may have used corrupted tilt counts: re-run them.
2. quantem.widget folder viewer now loads every master encoded on one GPU (0.1-2 GiB per 512²
   master, no paging or multi-GPU spreading).
3. quantem.live `cpu_stream` ptychography keeps sparse amplitude targets: upstream quantem.live
   reads them through encoded scan-row reads, bit-identical to the dense targets. denova Series
   peak 19.8 -> 32.9 GiB at full field (11.2 s vs 16.2 s); denova's uint32 HDF5 with sentinels
   above 65535 now needs `hot_pixel_correction="zero"`.
4. One schema-1 `.qem` file in a denoising reproduction run still needs the schema-1 reader;
   delete the reader once that file is re-exported.
