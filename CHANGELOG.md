# Changelog

One line per release candidate: the main user-facing thing that changed. Newest
first. Add an entry under **Unreleased** as you land a change; move it under the
new `rcN` heading when that rc is published to TestPyPI.

## Unreleased

## rc14 - 2026-10-08

Includes everything since rc5; rc6 to rc13 were published without rolling this file.

- Exact detector products (`masked_sum_exact`, `reduce_frames_exact`,
  `reduce_frames_max`) on CuPy float arrays raise `TypeError`, as on Torch and
  NumPy; they returned truncated integers (0 for 0.75) before. The float ANS
  centre of mass is the absolute detector centre like every other source, 0 for
  an empty frame, so one empty frame no longer makes the DPC field NaN.
- CUDA SSB refuses scans larger than 1024 positions per side instead of
  center-cropping them, and prints one line when it pads a smaller or
  non-square scan with its mean pattern.
- WebGPU scan-ROI patterns read float32 data as float32 and sum counts exactly
  in 64 bits; a 32-bit sum wrapped above 2^32. `applySlots` reads back adopted
  slots and slots with spare RGBA capacity.
- The CPU HDF5 reference raises when the pixel mask shape differs from the
  frames instead of skipping the mask. The CPU movie writer uses the CUDA and
  MPS grid, so its frames no longer clip the last column of a scaled grid.
- `wait=False` on `frame` and `masked_sum` completes the result on sources that
  do not queue queries; it raised `TypeError` on dense CUDA series before.

- `[cuda]` installs CuPy for CUDA 13 (`cupy-cuda13x[ctk]>=14.0`) with PyTorch 2.11
  or newer, so CuPy and PyPI torch share one CUDA 13 toolkit with uv and pip alike
  (uv picked a CuPy without NVRTC before). CUDA needs an NVIDIA driver 580 or newer.
- MPS opens Arina `uint32` masters with flagged pixels, as CUDA does: the default
  median correction replaces the 0xFFFFFFFF flagged pixels on the GPU before every
  count is checked to fit `uint16`. It raised `NotImplementedError` before.
- PyTorch 2.3 or newer is a dependency: `import quantem.gpu` loads the readers
  that return PyTorch tensors.
- Restructure the package into layers (`device`, `formats`, `resident`, `io`,
  `detector`, `geometry`, `dpc`, `parallax`, `screening`, `ssb`, `remote`), with
  CUDA, MPS and WebGPU code in `cuda/`, `mps/` and `webgpu/` beside the science it
  implements. No `backends/`, `compute/` or underscore modules remain; private
  import paths changed without aliases, public spellings (`io.load`, `io.save`,
  `SSB`, `detector.*`, `dpc.*`, `parallax.run`, `screening.prepare`) did not. The
  native Swift and Android Vulkan sources moved to `native/swift` and
  `native/vulkan`; the Android app and Direct3D folders are removed.
- `io.load` keeps acquisitions ANS encoded on CUDA and MPS; dense data is the
  explicit CPU reference (`backend="cpu", representation="dense"`). Removed: the
  Python packed family (QGPUH5 and QGIX readers, packed residents, `uint4`),
  block-indexed count-rANS, the prepared 66-acquisition series and `source112`,
  `MultiChunkedFrames`, the `scan_indices`, `random_positions`, `drift`,
  `detector_bin`, `output` and `devices` load options, and dtype casts other than
  `scaled_uint16`. Re-export old packed files from their original acquisitions
  as `.qem`. `io.inspect` reports a master whose frame count is not a square scan
  as not ready and asks for `scan_shape`.
- The remote service keeps browsing and saved SSB results; the remote SSB and
  MAPED services, `serve-ssb-mps`, `prepare-browse` and compact residency are
  removed. `SSB` rejects raw residents and never releases data it borrowed;
  screening, centre of mass, DPC and series detector queries run on MPS encoded
  data; parallax and scan rotation take the encoded acquisition.

- Fit the bright-field disk automatically once per encoded acquisition for
  `detector.bf(data)`, `detector.adf(data)` and `detector.df(data)`. Reuse the
  geometry across calls; optional center/radius overrides affect only that call.
  Mutable arrays are refitted so measurement edits cannot leave stale geometry.

- Rename `detector.mean_dp(data)` to `detector.mean(data)` and
  `detector.auto_probe(mean_dp)` to `detector.fit_probe(mean_dp)`, preserving
  the disk estimator. Update callers together; the previous names are removed.
  BF/ADF/DF now use the shared session reduction for ANS sources. Reuse fitted
  geometry across detector calls to avoid repeated mean-pattern reductions.

- Decode selected detector streams directly for CUDA streamed integer ANS
  indexing, preserving exact values while avoiding full-pattern expansion
  for detector-pixel scan images and detector crops.

- Index loaded 4D acquisitions directly with integers, slices and ellipsis to
  obtain GPU tensors, e.g. `data[10, 12]`. Integer indexing now selects scan
  rows rather than tuple fields; access storage and metadata with `.data` and
  `.metadata`. Strides decode their bounding region; advanced indexing is
  unsupported.

- Remove the `ssb.compute` compatibility imports. Remove obsolete test-path
  aliases; test commands now use current paths or suite names.

- SSB C10/C12 values and search ranges now use nanometers at the public API.
  Earlier releases passed angstrom values under nm labels: divide manually
  retained C10/C12 numbers and search bounds by 10 when migrating. For example,
  an old input of 100 now becomes 10 nm. Angles remain radians. Saved fits must
  declare nm and use the current schema; rerun old fits rather than silently
  converting their units.
- CUDA SSB previews support 1x, 2x, 3x, 4x and 8x output sampling, including
  tilt/depth correction with C10/C12. Fitting and diagnostic loss remain on
  the native scan grid. Higher-order aberrations and upsampled MPS/WebGPU
  previews are not supported. Finer sampling does not guarantee finer resolution.
- SSB.open follows the ANS-only GPU acquisition policy. Re-export older
  prepared-packed files from their original acquisitions as .qem; there is no
  packed override. Remove the unreachable packed-loading branch and update
  loading tests to use the supported source formats.
- Fix optional master-path handling in the Swift real-acquisition test so the
  Metal test target compiles.

- Share exact SSB phase and calibration as versioned JSON/NumPy pairs, with
  Python and native Swift readers, source-content matching, checksum validation
  and non-overwriting publication. No reconstruction buffers are exported.

- Recognize EMPAD-G1/G2 float32 exports and supported EMD 1 datacubes with
  versioned metadata provenance; optionally apply a confirmed mean-dark reference
  on Metal without changing original packed measurements.
- Prepare native SSB directly from packed detector columns, report preparation
  progress, preserve calibration and manual higher-order controls in saved runs,
  and use the measured blocked four-row FFT schedule for full-aperture fitting.
  Unqualified persistent-scheduler experiments remain outside production code.
- Native Metal SSB accepts uint8, uint16, and uint32 BF columns without integer
  narrowing, and saves complete versioned reconstructions with calibration,
  optimization history, source identity and image checksums for local reopening.
- Share percentile selection, display-limit conversion, and reusable-buffer
  Metal range/histogram encoding through `MetalImageRuntime`. Mixed integer
  and float batches use one caller-owned command buffer; synchronous helpers
  use the same implementation. Colormap tables remain in `MetalDisplayKernels`.
  Float statistics now expose raw `valueRange` rather than ordered reduction bits.
- Open original little-endian uint32 bitshuffle/LZ4 Arina acquisitions into the
  same lossless packed Metal residents as uint8/uint16, decoding all 32 planes
  in bounded windows with exact UInt64 detector sums and fused uint32 DPC
  accumulation. Wider compressed reads and read-ahead are registered as the
  20260909 uint32 experiments (raw evidence in the private evidence archive);
  big-endian input is rejected explicitly.
- Import validated NXem companion metadata (`_em_metadata.h5`) for Arina
  masters: regular-scan sampling with explicit length units, beam energy,
  convergence angle, camera length and reciprocal sampling, bound to the
  companion's identity for catalog invalidation. Mismatched or unitless
  metadata is reported, not guessed.
- Add `NativeScientificExport` for writing full-resolution scalar planes and a
  UTF-8 JSON metadata record into one new HDF5 file, and `countSummary()` that
  reduces the exact per-scan totals on Metal into one UInt64 without rereading
  the packed 4D payload.
- Match the CPU histogram reference to the Metal bins (`floor(fraction * 256)`,
  maximum in the last bin, constant images in the center bin) and use the
  shader's signed log1p mapping for logarithmic thresholds. A nonempty UInt32
  range of only `UInt32.max` is no longer reported as empty.

- `session.masked_sum(..., block_stride=k)` on a paired native series sums every
  k-th 512-scan block of the mask (every k-th scan row of a 512-wide raster) and
  leaves the other rows of `out` untouched, at about 1/k of the device time. The
  rows written are exact; consecutive queries at one stride build on each other
  incrementally and a change of stride starts from a full plan. A viewer uses it
  to keep every tile moving with a fast detector drag and follows it with one
  exact batch when the pointer pauses.
- The paired residual decoder refills its bit reservoir once per three coded
  pairs from a prefetched window instead of checking before every symbol, and
  the packed warp reduction biases products as it multiplies: 28 percent fewer
  instructions and 26 to 28 percent less device time per detector update on
  recorded centre drags, byte-identical outputs (`docs/performance/data/paired-decoder-2026-09-09.json`).
  Incremental masks whose change touches no index leaf skip the index pass and
  copy the previous sums.
- Add the opt-in paired-count tANS resident layout (`quantem.gpu._compact.paired`):
  `PairedCounts` codes complete 512-scan blocks with 32 Poisson pair models and an
  adaptive polar interaction index, `detector.prepare` selects the paired query
  kernels automatically, and `PairedCounts.save`/`load` reopen the exact resident
  arrays without decoding. `io.load(..., representation="paired")` streams complete
  uint16 acquisitions (one path or a list) through a direct-I/O loader whose
  shard reads run ahead across files, and reopens saved paired resident forms
  from their `QGPUPAIR` magic. `PairedCounts.decode_blocks` and `PairedFeed`
  hand whole 512-scan count blocks (optionally as float32 amplitudes) to
  reconstruction consumers from a prefetch stream; `io.inspect` reports saved
  paired forms and `PairedLoader.stream` yields public results per acquisition
  for applications; native streamed queries accept `wait=False` with
  `session.finish()` so a viewer keeps the device busy while it plans the next
  mask. The default byte-rANS layout and every existing load path are unchanged.
- Add experimental native EMPAD XML/RAW loading into lossless float32-bit
  packed Metal residents, with full-source parity, compensated BF/ABF/ADF,
  CoM and mean diffraction. Cooperative packing and reductions reuse bounded
  staging; optional source-checksum metadata never replaces full source reads.
  On an Apple M5 (24 GB), overlapped first-use hashing reduced native
  full-resident presentation from 2.11-2.39 s to 1.53-1.61 s for a public
  4.36 GB acquisition, at unchanged 4.36 GB residency. This is not a cold-I/O,
  subsecond-first-open or universal 120 Hz claim. Reproduction and limitations
  are retained as experiment 20260908-empad-first-open in the private evidence
  archive.

- Load original Arina HDF5 acquisitions (bitshuffle uint8/uint16, no crop or
  bin) directly into exact lossless block-packed Metal residents on Apple GPUs,
  with every count round-trip verified and about 2.0-2.5 GB resident per full
  512x512x192x192 uint16 acquisition. Exact detector updates use bounded
  8x8 region sums within the resident's own memory, and multi-resident detector
  updates run as one Metal submission. Packing plans (disposable per-source
  layout metadata) are written off the load thread, so a first open no longer
  pays the write on the load path. Registered the Apple M5 load-capacity
  measurement: a full load is GPU-saturated (two concurrent loads gain only
  24% throughput), so subsecond loading on that device needs a faster decode
  kernel rather than more overlap.

- Route sealed CUDA packed loading and SSB through `io.load`, add source-preserving
  browse-registry preparation, and expose authenticated prepared CoM without
  dense detector expansion or new end-to-end timing claims.
- Retain dense loading alongside direct packed readers, consolidate native
  Metal/Vulkan and WebGPU implementations, and document per-representation
  operation limits. Add packed inspection, raw-reconstruction admission,
  exactness reporting, and cross-language buffer-lifetime regressions without
  changing benchmark claims or silently transcoding between representations.
- Add one backend-neutral 4D-STEM representation API across Python,
  Swift/Metal, WebGPU, and remote receipts. `io.load()` now returns
  `Dataset4dstemGPU`, reports `lossless_packed` or `dense` separately from dtype
  and residency, auto-detects prepared Lossless Pack Format sources, and uses
  the descriptive `detector_bin` spelling while retaining compatibility aliases.
- Add a loopback-only CUDA browse service for native applications, with exact
  virtual-detector and selected-diffraction transport, acquisition monitoring,
  crop/bin admission, automatic whole-dataset placement across multiple GPUs,
  and bounded per-device resident caches. The service has no `quantem.live`
  dependency and ships through the `[cuda,remote]` install extras. A single
  `quantem-gpu-remote` Conda environment is provided for workstation setup,
  and memory admission preserves native `uint32` capacity when an Arina source
  cannot be narrowed losslessly.
- Add drift-aware sparse ptychography batches to `quantem.gpu.io.load()`: a
  shared random position set can be paired with one dense integer or
  fractional `(row, col)` drift field per source, matching the scan grid.
  Raw diffraction patterns remain unchanged and corrected float32 probe
  positions are recorded in `metadata["drift_batch"]`.
- Add `load(..., output="torch")` for direct Torch tensors on CUDA, MPS, and
  CPU, including recursive conversion of multi-dataset results. Add the
  notebook-friendly `quantem.gpu.device.profile()` diagnostic for reporting
  host, Python environment, Torch version, and the resolved compute backend
  without manual printing.
- Organize public compute around strict scientific domains: `io`, `detector`,
  `dpc`, `parallax`, `screening`, `device`, and `SSB`. CUDA, MPS, and WebGPU
  implementations now live below each domain's `compute` or `backends`
  directory; deleted flat APIs and the top-level WebGPU package are not kept as
  compatibility aliases.
- Add native four-byte unsigned (`dtype='uint32'` / `u32`) load and virtual-image
  support across CUDA, MPS, and WebGPU, and add CUDA packed `dtype='u4'`
  output for true 4-bit counts (`0..15`). CUDA and MPS selected sums use wider
  internal accumulation before float32 display output; MPS and WebGPU load paths
  preserve native `uint32` unless the caller explicitly requests `uint8` browse
  clipping. WebGPU product-first low8 sidecars now reject `uint32` sources
  instead of silently dropping high bits. Public `dtype='u4'` now means packed
  two-counts-per-byte storage with exact range audit and CUDA BF/DF/CoM kernels;
  it no longer aliases NumPy's four-byte `<u4` storage token.
- Add MPS cache-miss generation for `screening.prepare()`. CUDA still
  uses the RawKernel reduction path; MPS now streams raw HDF5 row chunks through
  chunk-backed Metal BF/DF/CoM reducers, records timing/memory metadata, and
  matched CUDA on local anonymized real-data agreement: mean DP/BF/DF exact,
  CoM max abs error `7.63e-6`, and matched rotation/radius.
- Add WebGPU GPU-resident DPC row/col and iDPC reducers to the canonical
  domain-owned Show4DSTEM WebGPU engine. The browser path now computes CoM,
  global CoM mean, centered DPC components, and fixed-rotation iDPC in WGSL,
  with direct browser agreement against NumPy/CUDA references and a local
  anonymized full-512 no-bin real-data NVIDIA WebGPU stress run. The latest headed signoff
  reports DPC row/col max abs error `7.63e-6`, iDPC mean abs error `4.70e-6`
  (`3.05e-5` max, float32 FFT tolerance), GPU-resident display medians of
  about `14.9/13.2/13.2 ms`, and full recompute medians of
  `13.7/19.3/22.7 ms` for DPC row/DPC col/iDPC on an RTX PRO 6000
  Blackwell WebGPU adapter after batching the browser FFT passes into one
  command submission. The browser benchmark harness now has a
  `--require-local-profile` guard so local-file timing runs cannot silently
  accept the URL/fetch fallback path.
- Sign off the Show4DSTEM WebGPU local-H5 detector-bin path for explicit
  `detBin=2/4/8` on full `512x512x192x192` and true crop-256 real evidence.
  The WGSL load path now zeroes raw bad detector pixels before binning and
  keeps the binned output free of raw-detector bad-pixel indices. Headed Chrome
  on an RTX PRO 6000 Blackwell WebGPU adapter matched corrected-frame integer
  checksums exactly against the zero-bad-before-bin reference, with full-load
  count-audited low8 page profiles `1.199/1.212/1.106 s` and crop-256
  20-repeat medians `0.774/0.755/0.733 s` with p95
  `0.798/0.813/0.775 s`; native non-low8 `uint16` `detBin=2` was also exact
  at `2.651 s`.
- Sign off WebGPU product-first BF selected-block loading on true
  real-acquisition `1024x1024x192x192` evidence with BF radius `30`. Headed
  Chrome on an RTX PRO 6000 Blackwell WebGPU adapter matched an independent
  Python reference exactly (`max_abs=0`, `mean_abs=0`, mismatches `0`) with
  4-run median wall `4.92 s`, page/profile `4.85 s`, product stage `1.56 s`,
  selected compressed payload `6.88 GB`, and `4.19 MB` output. This is a
  product-first signoff, not full-stack no-bin browser browse/load signoff.
- Harden the WebGPU selected-block staging uploader so reused staging buffers
  are not remapped until the previous submitted copy work has completed. The
  browser product benchmark now validates fixture existence, mounted file
  count, required reference arrays, and product-debug hooks before reporting
  timings.
- Refresh the MPS SSB performance status on a 24 GB Apple M5 reference laptop:
  radius-30
  `512x512` object steering is real-time (`10.86 ms` mean), radius-30 exact
  phase/loss is reviewable but not CUDA-like (`76.28 ms` mean), and full-active
  `512x512` exact phase/loss remains slow (`528.90 ms` mean). The docs now keep
  BF policy, object-wave steering, and exact phase/loss timing separate.
- Record true real-acquisition `1024x1024x192x192` CUDA and MPS HDF5
  load/decode signoffs: no hidden bin/crop, `uint16` output, selected
  corrected frames bit-exact against direct HDF5, `77.31 GB` resident,
  `4.704 s` wall on CUDA and `4.617 s` wall through the chunk-backed MPS path.
- Keep Show4DSTEM browser VI/DPC ownership beside the detector and DPC domains: detector
  and scan mask builders now live with the canonical WebGPU compute source, and
  the widget source-contract tests verify the frontend does not reintroduce
  local BF/DF/DPC mask helper implementations.
- Add a CUDA RawKernel virtual-image backend for resident CuPy uint8/uint16
  4D-STEM data, wire `compute_backend(cupy_array)` to it, and add exact parity
  tests against the old CuPy selected-pixel reduction. Add a
  `virtual_image_kernel_support()` probe plus a maintainer checklist covering
  CUDA, MPS, and WebGPU browser paths, including the future
  `1024x1024x192x192 uint8` target. The CUDA path now uses warp-shuffle
  selected-pixel reducers, a custom total-count reducer, fused dense
  `total - complement` output, and per-viewer detector-index caching. On a
  local anonymized full 512x512x192x192 real-data benchmark, median BF/ADF/DF drag
  latency improved from 4.96/16.16/62.64 ms on the old widget Torch path to
  1.35/3.86/1.84 ms with bit-exact output. On a local seven-tilt detector-bin2
  benchmark, per-panel BF/ADF/DF medians are now 0.54/1.35/0.53 ms with
  max absolute error 0.
- Add a CUDA RawKernel CoM/DPC reducer for resident CuPy uint8/uint16 data.
  The fused kernel accumulates total intensity, detector-row moment, and
  detector-column moment in one detector pass and caches the full-detector CoM
  field per backend. On a local anonymized full 512x512x192x192 uint16 benchmark, DPC
  CoM improved from 200.42 ms to 12.39 ms with max absolute error 0; on a
  local seven-panel detector-bin2 benchmark, first full-grid DPC improved
  from 373.14 ms to 24.63 ms with max absolute error 0, and repeated DPC reads
  use the backend cache.
- Clarify the cross-backend CoM/DPC product-kernel tracker: MPS uses raw Metal
  `com_u8`/`com_u16`, while WebGPU already has WGSL masked CoM source under
  the DPC domain but still needs the same GPU-resident buffer/cache
  parity path as virtual-image dragging.
- Add canonical reusable WebGPU/TypeScript browser compute beside each
  scientific domain. The Show4DSTEM and ShowPtycho browser engines are shipped
  as package data for widget builds.
- Fix MPS SSB fixed-aberration loss reporting for cached 512x512 geometry. The
  cached path now treats the 512 column-kernel sum-of-squares as a scalar, so
  real-data MPS fixed phase/loss and sparse optimizer parity pass against the
  CUDA reference artifacts.
- Add MPS Metal uint8 virtual-image kernels and route
  `load(..., backend="mps", dtype="u8")` through chunk-backed Metal IO, so
  Show4DSTEM browse loads do not materialize a giant Torch-MPS tensor.
- Add the MPS dense-mask `total - complement` cache path so dark-field style
  Show4DSTEM drags use sparse complement reads on Metal, matching the CUDA
  kernel strategy.
- Make CUDA SSB batch variance deterministic for sparse 256/512/1024 row
  transforms, clarify the ShowPtycho UI handoff, and document that WebGPU/WGSL
  runs in the browser while reusable source lives beside its scientific domain.

## rc5 - 2026-07-14

- Add the first documentation site with install/backend tutorials, simplify the
  tutorial language around BF/DF/ADF/DPC, add movie rendering docs, and add a
  backend coverage matrix for CUDA, MPS, CPU, and remaining migration work.
- Add an Apple Metal/MPS MP4 rendering backend so `save_mp4(...,
  backend="auto")` tries CUDA/NVENC, then MPS/Metal, then CPU/ffmpeg.

## rc4 - 2026-07-14

- Correct installed package version reporting so `quantem.gpu.__version__`
  matches the `quantem.gpu` distribution version from TestPyPI installs.

## rc3 - 2026-07-14

- Add `load(path, scan_region=(row_start, row_stop, col_start, col_stop))` as
  the crop-first HDF5 API.
- Move MPS crop-first sparse HDF5 decode and the lazy multi-dataset MPS loader
  into `quantem.gpu.io`, leaving `quantem.widget.multidataset_mps` as a
  compatibility re-export.
- Add real-data CUDA/MPS parity tests for crop-first HDF5 IO and MPS SSB sparse
  optimizer objective checks on a full 512x512 acquisition.
- Match MPS SSB fixed-preview phase output to CUDA's mean-of-per-BF-phase
  contract, tighten real-data phase parity thresholds, and add a fused
  MLX/Metal correction kernel that reduces MPS sparse objective timing from
  about 26 ms/candidate to about 7 ms/candidate on a 24 GB Apple M5 reference
  laptop.

## rc2 - 2026-07-14

- Publish the first `quantem.gpu` release candidate to TestPyPI as the
  multi-backend accelerated STEM package for QuantEM (`cuda`, `mps`, `cpu`),
  with a Quick Start README showing install, device reporting, HDF5 crop load,
  virtual detector products, and widget migration usage.
- Move the HDF5 GPU IO/decompression hot path into `quantem.gpu.io`, including
  CUDA bitshuffle/LZ4 chunk decode, pinned-buffer master loading, scan-region
  crop loading, MPS Metal bitshuffle/LZ4 chunk IO, and CPU reference decode for
  parity.
- Add device policy helpers (`device_report`, `select_device`) and import-light
  lazy exports so `import quantem.gpu` works without CUDA/CuPy installed.
- Move BF/DF/ADF, mean diffraction pattern, masked-sum, virtual image, CoM/DPC,
  and iDPC compute paths into `quantem.gpu`, with parity tests against the
  legacy widget/live paths.
- Move SSB compute APIs from `quantem.live` into `quantem.gpu.ssb`, including
  CUDA reference parity, MPS/MLX preview and C10/C12/phi12 free-fit paths, and
  real-data parity/speed checks used during migration.
- Move MPS chunk-backed product compute and movie export helpers into
  `quantem.gpu`, leaving widget responsible for frontend display/export
  orchestration.
- Wire the `quantem.widget` migration branch to depend on
  `quantem.gpu>=0.0.1rc2`, so widget HDF5 loading and accelerated products can
  call the new package without changing public widget APIs.
- Add release automation for `gpu-v*` tags, TestPyPI trusted publishing through
  `release.yml`, MIT license packaging, and an NVIDIA nvCOMP CUDA LZ4
  BSD-3-Clause third-party notice.
