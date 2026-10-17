# Resident rANS product parity

`rans-products.ts` exercises the browser decoder and scientific products on a
caller-supplied WebGPU device. It runs in the headed `.qem` gate below on
fixtures that `make_qem_browser_fixture.generate` writes with the public encoder:
two acquisitions of a 32 x 32 scan on a 3 x 4 detector whose last pixel is
saturated and masked, cutting chunk and checkpoint boundaries. Point patterns,
virtual images, and scan-ROI sums and means are compared exactly against raw
counts and the stock dense backend. CoM, centered DPC and iDPC must equal the
same backend exactly (both divide the exact integer moments once); iDPC includes
explicit rotation and transpose. Single-pixel and empty masks exercise the
corresponding views.

A 131072-position saturated ROI tests dispatch splitting and sums above 32 bits.
A 512-column saturated detector tests weighted moments above 32 bits, with an
exact expected centroid of 255.5 pixels. No performance assertion is made.

On 2026-10-10 the real NVIDIA/Blackwell run passed every check with zero
tolerance. Before the .qem CoM divided its exact moments once (as the dense path
does), CoM and DPC differed from the dense backend by up to `1.20e-7` (one float32
ulp) and iDPC by up to `3.13e-7`.

## Integer .qem browser admission

Run the disposable, headed physical-adapter gate:

```bash
python tests/webgpu/run_qem_browser.py \
  --esbuild node_modules/.bin/esbuild \
  --chrome /path/to/chrome
```

`--expect-adapter` fails the gate on any other adapter, such as a software
fallback. On Linux, the Vulkan flags and `VK_ICD_FILENAMES` select the vendor's
device, for example:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python tests/webgpu/run_qem_browser.py \
  --esbuild node_modules/.bin/esbuild --chrome /usr/bin/google-chrome \
  --expect-adapter nvidia/blackwell --chrome-arg=--enable-features=Vulkan \
  --chrome-arg=--ignore-gpu-blocklist --chrome-arg=--disable-gpu-sandbox
```

This generates tiny synthetic uint8/uint16 .qem acquisitions, bundles the
canonical decoder, and removes fixtures and browser state on exit. It tests
zero, constant, literal, sparse-event, and entropy streams (including escaped
counts), scan checkpoints, multiple chunks, two blocks in one chunk (the second
starts mid-word once staged on the GPU), a tail, zero-byte payloads, patterns,
detector-mask updates, ROI sums/means, and corrupted-payload rejection.
The retained rANS product test runs in the same physical browser adapter.
A passed gate is numerical browser coverage, not a UI or performance claim.

`RansResidentSet.loadQemFile` / `loadQemFiles` admit the
QEMDATA1 envelope and runtime-column-rans-spatial-v2 integer codec. Headers and
payload chunks are authenticated before GPU upload: each 64 MiB chunk's stream
bytes are copied into resident GPU groups of at most 256 MiB only after its
SHA-256 matches, so the payload is read once and only hashed bytes are
decoded. A chunk larger than one group is split across groups on a word
boundary; each block a split cuts is copied whole on the GPU from the
authenticated groups. Nothing is read after authentication. The gate repeats
the two-block fixtures with group limits below one chunk on the real device.
Only encoded tables are prepared on the
host; measurement decoding remains in WGSL. QEM detector validity masks reach
the displayed products without rewriting stored counts. Series must share
geometry, dtype, and validity masks; this is decided from the authenticated
headers before any payload is read. `qemHttpFiles` serves the same files from a
Range-capable server, streamed through one uncached response per file.

Float32 QEM codecs are not supported by this browser adapter. Admission rejects
them with directions to use the native application or Python GPU session.

The browser image contract remains uint32 detector sums; sources exceeding that
bound are rejected. No universal format, full-acquisition throughput, or 120 FPS
claim follows from this small correctness gate.

The same headed gate runs `qem-series.ts` against current QEM fixtures.
It checks reversed acquisition order, exact batched detector/delta products,
borrowed `imageViewsU32` views equal to `readImageU32`, a `RansResidentSeries`
(one resident set per acquisition) driving the same compare-grid batch and
delta exactly, saved validity masks and scientific metadata, and
shape/dtype/mask mismatches rejected before GPU allocation. The fixture generator also supplies the
saturated QEM acquisition for the integer readback gate below.

## Exact quantitative detector readback

Use `await set.readImageU32(tilt)` for quantitative detector sums. It returns
a caller-owned snapshot of the canonical integer buffer in GPU queue order.
`readImage(tilt)` retains its float32 display behavior, which can round integer
sums above 2^24. Display normalization never modifies the integer buffer.

Generate the small saturated fixture with the public encoder (CPU only):

```python
import numpy as np
from quantem.gpu import io

io.save("saturated.qem", np.full((1, 1, 256, 256), 65535, np.uint16),
        format="quantem", compression="ans", backend="cpu")
```

Bundle `rans-integer-readback.ts` with
`--global-name=RansIntegerReadbackParity`, then run
`await RansIntegerReadbackParity.runRansIntegerReadbackParity(device, fixtureURL)`
on the verified hardware adapter. Here `fixtureURL` points to `saturated.qem`.
The zero-tolerance gate checks sums 16842495 and 4294836225, addition/removal
deltas, queue-ordered snapshots, unchanged counts after preview normalization,
float32 compatibility, and invalid acquisition indices.

The historical integer gate passed on NVIDIA/Blackwell on 2026-09-07. That
result predates the QEM container migration and does not certify the current
reader; run the current headed QEM gate above for a fresh result.

## Scan-ROI patterns of dense residents

`reduce-frames-parity.ts` exports `runReduceFramesParity(device)`; the headed
QEM gate above runs it. It reduces two-chunk uint16, uint32 and float32 stacks
whose integer per-pixel sums exceed 2^32 and compares every sum and mean
exactly against float64 sums rounded once to float32. A 32-bit accumulator
wraps these sums, and reading float32 data as packed uint16 counts fails them.

## Dense HDF5 loading and virtual images

`dense-h5-parity.ts` runs in the same headed gate on Arina-layout bitshuffle-LZ4
files that `make_qem_browser_fixture.generate_h5` writes with h5py and hdf5plugin
(uint8, uint16, uint32 and float32 at 64 x 64 and 96 x 96, uint16 at 128 x 128,
one masked hot pixel). A 96 x 96 frame is not a whole number of bitshuffle
blocks, so it ends in a shorter block with its own plane stride. The counts
include 255 < count <= 65535 and full uint16 frames. Every check is exact
against the raw counts:

- `runDenseDecodeParity`: the full-stack decode at native width, and the uint8
  output saturating at 255.
- `runProductFirstParity`: the widget's first virtual image, summed straight from
  the compressed chunks (pixel and group-mask kernels, full, disk, annulus, one-
  and two-pixel masks, a scan region). Counts above 255 need all 16 bit planes.
- `runResidentParity`: virtual images and centers of mass of the resident stack
  the widget builds from the same files, down to one-pixel masks.
- `runDenseVirtualImageParity`: totals above 2^24 on the direct and the
  large-aperture complement path equal the exact integer sum rounded once.

## SSB bright-field subsets

`ssb-bf-subset.ts` builds a synthetic 128 x 128 SSB folder (441 bright-field
uint16 columns, resident G(q,k) in chunks of 256) and runs in the same headed
gate. After a full-BF load, reconstructions at 100 and 300 BF pixels (drag
estimator and committed mean phase with loss) must equal an engine prepared with
only that subset to 1e-6 rad and 1e-7 relative loss (float32 summation order),
and returning to the full count must reproduce a fresh full engine exactly.
`runSsbBuildPeak` prepares 1793 bright-field columns (8 chunks) while counting
live GPU buffer bytes: the build peak must not exceed the bytes the ready engine
keeps, so the full-plane G(q,k) of every chunk is never resident at once.

## Decoder pipelines per device (CPU-only)

`pipeline-device-cache.ts` is a Node test with fake devices: a device created
after a device loss must compile its own decoder pipelines instead of reusing
the lost device's. Bundle with esbuild's `--platform=node --format=esm` and run
`node --test` on the output.

## .qem loading contracts (CPU-only)

`tests/contracts/test_webgpu_qem_loading.py` bundles each of these with esbuild
and runs it with `node --test`; no browser, GPU or real data is used.
`fake-gpu.ts` supplies a device with real mapped memory and recorded
allocations, and `qem-synthetic.ts` builds authenticated one-pixel .qem files
whose payloads cross 64 MiB chunks and the 256 MiB group limit cheaply.

- `qem-http.ts`: served files read through uncached HEAD and range requests
  admit the same streams as a local File; servers that ignore ranges, missing
  files and names outside the folder are rejected.
- `qem-checksum.ts`: transferred chunks retain exact bytes and SHA-256 hashes,
  out-of-order jobs return to the right caller, worker startup restrictions use
  the original WebCrypto path, and cancellation/runtime failures release workers
  without admitting unverified bytes. The headed browser gate additionally tests
  real worker transfers; `runQemChecksumParity(false)` tests a page with
  `Content-Security-Policy: worker-src 'none'`.
- QEM admission builds exact decoder checkpoints as complete GPU groups and
  their tables authenticate, overlapping later reads. The resident reuses those
  checkpoints, offsets and tables. Split chunks retain the original decoder
  path. `qem-resident.ts` tests ordering, reuse, cancellation and later-corruption
  cleanup; the real-browser gate rejects invalid ANS states even with valid SHA.
  `checkpointPrebuiltGroups` records the early groups; `checkpointMs` measures
  only the final pass, not total overlapped GPU work. Use `readyMs` for load timing.
- `qem-stream.ts`: fragmented BYOB and default streams arrive as exact chunks,
  short and long responses are rejected, stopping early cancels the download,
  admission verifies every chunk from one ranged response with four local reads
  in flight, and read buffers are recycled (at most four, never across streams)
  only after their checksum matched.
- `qem-resident.ts`: stream bytes are staged once and bound with their byte
  offset, tables equal a plain admission, failures release every staged buffer,
  oversized chunks are split with their cut blocks copied on the GPU while
  bytes served after authentication never reach the decoder, uploads and
  staged segments keep unit rows in group order, groups stay within 256 MiB,
  series forward staged payload per acquisition, and borrowed count views
  address the canonical uint32 images.
- `rans-batch.ts`: a compare-grid batch over separately loaded sets integrates
  every set's own image in one submission.
- `qem-series-lookahead.ts`: a series publishes its first file at once and the
  rest in file order with three loads in flight, rejects mismatched headers
  before any load, stops loads in flight without waiting for them, and releases
  every unpublished set on failure, mismatch, disposal or cancellation.

## Bounded payload read-ahead (CPU-only)

Bundle `rans-read-ahead.ts` with esbuild's `--platform=node --format=esm`, then
run the output with `node --test`. No browser, GPU or disk payload is used.
The controlled byte source resolves chunks out of order; tests compare every
copied byte, confirm at most four retained 32 MiB reads, and cover nonzero
ranges, tails, short reads, copy failures and out-of-order read failures.
Pending reads settle before a failed copy returns, so no late copy can touch a
released mapped destination.

The loader uses this fixed window without a new API option. At most 128 MiB of
payload-read staging is retained in addition to the currently mapped GPU
payload group. `payloadReadMs` sums overlapping per-request elapsed durations;
`payloadReadWaitMs` counts exposed await intervals and is not their sum.
These unit tests establish correctness and bounded concurrency only. Compare
real FileList loading separately before reporting any load-time improvement.
