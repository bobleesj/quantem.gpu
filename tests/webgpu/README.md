# Resident rANS product parity

`rans-products.ts` exercises the browser decoder and scientific products on a
caller-supplied WebGPU device. Bundle it with the repository's esbuild (installed
by `npm ci` at the repository root):

```bash
npx --no-install esbuild \
  tests/webgpu/rans-products.ts --bundle --format=iife \
  --global-name=RansProductParity --outfile=/tmp/rans-products-parity.js
```

Load that script into a browser test page, confirm the adapter is the intended
hardware (NVIDIA/Blackwell on the development host), then call:

```javascript
await RansProductParity.runRansProductParity(device)
```

The test creates small deterministic rANS exports in memory. Two acquisitions
mix entropy-coded and literal columns, cross checkpoint and block boundaries,
and contain a masked saturated detector pixel. Point patterns, virtual images,
and scan-ROI sums and means are compared exactly against raw counts and the
stock dense backend. CoM, centered DPC, magnitude, and iDPC compare against the
same backend at an absolute tolerance of `1e-6`; iDPC includes explicit rotation
and transpose. Single-pixel and empty masks exercise the corresponding views.

A 131072-position saturated ROI tests dispatch splitting and sums above 32 bits.
A 512-column saturated detector tests weighted moments above 32 bits, with an
exact expected centroid of 255.5 pixels. No performance assertion is made.

On 2026-09-07 the real NVIDIA/Blackwell run passed all checks. CoM/DPC maximum
absolute differences were at most `1.20e-7`; iDPC at most `3.13e-7`. Exact pattern,
ROI, and wide-moment checks passed with zero tolerance.

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
