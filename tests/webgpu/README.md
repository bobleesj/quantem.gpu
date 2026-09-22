# Resident rANS product parity

`rans-products.ts` exercises the browser decoder and scientific products on a
caller-supplied WebGPU device. Bundle it with the widget's installed esbuild:

```bash
/path/to/quantem.widget/node_modules/.bin/esbuild \
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

## Parallel display ranges

Bundle `parallel-display-range.ts` with `--global-name=ParallelRangeParity` and
call `await ParallelRangeParity.runParallelRangeParity(device)` on the confirmed
hardware device. The test compares ranges and normalized RGBA pixels exactly
against the retained single-workgroup region pipeline. Cases include signed
values, signed logarithms, nonfinite values, constant images, extreme finite
values, rectangular subregions, and seven slots recorded into one encoder.
It also verifies that repeated slot updates reuse the partial-range scratch.
All cases passed on NVIDIA/Blackwell on 2026-09-07; no performance assertion is
part of this correctness gate.

## Integer .qem browser admission

Run the disposable, headed physical-adapter gate:

```bash
python tests/webgpu/run_qem_browser.py \
  --esbuild /path/to/quantem.widget/node_modules/.bin/esbuild \
  --chrome /path/to/chrome
```

This generates tiny synthetic uint8/uint16 .qem acquisitions, bundles the
canonical decoder, and removes fixtures and browser state on exit. It tests
zero, constant, literal, sparse-event, and entropy streams (including escaped
counts), scan checkpoints, multiple chunks, a tail, zero-byte payloads,
patterns, detector-mask updates, ROI sums/means, and corrupted-payload rejection.
The retained rANS product test runs in the same physical browser adapter.
A passed gate is numerical browser coverage, not a UI or performance claim.

The existing `loadCountANS` / `loadCountANSFiles` entry points now admit the
QEMDATA1 envelope and runtime-column-rans-spatial-v2 integer codec. Headers and
payload chunks are authenticated before GPU upload. Only encoded tables are
prepared on the host; measurement decoding remains in WGSL. QEM detector
validity masks reach the displayed products without rewriting stored counts.
Series must share geometry, dtype, and validity masks.

Float32 QEM codecs are not supported by this browser adapter. Admission rejects
them with directions to use the native application or Python GPU session.
Old .ans containers are not accepted. The unrelated retained source112 tANS
representation is unchanged.

The browser image contract remains uint32 detector sums; sources exceeding that
bound are rejected. No universal format, full-acquisition throughput, or 120 FPS
claim follows from this small correctness gate.

The same headed gate runs `count-ans-series.ts` against current QEM fixtures.
It checks reversed acquisition order, exact batched detector/delta products,
saved validity masks and scientific metadata, and shape/dtype/mask mismatches
rejected before GPU allocation. The fixture generator also supplies the
saturated QEM acquisition for the integer readback gate below.

## Compare-preview normalization

The same `rans-products.ts` bundle exposes
`runRansDisplayNormalizationParity(device)`. Run on the confirmed hardware
adapter. It checks seven normalized display copies at two detector areas,
exact integer resident sums before/after normalization, unchanged default
single-view sum buffers, and a subsequent delta reusing the normalized copies.
It compares every preview float to the stock operation
`Float32(Float32(sum) / maskArea)`, including saturated masks with sums beyond
2^24. Preview division permits at most **one float32 ULP** difference because
hardware WGSL division need not round identically to JavaScript division.
Canonical integer sums and default unnormalized sum copies remain exact,
with zero tolerance. Linear GPU ranges allow one ULP and logarithmic ranges
allow two ULP against the CPU preview passed through the same GPU range shader.
The stock range-driven colormap shader, with a grayscale lookup table, must
produce RGBA channels within one 8-bit level. This bounds intensity-bin changes;
it does not assert that every arbitrary color palette has adjacent colors
within one level. The helper reports observed maximum errors separately.
No bitwise identity is claimed for normalized previews. Invalid areas and passing the
canonical integer buffer to the display helper must fail.

The widget should call `set.normalizeDisplayBuffers(buffers, maskArea)` once
after each rANS batch copy, before adopting compare display buffers. The next
delta refreshes these copies from canonical sums before normalization; repeated
normalization without that copy refresh is not a supported workflow. This
changes only floating-point previews, never the canonical integer counts.

## Batched direct compare canvases

Bundle `batch-canvas-parity.ts` with `--global-name=BatchCanvasParity` and call
`await BatchCanvasParity.runBatchCanvasParity(device)` on the authenticated
hardware adapter. Fake canvas contexts expose actual renderable GPU textures,
so this test requires neither desktop pointer control nor a visible canvas.
Seven distinct source shapes exercise independent render uniforms, single- and
multi-workgroup ranges, signed/nonnegative/constant/high values, linear/log
coloring, zoom, pan and smooth sampling. Every rendered RGBA byte must equal
the seven serial single-slot calls that reproduce the pre-batch submission
sequence. Each batch must submit once, versus seven submissions for serial.

Lifecycle checks reject duplicate indices and mismatched context counts, verify
empty/missing batches do not submit, and hold an unsubmitted range encoder
across an empty batch plus a throwing texture acquisition. That held encoder
must remain valid, failed presentation must not submit a partial frame, and a
subsequent normal batch must still render all seven surfaces.
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

## Preserved source112 dense/sparse kernels

Bundle `source112-kernels.ts` with `--global-name=Source112KernelParity`, then
call `await Source112KernelParity.runSource112KernelParity(device)` on a
verified hardware adapter. The fixture writer uses independent bit-by-bit
tANS packing and original compact offsets. The gate checks two batched native
16384-frame records, 67 dense and 35 sparse columns, 1/10-bit state transitions,
12-bit escapes, uint16 literals including 65535, sparse values 1..127, empty
streams, packed-position crossings, rank prefixes, full masks and mixed signed
deltas, packet-boundary patterns, paired-word gathers and carries, and rejected
truncated streams, invalid models and zero sparse values. Every count comparison
uses exact integer equality.

The complete synthetic gate, including malformed-stream rejection, passed on
NVIDIA/Blackwell GPU0 on 2026-09-07. This validates kernel arithmetic and the
binding contract; it does not establish full66 archive parity or throughput.
The 64-byte uniform and twelve-word record layout are documented alongside
`SOURCE112_WGSL` in `source112-kernels.ts`.

## Source112 CPU admission and resource lifecycle

`source112-lifecycle.ts` uses Node's test runner and in-memory metadata. It needs
no GPU or acquisition files. As with the other CPU WebGPU tests, bundle it first:

```bash
/path/to/quantem.widget/node_modules/.bin/esbuild \
  tests/webgpu/source112-lifecycle.ts --bundle --platform=node --format=cjs \
  --outfile=/tmp/source112-lifecycle.cjs
node --test /tmp/source112-lifecycle.cjs
```

The tests reject malformed metadata before allocation, verify cleanup of partial
loads and aborted admission, and exercise repeated all-panel display conversion
with one aligned uniform slab. Virtual payload files reject any attempted source
read. These checks cover ownership and command construction; they do not replace
integer parity on real hardware or establish interactive performance.

## Borrowed integer mean displays

`borrowed-count-display.ts` is a CPU-only Node test of device/extent admission,
source ownership, typed bindings and float fallback. Bundle with esbuild's
`--platform=node` and run `node --test` on the output.

`borrowed-count-display-parity.ts` exports `runBorrowedCountDisplayParity(device)`
for an identified hardware adapter. It compares integer views with canonical
GPU `f32(count) / divisor` conversion, including counts above float32 exact
integer range, linear/log scaling, smoothing, three mean areas, and both range
reduction paths. Every RGBA and range byte must agree; original uint32 counts
must remain unchanged. This is correctness evidence, not a throughput test.
