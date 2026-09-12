# Exact compact resident series

There are two CUDA source paths behind the existing `io.load` and
`detector.prepare` entry points. Ordinary H5 can be explicitly loaded into a
runtime encoded resident with `representation="encoded"`; portable encoded files retain
their original encoded storage. Equally shaped acquisitions can be prepared
as a list for joint native queries. See the
{ref}`H5 workflow <cuda-h5-ans-residency>`
for the current options and save/conversion limits.

The prepared-series path below preserves the earlier specialized deployment.
Its extraction measurements do not qualify the general H5 path. The source
integration retains both implementations and the existing dense/packed APIs;
it does not establish a new full-series loading or displayed-frame-rate result.

## Prepared-series source

Canonical format labels are independent of source identity. The loader also
recognizes the exact historical labels by their SHA-256 fingerprints, then
applies the same layout, completion, checksum and kernel compatibility checks.
Existing prepared records need no rewrite or re-encoding. The public query ABI
is `compact-series-v1`; native consumers must pin the corresponding installed
implementation digest when accepting a resident owner.

The private `_compact` backend packages the previously validated native prepared-series
query implementation. Its first supported input is a completed
`compact-prepared-series-v1` **query-ready** folder containing two record packs
and its own global metadata. It retains the full
`uint16[66,512,512,192,192]` logical source. Its dense codec is exact paired tANS;
sparse events retain original counts. Fine/coarse indexes accelerate queries.

This initial format fixes the acquisition count, detector shape, column ordering
and index layout. It does not yet encode arbitrary H5 inputs, save a new archive,
build indexes from a source-only archive, or support multi-GPU sharding.
Existing H5 and other QuantEM input paths are separate and remain available.
Unsupported compact operations must fail explicitly.

## Scientific contract v1

- Axes are acquisition, scan row, scan column, detector row, detector column.
- A binary detector mask is intersected with the stored original validity mask.
  Fractional annuli use float64 distances and inclusive inner/outer boundaries.
- Virtual sums return every acquisition and every scan position as exact uint32.
  `192 * 192 * 65535 < 2**32`; no count truncation is possible in this shape.
- Point diffraction returns original uint16 counts from the same scan position
  across all acquisitions. It does not zero original invalid-pixel counts.
- Each calculation is serialized on its selected device and completed before
  returning. Native default outputs own separate arrays. An explicit `out`
  borrows caller storage; callers must finish reading before reusing it.
- A delta seed refers to the immediately previous complete virtual image.
  A private previous-image workspace keeps that baseline separate from mutable
  caller-owned outputs (66 MiB for this series). Before writing into another
  output slot, copy that private baseline. Snapshot the completed calculation
  before returning it. Neither caller mutation nor slot rotation changes state.
- Full-scan mean DP, multi-position DP reductions, CoM and weighted masks are
  currently unsupported. No approximation or alternate calculation is substituted.

## Implementation and provenance

`_compact/provenance.json` identifies the active CUDA source hashes. The first
extraction preserves them byte for byte. Querying has no Python acquisition
loop. GPU compaction has one small scalar handoff; planning runs in a compiled
C++ routine. CPU planning is deliberately preserved in this extraction.

The planner builds once with `CXX` (default `c++`), requires C++17 on Linux, and
caches its library by source/compiler/platform identity under `XDG_CACHE_HOME`.
Importing the planner does not compile it. CUDA code uses CuPy's normal kernel
cache. Neither compiler executes checkpoint-supplied source code.

Metadata checks run before device allocation. Record packs stream through four
bounded page-locked slots with direct reads and one host-to-device copy into
final source allocations. Upload events fence slot reuse, including on errors.
The two packs can remain on one disk or be relative symlinks to separate disks.
Whole-payload hashing is a separate verification task, not hidden in the hot
load path. Small metadata checksums are always verified.

Extraction evidence is registered in `2026-lossless4d`, experiment
`0908-06-packaged-resident-backend`: 30 complete all-66 arrays equal independent
original-source references. In 100 identical fresh moving/resizing queries,
original/extracted wall p50 was 5.148/5.146 ms and p95 11.760/11.743 ms.
This is an extraction parity result, not a speedup or displayed-FPS claim.
The loaded GPU1 owner and all source addresses were preserved.

## Public workflow

```python
from quantem.gpu import detector, io

loaded = io.load("prepared-series", backend="cuda", device=1)
session = detector.prepare(loaded)
images = session.masked_sum(mask, output="native")
patterns = session.frame(256 * 512 + 256, output="native")
```

`device` uses CUDA's process-visible numbering. With physical GPU1 selected by
`CUDA_VISIBLE_DEVICES`, that device is logical 0. The source determines query
placement even if another CUDA device is current when a caller makes a request.

`session.series_shape` is `(66,)`; scan and detector shapes stay `(512,512)` and
`(192,192)`. `masked_sum(..., output="native")` and
`masked_sum_exact(..., output="native")` both preserve uint32 device counts.
The NumPy defaults remain float32 and uint64 respectively. `frame` returns
uint16. Native output is currently qualified only for the compact CUDA backend;
other backends explicitly reject this option until they implement its contract.

Applications can supply `out=buffer` with native output to rotate bounded device
buffers. Scientific source/index memory is never an allowed output destination.
Request IDs and lease release belong to the application transport; they are not
scientific cache keys. `session.timings` excludes output transport and display.
`session.backend_metadata` supplies format, query ABI, installed package version,
and an implementation digest captured when the resident source is constructed.
