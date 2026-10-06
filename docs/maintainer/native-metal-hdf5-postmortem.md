# Native Metal HDF5 Loader Postmortem

Date: 2026-08-13

Hardware: Apple M5 Max, 128 GB unified memory

Evidence: `512x512x192x192`, source-native `uint16`, three external HDF5
shards, no scan crop, detector binning, or precision reduction

## What was slow

The fused Bitshuffle/LZ4 decoder was not the main problem. The application
called otherwise reasonable kernels in a wasteful full-volume topology:

1. Decode the `uint16` source into a clipped `uint8` volume while auditing the
   source count range.
2. Discover that millions of valid counts exceed 255, discard the complete
   9.66 GB `uint8` volume, and decode the same compressed HDF5 data again into
   exact `uint16`.
3. Materialize a complete 19.327 GB scan-major `uint16` volume, then perform a
   separate serialized 19.327 GB scan-major-to-detector-major transpose.
4. Wait for batch command buffers inside the submission loop, preventing HDF5
   decode, detector-product work, and layout conversion from overlapping.

The first pass was scientifically cautious but operationally wrong. A full
speculative conversion is not a dtype probe. On the center tilt, the exact
audit found a maximum count of 1833 and 17,420,309 unmasked values above 255,
so the speculative `uint8` result could never be retained.

This is the central lesson: profile the whole pass graph and data lifetime
before micro-optimizing an individual kernel. A fast kernel called twice over
9.66 billion detector values is still a slow loader.

## Retained topology

The native application now chooses the source dtype before allocation and
performs one exact `uint16` decode. Each bounded batch:

1. maps compressed HDF5 bytes;
2. decodes exact counts into a scratch buffer;
3. computes the exact count audit, BF/ABF/DF products, and mean diffraction;
4. signals a shared Metal event; and
5. transposes a tiled batch directly into its final detector-major offset on a
   second command queue.

Eight 16,384-frame scratch slots keep the two queues busy without a complete
scan-major intermediate. The transpose moves 32x32 tiles with 16x16
threadgroups. Count auditing aggregates in threadgroup atomics and performs
only bounded global atomic updates instead of one global update per detector
value.

GPU work summed across concurrent queues can exceed wall time. That is expected
and must not be reported as a regression without first checking queue overlap.

## Result and parity gate

| Measurement | Before | Retained path |
| --- | ---: | ---: |
| One full tilt | `2.486-2.960 s` | `0.944 s` median, `0.928-0.950 s` |
| Resident result | `19.327 GB` exact `uint16` | `19.327 GB` exact `uint16` |
| Estimated one-load peak | about `38.65 GB` | about `28.99 GB` |
| Full-volume decodes | two for a `uint16` source | one |
| Full scan-major temporary | yes | no |

The final five-run timing uses fresh processes, one at a time. Total executable
launch averaged about 1.20 seconds, including roughly 0.25 seconds of
SwiftUI/catalog startup outside the measured loader.

Parity is exact on the retained path:

- source dtype remains `uint16`;
- center-tilt maximum count is 1833;
- center-tilt unmasked count above 255 is 17,420,309;
- center-tilt selected-frame hash is `ce268593ec0ac969` before and after;
- all seven tilt hashes and count audits are unchanged; and
- BF/ABF/DF use exact integer sums.

## Mandatory prevention checklist

Before editing a CUDA, MPS/Metal, or WebGPU kernel:

1. Draw the production pass graph from file or resident input to the
   user-visible output. Include fallback and audit branches.
2. List every full-volume allocation, read, write, conversion, transpose, and
   readback with its byte count. Treat every extra 10-20 GB pass as a primary
   suspect.
3. Select dtype and scientific evidence policy before allocating the output.
   Never run a complete lossy/compact conversion merely to discover whether it
   can be retained. Fuse an exact audit into the native pass or reuse a trusted,
   invalidated-with-the-source audit cache.
4. Time process startup/catalog/index preparation separately from mapping,
   decode, products, layout, finalization, and total user wall time.
5. Search for synchronization inside submission loops. Prefer bounded
   pipelining and explicit dependencies when independent stages can overlap.
6. Compute both resident and peak transient memory. A topology that needs two
   full raw volumes is not acceptable merely because each kernel is fast.
7. Freeze parity before tuning: dtype, count range, corrected-frame hashes,
   product sums, frame order, and bad-pixel behavior.
8. Benchmark fresh and warm states separately, and run fresh-process trials
   sequentially. Concurrent index helpers can contend for the same cache and
   invalidate timing.
9. Optimize individual kernels only after the topology, synchronization, and
   memory-traffic audit is complete.

## Bounded experiments

- Increasing the streaming batch to 32,768 frames reduced available pipeline
  depth and regressed wall time to `1.022-1.066 s`; retain 16,384.
- The useful improvement came from removing redundant passes and overlapping
  stages. Smaller launch-shape changes were secondary.
- A compact `uint8` browse path remains valid only for a native `uint8` source
  or an explicit, already-trusted count-range contract. It must never silently
  replace exact `uint16` microscope evidence.
