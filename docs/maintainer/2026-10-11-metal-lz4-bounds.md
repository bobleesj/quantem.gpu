# Metal LZ4 bounds - 2026-10-11

## Question

The CUDA LZ4 decoder bounds its reads and writes and raises `ValueError` for
damaged LZ4 bytes inside intact framing
([record](2026-10-11-lz4-read-bounds-and-chunk-walk.md)). The Metal decoders
bounded neither. What does a damaged block do on Metal, and what do the same
bounds cost there?

## Setup

- One Apple silicon laptop (M5 Max, 128 GB), macOS 26.4, Python 3.12,
  PyTorch 2.14.1, pyobjc-framework-Metal 12.2.2, h5py 3.16.0, hdf5plugin 7.1.0.
- Revisions: base `6e14725`; bounded `d75c476`.
- Metal has two LZ4 decoders on the Python path. Whole-block uint16 frames
  (192 x 192) take the fused kernel
  `h5lz4dc_unshuffle_u16_single_block_packed_h5` in `qh5idx.metal`, which
  decodes one 8 KiB block into threadgroup memory and unshuffles it into the
  output. Every other layout (uint8, uint32, a partial final block) takes
  `h5lz4dc_batched` in `bslz4.msl`, which decodes into a device scratch buffer
  that the unshuffle kernels then read.
- Data: one Arina acquisition, 512 x 512 scan, 192 x 192 detector, uint16
  (27 files; fused kernel). No acquisition for the other kernel was at hand, so
  three synthetic bitshuffle+LZ4 stacks were written with hdf5plugin from the
  first 8192 frames of that acquisition: the same counts as uint32
  (4096 frames), the detector binned 2 x 2 to 96 x 96 uint16 (8192 frames, a
  partial final block), and the counts clipped to uint8 (8192 frames).
- Timing: the decode modules of both revisions are loaded side by side in one
  process, each compiling its own kernel files. One prepared batch (the
  loader's batch: 8192 frames, 4096 for uint32) is copied into a decoder of
  each, and the decoders are launched in rotation, 300 launches each, the
  order rotating every round. GPU time is `GPUEndTime - GPUStartTime` of the
  command buffer. A second decoder of the base revision runs in the same
  rotation as a control. The two base decoders agree within 0.05% where one
  kernel is timed (the fused kernel, the LZ4 kernel alone) and differ by up to
  0.9% for the whole decode command of the scratch layouts, constantly within
  one process, so those rows are given against both.

## Before the change

Damage was written into the LZ4 bytes of one block after the host had walked
the chunk, so every length field stayed intact: the block zeroed, replaced by
random bytes, its second half zeroed, and its first 45 bytes replaced by one
literal and a match of 10219 bytes that repeats it (every read valid, the block
2 KiB too long). Four layouts, the first block of frame 2 and the last block of
the batch: 32 cases, in one throwaway process.

- All 32 decoded without an error. The Metal command completed without an
  error status, the damaged block held wrong counts (672 to 7825 wrong values
  per block), and a valid batch decoded exactly afterwards.
- The over-long match wrote outside its block in `h5lz4dc_batched`: 2254 to
  8885 bytes of the scratch changed in two neighbouring blocks, whose decoded
  counts were wrong as well, and 10005 to 12427 bytes changed past the last
  frame of the batch when the damaged block was the last one (inside the
  scratch here, because the probe allocated it twice as long as the batch).
- In the fused kernel the same stream runs past the 8192-byte threadgroup
  array. No other block's counts changed; what it overwrote is not visible
  from the host.
- A block whose length field was halved after the walk (a block that stops
  early) decoded without an error as well, with wrong counts in its frame.
- Through the public API, a master with one zeroed block on disk loaded
  without an error, as encoded counts and as scaled precision storage.

## The change

`bslz4.msl` `decompressStream` and a new
`bslz4DecompressStreamDirectBoundedToThreadgroup` in `qh5idx.metal` return
false for a stream that does not decode to exactly its block, with the bounds
of CUDA's `decompressStream`: literals lie inside the compressed stream, two
bytes remain before a match offset is read, a length extension that runs off
the stream is refused, a match copies from bytes the block already wrote
(`0 < offset <= decoded`), and writes stay inside the block. Both kernels take
an error buffer and record the lowest damaged frame of the batch and one past
the highest. `MPSDecompressor.load_prepared_frames` raises
`ValueError("Frames a to b: a compressed HDF5 chunk is truncated or corrupt
...")`; the frame numbers come from a new `frame_indices` entry of the plan
that `FrameReader.prepare` returns.

The bounds are not a line-for-line port. A refused sequence is clamped to the
room it has (`min` of its length and the bytes left in the stream and in the
block; no room at all for an offset outside `1..decoded`), so it copies only
bytes that are safe to copy, and decoding runs on to the end of the stream.
The loop has no early exit, and the result is `valid && decoded == block`.

The native Swift entry points of `qh5idx.metal` keep their decoders unchanged.

## Numbers

### Damaged input after the change

- The 32 cases above raise `ValueError` naming the frame; no byte of the
  scratch outside the damaged block changes; the next valid batch is exact.
- `tests/hardware/mps/test_streamed_h5_ans.py::test_damaged_lz4_bytes_raise_and_the_gpu_stays_usable`
  (the four damages and the block that stops early, on a 192 x 192 and a
  100 x 100 uint16 master) fails on the base with
  `zeroed: damaged LZ4 bytes decoded without an error` and passes after.
- `io.load` of a master with one zeroed block on disk raises
  `Frames 37 to 37: ...` for both representations, and the undamaged copy
  loads exactly afterwards.
- 240,000 randomly mutated blocks (byte flips, zero, 0xFF and random ranges,
  0xFF to the end of the stream, changed last bytes, crafted tokens; frames
  from empty to incompressible; three seeds, four layouts) were decoded one
  per batch and compared with a plain Python LZ4 decoder that has CUDA's
  bounds. Metal refused exactly the blocks the reference refuses (88,193) and
  named their frame, the accepted blocks (151,807) equal the reference's
  bytes, and no byte outside the mutated block changed. Each of the nine
  bounds refused blocks in every layout (3 to 14,910 blocks per bound and
  layout).

### Valid input

- Existing suites on MPS (`tests` without the CUDA, Swift, WebGPU and
  repository-layout folders; MPS, regional and real-acquisition gates on):
  base 682 passed and 138 skipped, bounded 683 passed and 138 skipped (the
  new test).
- SHA-256 of all 262,144 decoded frames of the acquisition: identical before
  and after. The first 2048 frames equal h5py's own decode. The three
  synthetic stacks: identical digests, and every frame equals h5py's decode.

### Cost

GPU time of one batch, ms, minimum / median of 300 launches. "Early return" is
the line-for-line port of the CUDA bounds, measured in the same rotation.
Percentages compare minima with the base; where the two base decoders differ,
with both.

Fused kernel (the acquisition; 8192 frames, 604 MB decoded):

| Measure | base | base, second decoder | bounded | early return |
|---|---|---|---|---|
| decode command | 5.834 / 5.844 | 5.835 / 5.844 | 6.024 / 6.032 (+3.2%) | 7.106 / 7.114 (+21.8%) |

`h5lz4dc_batched` (synthetic stacks):

| Stack | Measure | base | base, second decoder | bounded | early return |
|---|---|---|---|---|---|
| uint32 192 x 192, 4096 frames, 604 MB | LZ4 kernel alone | 4.059 / 4.079 | 4.057 / 4.078 | 4.173 / 4.198 (+2.8%) | 4.490 / 4.512 (+10.6%) |
| | decode command | 7.277 / 7.301 | 7.245 / 7.268 | 7.362 / 7.389 (+1.2 to +1.6%) | 7.674 / 7.702 (+5.5 to +5.9%) |
| uint16 96 x 96, 8192 frames, 151 MB | LZ4 kernel alone | 1.459 / 1.464 | 1.459 / 1.464 | 1.520 / 1.526 (+4.2%) | 1.719 / 1.724 (+17.8%) |
| | decode command | 2.335 / 2.340 | 2.315 / 2.319 | 2.384 / 2.391 (+2.1 to +3.0%) | 2.574 / 2.581 (+10.3 to +11.2%) |
| uint8 192 x 192, 8192 frames, 302 MB | LZ4 kernel alone | 4.176 / 4.183 | 4.176 / 4.183 | 4.403 / 4.408 (+5.4%) | 5.095 / 5.101 (+22.0%) |
| | decode command | 7.023 / 7.033 | 7.013 / 7.021 | 7.237 / 7.247 (+3.0 to +3.2%) | 7.931 / 7.940 (+12.9 to +13.1%) |

The decode command is the LZ4 kernel followed by the unshuffle kernels. An
earlier process gave the same picture: +3.2% for the fused kernel, +2.9%,
+4.2% and +5.4% for the LZ4 kernel alone, +1.7%, +2.7% and +3.2% for the three
decode commands, with the two base decoders within 0.1% of each other in
every row.

Whole calls, wall time, minimum / median:

| Call | base | bounded |
|---|---|---|
| `load_prepared_frames`, acquisition batch, ms (30 calls; the second base decoder gave 16.10 / 17.17) | 15.52 / 16.86 | 16.05 / 17.20 |
| `io.load` of the acquisition, s (8 warm loads in 4 alternating processes) | 1.574 / 1.582 | 1.565 / 1.585 |

The kernel difference predicts 6 ms per load (32 batches of 0.19 ms), inside
the 30 ms spread of the warm loads.

### Where the cost of bounds comes from

Fused kernel, the acquisition batch, minimum GPU time of 200 launches against
the base (5.836 ms):

| Formulation | ms | vs base |
|---|---|---|
| final length check and error report only | 5.816 | -0.3% |
| write bounds only, early return | 6.251 | +7.1% |
| read bounds only, early return | 6.951 | +19.1% |
| all bounds, early return | 7.101 | +21.7% |
| all bounds, early return, extension loops unchecked | 6.674 | +14.4% |
| extension loops stop by their own condition, checks return | 6.770 | +16.0% |
| the same, checks leave by `break` | 6.841 | +17.2% |
| no exit: refused sequence skipped in an `if`/`else`, read position moved to the stream end | 6.463 | +10.7% |
| no exit, no branch: lengths zeroed by select, read position moved to the stream end | 6.181 | +5.9% |
| the same, decoding continues instead of moving the read position | 6.112 | +4.7% |
| lengths clamped by `min`, decoding continues | 6.101 | +4.5% |
| the same, a match length that runs off the stream set in the extension branch (shipped) | 6.019 | +3.2% |
| the same, refusals counted in an integer instead of a bool | 6.052 | +3.7% |
| select form with unbounded extension loops (unsafe, for attribution) | 6.027 | +3.3% |

The comparisons are nearly free; the exits are not. Two early returns in the
main loop cost 7%. With every bound in place, the forms that leave the loop
cost 16.0 to 21.7% and the forms that do not cost 3.2 to 10.7%. The stream
bound inside the length-extension loops is 1.4% of the kernel.

## Conclusion

- Before the change, damaged LZ4 bytes inside intact framing loaded on Metal
  without any error, with wrong counts in the damaged block, and a stream that
  decodes to more than its block overwrote neighbouring blocks and bytes past
  the batch. Now both Metal decoders refuse the same streams as CUDA, the load
  raises `ValueError` naming the frames, and the process keeps decoding.
- Valid data decodes to the same bytes.
- The bounds cost 3.2% of the fused kernel (0.19 ms per 8192 frames) and 2.8 to
  5.4% of `h5lz4dc_batched` (1.2 to 3.2% of its decode command). A whole load
  does not change beyond its run-to-run spread.
- On this Apple GPU the cost of a bound is its control flow. The CUDA form
  (return at the first refused sequence) costs 22% of the fused kernel, against
  11% of the CUDA kernel for the same bounds; the clamped form costs 3%.
- A bound detects damage that breaks the LZ4 structure. In the randomized run
  63% of the mutated blocks were still valid LZ4 streams of the right length
  (the test frames are rich in literals) and decoded to changed counts without
  an error, on Metal as in the reference. These files carry no checksum, so no
  decoder can see that.

## Rejected

- The line-for-line port of the CUDA bounds: 22% of the fused kernel and 11 to
  22% of `h5lz4dc_batched`.
- Leaving the loop by `break`, skipping a refused sequence with `if`/`else`,
  and counting refusals in an integer: all slower than the clamped form
  (table above).
- Moving the read position to the stream end at the first refused sequence, so
  a damaged stream stops early: 1.2% slower on valid streams than decoding on.
  A damaged stream is at most its own length of extra work.
- A zero byte appended to the compressed buffer so that the length-extension
  loops need no bound (not built). A run of 0xFF bytes, as an erased region
  reads, would then send every block's loop to the end of the batch buffer.
  The bound costs 1.4%.
- A `bounded` template flag on the shared direct decoder instead of a second
  function. The clamped form differs from the original in its loop structure,
  so one body would test the flag at every bound, and the native entry points
  would no longer compile from untouched text.
- Reporting failure the CUDA way, one flag: two atomic updates on the failure
  path give the frame range at no cost on valid streams (the form with only
  the final check and the report measured -0.3%).

## Not done

- The native Swift entry points of `qh5idx.metal` still decode without bounds,
  except the family built on `bslz4DecompressFullSerialCheckedToDevice`, which
  validates and reports failure already: the kernels that call
  `bslz4DecompressStreamToThreadgroup`,
  `bslz4DecompressStreamDirectToThreadgroup`, the prefix decoders and
  `bslz4DecompressFullSerialFastToDevice`. They have no error buffer, and the
  native application was not built or measured here. The four kernels on the
  direct decoder have the topology of the fused kernel measured above.
- `MPSDecompressor.load_prepared_frames` waits for its Metal command without
  reading the command's error. No command reported one in any run here.
- The CUDA test has three damages and a separate test for the block that
  stops early; the over-long match was added for Metal only.
- The cost was measured on one Apple GPU. The exit cost may differ on other
  Apple GPU generations; the clamped form has no exit to pay for.
- No real uint32, uint8 or partial-block acquisition was timed; the three
  stacks are re-encodings of real frames.
