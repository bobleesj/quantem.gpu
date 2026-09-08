# Exact merged float32 compression capacity

**Result: the tested candidates still do not fit a 24 GB Mac.** No native
NumPy loader or new Metal format was implemented or promoted in this run.

The original is a complete C-order `(512, 512, 192, 192)` float32 NumPy array,
38,654,705,792 bytes including its 128-byte header. Its SHA-256 is
`24d7ec1fe83ec12faddf026a92b8f97f003202dc68a326664ec35f3a0d8c7057`.
No values were rounded, clipped, binned, cropped, or converted to float16.

| Representation | Bytes | Evidence |
|---|---:|---|
| Original float32 values | 38,654,705,664 | Full source length and hash |
| Existing 128-word XOR layout | 37,637,571,104 | Prior complete capacity audit |
| Byte shuffle + Zstandard level 3 | 30,921,859,858 | Every block compressed and decoded exactly |
| Index for the tested independent eight-frame blocks | 262,152 | 32,769 UInt64 offsets |

The new CPU compression result is about 20% smaller than dense values and 17.8%
smaller than existing XOR storage. **It is not a GPU-resident implementation or
a measured loading improvement.** It remains above physical memory before
decoder state, staging, app, OS, and Metal alignment. This experiment does not
prove that every possible lossless codec must fail.

## Method

`probe.py` selects 32 stratified eight-frame blocks using seed 20260908, covering
37,748,736 original bytes across the entire scan. Six candidates combine raw,
byte-plane or bit-plane ordering with no predictor, neighboring-scan XOR, or
modular UInt32 subtraction of neighboring scan frames. Predictors operate on
float **bits**, not rounded numeric values. Each compressed block is decoded,
its transformation inverted, and compared byte-for-byte with its source.

The best sample candidate was byte-shuffle/Zstandard-3, projecting 31.01 GB.
`full_audit.py` then applied that exact candidate to all 32,768 independent
blocks. It authenticated the complete source and verified all 9,663,676,416
decompressed words. Compressed blocks were discarded after checking, not saved
as a second full dataset. This CPU audit ran beside the authoritative source,
not as a Mac performance benchmark.

The first sampling attempt failed due to a noncontiguous inverse byte view.
`failed-attempt.json` preserves that failure; the corrected run repeated every
round trip without relaxing exactness. No result was claimed from the failure.

Script SHA-256:

- `probe.py`: `c3dd7eb730ae1c9ca9a3fdcb461587bd6f5761738cbc3faab7833252d38ce081`
- `full_audit.py`: `f61778fbf70c1593e39dbdd6a459e81d9e712c60ed816a523379de74ea79d7b6`

## Consequence for native support

Do not add this codec to the native loader as a solution to the memory target.
A useful follow-up must either demonstrate a substantially smaller exact
representation first, or explicitly obtain a different memory/streaming
contract. Disk-backed or preview-only access cannot satisfy full-resident
loading acceptance. A lossy representation requires a separate explicit
scientific decision and cannot replace the source silently.
