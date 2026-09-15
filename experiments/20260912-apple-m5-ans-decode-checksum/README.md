# Decode versus image accumulation

FC14 is an opt-in diagnostic, not an image implementation. Every selected
stream is decoded with existing bounds, escape and terminal checks. A lane
accumulates the decoded values locally, followed by one SIMD sum per packet.
This removes per-pair SIMD sums, shared partial accumulation and image writes.
Each packet checksum is compared to the wrapping UInt32 sum of 512 ordinary
image pixels. This checksum check is weaker than full-image parity.

Seven full uint16 `(512,512,192,192)` acquisitions remain resident. No crop or
binning. Masks contain positive contributions only, including the symmetric
difference region of small detector movements. These are not signed drag
updates. Each ordinary control resets to zero before timing the absolute mask;
checksum execution is bracketed by two full-image controls. First command is
warmup, leaving four repeats. All-seven return includes host work; per-source
GPU timings are also retained in results.jsonl.

| Region | Image control A1 ms | Checksum diagnostic ms | Image control A2 ms |
|---|---:|---:|---:|
| BF full | 331.95 | 299.96 | 332.23 |
| BF small changed region | 9.92 | 7.62 | 9.20 |
| ADF full | 831.19 | 728.45 | 814.66 |
| ADF small changed region | 25.01 | 21.88 | 22.99 |

Removing image reduction does not eliminate the dominant latency. The decoder
and its selected-stream traversal remain the main target. Timing subtraction
is not an exact decomposition: removing operations changes compilation,
register use and scheduling; host output work also differs. No cache/occupancy
counter or theoretical latency floor was measured. No production speedup is
claimed from this ablation.

71,680 packet checksums and 140 repeated full-image comparisons passed.
A separate ordinary trajectory passed 280 frozen independent full-map hashes.
Resident representation is unchanged. Checksum output is 2048 bytes/source;
temporary selected-index and coefficient buffers are also allocated, not a
decoded 4D volume. The installed application was not modified.
