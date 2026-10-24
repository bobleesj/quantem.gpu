# QEM conversion and complete lossless verification, 2026-10-23

## Question and workload

Can a full 2048 × 2048 scan with 192 × 192 detector pixels be converted and
verified without the previous thirteen-minute wait? The acquisition contains
154,618,822,656 unsigned integer measurements (288 GiB at its original uint16
precision). All six flagged detector pixels remain part of the comparison.
There is no cropping, binning, correction, sampling, or relaxed tolerance.

Private acquisition identifiers are replaced by acquisition labels and shapes. These labels
identify matched inputs in the private experiment, not public downloadable data.
The matrix covers all eleven acquisitions, from 256² × 192² through
512² × 96², 512² × 192², 1024² × 192², and 2048² × 192² inputs. The 96²
case is a native detector shape, not a downsampled acquisition.

## Reusable measurement

Run `scripts/benchmark_qem.py SOURCE --backend mps --output PRIVATE_REPORT.json`
or select `--backend cuda`. The command performs the canonical conversion,
full source comparison, and three separate encoded QEM reloads. Verification
includes reopening the saved file; the nested load stage is not an additional
stage to add to the total. Saving includes the canonical writer's durability
work. Every generated QEM is held in temporary storage and removed on exit.

For native Swift/Metal, build `bash scripts/build_qem_roundtrip.sh`, then select
`--backend native-metal --native-binary BINARY`. Native loading, saving, and
reopening are timed by the existing executable. Its own checks compare seven
complete diffraction frames, full BF/ADF/DF images, masks, and metadata. The
runner additionally compares **every saved count** against the original source
using the Python MPS verifier. This last timing is explicitly MPS, not native
Swift verification. Three additional native QEM reopens use fresh processes. The native wall time
includes product checks and should
not be ranked against the Python conversion total as an identical workload.

Run the complete collection with `scripts/benchmark_io_scale.py SOURCE_FOLDER
--backend mps --output-dir PRIVATE_DIRECTORY --reps 3` (one shell command).
Use `cuda` or `native-metal` for the other independent reports, supplying the
native executable for Swift. The resulting backend reports are reviewed in
[MPS](2026-10-23-io-mps.md), [CUDA](2026-10-23-io-cuda.md), and
[Swift/Metal](2026-10-23-io-swift-metal.md).

The runners do not evict OS caches. First observations are not cold-disk
measurements. CUDA and Apple measurements use different devices and storage;
these rows compare implementations on their respective systems, not GPU-only
throughput. Raw reports include source paths and belong in the private archive.

## Implementation

Profiling the original verifier on the 256² input found 7.35 of 10.30 seconds
in h5py source decoding and another 1.49 seconds in int64 conversion. The MPS
candidate instead decodes original bitshuffle/LZ4 frames on Metal, prefetches
the next bounded batch, decodes saved ANS counts, and compares unsigned values
without discarding high source bytes. Only block mismatch counts return to the
host. Two source buffers and one saved-count buffer bound staging memory.
Source-file SHA-256 computation also uses four independent workers.

Eligibility is deliberately narrow: native unsigned 8/16/32-bit source counts,
one complete frame per chunk, bitshuffle/LZ4 as the sole filter, compatible
stored precision, and no pixel correction. Other layouts and correction
references retain the existing exhaustive verifier. CUDA now uses the same eligibility gate, bounded GPU decoding, source prefetch,
and exact unsigned comparison. Its reference baseline remains separately recorded.

Negative tests introduce high-byte errors at batch boundaries and in flagged
pixels, require the exact mismatch count, and compare with the original h5py
reference. Failure tests cover both GPU-command and prefetched-source errors
and require buffer allocations to return to baseline.

## Evidence and limitations

The private experiment is `20261023-qem-backend-verification`. The implementation
is an uncommitted candidate based on `d2db0bdb4c9859230d3e9ba2f6b23fcfae8eb0f1`;
its patch, file fingerprints, commands, hardware reports, and raw measurements
are retained with that experiment. This is a source-tree benchmark, not an
installed-wheel or application UI qualification. The measured final matrix fingerprint
is `8efd434d9626dd30cfffaeb377fcafd6ca2042b4748ce2910605457398fdc201`.
A subsequent import-order and line-wrapping cleanup in `io/convert.py` gives
`502b85399b55501f0a4aaec4fd44ce02d5e01ac8b7c2e0b03fb475c5ba72312b`;
only import ordering and wrapping changed; numerical kernels and comparison logic are unchanged. Both source snapshots
are retained; timings remain attributed to the measured snapshot.

## MPS before and after

The physical device is Apple M5 Max with 128 GB unified memory. The first
optimized full-size observation completed conversion and verification in
68.65 s, versus 780.25 s for the earlier canonical converter. Exhaustive
verification fell from 698.95 s to 24.50 s, including a 4.97 s QEM reopen.
Two further exhaustive checks took 25.03 and 26.80 s. All three compared
154,618,822,656 values with zero mismatches. Repeated QEM loads took 5.03,
3.43, and 3.42 s. The saved file was 38,332,680,910 bytes.

A subsequent failure-cleanup fix drops buffer references retained by exception
tracebacks; it changes no comparison arithmetic. That build's full-size
conversion took 83.95 s: source load 25.42 s, save 21.79 s, and verification
28.39 s (including 5.45 s to reopen). Its reloads took 4.10, 2.77, and 2.72 s.
Every value again matched exactly. This run shared storage bandwidth with the
input transfer for CUDA testing. The earlier and later observations are retained
separately; save-time variation is not attributed to an unchanged writer.

The later scale run completed the full-size canonical conversion in 58.62 s,
including 20.69 s of exhaustive verification. Its repeated raw HDF5 load
median was 11.82 s, versus 23.18 s for a separate fresh-process load with an
empty frame-index cache. These cache boundaries and storage observations stay
separate from the earlier conversion measurements above.

The original verifier was independently checked against source data through
h5py; the optimized path uses a different GPU source decoder. On the 256²
profile case, 2,415,919,104 values matched in 0.30 s versus 10.30 s for the
original verifier (verification call only; QEM creation excluded, its reopen included).

## Acceptance scope

The final focused MPS collection, streamed-HDF5, benchmark-runner and public-repository
checks passed 48 tests, with one CUDA-only lifetime check skipped on MPS. The
physical CUDA collection checks passed 16 tests, with two Metal-only lifetime
checks skipped. The final native acceptance gate passed all nine tests. Broader physical MPS acceptance reports 145 passed, six failed,
and 23 blocked tests. The failures are one uint64 fixed-SSB case and five FP8
product cases, reproduced on the unchanged base in the preceding loading
experiment. CUDA acceptance reports 145 passed, five failed FP8 product tests,
and 23 blocked tests. Missing public collections remain blocked, not passed.
These gates cover more formats than the integer acquisition benchmark and
prevent treating its success as universal-format or release qualification.

## Native writer storage fix

The initial full-size Swift/Metal round trip failed for insufficient disk space:
the original writer staged an entire body and then copied it into the final
container, temporarily requiring two copies. The candidate plans the exact
header size with fixed-length checksum placeholders, writes the encoded body
once, and fills in the authenticated header before synchronization and atomic
publication. It retains the existing on-disk format and all block checksums.
Cancellation removes the unpublished temporary file; an existing destination
is never replaced. The small real-acquisition test passed cancellation cleanup,
existing-file protection, native reopen, and full independent count comparison.

## Matched CUDA verification

On the RTX PRO 6000 Blackwell Max-Q Workstation Edition, the reference converter
compared every count exactly but took 1,170.90 s for verification (1,345.16 s
for conversion plus verification). Its observed source load, save and QEM
reopen times were 53.53, 84.21 and 31.18 s. These storage-sensitive observations
were taken during other storage activity and are not an isolated disk baseline.
The candidate then verified the **same saved file on the same physical GPU**
using the new bounded CUDA source decoder in **45.14 s and 49.49 s**. Both candidate checks again compared
all 154,618,822,656 values, with zero mismatches. The raw timing records remain
separate from the scale table, which uses the other recorded physical CUDA GPU.
The reference QEM was retained by a temporary hard link for this comparison and
removed after both checks.
