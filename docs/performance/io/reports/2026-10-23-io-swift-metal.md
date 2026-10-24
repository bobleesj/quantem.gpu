# Native Swift/Metal IO benchmark, 2026-10-23

Physical device: **Apple M5 Max**. Source fingerprint: `8efd434d9626dd30cfffaeb377fcafd6ca2042b4748ce2910605457398fdc201`.
Runtime: Apple Swift 6.3.3, release build, macOS 26.4; 128 GB unified memory.

Private experiment: `20261023-qem-backend-verification`. All input files were SHA-256 matched across systems.

This report keeps first observations separate from the median of repetitions two and three. OS caches were uncontrolled; these are not cold-disk results. Three repetitions are a small diagnostic sample, not a tail-latency guarantee.

Every successful conversion compares every source count exactly, including flagged pixels. Loading uses native precision and explicitly disables pixel correction; no cropping or binning is applied.

[Shared definitions and implementation](2026-10-23-qem-verification.md). Reproduce with `scripts/benchmark_io_scale.py SOURCE_FOLDER --backend native-metal --output-dir PRIVATE_DIRECTORY --reps 3` plus `--native-binary QEM_ROUNDTRIP_BINARY`.

Native HDF5 timings include catalog preparation and full ANS residency with spatial indexes. Repetitions use fresh processes and reuse the per-acquisition catalog directory. QEM reloads use the native authenticated reader. Full source verification is independently performed by Python MPS and is labeled separately.

QEM measurements retain their original source snapshots: `8efd434d9626dd30cfffaeb377fcafd6ca2042b4748ce2910605457398fdc201`. The later driver revision makes the raw-count loading policy explicit and benchmarks the two expanded copies; it does not change their GPU arithmetic.

Initial QEM reopen is measured inside verification (Python) or the native round trip. The dedicated reloads occur afterward; their first observation is not the first-ever opening of the file. An unavailable isolated reopen timing is shown as a dash.

## Loading

| Acquisition | Scan × detector | HDF5 first (s) | HDF5 repeated median (s) | Initial QEM reopen (s) | Dedicated QEM reload 1 (s) | QEM repeated median (s) |
|---|---|---:|---:|---:|---:|---:|
| acquisition-03 | 256² × 192² | 0.58 | 0.34 | 0.06 | 0.07 | 0.07 |
| acquisition-01 | 512² × 192² | 1.69 | 1.32 | 0.27 | 0.26 | 0.27 |
| acquisition-02 | 512² × 192² | 1.60 | 1.26 | 0.25 | 0.25 | 0.25 |
| acquisition-04 | 512² × 192² | 1.72 | 1.32 | 0.29 | 0.29 | 0.29 |
| acquisition-05 | 512² × 192² | 1.97 | 1.49 | 0.29 | 0.29 | 0.29 |
| acquisition-11 | 512² × 192² | 1.69 | 1.34 | 0.25 | 0.26 | 0.25 |
| acquisition-08 | 1024² × 192² | 6.97 | 5.84 | 2.50 | 1.15 | 1.23 |
| acquisition-09 | 1024² × 192² | 7.12 | 5.73 | 2.56 | 1.27 | 1.26 |
| acquisition-10 | 2048² × 192² | 27.92 | 23.41 | 23.77 | 11.46 | 4.73 |

## Encoding, saving and verification

The encode column is the native **fused decode-and-encode** stage. It cannot isolate ANS encoding alone. Verification is MPS wall time, including QEM reopen. Native save excludes its separate product/parity checks.

| Acquisition | Encode (s) | Index (s) | QEM save (s) | Full verification (s) | Conversion total (s) | QEM GiB | HDF5 / QEM | Logical uint16 / QEM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acquisition-03 | 0.14 | — | 0.32 | 0.40 | — | 0.456 | 1.47× | 9.88× |
| acquisition-01 | 0.56 | — | 1.32 | 1.05 | — | 1.913 | 1.48× | 9.41× |
| acquisition-02 | 0.53 | — | 1.30 | 1.02 | — | 1.920 | 1.48× | 9.38× |
| acquisition-04 | 0.56 | — | 1.50 | 1.03 | — | 2.193 | 1.43× | 8.21× |
| acquisition-05 | 0.63 | — | 1.44 | 1.02 | — | 2.200 | 1.43× | 8.18× |
| acquisition-11 | 0.56 | — | 1.32 | 1.07 | — | 1.915 | 1.48× | 9.40× |
| acquisition-08 | 2.50 | — | 6.59 | 3.72 | — | 8.767 | 1.43× | 8.21× |
| acquisition-09 | 2.50 | — | 6.08 | 4.06 | — | 8.863 | 1.43× | 8.12× |
| acquisition-10 | 10.17 | — | 26.60 | 19.77 | — | 35.702 | 1.42× | 8.07× |

Ratios above one mean smaller QEM storage. The HDF5 ratio includes the acquisition master and detector shards. Logical storage is the complete original uint16 tensor. Conversion totals include source hashing and metadata work; they are not sums of the overlapping diagnostic columns.

A Python-style conversion total is omitted: native loading, saving and reopening surround additional product checks, and exhaustive independent verification uses MPS. The native writer uses a single temporary body and preserves all block checksums.

Unsupported native inputs: acquisition-06, acquisition-07. Both use a 96 × 96 detector; the native HDF5 decoder rejects incomplete 4096-value bitshuffle blocks. No crop or binning workaround was used.

## QEM loading for the 96-square detector inputs

Both inputs rejected by native HDF5 ingestion successfully reopen as QEM on Swift/Metal. The disposable files were produced on MPS and verified against every original count before native loading. This qualifies the encoded-file reader separately from native HDF5 conversion.

| Acquisition | Producer | Native QEM reload 1 (s) | Native repeated median (s) | Count verification |
|---|---|---:|---:|---|
| acquisition-06 | MPS | 0.21 | 0.21 | 2,415,919,104, zero mismatches |
| acquisition-07 | MPS | 0.21 | 0.20 | 2,415,919,104, zero mismatches |

The native command is `metal-runtime-ans-benchmark INPUT.qem --qem-load-only`. It authenticates the complete encoded body and loads GPU residency; the exhaustive source-count comparison above uses MPS.

## Qualification limits

9 successful acquisitions, 282,662,535,168 values exhaustively compared, zero mismatches. Each QEM is removed after testing. HDF5 loading repetitions validate shape/dtype; full count verification is performed by the separate conversion workflow, not repeated during each load timing.

These measurements use a source checkout, not an installed release. Broader typed/FP8 failures and unavailable real-data gates remain visible in the [shared acceptance record](2026-10-23-qem-verification.md). Background system/storage activity and existing memory pressure are retained in private telemetry; these are observed workflow timings, not isolated hardware rankings.
