# Strict ANS acquisition acceptance

This gate checks original acquisition → compressed device residency → detector
queries → `.qem` export → reopen. It produces JSON, a small Markdown table, and
a pytest log outside the repository. It does not launch an application or
certify frame rate, SSB, every vendor dialect, or an entire detector generation.

## Run on physical devices

Use the candidate checkout and an environment containing its normal runtime
dependencies plus pytest. The runner checks the imported package location so
an older installed copy cannot silently stand in for the candidate.

```bash
python scripts/check_ans_io.py --backend mps \
  --zenodo-root /path/to/local/zenodo-7464234 \
  --tcmep-root /path/to/local/zenodo-15084123 \
  --output /tmp/ans-mps.json

CUDA_VISIBLE_DEVICES=0 python scripts/check_ans_io.py --backend cuda \
  --zenodo-root /path/to/local/zenodo-7464234 \
  --tcmep-root /path/to/local/zenodo-15084123 \
  --output /tmp/ans-cuda.json

python scripts/check_ans_io.py --backend metal --output /tmp/ans-metal.json
```

`mps` means the Python Apple-GPU path, including its Metal kernels and native
tensor outputs. `metal` means the independently built native Swift/Metal
reader/exporter. These are separate acceptance paths, not interchangeable
evidence. CUDA must run on a physical NVIDIA device; CPU fallback is not a pass.
Choose an available CUDA device without interrupting another workload. These
are correctness tests, not isolated performance measurements.

Copy the small reports to one machine and combine the same candidate:

```bash
python scripts/check_ans_io.py \
  --combine /tmp/ans-cuda.json /tmp/ans-mps.json /tmp/ans-metal.json \
  --output /tmp/ans-summary.json
```

The accompanying `ans-summary.md` has CUDA, Python MPS/Metal, and native
Swift/Metal columns. Source fingerprints must match. Changing code during a
run invalidates its acceptance result. Reports are never overwritten; choose
a fresh output name for each attempt.

## What each row establishes

| Workflow | Evidence required |
|---|---|
| Original count HDF5 | Chunked bitshuffle/LZ4 input, default ANS residency, exact selected DP and virtual sum, mean DP, metadata and saved-copy reopening |
| NumPy uint8/uint16 | Native patterns, mean DP, saved-copy reopening; native Swift also checks every small-fixture DP and three detector masks |
| NumPy int32 counts | Complete bounded range audit, exact narrowing only, no clipping, exported provenance |
| Float32 arrays | Rectangular detector geometries, fractions and IEEE bit patterns preserved, selected queries and export/reopen |
| EMPAD-G1 and processed EMPAD2 | Separate layout checks, original XML retention and no fractional-to-integer conversion |
| DM3 and DM4 | Separate synthetic reader/export workflows; neither proves the other nor every K3 acquisition variant |
| EMD arrays | Explicit array selection, axis metadata, known-unit normalization and unknown-unit preservation; not Velox events |
| Float `.qem` | Bitwise exact measurements including signed zero and NaN payloads; Python also checks metadata, reductions and resource lifetime |
| ANS-only ingestion | No reference-encoder fallback, multi-window ingestion, observed device uploads and encoder input bounded by the declared scratch limit, dense/packed requests rejected before upload |
| Published real collection | All 22 acquisitions in Zenodo 7464234, all detector values compared after export/reopen, retained companion metadata |
| Published TCMEP collection | All 21 acquisitions in Zenodo 15084123, full-value export/reopen comparison, prepared-stack geometry and calibration companions |
| Exact promoted float64 | Complete bitwise round-trip audit before allocation; nonrepresentable values rejected, narrowing provenance retained |

Synthetic checks remain small. Real-data checks compare one scan row at a time;
they never construct a full CPU reconstruction oracle or an uncompressed GPU
acquisition. Encoder-input instrumentation and resident-profile assertions are
not a universal allocator trace of every driver allocation.

## Strict outcomes and remaining boundaries

- `passed`: every collected check for the selected row completed successfully.
- `failed`: a check or collection failed.
- `blocked`: a required fixture/device was missing or a test skipped/xfail-ed.
- `not-run`: no execution result exists.
- `not-covered`: this gate has no test for that runtime/workflow combination.

Skipped, deselected, empty, or expected-failure checks cannot make a selected
gate pass. A zero pytest exit code alone is insufficient. A native subset pass
is **not** complete native-format support. `not-covered` cells remain visible
even when every selected test passes. The table describes execution coverage,
not a Cartesian claim that every tested shape works with every tested format.

The native gate currently covers uint8/uint16 NumPy export/reopening and a
128 × 128 float32 `.qem` bit oracle. The native NumPy reader does not yet accept
the original int32 simulations. Raw EMPAD2/G3 sensor words need matching
calibration and separately verified decoding. Application folder discovery,
interaction, microphone/voice controls, 120 FPS and SSB require their own tests.
The native float reader does not yet cover the TCMEP gzip originals or its
256 × 256 float detectors. A Python MPS pass does not close those native gaps.

Historical QEM references using retired codecs remain frozen. The native
float test exports a current `.qem` from the original frozen NumPy bit oracle;
it does not rewrite the historical fixture or enable backward compatibility.

## Real-data setup and cleanup

Retain the published archive's original structure and a
`local-download-receipt.json` containing `archive_md5`,
`archive_checksum_verified`, and `files` entries with relative `path`, `bytes`
and `sha256`. The expected archive checksum is pinned in the test. Each tested
acquisition's size and SHA-256 must match its receipt before GPU loading.
Do not put source acquisitions or machine-specific paths in Git.

The separate 3D potential is a reference, not a 4D detector acquisition. Keep
XML and `para.txt` companions. The XML is automatically read; these tests
explicitly attach `para.txt` text to export metadata without guessing calibration
units or overriding the XML. This is not an automatic calibration-text parser.

Every export lives in disposable test storage and is removed after its test,
including failures. Originals and intentional saved files are untouched. The
real float tests require space for one export and temporary writer storage,
plus a 5 GiB reserve. Keep only small JSON/Markdown/log evidence. Review the
collection's original license and agreement before redistribution.
