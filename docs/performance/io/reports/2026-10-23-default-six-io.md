# Six default acquisition IO comparison, 2026-10-23

Historical six-acquisition evidence. The [current five-acquisition selection](2026-10-23-default-five-io.md) excludes 96 × 96 detector inputs from future runs and active examples; the completed measurements below are retained unchanged.

The selected default set is acquisition-03, acquisition-06, acquisition-01, acquisition-04, acquisition-08 and acquisition-10. This is a new run; earlier eleven-acquisition observations remain historical.

The [compression comparison](2026-10-23-qem-compression.md) documents complete
HDF5 and ANS QEM sizes, compression ratios, the expanding input, and why size
savings differ from loading speedups.

CUDA: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, CUDA runtime 12.9, CuPy 14.2.0, Python 3.14.6 and PyTorch 2.13.0. MPS and Swift/Metal: Apple M5 Max, 128 GB unified memory, macOS 26.4; Python 3.13.13/PyTorch 2.13.0 or Swift 6.3.3 release build. The CUDA run held the physical GPU's cooperative lock; Apple backends ran serially.

Private experiment: `20261023-six-default-io`. Base revision: `d2db0bdb4c9859230d3e9ba2f6b23fcfae8eb0f1`. Identical measured source fingerprint: `2873ac0ffac23c49c89a7d5311ffbcc8836ead6af640acf6df10f5a292290df8`. Native binary hashes, source/selection manifests, telemetry and all individual measurements are retained privately.

| Acquisition | Scan × detector | Backend | HDF5 first / repeated (s) | QEM conversion + full verification (s) | QEM reload first / repeated (s) |
|---|---|---|---:|---:|---:|
| acquisition-03 | 256² × 192² | CUDA | 4.29 / 0.21 | 3.35 | 0.08 / 0.08 |
| acquisition-03 | 256² × 192² | MPS | 2.49 / 0.24 | 1.04 | 0.03 / 0.03 |
| acquisition-03 | 256² × 192² | Swift/Metal | 0.58 / 0.38 | 1.76* | 0.07 / 0.07 |
| acquisition-06 | 512² × 96² | CUDA | 3.26 / 0.24 | 2.34† | 0.22 / 0.22 |
| acquisition-06 | 512² × 96² | MPS | 2.68 / 0.36 | 1.83† | 0.12 / 0.11 |
| acquisition-06 | 512² × 96² | Swift/Metal | Unsupported | Unsupported | 0.23 / 0.22‡ |
| acquisition-01 | 512² × 192² | CUDA | 3.72 / 0.44 | 4.03 | 0.28 / 0.26 |
| acquisition-01 | 512² × 192² | MPS | 3.12 / 0.77 | 2.89 | 0.12 / 0.12 |
| acquisition-01 | 512² × 192² | Swift/Metal | 1.63 / 1.41 | 4.80* | 0.27 / 0.27 |
| acquisition-04 | 512² × 192² | CUDA | 3.85 / 0.55 | 4.03 | 0.29 / 0.29 |
| acquisition-04 | 512² × 192² | MPS | 3.17 / 0.78 | 3.08 | 0.15 / 0.14 |
| acquisition-04 | 512² × 192² | Swift/Metal | 1.91 / 1.53 | 4.64* | 0.29 / 0.29 |
| acquisition-08 | 1024² × 192² | CUDA | 7.40 / 1.48 | 13.66 | 1.19 / 1.12 |
| acquisition-08 | 1024² × 192² | MPS | 7.51 / 3.68 | 10.73 | 0.52 / 0.52 |
| acquisition-08 | 1024² × 192² | Swift/Metal | 6.82 / 6.32 | 18.24* | 1.08 / 1.09 |
| acquisition-10 | 2048² × 192² | CUDA | 22.94 / 11.96 | 100.24 | 10.42 / 10.47 |
| acquisition-10 | 2048² × 192² | MPS | 15.44 / 12.11 | 52.05 | 3.13 / 2.20 |
| acquisition-10 | 2048² × 192² | Swift/Metal | 30.46 / 24.28 | 100.98* | 10.81 / 4.54 |

All times are seconds. HDF5 and QEM repeated values are the median of observations two and three; conversion has one observation. First and repeated observations are retained separately. Disk caches were not evicted, so none of these are cold-disk claims. Three loads do not establish tail latency.

HDF5 loading ends at full encoded GPU residency with spatial indexes, at original uint16 precision and geometry, without pixel correction. Python repetitions share a process; native repetitions use fresh processes and reuse their catalog. Each first HDF5 observation starts with an empty frame-index/catalog directory. QEM reloads authenticate and load encoded GPU residency after conversion and verification have already opened the file; they do not expand the full logical tensor.

Conversion starts immediately before the producer workflow and stops after full source comparison and file-header inspection, before the dedicated reload trials. Python includes canonical metadata/source hashing and durable saving. * Native Swift/Metal runs its existing native round-trip/product checks and then exhaustive verification through MPS; this is a hybrid workflow, not a native Swift full-count verifier or an identical workload to Python.

† Acquisition-06 creates a QEM larger than its HDF5 input. The production converter declines that copy. Its measured benchmark workflow includes that initial attempt and a disposable save-and-verify retry. ‡ Native HDF5 decoding does not support this 96² detector layout; its QEM load measurement uses a separately MPS-produced copy whose every source count was verified first. Unsupported cells are not passes.

Every completed verification includes all source counts and flagged detector pixels, with zero mismatches, no crop, binning, approximation, or relaxed tolerance. Full-count verification is performed once per conversion workflow, not inside every HDF5 timing repetition. Temporary QEM files are removed after their trials. The acquired datasets remain unchanged.

These are source-checkout measurements on different computers and storage, not a GPU-only ranking or release qualification. Previously recorded broader FP8/uint64 acceptance failures and unavailable public collections remain unresolved; this rerun changes benchmark selection and timing instrumentation only.

Reproduce one backend with `scripts/benchmark_io_scale.py SOURCE_FOLDER --backend cuda --selection-manifest PRIVATE_SELECTION.json --fresh-index --output-dir NEW_PRIVATE_DIRECTORY --reps 3`. Use `mps` or `native-metal`; the native path also requires `--native-binary RELEASE_QEM_ROUNDTRIP`. The selection manifest holds stable IDs, master filenames and expected master SHA-256 values. The supplementary native QEM-only run and complete commands are archived with the experiment.
