# MPS uint32 Arina masters and the CUDA 13 `[cuda]` extra - 2026-10-07

Two findings from the cross-host Show4DSTEM test (quantem.widget rc39 on
quantem.gpu `81f401d`), fixed in `a5c5179` and `58bed8c`.

## 1. MPS refused the public gold_512 master

### Question

`io.load("gold_004_master.h5")` on an M5 Max raised `NotImplementedError: uint32
counts span [0, 4294967295] and cannot be stored exactly as uint16`. CUDA loads
the same file. Does MPS now produce the same counts as CUDA?

### Cause

`io/arrays.py` sent uint32 bitshuffle/LZ4 masters to the native reader only on
CUDA. On MPS they reached the bounded generic path, which audits the raw counts
on the host before any flagged-pixel correction; Arina stores 0xFFFFFFFF at the
four flagged pixels, so the audit failed.

### Change

MPS now takes the native reader for uint32 masters, like CUDA: Metal decodes the
uint32 batch, the stored-mask correction runs on the uint32 counts (the median
reads only valid neighbors, so a sentinel never enters a replacement), then the
`narrow_u32_u16` kernel copies every count into uint16 and flags any count above
65535 instead of clipping it. Batches hold 4096 scans for uint32 (8192 for
uint16), so the decoded bytes per batch are unchanged.

### Setup

- Data: public gold_512 (`bobleesj/quantem-data` revision `00179851`), 512 x 512
  scan, 192 x 192 detector, uint32, 27 data files, 4 flagged pixels. The Mac reads
  it from the Linux workstation over NFS.
- CUDA: Linux, RTX PRO 6000 Blackwell GPU 0 (shared with other jobs at 100 %
  utilization), CuPy 14.2.0 (CUDA 12.9), torch 2.13.0.
- MPS: Apple M5 Max (128 GB, macOS 26.4), torch 2.14.1, pyobjc-framework-Metal
  12.2.2.
- One script on both hosts: `io.load(master, backend=...)` with the default
  median correction, one `detector.prepare` session, masks from the CUDA BF fit
  (r = 49.3 px) reused on MPS, and a SHA-256 of every decoded count read in
  16-row scan blocks.

### Results

| Quantity | dtype | CUDA = MPS |
|---|---|---|
| Every decoded count (512 x 512 x 192 x 192) | uint16 | bit-equal (SHA-256 `8916ef9a...`) |
| Detector total | uint64 | bit-equal |
| Mean DP | float32 | bit-equal |
| BF, ABF, ADF, HAADF exact sums | uint64 | bit-equal |
| BF, ABF, ADF, HAADF virtual images | float32 | bit-equal |
| Frames at scan (128, 128), (256, 256), (384, 384) | uint16 | bit-equal |
| Flagged pixels | | 4 replaced by median on both |

| | CUDA (RTX PRO 6000) | MPS (M5 Max) |
|---|---|---|
| `io.load` wall time | 3.7 s | 8.3 s first run, 5.1 s second run (NFS source) |
| Peak device memory during load | 4.53 GB CuPy pool | 7.55 GB Metal allocated |
| Resident after load | 6.05 GB | 6.05 GB |
| Process peak RSS | 2.96 GB | 2.5-2.6 GB |
| Mean DP | 0.26 s | 0.14 s |
| One virtual image | 2-7 ms | 3-7 ms |
| One frame | 2-3 ms | 0.3 ms |

`Show4DSTEM(master)` on the M5 Max (widget rc39, quantem.gpu wheel of this branch)
now constructs in 6.6 s on a `DetectorSession` with the BF radius 49.337 that the
CUDA row reported.

`tests/hardware/test_uint32_flagged_master.py` writes an Arina-style master
(two data files, 1100 scans, 48 x 48 uint32 frames ending in a partial bitshuffle
block, a corner, an interior and two adjacent flagged pixels) and compares
CUDA or MPS (`QEM_TEST_BACKEND`) with h5py and a NumPy median. It passes on both;
on MPS at `81f401d` it raises the NotImplementedError above.

### Not changed

The bounded generic HDF5 path still audits uint32 counts before correction, so a
uint32 layout outside the native reader (for example an explicit `dataset_path`)
with flagged sentinels is still refused on MPS. CUDA reads such layouts through
its native reader.

## 2. uv installed a CuPy that could not compile kernels

### Question

`cupy-cuda12x[ctk]>=12.0` resolved differently under uv and pip. Does CuPy for
CUDA 13 give one working set with both?

### Results

Fresh Python 3.12 venvs, `pip install -e ".[cuda]"` and `uv pip install -e ".[cuda]"`,
driver 580.173.02 (CUDA 13.0):

| `[cuda]` requirement | Installer | torch | CuPy | CUDA toolkit wheels | Works |
|---|---|---|---|---|---|
| `cupy-cuda12x[ctk]>=12.0` | uv | 2.14.1 (CUDA 13) | cupy-cuda12x 13.6.0, no `[ctk]` | none for CuPy | no: `libnvrtc.so.12` missing |
| `cupy-cuda12x[ctk]>=12.0` | pip | 2.10.0 (CUDA 12.8) | cupy-cuda12x 14.2.0 | 12.8 | yes |
| `cupy-cuda13x[ctk]>=14.0` | uv | 2.14.1 (CUDA 13) | cupy-cuda13x 14.2.0 | 13.0.3 shared | yes |
| `cupy-cuda13x[ctk]>=14.0` | pip | 2.10.0 (CUDA 12.8) | cupy-cuda13x 14.2.0 | 13.4.2 for CuPy, 12.8 for torch | mixed majors |
| `cupy-cuda13x[ctk]>=14.0` + `torch>=2.11` | uv and pip | 2.14.1 (CUDA 13) | cupy-cuda13x 14.2.0 | 13.0.3 shared (NVRTC 13.0.88) | yes |

`[ctk]` first appears in CuPy 14.0. PyPI torch 2.11 to 2.14 pin `cuda-toolkit`
13.0.x; without the torch floor pip keeps the newest `cuda-toolkit` for CuPy and
backtracks torch to 2.10, the last CUDA 12 build. CUDA 13 wheels put every
library in `site-packages/nvidia/cu13/lib`, so `preload_libraries` now also
looks there (13 libraries found; 0 before).

With the same test-only packages as the CUDA 12 gate environment, both CUDA 13
venvs pass `pytest -m ""` on GPU 0 (1037 passed, 246 skipped, the same skips as
the CUDA 12 environment) and run `qem_portable.ipynb` and `ssb_gold_native.ipynb`
without errors.

### Found on the way

optuna 5.0.0, which a fresh install now resolves, changes the SSB aberration
search: `tests/hardware/cuda/test_ssb_units.py::test_fitted_c10_is_nm[100.0]`
fits C10 = -9.715 nm against -10 nm (2 % tolerance) with optuna 5.0.0 on both
CUDA 12 and CUDA 13 CuPy, and passes with optuna 4.9.0 on both. Not changed here.
