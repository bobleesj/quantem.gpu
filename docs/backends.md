# Backends

`quantem.gpu` supports three backend names:

| Backend | Purpose | Notes |
|---|---|---|
| `cuda` | NVIDIA GPU IO, decompression, reductions, and SSB reference paths | Uses CuPy/CUDA kernels where available. |
| `mps` | Apple Silicon Metal/MLX paths | Used for MacBook chunk-backed loading, BF/DF/DPC images, and MPS SSB preview/free-fit paths. |
| `cpu` | Portable reference fallback | Useful for metadata, availability, and parity checks; not the target for heavy workflows. |

Check the selected backend:

```python
import quantem.gpu as qgpu

report = qgpu.device_report()
print(report.selected)
print(report.cuda_available, report.cuda_device_count, report.cuda_error)
print(report.mps_available, report.mps_error)
```

Request a backend explicitly when you need honest failure:

```python
qgpu.select_device("cuda")  # raises if CUDA is unavailable
qgpu.select_device("mps")   # raises if MPS is unavailable
```

Use `backend="auto"` for normal scripts and `backend="cuda"` or
`backend="mps"` in parity/performance tests.

## Backend coverage

CUDA and MPS are the primary production backends. CPU exists for reference,
availability, and small fallback workflows.

| Area | CUDA | MPS | CPU | Further work |
|---|---|---|---|---|
| Device report and explicit selection | Implemented | Implemented | Implemented | Keep errors precise as dependencies change. |
| HDF5 master metadata, readiness, discovery | Implemented | Implemented | Implemented | Keep one shared API for widget/live callers. |
| Full HDF5 load and bitshuffle/LZ4 decompression | Implemented with CUDA kernels and device arrays | Implemented with Metal kernels and chunk-backed unified-memory arrays | Reference h5py/hdf5plugin decode | Broader real-data regression matrix. |
| `load(..., scan_region=...)` crop-first IO | Implemented | Implemented | Not the accelerated target | More held-out masters and malformed-file tests. |
| BF, DF, ADF, mean DP, masked sums | Implemented | Implemented for chunk-backed Mac data | Reference/small data path | More real-data parity reports. |
| CoM, DPC, and iDPC | Implemented | Implemented for chunk-backed Mac data | Reference/small data path | More real-data visual parity and performance signoff. |
| Ptychographic SSB fixed preview | Implemented/reference | Implemented with MLX/Metal path | Not a production target | More datasets and time-series checks. |
| Ptychographic SSB optimizer/free-fit | Implemented/reference | Implemented for current real-data parity shape | Not a production target | Broader scan sizes, more datasets, temporal/joint SSB validation. |
| GIF/MP4 movie rendering | CUDA/NVENC MP4 implemented | Metal NV12 rendering plus ffmpeg/VideoToolbox MP4 implemented | GIF and MP4 fallback via CPU/PIL/ffmpeg | Larger export benchmark matrix and widget button wiring. |
| `quantem.widget` display | Widget consumes arrays/results | Widget consumes arrays/results | Widget consumes arrays/results | Keep UI/export controls in widget, not in `quantem.gpu`. |
| `quantem.live` app/dashboard callers | Partially routed | Partially routed | Fallback only | Finish CLI/dashboard cleanup over time. |

The rule for new heavy work is: implement the compute or IO path in
`quantem.gpu`, then let widget/live call it.
