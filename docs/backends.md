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

WebGPU is a browser runtime, not a Python device backend. Reusable browser
compute sources live in `quantem.gpu.webgpu` and are bundled by
`quantem.widget` for anywidget and exported-HTML use.

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
| Resident virtual-image drag kernels | CuPy uint8/uint16 arrays use CUDA RawKernels for selected-pixel BF/DF/ADF, dense cached `total - complement`, and fused CoM/DPC moment reduction | Chunk-backed Metal uint8/uint16 selected-pixel reducers; dense DF uses cached `total - complement`; CoM/DPC uses raw Metal `com_u8`/`com_u16`; no giant Torch-MPS tensor for full no-bin browse loads | Reference/small data path | Add broader MPS/WebGPU side-by-side reports on more held-out datasets. |
| CoM, DPC, and iDPC | Implemented with a fused CUDA CoM kernel and backend cache | Implemented for chunk-backed Mac data | Reference/small data path | More real-data visual parity and performance signoff. |
| Ptychographic SSB fixed preview | Implemented/reference | Implemented with MLX/Metal path | Not a production target | More datasets and time-series checks. |
| Ptychographic SSB optimizer/free-fit | Implemented/reference | Implemented for current real-data parity shape | Not a production target | Broader scan sizes, more datasets, temporal/joint SSB validation. |
| GIF/MP4 movie rendering | CUDA/NVENC MP4 implemented | Metal NV12 rendering plus ffmpeg/VideoToolbox MP4 implemented | GIF and MP4 fallback via CPU/PIL/ffmpeg | Larger export benchmark matrix and widget button wiring. |
| WebGPU browser source ownership | Canonical source shipped in `quantem.gpu.webgpu`; widget consumes/bundles it | Same source ownership rule | Not applicable | Keep removing widget-local permanent backend copies as synced GPU-owned sources gain browser parity tests. |
| `quantem.widget` display | Widget consumes arrays/results | Widget consumes arrays/results | Widget consumes arrays/results | Keep UI/export controls in widget, not in `quantem.gpu`. |
| `quantem.live` app/dashboard callers | Partially routed | Partially routed | Fallback only | Finish CLI/dashboard cleanup over time. |

The rule for new heavy work is: implement the compute or IO path in
`quantem.gpu`, then let widget/live call it.
