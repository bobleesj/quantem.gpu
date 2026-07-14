# Backends

`quantem.gpu` supports three backend names:

| Backend | Purpose | Notes |
|---|---|---|
| `cuda` | NVIDIA GPU IO, decompression, reductions, and SSB reference paths | Uses CuPy/CUDA kernels where available. |
| `mps` | Apple Silicon Metal/MLX paths | Used for MacBook chunk-backed loading, products, and MPS SSB preview/free-fit paths. |
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
