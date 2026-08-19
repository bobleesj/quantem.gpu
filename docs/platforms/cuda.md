# CUDA

CUDA is the primary runtime for NVIDIA workstations, servers, and dedicated
large-memory GPUs. Python callers use the same public IO, detector, DPC, and SSB
APIs as other backends.

```python
from quantem.gpu import detector, io

loaded = io.load(
    "scan_master.h5",
    backend="cuda",
    dtype="u16",
    det_bin=1,
)
bright_field = detector.bf(loaded.data)
```

The returned detector data is device-resident. CUDA implementations own
bitshuffle/LZ4 decode, detector reductions, CoM/DPC, and SSB kernels; public
workflows do not expose kernel-launch or storage-worker tuning.

For remote native clients, keep the data and CUDA environment on the
workstation and connect through the loopback service over SSH. See
[Remote CUDA](../tutorials/remote_cuda.md).

Performance claims must name the GPU, driver/runtime, source shape/dtype,
crop/bin settings, allocated/reserved VRAM, total-card occupancy, cache state,
and parity artifact. See [Benchmark methodology](../performance/methodology.md).
