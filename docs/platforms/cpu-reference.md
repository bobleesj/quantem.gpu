# CPU reference

CPU is an explicit reference and portability path for small tests. It provides
an independent implementation for adjudicating accelerator outputs and file
round trips.

```python
from quantem.gpu import io

reference = io.load("small_master.h5", backend="cpu", dtype="u16")
```

Production accelerated APIs never silently fall back to CPU. A missing CUDA,
MPS, native Metal, or WebGPU capability must fail honestly rather than produce
a slower result that is later reported as GPU evidence.

Reference fixtures should be small, deterministic, and generated independently
of the backend under test. Frozen outputs include source hashes, parameters,
shape/dtype, and the metric used for comparison.
