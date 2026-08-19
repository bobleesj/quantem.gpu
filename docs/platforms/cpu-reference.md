# CPU reference

The CPU path is an explicit independent reference for small deterministic
fixtures and portable IO checks. It is not a silent production fallback.

```python
from quantem.gpu import io

reference = io.load("small_master.h5", backend="cpu", dtype="u16")
```

## Source map

- IO reference: `src/quantem/gpu/io/backends/cpu`
- backend-neutral detector/DPC reference: the public domain implementations
- frozen fixtures and comparisons: `tests/parity` and focused test modules

## Reference design

Reference code favors directness and independent arithmetic over sharing an
accelerator optimization. It preserves
$I[s_r,s_c,q_r,q_c]$ and `(row, column) ≡ (r, c)`, uses widened accumulators,
retains incomplete edge bins, and records the same provenance.

A reference fixture is small, deterministic, versioned, and generated through
an explicit recapture command. The backend being adjudicated never creates its
own golden. Missing accelerated capability fails honestly rather than running
the reference and later being reported as GPU evidence.
