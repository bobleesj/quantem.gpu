# CPU reference

The CPU path is an explicit independent reference for small deterministic
fixtures and portable IO checks. It is not a silent production fallback.

```python
from quantem.gpu import io

reference = io.load("small_master.h5", backend="cpu", dtype="u16")
```

## Dispatch and implementation layers

| Layer | CPU/reference source | Responsibility |
|---|---|---|
| Explicit selection | `src/quantem/gpu/io/backends/protocol.py` | accept `backend="cpu"`; automatic accelerator selection never chooses it |
| IO implementation | `src/quantem/gpu/io/backends/cpu/reference.py` | h5py/hdf5plugin decode, bad-pixel zeroing, integer detector-bin reference |
| Array detector reference | `src/quantem/gpu/detector/workflow.py::_ArrayComputeBackend` | NumPy mean diffraction, masks, exact sums, and CoM |
| DPC/iDPC reference | `src/quantem/gpu/dpc/workflow.py` | NumPy rotation, curl objective, and FFT integration |
| Frozen contracts | `tests/parity`, product and backend tests | adjudicate accelerator outputs without generating goldens from that accelerator |

The reference call path is:

```text
io.load(..., backend="cpu")
  → explicit protocol resolution
  → h5py + hdf5plugin decompression
  → NumPy array + shared LoadResult metadata
  → NumPy detector/DPC reference operations
```

## Reference design

Reference code favors directness and independent arithmetic over sharing an
accelerator optimization. It preserves
$I[s_r,s_c,q_r,q_c]$ and `(row, column) ≡ (r, c)`, uses widened accumulators,
retains incomplete edge bins, and records the same provenance.

A reference fixture is small, deterministic, versioned, and generated through
an explicit recapture command. The backend being adjudicated never creates its
own golden. Missing accelerated capability fails honestly rather than running
the reference and later being reported as GPU evidence.

## Arithmetic and independence

Integer detector reductions widen before summation. Bad pixels are zeroed in
the same scientific order as accelerated paths. Floating references state their
dtype, normalization, and tolerance. The reference favors direct array
expressions over sharing a fused accelerator kernel, because common code can
hide a common bug.

### Known incomplete-edge gap

The public scientific contract retains an incomplete final detector bin and
records its smaller contribution count. The current CPU loader's private
`_bin_sum` helper instead trims that incomplete edge. Until this implementation
gap is fixed across its focused fixtures, use detector shapes divisible by the
requested bin factor for CPU parity signoff. Do not redefine the public
contract around this limitation, and do not use this path to generate an
incomplete-edge golden.

## Focused checks

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_import_without_cupy.py \
  tests/io/test_load.py \
  tests/test_products_parity.py \
  tests/test_dpc_rotation_agreement.py
```

For each frozen fixture, record the generator revision, input checksum,
shape/dtype, exact parameters, expected-output checksum, comparison metric, and
tolerance. Updating a golden requires an explicit recapture reason and an
independent scientific review.
