# CPU reference

The package loads and saves scientific data on CUDA or Apple MPS only; the
public `io.load` and `io.save` have no CPU route, and `device="cpu"` raises.
The CPU decoders that adjudicate the GPU loads live with the tests, in
`tests/parity/cpu_reference.py`: h5py plus hdf5plugin for HDF5 masters, and
the portable QEM codec in `src/quantem/gpu/formats/qem/reference.py`, which
imports no accelerated codec code and uses frozen integer entropy tables. They
are a correctness reference for small deterministic fixtures, never a fallback.
See [QEM Python workflows](../api/qem-python.md).

```python
from tests.parity import cpu_reference

reference = cpu_reference.load("small_master.h5")
```

## Dispatch and implementation layers

| Layer | CPU/reference source | Responsibility |
|---|---|---|
| IO implementation | `tests/parity/cpu_reference.py` | h5py/hdf5plugin decode of complete acquisitions and the CPU reference of the bad-pixel fill |
| Array detector reference | `src/quantem/gpu/detector/tensors.py::ArrayBackend` | NumPy mean diffraction, masks, exact sums, and CoM in uint64/float64 |
| DPC/iDPC reference | `src/quantem/gpu/dpc/workflow.py` | NumPy rotation, curl objective, and FFT integration |
| Frozen contracts | `tests/parity`, product and backend tests | adjudicate accelerator outputs without generating goldens from that accelerator |

The reference call path is:

```text
cpu_reference.load(...)
  → h5py + hdf5plugin decompression
  → NumPy array + shared Dataset4dstem metadata
  → NumPy detector/DPC reference operations
```

## Reference design

Reference code favors directness and independent arithmetic over sharing an
accelerator optimization. It preserves
$I[R_r,R_c,k_r,k_c]$ and `(row, column) ≡ (r, c)`, uses widened accumulators,
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

## Focused checks

```bash
PYTHONPATH=src python -m pytest -q \
  tests/contracts/test_import_without_cupy.py \
  tests/contracts/io/test_load.py \
  tests/parity/test_products_parity.py \
  tests/parity/test_dpc_rotation_agreement.py
```

For each frozen fixture, record the generator revision, input checksum,
shape/dtype, exact parameters, expected-output checksum, comparison metric, and
tolerance. Updating a golden requires an explicit recapture reason and an
independent scientific review.
