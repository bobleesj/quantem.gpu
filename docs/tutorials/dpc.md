# Compute CoM, DPC, and iDPC

`quantem.gpu.dpc()` computes center of mass, rotation alignment, and integrated
DPC phase from loaded 4D-STEM data.

```python
from quantem.gpu import dpc, load

result = load("scan_master.h5", backend="auto", det_bin=4)
dpc_result = dpc(result.data)

print(dpc_result.rotation_deg)
print(dpc_result.use_transpose)
print(dpc_result.elapsed)
```

On CUDA, the expensive CoM pass uses the custom `cuda_center_of_mass` kernel for
resident raw-count arrays. Auto-rotation is evaluated after CoM on the small
scan-shaped vector field: the search uses analytic curl/divergence moments, so
it does not allocate one rotated full-field map per candidate angle.

Display the outputs with `quantem.widget`:

```python
from quantem.widget import Show2D

Show2D(dpc_result.phase)
Show2D(dpc_result.com_row)
Show2D(dpc_result.com_col)
```

For parity reports, compare:

- `phase`
- `com_row`
- `com_col`
- `rotation_deg`
- `use_transpose`

Use real data for signoff. Small synthetic arrays are useful for unit tests, but
they are not enough to prove production DPC behavior.
