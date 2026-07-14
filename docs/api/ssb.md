# SSB API

Primary imports:

```python
from quantem.gpu.ssb import (
    SSB,
    SSBEngine,
    defocus_sweep,
    ssb,
    ssb_fit_mps,
    ssb_preview_mps,
    ssb_series,
)
```

Use `ssb()` or `SSBEngine` for CUDA/reference workflows. Use `ssb_preview_mps`
and `ssb_fit_mps` for the Apple GPU path.

The SSB API is intentionally pure compute. It should return arrays, metrics, and
metadata that widget/live callers can display or save.

Signoff expectations:

- Use real data, not only synthetic controls.
- Compare CUDA and MPS on the same BF-pixel selection.
- Include images and difference maps, not only scalar tables.
- Do not use fast mode for parity claims.
- Keep temporal/joint SSB experiments separate until the improvement metric is
  clear.
