# Run SSB compute

SSB migration is compute-only. The engine should not contain any widget UI.
Display, export, and interactions stay in `quantem.widget`.

The high-level API is available from `quantem.gpu.ssb`:

```python
from quantem.gpu.ssb import ssb

result = ssb(
    data,
    voltage_kV=300,
    semiangle_mrad=21.4,
    scan_sampling_A=0.5,
)

object_wave = result.object_wave
phase = result.phase
```

For Apple Silicon MPS data, use the MPS entry points when testing the Mac path:

```python
from quantem.gpu import load
from quantem.gpu.ssb import ssb_fit_mps, ssb_preview_mps

loaded = load("scan_master.h5", backend="mps", det_bin=1)
preview = ssb_preview_mps(loaded.data)
fit = ssb_fit_mps(loaded.data)
```

Parity expectations:

- CUDA remains the reference path for production SSB parity.
- MPS fixed-preview and optimizer paths must be compared against CUDA on the
  same real data and the same BF-pixel selection.
- Do not use fast-mode shortcuts for signoff.
- Reports should include phase images, difference maps, loss, C10, C12, phi12,
  load time, fit time, and BF pixel count.

Temporal or joined SSB work should stay on a separate verification track until
it has real time-series metrics showing whether it improves over per-frame SSB
or a simple complex/phase average.
