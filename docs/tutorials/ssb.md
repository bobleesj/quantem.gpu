# Ptychographic SSB

Ptychographic SSB reconstruction is compute-only in quantem.gpu. Widget UI,
HTML export, and interactions stay in quantem.widget.

The same API selects CUDA or MPS without changing scientific parameters:

~~~python
from quantem.gpu import SSB

workflow = SSB.open(
    "scan_master.h5",
    backend="mps",
    voltage_kV=300,
    semiangle_mrad=21.4,
    scan_sampling_A=0.5,
)
result = workflow.fit(trials=200, refinement="nelder-mead")
~~~

Changing `backend` to `"cuda"` preserves the full-BF exact objective,
parameter units, and the single `SSBResult` type. Browser WebGPU uses the same
serialized plan through the canonical `quantem showptycho` CLI; unsupported
fit capability fails explicitly.

For known aberrations, skip fitting and reconstruct directly:

~~~python
fixed = workflow.reconstruct(
    {"C10": 12.5, "C12": 3.0, "phi12": 0.25},
)
~~~

The interactive widget path uses `workflow.preview(aberrations)` for transient
phase/loss updates without inventing a backend-specific API or result class.

Display the shared result with the widget layer:

~~~python
from quantem.widget import Show2D

Show2D(result.phase)
Show2D(result.amplitude)
~~~

Parity reports must use the same detector selection and include object-phase
difference maps, full-BF loss, C10, C12, phi12, load time, fit time, and BF
pixel count. Approximate preview modes are not calibration signoff.
