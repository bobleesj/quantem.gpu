# Python API reference

Use this section to check parameters, units, return values, and supported
options. For a first session, follow [I/O, DPC and SSB on one page](../python-workflow.md).
The examples use one public Python API on supported MPS and CUDA paths.

| Task | Entry point | Returns | Details |
|---|---|---|---|
| Load and select measurements | `io.load`, array indexing | `Dataset4dstemGPU`; selections are GPU Torch tensors | [I/O](io.md) |
| Save or inspect a file | `io.save`, `io.inspect` | Saved acquisition or header report | [Save and share](qem-python.md) |
| Mean diffraction, BF, ADF, DF | `detector.mean`, `bf`, `adf`, `df` | Reduced NumPy images | [Detectors](images_dpc.md) |
| CoM and integrated DPC | `dpc.run` | `DPCResult` | [DPC](images_dpc.md) |
| Find aberrations; reconstruct phase | `SSB.find_aberrations`, `SSB.reconstruct` | `SSBResult` | [SSB](ssb.md) |
| Render a movie | `movie` | Encoded artifact | [Movies](movie.md) |

Use `(row, col)` coordinates. `data.sampling`, `data.units`, and `data.origin`
describe the axes; `data.metadata` holds the full record. Unknown calibration
stays explicit. See the individual API page for dtype, ownership, and backend
limitations before interpreting a result.

## Supporting modules

| Module | Purpose | Details |
|---|---|---|
| `io` | Discovery, loading, indexing, inspection, saving | [I/O](io.md) |
| `detector`, `dpc` | Detector geometry, images, and phase gradients | [Detectors and DPC](images_dpc.md) |
| `device` | Detect or explicitly select an accelerator | [Device APIs](core.md) |
| `optics` | Relativistic electron wavelength | [Optics APIs](core.md) |
| `parallax` | Parallax reconstruction | [Parallax](core.md) |
| `geometry` | Scan-plane rotation of an acquisition | [Scan rotation](core.md) |
| `screening` | Reuse derived products for application integration | [Screening](core.md) |
| `display`, `movie` | Display math and export | [Movies](movie.md), [display operations](../kernels/display-export.md) |

Native Swift products, browser kernels, and byte-level formats are documented
under the [Developer guide](../developer/index.md). Ordinary Python workflows
use the public modules above rather than importing backend internals.
