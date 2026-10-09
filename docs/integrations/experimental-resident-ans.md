# Experimental lossless resident ANS

ANS means **asymmetric numeral systems**. Supported original acquisitions now
default to ANS on CUDA/MPS through `io.load(path)`. This page retains experimental
browser/archive integrations; their qualification is distinct from that default.
See the [current IO guide](../api/io.md) for supported formats and the
[acceptance gate](../maintainer/ans-io-acceptance.md) for runtime coverage.
Source integration does not mean these changes have been released on PyPI.

## Compatibility safeguards

The loader detects encoded sources. Explicit reference inputs retain their
separate contract. Encoding writes a new container atomically
and refuses to overwrite an existing file; original acquisitions remain intact.
Source ownership and cancellation are covered by regression tests. These are
specific safeguards, not a guarantee that every experimental hardware path is
qualified. The performance and validation limits below remain part of the feature.

## One implementation owner

`quantem.gpu` owns the count encoder, validated container reader, CUDA decoders,
WebGPU decoders, detector reductions, pattern gathering, and shared GPU display
math. Show4DSTEM owns browser permission prompts, progressive panel display,
selection, gestures, and scheduling. Its build copies the package's TypeScript
sources into the ignored `js/.generated/engine` directory and bundles them;
those generated files are not an independently maintained codec.

| Input | Package implementation | Client route |
| --- | --- | --- |
| [K3 saved copies](k3-dm4-qem.md), `quantem.qem` | CUDA and Metal encoding, checksummed reopening, spatial queries | Python CUDA/MPS and native Live4DSTEM macOS |
| Saved `.qem` copies | `io.save`, `io.load`, `io.qem`, CUDA count decoder, native reader | CUDA/MPS resident queries; native Live4DSTEM macOS |
| Retained detector-rANS manifest | WebGPU `rans.ts` | Show4DSTEM WebGPU export |

These are explicitly identified formats, not interchangeable byte streams.
The canonical writer is shared; compatibility readers retain older experiments
without rewriting source data. Full-series conversion of every retained archive
to the canonical container is not a completed acceptance gate.

## Encode and query canonical counts

```python
from quantem.gpu import detector, io

# resident: an encoded acquisition from io.load, native uint8/uint16 counts,
# (scan_row, scan_col, detector_row, detector_col)
io.save("acquisition.qem", resident)
with io.load("acquisition.qem", backend="cuda", device=0) as loaded:
    session = detector.prepare(loaded)
    pattern = session.frame(0, output="native")
    image = session.masked_sum_exact(binary_detector_mask, output="native")
```

`device=0` means the first CUDA-visible device; select the intended physical GPU
before launching Python. Saving copies the resident's encoded bytes without
re-encoding; it is not a claim of real-time full-acquisition compression. CUDA
residency retains encoded buffers and produces device results without
constructing a full dense acquisition. See [the container contract](../developer/count-ans.md) for dtype, checksums,
invalid pixels, conversion, and lifetime rules.

Compatible ANS acquisitions can be retained independently on MPS and queried
as one detector session without dense stacking:

```python
from quantem.gpu import detector, io

loaded = io.load(paths, backend="mps", stack=False)
session = detector.prepare(loaded)
patterns = session.frame(scan_row * scan_columns + scan_column)
images = session.masked_sums_exact(masks)  # (mask, acquisition, scan_row, scan_column)
```

All files must declare the same complete scan and detector geometry. The series
adapter runs each query on each acquisition in turn and stacks the results along
the acquisition axis, so every value equals the result of preparing that
acquisition alone; it returns only the requested point patterns or detector
products. Each `Dataset4dstemGPU` owner remains caller-owned and must be closed
after the session is finished. This is the supported
multi-file ANS API, not a claim that arbitrary HDF5 folders are encoded as ANS
automatically.

## Show4DSTEM WebGPU

The widget package supplies the browser export, using the same package reader:

```python
from quantem.widget.show4dstem_webgpu_export import export_show4dstem_rans_viewer

html = export_show4dstem_rans_viewer(
    ["acquisition.qem", "next-acquisition.qem"],
    "ans-viewer",
    frame_labels=["First", "Second"],
)
```

Open the generated launcher/viewer and use **Open QEM files** to grant access
to the linked `.qem` files in `ans-viewer/rans`. Export preserves native shape
and dtype, links the `.qem` files, and does not decode or duplicate the full
acquisition. Keep the source files available.

The canonical-file workflow above is the browser export integration.

Raw hardware sentinel counts are preserved. The established detector validity
mask is applied only to displayed scientific products. Integer native patterns
and detector sums require exact reference equality. Count means use explicitly
rounded float32 division; native sums remain exact for the validated series.

## Qualification and limits

Releasing the resident owner or closing the viewer releases its encoded source
and display buffers. Failed/cancelled admissions must release owned allocations.
Do not delete original acquisitions, canonical containers, or retained archive
payloads as part of branch or benchmark cleanup.

## Maintainer checks

- Build Show4DSTEM with `QUANTEM_GPU_SRC` pointing to this package's `src` tree.
- Run count-ANS integer round trips and browser-export tests for native dtypes.
- Run quantem.widget's display tests for display buffer ownership and resident
  pattern means; quantem.gpu no longer ships a browser display.
- On an identified real adapter, run native source parity.
- Drive selected and averaged scan drags with the button held, then verify the
  endpoint against original counts. Separately check detector drags, contrast,
  zoom, copy, progressive loading, cancellation, and device loss.
- Count fresh scientific output, not animation callbacks or repeated paints.
