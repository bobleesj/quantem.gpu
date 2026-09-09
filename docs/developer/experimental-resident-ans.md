# Experimental lossless resident ANS

ANS means **asymmetric numeral systems**. This feature is opt-in and experimental:
container compatibility, admission policies, and private adapters may change.
It is not a new default for ordinary HDF5, dense, or packed-data workflows.
Use the matching experimental `quantem.gpu` and `quantem.widget` revisions.
Source integration does not mean that these changes have been released on PyPI.

## Compatibility safeguards

Encoded sources are explicitly selected. Ordinary HDF5, dense and packed inputs
retain their established workflows. Encoding writes a new container atomically
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
| Canonical count-ANS v1 `.ans` files | `io.save`, `io.load`, private `_ans` modules, CUDA count decoder, WebGPU `count-ans.ts` | CUDA resident queries; Show4DSTEM WebGPU export |
| Retained detector-rANS manifest | Validated legacy adapter and WebGPU `rans.ts` | Existing resident CUDA owner or Show4DSTEM WebGPU export |
| Retained source112 tANS archive | Package exporter and WebGPU `source112.ts` | Experimental progressive Show4DSTEM resident series |

These are explicitly identified formats, not interchangeable byte streams.
The canonical writer is shared; compatibility readers retain older experiments
without rewriting source data. Full-series conversion of every retained archive
to the canonical container is not a completed acceptance gate.

Huffman checkpoints are an internal source112 acceleration representation for
selected entropy models. They do not make Huffman a mandatory codec for the
canonical ANS container or ordinary bitpacked data. General WebGPU bitpacked
input needs its own supported loader and parity tests; ANS support alone does
not establish that support.

## Encode and query canonical counts

```python
from quantem.gpu import io

# counts: native uint8/uint16, (scan_row, scan_col, detector_row, detector_col)
saved = io.save("acquisition.ans", counts, format="quantem", compression="ans", backend="cpu")
source = io.load(saved.path, backend="cuda", representation="ans", device=0).data
try:
    pattern = source.extract_diffraction_device(0, 0)
    image = source.detector_sum_device(binary_detector_mask)
finally:
    source.release()
```

`device=0` means the first CUDA-visible device; select the intended physical GPU
before launching Python. Encoding here is the bounded CPU reference encoder,
not a claim of real-time full-acquisition compression. CUDA residency retains
encoded buffers and produces device results without constructing a full dense
acquisition. See [the container contract](count-ans.md) for dtype, checksums,
invalid pixels, conversion, and lifetime rules.

## Show4DSTEM WebGPU

The widget package supplies the browser export, using the same package reader:

```python
from quantem.widget.show4dstem_webgpu_export import export_show4dstem_rans_viewer

html = export_show4dstem_rans_viewer(
    ["acquisition.ans", "next-acquisition.ans"],
    "ans-viewer",
    frame_labels=["First", "Second"],
)
```

Open the generated launcher/viewer and use **Open count-ANS files** to grant
access to the linked source files in `ans-viewer/rans`. Export preserves native
shape and dtype, links source containers, and does not decode or duplicate the
full acquisition. Keep the source files available. The exporter also accepts a
retained detector-rANS build manifest, explicitly through its legacy adapter.

The direct `Show4DSTEM(source)` CUDA route currently expects the existing
resident-owner protocol. Do not infer that every object returned by
`io.load(..., representation="ans")` implements that widget protocol. The
canonical-file workflow above is the browser export integration.

## Resident source112 DP behavior

For the retained source112 adapter, selected and averaged DPs now gather into
source-owned GPU buffers. The shared GPU display engine averages and renders
those buffers without a CPU readback per scan step. Metadata, histogram, and
hover values hydrate after a short pause; pending readouts are withheld instead
of labeling old values as the current pattern. Copy captures the visible GPU
canvas. Other source types retain their existing paths.

Raw hardware sentinel counts are preserved. The established detector validity
mask is applied only to displayed scientific products. Integer native patterns
and detector sums require exact reference equality. Count means use explicitly
rounded float32 division; native sums remain exact for the validated series.

## Qualification and limits

On the tested NVIDIA Blackwell adapter, three complete acquisitions loaded in
roughly 2.4–3.1 seconds and appeared progressively. The latest native-reference
gate compared 5,799,936 values exactly, including detector products, raw point
patterns, and retained display buffers. Averaged DP matched 442,368 reference
values exactly; a separate rounding check covered 319,059 integer ratios.

Continuous held dragging submitted about 47 selected DPs/s in the latest small
run; an earlier averaged run submitted about 37/s. Submissions are not physical
presentation measurements. Sustained 120 fps for both modes, all 66 acquisitions
under 20 seconds, real-time encoding, and multi-GPU browser placement are **not
qualified**. An earlier full66 load took about 62 seconds. These observations
are experimental evidence, not performance guarantees for other machines.

Releasing the resident owner or closing the viewer releases its encoded source
and display buffers. Failed/cancelled admissions must release owned allocations.
Do not delete original acquisitions, canonical containers, or retained archive
payloads as part of branch or benchmark cleanup.

## Maintainer checks

- Build Show4DSTEM with `QUANTEM_GPU_SRC` pointing to this package's `src` tree.
- Run count-ANS integer round trips and browser-export tests for native dtypes.
- Run `tests/webgpu/source112-pattern-buffer.ts` and
  `tests/webgpu/display-readback-lifecycle.ts` for ownership failures.
- On an identified real adapter, run native source parity and
  `tests/webgpu/resident-pattern-mean-parity.ts`.
- Drive selected and averaged scan drags with the button held, then verify the
  endpoint against original counts. Separately check detector drags, contrast,
  zoom, copy, progressive loading, cancellation, and device loss.
- Count fresh scientific output, not animation callbacks or repeated paints.
