# Experimental lossless resident ANS

ANS means **asymmetric numeral systems**. Supported original acquisitions now
default to ANS on CUDA/MPS through `io.load(path)`. This page retains experimental
browser/archive integrations; their qualification is distinct from that default.
See the [current IO guide](../api/io.md) for supported formats and the
[acceptance gate](../maintainer/ans-io-acceptance.md) for runtime coverage.
Source integration does not mean these changes have been released on PyPI.

## Compatibility safeguards

The loader detects encoded sources. Explicit reference and prepared packed inputs
retain their separate contracts. Encoding writes a new container atomically
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
| Saved `.qem` copies | `io.save`, `io.load`, `_streamed_file`, CUDA count decoder, native reader | CUDA/MPS resident queries; native Live4DSTEM macOS |
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
io.save("acquisition.qem", resident)
source = io.load("acquisition.qem", backend="cuda", device=0).data
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
acquisition. See [the container contract](../developer/count-ans.md) for dtype, checksums,
invalid pixels, conversion, and lifetime rules.

On Python MPS, exact binary-mask reductions decode into a bounded shared staging
buffer and queue the decode/reduce pairs in groups of 128 before waiting. This
removes the former per-chunk command-buffer synchronization while preserving
the same native-count decode, `uint64` output, scan order, and error checks. The
queue change is covered by the 2026-09-11 exact queue-stress record; it is an
implementation improvement, not evidence of full-acquisition or 120 Hz ANS
throughput. When several binary detector products are requested together,
`detector_sums_device` uses one decode per source chunk and a bounded multi-mask
reduction kernel, so BF/ABF/ADF-style products share the decoded counts without
an additional dense 4D buffer. Exact `uint64` parity is covered by the
2026-09-11 multi-mask record. A real multi-acquisition ANS archive is still
required before seven-tilt or 120 Hz claims can be qualified.

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
adapter overlaps independent Metal queues and returns only the requested point
patterns or detector products; each `FourDSTEMData` owner remains caller-owned
and must be closed after the session is finished. This is the supported
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

Open the generated launcher/viewer and use **Open count-ANS files** to grant
access to the linked source files in `ans-viewer/rans`. Export preserves native
shape and dtype, links source containers, and does not decode or duplicate the
full acquisition. Keep the source files available. The exporter also accepts a
retained detector-rANS build manifest, explicitly through its legacy adapter.

The direct `Show4DSTEM(source)` CUDA route currently expects the existing
resident-owner protocol. Do not infer that every object returned by
`io.load(..., representation="encoded")` implements that widget protocol. The
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
