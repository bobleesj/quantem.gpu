# Native encoded inputs and scaled output

Native Swift clients can compose encoded acquisition storage, float32 image
operations, and bounded scaled-output writing without Python or Torch.
Scientific algorithms and their stage order remain in the consuming package.
For example, the MAPED sequence lives in QuantEM's `QuantEMMAPED` product.

| Product | Public type | Responsibility |
|---|---|---|
| `Metal4DSTEMStreamingIO` | `MetalEncodedSource` | Exact uint8/uint16 residency, GPU median correction, complete means, bounded reads |
| `Metal4DSTEMStreamingIO` | `MetalHDF5Reader` | Borrowed bounded buffers from the existing GPU HDF5 decoder |
| `Metal4DSTEMStreamingIO` | `MetalPrecision` | Global range, scaled uint16 codes, restoration, complete GPU error measurements |
| `Metal4DSTEMStreamingIO` | `MetalHDF5Writer` | GPU bitshuffle/LZ4 compression and atomic standard HDF5 output |
| `Metal4DSTEMStreamingIO` | `MetalPackedSource` | Full packed output residency, bounded restored-intensity reads |
| `MetalScientificNumerics` | `MetalImageOperations`, `GPUImage` | Gaussian/Sobel filters, windows, MPSGraph FFTs, refined correlation, translation and weighted accumulation |

## Precision and lifetime

`MetalPrecision.includeRange` consumes float32 GPU regions. After the complete
range pass, `calibrate(shape:)` fixes one scale and offset. `convert` produces
uint16 codes and accumulates restored-value error metrics on the GPU. `finish`
requires exactly the declared number of values and returns the existing
`quantem_precision_v1` report schema. Constant ranges use scale 1. Nonfinite
and float32 subnormal input is rejected explicitly.

The kernels are shared with the Python Metal precision implementation. Rounding
compares the source against the code midpoint in float-float arithmetic;
rounding a float32 quotient alone can pick the wrong code near a half step.
Saved coefficient parsing uses `JSONDecoder` to avoid NSNumber double rounding.

`MetalHDF5Writer.append` accepts consecutive uint16 frame buffers. All compression
and chunk assembly run on Metal; C HDF5 writes completed bytes and metadata.
`finish(metadata:)` publishes a complete new file and refuses to overwrite an
existing destination. An unfinished writer removes its own temporary file.
The native reader currently requires detector pixel counts divisible by 4096.
Float16 export is not yet part of this Swift API.

`MetalPackedSource.load` consumes the saved report, reads the file once, and
packs its codes without a second conversion. `read` restores up to 4096 frames
on the GPU. Call `releaseResidentStorage` when the owner is finished. A scientific
workflow should release owned inputs before reopening a full output if those
inputs are no longer needed.

## Shared count codec

Swift and Python MPS initialize their encoding and decoding tables on the GPU
from the same immutable model frequencies. The runtime performs no host
histogram, model fitting, or probability-table construction. The independent
NumPy reference remains only for tests and constant generation. Hot-pixel
correction excludes other masked neighbors from the local 3×3 median and
preserves native integer values exactly after that explicit correction.

The native MAPED consumer's `native/Tests/QuantEMMAPEDTests` checks integer
reads, correction, means, image operators, scaled codes, HDF5 reopening and
lifetime. Its `native/Tests/compare_torch.py` checks real seven-tilt outputs
against the existing Torch MPS workflow and the Python file loader. QuantEM.GPU's
`tests/hardware/mps/test_precision.py` and `test_ans_resident.py` protect the
shared kernels and compare their outputs with independent NumPy references.
These scoped tests do not constitute a minimum-memory Mac or application release
qualification.
