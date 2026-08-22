# Native Metal image endpoints

These Swift package products are the reusable Apple-GPU endpoints for native
clients. They own mathematics and GPU residency. They do not own SwiftUI,
gestures, cache policy, or Python.

Native clients should call these APIs instead of copying Metal source or
launching a local Python backend.

## Products

| Product | Endpoint | Use |
|---|---|---|
| `MetalImageFFT` | `MetalImageFFT.logMagnitude` | Browser FFT of an already-transferred 2D product |
| `MetalImageRuntime` | `MetalHistogramContrast`, `MetalDisplayStatistics` | Histogram windows, range, and display contracts |
| `MetalDisplayKernels` | LUT / histogram / render shaders | Colormap and display kernels |
| `Metal4DSTEMKernels` | decode / detector / CoM / iDPC kernels | Raw-data science on Apple GPU |
| `Native4DSTEMIO` | catalog, QH5 index, EMD calibration | Native HDF5/EMD discovery |

Import the products from the repository-root Swift package. Do not copy
`.metal` files into an application.

## Browser FFT

The display contract is `fftshift(log1p(abs(fft2(source))))` in row-major
`Float32`. Forward transforms are unnormalized. Reciprocal-space labels come
from the caller’s scan calibration, not from FFT amplitude.

```swift
import Metal
import MetalImageFFT

let fft = try MetalImageFFT(device: device)
try fft.prewarm(rows: scanRows, columns: scanCols)
let result = try fft.logMagnitude(
  source: brightField.values,
  rows: scanRows,
  columns: scanCols,
  scalarType: .uint32,
  output: existingFFT.values
)
```

`source` is a GPU-resident `MTLBuffer`. Typical Browser products:

| Product | `scalarType` | Typical scan |
|---|---|---|
| BF / ABF / ADF / custom | `.uint32` | `512×512` |
| CoM / DPC / iDPC | `.float32` | same scan shape |

`output` is the in-place destination. Pass the already-displayed FFT buffer so
warm updates do not allocate. `result.minimum` is `0`; `result.maximum` is the
GPU-reduced log-magnitude peak.

Call `prewarm(rows:columns:)` once after the scan shape is known, during load,
so the first visible FFT is not an MPSGraph compile hitch.

## 512×512 performance gate

Browser BF/ADF images are commonly `512×512`. After prewarming, the FFT gate is
completion within **8.33 ms** for a 120 Hz frame. This is an acceptance budget,
not a copied benchmark result. Current measurements, device identity, revision,
distribution, and parity live in [Verified benchmark results](../performance/results.md).

Release checks:

```bash
swift test --filter MetalImageFFTTests
swift run metal-image-fft-benchmark 512 512 12
python src/quantem/gpu/swift/Benchmarks/MetalImageFFTBenchmark/compare_torch_fft.py 512 512 12
```

`testWarm512UInt32BrightFieldFFTStaysInside120Hz` is the BF/ADF gate. Do not
treat CPU `fft2`, software adapters, a first-compile hitch, or a historical
table copied into an API contract as 120 Hz evidence.

## Histogram and contrast

```swift
import MetalImageRuntime

let statistics = try MetalDisplayStatistics(device: device)
let analyzed = try statistics.analyzeUInt32(
  values: brightField.values,
  rows: scanRows,
  columns: scanCols,
  scale: .linear
)
let window = MetalHistogramContrast.percentileWindow(
  bins: analyzed.bins,
  lowerPercentile: 0.01,
  upperPercentile: 0.99
)
```

Contrast, colormap, pan, and zoom stay in the client. They must not recompute
the FFT or request raw 4D data.

## Client rules

- Local macOS/iOS Explore is Swift + Metal + native HDF5 only.
- Do not bundle or launch Python, NumPy, or h5py in the app.
- Linux CUDA hosts may use Python `quantem.gpu`; that stays on the service host.
- The app owns cache, latest-wins scheduling, and SwiftUI. This package owns
  the buffer-to-buffer math.
