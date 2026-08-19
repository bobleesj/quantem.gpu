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

## 512×512 performance contract

Browser BF/ADF images are commonly `512×512`. That FFT must stay inside a
120 Hz frame after the graph is warm.

Measured on Apple M5 Max, in-place `logMagnitude`, same display contract as
PyTorch `fftshift(log1p(abs(fft2(x))))`:

| Shape | Dtype | Metal p50 | Metal FPS | Torch MPS p50 | Torch MPS FPS |
|---|---|---:|---:|---:|---:|
| 256×256 | float32 | 0.35 ms | 2850 | 0.44 ms | 2252 |
| **512×512** | **uint32 BF** | **≤ 8.33 ms required** | **≥ 120** | 0.58 ms | 1712 |
| 512×512 | float32 | 0.31 ms | 3236 | 0.58 ms | 1712 |
| 2048×2048 | float32 | 0.90 ms | 1111 | 2.29 ms | 437 |

Release checks:

```bash
swift test --filter MetalImageFFTTests
swift run metal-image-fft-benchmark 512 512 12
python src/quantem/gpu/swift/Benchmarks/MetalImageFFTBenchmark/compare_torch_fft.py 512 512 12
```

`testWarm512UInt32BrightFieldFFTStaysInside120Hz` is the BF/ADF gate. Do not
treat CPU `fft2`, software adapters, or a first-compile hitch as 120 Hz
evidence.

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
