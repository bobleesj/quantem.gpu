# Native Direct3D FFT extraction (not release-qualified)

This .NET 8 target runs HLSL on a caller-owned D3D11 device/context. It does
not require Python, SSH, WinUI, or a Windows compute server. It has no CPU
fallback and does not choose a hardware adapter; the consumer enforces its
hardware policy. Vortice packages remain pinned to 3.8.3.

```csharp
using var fft = new Quantem.Gpu.Display.Direct3D.ImageFft(device, context);
var output = fft.LogMagnitude(sourceView, rows, columns,
    Quantem.Gpu.Display.Direct3D.ScalarType.UInt32);
// Consume output on the same context before reusing this shape.
```

Source is a row-major structured uint32 buffer, with float32 supplied as IEEE
bits when selected. Output is float32 log(1 + abs(fftshift(fft2(source)))).
Both axes must be powers of two >= 2 within the D3D dispatch limit. Unsupported
shapes fail; there is no hidden padding, binning, crop, or CPU retry. Uint32
input is converted to float32 for the FFT, as in the frozen implementation;
this is not an integer-exact reduction. Nonfinite input and output become zero.

The caller keeps device/context and input alive, serializes context use, and
provides same-device views with exactly rows*columns words. The library owns
shaders and shape-keyed scratch buffers. Returned output is borrowed, remains
resident, and is overwritten on the next same-shape call. The caller must
unbind any output SRV before another call and consume results before reuse.
The compute shader slots are cleared, not restored. Dispose the FFT instance
to release its scratch buffers; the borrowed device/context are not disposed.

This initial extraction intentionally leaves admission/eviction and UI state
with the consumer. The library caches one workspace per used shape until
disposal. It performs no source upload, readback, transport, or presentation.
The existing Windows wrapper still performs its prior one-time final readback.
Removing that transfer is separate measured work, not an extraction claim.

The four HLSL shader bodies are unchanged from Windows source cd780db.
Tests use an independent direct DFT for square/rectangular uint32/float32
inputs, nonfinite handling, source immutability, deterministic warm results,
buffer reuse, and borrowed-device lifetime. Frozen max absolute error is
< 2e-3, inherited from the existing Windows FFT test, not a new tolerance.

```sh
dotnet run --project tests/direct3d/Display.Tests.csproj -c Release
```

Non-Windows execution prints SKIP for hardware parity. Compilation alone is
not scientific validation. Parallels cannot prove physical NVIDIA/120 Hz.
Real-data and exact-package headed Windows evidence are required before
removing the consumer's old implementation or publishing a native package.
This target is not packable yet. Histogram/reduction/packing, other scientific
domains, and Python backend registration are not implemented by this target.
