using System.Numerics;
using System.Runtime.InteropServices;
using Quantem.Gpu.Display.Direct3D;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;

if (!OperatingSystem.IsWindows())
{
    Console.WriteLine("SKIP hardware FFT: Windows D3D11 required; compilation is not GPU parity.");
    return;
}
using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory1>();
IDXGIAdapter1? selected = null;
for (uint index = 0; factory.EnumAdapters1(index, out IDXGIAdapter1? candidate).Success; index++)
{
    if (candidate is null) continue;
    if ((candidate.Description1.Flags & AdapterFlags.Software) == 0)
    {
        selected = candidate;
        break;
    }
    candidate.Dispose();
}
using var adapter = selected ?? throw new Exception("No hardware D3D11 adapter; WARP is forbidden.");
D3D11.D3D11CreateDevice(adapter, DriverType.Unknown, DeviceCreationFlags.None,
    [FeatureLevel.Level_11_1, FeatureLevel.Level_11_0], out var device,
    out var featureLevel, out var context).CheckError();
using (device)
using (context)
{
    using (var fft = new ImageFft(device, context))
    {
        foreach (var (rows, columns) in new[] { (2, 2), (4, 8), (8, 4), (16, 16) })
        foreach (var type in new[] { ScalarType.UInt32, ScalarType.Float32 })
        {
            var count = rows * columns;
            var words = new uint[count];
            var values = new double[count];
            for (var i = 0; i < count; i++)
            {
                if (type == ScalarType.UInt32)
                {
                    words[i] = (uint)((i * 31 + 7) % 257);
                    values[i] = words[i];
                }
                else
                {
                    var value = (float)(Math.Sin(i * 0.37) * 5 + i % 3);
                    if (i == 2) value = float.NaN;
                    if (i == 3) value = float.PositiveInfinity;
                    words[i] = BitConverter.SingleToUInt32Bits(value);
                    values[i] = float.IsFinite(value) ? value : 0;
                }
            }
            using var source = device.CreateBuffer<uint>(words, BindFlags.ShaderResource,
                ResourceUsage.Immutable, CpuAccessFlags.None, ResourceOptionFlags.BufferStructured);
            using var view = device.CreateShaderResourceView(source,
                new ShaderResourceViewDescription(ShaderResourceViewDimension.Buffer,
                    Format.Unknown, 0, (uint)count));
            using var staging = device.CreateBuffer((uint)(count * 4), BindFlags.None,
                ResourceUsage.Staging, CpuAccessFlags.Read);
            byte[] Read(ID3D11Buffer result)
            {
                context.CopyResource(staging, result);
                var mapped = context.Map(staging, MapMode.Read, Vortice.Direct3D11.MapFlags.None);
                try { return mapped.AsSpan<byte>(count * 4).ToArray(); }
                finally { context.Unmap(staging, 0); }
            }
            var result = fft.LogMagnitude(view, rows, columns, type);
            var first = Read(result);
            var repeated = fft.LogMagnitude(view, rows, columns, type);
            if (result.NativePointer != repeated.NativePointer || !first.SequenceEqual(Read(repeated)))
                throw new Exception("Warm FFT failed buffer reuse or deterministic-output parity.");
            var actual = MemoryMarshal.Cast<byte, float>(first);
            double maximumError = 0;
            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
            {
                var frequencyRow = (row + rows / 2) % rows;
                var frequencyColumn = (column + columns / 2) % columns;
                var sum = Complex.Zero;
                for (var r = 0; r < rows; r++)
                for (var c = 0; c < columns; c++)
                    sum += values[r * columns + c] * Complex.FromPolarCoordinates(1,
                        -2 * Math.PI * ((double)frequencyRow * r / rows + (double)frequencyColumn * c / columns));
                var expected = Math.Log(1 + sum.Magnitude);
                var value = actual[row * columns + column];
                if (!float.IsFinite(value)) throw new Exception("FFT output is nonfinite.");
                maximumError = Math.Max(maximumError, Math.Abs(value - expected));
            }
            if (maximumError >= 2e-3)
                throw new Exception($"Frozen Windows FFT tolerance failed: {maximumError:G9}.");
            if (!Read(source).SequenceEqual(MemoryMarshal.AsBytes(words.AsSpan()).ToArray()))
                throw new Exception("FFT modified the scientific source buffer.");
            Console.WriteLine($"PASS {rows}x{columns} {type}: max_error={maximumError:G9}; stable buffer/output, source unchanged");
        }
    }
    // The library must not dispose the borrowed device/context.
    using var afterDispose = device.CreateBuffer(16, BindFlags.None, ResourceUsage.Staging, CpuAccessFlags.Read);
    context.Flush();
}
Console.WriteLine($"PASS borrowed D3D lifetime; {adapter.Description1.Description}, {featureLevel}. Virtual adapters are not physical NVIDIA/120 Hz proof.");
