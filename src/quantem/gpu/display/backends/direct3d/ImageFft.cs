using Vortice.D3DCompiler;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;

namespace Quantem.Gpu.Display.Direct3D;

public enum ScalarType { UInt32, Float32 }

/// <summary>
/// Enqueues float32 log(1 + abs(fftshift(fft2(source)))) on a borrowed D3D11
/// device/context. Does not upload, read back, present, or select an adapter.
/// Nonfinite inputs become zero, matching the frozen Windows implementation.
/// </summary>
/// <remarks>
/// Caller serializes this instance and all access to the borrowed immediate
/// context, and keeps device/context alive until disposal. Output is borrowed
/// and overwritten by the next call of the same shape; consume it before reuse.
/// Dispose after all queued consumers have been submitted. No CPU fallback.
/// </remarks>
public sealed class ImageFft : IDisposable
{
    private const string InitializeShader = """
        StructuredBuffer<uint> Source : register(t0);
        StructuredBuffer<uint> Parameters : register(t1);
        RWStructuredBuffer<float2> Output : register(u0);

        [numthreads(256, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID)
        {
            uint rows = Parameters[0];
            uint columns = Parameters[1];
            uint count = rows * columns;
            uint index = id.x;
            if (index >= count) return;
            uint row = index / columns;
            uint column = index % columns;
            uint logColumns = Parameters[3];
            uint sourceColumn = reversebits(column) >> (32 - logColumns);
            uint raw = Source[row * columns + sourceColumn];
            float value = Parameters[2] == 0 ? (float)raw : asfloat(raw);
            if (isnan(value) || isinf(value)) value = 0.0f;
            Output[index] = float2(value, 0.0f);
        }
        """;

    private const string BitReverseColumnsShader = """
        StructuredBuffer<float2> Source : register(t0);
        StructuredBuffer<uint> Parameters : register(t1);
        RWStructuredBuffer<float2> Output : register(u0);

        [numthreads(256, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID)
        {
            uint rows = Parameters[0];
            uint columns = Parameters[1];
            uint count = rows * columns;
            uint index = id.x;
            if (index >= count) return;
            uint row = index / columns;
            uint column = index % columns;
            uint sourceRow = reversebits(row) >> (32 - Parameters[4]);
            Output[index] = Source[sourceRow * columns + column];
        }
        """;

    private const string ButterflyShader = """
        StructuredBuffer<float2> Source : register(t0);
        StructuredBuffer<uint> Parameters : register(t1);
        RWStructuredBuffer<float2> Output : register(u0);

        float2 multiplyComplex(float2 a, float2 b)
        {
            return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }

        [numthreads(256, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID)
        {
            uint rows = Parameters[0];
            uint columns = Parameters[1];
            uint count = rows * columns;
            uint butterfly = id.x;
            if (butterfly >= count / 2) return;
            uint axis = Parameters[5];
            uint stage = Parameters[6];
            uint length = axis == 0 ? columns : rows;
            uint butterfliesPerLine = length / 2;
            uint lineIndex = butterfly / butterfliesPerLine;
            uint local = butterfly % butterfliesPerLine;
            uint span = 1u << (stage + 1);
            uint halfSpan = span >> 1;
            uint group = local / halfSpan;
            uint offset = local % halfSpan;
            uint first;
            uint second;
            if (axis == 0)
            {
                first = lineIndex * columns + group * span + offset;
                second = first + halfSpan;
            }
            else
            {
                first = (group * span + offset) * columns + lineIndex;
                second = first + halfSpan * columns;
            }
            float angle = -6.2831853071795864769f * (float)offset / (float)span;
            float sine;
            float cosine;
            sincos(angle, sine, cosine);
            float2 even = Source[first];
            float2 odd = multiplyComplex(Source[second], float2(cosine, sine));
            Output[first] = even + odd;
            Output[second] = even - odd;
        }
        """;

    private const string MagnitudeShader = """
        StructuredBuffer<float2> Source : register(t0);
        StructuredBuffer<uint> Parameters : register(t1);
        RWStructuredBuffer<uint> Output : register(u0);

        [numthreads(256, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID)
        {
            uint rows = Parameters[0];
            uint columns = Parameters[1];
            uint count = rows * columns;
            uint index = id.x;
            if (index >= count) return;
            uint outputRow = index / columns;
            uint outputColumn = index % columns;
            uint sourceRow = (outputRow + (rows + 1) / 2) % rows;
            uint sourceColumn = (outputColumn + (columns + 1) / 2) % columns;
            float2 value = Source[sourceRow * columns + sourceColumn];
            float magnitude = log(1.0f + length(value));
            Output[index] = asuint(isnan(magnitude) || isinf(magnitude) ? 0.0f : magnitude);
        }
        """;


    private readonly ID3D11Device _device;
    private readonly ID3D11DeviceContext _context;
    private readonly ID3D11ComputeShader _initialize;
    private readonly ID3D11ComputeShader _bitReverseColumns;
    private readonly ID3D11ComputeShader _butterfly;
    private readonly ID3D11ComputeShader _magnitude;
    private readonly ID3D11Buffer _parameters;
    private readonly ID3D11ShaderResourceView _parameterView;
    private readonly Dictionary<(int Rows, int Columns), Workspace> _workspaces = [];
    private readonly uint[] _parameterValues = new uint[8];
    private bool _disposed;

    public ImageFft(ID3D11Device device, ID3D11DeviceContext context)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(context);
        _device = device;
        _context = context;
        _initialize = Compile(InitializeShader, "Quantem.Gpu.FFT.initialize.hlsl");
        _bitReverseColumns = Compile(BitReverseColumnsShader, "Quantem.Gpu.FFT.bit-reverse-columns.hlsl");
        _butterfly = Compile(ButterflyShader, "Quantem.Gpu.FFT.butterfly.hlsl");
        _magnitude = Compile(MagnitudeShader, "Quantem.Gpu.FFT.magnitude.hlsl");
        _parameters = _device.CreateBuffer(
            8 * sizeof(uint), BindFlags.ShaderResource, ResourceUsage.Dynamic,
            CpuAccessFlags.Write, ResourceOptionFlags.BufferStructured, sizeof(uint));
        _parameterView = _device.CreateShaderResourceView(
            _parameters,
            new ShaderResourceViewDescription(ShaderResourceViewDimension.Buffer, Format.Unknown, 0, 8));
    }

    /// <summary>
    /// Source is a same-device structured uint buffer, one 32-bit word per
    /// row-major pixel; float input is passed as IEEE754 bits. Returns the
    /// same-size structured buffer of float32 log-magnitude bits, GPU-resident.
    /// Only power-of-two dimensions >= 2 are supported; no padding/cropping.
    /// Uint32-to-float32 conversion is intrinsic to this FFT, not count-preserving
    /// integer arithmetic. Original source bytes are never modified.
    /// </summary>
    public ID3D11Buffer LogMagnitude(
        ID3D11ShaderResourceView source, int rows, int columns, ScalarType scalarType)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(source);
        if (rows < 2 || columns < 2
            || !System.Numerics.BitOperations.IsPow2((uint)rows)
            || !System.Numerics.BitOperations.IsPow2((uint)columns))
            throw new ArgumentException("FFT requires power-of-two rows and columns >= 2; no padding or CPU fallback is applied.");
        if (scalarType is not (ScalarType.UInt32 or ScalarType.Float32))
            throw new ArgumentOutOfRangeException(nameof(scalarType));
        var count = checked(rows * columns);
        if ((long)count > 65535L * 256)
            throw new ArgumentOutOfRangeException(nameof(rows), "FFT exceeds the D3D11 one-dimensional dispatch limit.");
        var parameters = _parameterValues;
        parameters[0] = (uint)rows;
        parameters[1] = (uint)columns;
        parameters[2] = scalarType == ScalarType.UInt32 ? 0u : 1u;
        parameters[3] = (uint)System.Numerics.BitOperations.Log2((uint)columns);
        parameters[4] = (uint)System.Numerics.BitOperations.Log2((uint)rows);
        if (!_workspaces.TryGetValue((rows, columns), out var workspace))
        {
            workspace = new Workspace(_device, count);
            _workspaces.Add((rows, columns), workspace);
        }
        SetParameters(parameters);
        Dispatch(_initialize, source, workspace.First.Uav, count);
        var input = workspace.First;
        var output = workspace.Second;
        for (var stage = 0; stage < parameters[3]; stage++)
        {
            parameters[5] = 0;
            parameters[6] = (uint)stage;
            SetParameters(parameters);
            Dispatch(_butterfly, input.View, output.Uav, count / 2);
            (input, output) = (output, input);
        }
        SetParameters(parameters);
        Dispatch(_bitReverseColumns, input.View, output.Uav, count);
        (input, output) = (output, input);
        for (var stage = 0; stage < parameters[4]; stage++)
        {
            parameters[5] = 1;
            parameters[6] = (uint)stage;
            SetParameters(parameters);
            Dispatch(_butterfly, input.View, output.Uav, count / 2);
            (input, output) = (output, input);
        }
        SetParameters(parameters);
        Dispatch(_magnitude, input.View, workspace.MagnitudeUav, count);
        return workspace.Magnitude;
    }

    private void Dispatch(ID3D11ComputeShader shader, ID3D11ShaderResourceView source, ID3D11UnorderedAccessView output, int workItems)
    {
        _context.CSSetShader(shader);
        _context.CSSetShaderResources(0, [source, _parameterView]);
        _context.CSSetUnorderedAccessView(0, output);
        _context.Dispatch(checked((uint)((workItems + 255) / 256)), 1, 1);
        _context.CSUnsetUnorderedAccessView(0);
        _context.CSUnsetShaderResources(0, 2);
        _context.CSSetShader(null);
    }

    private void SetParameters(uint[] values)
        => _parameters.SetData(_context, values, MapMode.WriteDiscard);

    private ID3D11ComputeShader Compile(string source, string name)
    {
        var bytecode = Compiler.Compile(source, "main", name, "cs_5_0");
        return _device.CreateComputeShader(bytecode.Span);
    }


    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var workspace in _workspaces.Values) workspace.Dispose();
        _workspaces.Clear();
        _parameterView.Dispose();
        _parameters.Dispose();
        _magnitude.Dispose();
        _butterfly.Dispose();
        _bitReverseColumns.Dispose();
        _initialize.Dispose();
        // Device and context are borrowed, never disposed here.
    }

    private sealed class ComplexBuffer(ID3D11Buffer buffer, ID3D11ShaderResourceView view, ID3D11UnorderedAccessView uav) : IDisposable
    {
        public ID3D11ShaderResourceView View { get; } = view;
        public ID3D11UnorderedAccessView Uav { get; } = uav;
        public void Dispose() { Uav.Dispose(); View.Dispose(); buffer.Dispose(); }
    }

    private sealed class Workspace : IDisposable
    {
        public Workspace(ID3D11Device device, int count)
        {
            First = CreateComplex(device, count);
            Second = CreateComplex(device, count);
            Magnitude = device.CreateBuffer(
                checked((uint)(count * sizeof(uint))), BindFlags.UnorderedAccess | BindFlags.ShaderResource,
                ResourceUsage.Default, CpuAccessFlags.None,
                ResourceOptionFlags.BufferStructured, sizeof(uint));
            MagnitudeUav = device.CreateUnorderedAccessView(
                Magnitude,
                new UnorderedAccessViewDescription(
                    UnorderedAccessViewDimension.Buffer, Format.Unknown, 0, checked((uint)count)));
        }

        public ComplexBuffer First { get; }
        public ComplexBuffer Second { get; }
        public ID3D11Buffer Magnitude { get; }
        public ID3D11UnorderedAccessView MagnitudeUav { get; }

        private static ComplexBuffer CreateComplex(ID3D11Device device, int count)
        {
            var buffer = device.CreateBuffer(
                checked((uint)(count * 2 * sizeof(float))),
                BindFlags.ShaderResource | BindFlags.UnorderedAccess,
                ResourceUsage.Default, CpuAccessFlags.None,
                ResourceOptionFlags.BufferStructured, 2 * sizeof(float));
            var view = device.CreateShaderResourceView(
                buffer,
                new ShaderResourceViewDescription(
                    ShaderResourceViewDimension.Buffer, Format.Unknown, 0, checked((uint)count)));
            var uav = device.CreateUnorderedAccessView(
                buffer,
                new UnorderedAccessViewDescription(
                    UnorderedAccessViewDimension.Buffer, Format.Unknown, 0, checked((uint)count)));
            return new ComplexBuffer(buffer, view, uav);
        }

        public void Dispose()
        {
            MagnitudeUav.Dispose();
            Magnitude.Dispose();
            Second.Dispose();
            First.Dispose();
        }
    }
}
