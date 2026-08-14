import Foundation
import Metal
import MetalDisplayKernels

enum BenchmarkError: LocalizedError {
    case unavailable(String)

    var errorDescription: String? {
        switch self {
        case .unavailable(let message): message
        }
    }
}

private struct Measurement {
    let wallMilliseconds: Double
    let gpuMilliseconds: Double
}

@main
struct MetalDisplayBenchmark {
    static func main() throws {
        let side = benchmarkSide()
        let count = side * side
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkError.unavailable("No Metal device is available.")
        }
        guard let queue = device.makeCommandQueue() else {
            throw BenchmarkError.unavailable("Could not create a Metal command queue.")
        }
        let library = try MetalDisplayKernels.makeLibrary(device: device)
        let rangePipeline = try computePipeline(
            device: device,
            library: library,
            name: MetalDisplayKernels.rangeFunction
        )
        let histogramPipeline = try computePipeline(
            device: device,
            library: library,
            name: MetalDisplayKernels.histogramFunction
        )
        let renderPipeline = try makeRenderPipeline(device: device, library: library)
        let values = (0 ..< count).map { UInt32(($0 * 17) % 4096) }
        let valueBuffer = try makeBuffer(device: device, values: values)
        let rangeBuffer = try require(
            device.makeBuffer(length: 2 * MemoryLayout<UInt32>.stride),
            "Could not allocate the range buffer."
        )
        let binsBuffer = try require(
            device.makeBuffer(length: 256 * MemoryLayout<UInt32>.stride),
            "Could not allocate the histogram buffer."
        )
        let lutBuffer = try MetalDisplayKernels.makeLUTBuffer(
            device: device,
            colormap: .viridis
        )
        let texture = try makeTexture(device: device, side: side)

        try calculateRange(
            queue: queue,
            pipeline: rangePipeline,
            values: valueBuffer,
            range: rangeBuffer,
            count: count
        )
        let range = rangeBuffer.contents().bindMemory(to: UInt32.self, capacity: 2)
        guard range[0] == 0, range[1] == 4095 else {
            throw BenchmarkError.unavailable(
                "Range parity failed: expected 0...4095, got \(range[0])...\(range[1])."
            )
        }

        var parameters = MetalDisplayParameters(
            rows: side,
            cols: side,
            low: range[0],
            high: range[1],
            scale: .linear
        )
        let linearMilliseconds = try benchmarkRender(
            queue: queue,
            pipeline: renderPipeline,
            values: valueBuffer,
            parameters: &parameters,
            lut: lutBuffer,
            texture: texture
        )
        parameters.scaleMode = MetalDisplayScale.logarithmic.rawValue
        let logMilliseconds = try benchmarkRender(
            queue: queue,
            pipeline: renderPipeline,
            values: valueBuffer,
            parameters: &parameters,
            lut: lutBuffer,
            texture: texture
        )
        let histogramMilliseconds = try benchmarkHistogram(
            queue: queue,
            pipeline: histogramPipeline,
            values: valueBuffer,
            bins: binsBuffer,
            parameters: &parameters,
            count: count
        )
        let bins = binsBuffer.contents().bindMemory(to: UInt32.self, capacity: 256)
        let histogramSum = (0 ..< 256).reduce(UInt64(0)) { $0 + UInt64(bins[$1]) }
        guard histogramSum == UInt64(count) else {
            throw BenchmarkError.unavailable(
                "Histogram parity failed: expected \(count), got \(histogramSum)."
            )
        }

        print("DISPLAY device=\(device.name) shape=\(side)x\(side) dtype=uint32 lut=viridis")
        print("PARITY range=\(range[0]):\(range[1]) histogram_sum=\(histogramSum)")
        print(
            String(
                format: "WALL linear_render_median_ms=%.4f "
                    + "log_render_median_ms=%.4f histogram_median_ms=%.4f",
                median(linearMilliseconds.map(\.wallMilliseconds)),
                median(logMilliseconds.map(\.wallMilliseconds)),
                median(histogramMilliseconds.map(\.wallMilliseconds))
            )
        )
        print(
            String(
                format: "GPU linear_render_median_ms=%.4f "
                    + "log_render_median_ms=%.4f histogram_median_ms=%.4f",
                median(linearMilliseconds.map(\.gpuMilliseconds)),
                median(logMilliseconds.map(\.gpuMilliseconds)),
                median(histogramMilliseconds.map(\.gpuMilliseconds))
            )
        )
    }

    private static func benchmarkSide() -> Int {
        guard let argument = CommandLine.arguments.dropFirst().first,
              let value = Int(argument), value > 0
        else { return 512 }
        return value
    }

    private static func computePipeline(
        device: MTLDevice,
        library: MTLLibrary,
        name: String
    ) throws -> MTLComputePipelineState {
        let function = try require(
            library.makeFunction(name: name),
            "Missing Metal function \(name)."
        )
        return try device.makeComputePipelineState(function: function)
    }

    private static func makeRenderPipeline(
        device: MTLDevice,
        library: MTLLibrary
    ) throws -> MTLRenderPipelineState {
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = try require(
            library.makeFunction(name: MetalDisplayKernels.vertexFunction),
            "Missing display vertex function."
        )
        descriptor.fragmentFunction = try require(
            library.makeFunction(name: MetalDisplayKernels.fragmentFunction),
            "Missing display fragment function."
        )
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        return try device.makeRenderPipelineState(descriptor: descriptor)
    }

    private static func makeTexture(device: MTLDevice, side: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: side,
            height: side,
            mipmapped: false
        )
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        return try require(
            device.makeTexture(descriptor: descriptor),
            "Could not allocate output texture."
        )
    }

    private static func calculateRange(
        queue: MTLCommandQueue,
        pipeline: MTLComputePipelineState,
        values: MTLBuffer,
        range: MTLBuffer,
        count: Int
    ) throws {
        let pointer = range.contents().bindMemory(to: UInt32.self, capacity: 2)
        pointer[0] = .max
        pointer[1] = 0
        let commandBuffer = try require(
            queue.makeCommandBuffer(),
            "Could not create command buffer."
        )
        let encoder = try require(
            commandBuffer.makeComputeCommandEncoder(),
            "Could not create range encoder."
        )
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(range, offset: 0, index: 1)
        var valueCount = UInt32(count)
        withUnsafeBytes(of: &valueCount) { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
        }
        dispatch(encoder, pipeline: pipeline, count: count)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try requireCompleted(commandBuffer)
    }

    private static func benchmarkRender(
        queue: MTLCommandQueue,
        pipeline: MTLRenderPipelineState,
        values: MTLBuffer,
        parameters: inout MetalDisplayParameters,
        lut: MTLBuffer,
        texture: MTLTexture
    ) throws -> [Measurement] {
        var measurements: [Measurement] = []
        for iteration in 0 ..< 110 {
            let commandBuffer = try require(
                queue.makeCommandBuffer(),
                "Could not create command buffer."
            )
            let descriptor = MTLRenderPassDescriptor()
            descriptor.colorAttachments[0].texture = texture
            descriptor.colorAttachments[0].loadAction = .dontCare
            descriptor.colorAttachments[0].storeAction = .store
            let encoder = try require(
                commandBuffer.makeRenderCommandEncoder(descriptor: descriptor),
                "Could not create render encoder."
            )
            encoder.setRenderPipelineState(pipeline)
            encoder.setFragmentBuffer(values, offset: 0, index: 0)
            withUnsafeBytes(of: &parameters) { bytes in
                encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: 1)
            }
            encoder.setFragmentBuffer(lut, offset: 0, index: 2)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()
            let started = DispatchTime.now().uptimeNanoseconds
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsed = DispatchTime.now().uptimeNanoseconds - started
            try requireCompleted(commandBuffer)
            if iteration >= 10 {
                measurements.append(
                    Measurement(
                        wallMilliseconds: Double(elapsed) / 1_000_000,
                        gpuMilliseconds: max(
                            0,
                            commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
                        ) * 1000
                    )
                )
            }
        }
        return measurements
    }

    private static func benchmarkHistogram(
        queue: MTLCommandQueue,
        pipeline: MTLComputePipelineState,
        values: MTLBuffer,
        bins: MTLBuffer,
        parameters: inout MetalDisplayParameters,
        count: Int
    ) throws -> [Measurement] {
        var measurements: [Measurement] = []
        for iteration in 0 ..< 110 {
            memset(bins.contents(), 0, bins.length)
            let commandBuffer = try require(
                queue.makeCommandBuffer(),
                "Could not create command buffer."
            )
            let encoder = try require(
                commandBuffer.makeComputeCommandEncoder(),
                "Could not create histogram encoder."
            )
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(values, offset: 0, index: 0)
            encoder.setBuffer(bins, offset: 0, index: 1)
            withUnsafeBytes(of: &parameters) { bytes in
                encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
            }
            dispatch(encoder, pipeline: pipeline, count: count)
            encoder.endEncoding()
            let started = DispatchTime.now().uptimeNanoseconds
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsed = DispatchTime.now().uptimeNanoseconds - started
            try requireCompleted(commandBuffer)
            if iteration >= 10 {
                measurements.append(
                    Measurement(
                        wallMilliseconds: Double(elapsed) / 1_000_000,
                        gpuMilliseconds: max(
                            0,
                            commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
                        ) * 1000
                    )
                )
            }
        }
        return measurements
    }

    private static func makeBuffer<T>(device: MTLDevice, values: [T]) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
            try require(
                device.makeBuffer(
                    bytes: bytes.baseAddress!,
                    length: bytes.count,
                    options: .storageModeShared
                ),
                "Could not allocate input buffer."
            )
        }
    }

    private static func dispatch(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        count: Int
    ) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, max(1, count))
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    private static func require<T>(_ value: T?, _ message: String) throws -> T {
        guard let value else { throw BenchmarkError.unavailable(message) }
        return value
    }

    private static func requireCompleted(_ commandBuffer: MTLCommandBuffer) throws {
        guard commandBuffer.status == .completed else {
            throw BenchmarkError.unavailable(
                commandBuffer.error?.localizedDescription ?? "Metal command did not complete."
            )
        }
    }
}
