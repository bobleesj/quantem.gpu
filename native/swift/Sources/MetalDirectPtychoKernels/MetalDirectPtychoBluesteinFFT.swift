import Foundation
import Metal

/// Exact-grid Fourier preparation for large non-power-of-two scan axes.
///
/// MPSGraph's large irregular transforms exceeded the Fourier evidence error
/// bound in independent impulse tests. Bluestein's convolution retains the
/// original grid and every sample; its power-of-two padding is internal only.
/// All data transforms and the convolution-kernel FFT run on Metal. The CPU
/// builds only bounded, data-independent chirp/twiddle tables in Double before
/// rounding them once to Float. No dense input or spectrum crosses the CPU.
final class MetalDirectPtychoBluesteinFFT {
  private struct Parameters {
    var rows, columns, count, convolutionCount, axis, kernelOnly: UInt32
  }

  private struct AxisPlan {
    let chirp: MTLBuffer
    let twiddle: MTLBuffer
    let kernel: MTLBuffer
    let parameters: Parameters
  }

  let kernel: MetalKernelIdentity
  private let pipeline: MTLComputePipelineState
  private var plans: [AxisPlan] = []

  /// Keep the qualified small-grid and native square paths unchanged.
  static func isRequired(rows: Int, columns: Int) -> Bool {
    [rows, columns].contains { $0 > 256 && ($0 & ($0 - 1)) != 0 }
  }

  var tableBytes: Int {
    plans.reduce(0) { $0 + $1.chirp.length + $1.twiddle.length + $1.kernel.length }
  }

  init(device: MTLDevice, queue: MTLCommandQueue, rows: Int, columns: Int) throws {
    guard (3...512).contains(rows), (3...512).contains(columns) else {
      throw MetalDirectPtychoError.invalidGeometry(
        "Bluestein preparation requires scan axes between 3 and 512")
    }
    let entry = DirectPtychoKernelRegistry.rectangularFourier
    kernel = MetalKernelIdentity(
      name: entry.name, version: entry.version,
      sourceSHA256: try DirectPtychoKernelRegistry.sourceSHA256(entry),
      variant: "rows=\(rows) columns=\(columns) dc=centered dtype=complex64")
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try device.makeLibrary(
      source: MetalDirectPtychoShaderSource.load(["direct_bluestein"]), options: options)
    guard let function = library.makeFunction(name: "direct_ptycho_bluestein") else {
      throw MetalDirectPtychoError.missingFunction("direct_ptycho_bluestein")
    }
    pipeline = try device.makeComputePipelineState(function: function)
    guard let command = queue.makeCommandBuffer() else { throw MetalDirectPtychoError.commandQueue }
    for (axis, count) in [(1, columns), (0, rows)] {
      var convolutionCount = 1
      while convolutionCount < 2 * count - 1 { convolutionCount *= 2 }
      guard pipeline.maxTotalThreadsPerThreadgroup >= convolutionCount / 2,
        device.maxThreadgroupMemoryLength >= convolutionCount * 8
      else {
        throw MetalDirectPtychoError.invalidGeometry(
          "Metal device cannot prepare this exact rectangular FFT")
      }
      let chirp = try Self.makeTable(
        device: device,
        values: (0..<count).map { j in
          let angle = Double.pi * Double((j * j) % (2 * count)) / Double(count)
          return SIMD2<Float>(Float(cos(angle)), Float(-sin(angle)))
        })
      let twiddle = try Self.makeTable(
        device: device,
        values: (0..<convolutionCount / 2).map { j in
          let angle = 2 * Double.pi * Double(j) / Double(convolutionCount)
          return SIMD2<Float>(Float(cos(angle)), Float(-sin(angle)))
        })
      guard
        let kernel = device.makeBuffer(length: convolutionCount * 8, options: .storageModePrivate)
      else {
        throw MetalDirectPtychoError.allocation("rectangular FFT convolution kernel")
      }
      let parameters = Parameters(
        rows: UInt32(rows), columns: UInt32(columns), count: UInt32(count),
        convolutionCount: UInt32(convolutionCount), axis: UInt32(axis), kernelOnly: 0)
      let plan = AxisPlan(chirp: chirp, twiddle: twiddle, kernel: kernel, parameters: parameters)
      plans.append(plan)
      encodeAxis(
        command: command, input: kernel, output: kernel, plan: plan, batch: 1, kernelOnly: true)
    }
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error {
      throw MetalDirectPtychoError.commandExecution(error.localizedDescription)
    }
  }

  /// Transform each row then column, reusing the engine's existing two workspaces.
  func encode(command: MTLCommandBuffer, input: MTLBuffer, temporary: MTLBuffer, batch: Int) {
    encodeAxis(command: command, input: input, output: temporary, plan: plans[0], batch: batch)
    encodeAxis(command: command, input: temporary, output: input, plan: plans[1], batch: batch)
  }

  private func encodeAxis(
    command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer,
    plan: AxisPlan, batch: Int, kernelOnly: Bool = false
  ) {
    var parameters = plan.parameters
    parameters.kernelOnly = kernelOnly ? 1 : 0
    let encoder = command.makeComputeCommandEncoder()!
    encoder.label = "Exact-grid Bluestein FFT"
    encoder.setComputePipelineState(pipeline)
    for (index, buffer) in [input, output, plan.chirp, plan.twiddle, plan.kernel].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 5)
    encoder.setThreadgroupMemoryLength(Int(parameters.convolutionCount) * 8, index: 0)
    let lines = kernelOnly ? 1 : Int(parameters.axis == 1 ? parameters.rows : parameters.columns)
    encoder.dispatchThreadgroups(
      MTLSize(width: lines, height: batch, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: Int(parameters.convolutionCount) / 2, height: 1, depth: 1))
    encoder.endEncoding()
  }

  private static func makeTable(device: MTLDevice, values: [SIMD2<Float>]) throws -> MTLBuffer {
    guard
      let buffer = values.withUnsafeBytes({ bytes in
        device.makeBuffer(
          bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
      })
    else { throw MetalDirectPtychoError.allocation("rectangular FFT twiddle table") }
    return buffer
  }
}
