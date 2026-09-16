import Foundation
import Metal
import Metal4DSTEMKernels

/// Exact full-mask queries over the saved 8/32-pixel integer sums.
/// Only edge residual columns are decoded; no detector evidence is dropped.
final class RuntimeSpatialQuery {
  let queue: MTLCommandQueue
  let leafPipeline: MTLComputePipelineState
  let rootPipeline: MTLComputePipelineState
  let indexPipeline: MTLComputePipelineState
  let parallelIndex: Bool
  let mask: MTLBuffer
  let valid: MTLBuffer
  let leaves: MTLBuffer
  let fields: MTLBuffer
  let fieldCoefficients: MTLBuffer
  let pixels: MTLBuffer
  let pixelCoefficients: MTLBuffer
  let counts: MTLBuffer
  let shape: [Int]
  let fieldCount: Int
  let tileCount: Int
  let rootCount: Int

  init(device: MTLDevice, shape: [Int], validity: [UInt8]) throws {
    self.shape = shape
    tileCount = ((shape[0] + 7) / 8) * ((shape[1] + 7) / 8)
    rootCount = ((shape[0] + 31) / 32) * ((shape[1] + 31) / 32)
    fieldCount = tileCount + rootCount
    guard let queue = device.makeCommandQueue() else {
      throw MetalRuntimeANSResidentSource.invalid("Cannot create the spatial query queue.")
    }
    self.queue = queue
    let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalRuntimeANSResidentSource.invalid("Missing spatial kernel \(name).")
      }
      return try device.makeComputePipelineState(function: function)
    }
    leafPipeline = try pipeline("camera_mask_leaves")
    rootPipeline = try pipeline("camera_mask_roots")
    parallelIndex = ProcessInfo.processInfo.environment["QGPU_K3_INDEX_SIMD"] != "0"
    indexPipeline = try pipeline(parallelIndex ? "camera_index_sum_simd" : "camera_index_sum")
    func buffer(_ bytes: Int) throws -> MTLBuffer {
      try MetalRuntimeANSResidentSource.sharedBuffer(
        device: device, bytes: max(4, bytes), label: "camera spatial plan")
    }
    mask = try buffer(validity.count)
    valid = try buffer(validity.count)
    leaves = try buffer(tileCount * 4)
    fields = try buffer(fieldCount * 4)
    fieldCoefficients = try buffer(fieldCount * 4)
    pixels = try buffer(validity.count * 4)
    pixelCoefficients = try buffer(validity.count * 4)
    counts = try buffer(8)
    _ = validity.withUnsafeBytes { memcpy(valid.contents(), $0.baseAddress!, $0.count) }
  }

  func update(
    _ values: [UInt8], source: MetalRuntimeANSResidentSource,
    output: MTLBuffer
  ) throws -> MetalRuntimeANSDetectorMetrics {
    let started = CFAbsoluteTimeGetCurrent()
    _ = values.withUnsafeBytes { memcpy(mask.contents(), $0.baseAddress!, $0.count) }
    memset(counts.contents(), 0, 8)
    var dimensions = [UInt32(shape[0]), UInt32(shape[1])]
    guard let plan = queue.makeCommandBuffer(), let encoder = plan.makeComputeCommandEncoder()
    else {
      throw MetalRuntimeANSResidentSource.invalid("Cannot prepare the camera mask.")
    }
    encoder.setComputePipelineState(leafPipeline)
    for (index, buffer) in [mask, valid, leaves, pixels, pixelCoefficients, counts].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBytes(&dimensions, length: 8, index: 6)
    encoder.dispatchThreadgroups(
      MTLSize(width: tileCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.setComputePipelineState(rootPipeline)
    for (index, buffer) in [leaves, fields, fieldCoefficients, counts].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBytes(&dimensions, length: 8, index: 4)
    encoder.dispatchThreadgroups(
      MTLSize(width: rootCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    plan.commit()
    plan.waitUntilCompleted()
    guard plan.status == .completed else {
      throw MetalRuntimeANSResidentSource.invalid("Camera mask planning failed.")
    }
    let sizes = counts.contents().assumingMemoryBound(to: UInt32.self)
    let fieldCountSelected = Int(sizes[0])
    let residualCount = Int(sizes[1])
    guard fieldCountSelected <= fieldCount, residualCount <= values.count,
      let command = queue.makeCommandBuffer(), let sums = command.makeComputeCommandEncoder(),
      let failure = source.failure
    else { throw MetalRuntimeANSResidentSource.invalid("Invalid camera mask plan.") }
    memset(failure.contents(), 0, 4)
    sums.setComputePipelineState(indexPipeline)
    for chunk in source.chunks {
      var parameters = [
        UInt32(chunk.scanCount), UInt32(fieldCount), UInt32(fieldCountSelected),
        UInt32(chunk.firstScan),
      ]
      for (index, buffer) in (chunk.spatial + [fields, fieldCoefficients, output]).enumerated() {
        sums.setBuffer(buffer, offset: 0, index: index)
      }
      sums.setBytes(&parameters, length: 16, index: 6)
      if parallelIndex {
        sums.dispatchThreadgroups(
          MTLSize(width: chunk.scanCount, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      } else {
        sums.dispatchThreads(
          MTLSize(width: chunk.scanCount, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      }
    }
    sums.endEncoding()
    if residualCount > 0 {
      guard let residual = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
        throw MetalRuntimeANSResidentSource.invalid("Cannot decode camera edge residuals.")
      }
      try source.encodeDetectorDelta(
        selected: pixels, coefficients: pixelCoefficients,
        changed: residualCount, output: output, encoder: residual)
      residual.endEncoding()
    }
    command.commit()
    command.waitUntilCompleted()
    try source.checkFailure(command)
    return MetalRuntimeANSDetectorMetrics(
      changedDetectorPixels: residualCount,
      gpuMilliseconds: (plan.gpuEndTime - plan.gpuStartTime + command.gpuEndTime
        - command.gpuStartTime) * 1000,
      wallMilliseconds: (CFAbsoluteTimeGetCurrent() - started) * 1000,
      acquisitionCount: 1, submissionCount: 2)
  }
}
