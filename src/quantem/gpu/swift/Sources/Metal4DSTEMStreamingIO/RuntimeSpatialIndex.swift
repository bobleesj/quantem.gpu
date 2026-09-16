import Foundation
import Metal
import Metal4DSTEMKernels

/// Build the CUDA-compatible compact spatial fields while each native chunk is available.
final class RuntimeSpatialIndex {
  let queue: MTLCommandQueue
  let device: MTLDevice
  let shape: [Int]
  let valid: MTLBuffer
  let fields: Int
  let pipelines: [MTLComputePipelineState]
  init(device: MTLDevice, shape: [Int], validity: [UInt8]) throws {
    self.device = device; self.shape = shape
    fields = ((shape[0] + 7) / 8) * ((shape[1] + 7) / 8) + ((shape[0] + 31) / 32) * ((shape[1] + 31) / 32)
    guard let queue = device.makeCommandQueue() else { throw MetalRuntimeANSResidentSource.invalid("Cannot build spatial indexes.") }
    self.queue = queue
    valid = try MetalRuntimeANSResidentSource.sharedBuffer(device: device, bytes: validity.count, label: "camera validity")
    let library = try Metal4DSTEMKernels.makeRuntimeANSLibrary(device: device)
    pipelines = try ["camera_fields", "camera_field_widths", "camera_pack_fields"].map {
      guard let function = library.makeFunction(name: $0) else { throw MetalRuntimeANSResidentSource.invalid("Missing camera index kernel.") }
      return try device.makeComputePipelineState(function: function)
    }
    _ = validity.withUnsafeBytes { memcpy(valid.contents(), $0.baseAddress!, $0.count) }
  }
  func build(raw: MTLBuffer, scans: Int, itemBytes: Int) throws -> [MTLBuffer] {
    let streams = ((scans + 511) / 512) * fields
    func buffer(_ bytes: Int) throws -> MTLBuffer {
      try MetalRuntimeANSResidentSource.sharedBuffer(device: device, bytes: max(4, bytes), label: "camera spatial index")
    }
    let values = try buffer(scans * fields * 4), widths = try buffer(streams)
    let lengths = try buffer(streams * 4), starts = try buffer((streams + 1) * 8)
    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
      throw MetalRuntimeANSResidentSource.invalid("Cannot build spatial index widths.")
    }
    var shape = [UInt32(scans), UInt32(self.shape[0]), UInt32(self.shape[1]), UInt32(itemBytes)]
    var layout = [UInt32(scans), UInt32(fields)]
    encoder.setComputePipelineState(pipelines[0])
    for (i, b) in [raw, valid, values].enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
    encoder.setBytes(&shape, length: 16, index: 3)
    encoder.dispatchThreadgroups(MTLSize(width: scans * fields, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.setComputePipelineState(pipelines[1])
    for (i, b) in [values, widths, lengths].enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
    encoder.setBytes(&layout, length: 8, index: 3)
    encoder.dispatchThreads(MTLSize(width: streams, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
    guard command.status == .completed else { throw MetalRuntimeANSResidentSource.invalid("Camera index reduction failed.") }
    let sizes = lengths.contents().assumingMemoryBound(to: UInt32.self)
    let offsets = starts.contents().assumingMemoryBound(to: UInt64.self)
    offsets[0] = 0
    for i in 0..<streams { offsets[i + 1] = offsets[i] + UInt64(sizes[i]) }
    let words = try buffer(Int(offsets[streams]) * 4)
    guard let pack = queue.makeCommandBuffer(), let encoding = pack.makeComputeCommandEncoder() else {
      throw MetalRuntimeANSResidentSource.invalid("Cannot pack camera indexes.")
    }
    encoding.setComputePipelineState(pipelines[2])
    for (i, b) in [values, widths, starts, words].enumerated() { encoding.setBuffer(b, offset: 0, index: i) }
    encoding.setBytes(&layout, length: 8, index: 4)
    encoding.dispatchThreads(MTLSize(width: streams, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoding.endEncoding(); pack.commit(); pack.waitUntilCompleted()
    guard pack.status == .completed else { throw MetalRuntimeANSResidentSource.invalid("Camera index packing failed.") }
    return [words, starts, widths]
  }
}
