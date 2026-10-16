import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels

/// Bounded detector-major extraction; the resident owns serialization.
final class PairedRuntimeDetectorColumns {
  private let pipeline: MTLComputePipelineState

  init(device: MTLDevice, library: MTLLibrary, compactOffsets: Bool) throws {
    let constants = MTLFunctionConstantValues()
    var compact = compactOffsets
    constants.setConstantValue(
      &compact, type: .bool,
      index: Metal4DSTEMKernels.pairedRuntimeTANSCompactOffsetsFunctionConstantIndex)
    pipeline = try device.makeComputePipelineState(function: library.makeFunction(
      name: "paired_runtime_tans_detector_columns", constantValues: constants))
  }

  func read(
    queue: MTLCommandQueue, payload: MTLBuffer, offsets: MTLBuffer,
    modes: MTLBuffer, table: MTLBuffer, detectorPixels: Int, scanCount: Int,
    selectedRanks: [UInt32], into output: MTLBuffer
  ) throws {
    guard !selectedRanks.isEmpty, selectedRanks.count <= 32,
      detectorPixels > 0, scanCount > 0, scanCount % 512 == 0,
      selectedRanks.allSatisfy({ UInt64($0) < UInt64(detectorPixels) }),
      output.length / 4 / scanCount >= selectedRanks.count,
      output.device.registryID == queue.device.registryID
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Detector columns need 1...32 valid pixels and a same-device UInt32 buffer for every scan position.")
    }
    guard let failure = queue.device.makeBuffer(length: 4, options: .storageModeShared),
      let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Cannot allocate detector-column extraction. Close inactive datasets and retry.")
    }
    failure.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    encoder.setComputePipelineState(pipeline)
    for (index, buffer) in [payload, offsets, modes, table, output, failure].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    var parameters = [detectorPixels, scanCount / 512, payload.length, selectedRanks.count]
      .map(UInt32.init)
    var selected = selectedRanks
    encoder.setBytes(&parameters, length: parameters.count * 4, index: 6)
    encoder.setBytes(&selected, length: selected.count * 4, index: 7)
    encoder.dispatchThreads(
      MTLSize(width: selected.count * scanCount / 512, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    let status = failure.contents().load(as: UInt32.self)
    guard command.status == .completed, command.error == nil, status == 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Detector-column decoding failed (code \(status)): \(command.error?.localizedDescription ?? "invalid encoded stream"). Reopen the acquisition.")
    }
  }
}
