import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels

/// Bounded exact scan reduction shared by paired-resident mean and region DPs.
/// Calls are serialized by the owning resident's state lock.
final class PairedRuntimeMeanDiffraction {
  private let partialsPipeline: MTLComputePipelineState
  private let combinePipeline: MTLComputePipelineState

  init(device: MTLDevice, library: MTLLibrary, compactOffsets: Bool) throws {
    let constants = MTLFunctionConstantValues()
    var compact = compactOffsets
    constants.setConstantValue(
      &compact, type: .bool,
      index: Metal4DSTEMKernels.pairedRuntimeTANSCompactOffsetsFunctionConstantIndex)
    partialsPipeline = try device.makeComputePipelineState(
      function: library.makeFunction(
        name: "paired_runtime_tans_region_partials", constantValues: constants))
    combinePipeline = try device.makeComputePipelineState(
      function: library.makeFunction(
        name: "paired_runtime_tans_region_combine", constantValues: constants))
  }

  func mean(
    queue: MTLCommandQueue, payload: MTLBuffer, offsets: MTLBuffer,
    modes: MTLBuffer, table: MTLBuffer, shape: [Int], pixelOfStreamRank: [UInt32]?,
    rows: Range<Int>, columns: Range<Int>, regionShape: MetalScanRegionShape
  ) throws -> MetalCompactH5MeanDiffraction {
    let started = CFAbsoluteTimeGetCurrent()
    guard !rows.isEmpty, !columns.isEmpty,
      rows.lowerBound >= 0, rows.upperBound <= shape[0],
      columns.lowerBound >= 0, columns.upperBound <= shape[1],
      regionShape != .circle || rows.count == columns.count
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Choose a nonempty scan region inside the image; circle bounds must be square.")
    }
    let pixels = shape[2] * shape[3]
    let scans = shape[0] * shape[1]
    let packetScans = PairedRuntimeTANSRecordABI.streamScans
    let batchPackets = 16
    func buffer(_ bytes: Int, _ label: String) throws -> MTLBuffer {
      guard let result = queue.device.makeBuffer(length: bytes, options: .storageModeShared) else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Cannot allocate \(label). Close inactive datasets and retry.")
      }
      result.label = label
      return result
    }
    let membership = try buffer(scans, "Mean diffraction scan membership")
    let selectedCounts = try buffer(scans / packetScans * 4, "Mean diffraction packet membership")
    memset(selectedCounts.contents(), 0, selectedCounts.length)
    let counts = selectedCounts.contents().assumingMemoryBound(to: UInt32.self)
    memset(membership.contents(), 0, membership.length)
    let selected = membership.contents().assumingMemoryBound(to: UInt8.self)
    for row in rows {
      for column in columns {
        let dr = 2 * row + 1 - rows.lowerBound - rows.upperBound
        let dc = 2 * column + 1 - columns.lowerBound - columns.upperBound
        if regionShape == .rectangle || dr * dr + dc * dc <= rows.count * rows.count {
          selected[row * shape[1] + column] = 1
          counts[(row * shape[1] + column) / packetScans] += 1
        }
      }
    }
    let partials = try buffer(batchPackets * pixels * 4, "Mean diffraction packet sums")
    let output = try buffer(pixels * 8, "Mean diffraction exact sums")
    let failure = try buffer(4, "Mean diffraction validation")
    memset(output.contents(), 0, output.length)
    memset(failure.contents(), 0, failure.length)
    guard let command = queue.makeCommandBuffer() else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Cannot create mean diffraction command. Retry.")
    }
    let firstPacket = (rows.lowerBound * shape[1] + columns.lowerBound) / packetScans
    let stopPacket = ((rows.upperBound - 1) * shape[1] + columns.upperBound - 1) / packetScans + 1
    var dispatches = 0
    for first in stride(from: firstPacket, to: stopPacket, by: batchPackets) {
      let count = min(batchPackets, stopPacket - first)
      guard let decode = command.makeComputeCommandEncoder() else {
        throw Metal4DSTEMStreamingIOError.invalidRequest("Cannot encode mean diffraction. Retry.")
      }
      decode.setComputePipelineState(partialsPipeline)
      for (index, source) in [payload, offsets, modes, table, partials, failure].enumerated() {
        decode.setBuffer(source, offset: 0, index: index)
      }
      var parameters = [pixels, scans / packetScans, payload.length, first, count].map(UInt32.init)
      decode.setBytes(&parameters, length: parameters.count * 4, index: 6)
      decode.setBuffer(membership, offset: 0, index: 7)
      decode.setBuffer(selectedCounts, offset: 0, index: 8)
      decode.dispatchThreads(
        MTLSize(width: count * pixels, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      decode.endEncoding()
      guard let combine = command.makeComputeCommandEncoder() else {
        throw Metal4DSTEMStreamingIOError.invalidRequest("Cannot combine mean diffraction. Retry.")
      }
      combine.setComputePipelineState(combinePipeline)
      combine.setBuffer(partials, offset: 0, index: 0)
      combine.setBuffer(output, offset: 0, index: 1)
      var combineParameters = [UInt32(pixels), UInt32(count)]
      combine.setBytes(&combineParameters, length: 8, index: 2)
      combine.dispatchThreads(
        MTLSize(width: pixels, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      combine.endEncoding()
      dispatches += 2
    }
    command.commit()
    command.waitUntilCompleted()
    let code = failure.contents().load(as: UInt32.self)
    guard command.status == .completed, command.error == nil, code == 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Mean diffraction failed (code \(code)): \(command.error?.localizedDescription ?? "invalid encoded stream"). Reload the acquisition."
      )
    }
    let stored = UnsafeBufferPointer(
      start: output.contents().assumingMemoryBound(to: UInt64.self), count: pixels)
    var sums = Array(stored)
    if let pixelOfStreamRank {
      for (rank, pixel) in pixelOfStreamRank.enumerated() { sums[Int(pixel)] = stored[rank] }
    }
    let count = regionShape.sampleCount(rowCount: rows.count, columnCount: columns.count)
    return MetalCompactH5MeanDiffraction(
      detectorSum: sums,
      mean: sums.map { Float(Double($0) / Double(count)) },
      wallMilliseconds: (CFAbsoluteTimeGetCurrent() - started) * 1_000,
      gpuMilliseconds: max(0, command.gpuEndTime - command.gpuStartTime) * 1_000,
      dispatchCount: dispatches, readbackBytes: UInt64(output.length))
  }
}
