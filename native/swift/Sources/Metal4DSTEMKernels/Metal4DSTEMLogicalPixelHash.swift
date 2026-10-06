import CryptoKit
import Foundation
import Metal

/// Errors produced while hashing an exact working volume in logical pixel order.
public enum Metal4DSTEMLogicalPixelHashError: Error, Equatable, Sendable {
  case invalidPlan
  case invalidStagingByteLimit
  case bufferCountMismatch(expected: Int, actual: Int)
  case bufferTooSmall(index: Int, expected: UInt64, actual: Int)
  case unsupportedLayout(Metal4DSTEMOutputLayout)
  case metalUnavailable(String)
}

/// Backend-neutral full-volume parity for exact 4D-STEM integer counts.
///
/// The digest contains payload bytes only, in logical
/// `[scan_row, scan_column, detector_row, detector_column]` C order with
/// `detector_column` varying fastest. Samples use their declared integer width
/// in little-endian byte order. Shape, dtype, and this schema must be recorded
/// and validated separately; shard boundaries and packed-word padding are not
/// part of the digest.
public enum Metal4DSTEMLogicalPixelHash {
  public static let schema = "quantem.gpu.4dstem-logical-pixels/v1"
  public static let defaultMaximumStagingBytes = 16 * 1024 * 1024

  /// Hash detector-word-major Metal shards without materializing another volume.
  ///
  /// A bounded shared staging buffer converts one or more scan positions at a
  /// time. The source shards may use shared or private Metal storage.
  public static func sha256(
    buffers: [MTLBuffer],
    shardPlan: Metal4DSTEMExactBinningShardPlan,
    device: MTLDevice,
    maximumStagingBytes: Int = defaultMaximumStagingBytes
  ) throws -> String {
    let provenance = shardPlan.provenance
    guard provenance.outputScanRows > 0,
      provenance.outputScanColumns > 0,
      provenance.outputDetectorRows > 0,
      provenance.outputDetectorColumns > 0
    else { throw Metal4DSTEMLogicalPixelHashError.invalidPlan }
    try shardPlan.validate(provenance: provenance)
    guard buffers.count == shardPlan.shards.count else {
      throw Metal4DSTEMLogicalPixelHashError.bufferCountMismatch(
        expected: shardPlan.shards.count,
        actual: buffers.count
      )
    }

    let detectorPixelsResult = provenance.outputDetectorRows
      .multipliedReportingOverflow(by: provenance.outputDetectorColumns)
    guard !detectorPixelsResult.overflow else {
      throw Metal4DSTEMLogicalPixelHashError.invalidPlan
    }
    let detectorPixels = detectorPixelsResult.partialValue
    let bytesPerSample: Int
    let functionName: String
    switch provenance.outputLayout {
    case .detectorWordMajorPackedUInt16:
      guard provenance.outputDtype == .uint16 else {
        throw Metal4DSTEMLogicalPixelHashError.invalidPlan
      }
      bytesPerSample = MemoryLayout<UInt16>.stride
      functionName = Metal4DSTEMKernels.extractU16FramesFunction
    case .detectorWordMajorUInt32:
      guard provenance.outputDtype == .uint32 else {
        throw Metal4DSTEMLogicalPixelHashError.invalidPlan
      }
      bytesPerSample = MemoryLayout<UInt32>.stride
      functionName = Metal4DSTEMKernels.extractU32FramesFunction
    }
    let frameBytesResult = detectorPixels.multipliedReportingOverflow(
      by: bytesPerSample
    )
    guard !frameBytesResult.overflow else {
      throw Metal4DSTEMLogicalPixelHashError.invalidPlan
    }
    let frameBytes = frameBytesResult.partialValue
    guard maximumStagingBytes >= frameBytes else {
      throw Metal4DSTEMLogicalPixelHashError.invalidStagingByteLimit
    }
    let batchCapacity = max(1, maximumStagingBytes / frameBytes)
    let stagingBytesResult = frameBytes.multipliedReportingOverflow(
      by: batchCapacity
    )
    guard !stagingBytesResult.overflow,
      let staging = device.makeBuffer(
        length: stagingBytesResult.partialValue,
        options: .storageModeShared
      ),
      let queue = device.makeCommandQueue()
    else {
      throw Metal4DSTEMLogicalPixelHashError.metalUnavailable(
        "Could not allocate canonical-hash staging resources."
      )
    }
    let library: MTLLibrary
    let pipeline: MTLComputePipelineState
    do {
      library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
      guard let function = library.makeFunction(name: functionName) else {
        throw Metal4DSTEMLogicalPixelHashError.metalUnavailable(
          "Missing canonical-hash extraction function \(functionName)."
        )
      }
      pipeline = try device.makeComputePipelineState(function: function)
    } catch let error as Metal4DSTEMLogicalPixelHashError {
      throw error
    } catch {
      throw Metal4DSTEMLogicalPixelHashError.metalUnavailable(
        "Could not compile canonical-hash extraction: \(error.localizedDescription)"
      )
    }

    var hasher = SHA256()
    for (index, pair) in zip(shardPlan.shards, buffers).enumerated() {
      let (shard, buffer) = pair
      guard UInt64(buffer.length) >= shard.payloadBytes else {
        throw Metal4DSTEMLogicalPixelHashError.bufferTooSmall(
          index: index,
          expected: shard.payloadBytes,
          actual: buffer.length
        )
      }
      guard shard.outputLayout == provenance.outputLayout,
        let scanCount = UInt32(exactly: shard.outputScanPositionCount),
        let pixelCount = UInt32(exactly: detectorPixels)
      else { throw Metal4DSTEMLogicalPixelHashError.invalidPlan }

      var scanStartValue = 0
      while scanStartValue < shard.outputScanPositionCount {
        let batchCount = min(
          batchCapacity,
          shard.outputScanPositionCount - scanStartValue
        )
        guard let scanStart = UInt32(exactly: scanStartValue),
          let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder()
        else {
          throw Metal4DSTEMLogicalPixelHashError.metalUnavailable(
            "Could not encode canonical-hash extraction."
          )
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(staging, offset: 0, index: 1)
        var mutableScanStart = scanStart
        var mutableScanCount = scanCount
        var mutablePixelCount = pixelCount
        encoder.setBytes(
          &mutableScanStart,
          length: MemoryLayout<UInt32>.stride,
          index: 2
        )
        encoder.setBytes(
          &mutableScanCount,
          length: MemoryLayout<UInt32>.stride,
          index: 3
        )
        encoder.setBytes(
          &mutablePixelCount,
          length: MemoryLayout<UInt32>.stride,
          index: 4
        )
        encoder.dispatchThreads(
          MTLSize(width: detectorPixels, height: batchCount, depth: 1),
          threadsPerThreadgroup: MTLSize(
            width: min(pipeline.threadExecutionWidth, detectorPixels),
            height: 1,
            depth: 1
          )
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
          throw Metal4DSTEMLogicalPixelHashError.metalUnavailable(
            "Canonical-hash extraction failed: "
              + (command.error?.localizedDescription ?? "unknown Metal error")
          )
        }
        let usedBytes = frameBytes * batchCount
        hasher.update(
          bufferPointer: UnsafeRawBufferPointer(
            start: staging.contents(),
            count: usedBytes
          )
        )
        scanStartValue += batchCount
      }
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
