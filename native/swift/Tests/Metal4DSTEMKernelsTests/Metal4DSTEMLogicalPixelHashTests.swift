import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMKernels

final class Metal4DSTEMLogicalPixelHashTests: XCTestCase {
  func testPackedUInt16HashMatchesCanonicalFrameMajorBytesAcrossShards() throws {
    let device = try metalDevice()
    let shardPlan = try makeUInt16ShardPlan()
    let values = (0..<90).map(UInt16.init)
    let shared = try makePackedUInt16Buffers(
      device: device,
      shardPlan: shardPlan,
      values: values,
      storageMode: .shared,
      paddingLane: 0xBEEF
    )
    let privateBuffers = try makePackedUInt16Buffers(
      device: device,
      shardPlan: shardPlan,
      values: values,
      storageMode: .private,
      paddingLane: 0x1234
    )

    let sharedHash = try Metal4DSTEMLogicalPixelHash.sha256(
      buffers: shared,
      shardPlan: shardPlan,
      device: device,
      maximumStagingBytes: 60
    )
    let privateHash = try Metal4DSTEMLogicalPixelHash.sha256(
      buffers: privateBuffers,
      shardPlan: shardPlan,
      device: device,
      maximumStagingBytes: 60
    )

    XCTAssertEqual(Metal4DSTEMLogicalPixelHash.schema, "quantem.gpu.4dstem-logical-pixels/v1")
    XCTAssertEqual(
      sharedHash,
      "cae677456d8bcae7bd20864213c1c380b694f075e1980e2645a707db8301b977"
    )
    XCTAssertEqual(privateHash, sharedHash)
    XCTAssertEqual(sharedHash, sha256LittleEndian(values))
  }

  func testUInt32HashPreservesFullWidthValues() throws {
    let device = try metalDevice()
    let shardPlan = try makeUInt32ShardPlan()
    let values: [UInt32] = [
      0, 255, 256, 65_535, 65_536, 80_000,
      100_000, 1, 4_000_000, UInt32.max, 17, 33,
    ]
    let buffers = try makeUInt32Buffers(
      device: device,
      shardPlan: shardPlan,
      values: values
    )

    XCTAssertEqual(
      try Metal4DSTEMLogicalPixelHash.sha256(
        buffers: buffers,
        shardPlan: shardPlan,
        device: device,
        maximumStagingBytes: 24
      ),
      sha256LittleEndian(values)
    )
  }

  func testHashFailsClosedOnMissingShardAndUndersizedStaging() throws {
    let device = try metalDevice()
    let shardPlan = try makeUInt16ShardPlan()
    let values = (0..<90).map(UInt16.init)
    let buffers = try makePackedUInt16Buffers(
      device: device,
      shardPlan: shardPlan,
      values: values,
      storageMode: .shared,
      paddingLane: 0
    )

    XCTAssertThrowsError(
      try Metal4DSTEMLogicalPixelHash.sha256(
        buffers: Array(buffers.dropLast()),
        shardPlan: shardPlan,
        device: device
      )
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMLogicalPixelHashError,
        .bufferCountMismatch(expected: 2, actual: 1)
      )
    }
    XCTAssertThrowsError(
      try Metal4DSTEMLogicalPixelHash.sha256(
        buffers: buffers,
        shardPlan: shardPlan,
        device: device,
        maximumStagingBytes: 29
      )
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMLogicalPixelHashError,
        .invalidStagingByteLimit
      )
    }
  }

  private func makeUInt16ShardPlan() throws -> Metal4DSTEMExactBinningShardPlan {
    let loadPlan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 2,
      sourceScanColumns: 3,
      detectorRows: 3,
      detectorColumns: 5,
      sourceBytesPerValue: 2,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 2, sourceColumns: 3),
      detectorBin: 1
    )
    let audit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "a", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: 65_535,
      pixelsAbove255: 10
    )
    let provenance = try Metal4DSTEMExactBinner.provenance(
      plan: loadPlan,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )
    return try Metal4DSTEMExactBinningShardPlan(
      provenance: provenance,
      maximumShardBytes: 96
    )
  }

  private func makeUInt32ShardPlan() throws -> Metal4DSTEMExactBinningShardPlan {
    let loadPlan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 1,
      sourceScanColumns: 2,
      detectorRows: 4,
      detectorColumns: 6,
      sourceBytesPerValue: 2,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 1, sourceColumns: 2),
      detectorBin: 2
    )
    let audit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "b", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: 20_000,
      pixelsAbove255: 12
    )
    let provenance = try Metal4DSTEMExactBinner.provenance(
      plan: loadPlan,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint32
    )
    return try Metal4DSTEMExactBinningShardPlan(
      provenance: provenance,
      maximumShardBytes: 48
    )
  }

  private func makePackedUInt16Buffers(
    device: MTLDevice,
    shardPlan: Metal4DSTEMExactBinningShardPlan,
    values: [UInt16],
    storageMode: MTLStorageMode,
    paddingLane: UInt16
  ) throws -> [MTLBuffer] {
    let detectorPixels =
      shardPlan.provenance.outputDetectorRows
      * shardPlan.provenance.outputDetectorColumns
    let detectorWords = (detectorPixels + 1) / 2
    var buffers: [MTLBuffer] = []
    for shard in shardPlan.shards {
      let scanCount = shard.outputScanPositionCount
      var words = [UInt32](repeating: 0, count: detectorWords * scanCount)
      for localScan in 0..<scanCount {
        let globalScan = shard.outputScanPositionStart + localScan
        for pixel in 0..<detectorPixels {
          let value = UInt32(values[globalScan * detectorPixels + pixel])
          words[(pixel / 2) * scanCount + localScan] |=
            value << UInt32((pixel % 2) * 16)
        }
        if detectorPixels % 2 == 1 {
          words[(detectorWords - 1) * scanCount + localScan] |=
            UInt32(paddingLane) << 16
        }
      }
      buffers.append(
        try makeBuffer(device: device, words: words, storageMode: storageMode)
      )
    }
    return buffers
  }

  private func makeUInt32Buffers(
    device: MTLDevice,
    shardPlan: Metal4DSTEMExactBinningShardPlan,
    values: [UInt32]
  ) throws -> [MTLBuffer] {
    let detectorPixels =
      shardPlan.provenance.outputDetectorRows
      * shardPlan.provenance.outputDetectorColumns
    return try shardPlan.shards.map { shard in
      let scanCount = shard.outputScanPositionCount
      var words = [UInt32](repeating: 0, count: detectorPixels * scanCount)
      for localScan in 0..<scanCount {
        let globalScan = shard.outputScanPositionStart + localScan
        for pixel in 0..<detectorPixels {
          words[pixel * scanCount + localScan] =
            values[globalScan * detectorPixels + pixel]
        }
      }
      return try makeBuffer(device: device, words: words, storageMode: .shared)
    }
  }

  private func makeBuffer(
    device: MTLDevice,
    words: [UInt32],
    storageMode: MTLStorageMode
  ) throws -> MTLBuffer {
    let shared = try words.withUnsafeBytes { bytes in
      try XCTUnwrap(
        device.makeBuffer(
          bytes: bytes.baseAddress!,
          length: bytes.count,
          options: .storageModeShared
        )
      )
    }
    guard storageMode == .private else { return shared }
    let destination = try XCTUnwrap(
      device.makeBuffer(length: shared.length, options: .storageModePrivate)
    )
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeBlitCommandEncoder())
    encoder.copy(
      from: shared,
      sourceOffset: 0,
      to: destination,
      destinationOffset: 0,
      size: shared.length
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed)
    return destination
  }

  private func metalDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("No Metal device is available on this host.")
    }
    return device
  }

  private func sha256LittleEndian<T: FixedWidthInteger>(_ values: [T]) -> String {
    let littleEndian = values.map(\.littleEndian)
    let data = littleEndian.withUnsafeBytes { Data($0) }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
}
