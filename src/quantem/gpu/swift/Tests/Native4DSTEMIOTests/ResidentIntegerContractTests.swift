import CryptoKit
import Foundation
import Metal
import Metal4DSTEMKernels
import Metal4DSTEMStreamingIO
import XCTest

private struct ResidentIntegerFixture: Decodable {
  struct DetectorMask: Decodable {
    let id: String
    let maskU8: [UInt8]
    let expectedSumU64: [UInt64]
  }
  struct LargeSum: Decodable {
    struct Override: Decodable {
      let flatIndex: Int
      let value: UInt16
    }
    let shape: [Int]
    let fill: UInt16
    let overrides: [Override]
    let expectedFullSumU64: [UInt64]
  }
  let schema: String
  let shape: [Int]
  let rawFramesU16: [[UInt16]]
  let excludedDetectorFlatIndices: [Int]
  let expectedWorkingFramesU16: [[UInt16]]
  let detectorMasks: [DetectorMask]
  let requestOrder: [String]
  let selectedScanRowColumns: [[Int]]
  let expectedSelectedFramesU16: [[UInt16]]
  let invalidSelectedScanRowColumns: [[Int]]
  let largeSum: LargeSum
}

final class ResidentIntegerContractTests: XCTestCase {
  // No duplicate-counting selected-frame sum endpoint exists on this owner.
  // Radius conveniences are not compared: callers supply the exact mask bytes.
  // This exercises private resident shards, not loading/decompression or phone UI.
  func testPrivateMetalMasksAndSelectedDiffractionMatchFrozenUInt16() throws {
    let fixture = try loadFixture()
    var working = fixture.rawFramesU16
    for scan in working.indices {
      for pixel in fixture.excludedDetectorFlatIndices { working[scan][pixel] = 0 }
    }
    XCTAssertEqual(working, fixture.expectedWorkingFramesU16)
    let device = try metalDevice()
    let (plan, shards) = try privateShards(
      device: device, shape: fixture.shape, values: working.flatMap { $0 }
    )
    XCTAssertEqual(shards.count, 2, "The selected DPs must cross a shard boundary.")
    let interactions = try Metal4DSTEMShardedInteractions(device: device)
    let virtual = try output(device, count: working.count)
    let selected = try output(device, count: working[0].count)
    var previous: [UInt8]? = nil
    let masks = Dictionary(uniqueKeysWithValues: fixture.detectorMasks.map { ($0.id, $0) })

    for name in fixture.requestOrder {
      let request = try XCTUnwrap(masks[name])
      try interactions.updateVirtualDetector(
        shards: shards, plan: plan, previousMask: previous,
        nextMask: request.maskU8, into: virtual
      )
      previous = request.maskU8
      XCTAssertEqual(
        read(virtual, count: working.count).map(UInt64.init),
        request.expectedSumU64, name)
      for (index, coordinate) in fixture.selectedScanRowColumns.enumerated() {
        try interactions.extractDiffraction(
          shards: shards, plan: plan, scanRow: coordinate[0],
          scanColumn: coordinate[1], into: selected
        )
        XCTAssertEqual(
          read(selected, count: working[0].count),
          fixture.expectedSelectedFramesU16[index].map(UInt32.init), name)
        // This caller-owned small output must not alias the private source.
        memset(selected.contents(), 0, selected.length)
      }
    }

    let full = try XCTUnwrap(masks["full"])
    func recover() throws {
      try interactions.updateVirtualDetector(
        shards: shards, plan: plan, previousMask: nil,
        nextMask: full.maskU8, into: virtual
      )
      XCTAssertEqual(
        read(virtual, count: working.count).map(UInt64.init),
        full.expectedSumU64)
    }
    for coordinate in fixture.invalidSelectedScanRowColumns {
      XCTAssertThrowsError(
        try interactions.extractDiffraction(
          shards: shards, plan: plan, scanRow: coordinate[0],
          scanColumn: coordinate[1], into: selected
        ))
      try recover()
    }
    XCTAssertThrowsError(
      try interactions.updateVirtualDetector(
        shards: shards, plan: plan, previousMask: nil, nextMask: [1], into: virtual
      ))
    try recover()
    XCTAssertThrowsError(
      try interactions.extractDiffraction(
        shards: shards, plan: plan, scanRow: 0, scanColumn: 0,
        into: selected, shouldCancel: { true }
      ))
    try recover()
    XCTAssertThrowsError(
      try interactions.updateVirtualDetector(
        shards: shards, plan: plan, previousMask: nil,
        nextMask: full.maskU8, into: virtual, shouldCancel: { true }
      ))
    try recover()
    // All original frames remain available after failures and output edits.
    for scan in working.indices {
      try interactions.extractDiffraction(
        shards: shards, plan: plan, scanRow: scan / fixture.shape[1],
        scanColumn: scan % fixture.shape[1], into: selected
      )
      XCTAssertEqual(read(selected, count: working[0].count), working[scan].map(UInt32.init))
    }
  }

  func testPrivateMetalFullUInt16SumAboveFloat32Range() throws {
    let fixture = try loadFixture().largeSum
    var values = [UInt16](repeating: fixture.fill, count: fixture.shape.reduce(1, *))
    for override in fixture.overrides { values[override.flatIndex] = override.value }
    let device = try metalDevice()
    let (plan, shards) = try privateShards(device: device, shape: fixture.shape, values: values)
    let interactions = try Metal4DSTEMShardedInteractions(device: device)
    let scanCount = fixture.shape[0] * fixture.shape[1]
    let virtual = try output(device, count: scanCount)
    try interactions.updateVirtualDetector(
      shards: shards, plan: plan, previousMask: nil,
      nextMask: [UInt8](repeating: 1, count: fixture.shape[2] * fixture.shape[3]),
      into: virtual
    )
    let actual = read(virtual, count: scanCount).map(UInt64.init)
    XCTAssertEqual(actual, fixture.expectedFullSumU64)
    XCTAssertNotEqual(actual[1], UInt64(Float(actual[1])))
  }

  func testEmptyRebaseAndInvalidMasksPreserveExactRecovery() throws {
    let fixture = try loadFixture()
    var working = fixture.rawFramesU16
    for scan in working.indices {
      for pixel in fixture.excludedDetectorFlatIndices { working[scan][pixel] = 0 }
    }
    let device = try metalDevice()
    let (plan, shards) = try privateShards(
      device: device, shape: fixture.shape,
      values: working.flatMap { $0 }
    )
    let interactions = try Metal4DSTEMShardedInteractions(device: device)
    let scanCount = fixture.shape[0] * fixture.shape[1]
    let virtual = try output(device, count: scanCount)
    let masks = Dictionary(uniqueKeysWithValues: fixture.detectorMasks.map { ($0.id, $0) })
    let empty = try XCTUnwrap(masks["empty"])
    let full = try XCTUnwrap(masks["full"])
    memset(virtual.contents(), 0xff, virtual.length)
    XCTAssertThrowsError(
      try interactions.updateVirtualDetector(
        shards: shards, plan: plan, previousMask: nil, nextMask: empty.maskU8,
        into: virtual, shouldCancel: { true }
      ))
    XCTAssertEqual(read(virtual, count: scanCount), [UInt32](repeating: .max, count: scanCount))
    try interactions.updateVirtualDetector(
      shards: shards, plan: plan, previousMask: nil, nextMask: empty.maskU8, into: virtual
    )
    XCTAssertEqual(read(virtual, count: scanCount).map(UInt64.init), empty.expectedSumU64)
    let unchangedMilliseconds = try interactions.updateVirtualDetector(
      shards: shards, plan: plan, previousMask: empty.maskU8, nextMask: empty.maskU8, into: virtual
    )
    XCTAssertEqual(unchangedMilliseconds, 0)
    XCTAssertEqual(read(virtual, count: scanCount).map(UInt64.init), empty.expectedSumU64)

    var malformed = full.maskU8
    malformed[1] = 2
    for invalidPrevious in [false, true] {
      try interactions.updateVirtualDetector(
        shards: shards, plan: plan, previousMask: nil, nextMask: full.maskU8, into: virtual
      )
      XCTAssertThrowsError(
        try interactions.updateVirtualDetector(
          shards: shards, plan: plan,
          previousMask: invalidPrevious ? malformed : full.maskU8,
          nextMask: invalidPrevious ? full.maskU8 : malformed, into: virtual
        ))
      XCTAssertEqual(read(virtual, count: scanCount).map(UInt64.init), full.expectedSumU64)
      try interactions.updateVirtualDetector(
        shards: shards, plan: plan, previousMask: nil, nextMask: empty.maskU8, into: virtual
      )
      XCTAssertEqual(read(virtual, count: scanCount).map(UInt64.init), empty.expectedSumU64)
      try interactions.updateVirtualDetector(
        shards: shards, plan: plan, previousMask: empty.maskU8, nextMask: full.maskU8, into: virtual
      )
      XCTAssertEqual(read(virtual, count: scanCount).map(UInt64.init), full.expectedSumU64)
    }
  }

  private func loadFixture() throws -> ResidentIntegerFixture {
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<7 { root.deleteLastPathComponent() }
    let data = try Data(
      contentsOf: root.appendingPathComponent(
        "tests/parity/fixtures/resident_integer_products_v1.json"
      ))
    XCTAssertEqual(
      SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
      "bc6de8428ffaf53d5e996470c3947328afd072cb72a4416325ce61819a600ea2")
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let fixture = try decoder.decode(ResidentIntegerFixture.self, from: data)
    XCTAssertEqual(fixture.schema, "quantem.gpu.resident-integer-products/v1")
    return fixture
  }

  private func privateShards(
    device: MTLDevice, shape: [Int], values: [UInt16]
  ) throws -> (Metal4DSTEMExactBinningShardPlan, [MTLBuffer]) {
    let load = try Metal4DSTEMLoadPlan(
      sourceScanRows: shape[0], sourceScanColumns: shape[1],
      detectorRows: shape[2], detectorColumns: shape[3], sourceBytesPerValue: 2,
      scanRegion: .full(sourceRows: shape[0], sourceColumns: shape[1])
    )
    let identity = values.withUnsafeBytes { SHA256.hash(data: Data($0)) }
      .map { String(format: "%02x", $0) }.joined()
    let audit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: identity, sourceDtype: .uint16, badPixelIndices: [],
      maximumSourceCount: UInt32(values.max() ?? 0),
      pixelsAbove255: UInt64(values.filter { $0 > 255 }.count)
    )
    let provenance = try Metal4DSTEMExactBinner.provenance(
      plan: load, sourceAudit: audit, stagingDtype: .uint16, outputDtype: .uint16
    )
    let detectorPixels = shape[2] * shape[3]
    let detectorWords = (detectorPixels + 1) / 2
    let plan = try Metal4DSTEMExactBinningShardPlan(
      provenance: provenance, maximumShardBytes: UInt64(detectorWords * shape[1] * 4)
    )
    let queue = try XCTUnwrap(device.makeCommandQueue())
    var shards: [MTLBuffer] = []
    for shard in plan.shards {
      let scans = shard.outputScanPositionCount
      var words = [UInt32](repeating: 0, count: detectorWords * scans)
      for scan in 0..<scans {
        for pixel in 0..<detectorPixels {
          let value = values[(shard.outputScanPositionStart + scan) * detectorPixels + pixel]
          words[(pixel / 2) * scans + scan] |= UInt32(value) << UInt32((pixel % 2) * 16)
        }
      }
      let staging = try XCTUnwrap(
        device.makeBuffer(
          bytes: words, length: words.count * 4, options: .storageModeShared
        ))
      let resident = try XCTUnwrap(
        device.makeBuffer(length: staging.length, options: .storageModePrivate))
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
      blit.copy(
        from: staging, sourceOffset: 0, to: resident, destinationOffset: 0, size: staging.length)
      blit.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      XCTAssertEqual(
        command.status, .completed, command.error?.localizedDescription ?? "upload failed")
      XCTAssertEqual(resident.storageMode, .private)
      shards.append(resident)
    }
    return (plan, shards)
  }

  private func metalDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal device unavailable.")
    }
    return device
  }

  private func output(_ device: MTLDevice, count: Int) throws -> MTLBuffer {
    try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModeShared))
  }

  private func read(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
    Array(
      UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: count))
  }
}
