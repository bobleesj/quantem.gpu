import CryptoKit
import Foundation
import XCTest

@testable import Native4DSTEMIO

final class NativeLosslessPackV1ProducerTests: XCTestCase {
  func testInspectAndPlanAreShapeGenericAndFailClosed() throws {
    let fixture = try producerFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let producer = NativeLosslessPackV1Producer(cacheDirectory: fixture.cache)
    let inspection = try producer.inspect(input: fixture.source)

    XCTAssertEqual(inspection.sourceShape, [4, 8, 64, 64])
    XCTAssertEqual(inspection.sourceDtype, "uint16")
    XCTAssertEqual(inspection.sourceLogicalBytes, 262_144)
    XCTAssertGreaterThan(inspection.preparedIndexBytes, 0)
    XCTAssertGreaterThan(inspection.maximumCompressedBlockBytes, 0)
    XCTAssertEqual(inspection.badPixelIndices, [196])
    XCTAssertEqual(inspection.detectorMaskIdentityOrigin, "source-little-endian-u32-mask")
    XCTAssertNotEqual(
      inspection.detectorMaskIdentitySHA256,
      inspection.badPixelIdentitySHA256
    )
    XCTAssertEqual(inspection.calibration.scanRowSamplingNanometer, 0.8)
    XCTAssertEqual(inspection.calibration.scanColumnSamplingNanometer, 1.2)
    XCTAssertNotNil(inspection.calibration.detectorRowSampling)
    XCTAssertNotNil(inspection.calibration.detectorColumnSampling)

    let plan = try producer.plan(
      inspection: inspection,
      destination: fixture.output,
      maximumTransientBytes: 8 * 1024 * 1024,
      availableOutputDiskBytes: 64 * 1024 * 1024
    )
    XCTAssertEqual(plan.executionBackend, .cpuReference)
    XCTAssertEqual(plan.encodingProfile, .exactUInt8Bitpacked)
    XCTAssertEqual(plan.scanTile, 32)
    XCTAssertEqual(plan.scansPerShard, 32)
    XCTAssertEqual(plan.shardCount, 1)
    XCTAssertLessThan(plan.predictedPeakTransientBytes, 8 * 1024 * 1024)
    XCTAssertLessThan(plan.predictedOutputFileMaximumBytes, 64 * 1024 * 1024)
    XCTAssertLessThan(plan.predictedReceiptFileMaximumBytes, 64 * 1024 * 1024)
    XCTAssertEqual(
      plan.predictedOutputDiskMaximumBytes,
      plan.predictedOutputFileMaximumBytes + plan.predictedReceiptFileMaximumBytes
    )

    XCTAssertThrowsError(
      try producer.plan(
        inspection: inspection,
        destination: fixture.output,
        maximumTransientBytes: 1,
        availableOutputDiskBytes: 64 * 1024 * 1024
      )
    ) { error in
      guard case NativeLosslessPackV1ProducerError.insufficientMemory = error else {
        return XCTFail("Expected a memory preflight failure, received \(error)")
      }
    }
    XCTAssertThrowsError(
      try producer.plan(
        inspection: inspection,
        destination: fixture.output,
        maximumTransientBytes: 8 * 1024 * 1024,
        availableOutputDiskBytes: 1
      )
    ) { error in
      guard case NativeLosslessPackV1ProducerError.insufficientDisk = error else {
        return XCTFail("Expected an aggregate output-disk failure, received \(error)")
      }
    }
    XCTAssertThrowsError(
      try producer.plan(
        inspection: inspection,
        destination: fixture.output,
        maximumTransientBytes: 8 * 1024 * 1024,
        availableOutputDiskBytes: 64 * 1024 * 1024,
        executionBackend: .metal
      )
    ) { error in
      XCTAssertEqual(
        error as? NativeLosslessPackV1ProducerError,
        .unavailableExecutionBackend(.metal)
      )
    }

    let nestedDestination = fixture.root
      .appendingPathComponent("new", isDirectory: true)
      .appendingPathComponent("cache", isDirectory: true)
      .appendingPathComponent("planned.h5")
    XCTAssertNoThrow(
      try producer.plan(
        inspection: inspection,
        destination: nestedDestination,
        maximumTransientBytes: 8 * 1024 * 1024
      )
    )
    XCTAssertFalse(FileManager.default.fileExists(atPath: nestedDestination.path))
  }

  func testPlans512And1024ScanSourcesWithoutAllocatingTheirLogicalTensors() throws {
    let fixture = try producerFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let producer = NativeLosslessPackV1Producer(cacheDirectory: fixture.cache)
    let inspected = try producer.inspect(input: fixture.source)

    for (scanSide, expectedShards) in [(512, 64), (1024, 256)] {
      let inspection = scaledInspection(inspected, scanSide: scanSide)
      let plan = try producer.plan(
        inspection: inspection,
        destination: fixture.root.appendingPathComponent("planned-\(scanSide).h5"),
        maximumTransientBytes: 256 * 1024 * 1024,
        availableOutputDiskBytes: UInt64.max
      )
      XCTAssertEqual(plan.scansPerShard, 4096)
      XCTAssertEqual(plan.shardCount, expectedShards)
      XCTAssertLessThan(plan.predictedPeakTransientBytes, 256 * 1024 * 1024)
      XCTAssertLessThan(plan.predictedOutputDiskMaximumBytes, UInt64.max)
    }
  }

  func testOriginalHDF5ProducesExactLosslessPackV1() throws {
    let fixture = try producerFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let producer = NativeLosslessPackV1Producer(cacheDirectory: fixture.cache)
    let inspection = try producer.inspect(input: fixture.source)
    let plan = try producer.plan(
      inspection: inspection,
      destination: fixture.output,
      maximumTransientBytes: 8 * 1024 * 1024,
      availableOutputDiskBytes: 64 * 1024 * 1024
    )
    let receipt = try producer.produce(plan)

    XCTAssertEqual(receipt.schema, NativeLosslessPackV1ProductionReceipt.currentSchema)
    XCTAssertEqual(receipt.formatSchema, "quantem.gpu.lossless-pack-format/v1")
    XCTAssertEqual(receipt.encodingProfile, .exactUInt8Bitpacked)
    XCTAssertEqual(receipt.executionBackend, .cpuReference)
    XCTAssertFalse(receipt.gpuAccelerated)
    XCTAssertEqual(receipt.sourceShape, [4, 8, 64, 64])
    XCTAssertEqual(receipt.workingDtype, "uint8")
    XCTAssertEqual(receipt.badPixelIndices, [196])
    XCTAssertEqual(
      receipt.detectorMaskIdentitySHA256,
      inspection.detectorMaskIdentitySHA256
    )
    XCTAssertEqual(receipt.badPixelRawValues, [65_535])
    XCTAssertEqual(receipt.shards.count, 1)
    XCTAssertEqual(receipt.shards[0].maximumBitWidth, 8)
    XCTAssertLessThanOrEqual(
      receipt.observedPackedResidentBytes,
      receipt.predictedPackedResidentMaximumBytes
    )
    XCTAssertLessThanOrEqual(
      receipt.accountedPeakTransientBytes,
      receipt.predictedPeakTransientBytes
    )
    XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.output.path))
    XCTAssertTrue(
      FileManager.default.fileExists(
        atPath: fixture.output.appendingPathExtension("receipt.json").path
      )
    )
    let persistedReceipt = try JSONDecoder().decode(
      NativeLosslessPackV1ProductionReceipt.self,
      from: Data(
        contentsOf: fixture.output.appendingPathExtension("receipt.json")
      )
    )
    XCTAssertEqual(persistedReceipt, receipt)
    XCTAssertEqual(try fileSHA256(fixture.output), receipt.outputSHA256)

    let decoded = try decodePortableFixture(fixture.output)
    XCTAssertEqual(decoded.count, 32 * 64 * 64)
    for scan in 0..<32 {
      for pixel in 0..<(64 * 64) {
        let expected: UInt8 = pixel == 196 ? 0 : UInt8((pixel + scan) % 251)
        XCTAssertEqual(decoded[scan * 64 * 64 + pixel], expected)
      }
    }
    XCTAssertEqual(
      SHA256.hash(data: Data(decoded)).hex,
      receipt.workingLogicalSHA256
    )
    var raw = [UInt16]()
    raw.reserveCapacity(decoded.count)
    for scan in 0..<32 {
      for pixel in 0..<(64 * 64) {
        raw.append(pixel == 196 ? UInt16.max : UInt16((pixel + scan) % 251))
      }
    }
    let rawData = raw.withUnsafeBytes { Data($0) }
    XCTAssertEqual(SHA256.hash(data: rawData).hex, receipt.sourceRawLogicalSHA256)
  }

  func testCancellationAndSourceMutationPublishNothing() throws {
    let cancelledFixture = try producerFixture()
    defer { try? FileManager.default.removeItem(at: cancelledFixture.root) }
    let producer = NativeLosslessPackV1Producer(cacheDirectory: cancelledFixture.cache)
    let inspection = try producer.inspect(input: cancelledFixture.source)
    let plan = try producer.plan(
      inspection: inspection,
      destination: cancelledFixture.output,
      maximumTransientBytes: 8 * 1024 * 1024,
      availableOutputDiskBytes: 64 * 1024 * 1024
    )
    XCTAssertThrowsError(try producer.produce(plan, shouldCancel: { true })) { error in
      XCTAssertEqual(error as? NativeLosslessPackV1ProducerError, .cancelled)
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: cancelledFixture.output.path))
    XCTAssertFalse(
      FileManager.default.fileExists(
        atPath: cancelledFixture.output.appendingPathExtension("receipt.json").path
      )
    )

    let mutatedFixture = try producerFixture()
    defer { try? FileManager.default.removeItem(at: mutatedFixture.root) }
    let mutatedProducer = NativeLosslessPackV1Producer(cacheDirectory: mutatedFixture.cache)
    let mutatedInspection = try mutatedProducer.inspect(input: mutatedFixture.source)
    let mutatedPlan = try mutatedProducer.plan(
      inspection: mutatedInspection,
      destination: mutatedFixture.output,
      maximumTransientBytes: 8 * 1024 * 1024,
      availableOutputDiskBytes: 64 * 1024 * 1024
    )
    let handle = try FileHandle(forWritingTo: mutatedFixture.source)
    try handle.seek(toOffset: 4_096)
    try handle.write(contentsOf: Data([0xff]))
    try handle.close()
    XCTAssertThrowsError(try mutatedProducer.produce(mutatedPlan)) { error in
      guard case NativeLosslessPackV1ProducerError.sourceChanged = error else {
        return XCTFail("Expected a source-identity failure, received \(error)")
      }
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: mutatedFixture.output.path))
    XCTAssertFalse(
      FileManager.default.fileExists(
        atPath: mutatedFixture.output.appendingPathExtension("receipt.json").path
      )
    )
  }

  func testPythonReferenceParserParityWhenRequested() throws {
    guard ProcessInfo.processInfo.environment["QGPU_RUN_PYTHON_PARITY"] == "1" else {
      throw XCTSkip("Set QGPU_RUN_PYTHON_PARITY=1 for the cross-runtime contract check.")
    }
    let fixture = try producerFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let producer = NativeLosslessPackV1Producer(cacheDirectory: fixture.cache)
    let inspection = try producer.inspect(input: fixture.source)
    let plan = try producer.plan(
      inspection: inspection,
      destination: fixture.output,
      maximumTransientBytes: 8 * 1024 * 1024,
      availableOutputDiskBytes: 64 * 1024 * 1024
    )
    let receipt = try producer.produce(plan)
    let root = try repositoryRoot()
    let python =
      ProcessInfo.processInfo.environment["QGPU_PYTHON"].map {
        URL(fileURLWithPath: $0)
      } ?? root.appendingPathComponent(".venv/bin/python")
    guard FileManager.default.isExecutableFile(atPath: python.path) else {
      throw XCTSkip("The repository Python environment is not available.")
    }
    let script = """
      import hashlib
      import h5py
      import numpy as np
      from quantem.gpu.io._compact_h5 import CompactH5Index, CompactH5ReferenceDecoder
      source = CompactH5Index.from_file(r'\(fixture.output.path)')
      assert source.schema_version == 3
      assert source.shape == (4, 8, 64, 64)
      assert source.source_identity_sha256 == '\(receipt.sourceIdentitySHA256)'
      assert source.raw_reconstruction_available
      with h5py.File(r'\(fixture.source.path)', 'r') as original:
          mask = np.asarray(
              original['/entry/instrument/detector/detectorSpecific/pixel_mask'][...],
              dtype='<u4',
          )
      assert source.manifest['detector_mask_sha256'] == hashlib.sha256(
          mask.tobytes(order='C')
      ).hexdigest()
      decoder = CompactH5ReferenceDecoder(source)
      for scan in (0, 7, 31):
          row, column = divmod(scan, 8)
          for pixel in (0, 1, 196, 4095):
              detector_row, detector_column = divmod(pixel, 64)
              expected = 65535 if pixel == 196 else (pixel + scan) % 251
              assert decoder.raw_value(row, column, detector_row, detector_column) == expected
      """
    let process = Process()
    process.executableURL = python
    process.arguments = ["-c", script]
    var environment = ProcessInfo.processInfo.environment
    environment["PYTHONPATH"] = root.appendingPathComponent("src").path
    process.environment = environment
    let errors = Pipe()
    process.standardError = errors
    try process.run()
    process.waitUntilExit()
    let errorText = String(
      decoding: errors.fileHandleForReading.readDataToEndOfFile(),
      as: UTF8.self
    )
    XCTAssertEqual(process.terminationStatus, 0, errorText)
  }
}

private struct NativeProducerFixture {
  let root: URL
  let source: URL
  let cache: URL
  let output: URL
}

private func producerFixture() throws -> NativeProducerFixture {
  let root = FileManager.default.temporaryDirectory
    .appendingPathComponent(
      "NativeLosslessPackV1ProducerTests-\(UUID().uuidString)", isDirectory: true)
  try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  let source = root.appendingPathComponent("native_lossless_pack_v1_source.h5")
  try FileManager.default.copyItem(
    at: Bundle.module.url(
      forResource: "native_lossless_pack_v1_source",
      withExtension: "h5",
      subdirectory: "Fixtures"
    )!,
    to: source
  )
  let fixture = NativeProducerFixture(
    root: root,
    source: source,
    cache: root.appendingPathComponent("cache", isDirectory: true),
    output: root.appendingPathComponent("native-lossless-pack-v1.h5")
  )
  return fixture
}

private func repositoryRoot() throws -> URL {
  let fileManager = FileManager.default
  var candidate = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
  while candidate.path != "/" {
    if fileManager.fileExists(atPath: candidate.appendingPathComponent("pyproject.toml").path) {
      return candidate
    }
    candidate.deleteLastPathComponent()
  }
  throw XCTSkip("Could not locate the repository root for the Python parity check.")
}

private func scaledInspection(
  _ inspection: NativeLosslessPackV1SourceInspection,
  scanSide: Int
) -> NativeLosslessPackV1SourceInspection {
  let source = inspection.dataset
  let dataset = Native4DSTEMDataset(
    id: source.id,
    label: source.label,
    masterPath: source.masterPath,
    dataFiles: source.dataFiles,
    indexFiles: source.indexFiles,
    scanRows: scanSide,
    scanCols: scanSide,
    detectorRows: 192,
    detectorCols: 192,
    sourceDtype: source.sourceDtype,
    sourceBytes: source.sourceBytes,
    badPixelIndices: [],
    scanPixelSizeRowNanometer: source.scanPixelSizeRowNanometer,
    scanPixelSizeColNanometer: source.scanPixelSizeColNanometer,
    kPixelSizeRow: source.kPixelSizeRow,
    kPixelSizeCol: source.kPixelSizeCol,
    kPixelUnit: source.kPixelUnit,
    acquisitionDate: source.acquisitionDate,
    metadata: source.metadata,
    schemaIdentity: source.schemaIdentity,
    sourceIdentitySHA256: source.sourceIdentitySHA256,
    masterSHA256: source.masterSHA256,
    orderedMemberSHA256: source.orderedMemberSHA256,
    sourceScanCalibration: source.sourceScanCalibration,
    scalarImageRawPath: source.scalarImageRawPath,
    detectorMaskSHA256: nil
  )
  return NativeLosslessPackV1SourceInspection(
    dataset: dataset,
    sourceMembers: inspection.sourceMembers,
    sourceIdentitySHA256: inspection.sourceIdentitySHA256,
    sourceShape: [scanSide, scanSide, 192, 192],
    sourceDtype: "uint16",
    sourceLogicalBytes: UInt64(scanSide * scanSide * 192 * 192 * 2),
    preparedIndexBytes: inspection.preparedIndexBytes,
    maximumCompressedBlockBytes: inspection.maximumCompressedBlockBytes,
    badPixelIndices: [],
    badPixelIdentitySHA256: SHA256.hash(data: Data()).hex,
    detectorMaskIdentitySHA256: SHA256.hash(
      data: Data(repeating: 0, count: 192 * 192 * 4)
    ).hex,
    detectorMaskIdentityOrigin: "implicit-all-admitted-u32-mask",
    calibration: inspection.calibration
  )
}

private func decodePortableFixture(_ url: URL) throws -> [UInt8] {
  let file = try Data(contentsOf: url, options: .mappedIfSafe)
  XCTAssertEqual(Array(file.prefix(8)), [0x51, 0x47, 0x50, 0x55, 0x48, 0x35, 0, 1])
  let binaryOffset = Int(readLE32(file, at: 16))
  let binaryBytes = Int(readLE32(file, at: 20))
  let binary = file.subdata(in: binaryOffset..<(binaryOffset + binaryBytes))
  XCTAssertEqual(Array(binary.prefix(8)), [0x51, 0x47, 0x49, 0x58, 0, 0, 0, 3])
  XCTAssertEqual(readLE32(binary, at: 8), 1)
  XCTAssertEqual(readLE32(binary, at: 36), 32)
  XCTAssertEqual(readLE32(binary, at: 44), 1)
  let record = 84
  let payloadOffset = Int(readLE64(binary, at: record))
  let payloadBytes = Int(readLE64(binary, at: record + 8))
  let headersOffset = Int(readLE64(binary, at: record + 32))
  let headersBytes = Int(readLE64(binary, at: record + 40))
  let payload = words(file.subdata(in: payloadOffset..<(payloadOffset + payloadBytes)))
  let headers = words(file.subdata(in: headersOffset..<(headersOffset + headersBytes)))
  XCTAssertEqual(headers.count, 64 * 64 * 2)

  var result = [UInt8](repeating: 0, count: 32 * 64 * 64)
  for pixel in 0..<(64 * 64) {
    let firstWord = Int(headers[pixel * 2])
    let width = Int(headers[pixel * 2 + 1] & 15)
    guard width > 0 else { continue }
    for scan in 0..<32 {
      let bit = scan * width
      let word = firstWord + bit / 32
      let shift = bit % 32
      var value = payload[word] >> UInt32(shift)
      if shift + width > 32 {
        value |= payload[word + 1] << UInt32(32 - shift)
      }
      result[scan * 64 * 64 + pixel] = UInt8(value & ((1 << UInt32(width)) - 1))
    }
  }
  return result
}

private func words(_ data: Data) -> [UInt32] {
  stride(from: 0, to: data.count, by: 4).map { readLE32(data, at: $0) }
}

private func readLE32(_ data: Data, at offset: Int) -> UInt32 {
  data.withUnsafeBytes {
    $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian
  }
}

private func readLE64(_ data: Data, at offset: Int) -> UInt64 {
  data.withUnsafeBytes {
    $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self).littleEndian
  }
}

private func fileSHA256(_ url: URL) throws -> String {
  SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe)).hex
}

extension Sequence where Element == UInt8 {
  fileprivate var hex: String {
    map { String(format: "%02x", $0) }.joined()
  }
}
