import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

final class PairedRuntimeTANSRecordProviderTests: XCTestCase {
  func testPublishesPrivateExactRecordsAfterSuccessfulGPUProduction() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal is required for paired-runtime provider validation")
    }
    let fixture = try makeFixture(device: device)

    let provider = try PairedRuntimeTANSRecordProvider(
      descriptor: fixture.descriptor,
      decodingTable: fixture.decodingTable,
      records: fixture.records,
      receipt: fixture.receipt)

    XCTAssertEqual(provider.records.count, 16)
    XCTAssertEqual(provider.records.first?.firstScan, 0)
    XCTAssertEqual(provider.records.last?.firstScan, 15 * 16_384)
    XCTAssertEqual(provider.records[0].payload.storageMode, .private)
    XCTAssertEqual(provider.records[0].offsets.storageMode, .private)
    XCTAssertEqual(provider.records[0].modes.storageMode, .private)
    XCTAssertNil(provider.records[0].workGroups)
  }

  func testRejectsPayloadThatDisagreesWithProducerTerminalOffset() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal is required for paired-runtime provider validation")
    }
    let fixture = try makeFixture(device: device, terminalPayloadBytes: 65)

    XCTAssertThrowsError(
      try PairedRuntimeTANSRecordProvider(
        descriptor: fixture.descriptor,
        decodingTable: fixture.decodingTable,
        records: fixture.records,
        receipt: fixture.receipt)
    ) { error in
      XCTAssertTrue(
        String(describing: error).contains("terminal offset"),
        "Unexpected error: \(error)")
    }
  }

  func testRejectsProducerFailureBeforeResidentPublication() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal is required for paired-runtime provider validation")
    }
    let command = try completedCommand(device: device)
    let failure = try sharedBuffer(device: device, bytes: 4)
    failure.contents().storeBytes(of: UInt32(1), as: UInt32.self)
    let extents = (0..<16).map {
      PairedRuntimeTANSRecordExtent(
        acquisitionIndex: 0, recordInAcquisition: $0,
        terminalPayloadBytes: 64, workGroupCount: 0)
    }

    XCTAssertThrowsError(
      try PairedRuntimeTANSProducerReceipt(
        sourceIdentitySHA256: [String(repeating: "a", count: 64)],
        recordExtents: extents,
        completedCommand: command,
        failureFlag: failure))
  }

  private struct Fixture {
    let descriptor: PairedRuntimeTANSSeriesDescriptor
    let decodingTable: MTLBuffer
    let records: [PairedRuntimeTANSRecordBuffers]
    let receipt: PairedRuntimeTANSProducerReceipt
  }

  private func makeFixture(
    device: MTLDevice, terminalPayloadBytes: Int = 64
  ) throws -> Fixture {
    var validity = [UInt8](repeating: 1, count: 36_864)
    validity[3] = 0
    let descriptor = try PairedRuntimeTANSSeriesDescriptor(
      sourceIdentitySHA256: [String(repeating: "a", count: 64)],
      shape: [1, 512, 512, 192, 192],
      logicalDtype: .uint16,
      detectorValidity: validity)
    let decodingTable = try privateBuffer(
      device: device, bytes: descriptor.modelCount * 1_024 * 4)
    let payload = try privateBuffer(device: device, bytes: 64)
    let offsets = try privateBuffer(device: device, bytes: descriptor.offsetsBytesPerRecord)
    let modes = try privateBuffer(device: device, bytes: descriptor.modesBytesPerRecord)
    let records = (0..<16).map {
      PairedRuntimeTANSRecordBuffers(
        acquisitionIndex: 0,
        recordInAcquisition: $0,
        payload: payload,
        offsets: offsets,
        modes: modes,
        workGroups: nil)
    }
    let extents = (0..<16).map {
      PairedRuntimeTANSRecordExtent(
        acquisitionIndex: 0,
        recordInAcquisition: $0,
        terminalPayloadBytes: terminalPayloadBytes,
        workGroupCount: 0)
    }
    let failure = try sharedBuffer(device: device, bytes: 4)
    failure.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    let receipt = try PairedRuntimeTANSProducerReceipt(
      sourceIdentitySHA256: descriptor.sourceIdentitySHA256,
      recordExtents: extents,
      completedCommand: try completedCommand(device: device),
      failureFlag: failure)
    return Fixture(
      descriptor: descriptor,
      decodingTable: decodingTable,
      records: records,
      receipt: receipt)
  }

  private func completedCommand(device: MTLDevice) throws -> MTLCommandBuffer {
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    command.commit()
    command.waitUntilCompleted()
    return command
  }

  private func privateBuffer(device: MTLDevice, bytes: Int) throws -> MTLBuffer {
    try XCTUnwrap(device.makeBuffer(length: bytes, options: .storageModePrivate))
  }

  private func sharedBuffer(device: MTLDevice, bytes: Int) throws -> MTLBuffer {
    try XCTUnwrap(device.makeBuffer(length: bytes, options: .storageModeShared))
  }
}
