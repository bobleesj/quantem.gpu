import Foundation
import Metal
import Metal4DSTEMKernels

/// Fixed geometry consumed by the experimental paired-runtime owner kernels.
enum PairedRuntimeTANSRecordABI {
  static let scanRows = 512
  static let scanColumns = 512
  static let detectorRows = 192
  static let detectorColumns = 192
  static let recordScans = 16_384
  static let streamScans = 512
  static let streamsPerDetector = recordScans / streamScans
  static let recordsPerAcquisition = 16
  static let statesPerModel = PairedRuntimeTANSTables.stateCount
  static let workGroupDescriptorBytes = 16
}

/// Source-bound scientific metadata for exact paired-runtime records.
struct PairedRuntimeTANSSeriesDescriptor: Equatable, Sendable {
  let sourceIdentitySHA256: [String]
  let shape: [Int]
  let logicalDtype: Metal4DSTEMIntegerDType
  let detectorValidity: [UInt8]

  let modelCount = PairedRuntimeTANSTables.modelCount

  var acquisitionCount: Int { shape[0] }
  var detectorPixels: Int { shape[3] * shape[4] }
  var streamsPerRecord: Int {
    detectorPixels * PairedRuntimeTANSRecordABI.streamsPerDetector
  }
  var recordCount: Int {
    acquisitionCount * PairedRuntimeTANSRecordABI.recordsPerAcquisition
  }
  var offsetsBytesPerRecord: Int { (streamsPerRecord + 1) * 4 }
  var modesBytesPerRecord: Int { streamsPerRecord }

  init(
    sourceIdentitySHA256: [String], shape: [Int],
    logicalDtype: Metal4DSTEMIntegerDType, detectorValidity: [UInt8]
  ) throws {
    let detectorPixels =
      PairedRuntimeTANSRecordABI.detectorRows
      * PairedRuntimeTANSRecordABI.detectorColumns
    guard !sourceIdentitySHA256.isEmpty,
      sourceIdentitySHA256.count <= Int.max / PairedRuntimeTANSRecordABI.recordsPerAcquisition,
      shape
        == [
          sourceIdentitySHA256.count,
          PairedRuntimeTANSRecordABI.scanRows,
          PairedRuntimeTANSRecordABI.scanColumns,
          PairedRuntimeTANSRecordABI.detectorRows,
          PairedRuntimeTANSRecordABI.detectorColumns,
        ]
    else {
      throw pairedRuntimeTANSInvalid(
        "Paired-runtime shape must be [acquisition, 512, 512, 192, 192] with one "
          + "source identity per acquisition; got shape \(shape) and "
          + "\(sourceIdentitySHA256.count) identities.")
    }
    guard sourceIdentitySHA256.allSatisfy(Self.isLowercaseSHA256) else {
      throw pairedRuntimeTANSInvalid(
        "Every paired-runtime acquisition requires its exact lowercase SHA-256 identity.")
    }
    guard logicalDtype == .uint8 || logicalDtype == .uint16 else {
      throw pairedRuntimeTANSInvalid(
        "Paired-runtime tANS preserves native uint8 or uint16 counts; got "
          + "\(logicalDtype.rawValue).")
    }
    guard detectorValidity.count == detectorPixels,
      detectorValidity.allSatisfy({ $0 == 0 || $0 == 1 })
    else {
      throw pairedRuntimeTANSInvalid(
        "Detector validity must contain \(detectorPixels) binary bytes; got "
          + "\(detectorValidity.count).")
    }
    self.sourceIdentitySHA256 = sourceIdentitySHA256
    self.shape = shape
    self.logicalDtype = logicalDtype
    self.detectorValidity = detectorValidity
  }

  private static func isLowercaseSHA256(_ value: String) -> Bool {
    value.utf8.count == 64
      && value.utf8.allSatisfy { byte in
        (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte)
          || (UInt8(ascii: "a")...UInt8(ascii: "f")).contains(byte)
      }
  }
}

/// Host-visible terminal values emitted by one successful 16K GPU encode.
struct PairedRuntimeTANSRecordExtent: Equatable, Sendable {
  let acquisitionIndex: Int
  let recordInAcquisition: Int
  let terminalPayloadBytes: Int
  let workGroupCount: Int

  var firstScan: Int { recordInAcquisition * PairedRuntimeTANSRecordABI.recordScans }
  var scanCount: Int { PairedRuntimeTANSRecordABI.recordScans }
}

/// Proof that a GPU producer completed exact record construction without failure.
///
/// Only terminal values and this small failure flag return to the host. Payload,
/// offsets, stream modes, and optional grouping descriptors remain GPU-private.
struct PairedRuntimeTANSProducerReceipt {
  let sourceIdentitySHA256: [String]
  let deviceRegistryID: UInt64
  let recordExtents: [PairedRuntimeTANSRecordExtent]

  init(
    sourceIdentitySHA256: [String], recordExtents: [PairedRuntimeTANSRecordExtent],
    completedCommand: MTLCommandBuffer, failureFlag: MTLBuffer
  ) throws {
    guard completedCommand.status == .completed, completedCommand.error == nil else {
      throw pairedRuntimeTANSInvalid(
        "The paired-runtime producer command must complete before resident publication.")
    }
    let device = completedCommand.commandQueue.device
    guard failureFlag.device.registryID == device.registryID,
      failureFlag.storageMode == .shared, failureFlag.length >= 4,
      failureFlag.contents().load(as: UInt32.self) == 0
    else {
      throw pairedRuntimeTANSInvalid(
        "The paired-runtime producer reported an invalid stream or unreadable failure flag.")
    }
    guard !sourceIdentitySHA256.isEmpty,
      recordExtents.count
        == sourceIdentitySHA256.count * PairedRuntimeTANSRecordABI.recordsPerAcquisition
    else {
      throw pairedRuntimeTANSInvalid(
        "A producer receipt requires 16 terminal records for every source identity.")
    }
    self.sourceIdentitySHA256 = sourceIdentitySHA256
    deviceRegistryID = device.registryID
    self.recordExtents = recordExtents
  }
}

/// Private-capable buffers for one exact 16,384-scan paired-runtime record.
struct PairedRuntimeTANSRecordBuffers {
  let acquisitionIndex: Int
  let recordInAcquisition: Int
  let payload: MTLBuffer
  let offsets: MTLBuffer
  let modes: MTLBuffer
  let workGroups: MTLBuffer?

  var firstScan: Int { recordInAcquisition * PairedRuntimeTANSRecordABI.recordScans }
  var scanCount: Int { PairedRuntimeTANSRecordABI.recordScans }
}

/// Internal boundary for the future paired-runtime interactive owner.
protocol PairedRuntimeTANSRecordProviding: AnyObject {
  var descriptor: PairedRuntimeTANSSeriesDescriptor { get }
  var decodingTable: MTLBuffer { get }
  var records: [PairedRuntimeTANSRecordBuffers] { get }
  var receipt: PairedRuntimeTANSProducerReceipt { get }
}

/// Validated records ready for a paired-runtime owner query implementation.
final class PairedRuntimeTANSRecordProvider: PairedRuntimeTANSRecordProviding {
  let descriptor: PairedRuntimeTANSSeriesDescriptor
  let decodingTable: MTLBuffer
  let records: [PairedRuntimeTANSRecordBuffers]
  let receipt: PairedRuntimeTANSProducerReceipt

  init(
    descriptor: PairedRuntimeTANSSeriesDescriptor, decodingTable: MTLBuffer,
    records: [PairedRuntimeTANSRecordBuffers], receipt: PairedRuntimeTANSProducerReceipt
  ) throws {
    guard receipt.sourceIdentitySHA256 == descriptor.sourceIdentitySHA256,
      receipt.recordExtents.count == descriptor.recordCount,
      records.count == descriptor.recordCount,
      decodingTable.device.registryID == receipt.deviceRegistryID,
      decodingTable.length
        == descriptor.modelCount * PairedRuntimeTANSRecordABI.statesPerModel * 4
    else {
      throw pairedRuntimeTANSInvalid(
        "The paired-runtime receipt, records, table, and source identities must describe one series."
      )
    }
    for (index, pair) in zip(records, receipt.recordExtents).enumerated() {
      let expectedAcquisition = index / PairedRuntimeTANSRecordABI.recordsPerAcquisition
      let expectedRecord = index % PairedRuntimeTANSRecordABI.recordsPerAcquisition
      let record = pair.0
      let extent = pair.1
      guard record.acquisitionIndex == expectedAcquisition,
        record.recordInAcquisition == expectedRecord,
        extent.acquisitionIndex == expectedAcquisition,
        extent.recordInAcquisition == expectedRecord
      else {
        throw pairedRuntimeTANSInvalid(
          "Paired-runtime records and receipts must be acquisition-major at index \(index).")
      }
      try Self.validate(record: record, extent: extent, descriptor: descriptor, receipt: receipt)
    }
    self.descriptor = descriptor
    self.decodingTable = decodingTable
    self.records = records
    self.receipt = receipt
  }

  private static func validate(
    record: PairedRuntimeTANSRecordBuffers, extent: PairedRuntimeTANSRecordExtent,
    descriptor: PairedRuntimeTANSSeriesDescriptor, receipt: PairedRuntimeTANSProducerReceipt
  ) throws {
    let required = [record.payload, record.offsets, record.modes]
    guard required.allSatisfy({ $0.device.registryID == receipt.deviceRegistryID }),
      required.allSatisfy({ $0.length > 0 && $0.length <= $0.device.maxBufferLength }),
      record.offsets.length == descriptor.offsetsBytesPerRecord,
      record.modes.length == descriptor.modesBytesPerRecord
    else {
      throw pairedRuntimeTANSInvalid(
        "Every paired-runtime record buffer must be bounded and match the fixed stream ABI.")
    }
    let maximumPayloadBytes =
      PairedRuntimeTANSRecordABI.recordScans
      * descriptor.detectorPixels * MemoryLayout<UInt16>.stride
    guard (0...maximumPayloadBytes).contains(extent.terminalPayloadBytes),
      record.payload.length == max(4, Self.alignedUInt32Bytes(extent.terminalPayloadBytes))
    else {
      throw pairedRuntimeTANSInvalid(
        "A paired-runtime payload disagrees with its bounded terminal offset.")
    }
    if let workGroups = record.workGroups {
      guard (1...descriptor.streamsPerRecord).contains(extent.workGroupCount),
        workGroups.device.registryID == receipt.deviceRegistryID,
        workGroups.length
          == extent.workGroupCount * PairedRuntimeTANSRecordABI.workGroupDescriptorBytes
      else {
        throw pairedRuntimeTANSInvalid(
          "Paired-runtime work groups must be bounded fixed-size producer descriptors.")
      }
    } else if extent.workGroupCount != 0 {
      throw pairedRuntimeTANSInvalid(
        "The producer receipt declares paired-runtime work groups without a descriptor buffer.")
    }
  }

  private static func alignedUInt32Bytes(_ bytes: Int) -> Int {
    (bytes + 3) & ~3
  }
}

private func pairedRuntimeTANSInvalid(_ message: String) -> Metal4DSTEMStreamingIOError {
  .invalidRequest(message)
}
