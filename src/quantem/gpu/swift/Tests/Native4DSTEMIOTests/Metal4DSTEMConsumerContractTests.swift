import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

final class Metal4DSTEMConsumerContractTests: XCTestCase {
  private let sourceA = String(repeating: "a", count: 64)
  private let sourceB = String(repeating: "b", count: 64)

  func testCanonicalRepresentationNamesHaveNoLegacyAlias() throws {
    for value in ["dense", "packed", "ans"] {
      let encoded = Data("\"\(value)\"".utf8)
      let representation = try JSONDecoder().decode(
        Metal4DSTEMResidentRepresentation.self, from: encoded)
      XCTAssertEqual(representation.rawValue, value)
      XCTAssertEqual(try JSONEncoder().encode(representation), encoded)
    }
    XCTAssertThrowsError(
      try JSONDecoder().decode(
        Metal4DSTEMResidentRepresentation.self, from: Data("\"lossless_packed\"".utf8)))
  }

  func testResidentReceiptSeparatesScientificWorkingAndPhysicalBytes() throws {
    let receipt = Metal4DSTEMResidentReceipt(
      schema: Metal4DSTEMResidentReceipt.currentSchema,
      representation: .packed,
      sourceIdentitySHA256: sourceA,
      sourceShape: [512, 512, 192, 192],
      workingShape: [512, 512, 192, 192],
      sourceDtype: "uint16",
      workingDtype: "uint8",
      sourceLogicalTensorBytes: 19_327_352_832,
      workingLogicalTensorBytes: 9_663_676_416,
      physicalResidentBytes: 2_394_650_896,
      containerBytes: 2_394_887_216,
      storageSchema: "quantem.gpu.packed-detector-h5/v3",
      losslessExact: true,
      scanBin: 1,
      detectorBin: 1,
      crop: nil,
      detectorMaskCount: 1,
      detectorMaskSHA256: sourceB,
      detectorMaskSchema: "quantem.gpu.detector-mask-identity/opaque-v1",
      calibrationSchema: nil,
      calibrationSHA256: nil,
      provenanceSchema: "quantem.gpu.packed-detector-h5-manifest/v1",
      provenanceSHA256: String(repeating: "c", count: 64),
      sourceRawLogicalSHA256: String(repeating: "d", count: 64),
      workingLogicalSHA256: String(repeating: "e", count: 64),
      implementationRevision: nil
    )

    XCTAssertNoThrow(try receipt.validate())
    XCTAssertNotEqual(receipt.sourceLogicalTensorBytes, receipt.workingLogicalTensorBytes)
    XCTAssertNotEqual(receipt.workingLogicalTensorBytes, receipt.physicalResidentBytes)
  }

  func testResidentReceiptSupportsFull256AndExactDetectorBinTwo() throws {
    let full256 = Metal4DSTEMResidentReceipt(
      schema: Metal4DSTEMResidentReceipt.currentSchema,
      representation: .packed,
      sourceIdentitySHA256: sourceA,
      sourceShape: [512, 512, 256, 256],
      workingShape: [512, 512, 256, 256],
      sourceDtype: "uint16",
      workingDtype: "uint16",
      sourceLogicalTensorBytes: 34_359_738_368,
      workingLogicalTensorBytes: 34_359_738_368,
      physicalResidentBytes: 4_000_000_000,
      containerBytes: 4_000_065_536,
      storageSchema: "quantem.gpu.packed-detector-h5/v1",
      losslessExact: true,
      scanBin: 1,
      detectorBin: 1,
      crop: nil,
      detectorMaskCount: 0,
      detectorMaskSHA256: nil,
      detectorMaskSchema: nil,
      calibrationSchema: nil,
      calibrationSHA256: nil,
      provenanceSchema: "quantem.gpu.packed-detector-h5-manifest/v1",
      provenanceSHA256: String(repeating: "c", count: 64),
      sourceRawLogicalSHA256: String(repeating: "d", count: 64),
      workingLogicalSHA256: nil,
      implementationRevision: nil
    )
    XCTAssertNoThrow(try full256.validate())

    let detectorBinTwo = Metal4DSTEMResidentReceipt(
      schema: Metal4DSTEMResidentReceipt.currentSchema,
      representation: .dense,
      sourceIdentitySHA256: sourceA,
      sourceShape: [512, 512, 192, 192],
      workingShape: [512, 512, 96, 96],
      sourceDtype: "uint16",
      workingDtype: "uint16",
      sourceLogicalTensorBytes: 19_327_352_832,
      workingLogicalTensorBytes: 4_831_838_208,
      physicalResidentBytes: 4_831_838_208,
      containerBytes: nil,
      storageSchema: "quantem.gpu.indexed-resident-integer/v1",
      losslessExact: true,
      scanBin: 1,
      detectorBin: 2,
      crop: nil,
      detectorMaskCount: 0,
      detectorMaskSHA256: nil,
      detectorMaskSchema: nil,
      calibrationSchema: nil,
      calibrationSHA256: nil,
      provenanceSchema: "quantem.gpu.metal-4dstem-exact-binning/v1",
      provenanceSHA256: String(repeating: "e", count: 64),
      sourceRawLogicalSHA256: nil,
      workingLogicalSHA256: nil,
      implementationRevision: nil
    )
    XCTAssertNoThrow(try detectorBinTwo.validate())
  }

  func testExactProductsDeriveMeanDPAndCenteredDPC() throws {
    let products = Metal4DSTEMExactProducts(
      detectorSum: [6, 12],
      band1: [0, 0, 0, 0],
      band2: [0, 0, 0, 0],
      band4: [0, 0, 0, 0],
      total: [2, 2, 0, 4],
      detectorRowMoment: [0, 2, 0, 4],
      detectorColumnMoment: [2, 0, 0, 4]
    )

    XCTAssertEqual(try products.meanDiffractionPattern(frameCount: 3), [2, 4])
    let dpc = try products.centeredDPC(
      scanRows: 2,
      scanColumns: 2,
      detectorRows: 2,
      detectorColumns: 2
    )
    XCTAssertEqual(dpc.row, [-0.5, 0.5, -0.5, 0.5])
    XCTAssertEqual(dpc.column, [0.5, -0.5, -0.5, 0.5])
  }

  func testCenteredDPCRejectsImpossibleMoment() throws {
    let products = Metal4DSTEMExactProducts(
      detectorSum: [1],
      band1: [1],
      band2: [1],
      band4: [1],
      total: [1],
      detectorRowMoment: [2],
      detectorColumnMoment: [0]
    )

    XCTAssertThrowsError(
      try products.centeredDPC(
        scanRows: 1,
        scanColumns: 1,
        detectorRows: 2,
        detectorColumns: 2
      )
    )
  }

  func testCapabilitiesRequireEveryProduct() {
    let products = Metal4DSTEMResidentProduct.allCases.map {
      Metal4DSTEMResidentProductCapability(
        product: $0,
        availability: .residentOnDemand,
        numerics: .exactInteger
      )
    }
    let receipt = Metal4DSTEMResidentReceipt(
      schema: Metal4DSTEMResidentReceipt.currentSchema,
      representation: .dense,
      sourceIdentitySHA256: sourceA,
      sourceShape: [2, 2, 3, 4],
      workingShape: [2, 2, 3, 4],
      sourceDtype: "uint16",
      workingDtype: "uint16",
      sourceLogicalTensorBytes: 96,
      workingLogicalTensorBytes: 96,
      physicalResidentBytes: 96,
      containerBytes: nil,
      storageSchema: "quantem.gpu.indexed-resident-integer/v1",
      losslessExact: true,
      scanBin: 1,
      detectorBin: 1,
      crop: nil,
      detectorMaskCount: 0,
      detectorMaskSHA256: nil,
      detectorMaskSchema: nil,
      calibrationSchema: nil,
      calibrationSHA256: nil,
      provenanceSchema: nil,
      provenanceSHA256: nil,
      sourceRawLogicalSHA256: nil,
      workingLogicalSHA256: nil,
      implementationRevision: nil
    )
    let complete = Metal4DSTEMResidentCapabilities(
      schema: Metal4DSTEMResidentCapabilities.currentSchema,
      representation: .dense,
      sourceIdentitySHA256: sourceA,
      scanRows: 2,
      scanColumns: 2,
      detectorRows: 3,
      detectorColumns: 4,
      storageSchema: "quantem.gpu.indexed-resident-integer/v1",
      workingDtype: "uint16",
      exactIntegerBits: 16,
      logicalTensorBytes: 96,
      completeSourceResident: true,
      residentBytes: 96,
      residentStorageBytes: 96,
      lossless: true,
      residentReceipt: receipt,
      products: products
    )
    XCTAssertTrue(complete.fullInteractiveResident)

    let lossy = Metal4DSTEMResidentCapabilities(
      schema: complete.schema,
      representation: complete.representation,
      sourceIdentitySHA256: complete.sourceIdentitySHA256,
      scanRows: complete.scanRows,
      scanColumns: complete.scanColumns,
      detectorRows: complete.detectorRows,
      detectorColumns: complete.detectorColumns,
      storageSchema: complete.storageSchema,
      workingDtype: complete.workingDtype,
      exactIntegerBits: complete.exactIntegerBits,
      logicalTensorBytes: complete.logicalTensorBytes,
      completeSourceResident: true,
      residentBytes: complete.residentBytes,
      residentStorageBytes: complete.residentStorageBytes,
      lossless: false,
      residentReceipt: receipt,
      products: products
    )
    XCTAssertFalse(lossy.fullInteractiveResident)

    let missing = Metal4DSTEMResidentCapabilities(
      schema: complete.schema,
      representation: complete.representation,
      sourceIdentitySHA256: complete.sourceIdentitySHA256,
      scanRows: complete.scanRows,
      scanColumns: complete.scanColumns,
      detectorRows: complete.detectorRows,
      detectorColumns: complete.detectorColumns,
      storageSchema: complete.storageSchema,
      workingDtype: complete.workingDtype,
      exactIntegerBits: complete.exactIntegerBits,
      logicalTensorBytes: complete.logicalTensorBytes,
      completeSourceResident: true,
      residentBytes: complete.residentBytes,
      residentStorageBytes: complete.residentStorageBytes,
      lossless: complete.lossless,
      residentReceipt: receipt,
      products: Array(products.dropLast())
    )
    XCTAssertFalse(missing.fullInteractiveResident)
  }

  func testPublicationRecorderRejectsStaleABA() throws {
    let recorder = Metal4DSTEMPublicationRecorder(signpostsEnabled: false)
    XCTAssertTrue(
      try recorder.begin(
        generation: 1,
        sourceIdentitySHA256: sourceA,
        representation: .dense,
        counters: .init(
          sourceBytes: 10,
          residentBytes: 20,
          peakProcessRSSBytes: 30,
          peakDeviceAllocatedBytes: 40
        )
      )
    )
    XCTAssertTrue(try recorder.record(generation: 1, milestone: .sourceAdmitted))
    XCTAssertTrue(try recorder.record(generation: 1, milestone: .residentReady))
    XCTAssertTrue(
      try recorder.begin(
        generation: 2,
        sourceIdentitySHA256: sourceB,
        representation: .packed
      )
    )
    XCTAssertFalse(
      try recorder.record(generation: 1, milestone: .firstResidentPresent)
    )
    XCTAssertTrue(
      try recorder.begin(
        generation: 3,
        sourceIdentitySHA256: sourceA,
        representation: .dense
      )
    )
    XCTAssertTrue(try recorder.record(generation: 3, milestone: .residentReady))
    XCTAssertTrue(
      try recorder.record(generation: 3, milestone: .firstResidentPresent)
    )

    XCTAssertEqual(
      recorder.events().map(\.milestone),
      [
        .requested,
        .sourceAdmitted,
        .residentReady,
        .requested,
        .supersededRejected,
        .requested,
        .residentReady,
        .firstResidentPresent,
      ]
    )
    XCTAssertEqual(recorder.events().first?.counters.residentBytes, 20)
    XCTAssertEqual(recorder.events().first?.counters.peakProcessRSSBytes, 30)
    XCTAssertEqual(recorder.events().first?.counters.peakDeviceAllocatedBytes, 40)
    XCTAssertEqual(recorder.events()[4].sourceIdentitySHA256, sourceA)
  }

  func testPresentAndRecoveryRequireOrderedMilestones() throws {
    let recorder = Metal4DSTEMPublicationRecorder(signpostsEnabled: false)
    XCTAssertTrue(
      try recorder.begin(
        generation: 7,
        sourceIdentitySHA256: sourceA,
        representation: .dense
      )
    )
    XCTAssertThrowsError(
      try recorder.record(generation: 7, milestone: .firstResidentPresent)
    )
    XCTAssertThrowsError(try recorder.record(generation: 7, milestone: .recoveryReady))
    XCTAssertTrue(try recorder.record(generation: 7, milestone: .deviceLost))
    XCTAssertTrue(try recorder.record(generation: 7, milestone: .recoveryReady))
    XCTAssertTrue(
      try recorder.record(generation: 7, milestone: .firstResidentPresent)
    )
  }

  func testTimingSummaryUsesNearestRank() throws {
    let summary = try Metal4DSTEMTimingSummary(
      samplesSeconds: [0.4, 0.1, 0.3, 0.2, 0.5]
    )
    XCTAssertEqual(summary.sampleCount, 5)
    XCTAssertEqual(summary.p50Seconds, 0.3)
    XCTAssertEqual(summary.p95Seconds, 0.5)
    XCTAssertEqual(summary.maximumSeconds, 0.5)
    XCTAssertNotEqual(
      Metal4DSTEMTimingBoundary.coldArbitraryToResidentReady.rawValue,
      Metal4DSTEMTimingBoundary.preparedReopenToResidentReady.rawValue
    )
    XCTAssertThrowsError(
      try Metal4DSTEMTimingSummary(samplesSeconds: [Double.nan])
    )
  }

  func testDPCProcessorPublishesResidentPhaseAndFFTSeams() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("This functional DPC test requires a Metal device.")
    }
    let processor = try Metal4DSTEMDPCProcessor(device: device)
    let row: [Float] = [
      -1.5, -0.5, 0.5, 1.5,
      -1.5, -0.5, 0.5, 1.5,
      -1.5, -0.5, 0.5, 1.5,
      -1.5, -0.5, 0.5, 1.5,
    ]
    let column: [Float] = [
      -1.5, -1.5, -1.5, -1.5,
      -0.5, -0.5, -0.5, -0.5,
      0.5, 0.5, 0.5, 0.5,
      1.5, 1.5, 1.5, 1.5,
    ]
    let expected: [Float] = [
      0, 0.58474338, 0.58474338, 0,
      -0.58474338, 0, 0, -0.58474338,
      -0.58474338, 0, 0, -0.58474338,
      0, 0.58474338, 0.58474338, 0,
    ]
    let result = try processor.process(
      centeredDPC: Metal4DSTEMCenteredDPC(row: row, column: column),
      configuration: Metal4DSTEMDPCConfiguration(
        scanRows: 4,
        scanColumns: 4,
        rotationDegrees: 17,
        transposeComponents: false
      )
    )

    let phase = result.phaseBuffer.contents().bindMemory(to: Float.self, capacity: 16)
    for index in expected.indices {
      XCTAssertEqual(phase[index], expected[index], accuracy: 2e-5)
    }
    XCTAssertEqual(result.phaseBuffer.storageMode, .shared)
    XCTAssertEqual(result.gradientFFTBuffer.storageMode, .private)
    XCTAssertEqual(result.phaseFFTBuffer.storageMode, .private)
    XCTAssertEqual(result.metrics.fftDispatchCount, 13)
    XCTAssertEqual(result.metrics.totalDispatchCount, 16)
    XCTAssertEqual(result.metrics.uploadBytes, 128)
    XCTAssertEqual(result.metrics.readbackBytes, 0)
    XCTAssertEqual(result.metrics.synchronizationCount, 1)
  }
}
