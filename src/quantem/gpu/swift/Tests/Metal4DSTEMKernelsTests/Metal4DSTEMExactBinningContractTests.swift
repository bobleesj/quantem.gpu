import Foundation
import XCTest

@testable import Metal4DSTEMKernels

final class Metal4DSTEMExactBinningContractTests: XCTestCase {
  func testLoadPlanAccountsForExactDetectorSumBinning() throws {
    let region = try Metal4DSTEMScanRegion.full(sourceRows: 8, sourceColumns: 8)
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 8,
      sourceScanColumns: 8,
      detectorRows: 5,
      detectorColumns: 7,
      sourceBytesPerValue: 2,
      scanRegion: region,
      detectorBin: 2
    )

    XCTAssertEqual(plan.outputDetectorRows, 3)
    XCTAssertEqual(plan.outputDetectorColumns, 4)
    XCTAssertEqual(plan.outputDetectorPixels, 12)
    XCTAssertEqual(plan.residentBytesPerValue, MemoryLayout<UInt32>.stride)
    XCTAssertEqual(plan.residentVolumeBytes, 64 * 12 * 4)
    XCTAssertEqual(plan.detectorContributionCount(outputRow: 0, outputColumn: 0), 4)
    XCTAssertEqual(plan.detectorContributionCount(outputRow: 2, outputColumn: 3), 1)
    XCTAssertTrue(plan.provenanceLabel.contains("detector-sum bin 2×2"))
  }

  func testExactBinnerFullBin2ProvenanceAndSampling() throws {
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 512,
      sourceScanColumns: 512,
      detectorRows: 192,
      detectorColumns: 192,
      sourceBytesPerValue: 2,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 512, sourceColumns: 512),
      detectorBin: 2
    )
    let audit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "a", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [9, 2],
      maximumSourceCount: 53,
      pixelsAbove255: 0
    )
    XCTAssertEqual(
      audit.auditSHA256,
      "feea583af9482886a18e26dfd0b398c842057a17092e77e712feddce2280147c"
    )

    let provenance = try Metal4DSTEMExactBinner.provenance(
      plan: plan,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )

    XCTAssertEqual(provenance.outputScanRows, 512)
    XCTAssertEqual(provenance.outputScanColumns, 512)
    XCTAssertEqual(provenance.outputDetectorRows, 96)
    XCTAssertEqual(provenance.outputDetectorColumns, 96)
    XCTAssertEqual(provenance.scanBin, 1)
    XCTAssertEqual(provenance.detectorBin, 2)
    XCTAssertEqual(provenance.sourceDtype, .uint16)
    XCTAssertEqual(provenance.stagingDtype, .uint16)
    XCTAssertEqual(provenance.outputDtype, .uint16)
    XCTAssertEqual(provenance.stagingLayout, .frameMajorSelectedColumns)
    XCTAssertEqual(provenance.outputLayout, .detectorWordMajorPackedUInt16)
    XCTAssertEqual(provenance.badPixelApplication, .alreadyZeroedUsingAudit)
    XCTAssertEqual(provenance.badPixelIndices, [2, 9])
    XCTAssertEqual(provenance.maximumOutputCount, 212)
    XCTAssertEqual(provenance.outputPayloadBytes, 4_831_838_208)

    let shardPlan = try Metal4DSTEMExactBinningShardPlan(
      provenance: provenance,
      maximumShardBytes: 603_979_776
    )
    XCTAssertEqual(shardPlan.bytesPerOutputScanRow, 9_437_184)
    XCTAssertEqual(shardPlan.shards.count, 8)
    XCTAssertEqual(shardPlan.totalPayloadBytes, 4_831_838_208)
    XCTAssertEqual(shardPlan.maximumActualShardBytes, 603_979_776)
    for (index, shard) in shardPlan.shards.enumerated() {
      XCTAssertEqual(shard.index, index)
      XCTAssertEqual(shard.outputScanRowStart, index * 64)
      XCTAssertEqual(shard.outputScanRowStop, (index + 1) * 64)
      XCTAssertEqual(shard.outputScanPositionStart, index * 64 * 512)
      XCTAssertEqual(shard.outputScanPositionCount, 64 * 512)
      XCTAssertEqual(shard.payloadBytes, 603_979_776)
      XCTAssertEqual(shard.outputLayout, .detectorWordMajorPackedUInt16)
    }
    try shardPlan.validate(provenance: provenance)
    try shardPlan.validate(sourceAudit: audit)
    XCTAssertEqual(
      try JSONDecoder().decode(
        Metal4DSTEMExactBinningShardPlan.self,
        from: JSONEncoder().encode(shardPlan)
      ),
      shardPlan
    )

    let sampling = try provenance.propagatingSampling(
      sourceScan: Metal4DSTEMAxisSampling(
        row: 0.08,
        column: 0.11,
        unit: "nm",
        provenance: "acquisition metadata",
        evidence: "fixture scan calibration"
      ),
      sourceDetector: Metal4DSTEMAxisSampling(
        row: 0.02,
        column: 0.03,
        unit: "1/nm",
        provenance: "acquisition metadata",
        evidence: "fixture diffraction calibration"
      )
    )
    XCTAssertEqual(sampling.scanState, .unchanged)
    XCTAssertEqual(sampling.detectorState, .uniformlyScaled)
    XCTAssertEqual(sampling.workingScan?.row, 0.08)
    XCTAssertEqual(sampling.workingScan?.column, 0.11)
    XCTAssertEqual(sampling.workingDetector?.row, 0.04)
    XCTAssertEqual(sampling.workingDetector?.column, 0.06)
    XCTAssertEqual(sampling.firstWorkingDetectorCenterRowInSourcePixels, 0.5)
    XCTAssertEqual(sampling.firstWorkingDetectorCenterColumnInSourcePixels, 0.5)
    try provenance.validate(sourceAudit: audit)
    XCTAssertEqual(
      try JSONDecoder().decode(
        Metal4DSTEMExactBinningProvenance.self,
        from: JSONEncoder().encode(provenance)
      ),
      provenance
    )

    var object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(provenance))
        as? [String: Any]
    )
    object["outputDetectorRows"] = 95
    let tampered = try JSONDecoder().decode(
      Metal4DSTEMExactBinningProvenance.self,
      from: JSONSerialization.data(withJSONObject: object)
    )
    XCTAssertThrowsError(try tampered.validate(sourceAudit: audit)) { error in
      XCTAssertEqual(error as? Metal4DSTEMExactBinnerError, .invalidProvenance)
    }
  }

  func testExactBinningShardPlanFailsClosed() throws {
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 4,
      sourceScanColumns: 3,
      detectorRows: 3,
      detectorColumns: 5,
      sourceBytesPerValue: 2,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 4, sourceColumns: 3),
      detectorBin: 2
    )
    let audit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "7", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: 11,
      pixelsAbove255: 0
    )
    let provenance = try Metal4DSTEMExactBinner.provenance(
      plan: plan,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )
    XCTAssertEqual(provenance.outputPayloadBytes, 144)
    XCTAssertThrowsError(
      try Metal4DSTEMExactBinningShardPlan(
        provenance: provenance,
        maximumShardBytes: 35
      )
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMExactBinnerError,
        .maximumShardBytesTooSmall(required: 36, actual: 35)
      )
    }

    let shardPlan = try Metal4DSTEMExactBinningShardPlan(
      provenance: provenance,
      maximumShardBytes: 72
    )
    XCTAssertEqual(shardPlan.shards.map(\.outputScanRowStart), [0, 2])
    XCTAssertEqual(shardPlan.shards.map(\.outputScanRowStop), [2, 4])

    let differentAudit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "8", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: 11,
      pixelsAbove255: 0
    )
    let differentProvenance = try Metal4DSTEMExactBinner.provenance(
      plan: plan,
      sourceAudit: differentAudit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )
    XCTAssertThrowsError(
      try shardPlan.validate(provenance: differentProvenance)
    ) { error in
      XCTAssertEqual(error as? Metal4DSTEMExactBinnerError, .invalidShardPlan)
    }
  }

  func testExactBinnerUInt16BoundaryAndTamperedAuditFailClosed() throws {
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 1,
      sourceScanColumns: 1,
      detectorRows: 2,
      detectorColumns: 2,
      sourceBytesPerValue: 2,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 1, sourceColumns: 1),
      detectorBin: 2
    )
    let accepted = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "b", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: 16_383,
      pixelsAbove255: 1
    )
    XCTAssertEqual(
      try Metal4DSTEMExactBinner.provenance(
        plan: plan,
        sourceAudit: accepted,
        stagingDtype: .uint16,
        outputDtype: .uint16
      ).maximumOutputCount,
      65_532
    )

    let rejected = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "b", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: 16_384,
      pixelsAbove255: 1
    )
    XCTAssertThrowsError(
      try Metal4DSTEMExactBinner.provenance(
        plan: plan,
        sourceAudit: rejected,
        stagingDtype: .uint16,
        outputDtype: .uint16
      )
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMExactBinnerError,
        .outputRangeOverflow(maximum: 65_536, dtype: .uint16)
      )
    }
    let wide = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "c", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: UInt32(UInt16.max),
      pixelsAbove255: 1
    )
    XCTAssertEqual(
      try Metal4DSTEMExactBinner.provenance(
        plan: plan,
        sourceAudit: wide,
        stagingDtype: .uint16,
        outputDtype: .uint32
      ).maximumOutputCount,
      262_140
    )

    var object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(accepted))
        as? [String: Any]
    )
    object["maximumSourceCount"] = 16_382
    let tampered = try JSONDecoder().decode(
      Metal4DSTEMExactSourceAudit.self,
      from: JSONSerialization.data(withJSONObject: object)
    )
    XCTAssertThrowsError(try tampered.validate()) { error in
      XCTAssertEqual(error as? Metal4DSTEMExactBinnerError, .invalidAuditIdentity)
    }
  }

  func testExactBinnerRejectsDtypeMismatchAndNonuniformSamplingClaim() throws {
    let mismatchedPlan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 1,
      sourceScanColumns: 1,
      detectorRows: 5,
      detectorColumns: 7,
      sourceBytesPerValue: 1,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 1, sourceColumns: 1),
      detectorBin: 2
    )
    let audit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "d", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: 10,
      pixelsAbove255: 0
    )
    XCTAssertThrowsError(
      try Metal4DSTEMExactBinner.provenance(
        plan: mismatchedPlan,
        sourceAudit: audit,
        stagingDtype: .uint16,
        outputDtype: .uint16
      )
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMExactBinnerError,
        .sourceDtypeMismatch(expectedBytes: 1, actual: .uint16)
      )
    }

    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 1,
      sourceScanColumns: 1,
      detectorRows: 5,
      detectorColumns: 7,
      sourceBytesPerValue: 2,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 1, sourceColumns: 1),
      detectorBin: 2
    )
    let provenance = try Metal4DSTEMExactBinner.provenance(
      plan: plan,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )
    let sampling = try provenance.propagatingSampling(
      sourceScan: nil,
      sourceDetector: Metal4DSTEMAxisSampling(
        row: 0.01,
        column: 0.02,
        unit: "1/nm",
        provenance: "fixture",
        evidence: "odd detector geometry"
      )
    )
    XCTAssertEqual(sampling.detectorState, .nonuniformEdgeBins)
    XCTAssertNil(sampling.workingDetector)
  }

  func testExactBinnerRejectsUnprovedNarrowStagingAndInvalidAuditGeometry() throws {
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 1,
      sourceScanColumns: 1,
      detectorRows: 2,
      detectorColumns: 2,
      sourceBytesPerValue: 2,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 1, sourceColumns: 1),
      detectorBin: 2
    )
    let wideAudit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "f", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: 300,
      pixelsAbove255: 1
    )
    XCTAssertThrowsError(
      try Metal4DSTEMExactBinner.provenance(
        plan: plan,
        sourceAudit: wideAudit,
        stagingDtype: .uint8,
        outputDtype: .uint16
      )
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMExactBinnerError,
        .stagingRangeOverflow(maximum: 300, dtype: .uint8)
      )
    }

    let outOfRangeBadPixel = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "f", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [4],
      maximumSourceCount: 10,
      pixelsAbove255: 0
    )
    XCTAssertThrowsError(
      try Metal4DSTEMExactBinner.provenance(
        plan: plan,
        sourceAudit: outOfRangeBadPixel,
        stagingDtype: .uint16,
        outputDtype: .uint16
      )
    ) { error in
      XCTAssertEqual(error as? Metal4DSTEMExactBinnerError, .invalidAuditValues)
    }

    XCTAssertThrowsError(
      try Metal4DSTEMExactSourceAudit(
        sourceIdentitySHA256: String(repeating: "f", count: 64),
        sourceDtype: .uint16,
        badPixelIndices: [],
        maximumSourceCount: 300,
        pixelsAbove255: 0
      )
    ) { error in
      XCTAssertEqual(error as? Metal4DSTEMExactBinnerError, .invalidAuditValues)
    }
  }

  func testSamplingPropagationRejectsNonfiniteScaledValues() throws {
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 1,
      sourceScanColumns: 1,
      detectorRows: 2,
      detectorColumns: 2,
      sourceBytesPerValue: 2,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 1, sourceColumns: 1),
      detectorBin: 2
    )
    let audit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "1", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: 10,
      pixelsAbove255: 0
    )
    let provenance = try Metal4DSTEMExactBinner.provenance(
      plan: plan,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )
    let sourceSampling = try Metal4DSTEMAxisSampling(
      row: Double.greatestFiniteMagnitude,
      column: 1,
      unit: "1/nm",
      provenance: "synthetic boundary",
      evidence: "overflow test"
    )
    XCTAssertThrowsError(
      try provenance.propagatingSampling(
        sourceScan: nil,
        sourceDetector: sourceSampling
      )
    ) { error in
      XCTAssertEqual(error as? Metal4DSTEMExactBinnerError, .invalidCalibration)
    }
  }
}
