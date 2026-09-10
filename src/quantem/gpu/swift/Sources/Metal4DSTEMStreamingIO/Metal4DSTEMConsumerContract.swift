import CryptoKit
import Foundation
import Metal4DSTEMKernels
import os

/// Backend-neutral representation of a complete 4D-STEM generation.
public enum Metal4DSTEMResidentRepresentation: String, Codable, Sendable {
  case dense
  case packed
  case ans
}

/// Backend-neutral scientific and memory receipt for one resident generation.
public struct Metal4DSTEMResidentReceipt: Codable, Equatable, Sendable {
  public static let currentSchema = "quantem.gpu.4dstem-resident-receipt/v3"

  public let schema: String
  public let representation: Metal4DSTEMResidentRepresentation
  public let sourceIdentitySHA256: String
  public let sourceShape: [Int]
  public let workingShape: [Int]
  public let sourceDtype: String
  public let workingDtype: String
  public let sourceLogicalTensorBytes: UInt64
  public let workingLogicalTensorBytes: UInt64
  public let physicalResidentBytes: UInt64
  public let containerBytes: UInt64?
  public let storageSchema: String
  public let losslessExact: Bool
  public let scanBin: Int
  public let detectorBin: Int
  public let crop: [Int]?
  public let detectorMaskCount: Int
  /// Opaque unless `detectorMaskSchema` names an explicit byte encoding.
  public let detectorMaskSHA256: String?
  public let detectorMaskSchema: String?
  public let calibrationSchema: String?
  public let calibrationSHA256: String?
  public let provenanceSchema: String?
  public let provenanceSHA256: String?
  public let sourceRawLogicalSHA256: String?
  public let workingLogicalSHA256: String?
  public let implementationRevision: String?

  public func validate() throws {
    if sourceDtype == "float32" || workingDtype == "float32" {
      guard sourceDtype == workingDtype, scanBin == 1, detectorBin == 1, crop == nil else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "An exact float32 resident must preserve the original dtype and full tensor shape."
        )
      }
    }
    guard schema == Self.currentSchema,
      Self.validSHA256(sourceIdentitySHA256),
      sourceShape.count == 4, workingShape.count == 4,
      sourceShape.allSatisfy({ $0 > 0 }), workingShape.allSatisfy({ $0 > 0 }),
      let sourceBytesPerValue = Self.bytesPerValue(sourceDtype),
      let workingBytesPerValue = Self.bytesPerValue(workingDtype),
      scanBin > 0, detectorBin > 0,
      detectorMaskCount >= 0, physicalResidentBytes > 0,
      containerBytes == nil || containerBytes! > 0,
      !storageSchema.isEmpty, losslessExact
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Resident receipt metadata is incomplete or unsupported."
      )
    }
    let sourceBytes = try Self.logicalBytes(
      shape: sourceShape, bytesPerValue: sourceBytesPerValue
    )
    let workingBytes = try Self.logicalBytes(
      shape: workingShape, bytesPerValue: workingBytesPerValue
    )
    let cropRows: Int
    let cropColumns: Int
    if let crop {
      guard crop.count == 4,
        0 <= crop[0], crop[0] < crop[1], crop[1] <= sourceShape[0],
        0 <= crop[2], crop[2] < crop[3], crop[3] <= sourceShape[1]
      else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Resident receipt crop is not a valid half-open scan region."
        )
      }
      cropRows = crop[1] - crop[0]
      cropColumns = crop[3] - crop[2]
    } else {
      cropRows = sourceShape[0]
      cropColumns = sourceShape[1]
    }
    guard sourceBytes == sourceLogicalTensorBytes,
      workingBytes == workingLogicalTensorBytes,
      workingShape == [
        Self.ceilingDivide(cropRows, by: scanBin),
        Self.ceilingDivide(cropColumns, by: scanBin),
        Self.ceilingDivide(sourceShape[2], by: detectorBin),
        Self.ceilingDivide(sourceShape[3], by: detectorBin),
      ],
      detectorMaskCount == 0 || Self.validSHA256(detectorMaskSHA256),
      detectorMaskSHA256 == nil || Self.validSHA256(detectorMaskSHA256),
      (detectorMaskSchema == nil) == (detectorMaskSHA256 == nil),
      detectorMaskSchema == nil || !detectorMaskSchema!.isEmpty,
      (calibrationSchema == nil) == (calibrationSHA256 == nil),
      calibrationSchema == nil || !calibrationSchema!.isEmpty,
      calibrationSHA256 == nil || Self.validSHA256(calibrationSHA256),
      (provenanceSchema == nil) == (provenanceSHA256 == nil),
      provenanceSchema == nil || !provenanceSchema!.isEmpty,
      provenanceSHA256 == nil || Self.validSHA256(provenanceSHA256),
      sourceRawLogicalSHA256 == nil || Self.validSHA256(sourceRawLogicalSHA256),
      workingLogicalSHA256 == nil || Self.validSHA256(workingLogicalSHA256),
      representation != .dense || physicalResidentBytes == workingLogicalTensorBytes
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Resident receipt geometry, byte counts, or identities are inconsistent."
      )
    }
  }

  fileprivate static func detectorMaskSHA256(_ indices: [Int]) -> String? {
    guard !indices.isEmpty else { return nil }
    var data = Data(capacity: indices.count * MemoryLayout<UInt32>.stride)
    for index in indices {
      var value = UInt32(index).littleEndian
      withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  fileprivate static func metadataSHA256<T: Encodable>(_ value: T) -> String? {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    guard let data = try? encoder.encode(value) else { return nil }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private static func ceilingDivide(_ value: Int, by divisor: Int) -> Int {
    value / divisor + (value % divisor == 0 ? 0 : 1)
  }

  private static func validSHA256(_ value: String?) -> Bool {
    guard let value else { return false }
    return value.utf8.count == 64
      && value.utf8.allSatisfy {
        (48...57).contains($0) || (97...102).contains($0)
      }
  }

  private static func bytesPerValue(_ dtype: String) -> UInt64? {
    switch dtype {
    case "uint8": 1
    case "uint16": 2
    case "uint32", "float32": 4
    case "uint64": 8
    default: nil
    }
  }

  fileprivate static func logicalBytes(
    shape: [Int], bytesPerValue: UInt64
  ) throws -> UInt64 {
    var result = bytesPerValue
    for dimension in shape {
      let product = result.multipliedReportingOverflow(by: UInt64(dimension))
      guard !product.overflow else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Resident receipt logical byte count overflows UInt64."
        )
      }
      result = product.partialValue
    }
    return result
  }
}

/// Product roles required by a complete interactive 4D-STEM consumer.
public enum Metal4DSTEMResidentProduct: String, CaseIterable, Codable, Sendable {
  case diffractionPattern = "diffraction-pattern"
  case brightField = "bright-field"
  case annularBrightField = "annular-bright-field"
  case annularDarkField = "annular-dark-field"
  case meanDiffractionPattern = "mean-diffraction-pattern"
  case total = "total"
  case centerOfMass = "center-of-mass"
  case differentialPhaseContrast = "dpc"
  case integratedDifferentialPhaseContrast = "idpc"
  case fastFourierTransform = "fft"
}

/// When a product can be consumed after resident-ready publication.
public enum Metal4DSTEMProductAvailability: String, Codable, Sendable {
  case immediate
  case residentOnDemand = "resident-on-demand"
  case unavailable
}

/// Numerical boundary advertised for one resident product.
public enum Metal4DSTEMProductNumerics: String, Codable, Sendable {
  case exactInteger
  case exactFloat32Bits = "exact-float32-bits"
  case exactIntegerThenFloat32 = "exact-integer-then-float32"
  case frozenFloat32 = "frozen-float32"
}

public struct Metal4DSTEMResidentProductCapability: Codable, Equatable, Sendable {
  public let product: Metal4DSTEMResidentProduct
  public let availability: Metal4DSTEMProductAvailability
  public let numerics: Metal4DSTEMProductNumerics

  public init(
    product: Metal4DSTEMResidentProduct,
    availability: Metal4DSTEMProductAvailability,
    numerics: Metal4DSTEMProductNumerics
  ) {
    self.product = product
    self.availability = availability
    self.numerics = numerics
  }
}

/// UI-free capability receipt for one fully published resident generation.
public struct Metal4DSTEMResidentCapabilities: Codable, Equatable, Sendable {
  public static let currentSchema = "quantem.gpu.apple-4dstem-resident-capabilities/v4"

  public let schema: String
  public let representation: Metal4DSTEMResidentRepresentation
  public let sourceIdentitySHA256: String
  public let scanRows: Int
  public let scanColumns: Int
  public let detectorRows: Int
  public let detectorColumns: Int
  public let storageSchema: String
  public let workingDtype: String
  public let exactIntegerBits: Int
  public let logicalTensorBytes: UInt64
  public let completeSourceResident: Bool
  public let residentBytes: UInt64
  public let residentStorageBytes: UInt64
  public let lossless: Bool
  public let residentReceipt: Metal4DSTEMResidentReceipt
  public let products: [Metal4DSTEMResidentProductCapability]

  public var fullInteractiveResident: Bool {
    (try? residentReceipt.validate()) != nil
      && completeSourceResident
      && lossless
      && logicalTensorBytes > 0
      && residentStorageBytes > 0
      && residentReceipt.losslessExact
      && residentReceipt.representation == representation
      && residentReceipt.sourceIdentitySHA256 == sourceIdentitySHA256
      && residentReceipt.workingShape == [
        scanRows, scanColumns, detectorRows, detectorColumns,
      ]
      && residentReceipt.workingDtype == workingDtype
      && residentReceipt.workingLogicalTensorBytes == logicalTensorBytes
      && residentReceipt.physicalResidentBytes == residentStorageBytes
      && residentReceipt.storageSchema == storageSchema
      && Set(products.map(\.product)) == Set(Metal4DSTEMResidentProduct.allCases)
      && products.allSatisfy { $0.availability != .unavailable }
  }

  /// Describe a full float32 EMPAD tensor without advertising missing products.
  /// The scientific source identity covers dtype, shape and original detector
  /// words, not footer bytes or acquisition filenames. CoM supplies the input
  /// for the shared DPC/iDPC kernels; 2D products supply the shared FFT kernels.
  /// Availability is not consumer or release qualification.
  public static func empad(_ source: MetalEMPADResidentSource) throws -> Self {
    guard !source.isReleased else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "EMPAD resident storage was released. Reload it before requesting capabilities.")
    }
    let shape = [source.source.scanRows, source.source.scanColumns, 128, 128]
    let logicalBytes = try Metal4DSTEMResidentReceipt.logicalBytes(shape: shape, bytesPerValue: 4)
    let storageSchema = "quantem.gpu.empad-xor-row-packed/v1"
    let receipt = Metal4DSTEMResidentReceipt(
      schema: Metal4DSTEMResidentReceipt.currentSchema, representation: .packed,
      sourceIdentitySHA256: source.sourceIdentitySHA256,
      sourceShape: shape, workingShape: shape, sourceDtype: "float32", workingDtype: "float32",
      sourceLogicalTensorBytes: logicalBytes, workingLogicalTensorBytes: logicalBytes,
      physicalResidentBytes: source.residentBytes,
      containerBytes: UInt64(source.source.sourceBytes),
      storageSchema: storageSchema, losslessExact: true, scanBin: 1, detectorBin: 1, crop: nil,
      detectorMaskCount: 0, detectorMaskSHA256: nil, detectorMaskSchema: nil,
      calibrationSchema: nil, calibrationSHA256: nil,
      provenanceSchema: "quantem.gpu.empad-tensor/v1",
      provenanceSHA256: source.sourceIdentitySHA256,
      sourceRawLogicalSHA256: source.logicalSHA256, workingLogicalSHA256: source.logicalSHA256,
      implementationRevision: nil)
    try receipt.validate()
    let products = Metal4DSTEMResidentProduct.allCases.map { product in
      Metal4DSTEMResidentProductCapability(
        product: product,
        availability: .residentOnDemand,
        numerics: product == .diffractionPattern ? .exactFloat32Bits : .frozenFloat32)
    }
    return Self(
      schema: currentSchema, representation: .packed,
      sourceIdentitySHA256: source.sourceIdentitySHA256,
      scanRows: shape[0], scanColumns: shape[1], detectorRows: 128, detectorColumns: 128,
      storageSchema: storageSchema, workingDtype: "float32", exactIntegerBits: 0,
      logicalTensorBytes: logicalBytes, completeSourceResident: true,
      residentBytes: source.residentBytes, residentStorageBytes: source.residentBytes,
      lossless: true, residentReceipt: receipt, products: products)
  }

  /// Describe an exact indexed/sharded uint16 result without taking ownership.
  public static func indexed(
    _ result: Metal4DSTEMIndexedBinnedLoadResult
  ) throws -> Self {
    let provenance = result.binningProvenance
    let sourceShape = [
      provenance.sourceScanRows,
      provenance.sourceScanColumns,
      provenance.sourceDetectorRows,
      provenance.sourceDetectorColumns,
    ]
    let workingShape = [
      provenance.outputScanRows,
      provenance.outputScanColumns,
      provenance.outputDetectorRows,
      provenance.outputDetectorColumns,
    ]
    let sourceLogicalTensorBytes = try Metal4DSTEMResidentReceipt.logicalBytes(
      shape: sourceShape,
      bytesPerValue: UInt64(provenance.sourceDtype.bytesPerValue)
    )
    let workingLogicalTensorBytes = try Metal4DSTEMResidentReceipt.logicalBytes(
      shape: workingShape,
      bytesPerValue: UInt64(provenance.outputDtype.bytesPerValue)
    )
    let region = provenance.scanRegion
    let crop: [Int]? =
      region.rowStart == 0 && region.rowStop == provenance.sourceScanRows
        && region.columnStart == 0 && region.columnStop == provenance.sourceScanColumns
      ? nil : [region.rowStart, region.rowStop, region.columnStart, region.columnStop]
    let detectorMaskSHA256 = Metal4DSTEMResidentReceipt.detectorMaskSHA256(
      result.sourceAudit.badPixelIndices
    )
    let samplingSHA256 = Metal4DSTEMResidentReceipt.metadataSHA256(
      result.samplingPropagation
    )
    let representation: Metal4DSTEMResidentRepresentation =
      result.metrics.workingPayloadBytes == workingLogicalTensorBytes
      ? .dense : .packed
    let receipt = Metal4DSTEMResidentReceipt(
      schema: Metal4DSTEMResidentReceipt.currentSchema,
      representation: representation,
      sourceIdentitySHA256: provenance.sourceIdentitySHA256,
      sourceShape: sourceShape,
      workingShape: workingShape,
      sourceDtype: provenance.sourceDtype.rawValue,
      workingDtype: provenance.outputDtype.rawValue,
      sourceLogicalTensorBytes: sourceLogicalTensorBytes,
      workingLogicalTensorBytes: workingLogicalTensorBytes,
      physicalResidentBytes: result.metrics.workingPayloadBytes,
      containerBytes: nil,
      storageSchema: "quantem.gpu.indexed-resident-integer/v1",
      losslessExact: true,
      scanBin: provenance.scanBin,
      detectorBin: provenance.detectorBin,
      crop: crop,
      detectorMaskCount: result.sourceAudit.badPixelIndices.count,
      detectorMaskSHA256: detectorMaskSHA256,
      detectorMaskSchema: result.sourceAudit.badPixelIndices.isEmpty
        ? nil : "quantem.gpu.detector-mask-flat-u32le/v1",
      calibrationSchema: samplingSHA256 == nil
        ? nil : "quantem.gpu.metal-4dstem-sampling-propagation/v1",
      calibrationSHA256: samplingSHA256,
      provenanceSchema: Metal4DSTEMExactBinningProvenance.currentSchema,
      provenanceSHA256: Metal4DSTEMResidentReceipt.metadataSHA256(provenance),
      sourceRawLogicalSHA256: nil,
      workingLogicalSHA256: nil,
      implementationRevision: nil
    )
    try receipt.validate()
    let productCapabilities = [
      capability(.diffractionPattern, .residentOnDemand, .exactInteger),
      capability(.brightField, .immediate, .exactInteger),
      capability(.annularBrightField, .immediate, .exactInteger),
      capability(.annularDarkField, .immediate, .exactInteger),
      capability(.meanDiffractionPattern, .immediate, .exactIntegerThenFloat32),
      capability(.total, .immediate, .exactInteger),
      capability(.centerOfMass, .immediate, .exactIntegerThenFloat32),
      capability(.differentialPhaseContrast, .immediate, .exactIntegerThenFloat32),
      capability(.integratedDifferentialPhaseContrast, .residentOnDemand, .frozenFloat32),
      capability(.fastFourierTransform, .residentOnDemand, .frozenFloat32),
    ]
    return Self(
      schema: currentSchema,
      representation: representation,
      sourceIdentitySHA256: result.nativeProductProvenance.sourceIdentitySHA256,
      scanRows: provenance.outputScanRows,
      scanColumns: provenance.outputScanColumns,
      detectorRows: provenance.outputDetectorRows,
      detectorColumns: provenance.outputDetectorColumns,
      storageSchema: "quantem.gpu.indexed-resident-integer/v1",
      workingDtype: provenance.outputDtype.rawValue,
      exactIntegerBits: provenance.outputDtype.bytesPerValue * 8,
      logicalTensorBytes: UInt64(provenance.outputScanRows)
        * UInt64(provenance.outputScanColumns)
        * UInt64(provenance.outputDetectorRows)
        * UInt64(provenance.outputDetectorColumns)
        * UInt64(provenance.outputDtype.bytesPerValue),
      completeSourceResident: true,
      residentBytes: result.metrics.workingPayloadBytes,
      residentStorageBytes: result.metrics.workingPayloadBytes,
      lossless: true,
      residentReceipt: receipt,
      products: productCapabilities
    )
  }

  /// Describe a complete lossless-packed source without widening its format.
  public static func compact(
    _ source: MetalCompactH5ResidentSource
  ) throws -> Self {
    let metadata = source.metadata
    let supportedFormat =
      (metadata.schema == "quantem.gpu.packed-detector-h5/v1"
        && metadata.sourceDtype == "uint16" && metadata.workingDtype == "uint16")
      || (metadata.schema == "quantem.gpu.packed-detector-h5/v3"
        && ["uint8", "uint16", "uint32"].contains(metadata.sourceDtype ?? "")
        && ["uint8", "uint16", "uint32"].contains(metadata.workingDtype))
    guard supportedFormat else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Lossless-packed capabilities require a supported exact unsigned-count storage schema."
      )
    }
    guard
      metadata.rawAccessMode == "exact_no_exclusions"
        || metadata.rawAccessMode == "exact_exclusion_constants"
        || metadata.rawAccessMode == "exact_retained_payload"
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Compact resident capabilities require lossless access to every source count."
      )
    }
    let shape = [
      metadata.scanRows, metadata.scanColumns,
      metadata.detectorRows, metadata.detectorColumns,
    ]
    let sourceLogicalTensorBytes = try Metal4DSTEMResidentReceipt.logicalBytes(
      shape: shape,
      bytesPerValue: metadata.sourceDtype == "uint8"
        ? 1 : (metadata.sourceDtype == "uint32" ? 4 : 2)
    )
    let workingBytesPerValue: UInt64 =
      metadata.workingDtype == "uint8" ? 1 : (metadata.workingDtype == "uint32" ? 4 : 2)
    let workingLogicalTensorBytes = try Metal4DSTEMResidentReceipt.logicalBytes(
      shape: shape, bytesPerValue: workingBytesPerValue
    )
    let receipt = Metal4DSTEMResidentReceipt(
      schema: Metal4DSTEMResidentReceipt.currentSchema,
      representation: .packed,
      sourceIdentitySHA256: metadata.sourceIdentitySHA256,
      sourceShape: shape,
      workingShape: shape,
      sourceDtype: metadata.sourceDtype ?? "",
      workingDtype: metadata.workingDtype,
      sourceLogicalTensorBytes: sourceLogicalTensorBytes,
      workingLogicalTensorBytes: workingLogicalTensorBytes,
      physicalResidentBytes: source.loadMetrics.totalResidentBytes,
      containerBytes: metadata.sourceBytes,
      storageSchema: metadata.schema,
      losslessExact: true,
      scanBin: 1,
      detectorBin: 1,
      crop: nil,
      detectorMaskCount: metadata.excludedDetectorPixels.count,
      detectorMaskSHA256: metadata.detectorMaskSHA256,
      detectorMaskSchema: metadata.detectorMaskSHA256 == nil
        ? nil : "quantem.gpu.detector-mask-identity/opaque-v1",
      calibrationSchema: metadata.detectorCalibrationSchema,
      calibrationSHA256: metadata.detectorCalibrationSHA256,
      provenanceSchema: "quantem.gpu.packed-detector-h5-manifest/v1",
      provenanceSHA256: metadata.manifestSHA256,
      sourceRawLogicalSHA256: metadata.sourceRawLogicalSHA256,
      workingLogicalSHA256: metadata.workingLogicalSHA256,
      implementationRevision: nil
    )
    try receipt.validate()
    let preparedNames = Set(
      metadata.preparedDetectorProducts?.products.map(\.name) ?? []
    )
    let hasPreparedDPC = metadata.preparedDPCMoments != nil
    let productCapabilities = [
      capability(.diffractionPattern, .residentOnDemand, .exactInteger),
      capability(
        .brightField,
        preparedNames.contains("bf") ? .immediate : .residentOnDemand,
        .exactInteger
      ),
      capability(
        .annularBrightField,
        preparedNames.contains("abf") ? .immediate : .residentOnDemand,
        .exactInteger
      ),
      capability(
        .annularDarkField,
        preparedNames.contains("adf") ? .immediate : .residentOnDemand,
        .exactInteger
      ),
      capability(
        .meanDiffractionPattern,
        .residentOnDemand,
        .exactIntegerThenFloat32
      ),
      capability(
        .total,
        hasPreparedDPC ? .immediate : .unavailable,
        .exactInteger
      ),
      capability(
        .centerOfMass,
        hasPreparedDPC ? .immediate : .unavailable,
        .exactIntegerThenFloat32
      ),
      capability(
        .differentialPhaseContrast,
        hasPreparedDPC ? .immediate : .unavailable,
        .exactIntegerThenFloat32
      ),
      capability(
        .integratedDifferentialPhaseContrast,
        hasPreparedDPC ? .residentOnDemand : .unavailable,
        .frozenFloat32
      ),
      capability(
        .fastFourierTransform,
        hasPreparedDPC ? .residentOnDemand : .unavailable,
        .frozenFloat32
      ),
    ]
    return Self(
      schema: currentSchema,
      representation: .packed,
      sourceIdentitySHA256: metadata.sourceIdentitySHA256,
      scanRows: metadata.scanRows,
      scanColumns: metadata.scanColumns,
      detectorRows: metadata.detectorRows,
      detectorColumns: metadata.detectorColumns,
      storageSchema: metadata.schema,
      workingDtype: metadata.workingDtype,
      exactIntegerBits: Int(workingBytesPerValue * 8),
      logicalTensorBytes: workingLogicalTensorBytes,
      completeSourceResident: true,
      residentBytes: source.loadMetrics.totalResidentBytes,
      residentStorageBytes: source.loadMetrics.totalResidentBytes,
      lossless: true,
      residentReceipt: receipt,
      products: productCapabilities
    )
  }

  private static func capability(
    _ product: Metal4DSTEMResidentProduct,
    _ availability: Metal4DSTEMProductAvailability,
    _ numerics: Metal4DSTEMProductNumerics
  ) -> Metal4DSTEMResidentProductCapability {
    Metal4DSTEMResidentProductCapability(
      product: product,
      availability: availability,
      numerics: numerics
    )
  }
}

/// Float display products derived only after exact integer accumulation.
public struct Metal4DSTEMCenteredDPC: Equatable, Sendable {
  public let row: [Float]
  public let column: [Float]

  public init(row: [Float], column: [Float]) {
    self.row = row
    self.column = column
  }
}

extension Metal4DSTEMExactProducts {
  /// Return a mean diffraction pattern from the exact detector sum.
  public func meanDiffractionPattern(frameCount: Int) throws -> [Float] {
    guard frameCount > 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Mean diffraction requires a positive complete-frame count."
      )
    }
    return detectorSum.map { Float(Double($0) / Double(frameCount)) }
  }

  /// Return centered row/column CoM maps from exact totals and moments.
  public func centeredDPC(
    scanRows: Int,
    scanColumns: Int,
    detectorRows: Int,
    detectorColumns: Int
  ) throws -> Metal4DSTEMCenteredDPC {
    let scanCountResult = scanRows.multipliedReportingOverflow(by: scanColumns)
    guard scanRows > 0, scanColumns > 0, detectorRows > 0, detectorColumns > 0,
      !scanCountResult.overflow,
      total.count == scanCountResult.partialValue,
      detectorRowMoment.count == total.count,
      detectorColumnMoment.count == total.count
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Centered DPC requires complete total and moment maps matching the scan shape."
      )
    }
    var row = [Double](repeating: 0, count: total.count)
    var column = [Double](repeating: 0, count: total.count)
    for index in total.indices {
      let count = total[index]
      let rowLimit = count.multipliedReportingOverflow(by: UInt64(detectorRows - 1))
      let columnLimit = count.multipliedReportingOverflow(
        by: UInt64(detectorColumns - 1)
      )
      guard !rowLimit.overflow, !columnLimit.overflow,
        detectorRowMoment[index] <= rowLimit.partialValue,
        detectorColumnMoment[index] <= columnLimit.partialValue,
        count != 0
          || (detectorRowMoment[index] == 0 && detectorColumnMoment[index] == 0)
      else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Centered DPC moments violate the exact detector bounds at scan index \(index)."
        )
      }
      if count > 0 {
        row[index] = Double(detectorRowMoment[index]) / Double(count)
        column[index] = Double(detectorColumnMoment[index]) / Double(count)
      }
    }
    let divisor = Double(total.count)
    let rowMean = row.reduce(0, +) / divisor
    let columnMean = column.reduce(0, +) / divisor
    return Metal4DSTEMCenteredDPC(
      row: row.map { Float($0 - rowMean) },
      column: column.map { Float($0 - columnMean) }
    )
  }
}

/// Stable publication milestones. Only consumers can acknowledge presentation.
public enum Metal4DSTEMPublicationMilestone: String, Codable, Sendable {
  case requested
  case sourceAdmitted = "source-admitted"
  case residentReady = "resident-ready"
  case firstResidentPresent = "first-resident-present"
  case supersededRejected = "superseded-rejected"
  case cancelled
  case failed
  case deviceLost = "device-lost"
  case recoveryReady = "recovery-ready"
}

/// Non-interchangeable load, reopen, switching, and presentation timings.
public enum Metal4DSTEMTimingBoundary: String, Codable, Sendable {
  case coldArbitraryToResidentReady = "cold-arbitrary-to-resident-ready"
  case coldArbitraryToFirstPresent = "cold-arbitrary-to-first-resident-present"
  case preparedCreation = "prepared-creation"
  case warmReopenToResidentReady = "warm-reopen-to-resident-ready"
  case warmReopenToFirstPresent = "warm-reopen-to-first-resident-present"
  case preparedReopenToResidentReady = "prepared-reopen-to-resident-ready"
  case preparedReopenToFirstPresent = "prepared-reopen-to-first-resident-present"
  case cacheReopenToResidentReady = "cache-reopen-to-resident-ready"
  case cacheReopenToFirstPresent = "cache-reopen-to-first-resident-present"
  case exactSwitchToResidentReady = "exact-switch-to-resident-ready"
  case exactSwitchToFirstPresent = "exact-switch-to-first-resident-present"
}

/// Counter snapshot attached to a publication event.
public struct Metal4DSTEMPublicationCounters: Codable, Equatable, Sendable {
  public let sourceBytes: UInt64?
  public let residentBytes: UInt64?
  public let processRSSBytes: UInt64?
  public let peakProcessRSSBytes: UInt64?
  public let compressedMemoryBytes: UInt64?
  public let swapBytes: UInt64?
  public let deviceAllocatedBytes: UInt64?
  public let peakDeviceAllocatedBytes: UInt64?
  public let storageReadBytes: UInt64?
  public let uploadBytes: UInt64?
  public let readbackBytes: UInt64?
  public let synchronizationCount: UInt64?

  public init(
    sourceBytes: UInt64? = nil,
    residentBytes: UInt64? = nil,
    processRSSBytes: UInt64? = nil,
    peakProcessRSSBytes: UInt64? = nil,
    compressedMemoryBytes: UInt64? = nil,
    swapBytes: UInt64? = nil,
    deviceAllocatedBytes: UInt64? = nil,
    peakDeviceAllocatedBytes: UInt64? = nil,
    storageReadBytes: UInt64? = nil,
    uploadBytes: UInt64? = nil,
    readbackBytes: UInt64? = nil,
    synchronizationCount: UInt64? = nil
  ) {
    self.sourceBytes = sourceBytes
    self.residentBytes = residentBytes
    self.processRSSBytes = processRSSBytes
    self.peakProcessRSSBytes = peakProcessRSSBytes
    self.compressedMemoryBytes = compressedMemoryBytes
    self.swapBytes = swapBytes
    self.deviceAllocatedBytes = deviceAllocatedBytes
    self.peakDeviceAllocatedBytes = peakDeviceAllocatedBytes
    self.storageReadBytes = storageReadBytes
    self.uploadBytes = uploadBytes
    self.readbackBytes = readbackBytes
    self.synchronizationCount = synchronizationCount
  }
}

public struct Metal4DSTEMPublicationEvent: Codable, Equatable, Sendable {
  public static let currentSchema = "quantem.gpu.apple-4dstem-publication/v2"

  public let schema: String
  public let generation: UInt64
  public let sourceIdentitySHA256: String
  public let representation: Metal4DSTEMResidentRepresentation
  public let milestone: Metal4DSTEMPublicationMilestone
  public let monotonicNanoseconds: UInt64
  public let counters: Metal4DSTEMPublicationCounters
  public let detail: String?
}

/// Thread-safe latest-wins publication and signpost recorder.
///
/// Loading code records `residentReady`. The consumer records
/// `firstResidentPresent` only after the exact generation was actually drawn.
/// A stale generation is retained as `supersededRejected` evidence and cannot
/// become resident-ready or presented.
public final class Metal4DSTEMPublicationRecorder: @unchecked Sendable {
  private enum State {
    case active
    case ready
    case presented
    case lost
    case terminal
  }

  private struct GenerationState {
    let sourceIdentitySHA256: String
    let representation: Metal4DSTEMResidentRepresentation
    var state: State
  }

  private let lock = NSLock()
  private let log: OSLog?
  private var latestGeneration: UInt64?
  private var generationState: GenerationState?
  private var sourceByGeneration: [UInt64: GenerationState] = [:]
  private var recordedEvents: [Metal4DSTEMPublicationEvent] = []

  public init(signpostsEnabled: Bool = true) {
    log =
      signpostsEnabled
      ? OSLog(
        subsystem: "org.ophusgroup.quantem.gpu",
        category: "4dstem-publication"
      ) : nil
  }

  @discardableResult
  public func begin(
    generation: UInt64,
    sourceIdentitySHA256: String,
    representation: Metal4DSTEMResidentRepresentation,
    counters: Metal4DSTEMPublicationCounters = .init()
  ) throws -> Bool {
    guard isPublicationSHA256(sourceIdentitySHA256) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Publication requires a lowercase source SHA-256 identity."
      )
    }
    lock.lock()
    defer { lock.unlock() }
    if let latestGeneration, generation <= latestGeneration {
      append(
        generation: generation,
        sourceIdentitySHA256: sourceIdentitySHA256,
        representation: representation,
        milestone: .supersededRejected,
        counters: counters,
        detail: "generation is not newer than the active request"
      )
      return false
    }
    latestGeneration = generation
    generationState = GenerationState(
      sourceIdentitySHA256: sourceIdentitySHA256,
      representation: representation,
      state: .active
    )
    sourceByGeneration[generation] = generationState
    append(
      generation: generation,
      sourceIdentitySHA256: sourceIdentitySHA256,
      representation: representation,
      milestone: .requested,
      counters: counters,
      detail: nil
    )
    return true
  }

  /// Record a milestone only when the generation remains current.
  @discardableResult
  public func record(
    generation: UInt64,
    milestone: Metal4DSTEMPublicationMilestone,
    counters: Metal4DSTEMPublicationCounters = .init(),
    detail: String? = nil
  ) throws -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard generation == latestGeneration, var current = generationState else {
      if let rejected = sourceByGeneration[generation] ?? generationState {
        append(
          generation: generation,
          sourceIdentitySHA256: rejected.sourceIdentitySHA256,
          representation: rejected.representation,
          milestone: .supersededRejected,
          counters: counters,
          detail: detail ?? "generation was superseded before publication"
        )
      }
      return false
    }
    guard milestone != .requested, milestone != .supersededRejected else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Use begin for requested generations; stale rejection is recorder-owned."
      )
    }
    let allowed: Bool
    switch milestone {
    case .sourceAdmitted:
      allowed = current.state == .active
    case .residentReady:
      allowed = current.state == .active
      if allowed { current.state = .ready }
    case .firstResidentPresent:
      allowed = current.state == .ready || current.state == .presented
      if allowed { current.state = .presented }
    case .deviceLost:
      allowed = current.state != .terminal
      if allowed { current.state = .lost }
    case .recoveryReady:
      allowed = current.state == .lost
      if allowed { current.state = .ready }
    case .cancelled, .failed:
      allowed = current.state != .terminal
      if allowed { current.state = .terminal }
    case .requested, .supersededRejected:
      allowed = false
    }
    guard allowed else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Publication milestone \(milestone.rawValue) is invalid for the current generation state."
      )
    }
    generationState = current
    sourceByGeneration[generation] = current
    append(
      generation: generation,
      sourceIdentitySHA256: current.sourceIdentitySHA256,
      representation: current.representation,
      milestone: milestone,
      counters: counters,
      detail: detail
    )
    return true
  }

  public func events() -> [Metal4DSTEMPublicationEvent] {
    lock.lock()
    defer { lock.unlock() }
    return recordedEvents
  }

  private func append(
    generation: UInt64,
    sourceIdentitySHA256: String,
    representation: Metal4DSTEMResidentRepresentation,
    milestone: Metal4DSTEMPublicationMilestone,
    counters: Metal4DSTEMPublicationCounters,
    detail: String?
  ) {
    let event = Metal4DSTEMPublicationEvent(
      schema: Metal4DSTEMPublicationEvent.currentSchema,
      generation: generation,
      sourceIdentitySHA256: sourceIdentitySHA256,
      representation: representation,
      milestone: milestone,
      monotonicNanoseconds: DispatchTime.now().uptimeNanoseconds,
      counters: counters,
      detail: detail
    )
    recordedEvents.append(event)
    if let log {
      os_signpost(
        .event,
        log: log,
        name: "QuantEM4DSTEMPublication",
        "generation=%llu milestone=%{public}s source=%{public}s representation=%{public}s",
        generation,
        milestone.rawValue,
        sourceIdentitySHA256,
        representation.rawValue
      )
    }
  }
}

private func isPublicationSHA256(_ value: String) -> Bool {
  value.utf8.count == 64
    && value.utf8.allSatisfy {
      (48...57).contains($0) || (97...102).contains($0)
    }
}

/// Nearest-rank latency summary for one explicitly named timing boundary.
public struct Metal4DSTEMTimingSummary: Codable, Equatable, Sendable {
  public let sampleCount: Int
  public let p50Seconds: Double
  public let p95Seconds: Double
  public let maximumSeconds: Double

  public init(samplesSeconds: [Double]) throws {
    guard !samplesSeconds.isEmpty,
      samplesSeconds.allSatisfy({ $0.isFinite && $0 >= 0 })
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "A timing summary requires one or more finite nonnegative samples."
      )
    }
    let sorted = samplesSeconds.sorted()
    func percentile(_ probability: Double) -> Double {
      let rank = max(1, Int(ceil(probability * Double(sorted.count))))
      return sorted[min(rank - 1, sorted.count - 1)]
    }
    sampleCount = sorted.count
    p50Seconds = percentile(0.50)
    p95Seconds = percentile(0.95)
    maximumSeconds = sorted.last!
  }
}
