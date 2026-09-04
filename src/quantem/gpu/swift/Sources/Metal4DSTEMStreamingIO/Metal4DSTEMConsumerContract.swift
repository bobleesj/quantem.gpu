import Foundation
import os

/// Exact Apple representation that owns a complete resident 4D-STEM generation.
public enum Metal4DSTEMResidentRepresentation: String, Codable, Sendable {
  case compactQGIXV3UInt8 = "compact-qgix-v3-uint8"
  case indexedResidentInteger = "indexed-resident-integer"
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
  public static let currentSchema = "quantem.gpu.apple-4dstem-resident-capabilities/v2"

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
  public let products: [Metal4DSTEMResidentProductCapability]

  public var fullInteractiveResident: Bool {
    completeSourceResident
      && lossless
      && logicalTensorBytes > 0
      && residentStorageBytes > 0
      && Set(products.map(\.product)) == Set(Metal4DSTEMResidentProduct.allCases)
      && products.allSatisfy { $0.availability != .unavailable }
  }

  /// Describe an exact indexed/sharded uint16 result without taking ownership.
  public static func indexed(
    _ result: Metal4DSTEMIndexedBinnedLoadResult
  ) -> Self {
    let provenance = result.binningProvenance
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
      representation: .indexedResidentInteger,
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
      products: productCapabilities
    )
  }

  /// Describe a complete QGIX v3 uint8 source without widening its format.
  public static func compact(
    _ source: MetalCompactH5ResidentSource
  ) throws -> Self {
    let metadata = source.metadata
    guard metadata.schema == "quantem.gpu.packed-detector-h5/v3" else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Compact resident capabilities require exact QGIX v3 uint8 source semantics."
      )
    }
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
      representation: .compactQGIXV3UInt8,
      sourceIdentitySHA256: metadata.sourceIdentitySHA256,
      scanRows: metadata.scanRows,
      scanColumns: metadata.scanColumns,
      detectorRows: metadata.detectorRows,
      detectorColumns: metadata.detectorColumns,
      storageSchema: "quantem.gpu.packed-detector-h5/v3",
      workingDtype: "uint8",
      exactIntegerBits: 8,
      logicalTensorBytes: UInt64(metadata.scanRows)
        * UInt64(metadata.scanColumns)
        * UInt64(metadata.detectorRows)
        * UInt64(metadata.detectorColumns),
      completeSourceResident: true,
      residentBytes: source.loadMetrics.totalResidentBytes,
      residentStorageBytes: source.loadMetrics.totalResidentBytes,
      lossless: true,
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
  public static let currentSchema = "quantem.gpu.apple-4dstem-publication/v1"

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
