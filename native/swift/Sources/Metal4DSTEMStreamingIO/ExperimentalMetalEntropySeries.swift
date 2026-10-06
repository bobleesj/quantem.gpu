import Foundation
import Metal

/// Experimental native Metal consumer for the fixed exact tANS archive profile.
/// As of 2026-09-08 this SPI is not a stable API, general encoder, or release.
/// SHA-256 checks establish archive integrity, not trusted authorship.
/// Callers must serialize load, queries and release off the UI thread, and own
/// memory admission, source provenance, calibration and publication policy.
@_spi(EntropySeriesPrototype)
public final class ExperimentalMetalEntropySeries {
  private let resident: MetalTANSResidentSeries

  public init(
    directory: URL, acquisitions: [Int], device: MTLDevice,
    maximumAdditionalBytes: UInt64
  ) throws {
    resident = try MetalTANSResidentSeries(
      directory: directory, acquisitions: acquisitions,
      device: device, maximumAdditionalBytes: maximumAdditionalBytes)
  }

  public var residentBytes: Int { resident.residentBytes }
  public var loadSeconds: Double { resident.loadSeconds }
  public var shape: [Int] { resident.shape }
  public var acquisitionIndices: [Int] { resident.acquisitionIndices }
  public var detectorIndexBytes: Int { resident.experimentalTileIndexBytes }

  /// Build a bounded exact detector-query index without changing source counts.
  /// This one-time preparation includes exact GPU packing/roundtrip checks and
  /// must run on the caller's serialized worker, never the main/UI thread.
  /// The caller admits the extra index and at least 1 GiB preparation headroom.
  /// A failure preserves the prior resident/index and never enables a raw path.
  public func prepareDetectorIndex(maximumIndexBytes: UInt64) throws {
    try resident.prepareExperimentalTileIndex(maximumIndexBytes: maximumIndexBytes)
    resident.experimentalDetectorStreamsPerLane = 32
    resident.experimentalUseTileIndex = true
  }

  /// Independent UInt32 2D image buffers in acquisition order; exact UInt16 counts.
  /// Waits for Metal completion, never downloads images or retains a raw 4D volume.
  public func diffractionImages(scanRow: Int, scanColumn: Int) throws -> [MTLBuffer] {
    try resident.diffractionImages(scanRow: scanRow, scanColumn: scanColumn)
  }

  /// Exact UInt32 virtual-detector images for the supplied native detector mask.
  /// The mask is caller-owned and must explicitly encode validity and aperture
  /// policy. Results stay on Metal; no raw 4D resident or CPU readback is made.
  public func detectorImages(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool = false
  ) throws -> [MTLBuffer] {
    try resident.detectorImages(
      mask: mask, maximumAdditionalBytes: maximumAdditionalBytes,
      rebase: rebase)
  }

  /// Exact interactive subset. The returned buffers are in the requested
  /// acquisition order and preserve the same integer/count contract.
  /// Indices must be unique and belong to the loaded acquisition set.
  public func detectorImages(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool = false, selectedAcquisitions: [Int]
  ) throws -> [MTLBuffer] {
    try resident.detectorImages(
      mask: mask, maximumAdditionalBytes: maximumAdditionalBytes,
      rebase: rebase, selectedAcquisitions: selectedAcquisitions)
  }

  public var validDetectorMask: [UInt8] { resident.validDetectorMask }
  public var lastDetectorGPUSeconds: Double { resident.lastDetectorGPUSeconds }
  public var lastDetectorDecodedColumns: Int { resident.lastDetectorDecodedColumns }
  public var lastDetectorUsedPrevious: Bool { resident.lastDetectorUsedPrevious }

  public func release() { resident.releaseResidentStorage() }
}
