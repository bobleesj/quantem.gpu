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
  /// SHA-256 of the archive manifest; the app keys its derived caches by it.
  public var archiveCheckpointSHA256: String { resident.archiveCheckpointSHA256 }

  /// Exact tile layouts the detector index can be built in. Both hold only
  /// lossless tile sums. `uniform8` holds 576 exact 8x8 tiles instead of 100
  /// mixed ones: several times the memory, and far fewer residual columns when
  /// a large annulus is recomputed from the index.
  public enum DetectorIndexLayout: String, Sendable {
    case centerFine
    case uniform8

    var tileLayout: TANSExactTileIndex.Layout { self == .uniform8 ? .uniform8 : .centerFine }

    /// Planner price of one exact tile add. With 576 small tiles the flat 0.5
    /// buys tiles that cost more than the columns they remove at mid-size
    /// steps; 2.0 was equal or faster at every measured step (1-16 px) over 66
    /// acquisitions. The price only ranks exact decompositions.
    var plannerTileCost: Double { self == .uniform8 ? 2.0 : 0.5 }
  }

  /// Outcome of `prepareDetectorIndex(maximumIndexBytes:cacheDirectory:)`.
  public struct DetectorIndexPreparation: Sendable {
    public let restoredFromCache: Bool
    public let seconds: Double
    public let cacheWritten: Bool
    public let cacheError: String?
  }

  /// Prepare the exact detector index, restoring it from `cacheDirectory` when
  /// a matching, digest-verified cache exists and writing one after a fresh
  /// build otherwise. Restore failures never skip the build; write failures are
  /// reported, not thrown. Same serialization rules as the plain preparation.
  public func prepareDetectorIndex(
    maximumIndexBytes: UInt64, cacheDirectory: URL?, layout: DetectorIndexLayout = .centerFine
  ) throws -> DetectorIndexPreparation {
    let started = ProcessInfo.processInfo.systemUptime
    var cacheError: String?
    if let cacheDirectory {
      do {
        try resident.importExperimentalTileIndex(
          from: cacheDirectory, maximumIndexBytes: maximumIndexBytes, layout: layout.tileLayout)
        resident.experimentalTileCost = layout.plannerTileCost
        enableIndexQueries()
        return DetectorIndexPreparation(
          restoredFromCache: true, seconds: ProcessInfo.processInfo.systemUptime - started,
          cacheWritten: false, cacheError: nil)
      } catch {
        cacheError = "restore: \(error)"
      }
    }
    try prepareDetectorIndex(maximumIndexBytes: maximumIndexBytes, layout: layout)
    var written = false
    if let cacheDirectory {
      do {
        try resident.exportExperimentalTileIndex(to: cacheDirectory, layout: layout.tileLayout)
        written = true
      } catch {
        cacheError = (cacheError.map { $0 + "; " } ?? "") + "write: \(error)"
      }
    }
    return DetectorIndexPreparation(
      restoredFromCache: false, seconds: ProcessInfo.processInfo.systemUptime - started,
      cacheWritten: written, cacheError: cacheError)
  }

  private func enableIndexQueries() {
    resident.experimentalDetectorStreamsPerLane = 32
    resident.experimentalUseTileIndex = true
  }

  /// Build a bounded exact detector-query index without changing source counts.
  /// This one-time preparation includes exact GPU packing/roundtrip checks and
  /// must run on the caller's serialized worker, never the main/UI thread.
  /// The caller admits the extra index and at least 1 GiB preparation headroom.
  /// A failure preserves the prior resident/index and never enables a raw path.
  public func prepareDetectorIndex(
    maximumIndexBytes: UInt64, layout: DetectorIndexLayout = .centerFine
  ) throws {
    try resident.prepareExperimentalTileIndex(
      maximumIndexBytes: maximumIndexBytes, layout: layout.tileLayout)
    resident.experimentalTileCost = layout.plannerTileCost
    resident.experimentalDetectorStreamsPerLane = 32
    resident.experimentalUseTileIndex = true
    #if QUANTEM_ENTROPY_SUBMISSION_OVERLAP
      // Local QA build only: keep submission mechanics out of the scientist API.
      // Exact all-acquisition completion remains the return/publication boundary.
      resident.experimentalDetectorRecordsPerCommand = 64
    #endif
    #if QUANTEM_ENTROPY_MIXED_TAILS
      // Local all66 QA only. No source/geometry/precision or stable API change.
      resident.experimentalMixedModelTails = true
      resident.experimentalMixedTailSavingsDivisor = 8
      resident.experimentalSeparateMixedDispatches = true
      resident.experimentalMixedOnlySpecialization = true
    #endif
  }

  /// Interactive residuals are small and often span many entropy models, so
  /// fixed 32-lane model groups run mostly padded lanes. Mixed-model tails pack
  /// the partial groups of all models into shared groups whose lanes look up
  /// their own table. Exactness is unchanged (same integer contributions);
  /// only launch geometry changes. Requires the prepared detector index path.
  ///
  /// `chooseCheaperBase` additionally compares, for every query, the exact
  /// decomposition that continues from the previous image against the one that
  /// recomputes from the tile index, and uses whichever leaves less entropy
  /// work. Both are exact and produce identical counts; only which one is
  /// dispatched changes. It matters when the geometry moved far since the last
  /// publication, where continuing from the previous image can cost more than
  /// starting again: measured 4,534 instead of 5,801 residual columns and
  /// 165 ms instead of 209 ms for a 16-pixel annulus step over 66 acquisitions.
  public func configureInteractiveGrouping(
    mixedModelTails: Bool, chooseCheaperBase: Bool = false
  ) {
    resident.experimentalMixedModelTails = mixedModelTails
    // No savings gate: an all-acquisition query spreads its residual over many
    // entropy models, and the 1/8 gate this used to apply left the regrouping
    // switched off for nearly every such frame (measured 0.3% of groups
    // regrouped). Regrouping every partial group is a launch-geometry change
    // only; the integer contributions per lane are identical.
    resident.experimentalMixedTailSavingsDivisor = 0
    resident.experimentalSeparateMixedDispatches = mixedModelTails
    resident.experimentalMixedOnlySpecialization = mixedModelTails
    // Prefetching the next decoder entry overlaps the table read with the
    // current symbol's arithmetic. Same table, same symbols, same order.
    resident.experimentalPrefetchDecoderEntry = mixedModelTails
    resident.experimentalPlanAfterIndex = chooseCheaperBase
    // Interactive queries use the packet-owner kernel: one SIMD group per
    // (record, packet) decodes every dense group of the record and writes each
    // output scan once, with no device atomics. Same streams, symbols and
    // integer contributions; only the addition order changes (exact in uint32).
    // Cross-checked bit for bit against the shared-model kernel on all 66
    // acquisitions. QUANTEM_TANS_PACKET_OWNER=0 restores the shared-model
    // kernel. Later atlas fields are built with whichever kernel is selected;
    // the index built before this call used the shared-model kernel unless
    // QUANTEM_TANS_PACKET_OWNER=1.
    resident.experimentalPacketOwnerKernel =
      ProcessInfo.processInfo.environment["QUANTEM_TANS_PACKET_OWNER"] != "0"
  }

  /// Exact annulus atlas (experimental). Complete exact images of caller-chosen
  /// masks, packed losslessly and held beside the source. A query may start
  /// from the stored image whose mask is closest to the request and decode only
  /// the pixels where the two masks differ; the identity uses mask bytes only,
  /// so every published count is exact for any request. A fast drag then costs
  /// about the same per frame whatever its speed (measured over 66
  /// acquisitions at 2.5-3.3 px steps: 12.7-16.7 ms GPU instead of 36-54 ms).
  ///
  /// `beginDetectorAtlas` reserves an explicit byte budget (the caller admits
  /// it, plus 1 GiB headroom) and drops any prior atlas. Requires the prepared
  /// detector index. Same serialization rules as every other call.
  public func beginDetectorAtlas(maximumBytes: UInt64) throws {
    try resident.beginExperimentalAtlas(maximumBytes: maximumBytes)
    resident.experimentalUseAtlas = true
  }

  /// Add the exact image of one mask for every loaded acquisition. It is
  /// computed into temporary outputs: no returned image, and no seed a later
  /// query continues from, changes. One field is one unseeded exact query plus
  /// packing and a bit-for-bit audit, so callers add fields while idle.
  public func appendDetectorAtlasField(mask: [UInt8]) throws {
    try resident.appendExperimentalAtlasField(mask: mask)
  }

  public var detectorAtlasFieldCount: Int { resident.experimentalAtlasFieldCount }
  /// 1 for detector pixels stored as dense (tANS) columns, 0 for sparse-event
  /// columns: planning metadata for predicting decode cost. A residual's dense
  /// columns set its cost; up to 32 of them decode as one model group per record.
  public var detectorDenseMask: [UInt8] { resident.detectorDenseMask }
  public var detectorAtlasBytes: Int { resident.experimentalAtlasBytes }
  /// The atlas field the last query started from, or nil.
  public var lastDetectorAtlasField: Int? { resident.lastDetectorAtlasField }

  /// Independent UInt32 2D image buffers in acquisition order; exact UInt16 counts.
  /// Waits for Metal completion, never downloads images or retains a raw 4D volume.
  public func diffractionImages(scanRow: Int, scanColumn: Int) throws -> [MTLBuffer] {
    try resident.diffractionImages(scanRow: scanRow, scanColumn: scanColumn)
  }

  /// Exact diffraction for a loaded subset, in requested order. Intended for
  /// interactive inspection of one acquisition while the rest catch up later;
  /// it never changes counts, coverage or the retained series.
  public func diffractionImages(
    scanRow: Int, scanColumn: Int, selectedAcquisitions: [Int]
  ) throws -> [MTLBuffer] {
    try resident.diffractionImages(
      scanRow: scanRow, scanColumn: scanColumn, selectedAcquisitions: selectedAcquisitions)
  }

  /// Exact UInt32 virtual-detector images for the supplied native detector mask.
  /// The mask is caller-owned and must explicitly encode validity and aperture
  /// policy. Results stay on Metal; no raw 4D resident or CPU readback is made.
  /// Delta seeds are kept per acquisition, so single-acquisition queries and
  /// later subset or full catch-up queries each continue from their own last
  /// completed image. Each acquisition rotates through three complete output
  /// images: a returned buffer is not written again until that acquisition
  /// has completed two further queries.
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

  /// A submitted exact detector query (see `submitDetectorImages`).
  public final class DetectorSubmission: @unchecked Sendable {
    let pending: MetalTANSResidentSeries.PendingDetectorQuery
    /// Requested acquisitions, in the order their images will be returned.
    public let acquisitions: [Int]

    init(_ pending: MetalTANSResidentSeries.PendingDetectorQuery) {
      self.pending = pending
      acquisitions = pending.outputAcquisitionIndices
    }

    /// Submission order within this series.
    public var sequence: UInt64 { pending.sequence }

    /// Calls `handler` once, on an arbitrary thread, when every command of
    /// the query and of every query submitted before it has completed on the
    /// GPU, successfully or not (at once if they already have). After it,
    /// `finishDetectorImages` never blocks. Completion is not verification:
    /// `finishDetectorImages` returns the images or the failure.
    public func whenGPUCompleted(_ handler: @escaping @Sendable () -> Void) {
      pending.completion.notify(handler)
    }
  }

  /// Complete images of one finished detector query, in requested order, and
  /// that query's own diagnostics.
  public struct DetectorQueryResult: @unchecked Sendable {
    public let images: [MTLBuffer]
    /// First-command GPU-start to final-command GPU-end span (see
    /// `lastDetectorGPUSeconds`).
    public let gpuSeconds: Double
    public let decodedColumns: Int
    public let usedPrevious: Bool
    public let atlasField: Int?
    public let commandTiming: [String: Double]
  }

  /// Unfinished pipelined queries allowed at once. Two keep every rollback
  /// exact within the three-slot output ring of each acquisition.
  public static var maximumDetectorQueriesInFlight: Int {
    MetalTANSResidentSeries.maximumDetectorQueriesInFlight
  }

  /// Pipelined detector queries submitted and not yet finished.
  public var detectorQueriesInFlight: Int { resident.detectorQueriesInFlight }

  /// Pipelined exact interactive subset: plans, encodes and commits the query
  /// and returns without waiting, so the caller can prepare the next one while
  /// the GPU works. Counts are identical to `detectorImages`: the next query's
  /// delta seed is this query's mask and images, and the single detector queue
  /// orders its reads after these writes. Finish every submission, in any
  /// order, with `finishDetectorImages`; at most
  /// `maximumDetectorQueriesInFlight` may be unfinished. A returned image is
  /// rewritten by the third later query of its acquisition, so read or copy it
  /// before submitting that query. `detectorImages` and the index build write
  /// the output ring outside that limit, so they throw while a submission is
  /// unfinished; atlas calls wait for and verify submitted queries.
  public func submitDetectorImages(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool = false, selectedAcquisitions: [Int]
  ) throws -> DetectorSubmission {
    DetectorSubmission(
      try resident.submitDetectorImages(
        mask: mask, maximumAdditionalBytes: maximumAdditionalBytes,
        rebase: rebase, selectedAcquisitions: selectedAcquisitions))
  }

  /// Wait for a submitted query (and every earlier one; this blocks only if
  /// the GPU has not completed them, see `whenGPUCompleted`), verify it and
  /// return its complete images. If it or an earlier query it was seeded from
  /// failed, this throws and every affected seed returns to its prior state;
  /// images returned by earlier finished queries are untouched.
  public func finishDetectorImages(_ submission: DetectorSubmission) throws
    -> DetectorQueryResult
  {
    let result = try resident.finishDetectorImages(submission.pending)
    return DetectorQueryResult(
      images: result.images, gpuSeconds: result.gpuSeconds,
      decodedColumns: result.decodedColumns, usedPrevious: result.usedPrevious,
      atlasField: result.atlasField, commandTiming: result.timing)
  }

  /// Test-only failure injection: listed submissions are treated as failed on
  /// the GPU when verified.
  var experimentalFailedDetectorQuerySequences: Set<UInt64> {
    get { resident.experimentalFailedDetectorQuerySequences }
    set { resident.experimentalFailedDetectorQuerySequences = newValue }
  }

  /// Test-only failure injection: a query's batch throws after this many of
  /// its commands were committed.
  var experimentalFailAfterSubmittedCommands: Int? {
    get { resident.experimentalFailAfterSubmittedCommands }
    set { resident.experimentalFailAfterSubmittedCommands = newValue }
  }

  /// Test-only: entropy records per detector command (0: one per seed group).
  var experimentalDetectorRecordsPerCommand: Int {
    get { resident.experimentalDetectorRecordsPerCommand }
    set { resident.experimentalDetectorRecordsPerCommand = newValue }
  }

  /// Test-only: the exact mask an acquisition's next query starts from.
  func experimentalDetectorSeedMask(acquisition: Int) -> [UInt8]? {
    resident.experimentalDetectorSeedMask(acquisition: acquisition)
  }

  public var validDetectorMask: [UInt8] { resident.validDetectorMask }
  /// First-command GPU-start to final-command GPU-end span. Driver/scheduling
  /// gaps can be included; this is not an isolated shader-active timer or FPS.
  public var lastDetectorGPUSeconds: Double { resident.lastDetectorGPUSeconds }
  /// Diagnostic stages of the last completed query, in milliseconds except
  /// explicit count/flag keys. Command-interval sums are not active-cycle
  /// counters; the first-to-last GPU span can include scheduling gaps.
  public var lastDetectorCommandTiming: [String: Double] { resident.lastDetectorCommandTiming }
  public var lastDetectorDecodedColumns: Int { resident.lastDetectorDecodedColumns }
  public var lastDetectorUsedPrevious: Bool { resident.lastDetectorUsedPrevious }

  public func release() { resident.releaseResidentStorage() }
}
