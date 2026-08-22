import Foundation

/// A validated `QH5IDX01` sidecar and its block-offset words.
public struct NativeQH5Index: Sendable {
  public let metadata: NativeQH5IndexMetadata
  public let metadataWords: [UInt32]

  /// Open an index only when it still describes the exact source file.
  public static func open(
    sourceURL: URL,
    indexURL: URL
  ) throws -> Self {
    let data = try Data(contentsOf: indexURL, options: .mappedIfSafe)
    guard data.count >= 16, data.prefix(8) == qh5IndexMagic else {
      throw Native4DSTEMIOError.invalidData(
        "Missing or invalid QH5 index: \(indexURL.lastPathComponent)"
      )
    }
    let jsonBytes = Int(readLittleEndianUInt32(data, at: 8))
    let wordCount = Int(readLittleEndianUInt32(data, at: 12))
    let jsonEnd = 16.addingReportingOverflow(jsonBytes)
    guard !jsonEnd.overflow else {
      throw Native4DSTEMIOError.invalidData(
        "QH5 index JSON length overflows: \(indexURL.lastPathComponent)"
      )
    }
    let alignedJSONEnd = jsonEnd.partialValue.addingReportingOverflow(3)
    guard !alignedJSONEnd.overflow else {
      throw Native4DSTEMIOError.invalidData(
        "QH5 index alignment overflows: \(indexURL.lastPathComponent)"
      )
    }
    let binaryStart = alignedJSONEnd.partialValue & ~3
    let binaryByteCount = wordCount.multipliedReportingOverflow(
      by: MemoryLayout<UInt32>.stride
    )
    let binaryEnd = binaryStart.addingReportingOverflow(binaryByteCount.partialValue)
    guard jsonBytes >= 0, wordCount >= 0, !binaryByteCount.overflow,
      !binaryEnd.overflow, jsonEnd.partialValue <= data.count,
      binaryStart >= jsonEnd.partialValue, binaryEnd.partialValue == data.count
    else {
      throw Native4DSTEMIOError.invalidData(
        "Truncated or trailing QH5 index data: \(indexURL.lastPathComponent)"
      )
    }
    let metadata = try JSONDecoder().decode(
      NativeQH5IndexMetadata.self,
      from: data.subdata(in: 16..<jsonEnd.partialValue)
    )
    let sourceIdentity = try nativeFileIdentity(for: sourceURL)
    guard metadata.sourcePath == sourceIdentity.path,
      metadata.sourceBytes == sourceIdentity.bytes,
      metadata.sourceMtimeNs == sourceIdentity.modificationNanoseconds
    else {
      throw Native4DSTEMIOError.invalidData(
        "QH5 index \(indexURL.lastPathComponent) is stale for \(sourceURL.lastPathComponent)"
      )
    }
    var words = [UInt32]()
    words.reserveCapacity(wordCount)
    for offset in stride(
      from: binaryStart,
      to: binaryEnd.partialValue,
      by: MemoryLayout<UInt32>.stride
    ) {
      words.append(readLittleEndianUInt32(data, at: offset))
    }
    try validate(metadata: metadata, words: words, indexURL: indexURL)
    return Self(metadata: metadata, metadataWords: words)
  }

  private static func validate(
    metadata: NativeQH5IndexMetadata,
    words: [UInt32],
    indexURL: URL
  ) throws {
    func invalid(_ reason: String) throws -> Never {
      throw Native4DSTEMIOError.invalidData(
        "Invalid QH5 index \(indexURL.lastPathComponent): \(reason)"
      )
    }
    guard metadata.detRows > 0, metadata.detCols > 0, metadata.nFrames > 0,
      metadata.srcDtype == "uint8" || metadata.srcDtype == "uint16",
      metadata.blockElems > 0, metadata.nBlocksPerFrame > 0,
      !metadata.chunks.isEmpty
    else { try invalid("shape, dtype, block geometry, and frame count must be positive") }
    let detectorPixels = metadata.detRows.multipliedReportingOverflow(
      by: metadata.detCols
    )
    guard !detectorPixels.overflow else { try invalid("detector geometry overflows Int") }
    let expectedBlocks = (detectorPixels.partialValue - 1) / metadata.blockElems + 1
    guard detectorPixels.partialValue >= metadata.blockElems,
      metadata.nBlocksPerFrame == expectedBlocks,
      (metadata.srcDtype == "uint8" && detectorPixels.partialValue.isMultiple(of: 32))
        || (metadata.srcDtype == "uint16" && metadata.blockElems == 4096
          && detectorPixels.partialValue.isMultiple(of: metadata.blockElems))
    else { try invalid("block geometry is incompatible with the detector shape and dtype") }
    var expectedFrame = 0
    var expectedWord = 0
    var previousChunkEnd: UInt64 = 0
    for chunk in metadata.chunks {
      let expectedWords = chunk.nFrames.multipliedReportingOverflow(
        by: metadata.nBlocksPerFrame
      )
      let expectedWordPairs = expectedWords.partialValue.multipliedReportingOverflow(by: 2)
      guard chunk.startFrame == expectedFrame, chunk.nFrames > 0,
        chunk.rangeStart < chunk.rangeEnd, chunk.rangeEnd <= metadata.sourceBytes,
        chunk.rangeStart >= previousChunkEnd,
        chunk.metaOffsetWords == expectedWord, !expectedWords.overflow,
        !expectedWordPairs.overflow, chunk.metaWords == expectedWordPairs.partialValue
      else { try invalid("chunks must cover frames and metadata words exactly once") }
      let wordStop = chunk.metaOffsetWords.addingReportingOverflow(chunk.metaWords)
      guard !wordStop.overflow, wordStop.partialValue <= words.count else {
        try invalid("chunk metadata range exceeds the index word payload")
      }
      var previousPayloadEnd = chunk.rangeStart
      for word in stride(from: chunk.metaOffsetWords, to: wordStop.partialValue, by: 2) {
        let relativeStart = UInt64(words[word])
        let compressedBytes = UInt64(words[word + 1])
        let payloadStart = chunk.rangeStart.addingReportingOverflow(relativeStart)
        let payloadEnd = payloadStart.partialValue.addingReportingOverflow(compressedBytes)
        guard compressedBytes > 0, !payloadStart.overflow, !payloadEnd.overflow,
          payloadStart.partialValue >= chunk.rangeStart,
          payloadStart.partialValue >= previousPayloadEnd,
          payloadEnd.partialValue <= chunk.rangeEnd
        else { try invalid("compressed block metadata exceeds its source range") }
        previousPayloadEnd = payloadEnd.partialValue
      }
      guard previousPayloadEnd == chunk.rangeEnd else {
        try invalid("compressed block metadata does not reach the chunk range end")
      }
      let frameStop = expectedFrame.addingReportingOverflow(chunk.nFrames)
      guard !frameStop.overflow else { try invalid("frame coverage overflows Int") }
      expectedFrame = frameStop.partialValue
      expectedWord = wordStop.partialValue
      previousChunkEnd = chunk.rangeEnd
    }
    guard expectedFrame == metadata.nFrames, expectedWord == words.count else {
      try invalid("final frame or metadata-word coverage is incomplete")
    }
  }

  private static func readLittleEndianUInt32(_ data: Data, at offset: Int) -> UInt32 {
    data.withUnsafeBytes { raw in
      raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian
    }
  }
}

/// A deterministic partition of the logical native volume into bounded frames.
public struct Native4DSTEMFrameWindowPlan: Equatable, Sendable {
  public let scanRows: Int
  public let scanColumns: Int
  public let detectorRows: Int
  public let detectorColumns: Int
  public let sourceBytesPerValue: Int
  public let maximumDecodedBytes: UInt64
  public let alignsToScanRows: Bool
  public let decodedBytesPerFrame: UInt64
  public let logicalDecodedBytes: UInt64
  public let frameRanges: [Range<Int>]

  public init(
    scanRows: Int,
    scanColumns: Int,
    detectorRows: Int,
    detectorColumns: Int,
    sourceBytesPerValue: Int,
    maximumDecodedBytes: UInt64,
    alignToScanRows: Bool = true
  ) throws {
    guard scanRows > 0, scanColumns > 0, detectorRows > 0, detectorColumns > 0,
      sourceBytesPerValue == 1 || sourceBytesPerValue == 2
    else {
      throw Native4DSTEMIOError.invalidData(
        "Indexed-source shapes must be positive and source bytes per value must be 1 or 2"
      )
    }
    let detectorPixels = try Self.product(
      UInt64(detectorRows), UInt64(detectorColumns), label: "detector pixels"
    )
    let frameBytes = try Self.product(
      detectorPixels, UInt64(sourceBytesPerValue), label: "decoded frame bytes"
    )
    let scanFrames = try Self.product(
      UInt64(scanRows), UInt64(scanColumns), label: "scan frames"
    )
    let logicalBytes = try Self.product(
      scanFrames, frameBytes, label: "logical decoded bytes"
    )
    guard maximumDecodedBytes >= frameBytes else {
      throw Native4DSTEMIOError.invalidData(
        "The decoded-window ceiling must hold at least one native detector frame (\(frameBytes) bytes)"
      )
    }
    var framesPerWindow = maximumDecodedBytes / frameBytes
    if alignToScanRows {
      guard framesPerWindow >= UInt64(scanColumns) else {
        throw Native4DSTEMIOError.invalidData(
          "A scan-row-aligned window must hold at least one complete scan row (\(frameBytes * UInt64(scanColumns)) bytes)"
        )
      }
      framesPerWindow -= framesPerWindow % UInt64(scanColumns)
    }
    framesPerWindow = min(framesPerWindow, scanFrames)
    guard let totalFrames = Int(exactly: scanFrames),
      let step = Int(exactly: framesPerWindow), step > 0
    else {
      throw Native4DSTEMIOError.invalidData(
        "Indexed-source frame geometry exceeds this process's addressable range"
      )
    }
    var ranges: [Range<Int>] = []
    ranges.reserveCapacity((totalFrames - 1) / step + 1)
    var start = 0
    while start < totalFrames {
      let remaining = totalFrames - start
      let stop = remaining > step ? start + step : totalFrames
      ranges.append(start..<stop)
      start = stop
    }
    self.scanRows = scanRows
    self.scanColumns = scanColumns
    self.detectorRows = detectorRows
    self.detectorColumns = detectorColumns
    self.sourceBytesPerValue = sourceBytesPerValue
    self.maximumDecodedBytes = maximumDecodedBytes
    self.alignsToScanRows = alignToScanRows
    decodedBytesPerFrame = frameBytes
    logicalDecodedBytes = logicalBytes
    frameRanges = ranges
  }

  private static func product(
    _ lhs: UInt64,
    _ rhs: UInt64,
    label: String
  ) throws -> UInt64 {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else {
      throw Native4DSTEMIOError.invalidData("Native 4D-STEM \(label) overflows UInt64")
    }
    return result.partialValue
  }
}

/// One source/index pair with a stable global scan-frame range.
public struct Native4DSTEMIndexedShard: Sendable {
  public let sourceURL: URL
  public let indexURL: URL
  public let index: NativeQH5Index
  public let globalFrameRange: Range<Int>
}

/// Exact index data needed to decode part of one prepared QH5 chunk.
public struct Native4DSTEMIndexedSlice: Equatable, Sendable {
  public let shardIndex: Int
  public let chunkIndex: Int
  public let globalFrameRange: Range<Int>
  /// Frame range relative to the beginning of `shardIndex`.
  public let shardFrameRange: Range<Int>
  /// Word range in that shard's `NativeQH5Index.metadataWords`.
  public let metadataWordRange: Range<Int>
  /// Complete compressed source range that backs `chunkIndex`.
  public let chunkCompressedByteRange: Range<UInt64>
}

/// One bounded decoded window. All ranges are half open and nonoverlapping.
public struct Native4DSTEMIndexedWindow: Equatable, Sendable {
  public let globalFrameRange: Range<Int>
  public let decodedBytes: UInt64
  public let slices: [Native4DSTEMIndexedSlice]
}

/// Validated indexed access to a complete logical 4D-STEM dataset.
///
/// Opening this value reads prepared index sidecars but does not map source
/// files, allocate Metal buffers, decode detector frames, or select a device
/// policy. Consumers provide an explicit transient decoded-byte ceiling.
public struct Native4DSTEMIndexedSource: Sendable {
  public let dataset: Native4DSTEMDataset
  public let shards: [Native4DSTEMIndexedShard]
  public let logicalFrameCount: Int
  public let sourceBytesPerValue: Int
  public let decodedBytesPerFrame: UInt64
  public let logicalDecodedBytes: UInt64

  public static func open(dataset: Native4DSTEMDataset) throws -> Self {
    guard dataset.dataFiles.count == dataset.indexFiles.count,
      !dataset.dataFiles.isEmpty
    else {
      throw Native4DSTEMIOError.invalidData(
        "\(dataset.label) requires one prepared QH5 index for every data file"
      )
    }
    let sourceURLs = dataset.dataFiles.map {
      nativeCanonicalURL(URL(fileURLWithPath: $0))
    }
    let indexURLs = dataset.indexFiles.map {
      nativeCanonicalURL(URL(fileURLWithPath: $0))
    }
    guard Set(sourceURLs.map(\.path)).count == sourceURLs.count,
      Set(indexURLs.map(\.path)).count == indexURLs.count
    else {
      throw Native4DSTEMIOError.invalidData(
        "\(dataset.label) repeats a source or prepared-index path"
      )
    }
    let sourceBytesPerValue: Int
    switch dataset.sourceDtype {
    case "uint8": sourceBytesPerValue = 1
    case "uint16": sourceBytesPerValue = 2
    default:
      throw Native4DSTEMIOError.invalidData(
        "\(dataset.label) source dtype \(dataset.sourceDtype) is not indexed-load compatible"
      )
    }
    let plan = try Native4DSTEMFrameWindowPlan(
      scanRows: dataset.scanRows,
      scanColumns: dataset.scanCols,
      detectorRows: dataset.detectorRows,
      detectorColumns: dataset.detectorCols,
      sourceBytesPerValue: sourceBytesPerValue,
      maximumDecodedBytes: .max,
      alignToScanRows: false
    )
    var nextGlobalFrame = 0
    var shards: [Native4DSTEMIndexedShard] = []
    shards.reserveCapacity(dataset.dataFiles.count)
    for (sourceURL, indexURL) in zip(sourceURLs, indexURLs) {
      let index = try NativeQH5Index.open(sourceURL: sourceURL, indexURL: indexURL)
      guard index.metadata.detRows == dataset.detectorRows,
        index.metadata.detCols == dataset.detectorCols,
        index.metadata.srcDtype == dataset.sourceDtype
      else {
        throw Native4DSTEMIOError.invalidData(
          "\(indexURL.lastPathComponent) disagrees with \(dataset.label) detector geometry or dtype"
        )
      }
      let stop = nextGlobalFrame.addingReportingOverflow(index.metadata.nFrames)
      guard !stop.overflow else {
        throw Native4DSTEMIOError.invalidData(
          "\(dataset.label) global frame coverage overflows Int"
        )
      }
      shards.append(
        Native4DSTEMIndexedShard(
          sourceURL: sourceURL,
          indexURL: indexURL,
          index: index,
          globalFrameRange: nextGlobalFrame..<stop.partialValue
        )
      )
      nextGlobalFrame = stop.partialValue
    }
    guard nextGlobalFrame == plan.frameRanges.last?.upperBound else {
      throw Native4DSTEMIOError.invalidData(
        "\(dataset.label) indexes cover \(nextGlobalFrame) frames; expected \(dataset.scanRows * dataset.scanCols)"
      )
    }
    return Self(
      dataset: dataset,
      shards: shards,
      logicalFrameCount: nextGlobalFrame,
      sourceBytesPerValue: sourceBytesPerValue,
      decodedBytesPerFrame: plan.decodedBytesPerFrame,
      logicalDecodedBytes: plan.logicalDecodedBytes
    )
  }

  /// Partition every logical scan frame under the caller's transient ceiling.
  public func windows(
    maximumDecodedBytes: UInt64,
    alignToScanRows: Bool = true
  ) throws -> [Native4DSTEMIndexedWindow] {
    let plan = try Native4DSTEMFrameWindowPlan(
      scanRows: dataset.scanRows,
      scanColumns: dataset.scanCols,
      detectorRows: dataset.detectorRows,
      detectorColumns: dataset.detectorCols,
      sourceBytesPerValue: sourceBytesPerValue,
      maximumDecodedBytes: maximumDecodedBytes,
      alignToScanRows: alignToScanRows
    )
    return try plan.frameRanges.map { try window(for: $0) }
  }

  /// Resolve one `(scanRow, scanColumn)` detector frame without changing resolution.
  public func frameWindow(
    scanRow: Int,
    scanColumn: Int
  ) throws -> Native4DSTEMIndexedWindow {
    guard (0..<dataset.scanRows).contains(scanRow),
      (0..<dataset.scanCols).contains(scanColumn)
    else {
      throw Native4DSTEMIOError.invalidData(
        "Requested scan coordinate (\(scanRow), \(scanColumn)) is outside "
          + "\(dataset.scanRows)×\(dataset.scanCols)"
      )
    }
    let frame = scanRow * dataset.scanCols + scanColumn
    return try window(for: frame..<(frame + 1))
  }

  private func window(
    for globalRange: Range<Int>
  ) throws -> Native4DSTEMIndexedWindow {
    guard globalRange.lowerBound >= 0,
      globalRange.lowerBound < globalRange.upperBound,
      globalRange.upperBound <= logicalFrameCount
    else {
      throw Native4DSTEMIOError.invalidData("Indexed frame window is outside the logical scan")
    }
    var slices: [Native4DSTEMIndexedSlice] = []
    for (shardIndex, shard) in shards.enumerated() {
      let shardStart = max(globalRange.lowerBound, shard.globalFrameRange.lowerBound)
      let shardStop = min(globalRange.upperBound, shard.globalFrameRange.upperBound)
      guard shardStart < shardStop else { continue }
      for (chunkIndex, chunk) in shard.index.metadata.chunks.enumerated() {
        let chunkGlobalStart = shard.globalFrameRange.lowerBound + chunk.startFrame
        let chunkGlobalStop = chunkGlobalStart + chunk.nFrames
        let start = max(shardStart, chunkGlobalStart)
        let stop = min(shardStop, chunkGlobalStop)
        guard start < stop else { continue }
        let localStart = start - shard.globalFrameRange.lowerBound
        let localStop = stop - shard.globalFrameRange.lowerBound
        let chunkFrameOffset = start - chunkGlobalStart
        let frameCount = stop - start
        let wordsPerFrame = shard.index.metadata.nBlocksPerFrame * 2
        let metadataStart = chunk.metaOffsetWords + chunkFrameOffset * wordsPerFrame
        let metadataStop = metadataStart + frameCount * wordsPerFrame
        slices.append(
          Native4DSTEMIndexedSlice(
            shardIndex: shardIndex,
            chunkIndex: chunkIndex,
            globalFrameRange: start..<stop,
            shardFrameRange: localStart..<localStop,
            metadataWordRange: metadataStart..<metadataStop,
            chunkCompressedByteRange: chunk.rangeStart..<chunk.rangeEnd
          )
        )
      }
    }
    guard slices.first?.globalFrameRange.lowerBound == globalRange.lowerBound,
      slices.last?.globalFrameRange.upperBound == globalRange.upperBound,
      zip(slices, slices.dropFirst()).allSatisfy({
        $0.globalFrameRange.upperBound == $1.globalFrameRange.lowerBound
      })
    else {
      throw Native4DSTEMIOError.invalidData(
        "Prepared QH5 indexes do not cover the requested logical frame window exactly once"
      )
    }
    let frameCount = UInt64(globalRange.count)
    let decoded = frameCount.multipliedReportingOverflow(by: decodedBytesPerFrame)
    guard !decoded.overflow else {
      throw Native4DSTEMIOError.invalidData("Decoded frame-window bytes overflow UInt64")
    }
    return Native4DSTEMIndexedWindow(
      globalFrameRange: globalRange,
      decodedBytes: decoded.partialValue,
      slices: slices
    )
  }
}
