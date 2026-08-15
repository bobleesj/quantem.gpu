import Darwin
import Foundation

private let qh5IndexMagic = Data("QH5IDX01".utf8)

struct QH5IndexChunk: Codable, Equatable {
  let startFrame: Int
  let nFrames: Int
  let rangeStart: UInt64
  let rangeEnd: UInt64
  let metaOffsetWords: Int
  let metaWords: Int
}

struct QH5IndexMetadata: Codable, Equatable {
  let sourcePath: String
  let sourceBytes: UInt64
  let sourceMtimeNs: UInt64
  let detRows: Int
  let detCols: Int
  let nFrames: Int
  let srcDtype: String
  let blockElems: Int
  let nBlocksPerFrame: Int
  let chunks: [QH5IndexChunk]
}

enum QH5IndexWriter {
  static func prepare(
    source: URL,
    destination: URL
  ) throws -> QH5IndexMetadata {
    let identity = try nativeFileIdentity(for: source)
    if let cached = try cachedMetadata(at: destination),
      cached.sourcePath == identity.path,
      cached.sourceBytes == identity.bytes,
      cached.sourceMtimeNs == identity.modificationNanoseconds
    {
      return cached
    }

    let stack = try NativeHDF5Bridge.inspectStack(
      at: source,
      includeChunks: true
    )
    guard stack.chunks.count == stack.frameCount else {
      throw Native4DSTEMIOError.invalidData(
        "\(source.lastPathComponent) has \(stack.chunks.count) HDF5 chunks for \(stack.frameCount) frames"
      )
    }
    let detectorPixels = try multiplied(
      stack.detectorRows,
      stack.detectorColumns,
      label: "detector size"
    )
    let result = try withMappedFile(source) { rawBytes -> (QH5IndexMetadata, [UInt32]) in
      guard let base = rawBytes.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
        throw Native4DSTEMIOError.invalidData("\(source.lastPathComponent) is empty")
      }
      func exactOffset(_ value: UInt64, label: String) throws -> Int {
        guard let offset = Int(exactly: value), offset >= 0, offset <= rawBytes.count else {
          throw Native4DSTEMIOError.invalidData(
            "\(source.lastPathComponent) has an out-of-range \(label)"
          )
        }
        return offset
      }
      func readBE32(_ offset: Int) throws -> UInt32 {
        guard offset >= 0, offset <= rawBytes.count - 4 else {
          throw Native4DSTEMIOError.invalidData(
            "\(source.lastPathComponent) has a truncated bitshuffle/LZ4 header"
          )
        }
        return UInt32(base[offset]) << 24
          | UInt32(base[offset + 1]) << 16
          | UInt32(base[offset + 2]) << 8
          | UInt32(base[offset + 3])
      }

      let firstOffset = try exactOffset(stack.chunks[0].offset, label: "first chunk offset")
      let blockBytes = Int(try readBE32(firstOffset + 8))
      guard blockBytes > 0, blockBytes.isMultiple(of: stack.sourceBytes) else {
        throw Native4DSTEMIOError.invalidData(
          "\(source.lastPathComponent) has invalid bitshuffle block geometry"
        )
      }
      let blockElements = blockBytes / stack.sourceBytes
      let blocksPerFrame = (detectorPixels + blockElements - 1) / blockElements
      if stack.sourceBytes == 2 && blockElements != 4096 {
        throw Native4DSTEMIOError.invalidData(
          "\(source.lastPathComponent) has \(blockElements) values per bitshuffle block; expected 4096"
        )
      }
      // A 192x192 uint8 frame ends with a valid 4096-value tail after four
      // full 8192-value blocks. The uint8 Metal kernel handles that tail with
      // its actual bit-plane stride; keep the stricter full-block contract for
      // the unchanged uint16 hot path.
      guard detectorPixels >= blockElements,
        stack.sourceBytes == 2 || detectorPixels.isMultiple(of: 32)
      else {
        throw Native4DSTEMIOError.invalidData(
          "\(source.lastPathComponent) detector size \(stack.detectorRows)x\(stack.detectorColumns) is incompatible with \(blockElements)-value bitshuffle blocks"
        )
      }
      if stack.sourceBytes == 2 && !detectorPixels.isMultiple(of: blockElements) {
        throw Native4DSTEMIOError.invalidData(
          "\(source.lastPathComponent) detector size \(stack.detectorRows)x\(stack.detectorColumns) does not contain complete \(blockElements)-value uint16 bitshuffle blocks"
        )
      }
      let decodedFrameBytes = try multiplied(
        detectorPixels,
        stack.sourceBytes,
        label: "decoded frame bytes"
      )
      let framesPerChunk = max(1, (1 << 30) / decodedFrameBytes)
      let metadataWords = try multiplied(
        try multiplied(stack.frameCount, blocksPerFrame, label: "index block count"),
        2,
        label: "index word count"
      )
      var words = [UInt32](repeating: 0, count: metadataWords)
      var outputChunks: [QH5IndexChunk] = []
      outputChunks.reserveCapacity((stack.frameCount + framesPerChunk - 1) / framesPerChunk)
      var metadataOffset = 0

      for start in stride(from: 0, to: stack.frameCount, by: framesPerChunk) {
        let stop = min(stack.frameCount, start + framesPerChunk)
        let rangeStart = stack.chunks[start..<stop].map(\.offset).min()!
        var rangeEnd: UInt64 = 0
        let chunkMetadataStart = metadataOffset
        for frame in start..<stop {
          let rawChunk = stack.chunks[frame]
          let frameStart = try exactOffset(rawChunk.offset, label: "chunk offset")
          let frameSize = try exactOffset(rawChunk.size, label: "chunk size")
          guard frameStart <= rawBytes.count - frameSize else {
            throw Native4DSTEMIOError.invalidData(
              "\(source.lastPathComponent) detector chunk extends past end of file"
            )
          }
          var position = frameStart + 12
          for _ in 0..<blocksPerFrame {
            let compressedBytes = Int(try readBE32(position))
            let payload = position + 4
            guard payload <= rawBytes.count - compressedBytes else {
              throw Native4DSTEMIOError.invalidData(
                "\(source.lastPathComponent) has a truncated bitshuffle/LZ4 payload"
              )
            }
            let relative = UInt64(payload) - rangeStart
            guard let relativeWord = UInt32(exactly: relative),
              let compressedWord = UInt32(exactly: compressedBytes)
            else {
              throw Native4DSTEMIOError.invalidData(
                "\(source.lastPathComponent) QH5 chunk window exceeds 32-bit metadata"
              )
            }
            words[metadataOffset] = relativeWord
            words[metadataOffset + 1] = compressedWord
            metadataOffset += 2
            position = payload + compressedBytes
          }
          guard position == frameStart + frameSize else {
            throw Native4DSTEMIOError.invalidData(
              "\(source.lastPathComponent) has a malformed bitshuffle/LZ4 chunk payload"
            )
          }
          rangeEnd = max(rangeEnd, UInt64(position))
        }
        outputChunks.append(
          QH5IndexChunk(
            startFrame: start,
            nFrames: stop - start,
            rangeStart: rangeStart,
            rangeEnd: rangeEnd,
            metaOffsetWords: chunkMetadataStart,
            metaWords: metadataOffset - chunkMetadataStart
          )
        )
      }
      let metadata = QH5IndexMetadata(
        sourcePath: identity.path,
        sourceBytes: identity.bytes,
        sourceMtimeNs: identity.modificationNanoseconds,
        detRows: stack.detectorRows,
        detCols: stack.detectorColumns,
        nFrames: stack.frameCount,
        srcDtype: stack.sourceDtype,
        blockElems: blockElements,
        nBlocksPerFrame: blocksPerFrame,
        chunks: outputChunks
      )
      return (metadata, words)
    }
    try write(metadata: result.0, words: result.1, to: destination)
    return result.0
  }

  private static func cachedMetadata(at destination: URL) throws -> QH5IndexMetadata? {
    guard FileManager.default.fileExists(atPath: destination.path) else { return nil }
    let data = try Data(contentsOf: destination, options: .mappedIfSafe)
    guard data.count >= 16, data.prefix(8) == qh5IndexMagic else { return nil }
    let jsonLength = Int(readLE32(data, at: 8))
    guard jsonLength >= 0, jsonLength <= data.count - 16 else { return nil }
    return try? JSONDecoder().decode(
      QH5IndexMetadata.self,
      from: data.subdata(in: 16..<(16 + jsonLength))
    )
  }

  private static func write(
    metadata: QH5IndexMetadata,
    words: [UInt32],
    to destination: URL
  ) throws {
    let json = try JSONEncoder().encode(metadata)
    guard let jsonLength = UInt32(exactly: json.count),
      let wordCount = UInt32(exactly: words.count)
    else {
      throw Native4DSTEMIOError.invalidData("QH5 index metadata exceeds the format limit")
    }
    try FileManager.default.createDirectory(
      at: destination.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    var output = Data()
    output.reserveCapacity(16 + (json.count + 3) / 4 * 4 + words.count * 4)
    output.append(qh5IndexMagic)
    appendLE32(jsonLength, to: &output)
    appendLE32(wordCount, to: &output)
    output.append(json)
    output.append(Data(repeating: 0, count: (4 - json.count % 4) % 4))
    words.withUnsafeBytes { output.append(contentsOf: $0) }
    try output.write(to: destination, options: .atomic)
  }

  private static func readLE32(_ data: Data, at offset: Int) -> UInt32 {
    data.withUnsafeBytes { raw in
      raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian
    }
  }

  private static func appendLE32(_ value: UInt32, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
  }

  private static func multiplied(_ lhs: Int, _ rhs: Int, label: String) throws -> Int {
    let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else {
      throw Native4DSTEMIOError.invalidData("HDF5 \(label) exceeds this Mac's addressable range")
    }
    return value
  }

  private static func withMappedFile<Result>(
    _ url: URL,
    body: (UnsafeRawBufferPointer) throws -> Result
  ) throws -> Result {
    let descriptor = url.path.withCString { Darwin.open($0, O_RDONLY) }
    guard descriptor >= 0 else {
      throw Native4DSTEMIOError.invalidData("Could not open \(url.path)")
    }
    var status = stat()
    guard fstat(descriptor, &status) == 0, status.st_size > 0,
      let byteCount = Int(exactly: status.st_size)
    else {
      close(descriptor)
      throw Native4DSTEMIOError.invalidData("Could not inspect \(url.path)")
    }
    let address = mmap(nil, byteCount, PROT_READ, MAP_PRIVATE, descriptor, 0)
    close(descriptor)
    guard address != MAP_FAILED, let address else {
      throw Native4DSTEMIOError.invalidData("Could not memory-map \(url.path)")
    }
    defer { munmap(address, byteCount) }
    return try body(UnsafeRawBufferPointer(start: address, count: byteCount))
  }
}
