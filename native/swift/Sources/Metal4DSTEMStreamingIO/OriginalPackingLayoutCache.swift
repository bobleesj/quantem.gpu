import Compression
import CryptoKit
import Darwin
import Foundation

/// Disposable packing instructions, never count payload or a ready resident.
/// The caller supplies the location and freshly checked source binding.
enum OriginalPackingLayoutCache {
  private static let magic = Data([81, 71, 76, 65, 89, 79, 85, 49])  // QGLAYOU1
  private static let version: UInt32 = 1
  private static let preludeBytes = 48
  private static let maximumJSONBytes = 64 << 10
  private static let reservedBytes = preludeBytes + maximumJSONBytes
  private static let maximumFileBytes = 192 << 20
  private static let maximumHeaderBytes = 4 << 20
  private static let blockBytes = 8192
  private static let maximumCompressedBlockBytes = blockBytes * 2
  private enum Failure: Error { case invalid }

  struct SourceStamp: Codable, Equatable {
    let device, inode, bytes: UInt64
    let modificationSeconds, modificationNanoseconds: Int64
    let changeSeconds, changeNanoseconds: Int64

    init(url: URL) throws {
      let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
      guard descriptor >= 0 else { throw Failure.invalid }
      defer { Darwin.close(descriptor) }
      try self.init(descriptor: descriptor)
    }

    fileprivate init(descriptor: Int32) throws {
      var value = stat()
      guard fstat(descriptor, &value) == 0,
        value.st_mode & S_IFMT == S_IFREG, value.st_size >= 0
      else { throw Failure.invalid }
      device = UInt64(truncatingIfNeeded: value.st_dev)
      inode = UInt64(value.st_ino)
      bytes = UInt64(value.st_size)
      modificationSeconds = Int64(value.st_mtimespec.tv_sec)
      modificationNanoseconds = Int64(value.st_mtimespec.tv_nsec)
      changeSeconds = Int64(value.st_ctimespec.tv_sec)
      changeNanoseconds = Int64(value.st_ctimespec.tv_nsec)
    }
  }

  struct Binding: Codable, Equatable {
    let sourceIdentitySHA256: String
    // Master first, then the exact ordered data members. No source paths stored.
    let sourceStamps: [SourceStamp]
    let scanRows, scanColumns, detectorRows, detectorColumns: Int
    let sourceBytesPerValue: Int
    let framesPerWindow: Int
    let headerWordsPerPixel: Int
    let layout: String

    static func capture(
      sourceIdentity: String, sourceFiles: [URL],
      scanRows: Int, scanColumns: Int, detectorRows: Int, detectorColumns: Int,
      sourceDtype: String, framesPerWindow: Int, headerWordsPerPixel: Int
    ) throws -> Self {
      let sourceBytes: Int
      switch sourceDtype {
      case "uint8": sourceBytes = 1
      case "uint16": sourceBytes = 2
      default: throw Failure.invalid
      }
      let value = Self(
        sourceIdentitySHA256: sourceIdentity,
        sourceStamps: try sourceFiles.map { try SourceStamp(url: $0) },
        scanRows: scanRows, scanColumns: scanColumns, detectorRows: detectorRows,
        detectorColumns: detectorColumns, sourceBytesPerValue: sourceBytes,
        framesPerWindow: framesPerWindow, headerWordsPerPixel: headerWordsPerPixel,
        layout: "direct-bitpacked-u32/v3:tile32:nibble15-is16:le")
      guard value.geometry != nil else { throw Failure.invalid }
      return value
    }

    func isCurrent(sourceFiles: [URL]) -> Bool {
      (try? sourceFiles.map { try SourceStamp(url: $0) }) == sourceStamps
    }

    fileprivate var geometry: Geometry? {
      guard sourceIdentitySHA256.count == 64,
        sourceIdentitySHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
        !sourceStamps.isEmpty, sourceStamps.count <= 1024,
        scanRows > 0, scanColumns > 0, detectorRows > 0, detectorColumns > 0,
        [1, 2].contains(sourceBytesPerValue), framesPerWindow > 0,
        framesPerWindow <= 4096, framesPerWindow.isMultiple(of: 32),
        layout == "direct-bitpacked-u32/v3:tile32:nibble15-is16:le"
      else { return nil }
      let (scans, scanOverflow) = scanRows.multipliedReportingOverflow(by: scanColumns)
      let (pixels, pixelOverflow) = detectorRows.multipliedReportingOverflow(by: detectorColumns)
      let tiles = framesPerWindow / 32
      let stride = (tiles + 31) / 32 + (tiles + 7) / 8
      let (words, wordOverflow) = pixels.multipliedReportingOverflow(by: stride)
      let (headerBytes, byteOverflow) = words.multipliedReportingOverflow(by: 4)
      guard !scanOverflow, !pixelOverflow, !wordOverflow, !byteOverflow,
        scans.isMultiple(of: framesPerWindow), scans / framesPerWindow <= 256,
        headerWordsPerPixel == stride, headerBytes > 0, headerBytes <= maximumHeaderBytes
      else { return nil }
      let (counts, countOverflow) = pixels.multipliedReportingOverflow(by: framesPerWindow)
      let (denseBytes, denseOverflow) = counts.multipliedReportingOverflow(by: sourceBytesPerValue)
      guard !countOverflow, !denseOverflow, denseBytes.isMultiple(of: 4),
        denseBytes / 4 > 0, denseBytes / 4 <= Int(UInt32.max)
      else { return nil }
      return Geometry(
        windowCount: scans / framesPerWindow, headerBytes: headerBytes,
        blocks: (headerBytes + blockBytes - 1) / blockBytes,
        maximumPayloadWords: UInt32(denseBytes / 4))
    }
  }

  fileprivate struct Geometry {
    let windowCount, headerBytes, blocks: Int
    let maximumPayloadWords: UInt32
    var paddedHeaderBytes: Int { blocks * blockBytes }
    var metadataBytes: Int { blocks * 8 }
  }

  struct Window {
    let compressed: Data
    let metadata: [UInt32]
    let headerBytes: Int
    let paddedHeaderBytes: Int
    let payloadWordCount: UInt32

    /// Decode disposable instructions, not microscope counts. The caller owns
    /// one spare output byte so an overlong block cannot look exactly complete.
    /// Failure may modify that window; never use it before range validation.
    func decodeHeaders(into destination: UnsafeMutableRawBufferPointer) -> Bool {
      guard headerBytes > 0, headerBytes <= maximumHeaderBytes,
        paddedHeaderBytes >= headerBytes, paddedHeaderBytes <= maximumHeaderBytes,
        paddedHeaderBytes.isMultiple(of: blockBytes), paddedHeaderBytes - headerBytes < blockBytes,
        destination.count >= paddedHeaderBytes + 1, let output = destination.baseAddress,
        metadata.count == paddedHeaderBytes / blockBytes * 2, !compressed.isEmpty
      else { return false }
      var nextOffset = 0
      for block in 0..<(metadata.count / 2) {
        let offset = Int(metadata[block * 2])
        let count = Int(metadata[block * 2 + 1])
        guard offset == nextOffset, count > 0, count <= maximumCompressedBlockBytes,
          offset <= compressed.count, count <= compressed.count - offset
        else { return false }
        nextOffset += count
      }
      guard nextOffset == compressed.count else { return false }
      return compressed.withUnsafeBytes { input in
        for block in 0..<(metadata.count / 2) {
          let offset = Int(metadata[block * 2])
          let count = Int(metadata[block * 2 + 1])
          let decoded = compression_decode_buffer(
            output.advanced(by: block * blockBytes).assumingMemoryBound(to: UInt8.self),
            blockBytes + 1,
            input.baseAddress!.advanced(by: offset).assumingMemoryBound(to: UInt8.self), count,
            nil, COMPRESSION_LZ4_RAW)
          guard decoded == blockBytes else { return false }
        }
        return destination[headerBytes..<paddedHeaderBytes].allSatisfy { $0 == 0 }
      }
    }
  }

  private struct Record: Codable {
    let offset: Int
    let compressedBytes: Int
    let payloadWordCount: UInt32
    let checksum: String
  }
  private struct Manifest: Codable {
    let version: UInt32
    let binding: Binding
    let records: [Record]
  }

  /// A read failure invalidates this reader; callers recompute, never trust a
  /// partially read record. Compression is decoded later by the checked GPU path.
  final class Reader {
    private let handle: FileHandle
    private let stamp: SourceStamp
    private let geometry: Geometry
    private let records: [Record]
    private var invalid = false

    init?(url: URL, binding: Binding) {
      guard let geometry = binding.geometry else { return nil }
      let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
      guard descriptor >= 0 else { return nil }
      let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
      do {
        let stamp = try SourceStamp(descriptor: descriptor)
        guard stamp.bytes >= UInt64(reservedBytes), stamp.bytes <= UInt64(maximumFileBytes) else {
          return nil
        }
        let prelude = try readExactly(handle, count: preludeBytes)
        guard prelude.prefix(8) == magic, readU32(prelude, 12) == version else { return nil }
        let count = Int(readU32(prelude, 8))
        guard count > 0, count <= maximumJSONBytes else { return nil }
        let json = try readExactly(handle, count: count)
        guard Data(SHA256.hash(data: json)) == prelude.subdata(in: 16..<48) else { return nil }
        let manifest = try JSONDecoder().decode(Manifest.self, from: json)
        guard manifest.version == version, manifest.binding == binding,
          manifest.records.count == geometry.windowCount
        else { return nil }
        var offset = reservedBytes
        for record in manifest.records {
          guard record.offset == offset, record.compressedBytes > 0,
            record.compressedBytes <= geometry.blocks * maximumCompressedBlockBytes,
            record.payloadWordCount > 0, record.payloadWordCount <= geometry.maximumPayloadWords,
            record.checksum.count == 64
          else { return nil }
          offset += geometry.metadataBytes + record.compressedBytes
          guard offset <= maximumFileBytes else { return nil }
        }
        guard UInt64(offset) == stamp.bytes,
          try SourceStamp(descriptor: descriptor) == stamp
        else { return nil }
        self.handle = handle
        self.stamp = stamp
        self.geometry = geometry
        records = manifest.records
      } catch { return nil }
    }

    func read(window index: Int) -> Window? {
      guard !invalid, records.indices.contains(index) else { return nil }
      do {
        guard try SourceStamp(descriptor: handle.fileDescriptor) == stamp else {
          throw Failure.invalid
        }
        let record = records[index]
        try handle.seek(toOffset: UInt64(record.offset))
        let metadataBytes = try readExactly(handle, count: geometry.metadataBytes)
        let compressed = try readExactly(handle, count: record.compressedBytes)
        guard
          checksum(
            index: index, geometry: geometry, payloadWords: record.payloadWordCount,
            metadata: metadataBytes, compressed: compressed) == record.checksum
        else { throw Failure.invalid }
        var metadata: [UInt32] = []
        metadata.reserveCapacity(geometry.blocks * 2)
        var nextOffset = 0
        for block in 0..<geometry.blocks {
          let offset = Int(readU32(metadataBytes, block * 8))
          let length = Int(readU32(metadataBytes, block * 8 + 4))
          guard offset == nextOffset, length > 0, length <= maximumCompressedBlockBytes,
            offset <= compressed.count, length <= compressed.count - offset
          else { throw Failure.invalid }
          metadata.append(UInt32(offset))
          metadata.append(UInt32(length))
          nextOffset += length
        }
        guard nextOffset == compressed.count,
          try SourceStamp(descriptor: handle.fileDescriptor) == stamp
        else { throw Failure.invalid }
        return Window(
          compressed: compressed, metadata: metadata, headerBytes: geometry.headerBytes,
          paddedHeaderBytes: geometry.paddedHeaderBytes, payloadWordCount: record.payloadWordCount)
      } catch {
        invalid = true
        return nil
      }
    }
  }

  /// Single-owner streaming writer. A failed or abandoned write removes only
  /// its uniquely created temporary file; the previous complete cache survives.
  /// Mutable state is confined to the serial queue after initialization.
  /// Queued closures retain the writer, so deinitialization cannot race them.
  final class Writer: @unchecked Sendable {
    private let destination, temporary: URL
    private let destinationStamp: SourceStamp?
    private let binding: Binding
    private let geometry: Geometry
    private var handle: FileHandle?
    private var records: [Record] = []
    private var fileBytes = reservedBytes
    private var scratch: [UInt8]
    // Compression and file writes run on this serial queue so a window's plan
    // (about 1 MiB of headers, 3-4 ms to encode and write) never delays the
    // next Metal submission. `finish` drains the queue before publishing.
    private let queue = DispatchQueue(
      label: "qgpu.original-packing.plan-writer", qos: .userInitiated)

    init?(url: URL, binding: Binding) {
      guard let geometry = binding.geometry else { return nil }
      let scratchBytes = compression_encode_scratch_buffer_size(COMPRESSION_LZ4_RAW)
      guard scratchBytes <= 1 << 20 else { return nil }
      let parent = url.deletingLastPathComponent()
      let destinationStamp: SourceStamp?
      do {
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let properties = try parent.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard properties.isDirectory == true, properties.isSymbolicLink != true else { return nil }
        destinationStamp = try ownedDestinationStamp(url)
      } catch { return nil }
      let temporary = parent.appendingPathComponent(".packing-layout-\(UUID().uuidString).partial")
      let descriptor = Darwin.open(
        temporary.path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR)
      guard descriptor >= 0 else { return nil }
      let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
      do { try handle.write(contentsOf: Data(count: reservedBytes)) } catch {
        try? handle.close()
        try? FileManager.default.removeItem(at: temporary)
        return nil
      }
      destination = url
      self.temporary = temporary
      self.binding = binding
      self.destinationStamp = destinationStamp
      self.geometry = geometry
      self.handle = handle
      scratch = [UInt8](repeating: 0, count: max(1, scratchBytes))
    }

    deinit { abandon() }

    /// Queues the window asynchronously. A failed write is reported by `finish`.
    @discardableResult
    func append(headerData: Data, payloadWordCount: UInt32) -> Bool {
      queue.async { self.appendNow(headerData: headerData, payloadWordCount: payloadWordCount) }
      return true
    }

    private func appendNow(headerData: Data, payloadWordCount: UInt32) {
      guard let handle else { return }
      do {
        guard records.count < geometry.windowCount, headerData.count == geometry.headerBytes,
          payloadWordCount > 0, payloadWordCount <= geometry.maximumPayloadWords
        else { throw Failure.invalid }
        var compressed = Data()
        compressed.reserveCapacity(headerData.count)
        var metadata = Data()
        metadata.reserveCapacity(geometry.metadataBytes)
        var output = [UInt8](repeating: 0, count: maximumCompressedBlockBytes)
        var tail = [UInt8](repeating: 0, count: blockBytes)
        try headerData.withUnsafeBytes { raw in
          for block in 0..<geometry.blocks {
            let first = block * blockBytes
            let count = min(blockBytes, raw.count - first)
            func encode(_ input: UnsafePointer<UInt8>) throws {
              let size = output.withUnsafeMutableBufferPointer { destination in
                scratch.withUnsafeMutableBytes { scratch in
                  compression_encode_buffer(
                    destination.baseAddress!, destination.count,
                    input, blockBytes, scratch.baseAddress, COMPRESSION_LZ4_RAW)
                }
              }
              guard size > 0, size <= maximumCompressedBlockBytes else { throw Failure.invalid }
              appendU32(UInt32(compressed.count), to: &metadata)
              appendU32(UInt32(size), to: &metadata)
              compressed.append(contentsOf: output.prefix(size))
            }
            if count == blockBytes {
              try encode(raw.baseAddress!.advanced(by: first).assumingMemoryBound(to: UInt8.self))
            } else {
              tail.withUnsafeMutableBytes { $0.copyBytes(from: raw[first..<(first + count)]) }
              try tail.withUnsafeBufferPointer { try encode($0.baseAddress!) }
            }
          }
        }
        let nextBytes = fileBytes + metadata.count + compressed.count
        guard nextBytes <= maximumFileBytes else { throw Failure.invalid }
        let record = Record(
          offset: fileBytes, compressedBytes: compressed.count,
          payloadWordCount: payloadWordCount,
          checksum: checksum(
            index: records.count, geometry: geometry, payloadWords: payloadWordCount,
            metadata: metadata, compressed: compressed))
        try handle.write(contentsOf: metadata)
        try handle.write(contentsOf: compressed)
        records.append(record)
        fileBytes = nextBytes
      } catch {
        abandon()
      }
    }

    @discardableResult
    func finish() -> Bool {
      queue.sync { finishNow() }
    }

    private func finishNow() -> Bool {
      guard let handle else { return false }
      do {
        guard records.count == geometry.windowCount else { throw Failure.invalid }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = try encoder.encode(
          Manifest(version: version, binding: binding, records: records))
        guard json.count <= maximumJSONBytes else { throw Failure.invalid }
        var prelude = magic
        appendU32(UInt32(json.count), to: &prelude)
        appendU32(version, to: &prelude)
        prelude.append(contentsOf: SHA256.hash(data: json))
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: prelude)
        try handle.write(contentsOf: json)
        try handle.synchronize()
        try handle.close()
        self.handle = nil
        // Refresh only an unchanged, recognized cache. A missing destination must
        // remain missing at the atomic rename, including when another writer wins.
        guard try ownedDestinationStamp(destination) == destinationStamp else {
          throw Failure.invalid
        }
        let result =
          destinationStamp == nil
          ? Darwin.renamex_np(temporary.path, destination.path, UInt32(RENAME_EXCL))
          : Darwin.rename(temporary.path, destination.path)
        guard result == 0 else { throw Failure.invalid }
        return true
      } catch {
        abandon()
        return false
      }
    }

    private func abandon() {
      try? handle?.close()
      handle = nil
      try? FileManager.default.removeItem(at: temporary)
    }
  }

  /// Recognition deliberately does not require valid record checksums: damaged
  /// disposable caches may be refreshed, but data files and aliases may not.
  private static func ownedDestinationStamp(_ url: URL) throws -> SourceStamp? {
    let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
    guard descriptor >= 0 else {
      if errno == ENOENT { return nil }
      throw Failure.invalid
    }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    var attributes = stat()
    guard fstat(descriptor, &attributes) == 0, attributes.st_nlink == 1 else {
      throw Failure.invalid
    }
    let stamp = try SourceStamp(descriptor: descriptor)
    guard stamp.bytes >= UInt64(reservedBytes), stamp.bytes <= UInt64(maximumFileBytes) else {
      throw Failure.invalid
    }
    let prelude = try readExactly(handle, count: preludeBytes)
    let jsonBytes = Int(readU32(prelude, 8))
    guard prelude.prefix(8) == magic, readU32(prelude, 12) == version,
      jsonBytes > 0, jsonBytes <= maximumJSONBytes,
      try SourceStamp(descriptor: descriptor) == stamp
    else { throw Failure.invalid }
    return stamp
  }

  private static func readExactly(_ handle: FileHandle, count: Int) throws -> Data {
    guard count > 0, count <= maximumHeaderBytes * 2 + 4096,
      let data = try handle.read(upToCount: count), data.count == count
    else { throw Failure.invalid }
    return data
  }
  private static func readU32(_ data: Data, _ offset: Int) -> UInt32 {
    data.withUnsafeBytes {
      UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }
  }
  private static func appendU32(_ value: UInt32, to data: inout Data) {
    var value = value.littleEndian
    withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
  }
  private static func checksum(
    index: Int, geometry: Geometry, payloadWords: UInt32,
    metadata: Data, compressed: Data
  ) -> String {
    var domain = Data("packing-layout-window/v1\0".utf8)
    for value in [
      UInt32(index), UInt32(geometry.headerBytes), UInt32(geometry.paddedHeaderBytes), payloadWords,
    ] {
      appendU32(value, to: &domain)
    }
    var digest = SHA256()
    digest.update(data: domain)
    digest.update(data: metadata)
    digest.update(data: compressed)
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
