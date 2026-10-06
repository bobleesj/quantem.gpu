import CryptoKit
import Darwin
import Foundation

/// An optional, disposable native acceleration artifact, not a scientific format.
/// The source HDF5 remains authoritative. Publication is atomic and no-clobber.
struct CompactNativeCache: Codable, Sendable {
  struct Shard: Codable, Sendable {
    let payloadOffset: UInt64
    let payloadBytes: UInt64
    let payloadSHA256: String
    let descriptorsOffset: UInt64
    let descriptorsBytes: UInt64
    let descriptorsSHA256: String
  }

  static let headerBytes = 65_536
  static let magic = Data("QGMC0001".utf8)
  let schema: String
  let sourceSignature: String
  let sourceManifestSHA256: String
  let fileBytes: UInt64
  let shards: [Shard]

  static func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  static func invalid(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest("Native compact cache: " + message)
  }

  static func stamp(_ file: FileHandle) throws -> String {
    var state = stat()
    guard fstat(file.fileDescriptor, &state) == 0 else {
      throw invalid("could not stat the cache.")
    }
    return "\(state.st_size):\(state.st_mtimespec.tv_sec):\(state.st_mtimespec.tv_nsec)"
      + ":\(state.st_ctimespec.tv_sec):\(state.st_ctimespec.tv_nsec)"
  }

  /// Bind the exact JSON/binary index and source size/modification stamp.
  /// This is conservative cache invalidation, not a replacement for payload SHA.
  static func signature(sourceURL: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: sourceURL)
    defer { try? file.close() }
    var state = stat()
    guard fstat(file.fileDescriptor, &state) == 0 else {
      throw invalid("could not stat the source.")
    }
    let prelude = try read(file, count: 24)
    let offset = Int(
      prelude.withUnsafeBytes {
        UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 16, as: UInt32.self))
      })
    let count = Int(
      prelude.withUnsafeBytes {
        UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 20, as: UInt32.self))
      })
    guard offset >= 24, count >= 76, offset <= 16 * 1024 * 1024,
      count <= 16 * 1024 * 1024, offset + count <= state.st_size
    else { throw invalid("source index cannot be bound safely.") }
    var hasher = SHA256()
    hasher.update(data: prelude)
    hasher.update(data: try read(file, count: offset + count - 24))
    hasher.update(
      data: Data(
        "\(state.st_size):\(state.st_mtimespec.tv_sec):\(state.st_mtimespec.tv_nsec)".utf8
      ))
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }

  static func read(_ file: FileHandle, count: Int) throws -> Data {
    var data = Data()
    while data.count < count {
      guard let part = try file.read(upToCount: count - data.count), !part.isEmpty else {
        throw invalid("file is truncated; remove this cache and prepare it again.")
      }
      data.append(part)
    }
    return data
  }

  static func read(from file: FileHandle) throws -> CompactNativeCache {
    let prelude = try read(file, count: 48)
    guard prelude.prefix(8) == magic else {
      throw invalid("unsupported header; remove this cache and prepare it again.")
    }
    let bytes = prelude.withUnsafeBytes {
      UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self))
    }
    guard bytes > 0, bytes <= UInt64(headerBytes - 48) else {
      throw invalid("metadata length is invalid.")
    }
    let json = try read(file, count: Int(bytes))
    guard Data(SHA256.hash(data: json)) == prelude[16..<48] else {
      throw invalid("metadata SHA-256 mismatch.")
    }
    let cache = try JSONDecoder().decode(Self.self, from: json)
    var state = stat()
    guard fstat(file.fileDescriptor, &state) == 0, state.st_size >= headerBytes,
      cache.schema == "quantem.gpu.native-compact-cache/v1",
      cache.fileBytes == UInt64(state.st_size), !cache.shards.isEmpty
    else { throw invalid("incomplete cache or unsupported schema.") }
    var end = UInt64(headerBytes)
    for shard in cache.shards {
      for (offset, count, sha) in [
        (shard.payloadOffset, shard.payloadBytes, shard.payloadSHA256),
        (shard.descriptorsOffset, shard.descriptorsBytes, shard.descriptorsSHA256),
      ] {
        guard offset == end, count > 0, count.isMultiple(of: 4),
          offset <= cache.fileBytes, count <= cache.fileBytes - offset,
          sha.count == 64, sha.allSatisfy({ "0123456789abcdef".contains($0) })
        else { throw invalid("invalid, overlapping, or unauthenticated shard range.") }
        end = offset + count
      }
    }
    guard end == cache.fileBytes else { throw invalid("unaccounted trailing bytes.") }
    return cache
  }

  func writeHeader(to file: FileHandle) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let json = try encoder.encode(self)
    guard json.count <= Self.headerBytes - 48 else {
      throw Self.invalid("too many shards for this cache version.")
    }
    var prelude = Self.magic
    var count = UInt64(json.count).littleEndian
    withUnsafeBytes(of: &count) { prelude.append(contentsOf: $0) }
    prelude.append(contentsOf: SHA256.hash(data: json))
    prelude.append(json)
    prelude.append(Data(count: Self.headerBytes - prelude.count))
    try file.seek(toOffset: 0)
    try file.write(contentsOf: prelude)
  }
}
