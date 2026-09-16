import CryptoKit
import Darwin
import Foundation

/// Checksummed QEM envelope. Codec readers validate their own array layouts.
/// Example: `try NativeQEMFile(url: url).verifiedMapping()`.
public struct NativeQEMFile {
  public let url: URL
  public let header: [String: Any]
  public let bodyStart: Int
  public let bodyBytes: Int
  public let codec: String
  public let identity: String
  private let checksums: [String]
  public static let blockBytes = 64 << 20

  public static func matches(_ url: URL) -> Bool {
    guard let file = try? FileHandle(forReadingFrom: url) else { return false }
    defer { try? file.close() }
    return (try? file.read(upToCount: 8)) == NativeQEMMetadata.magic
  }

  public init(url: URL) throws {
    self.url = url
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    let prefix = try file.read(upToCount: 56) ?? Data()
    guard prefix.count == 56, prefix.prefix(8) == NativeQEMMetadata.magic else {
      throw Native4DSTEMIOError.invalidData("Choose a complete QuantEM (.qem) file.")
    }
    let length = prefix.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self).littleEndian }
    let start = prefix.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 16, as: UInt64.self).littleEndian }
    guard length > 0, length <= 16 << 20, start == 56 + length else {
      throw Native4DSTEMIOError.invalidData("Invalid QEM header size; recopy the complete file.")
    }
    let json = try file.read(upToCount: Int(length)) ?? Data()
    let digest = SHA256.hash(data: json)
    guard json.count == Int(length), Data(digest) == prefix.suffix(32),
      let header = try JSONSerialization.jsonObject(with: json) as? [String: Any],
      let shape = header["shape"] as? [Int], shape.count == 4,
      shape.allSatisfy({ $0 > 0 && $0 < 1 << 24 }),
      let codec = header["codec"] as? String,
      let bytes = header["bytes"] as? Int, bytes > 0, bytes < Int.max - Int(start),
      let checksums = header["sha256"] as? [String],
      checksums.count == (bytes - 1) / Self.blockBytes + 1,
      try file.seekToEnd() == UInt64(bytes) + start
    else { throw Native4DSTEMIOError.invalidData("Invalid QEM header or file length; recopy the complete file.") }
    try NativeQEMMetadata.validate(header, shape: shape)
    self.header = header; self.codec = codec; bodyBytes = bytes; bodyStart = Int(start)
    self.checksums = checksums
    identity = digest.map { String(format: "%02x", $0) }.joined()
  }

  public func verifiedMapping() throws -> Data {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    guard data.count == bodyStart + bodyBytes else {
      throw Native4DSTEMIOError.invalidData("QEM file changed during loading; reopen it.")
    }
    // Same bounded parallel verification as the integer snapshot reader. No
    // decode or unchecked byte is submitted to Metal before all workers join.
    let expected = checksums, start = bodyStart, length = bodyBytes
    try data.withUnsafeBytes { bytes in
      nonisolated(unsafe) let pointer = bytes.baseAddress!
      nonisolated(unsafe) var failedBlock: Int?
      let lock = NSLock()
      let workers = min(8, expected.count)
      DispatchQueue.concurrentPerform(iterations: workers) { worker in
        for number in stride(from: worker, to: expected.count, by: workers) {
          let offset = number * Self.blockBytes
          let count = min(Self.blockBytes, length - offset)
          let view = Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: pointer.advanced(by: start + offset)),
                          count: count, deallocator: .none)
          let digest = SHA256.hash(data: view).map { String(format: "%02x", $0) }.joined()
          if digest != expected[number] {
            lock.lock(); failedBlock = number; lock.unlock()
          }
        }
      }
      if let failedBlock {
        throw Native4DSTEMIOError.invalidData("QEM checksum mismatch in block \(failedBlock); recopy the file.")
      }
    }
    return data
  }
}

/// Bounded writer for already-encoded bytes; originals and existing copies are kept.
/// Example: create, append encoded arrays, then `finish(header:)`.
public final class NativeQEMWriter {
  private let destination: URL
  private let bodyURL: URL
  private let file: FileHandle
  private var digest = SHA256()
  private var blockCount = 0
  private var checksums = [String]()
  public private(set) var bodyBytes = 0

  public init(destination: URL) throws {
    self.destination = destination
    guard !FileManager.default.fileExists(atPath: destination.path) else {
      throw Native4DSTEMIOError.invalidData("\(destination.lastPathComponent) already exists; choose another name.")
    }
    bodyURL = destination.deletingLastPathComponent().appendingPathComponent(".\(UUID().uuidString).qem-body")
    guard FileManager.default.createFile(atPath: bodyURL.path, contents: nil) else {
      throw Native4DSTEMIOError.invalidData("Cannot create a QEM copy; choose a writable folder.")
    }
    file = try FileHandle(forUpdating: bodyURL)
  }

  deinit { try? file.close(); try? FileManager.default.removeItem(at: bodyURL) }

  @discardableResult
  public func append(_ bytes: Data, shouldCancel: () -> Bool = { false }) throws -> Int {
    let first = bodyBytes
    var offset = 0
    while offset < bytes.count {
      if shouldCancel() { throw CancellationError() }
      let count = min(NativeQEMFile.blockBytes - blockCount, bytes.count - offset)
      let part = bytes.subdata(in: offset..<offset + count)
      try file.write(contentsOf: part); digest.update(data: part)
      offset += count; blockCount += count; bodyBytes += count
      if blockCount == NativeQEMFile.blockBytes {
        checksums.append(digest.finalize().map { String(format: "%02x", $0) }.joined())
        digest = SHA256(); blockCount = 0
      }
    }
    return first
  }

  public func finish(header: [String: Any], shouldCancel: () -> Bool = { false }) throws {
    if shouldCancel() { throw CancellationError() }
    var header = header
    if blockCount > 0 { checksums.append(digest.finalize().map { String(format: "%02x", $0) }.joined()) }
    header["container"] = NativeQEMMetadata.container; header["container_version"] = 1
    header["bytes"] = bodyBytes; header["sha256"] = checksums
    let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    guard json.count <= 16 << 20 else { throw Native4DSTEMIOError.invalidData("QEM metadata exceeds 16 MiB; no copy was published.") }
    var prefix = NativeQEMMetadata.magic
    for number in [json.count, json.count + 56] {
      var value = UInt64(number).littleEndian
      withUnsafeBytes(of: &value) { prefix.append(contentsOf: $0) }
    }
    prefix.append(contentsOf: SHA256.hash(data: json)); prefix.append(json)
    let temporary = destination.deletingLastPathComponent().appendingPathComponent(".\(UUID().uuidString).qem-partial")
    defer { try? FileManager.default.removeItem(at: temporary) }
    guard FileManager.default.createFile(atPath: temporary.path, contents: nil) else {
      throw Native4DSTEMIOError.invalidData("Cannot write QEM output; check free space and folder permissions.")
    }
    let output = try FileHandle(forWritingTo: temporary)
    defer { try? output.close() }
    try output.write(contentsOf: prefix); try file.seek(toOffset: 0)
    while let block = try file.read(upToCount: NativeQEMFile.blockBytes), !block.isEmpty {
      if shouldCancel() { throw CancellationError() }
      try output.write(contentsOf: block)
    }
    try output.synchronize()
    if shouldCancel() { throw CancellationError() }
    guard link(temporary.path, destination.path) == 0 else {
      throw Native4DSTEMIOError.invalidData("Cannot publish QEM output: \(String(cString: strerror(errno))). Existing files were kept.")
    }
  }
}
