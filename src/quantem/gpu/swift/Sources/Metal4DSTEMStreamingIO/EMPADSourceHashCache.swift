import CryptoKit
import Foundation
import Native4DSTEMIO

/// Disposable identity metadata only. Every detector word is read and packed
/// on every load, whether or not its previously computed hash can be reused.
struct EMPADSourceHashCache: Codable {
  let schema: String
  let snapshot: Data
  let logicalSHA256: String
  let checksum: String

  private static let version = "quantem.gpu.empad-source-hash/v1"

  private static func checksum(snapshot: Data, hash: String) -> String {
    var digest = SHA256()
    digest.update(data: Data(version.utf8))
    digest.update(data: snapshot)
    digest.update(data: Data(hash.utf8))
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }

  static func safeURL(_ proposed: URL?, source: NativeEMPADSource) -> URL? {
    guard let proposed, proposed.isFileURL else { return nil }
    let url = proposed.standardizedFileURL.resolvingSymlinksInPath()
    let protected = [source.rawURL, source.metadataURL].compactMap { $0 }
      .map { $0.standardizedFileURL.resolvingSymlinksInPath() }
    return protected.contains(url) ? nil : url
  }

  static func read(_ url: URL?, snapshot: Data) -> String? {
    guard let url, let handle = try? FileHandle(forReadingFrom: url) else { return nil }
    defer { try? handle.close() }
    guard let data = try? handle.read(upToCount: 16385), data.count <= 16384,
      let record = try? JSONDecoder().decode(Self.self, from: data),
      record.schema == version, record.snapshot == snapshot,
      record.logicalSHA256.count == 64,
      record.logicalSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      record.checksum == checksum(snapshot: snapshot, hash: record.logicalSHA256)
    else { return nil }
    return record.logicalSHA256
  }

  static func write(_ url: URL?, snapshot: Data, hash: String) {
    guard let url else { return }
    let record = Self(schema: version, snapshot: snapshot, logicalSHA256: hash,
      checksum: checksum(snapshot: snapshot, hash: hash))
    guard let data = try? JSONEncoder().encode(record) else { return }
    // Cache permissions/corruption cannot prevent a valid original-source load.
    do {
      try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
      try data.write(to: url, options: .atomic)
    } catch {}
  }
}
