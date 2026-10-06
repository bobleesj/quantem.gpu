import CryptoKit
import Foundation
import Metal

/// Reuse compilation without sharing mutable scientific buffers. The source
/// digest prevents an edited kernel at the same resource path using old code.
final class MetalLoadingLibraryCache: @unchecked Sendable {
  static let shared = MetalLoadingLibraryCache()
  private let lock = NSLock()
  private var libraries: [String: MTLLibrary] = [:]

  func library(device: MTLDevice, source: String, strict: Bool) throws -> MTLLibrary {
    let digest = SHA256.hash(data: Data(source.utf8)).map { String(format: "%02x", $0) }.joined()
    let key = "\(device.registryID):\(strict):\(digest)"
    lock.lock()
    defer { lock.unlock() }
    if let existing = libraries[key] { return existing }
    let options = MTLCompileOptions()
    if strict { options.fastMathEnabled = false }
    let library = try device.makeLibrary(source: source, options: strict ? options : nil)
    // Bound developer hot-reload/multiple-device use. Existing clients retain
    // their own library; eviction never releases an active resident's buffers.
    if libraries.count >= 16 { libraries.removeAll() }
    libraries[key] = library
    return library
  }
}
