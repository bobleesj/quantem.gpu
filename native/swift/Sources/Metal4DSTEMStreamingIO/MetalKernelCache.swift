import Foundation
import Metal

/// Compiled kernel libraries and pipelines shared per device. Their sources are
/// constant, and compiled Metal objects are immutable and safe to share, so regions
/// and residents created in a loop no longer recompile the same library.
enum MetalKernelCache {
  private static let lock = NSLock()
  nonisolated(unsafe) private static var libraries: [String: MTLLibrary] = [:]
  nonisolated(unsafe) private static var pipelines: [String: MTLComputePipelineState] = [:]

  static func library(device: MTLDevice, key: String, source: () throws -> String) throws
    -> MTLLibrary
  {
    let name = "\(device.registryID)/\(key)"
    lock.lock()
    defer { lock.unlock() }
    if let cached = libraries[name] { return cached }
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try device.makeLibrary(source: try source(), options: options)
    libraries[name] = library
    return library
  }

  static func pipeline(device: MTLDevice, library: MTLLibrary, key: String, function: String)
    throws -> MTLComputePipelineState
  {
    let name = "\(device.registryID)/\(key)/\(function)"
    lock.lock()
    defer { lock.unlock() }
    if let cached = pipelines[name] { return cached }
    guard let entry = library.makeFunction(name: function) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest("Missing kernel \(function).")
    }
    let pipeline = try device.makeComputePipelineState(function: entry)
    pipelines[name] = pipeline
    return pipeline
  }
}
