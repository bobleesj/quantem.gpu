import Foundation
import Metal
import Metal4DSTEMKernels

/// Reuse compiled code across opens without retaining any dataset buffers.
/// Only the most recently used device is cached; existing residents own their
/// pipelines independently if a caller subsequently switches devices.
final class CompactH5KernelCache: @unchecked Sendable {
  static let shared = CompactH5KernelCache()
  private let lock = NSLock()
  private var device: MTLDevice?
  private var library: MTLLibrary?
  private var pipelines: [String: MTLComputePipelineState] = [:]
  private var metadata: CompactH5MetadataKernels?

  private init() {}

  private func libraryLocked(device: MTLDevice) throws -> MTLLibrary {
    if let previous = self.device, previous === device, let library { return library }
    let compiled = try Metal4DSTEMKernels.makeCompactH5Library(device: device)
    pipelines.removeAll()
    metadata = nil
    self.device = device
    library = compiled
    return compiled
  }

  func library(device: MTLDevice) throws -> MTLLibrary {
    try lock.withLock { try libraryLocked(device: device) }
  }

  func pipeline(name: String, device: MTLDevice) throws -> MTLComputePipelineState {
    try lock.withLock {
      let library = try libraryLocked(device: device)
      if let cached = pipelines[name] { return cached }
      guard let function = library.makeFunction(name: name) else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Compact Metal function \(name) is missing. Rebuild the bundled kernels.")
      }
      let compiled = try device.makeComputePipelineState(function: function)
      pipelines[name] = compiled
      return compiled
    }
  }

  func metadata(device: MTLDevice) throws -> CompactH5MetadataKernels {
    try lock.withLock {
      let library = try libraryLocked(device: device)
      if let metadata { return metadata }
      let compiled = try CompactH5MetadataKernels(device: device, library: library)
      metadata = compiled
      return compiled
    }
  }
}
