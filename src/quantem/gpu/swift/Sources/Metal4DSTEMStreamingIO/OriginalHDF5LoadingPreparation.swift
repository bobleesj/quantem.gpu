import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels

/// Prepare immutable loading resources before an acquisition is selected.
///
/// Call off the main thread, for example from an application's startup worker:
/// `try MetalCompactH5Loader.prepareLoadingResources(device: device)`.
/// This reads kernel resources only, not acquisitions, and allocates no dense
/// volume or resident count buffers. Failure leaves ordinary loading available.
extension MetalCompactH5Loader {
  public static func prepareLoadingResources(device: MTLDevice) throws {
    _ = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    _ = try OriginalHDF5Packing.forLoad(device: device, cachePlans: false)
  }
}

final class OriginalHDF5LoadingPreparation: @unchecked Sendable {
  static let shared = OriginalHDF5LoadingPreparation()
  private let lock = NSLock()
  private var packers: [String: OriginalHDF5Packing] = [:]

  func packing(device: MTLDevice, cachePlans: Bool) throws -> OriginalHDF5Packing {
    // Diagnostic configuration may vary within a benchmark process. Never
    // reuse a pipeline compiled for a different set of dispatch options.
    let options = ProcessInfo.processInfo.environment.filter { $0.key.hasPrefix("QGPU_ORIGINAL_") }
      .sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: "\n")
    let key = "\(device.registryID):\(cachePlans):\(options)"
    lock.lock()
    defer { lock.unlock() }
    if let existing = packers[key] { return existing }
    let created = try OriginalHDF5Packing(device: device, cachePlans: cachePlans)
    if packers.count >= 4 { packers.removeAll() }
    packers[key] = created
    return created
  }
}
