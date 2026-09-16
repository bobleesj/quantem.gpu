import Foundation
import Metal
import Native4DSTEMIO

/// Lossless file conversion, independent of app dialogs, preferences and queues.
public enum MetalQEMExporter {
  /// Supply a validated reader and explicit correction policy, never inferred UI state.
  public enum Source {
    case counts(any NativeCountArray)
    case indexed(Native4DSTEMIndexedSource)
    case snapshot(NativeANSSnapshot)
    case empad(NativeEMPADSource, background: NativeEMPADSource?, alreadyCorrected: Bool)
  }

  /// Save a complete acquisition without changing its count representation.
  ///
  /// Example: `try MetalQEMExporter.save(.counts(NativeNPYSource(url: input)),
  /// to: output, device: device)`. The caller owns scheduling and cancellation.
  /// Existing destinations are never overwritten. Readers retain source metadata.
  /// Omit calibrationOverrides to preserve saved edits; pass a complete dictionary
  /// to replace them, or an empty dictionary to clear them in the new copy.
  public static func save(
    _ source: Source, to destination: URL, device: MTLDevice,
    maximumAdditionalBytes: UInt64? = nil,
    calibrationOverrides: NativeQEMCalibration.Overrides? = nil,
    shouldCancel: () -> Bool = { false },
    progress: (String) -> Void = { _ in }
  ) throws {
    if let calibrationOverrides { try NativeQEMCalibration.validate(calibrationOverrides) }
    guard !FileManager.default.fileExists(atPath: destination.path) else {
      throw Native4DSTEMIOError.invalidData(
        "\(destination.lastPathComponent) already exists; choose another destination.")
    }
    guard !shouldCancel() else { throw Native4DSTEMIOError.invalidData("Conversion cancelled.") }
    let available =
      UInt64(device.recommendedMaxWorkingSetSize) > UInt64(device.currentAllocatedSize)
      ? UInt64(device.recommendedMaxWorkingSetSize) - UInt64(device.currentAllocatedSize) : 0
    let budget = min(maximumAdditionalBytes ?? available, available)
    progress("Reading original counts…")
    if case .empad(let original, let reference, let alreadyCorrected) = source {
      guard reference == nil || !alreadyCorrected else {
        throw Native4DSTEMIOError.invalidData(
          "An already-corrected acquisition cannot also subtract a dark reference.")
      }
      let background = try reference.map {
        try MetalEMPADBackground.load(
          $0, device: device, memoryBudgetBytes: budget, shouldCancel: shouldCancel)
      }
      let resident = try MetalEMPADResidentSource.load(
        original, device: device,
        memoryBudgetBytes: budget, subtracting: background, shouldCancel: shouldCancel)
      defer { resident.releaseResidentStorage() }
      progress("Writing .qem file…")
      try resident.saveQEM(
        to: destination, userConfirmedBackgroundCorrected: alreadyCorrected,
        calibrationOverrides: calibrationOverrides,
        shouldCancel: shouldCancel)
      return
    }
    let resident: MetalRuntimeANSResidentSource
    switch source {
    case .counts(let array):
      resident = try .load(
        array: array, device: device, maximumAdditionalBytes: budget,
        shouldCancel: shouldCancel,
        progress: { done, total in
          if done == total || done % 4096 == 0 {
            progress("Encoding \(done * 100 / max(1, total))%")
          }
        })
    case .indexed(let indexed):
      resident = try .load(
        source: indexed, device: device, includeSpatialIndex: true,
        maximumAdditionalBytes: budget, shouldCancel: shouldCancel)
    case .snapshot(let snapshot):
      resident = try .load(snapshot: snapshot, device: device, maximumAdditionalBytes: budget)
    case .empad:
      preconditionFailure("EMPAD conversion is handled above")
    }
    defer { resident.releaseResidentStorage() }
    progress("Writing .qem file…")
    try resident.saveSnapshot(to: destination, calibrationOverrides: calibrationOverrides, shouldCancel: shouldCancel)
  }
}
