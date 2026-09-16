import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

@main struct QEMCalibrationRoundtrip {
  static func main() throws {
    guard CommandLine.arguments.count == 3, let device = MTLCreateSystemDefaultDevice() else {
      fatalError("Usage: qem-calibration-roundtrip counts.npy new-copy.qem")
    }
    let original = try NativeNPYSource(url: URL(fileURLWithPath: CommandLine.arguments[1]))
    precondition(original.dataset.sourceDtype == "uint16", "This exact-count fixture requires uint16 input")
    let output = URL(fileURLWithPath: CommandLine.arguments[2])
    let overrides: NativeQEMCalibration.Overrides = [
      NativeQEMCalibration.scanRow: .init(value: 0.45e-10, unit: "m", evidence: "measured scan row"),
      NativeQEMCalibration.scanColumn: .init(value: 0.55e-10, unit: "m", evidence: "measured scan column"),
      NativeQEMCalibration.detectorRow: .init(value: 0.31, unit: "1/Å", evidence: "diffraction standard"),
      NativeQEMCalibration.detectorColumn: .init(value: 0.32, unit: "1/Å", evidence: "diffraction standard"),
      "electron_source/accelerating_voltage": .init(value: 200000, unit: "V", evidence: "microscope setting"),
      "illumination_system/semi_convergence_angle": .init(value: 24.75, unit: "mrad", evidence: "aperture calibration"),
    ]
    try MetalQEMExporter.save(.counts(original), to: output, device: device, calibrationOverrides: overrides)
    let snapshot = try NativeANSSnapshot(url: output)
    let restored = try NativeQEMCalibration.read(metadata: snapshot.dataset.metadata ?? [:])
    precondition(restored == overrides, "Overrides must survive without local preferences")
    precondition(snapshot.dataset.sourceScanCalibration == original.dataset.sourceScanCalibration,
      "User calibration must not replace recorded calibration")
    let resident = try MetalRuntimeANSResidentSource.load(snapshot: snapshot, device: device)
    defer { resident.releaseResidentStorage() }
    let expected = try Data(contentsOf: original.url)
    for frame in 0..<(original.dataset.scanRows * original.dataset.scanCols) {
      let raw = try resident.extractRawDiffraction(scanRow: frame / original.dataset.scanCols, scanColumn: frame % original.dataset.scanCols)
      let pixels = original.dataset.detectorRows * original.dataset.detectorCols
      let values = expected.withUnsafeBytes { pointer in
        (0..<pixels).map { UInt32(pointer.loadUnaligned(fromByteOffset: original.dataOffset + (frame * pixels + $0) * 2, as: UInt16.self)) }
      }
      precondition(raw == values, "Saving calibration changed a measured count")
    }
    let preserved = output.deletingPathExtension().appendingPathExtension("preserved.qem")
    let reset = output.deletingPathExtension().appendingPathExtension("reset.qem")
    try resident.saveSnapshot(to: preserved)
    try resident.saveSnapshot(to: reset, calibrationOverrides: [:])
    let preservedSnapshot = try NativeANSSnapshot(url: preserved)
    let resetSnapshot = try NativeANSSnapshot(url: reset)
    let preservedValues = try NativeQEMCalibration.read(metadata: preservedSnapshot.dataset.metadata ?? [:])
    let resetValues = try NativeQEMCalibration.read(metadata: resetSnapshot.dataset.metadata ?? [:])
    precondition(preservedValues == overrides)
    precondition(resetValues.isEmpty)
    print("PASS: all raw diffraction counts exact; saved calibration preserved by default and cleared only explicitly")
  }
}
