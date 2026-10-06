import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

@main
struct QEMRoundtrip {
  static func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Native4DSTEMIOError.invalidData(message) }
  }
  static func hash<T>(_ array: [T]) -> String {
    array.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
  }
  static func detectorHashes(_ source: MetalRuntimeANSResidentSource, frames: [Int]) throws -> [String] {
    let series = try MetalRuntimeANSSeries(sources: [source])
    defer { series.release() }
    let shape = source.shape, pixels = shape[2] * shape[3]
    let invalid = Set(source.dataset.badPixelIndices)
    var hashes = [String]()
    for (inner, outer) in [(0.0, 0.2), (0.3, 0.5), (0.2, 1.0)] {
      let scale = Double(min(shape[2], shape[3]))
      let mask: [UInt8] = (0..<pixels).map {
        let y = Double($0 / shape[3]) - Double(shape[2] - 1) / 2
        let x = Double($0 % shape[3]) - Double(shape[3] - 1) / 2
        return x*x + y*y >= pow(inner*scale, 2) && x*x + y*y <= pow(outer*scale, 2) ? 1 : 0
      }
      let result = try series.updatePriorityVirtualDetectorBuffer(mask: mask, priorityIndex: 0)
      let values = Array(UnsafeBufferPointer(start: result.buffer.contents().assumingMemoryBound(to: UInt32.self), count: shape[0] * shape[1]))
      for frame in frames {
        let raw = try source.extractRawDiffraction(scanRow: frame / shape[1], scanColumn: frame % shape[1])
        let reference = raw.indices.reduce(UInt64(0)) { sum, pixel in
          sum + (mask[pixel] != 0 && !invalid.contains(pixel) ? UInt64(raw[pixel]) : 0)
        }
        try require(UInt64(values[frame]) == reference, "Integer detector reduction changed at \(frame)")
      }
      hashes.append(hash(values))
    }
    return hashes
  }
  static func main() {
    do { try run() }
    catch { fputs("FAIL \(error.localizedDescription)\n", stderr); exit(1) }
  }
  static func backgroundAndCorruptionChecks(device: MTLDevice) throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("qem-fixture-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    func raw(_ name: String, dark: Bool) throws -> NativeEMPADSource {
      let path = directory.appendingPathComponent(name + ".raw")
      var values = [Float](repeating: 0, count: 2 * 130 * 128)
      for frame in 0..<2 { for pixel in 0..<16384 {
        values[frame * 130 * 128 + pixel] = dark ? 0.5 : Float(pixel % 19 - 7) * 0.25 + Float(frame)
      } }
      try values.withUnsafeBytes { try Data($0).write(to: path) }
      return try NativeEMPADSource.open(path, scanShape: (1, 2))
    }
    let sample = try raw("sample", dark: false), dark = try raw("dark", dark: true)
    let budget = UInt64(device.recommendedMaxWorkingSetSize)
    let background = try MetalEMPADBackground.load(dark, device: device, memoryBudgetBytes: budget)
    let resident = try MetalEMPADResidentSource.load(sample, device: device, memoryBudgetBytes: budget, subtracting: background)
    let path = directory.appendingPathComponent("sample.qem")
    let calibration: NativeQEMCalibration.Overrides = [
      NativeQEMCalibration.scanRow: .init(value: 0.4e-10, unit: "m", evidence: "manual row calibration"),
      NativeQEMCalibration.scanColumn: .init(value: 0.6e-10, unit: "m", evidence: "manual column calibration"),
      "electron_source/accelerating_voltage": .init(value: 300000, unit: "V", evidence: "microscope setting"),
    ]
    try resident.saveQEM(to: path, calibrationOverrides: calibration)
    let expected = try sample.readFrames([1]).map { ($0 - 0.5).bitPattern }
    resident.releaseResidentStorage()
    // Remove only these generated fixtures, proving saved correction never
    // depends on the original sample/dark paths being available on reopen.
    try FileManager.default.removeItem(at: sample.rawURL)
    try FileManager.default.removeItem(at: dark.rawURL)
    let restored = try MetalEMPADResidentSource.load(NativeEMPADSource.open(path), device: device, memoryBudgetBytes: budget)
    let restoredCalibration = try NativeQEMCalibration.read(metadata: restored.source.microscopeMetadata)
    try require(restoredCalibration == calibration, "Saved user calibration was lost")
    defer { restored.releaseResidentStorage() }
    try require(restored.background?.identitySHA256 == background.identitySHA256, "Saved dark identity changed")
    let output = device.makeBuffer(length: 65536, options: .storageModeShared)!
    let command = device.makeCommandQueue()!.makeCommandBuffer()!
    try restored.encodeDiffraction(scanRow: 0, scanColumn: 1, into: output, command: command)
    command.commit(); command.waitUntilCompleted()
    try require(command.status == .completed, "Saved background query failed")
    let actual = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt32.self), count: 16384))
    try require(expected == actual, "Saved background was lost or applied twice")
    let complete = try Data(contentsOf: path)
    for (name, data) in [("truncated", Data(complete.dropLast())),
                         ("header", Data(complete.prefix(56)) + Data(repeating: 0, count: complete.count - 56)),
                         ("body", Data(complete.dropLast()) + Data([complete.last! ^ 1]))] {
      let bad = directory.appendingPathComponent(name + ".qem")
      try data.write(to: bad)
      var rejected = false
      do { _ = try NativeQEMFile(url: bad).verifiedMapping() } catch { rejected = true }
      try require(rejected, "Corrupt \(name) accepted")
    }
    print("PASS FIXTURE saved_background=exact originals_unavailable=pass negative_values=preserved corruption=3_rejected")
  }
  static func run() throws {
    setbuf(stdout, nil)
    guard CommandLine.arguments.count == 3, let device = MTLCreateSystemDefaultDevice() else {
      fatalError("Usage: qem-roundtrip original new-copy.qem")
    }
    let input = URL(fileURLWithPath: CommandLine.arguments[1])
    let output = URL(fileURLWithPath: CommandLine.arguments[2])
    try backgroundAndCorruptionChecks(device: device)
    let started = CFAbsoluteTimeGetCurrent()
    let empad = ["xml", "raw"].contains(input.pathExtension.lowercased()) || NativeEMPADSource.isEMDFloatAcquisition(input)
    if empad {
      let original = try NativeEMPADSource.open(input)
      if ProcessInfo.processInfo.environment["QEM_VERIFY_CANCELLATION"] == "1" {
        for boundary in [2, 5, 8, 11] {
          var checks = 0
          var cancelled = false
          do {
            let unexpected = try MetalEMPADResidentSource.load(
              original, device: device,
              memoryBudgetBytes: UInt64(device.recommendedMaxWorkingSetSize),
              shouldCancel: { checks += 1; return checks >= boundary })
            unexpected.releaseResidentStorage()
          } catch {
            cancelled = error.localizedDescription.lowercased().contains("cancel")
          }
          try require(cancelled, "EMPAD did not cancel at boundary \(boundary)")
        }
        print("PASS EMPAD cancellation at four read/pack boundaries")
      }
      let source = try MetalEMPADResidentSource.load(original, device: device,
        memoryBudgetBytes: UInt64(device.recommendedMaxWorkingSetSize))
      let loaded = CFAbsoluteTimeGetCurrent()
      try source.saveQEM(to: output)
      let saved = CFAbsoluteTimeGetCurrent()
      source.releaseResidentStorage()
      let restoredDescription = try NativeEMPADSource.open(output)
      let restored = try MetalEMPADResidentSource.load(restoredDescription, device: device,
        memoryBudgetBytes: UInt64(device.recommendedMaxWorkingSetSize))
      defer { restored.releaseResidentStorage() }
      let reopened = CFAbsoluteTimeGetCurrent()
      try require(restoredDescription.scanCalibration == original.scanCalibration, "EMPAD scan calibration changed")
      try require(restoredDescription.diffractionSamplingInverseNanometers == original.diffractionSamplingInverseNanometers, "EMPAD detector calibration changed")
      for (key, value) in original.microscopeMetadata {
        try require(restoredDescription.microscopeMetadata[key] == value, "EMPAD metadata lost: \(key)")
      }
      try require(restoredDescription.backgroundSubtractionEvidence?.statement == original.backgroundSubtractionEvidence?.statement, "Supplier correction evidence lost")
      try require(restoredDescription.backgroundSubtractionEvidence?.documentName == original.backgroundSubtractionEvidence?.documentName, "Supplier document name changed")
      let frames = Array(Set([0, 1, original.frameCount / 4, original.frameCount / 2,
        original.frameCount * 3 / 4, original.frameCount - 2, original.frameCount - 1])).sorted()
      let queue = device.makeCommandQueue()!
      let dp = device.makeBuffer(length: 65536, options: .storageModeShared)!
      for frame in frames {
        let command = queue.makeCommandBuffer()!
        try restored.encodeDiffraction(scanRow: frame / original.scanColumns, scanColumn: frame % original.scanColumns, into: dp, command: command)
        command.commit(); command.waitUntilCompleted()
        try require(command.status == .completed, "EMPAD query failed")
        let reference = try original.readFrames([frame]).map(\.bitPattern)
        let actual = Array(UnsafeBufferPointer(start: dp.contents().assumingMemoryBound(to: UInt32.self), count: 16384))
        try require(actual == reference, "EMPAD float bit patterns changed at \(frame)")
      }
      for (name, inner, outer) in [("BF", 0.0, 20.0), ("ADF", 30.0, 60.0), ("DF", 20.0, 100.0)] {
        let mask: [UInt8] = (0..<16384).map {
          let row = Double($0 / 128) - 63.5, col = Double($0 % 128) - 63.5
          let radius = row * row + col * col
          return radius >= inner * inner && radius <= outer * outer ? 1 : 0
        }
        let maskBuffer = mask.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        let image = device.makeBuffer(length: original.frameCount * 4, options: .storageModeShared)!
        let command = queue.makeCommandBuffer()!
        try restored.encodeVirtualImage(mask: maskBuffer, into: image, command: command)
        command.commit(); command.waitUntilCompleted()
        try require(command.status == .completed, "EMPAD detector failed")
        for frame in frames {
          let raw = try original.readFrames([frame])
          let expected = raw.indices.reduce(0.0) { $0 + (mask[$1] == 0 ? 0 : Double(raw[$1])) }
          let actual = Double(image.contents().assumingMemoryBound(to: Float.self)[frame])
          try require(abs(actual - expected) <= max(1e-5, abs(expected) * 1e-6), "EMPAD \(name) differs at \(frame)")
        }
      }
      print("PASS EMPAD load=\(loaded-started) save=\(saved-loaded) reopen=\(reopened-saved) bytes=\(restored.residentBytes) exact_frames=\(frames.count) BF_ADF_DF=pass")
    } else {
      let source: MetalRuntimeANSResidentSource
      if NativeANSSnapshot.matches(input) {
        source = try .load(snapshot: NativeANSSnapshot(url: input), device: device)
      } else if input.pathExtension.lowercased() == "dm4" {
        source = try .load(camera: NativeDM4Source(url: input), device: device)
      } else {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("qem-validation-index")
        let dataset = try Native4DSTEMCatalogBuilder(cacheDirectory: cache).prepare(input: input).datasets[0]
        source = try .load(source: Native4DSTEMIndexedSource.open(dataset: dataset), device: device, includeSpatialIndex: true)
      }
      let loaded = CFAbsoluteTimeGetCurrent()
      let expectedDataset = source.dataset
      let shape = source.shape, scans = shape[0] * shape[1]
      let frames = [0, 1, scans / 4, scans / 2, scans * 3 / 4, scans - 2, scans - 1]
      var expected = [String]()
      for frame in frames { expected.append(hash(try source.extractRawDiffraction(scanRow: frame / shape[1], scanColumn: frame % shape[1]))) }
      let products = try detectorHashes(source, frames: frames)
      let saveStart = CFAbsoluteTimeGetCurrent()
      try source.saveSnapshot(to: output)
      let saved = CFAbsoluteTimeGetCurrent()
      source.releaseResidentStorage()
      let file = try NativeANSSnapshot(url: output)
      try require(NativeANSSnapshot.matches(output), "Writer did not produce QEM")
      let restored = try MetalRuntimeANSResidentSource.load(snapshot: file, device: device)
      defer { restored.releaseResidentStorage() }
      let reopened = CFAbsoluteTimeGetCurrent()
      try require(restored.shape == shape, "Shape changed")
      try require(try detectorHashes(restored, frames: frames) == products, "Full BF/ADF/DF images changed")
      for (index, frame) in frames.enumerated() {
        try require(hash(try restored.extractRawDiffraction(scanRow: frame / shape[1], scanColumn: frame % shape[1])) == expected[index], "DP changed at \(frame)")
      }
      try require(file.dataset.sourceScanCalibration?.rowSamplingAngstrom == expectedDataset.sourceScanCalibration?.rowSamplingAngstrom, "Row calibration lost")
      try require(file.dataset.sourceScanCalibration?.columnSamplingAngstrom == expectedDataset.sourceScanCalibration?.columnSamplingAngstrom, "Column calibration lost")
      try require(file.dataset.kPixelSizeRow == expectedDataset.kPixelSizeRow && file.dataset.kPixelSizeCol == expectedDataset.kPixelSizeCol && file.dataset.kPixelUnit == expectedDataset.kPixelUnit, "Detector calibration lost")
      try require(file.dataset.badPixelIndices == expectedDataset.badPixelIndices, "Validity mask changed")
      let retained = file.metadata["source_metadata"] as? [String: String] ?? [:]
      for (key, value) in expectedDataset.metadata ?? [:] { try require(retained[key] == value, "Metadata lost: \(key)") }
      print("PASS INTEGER shape=\(shape) load=\(loaded-started) save=\(saved-saveStart) reopen=\(reopened-saved) bytes=\(restored.residentBytes) exact_frames=\(frames.count) full_BF_ADF_DF=pass metadata=\(retained.count)")
    }
  }
}
