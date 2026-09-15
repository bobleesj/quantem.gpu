import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMStreamingIO
import Native4DSTEMIO

private enum ConcurrencyProbeError: Error { case failed(String) }

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
  if !condition() { throw ConcurrencyProbeError.failed(message) }
}

@main
@available(macOS 15.0, *)
struct ResidentConcurrencyProbe {
  static func main() async throws {
    let arguments = Array(CommandLine.arguments.dropFirst())
    guard arguments.count == 2 else {
      throw ConcurrencyProbeError.failed("Usage: resident-concurrency-probe FOLDER INDEX_DIRECTORY")
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw ConcurrencyProbeError.failed("Metal unavailable")
    }
    setenv("QGPU_PAIRED_RUNTIME_POLAR_INDEX", "1", 1)
    setenv("QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS", "64", 1)
    let catalog = try Native4DSTEMCatalogBuilder(
      cacheDirectory: URL(fileURLWithPath: arguments[1])
    ).prepare(input: URL(fileURLWithPath: arguments[0]))
    guard let dataset = catalog.datasets.first else {
      throw ConcurrencyProbeError.failed("Fixture contains no acquisition")
    }
    let indexed = try Native4DSTEMIndexedSource.open(dataset: dataset)
    let resident = try MetalPairedRuntimeTANSResidentSource.load(
      source: indexed, device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory)
    let fullValid = resident.detectorValidityMask

    // Two identical updates begin from the same apparent zero state. Correct
    // serialization makes one apply the delta and the other observe no change;
    // both must publish the same complete product. DP and mode reads share the
    // resident concurrently and must not reset its failure buffer mid-query.
    async let updateA = resident.updateVirtualDetector(mask: fullValid)
    async let updateB = resident.updateVirtualDetector(mask: fullValid)
    async let diffraction = resident.extractRawDiffraction(scanRow: 0, scanColumn: 0)
    async let modes = resident.detectorStreamModeCounts(mask: fullValid)
    let (first, second, dp, histogram) = try await (updateA, updateB, diffraction, modes)
    try require(first.values == second.values,
      "Concurrent identical detector updates published different products")
    try require(dp.count == fullValid.count, "Concurrent diffraction extent changed")
    try require(histogram.reduce(UInt64(0), { $0 + UInt64($1) }) > 0,
      "Concurrent mode histogram is empty")
    try require(resident.lastPolarFieldCount > 0 && resident.lastPolarResidualCount == 0,
      "Full-valid-mask update did not exercise the polar-only path")

    // Restore a nontrivial pending update, then race that synchronous update
    // with release. Lock ordering may let either operation win; release must
    // ultimately own the terminal state without crashes or use-after-release.
    _ = try resident.updateVirtualDetector(mask: [UInt8](repeating: 0, count: fullValid.count))
    async let updateWon: Bool = {
      do {
        _ = try resident.updateVirtualDetector(mask: fullValid)
        return true
      } catch {
        return false
      }
    }()
    async let release: Void = resident.releaseResidentStorage()
    let (completedBeforeRelease, _) = await (updateWon, release)
    try require(resident.isReleased && resident.residentBytes == 0,
      "Concurrent release did not leave a terminal empty resident")
    do {
      _ = try resident.extractRawDiffraction(scanRow: 0, scanColumn: 0)
      throw ConcurrencyProbeError.failed("Released resident accepted a new query")
    } catch is ConcurrencyProbeError {
      throw ConcurrencyProbeError.failed("Released resident accepted a new query")
    } catch {
      // Expected public invalid-request error.
    }
    print("PASS same-resident serialization, polar-only update, and release race; update-before-release=\(completedBeforeRelease)")
  }
}
