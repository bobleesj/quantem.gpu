import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO
#if !SCIENTIFIC_NUMERICS_CHECK
import XCTest
#endif

/// Loading a file list preserves the existing indexed reader's corrected counts.
func checkEncodedSourceFiles(directory: URL) throws {
  guard let device = MTLCreateSystemDefaultDevice() else {
    throw Metal4DSTEMStreamingIOError.invalidRequest("This check requires a Metal device.")
  }
  let root = FileManager.default.temporaryDirectory.appendingPathComponent("encoded-files-" + UUID().uuidString)
  try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: root) }
  for name in ["fixture_master.h5", "fixture_data_000001.h5", "fixture_u8_data_000001.h5"] {
    try FileManager.default.copyItem(at: directory.appendingPathComponent(name),
      to: root.appendingPathComponent(name))
  }
  let files = ["fixture_master.h5", "fixture_u8_data_000001.h5"].map { root.appendingPathComponent($0) }
  let cache = root.appendingPathComponent("index")
  var visits: [Int] = []
  let loaded = try MetalEncodedSource.load(files: files, indexDirectory: cache, device: device,
    progress: { visits.append($0); _ = $1 })
  defer { loaded.forEach { $0.releaseResidentStorage() } }
  guard visits == [0, 1], loaded.count == files.count else {
    throw Metal4DSTEMStreamingIOError.invalidRequest("File order or load progress changed.")
  }
  for (file, actual) in zip(files, loaded) {
    let prepared = try Native4DSTEMCatalogBuilder(cacheDirectory: cache).prepare(input: file)
    let expected = try MetalEncodedSource.load(
      source: Native4DSTEMIndexedSource.open(dataset: prepared.datasets[0]), device: device)
    defer { expected.releaseResidentStorage() }
    let actualBytes = try actual.read(0..<actual.readyFrames)
    let expectedBytes = try expected.read(0..<expected.readyFrames)
    guard actual.shape == expected.shape, actual.itemBytes == expected.itemBytes,
      actual.hotPixelIndices == expected.hotPixelIndices, actual.sourceReadPasses == 1,
      actualBytes.length == expectedBytes.length,
      memcmp(actualBytes.contents(), expectedBytes.contents(), actualBytes.length) == 0
    else { throw Metal4DSTEMStreamingIOError.invalidRequest("File loading changed corrected counts or source metadata.") }
  }
  do {
    _ = try MetalEncodedSource.load(files: files, indexDirectory: cache, device: device,
      shouldCancel: { true })
    throw Metal4DSTEMStreamingIOError.invalidRequest("A cancelled file list started loading.")
  } catch Metal4DSTEMStreamingIOError.cancelled { }
  print("ENCODED_SOURCE_FILES_PASS files=2 uint8_uint16=true corrected_counts_exact=true cancelled=true")
}

#if !SCIENTIFIC_NUMERICS_CHECK
final class MetalEncodedHDF5LoadingTests: XCTestCase {
  func testFilesMatchIndexedLoading() throws {
    guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Requires Metal") }
    try checkEncodedSourceFiles(directory: Bundle.module.resourceURL!.appendingPathComponent("Fixtures"))
  }
}
#endif
