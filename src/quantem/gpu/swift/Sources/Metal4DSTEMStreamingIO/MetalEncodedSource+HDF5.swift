import Foundation
import Metal
import Native4DSTEMIO

extension MetalEncodedSource {
  /// Load original HDF5 acquisitions into encoded residency in file order.
  /// Each file must identify one acquisition. Correction and bounded encoding
  /// use the same native reader as `load(source:device:shouldCancel:)`.
  /// The caller owns the returned residents and releases them when finished.
  ///
  /// Example: `try MetalEncodedSource.load(files: files, indexDirectory: cache, device: device)`.
  public static func load(
    files: [URL], indexDirectory: URL, device: MTLDevice,
    shouldCancel: () -> Bool = { false }, progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> [MetalEncodedSource] {
    let catalog = Native4DSTEMCatalogBuilder(cacheDirectory: indexDirectory)
    return try files.enumerated().map { index, file in
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      progress(index, files.count)
      let prepared = try catalog.prepare(input: file)
      guard prepared.datasets.count == 1 else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Each file must identify one 4D acquisition.")
      }
      return try MetalEncodedSource.load(
        source: Native4DSTEMIndexedSource.open(dataset: prepared.datasets[0]), device: device,
        shouldCancel: shouldCancel)
    }
  }
}
