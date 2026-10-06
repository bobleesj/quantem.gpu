import Foundation

/// Benchmark-only source-page control shared by exact hashing and indexed IO.
///
/// Applications do not select this control. The indexed-load benchmark enables
/// it explicitly so raw timing records can state how source pages were handled.
package enum Native4DSTEMBenchmarkSourcePageControl: String, Sendable {
  case unspecified
  case macOSFNoCacheHashAndIndexedSourceDescriptors =
    "macos_f_nocache_hash_and_indexed_source_descriptors"

  package static let environmentVariable =
    "QUANTEM_GPU_BENCHMARK_UNCACHED_SOURCE_READS"

  package init(uncachedSourceReads: Bool) {
    self =
      uncachedSourceReads
      ? .macOSFNoCacheHashAndIndexedSourceDescriptors
      : .unspecified
  }

  package static var current: Self {
    Self(
      uncachedSourceReads:
        ProcessInfo.processInfo.environment[environmentVariable] == "1"
    )
  }

  package var cacheStateComponent: String {
    switch self {
    case .unspecified:
      "source_pages_unspecified"
    case .macOSFNoCacheHashAndIndexedSourceDescriptors:
      "source_page_control_\(rawValue)"
    }
  }
}
