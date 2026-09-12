import Foundation
import Metal

/// Diagnostic modes are never scientific images and must not advance source state.
enum TANSSubmissionProbeMode: UInt32 {
  case bindings = 1
  case denseWords = 2
}

struct TANSSubmissionProbeResult {
  let timing: [String: Double]
  /// Per-record checksum, dense word count, stream count and binding marker.
  let records: [UInt32]
}

struct TANSSubmissionProbeCompleted: Error {}
