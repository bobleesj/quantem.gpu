import Dispatch

/// Per-command-buffer GPU timeline captured by ``MetalSSBEngine`` when a
/// timeline recorder is installed. Wall values are monotonic seconds since
/// process start; GPU values are `MTLCommandBuffer` timestamps in seconds.
public struct SSBCommandBufferTimeline: Sendable {
  public let label: String
  public let wallStart: Double
  public let wallEnd: Double
  public let gpuStart: Double
  public let gpuEnd: Double
  public let kernelStart: Double
  public let kernelEnd: Double
}

/// One objective evaluation: its wall span, the GPU busy time summed over its
/// command buffers, and the per-command-buffer records in submission order.
public struct SSBEvaluationTimeline: Sendable {
  public let index: Int
  public let wallStart: Double
  public let wallEnd: Double
  public let gpuSeconds: Double
  public let brightfieldCount: Int
  public let commandBuffers: [SSBCommandBufferTimeline]
}

/// Measurement-only recorder. The engine appends one command-buffer record per
/// commit/wait and one evaluation record per objective call. The loss math is
/// untouched: with no recorder installed the engine does one optional-chaining
/// check per commit and nothing else.
///
/// Thread-safety: the SSB loss path evaluates one objective at a time on the
/// calling thread; this helper is not a synchronization primitive.
public final class SSBTimelineRecorder: @unchecked Sendable {
  public var onEvaluation: ((SSBEvaluationTimeline) -> Void)?

  private var commandBuffers: [SSBCommandBufferTimeline] = []
  private var evaluationWallStart = 0.0
  private var evaluationIndex = 0
  private var active = false

  public init() {}

  @inline(__always)
  public static func now() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) * 1e-9
  }

  func beginEvaluation() {
    evaluationIndex += 1
    evaluationWallStart = Self.now()
    commandBuffers.removeAll(keepingCapacity: true)
    active = true
  }

  func recordCommandBuffer(
    label: String,
    wallStart: Double,
    wallEnd: Double,
    gpuStart: Double,
    gpuEnd: Double,
    kernelStart: Double,
    kernelEnd: Double
  ) {
    guard active else { return }
    commandBuffers.append(
      SSBCommandBufferTimeline(
        label: label,
        wallStart: wallStart,
        wallEnd: wallEnd,
        gpuStart: gpuStart,
        gpuEnd: gpuEnd,
        kernelStart: kernelStart,
        kernelEnd: kernelEnd
      ))
  }

  func finishEvaluation(
    wallEnd: Double,
    gpuSeconds: Double,
    brightfieldCount: Int
  ) {
    guard active else { return }
    active = false
    let record = SSBEvaluationTimeline(
      index: evaluationIndex,
      wallStart: evaluationWallStart,
      wallEnd: wallEnd,
      gpuSeconds: gpuSeconds,
      brightfieldCount: brightfieldCount,
      commandBuffers: commandBuffers
    )
    commandBuffers.removeAll(keepingCapacity: true)
    onEvaluation?(record)
  }
}
