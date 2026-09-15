import Foundation
import Metal

/// Opt-in timestamp-only diagnostics for one paired-runtime detector command.
///
/// The profiler samples the command's existing compute-encoder boundaries. It
/// does not add a command buffer, a wait, or a dispatch, so the normal queue
/// topology remains intact. Sampling is diagnostic and may perturb scheduling.
final class PairedRuntimeDetectorProfiler {
  struct StageResult {
    let valid: Bool
    let reason: String?
    let stageNanoseconds: [String: Double]
    let commandNanoseconds: Double
    let encoderUnionNanoseconds: Double
    /// Absolute calibrated CPU-second intervals, keyed by encoder occurrence.
    /// The occurrence suffix preserves separate intervals when a stage name is
    /// used more than once in a command; callers can union these across
    /// resident queues without summing overlapping stage durations.
    let stageTimeline: [String: Double]
    let rawTimestamps: [UInt64]
    let calibratedNanosecondsPerTick: Double?
  }

  /// State retained until the command has completed and its samples resolve.
  final class Session {
    private struct Calibration {
      let cpuNanoseconds: UInt64
      let gpuTicks: UInt64
    }

    private let device: MTLDevice
    private let maxEncoders: Int
    private(set) var sampleBuffer: MTLCounterSampleBuffer?
    private var stageNames: [String] = []
    private var before: Calibration?
    private var invalidReason: String?

    fileprivate init(
      device: MTLDevice, counterSet: MTLCounterSet?, maxEncoders: Int,
      allocationError: String?
    ) {
      self.device = device
      self.maxEncoders = maxEncoders
      invalidReason = allocationError
      guard let counterSet, allocationError == nil else { return }
      let descriptor = MTLCounterSampleBufferDescriptor()
      descriptor.counterSet = counterSet
      descriptor.storageMode = .shared
      descriptor.sampleCount = 2 * maxEncoders
      do {
        sampleBuffer = try device.makeCounterSampleBuffer(descriptor: descriptor)
      } catch {
        invalidReason = "counter allocation failed: \(error.localizedDescription)"
      }
    }

    /// Create an encoder with stage-boundary timestamp attachments.
    ///
    /// The caller uses this at boundaries it already has (for example, each
    /// polar query encoder and the residual owner2 encoder). If attachment
    /// creation fails, an ordinary encoder is returned and the session is
    /// marked invalid rather than changing command ordering.
    func makeComputeEncoder(
      commandBuffer: MTLCommandBuffer, stage: String
    ) -> MTLComputeCommandEncoder? {
      guard let sampleBuffer, invalidReason == nil else {
        return commandBuffer.makeComputeCommandEncoder()
      }
      guard stageNames.count < maxEncoders else {
        invalidReason = "timestamp sample capacity exceeded"
        return commandBuffer.makeComputeCommandEncoder()
      }
      let descriptor = MTLComputePassDescriptor()
      guard let attachment = descriptor.sampleBufferAttachments[0] else {
        invalidReason = "could not create timestamp counter attachment"
        return commandBuffer.makeComputeCommandEncoder()
      }
      let sampleIndex = stageNames.count * 2
      attachment.sampleBuffer = sampleBuffer
      attachment.startOfEncoderSampleIndex = sampleIndex
      attachment.endOfEncoderSampleIndex = sampleIndex + 1
      guard let encoder = commandBuffer.makeComputeCommandEncoder(descriptor: descriptor) else {
        invalidReason = "could not create sampled compute encoder"
        return commandBuffer.makeComputeCommandEncoder()
      }
      stageNames.append(stage)
      return encoder
    }

    /// Capture the CPU/GPU calibration immediately before command submission.
    func willCommit() {
      guard sampleBuffer != nil, invalidReason == nil else { return }
      let timestamps = device.sampleTimestamps()
      before = Calibration(cpuNanoseconds: timestamps.cpu, gpuTicks: timestamps.gpu)
    }

    /// Resolve and calibrate all attached stage intervals after completion.
    func record(command: MTLCommandBuffer) -> StageResult {
      let commandNanoseconds = max(0, command.gpuEndTime - command.gpuStartTime) * 1.0e9
      guard let sampleBuffer else {
        return StageResult(
          valid: false, reason: invalidReason ?? "timestamp sampling unavailable",
          stageNanoseconds: [:], commandNanoseconds: commandNanoseconds,
          encoderUnionNanoseconds: 0, stageTimeline: [:], rawTimestamps: [],
          calibratedNanosecondsPerTick: nil)
      }
      guard invalidReason == nil else {
        return StageResult(
          valid: false, reason: invalidReason, stageNanoseconds: [:],
          commandNanoseconds: commandNanoseconds, encoderUnionNanoseconds: 0,
          stageTimeline: [:], rawTimestamps: [], calibratedNanosecondsPerTick: nil)
      }
      let sampleCount = stageNames.count * 2
      guard sampleCount > 0 else {
        return StageResult(
          valid: false, reason: "no profiled compute encoders", stageNanoseconds: [:],
          commandNanoseconds: commandNanoseconds, encoderUnionNanoseconds: 0,
          stageTimeline: [:], rawTimestamps: [], calibratedNanosecondsPerTick: nil)
      }
      let data: Data
      do {
        guard let resolved = try sampleBuffer.resolveCounterRange(0..<sampleCount),
          resolved.count == sampleCount * MemoryLayout<MTLCounterResultTimestamp>.stride
        else {
          return StageResult(
            valid: false, reason: "missing or incorrectly sized resolved timestamps",
            stageNanoseconds: [:], commandNanoseconds: commandNanoseconds,
            encoderUnionNanoseconds: 0, stageTimeline: [:], rawTimestamps: [],
            calibratedNanosecondsPerTick: nil)
        }
        data = resolved
      } catch {
        return StageResult(
          valid: false, reason: "counter resolve failed: \(error.localizedDescription)",
          stageNanoseconds: [:], commandNanoseconds: commandNanoseconds,
          encoderUnionNanoseconds: 0, stageTimeline: [:], rawTimestamps: [],
          calibratedNanosecondsPerTick: nil)
      }
      let timestamps = data.withUnsafeBytes {
        Array($0.bindMemory(to: MTLCounterResultTimestamp.self)).map(\.timestamp)
      }
      guard command.status == .completed, commandNanoseconds > 0,
        timestamps.count == sampleCount,
        timestamps.allSatisfy({ $0 != 0 && $0 != UInt64.max }),
        stride(from: 0, to: sampleCount, by: 2).allSatisfy({ timestamps[$0 + 1] > timestamps[$0] })
      else {
        return StageResult(
          valid: false, reason: "invalid command or timestamp interval",
          stageNanoseconds: [:], commandNanoseconds: commandNanoseconds,
          encoderUnionNanoseconds: 0, stageTimeline: [:], rawTimestamps: timestamps,
          calibratedNanosecondsPerTick: nil)
      }
      guard let before else {
        return StageResult(
          valid: false, reason: "missing CPU/GPU calibration before commit",
          stageNanoseconds: [:], commandNanoseconds: commandNanoseconds,
          encoderUnionNanoseconds: 0, stageTimeline: [:], rawTimestamps: timestamps,
          calibratedNanosecondsPerTick: nil)
      }
      let afterValues = device.sampleTimestamps()
      let after = Calibration(cpuNanoseconds: afterValues.cpu, gpuTicks: afterValues.gpu)
      guard after.cpuNanoseconds > before.cpuNanoseconds,
        after.gpuTicks > before.gpuTicks
      else {
        return StageResult(
          valid: false, reason: "invalid CPU/GPU calibration interval",
          stageNanoseconds: [:], commandNanoseconds: commandNanoseconds,
          encoderUnionNanoseconds: 0, stageTimeline: [:], rawTimestamps: timestamps,
          calibratedNanosecondsPerTick: nil)
      }
      let scale =
        Double(after.cpuNanoseconds - before.cpuNanoseconds)
        / Double(after.gpuTicks - before.gpuTicks)
      guard scale.isFinite, scale > 0,
        timestamps.allSatisfy({ $0 >= before.gpuTicks && $0 <= after.gpuTicks })
      else {
        return StageResult(
          valid: false, reason: "counter timestamps are outside calibration bracket",
          stageNanoseconds: [:], commandNanoseconds: commandNanoseconds,
          encoderUnionNanoseconds: 0, stageTimeline: [:], rawTimestamps: timestamps,
          calibratedNanosecondsPerTick: nil)
      }
      var stageNanoseconds: [String: Double] = [:]
      var stageTimeline: [String: Double] = [:]
      var spans: [(start: UInt64, end: UInt64)] = []
      for (index, stage) in stageNames.enumerated() {
        let start = timestamps[index * 2]
        let end = timestamps[index * 2 + 1]
        stageNanoseconds[stage, default: 0] += Double(end - start) * scale
        let startSeconds =
          Double(before.cpuNanoseconds) * 1.0e-9
          + Double(start - before.gpuTicks) * scale * 1.0e-9
        let endSeconds =
          Double(before.cpuNanoseconds) * 1.0e-9
          + Double(end - before.gpuTicks) * scale * 1.0e-9
        stageTimeline["gpu_\(stage)_\(index)_start_seconds"] = startSeconds
        stageTimeline["gpu_\(stage)_\(index)_end_seconds"] = endSeconds
        spans.append((start: start, end: end))
      }
      stageTimeline["gpu_calibration_before_seconds"] =
        Double(before.cpuNanoseconds) * 1.0e-9
      stageTimeline["gpu_calibration_after_seconds"] =
        Double(after.cpuNanoseconds) * 1.0e-9
      spans.sort { $0.start < $1.start }
      var unionTicks: UInt64 = 0
      var unionStart = spans[0].start
      var unionEnd = spans[0].end
      for span in spans.dropFirst() {
        if span.start > unionEnd {
          unionTicks += unionEnd - unionStart
          unionStart = span.start
          unionEnd = span.end
        } else {
          unionEnd = max(unionEnd, span.end)
        }
      }
      unionTicks += unionEnd - unionStart
      return StageResult(
        valid: true, reason: nil, stageNanoseconds: stageNanoseconds,
        commandNanoseconds: commandNanoseconds,
        encoderUnionNanoseconds: Double(unionTicks) * scale,
        stageTimeline: stageTimeline,
        rawTimestamps: timestamps, calibratedNanosecondsPerTick: scale)
    }
  }

  private let device: MTLDevice
  private let counterSet: MTLCounterSet?
  private let unavailableReason: String?

  static func makeIfRequested(device: MTLDevice) -> PairedRuntimeDetectorProfiler? {
    guard ProcessInfo.processInfo.environment["QGPU_PAIRED_RUNTIME_PROFILE"] == "1"
    else { return nil }
    return PairedRuntimeDetectorProfiler(device: device)
  }

  private init(device: MTLDevice) {
    self.device = device
    if !device.supportsCounterSampling(.atStageBoundary) {
      counterSet = nil
      unavailableReason = "stage-boundary sampling is unsupported"
    } else if let found = device.counterSets?.first(where: {
      $0.name == MTLCommonCounterSet.timestamp.rawValue
    }), found.counters.contains(where: { $0.name == MTLCommonCounter.timestamp.rawValue }) {
      counterSet = found
      unavailableReason = nil
    } else {
      counterSet = nil
      unavailableReason = "timestamp counter set is unavailable"
    }
  }

  /// Make one bounded per-command session. Thirty-two existing encoders use
  /// 64 opaque timestamp samples; paired polar+residual queries use fewer.
  func makeSession(maxEncoders: Int = 32) -> Session {
    let bounded = min(max(1, maxEncoders), 32)
    return Session(
      device: device, counterSet: counterSet, maxEncoders: bounded,
      allocationError: unavailableReason)
  }
}
