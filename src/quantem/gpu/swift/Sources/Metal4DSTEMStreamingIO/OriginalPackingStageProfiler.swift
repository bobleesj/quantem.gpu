import Darwin
import Foundation
import Metal

/// Opt-in diagnostics for the existing one-command original-count loader.
/// Sampling can perturb scheduling; these records are not uninstrumented timings.
final class OriginalPackingStageProfiler {
  private let device: MTLDevice
  private let counterSet: MTLCounterSet?
  private let unavailableReason: String?
  private let expectedWindows: Int
  private var recordedWindows = 0, validWindows = 0
  private var totals: [String: Double] = [:]
  private var commandSeconds = 0.0

  static func makeIfRequested(device: MTLDevice, expectedWindows: Int)
    -> OriginalPackingStageProfiler?
  {
    guard OriginalPackingDiagnostics.enabled("STAGE_PROFILE") else { return nil }
    return OriginalPackingStageProfiler(device: device, expectedWindows: expectedWindows)
  }

  private init(device: MTLDevice, expectedWindows: Int) {
    self.device = device
    self.expectedWindows = expectedWindows
    if !device.supportsCounterSampling(.atStageBoundary) {
      counterSet = nil
      unavailableReason = "stage-boundary sampling is unsupported"
    } else if let found = device.counterSets?.first(where: {
      $0.name == MTLCommonCounterSet.timestamp.rawValue
    }),
      found.counters.contains(where: { $0.name == MTLCommonCounter.timestamp.rawValue })
    {
      counterSet = found
      unavailableReason = nil
    } else {
      counterSet = nil
      unavailableReason = "timestamp counter set is unavailable"
    }
    emit([
      "record": "capability", "status": unavailableReason == nil ? "available" : "unsupported",
      "reason": unavailableReason ?? "", "device": device.name, "registry_id": device.registryID,
      "expected_windows": expectedWindows, "sampling": "existing_encoder_boundaries",
      "diagnostic_only": true,
    ])
  }

  struct Calibration {
    let cpu, gpu: UInt64

    init(device: MTLDevice) {
      let timestamps = device.sampleTimestamps()
      cpu = timestamps.cpu
      gpu = timestamps.gpu
    }

    var json: [String: Any] {
      ["cpu_nanoseconds": cpu, "gpu_ticks": gpu]
    }
  }

  final class Window {
    let ordinal: Int
    let sliceFrames: [Int]
    let stages: [String]
    let samples: MTLCounterSampleBuffer?
    var invalidReason: String?
    var before: Calibration?

    init(ordinal: Int, sliceFrames: [Int], samples: MTLCounterSampleBuffer?, reason: String?) {
      self.ordinal = ordinal
      self.sliceFrames = sliceFrames
      self.samples = samples
      invalidReason = reason
      stages =
        Array(repeating: "checked_lz4", count: sliceFrames.count)
        + [
          "range_validation", "gather_pack_verify_partials", "summary_reduction",
          "retained_header_blit",
        ]
    }

    func computeEncoder(_ command: MTLCommandBuffer, stage: Int) -> MTLComputeCommandEncoder? {
      guard let samples, invalidReason == nil else { return command.makeComputeCommandEncoder() }
      let descriptor = MTLComputePassDescriptor()
      if let attachment = descriptor.sampleBufferAttachments[0] {
        attachment.sampleBuffer = samples
        attachment.startOfEncoderSampleIndex = stage * 2
        attachment.endOfEncoderSampleIndex = stage * 2 + 1
        if let encoder = command.makeComputeCommandEncoder(descriptor: descriptor) {
          return encoder
        }
      }
      invalidReason = "could not create sampled compute encoder"
      return command.makeComputeCommandEncoder()
    }

    func blitEncoder(_ command: MTLCommandBuffer) -> MTLBlitCommandEncoder? {
      guard let samples, invalidReason == nil else { return command.makeBlitCommandEncoder() }
      let descriptor = MTLBlitPassDescriptor()
      if let attachment = descriptor.sampleBufferAttachments[0] {
        attachment.sampleBuffer = samples
        attachment.startOfEncoderSampleIndex = (stages.count - 1) * 2
        attachment.endOfEncoderSampleIndex = (stages.count - 1) * 2 + 1
        if let encoder = command.makeBlitCommandEncoder(descriptor: descriptor) { return encoder }
      }
      invalidReason = "could not create sampled blit encoder"
      return command.makeBlitCommandEncoder()
    }
  }

  func makeWindow(ordinal: Int, sliceFrames: [Int]) -> Window {
    var samples: MTLCounterSampleBuffer?
    var reason = unavailableReason
    if let counterSet {
      let descriptor = MTLCounterSampleBufferDescriptor()
      descriptor.counterSet = counterSet
      descriptor.storageMode = .shared
      descriptor.sampleCount = 2 * (sliceFrames.count + 4)
      do { samples = try device.makeCounterSampleBuffer(descriptor: descriptor) } catch {
        reason = "counter allocation failed: \(error.localizedDescription)"
      }
    }
    return Window(ordinal: ordinal, sliceFrames: sliceFrames, samples: samples, reason: reason)
  }

  func willCommit(_ window: Window?) {
    guard let window, window.samples != nil else { return }
    window.before = Calibration(device: device)
  }

  func record(_ window: Window?, command: MTLCommandBuffer, countErrors: UInt32) {
    guard let window else { return }
    recordedWindows += 1
    let after = window.samples == nil ? nil : Calibration(device: device)
    let gpuSeconds = command.gpuEndTime - command.gpuStartTime
    var row: [String: Any] = [
      "record": "window", "window": window.ordinal,
      "slice_frames": window.sliceFrames, "stages": window.stages,
      "command_gpu_start_seconds": command.gpuStartTime,
      "command_gpu_end_seconds": command.gpuEndTime, "command_gpu_seconds": gpuSeconds,
      "command_status": command.status.rawValue, "count_errors": countErrors,
    ]
    if let before = window.before { row["calibration_before"] = before.json }
    if let after { row["calibration_after"] = after.json }
    var reason = window.invalidReason
    var timestamps: [UInt64] = []
    if let samples = window.samples, reason == nil {
      do {
        if let data = try samples.resolveCounterRange(0..<window.stages.count * 2),
          data.count == window.stages.count * 2 * MemoryLayout<MTLCounterResultTimestamp>.stride
        {
          timestamps = data.withUnsafeBytes {
            Array($0.bindMemory(to: MTLCounterResultTimestamp.self)).map(\.timestamp)
          }
        } else {
          reason = "missing or incorrectly sized resolved timestamps"
        }
      } catch { reason = "counter resolve failed: \(error.localizedDescription)" }
    }
    row["raw_gpu_timestamps"] = timestamps
    if reason == nil {
      if command.status != .completed || !gpuSeconds.isFinite || gpuSeconds <= 0 {
        reason = "command did not return a positive completed GPU interval"
      } else if timestamps.contains(where: { $0 == 0 || $0 == UInt64.max }) {
        reason = "zero or error-valued counter sample"
      } else if (0..<window.stages.count).contains(where: {
        timestamps[$0 * 2 + 1] <= timestamps[$0 * 2]
      }) {
        reason = "zero-length or reversed encoder counter interval"
      }
    }
    var secondsPerTick: Double?
    if reason == nil, let before = window.before, let after,
      before.cpu > 0, before.gpu > 0, after.cpu > before.cpu, after.gpu > before.gpu
    {
      // Metal sampleTimestamps CPU values are already nanoseconds, not raw
      // mach_absolute_time ticks. Calibrate the GPU tick scale for this window.
      let cpuSeconds = Double(after.cpu - before.cpu) * 1e-9
      let scale = cpuSeconds / Double(after.gpu - before.gpu)
      if scale.isFinite && scale > 0,
        timestamps.allSatisfy({ $0 >= before.gpu && $0 <= after.gpu })
      {
        secondsPerTick = scale
      } else {
        reason = "counter timestamps are outside the calibration bracket"
      }
    } else if reason == nil {
      reason = "missing or invalid CPU/GPU timestamp calibration"
    }
    if let secondsPerTick, reason == nil {
      var intervals: [Double] = []
      var sums: [String: Double] = [:]
      for (index, stage) in window.stages.enumerated() {
        let seconds = Double(timestamps[index * 2 + 1] - timestamps[index * 2]) * secondsPerTick
        intervals.append(seconds)
        sums[stage, default: 0] += seconds
        totals[stage, default: 0] += seconds
      }
      validWindows += 1
      commandSeconds += gpuSeconds
      row["calibrated_seconds_per_gpu_tick"] = secondsPerTick
      row["encoder_seconds"] = intervals
      row["stage_seconds"] = sums
      row["unattributed_command_seconds"] = gpuSeconds - intervals.reduce(0, +)
      row["status"] = "valid"
    } else {
      row["status"] = unavailableReason == nil ? "invalid" : "unsupported"
    }
    row["reason"] = reason ?? ""
    emit(row)
  }

  func reportSummary() {
    emit([
      "record": "summary", "expected_windows": expectedWindows,
      "recorded_windows": recordedWindows, "valid_windows": validWindows,
      "invalid_or_unsupported_windows": recordedWindows - validWindows,
      "missing_windows": expectedWindows - recordedWindows,
      "valid_window_stage_seconds": totals, "valid_window_command_seconds": commandSeconds,
      "valid_window_unattributed_seconds": commandSeconds - totals.values.reduce(0, +),
      "status": validWindows == expectedWindows ? "complete" : "incomplete",
    ])
  }

  private func emit(_ row: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]),
      let text = String(data: data, encoding: .utf8)
    else { return }
    fputs("QGPU_ORIGINAL_STAGE_PROFILE \(text)\n", stderr)
  }
}
