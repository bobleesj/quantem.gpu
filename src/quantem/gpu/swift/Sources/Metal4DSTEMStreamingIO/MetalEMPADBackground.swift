import CryptoKit
import Foundation
import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

/// Arithmetic mean of a confirmed dark acquisition, in the sample's signal units.
///
/// The caller confirms that the sample is not already corrected and that the
/// reference uses the same detector, exposure, gain and preprocessing. This is
/// uniform mean-dark subtraction, not encoded EMPAD2 even/odd calibration.
/// Example: `try MetalEMPADBackground.load(dark, device: device, memoryBudgetBytes: budget)`.
public final class MetalEMPADBackground {
  public static let schema = "empad-mean-dark/v1"
  public let source: NativeEMPADSource
  public let identitySHA256: String
  package let values: MTLBuffer

  private init(source: NativeEMPADSource, values: MTLBuffer, identity: String) {
    self.source = source
    self.values = values
    self.identitySHA256 = identity
  }

  /// Stream a dark acquisition through a bounded GPU mean; never expand its cube.
  public static func load(
    _ source: NativeEMPADSource, device: MTLDevice,
    memoryBudgetBytes: UInt64, shouldCancel: () -> Bool = { false }
  ) throws -> MetalEMPADBackground {
    guard source.frameCount <= Int(UInt32.max) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Dark acquisition exceeds the supported frame range. Choose a smaller reference.")
    }
    if shouldCancel() { throw CancellationError() }
    let library = try Metal4DSTEMKernels.makeEMPADLibrary(device: device)
    guard let function = library.makeFunction(name: "empad_dark_mean"),
      UInt64(device.currentAllocatedSize) + 256 * 65536 + 196608 <= memoryBudgetBytes,
      let queue = device.makeCommandQueue(),
      let input = device.makeBuffer(length: 256 * 65536, options: .storageModeShared),
      let accumulator = device.makeBuffer(length: 16384 * 8, options: .storageModePrivate),
      let output = device.makeBuffer(length: 65536, options: .storageModeShared)
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Not enough Metal memory for background correction. Close another dataset and retry.")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    var digest = SHA256()
    digest.update(data: Data((schema + "\0").utf8))
    for first in stride(from: 0, to: source.frameCount, by: 256) {
      if shouldCancel() { throw CancellationError() }
      let count = min(256, source.frameCount - first)
      let bytes = UnsafeMutableRawBufferPointer(start: input.contents(), count: count * 65536)
      try source.readFrames(Array(first..<(first + count)), into: bytes)
      digest.update(bufferPointer: UnsafeRawBufferPointer(bytes))
      guard let command = queue.makeCommandBuffer(),
        let encoder = command.makeComputeCommandEncoder()
      else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Cannot create background correction command. Retry loading.")
      }
      var dimensions = SIMD3<UInt32>(UInt32(first), UInt32(count), UInt32(source.frameCount))
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(input, offset: 0, index: 0)
      encoder.setBuffer(accumulator, offset: 0, index: 1)
      encoder.setBuffer(output, offset: 0, index: 2)
      encoder.setBytes(&dimensions, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
      encoder.dispatchThreads(
        MTLSize(width: 16384, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Background mean failed: \(String(describing: command.error))")
      }
    }
    try source.validateUnchanged()
    if shouldCancel() { throw CancellationError() }
    // Only a 64 KiB calibration is checked on the host, never the full cube.
    let pixels = UnsafeBufferPointer(
      start: output.contents().assumingMemoryBound(to: Float.self), count: 16384)
    guard pixels.allSatisfy(\.isFinite) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Background contains non-finite measurements. Choose a finite dark reference; no correction was applied."
      )
    }
    digest.update(bufferPointer: UnsafeRawBufferPointer(start: output.contents(), count: 65536))
    return MetalEMPADBackground(
      source: source, values: output,
      identity: digest.finalize().map { String(format: "%02x", $0) }.joined())
  }
}
