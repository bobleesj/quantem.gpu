import CryptoKit
import Foundation
import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

/// Full EMPAD measurements in lossless, randomly addressable Metal word packing.
///
/// This format preserves float32 bit patterns, not integer counts. It stores
/// each detector row's common XOR prefix and suffix plus the remaining bits.
/// Compression is data-dependent and can be slightly larger than float32 for
/// incompressible rows. It never quantizes or clips measurements.
///
/// Example: `try MetalEMPADResidentSource.load(source, device: device,
/// memoryBudgetBytes: budget)` followed by encoding a selected DP or mask sum.
/// Calls, release and command completion must be serialized by the owner.
public final class MetalEMPADResidentSource {
  public let source: NativeEMPADSource
  public let residentBytes: UInt64
  /// SHA-256 of all original little-endian float32 detector words in scan order.
  public let logicalSHA256: String
  /// Tensor identity binding shape and dtype to `logicalSHA256`, not a file checksum.
  public let sourceIdentitySHA256: String
  public private(set) var isReleased = false
  private let device: MTLDevice
  private let diffractionPipeline: MTLComputePipelineState
  private let detectorPipeline: MTLComputePipelineState
  private var chunks: [Chunk]

  private struct Chunk {
    let firstFrame: Int
    let frameCount: Int
    let payload: MTLBuffer
    let descriptors: MTLBuffer
  }

  private init(
    source: NativeEMPADSource, device: MTLDevice,
    diffraction: MTLComputePipelineState, detector: MTLComputePipelineState, chunks: [Chunk],
    logicalSHA256: String
  ) {
    self.source = source
    self.device = device
    self.diffractionPipeline = diffraction
    self.detectorPipeline = detector
    self.chunks = chunks
    self.logicalSHA256 = logicalSHA256
    var identity = SHA256()
    identity.update(data: Data("quantem.gpu.empad-tensor/v1\0float32-le\0".utf8))
    for dimension in [source.scanRows, source.scanColumns, 128, 128] {
      var word = UInt64(dimension).littleEndian
      withUnsafeBytes(of: &word) { identity.update(bufferPointer: $0) }
    }
    identity.update(data: Data(logicalSHA256.utf8))
    sourceIdentitySHA256 = identity.finalize().map { String(format: "%02x", $0) }.joined()
    residentBytes = chunks.reduce(0) { $0 + UInt64($1.payload.length + $1.descriptors.length) }
  }

  /// Read every original detector pixel and finish packing before returning.
  ///
  /// The budget caps total current Metal allocation plus the next bounded
  /// staging window. It is not a process-RSS or operating-system page-cache cap.
  /// No full dense host or device volume, prepared 2D preview or disk sidecar is
  /// used. Cancellation throws and releases partial residents.
  public static func load(
    _ source: NativeEMPADSource, device: MTLDevice, memoryBudgetBytes: UInt64,
    shouldCancel: () -> Bool = { false }
  ) throws -> MetalEMPADResidentSource {
    guard source.frameCount <= Int(UInt32.max) else {
      throw failure("EMPAD scan exceeds the supported frame-address range.")
    }
    try checkCancellation(shouldCancel)
    try source.validateUnchanged()
    let library = try Metal4DSTEMKernels.makeEMPADLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw failure("EMPAD kernel is missing: \(name). Rebuild the backend resources.")
      }
      return try device.makeComputePipelineState(function: function)
    }
    let describe = try pipeline("empad_describe")
    let pack = try pipeline("empad_pack")
    let diffraction = try pipeline("empad_diffraction")
    let detector = try pipeline("empad_virtual_image")
    guard let queue = device.makeCommandQueue() else {
      throw failure("Metal command queue is unavailable.")
    }
    var chunks: [Chunk] = []
    var logicalDigest = SHA256()
    for first in stride(from: 0, to: source.frameCount, by: 64) {
      try checkCancellation(shouldCancel)
      let frameCount = min(64, source.frameCount - first)
      let blocks = frameCount * 128
      let sourceBytes = frameCount * 16384 * 4
      let descriptorBytes = blocks * MemoryLayout<SIMD4<UInt32>>.stride
      // Includes worst-case payload and private descriptor copy before reading.
      let additionalBytes = UInt64(sourceBytes * 2 + descriptorBytes * 2)
      let allocatedBytes = UInt64(device.currentAllocatedSize)
      guard allocatedBytes <= memoryBudgetBytes,
        additionalBytes <= memoryBudgetBytes - allocatedBytes
      else {
        throw failure(
          "The full EMPAD resident exceeds the Metal memory budget. Close another dataset or open a smaller acquisition; no pixels were reduced."
        )
      }
      try autoreleasepool {
        let values = try source.readFrames(Array(first..<(first + frameCount)))
        values.withUnsafeBytes { logicalDigest.update(bufferPointer: $0) }
        guard
          let input = values.withUnsafeBytes({ bytes in
            device.makeBuffer(
              bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
          }),
          let descriptors = device.makeBuffer(length: descriptorBytes, options: .storageModeShared),
          let analysis = queue.makeCommandBuffer(),
          let encoder = analysis.makeComputeCommandEncoder()
        else { throw failure("Metal could not allocate the bounded EMPAD packing window.") }
        encoder.setComputePipelineState(describe)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(descriptors, offset: 0, index: 1)
        dispatch(encoder, pipeline: describe, count: blocks)
        encoder.endEncoding()
        try finish(analysis)
        try checkCancellation(shouldCancel)
        let entries = descriptors.contents().bindMemory(to: SIMD4<UInt32>.self, capacity: blocks)
        var words: UInt32 = 0
        for block in 0..<blocks {
          entries[block].w = words
          words += entries[block].y * 4
        }
        guard
          let payload = device.makeBuffer(
            length: max(4, Int(words) * 4), options: .storageModePrivate),
          let residentDescriptors = device.makeBuffer(
            length: descriptorBytes, options: .storageModePrivate),
          let command = queue.makeCommandBuffer(),
          let packEncoder = command.makeComputeCommandEncoder()
        else {
          throw failure("Metal could not allocate the exact EMPAD resident. Free memory and retry.")
        }
        packEncoder.setComputePipelineState(pack)
        packEncoder.setBuffer(input, offset: 0, index: 0)
        packEncoder.setBuffer(descriptors, offset: 0, index: 1)
        packEncoder.setBuffer(payload, offset: 0, index: 2)
        dispatch(packEncoder, pipeline: pack, count: blocks)
        packEncoder.endEncoding()
        guard let copy = command.makeBlitCommandEncoder() else {
          throw failure("Metal descriptor upload failed.")
        }
        copy.copy(
          from: descriptors, sourceOffset: 0, to: residentDescriptors,
          destinationOffset: 0, size: descriptorBytes)
        copy.endEncoding()
        try finish(command)
        try checkCancellation(shouldCancel)
        chunks.append(
          Chunk(
            firstFrame: first, frameCount: frameCount,
            payload: payload, descriptors: residentDescriptors))
      }
    }
    try source.validateUnchanged()
    return MetalEMPADResidentSource(
      source: source, device: device,
      diffraction: diffraction, detector: detector, chunks: chunks,
      logicalSHA256: logicalDigest.finalize().map { String(format: "%02x", $0) }.joined())
  }

  /// Encode a complete selected float32 DP without reading it back to the CPU.
  /// Output needs 128×128×4 bytes. The caller waits for its command completion
  /// before publishing the frame or reusing the output buffer.
  public func encodeDiffraction(
    scanRow: Int, scanColumn: Int, into output: MTLBuffer, command: MTLCommandBuffer
  ) throws {
    guard !isReleased, (0..<source.scanRows).contains(scanRow),
      (0..<source.scanColumns).contains(scanColumn), output.length >= 16384 * 4,
      command.commandQueue.device.registryID == device.registryID,
      output.device.registryID == device.registryID
    else {
      throw Self.failure(
        "EMPAD diffraction needs a resident source, valid scan coordinates and a same-device 128×128 float32 output."
      )
    }
    let frame = scanRow * source.scanColumns + scanColumn
    guard
      let chunk = chunks.first(where: {
        ($0.firstFrame..<($0.firstFrame + $0.frameCount)).contains(frame)
      }),
      let encoder = command.makeComputeCommandEncoder()
    else { throw Self.failure("EMPAD selected-frame encoding failed.") }
    var local = UInt32(frame - chunk.firstFrame)
    encoder.setComputePipelineState(diffractionPipeline)
    encoder.setBuffer(chunk.payload, offset: 0, index: 0)
    encoder.setBuffer(chunk.descriptors, offset: 0, index: 1)
    encoder.setBuffer(output, offset: 0, index: 2)
    encoder.setBytes(&local, length: 4, index: 3)
    Self.dispatch(encoder, pipeline: diffractionPipeline, count: 16384)
    encoder.endEncoding()
  }

  /// Encode mask integration over all scan positions, directly from packing.
  ///
  /// Any nonzero uint8 mask entry selects that detector pixel. Output is
  /// float32 in scan-row-major order. Selected NaNs propagate; unselected
  /// pixels do not participate. Output requires `source.frameCount * 4` bytes.
  public func encodeVirtualImage(
    mask: MTLBuffer, into output: MTLBuffer, command: MTLCommandBuffer
  ) throws {
    guard !isReleased, mask !== output, mask.length >= 16384,
      output.length >= source.frameCount * 4,
      command.commandQueue.device.registryID == device.registryID,
      mask.device.registryID == device.registryID, output.device.registryID == device.registryID
    else {
      throw Self.failure(
        "EMPAD integration needs a resident source, a same-device 128×128 uint8 mask and a separate full-scan float32 output."
      )
    }
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw Self.failure("EMPAD detector encoding failed.")
    }
    encoder.setComputePipelineState(detectorPipeline)
    encoder.setBuffer(mask, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)
    for chunk in chunks {
      var offset = UInt32(chunk.firstFrame)
      encoder.setBuffer(chunk.payload, offset: 0, index: 0)
      encoder.setBuffer(chunk.descriptors, offset: 0, index: 1)
      encoder.setBytes(&offset, length: 4, index: 4)
      Self.dispatch(encoder, pipeline: detectorPipeline, count: chunk.frameCount)
    }
    encoder.endEncoding()
  }

  /// Release after outstanding commands finish. Encoding after release fails.
  public func releaseResidentStorage() {
    chunks.removeAll()
    isReleased = true
  }

  private static func dispatch(
    _ encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState, count: Int
  ) {
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(32, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
  }

  private static func finish(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    if command.status != .completed {
      throw failure(
        "EMPAD Metal command failed: \(command.error?.localizedDescription ?? "unknown GPU error")."
      )
    }
  }

  private static func checkCancellation(_ cancelled: () -> Bool) throws {
    if cancelled() { throw CancellationError() }
  }

  private static func failure(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest(message)
  }
}
