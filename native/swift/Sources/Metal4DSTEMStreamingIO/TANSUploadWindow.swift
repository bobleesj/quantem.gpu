import Foundation
import Metal

/// Caller-serialized bounded upload window. Each joined worker exclusively owns
/// its staging buffer. Immutable archive metadata is shared; result slots are
/// lock protected. No GPU command outlives its worker or an error return.
final class TANSUploadWindow: @unchecked Sendable {
  struct Loaded {
    let resident: MTLBuffer
    let metrics: TANSArchive.ReadMetrics
    let readSeconds: Double
    let uploadSeconds: Double
  }

  private let archive: TANSArchive
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let stages: [MTLBuffer]
  private let readPolicy: TANSArchive.SourceReadPolicy
  private let lock = NSLock()
  private var results: [Result<Loaded, Error>?] = []

  init(
    archive: TANSArchive, device: MTLDevice, queue: MTLCommandQueue,
    stageBytes: Int, concurrency: Int,
    readPolicy: TANSArchive.SourceReadPolicy = .systemCache
  ) throws {
    self.archive = archive
    self.device = device
    self.queue = queue
    self.readPolicy = readPolicy
    stages = try (0..<concurrency).map { slot in
      guard let buffer = device.makeBuffer(length: stageBytes, options: .storageModeShared) else {
        throw TANSArchive.invalid("Cannot allocate bounded encoded transfer staging")
      }
      buffer.label = "tANS compressed transfer slot \(slot)"
      return buffer
    }
  }

  func load(_ chunks: [TANSArchive.Chunk]) throws -> [Loaded] {
    precondition(chunks.count <= stages.count)
    results = Array(repeating: nil, count: chunks.count)
    defer { results.removeAll() }
    DispatchQueue.concurrentPerform(iterations: chunks.count) { slot in
      let result = Result { try autoreleasepool { try self.load(chunks[slot], slot: slot) } }
      self.lock.lock()
      self.results[slot] = result
      self.lock.unlock()
    }
    // concurrentPerform joins even failed workers. All committed Metal work has
    // completed; dropping successful siblings on failure cannot free in-flight data.
    return try results.map { result in
      guard let result else { throw TANSArchive.invalid("Missing joined upload result") }
      return try result.get()
    }
  }

  private func load(_ chunk: TANSArchive.Chunk, slot: Int) throws -> Loaded {
    var metrics = TANSArchive.ReadMetrics()
    let stage = stages[slot]
    let readStart = ProcessInfo.processInfo.systemUptime
    try archive.read(
      chunk, into: UnsafeMutableRawBufferPointer(start: stage.contents(), count: stage.length),
      metrics: &metrics, policy: readPolicy)
    let readSeconds = ProcessInfo.processInfo.systemUptime - readStart
    guard let resident = device.makeBuffer(length: chunk.recordBytes, options: .storageModePrivate)
    else {
      throw TANSArchive.invalid("Cannot allocate encoded chunk; no resident published")
    }
    resident.label = "tANS encoded acquisition \(chunk.acquisition) chunk \(chunk.chunk)"
    let uploadStart = ProcessInfo.processInfo.systemUptime
    guard let command = queue.makeCommandBuffer(),
      let blit = command.makeBlitCommandEncoder()
    else { throw TANSArchive.invalid("Cannot encode bounded tANS upload") }
    blit.copy(
      from: stage, sourceOffset: 0, to: resident, destinationOffset: 0, size: chunk.recordBytes)
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("tANS upload failed: \(String(describing: command.error))")
    }
    return Loaded(
      resident: resident, metrics: metrics, readSeconds: readSeconds,
      uploadSeconds: ProcessInfo.processInfo.systemUptime - uploadStart)
  }
}
