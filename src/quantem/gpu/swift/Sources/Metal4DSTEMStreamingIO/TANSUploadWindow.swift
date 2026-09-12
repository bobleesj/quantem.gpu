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
  private let untrackedEncodedSource: Bool
  private let destinationStorageMode: MTLStorageMode
  private let lock = NSLock()
  private var results: [Result<Loaded, Error>?] = []
  // Caller serialized; immutable during concurrentPerform, one disjoint buffer per slot.
  private var destinations: [MTLBuffer]?
  private var destinationOffsets: [Int]?

  init(
    archive: TANSArchive, device: MTLDevice, queue: MTLCommandQueue,
    stageBytes: Int, concurrency: Int,
    readPolicy: TANSArchive.SourceReadPolicy = .systemCache,
    untrackedEncodedSource: Bool = false,
    destinationStorageMode: MTLStorageMode = .private
  ) throws {
    self.archive = archive
    self.device = device
    self.queue = queue
    self.readPolicy = readPolicy
    self.untrackedEncodedSource = untrackedEncodedSource
    guard
      destinationStorageMode == .private
        || (destinationStorageMode == .shared && device.hasUnifiedMemory && !untrackedEncodedSource)
    else { throw TANSArchive.invalid("Unsupported exact encoded destination storage mode") }
    self.destinationStorageMode = destinationStorageMode
    stages = try (0..<concurrency).map { slot in
      guard let buffer = device.makeBuffer(length: stageBytes, options: .storageModeShared) else {
        throw TANSArchive.invalid("Cannot allocate bounded encoded transfer staging")
      }
      buffer.label = "tANS compressed transfer slot \(slot)"
      return buffer
    }
  }

  func load(
    _ chunks: [TANSArchive.Chunk], destinations: [MTLBuffer]? = nil,
    destinationOffsets: [Int]? = nil
  ) throws -> [Loaded] {
    precondition(chunks.count <= stages.count)
    // Only fresh invocation-owned disjoint destinations may be supplied.
    if let destinations {
      guard !untrackedEncodedSource else {
        throw TANSArchive.invalid("Caller-owned upload slices must remain tracked")
      }
      try Self.validateDestinations(
        destinations, offsets: destinationOffsets ?? Array(repeating: 0, count: chunks.count),
        lengths: chunks.map(\.recordBytes), expectedStorageMode: destinationStorageMode)
    } else if destinationOffsets != nil {
      throw TANSArchive.invalid("Upload offsets require caller-owned encoded destinations")
    } else if destinationStorageMode != .private {
      throw TANSArchive.invalid("Shared encoded storage requires explicit coalesced destinations")
    }
    results = Array(repeating: nil, count: chunks.count)
    self.destinations = destinations
    self.destinationOffsets = destinationOffsets
    defer {
      results.removeAll()
      self.destinations = nil
      self.destinationOffsets = nil
    }
    DispatchQueue.concurrentPerform(iterations: chunks.count) { slot in
      let result = Result {
        try autoreleasepool {
          try self.load(
            chunks[slot], slot: slot, destination: self.destinations?[slot],
            destinationOffset: self.destinationOffsets?[slot] ?? 0)
        }
      }
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

  /// Joined workers may share a buffer only when their write slices are disjoint.
  static func validateDestinations(
    _ buffers: [MTLBuffer], offsets: [Int], lengths: [Int],
    expectedStorageMode: MTLStorageMode = .private
  ) throws {
    guard buffers.count == offsets.count, buffers.count == lengths.count,
      expectedStorageMode == .private || expectedStorageMode == .shared
    else {
      throw TANSArchive.invalid("Encoded upload slice counts do not match")
    }
    for index in buffers.indices {
      let buffer = buffers[index]
      let offset = offsets[index]
      let length = lengths[index]
      guard buffer.storageMode == expectedStorageMode, buffer.hazardTrackingMode == .tracked,
        offset >= 0, offset % 4 == 0, length > 0,
        offset <= buffer.length, length <= buffer.length - offset
      else { throw TANSArchive.invalid("Encoded upload slice exceeds its tracked destination") }
      for prior in 0..<index where buffers[prior] === buffer {
        guard offset >= offsets[prior] + lengths[prior] || offsets[prior] >= offset + length else {
          throw TANSArchive.invalid("Concurrent encoded upload slices overlap")
        }
      }
    }
  }

  private func load(
    _ chunk: TANSArchive.Chunk, slot: Int, destination: MTLBuffer?, destinationOffset: Int
  ) throws -> Loaded {
    var metrics = TANSArchive.ReadMetrics()
    let stage = stages[slot]
    let readStart = ProcessInfo.processInfo.systemUptime
    try archive.read(
      chunk, into: UnsafeMutableRawBufferPointer(start: stage.contents(), count: stage.length),
      metrics: &metrics, policy: readPolicy)
    let readSeconds = ProcessInfo.processInfo.systemUptime - readStart
    guard
      let resident = destination
        ?? device.makeBuffer(
          length: chunk.recordBytes,
          options: untrackedEncodedSource
            ? [.storageModePrivate, .hazardTrackingModeUntracked] : .storageModePrivate)
    else {
      throw TANSArchive.invalid("Cannot allocate encoded chunk; no resident published")
    }
    resident.label = "tANS encoded acquisition \(chunk.acquisition) chunk \(chunk.chunk)"
    let uploadStart = ProcessInfo.processInfo.systemUptime
    guard let command = queue.makeCommandBuffer(),
      let blit = command.makeBlitCommandEncoder()
    else { throw TANSArchive.invalid("Cannot encode bounded tANS upload") }
    blit.copy(
      from: stage, sourceOffset: 0, to: resident, destinationOffset: destinationOffset,
      size: chunk.recordBytes)
    blit.endEncoding()
    command.commit()
    // For the opt-in untracked immutable source, this completion is the
    // explicit write-to-read synchronization boundary. No source is exposed
    // until its upload completes; all later access is read-only.
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("tANS upload failed: \(String(describing: command.error))")
    }
    return Loaded(
      resident: resident, metrics: metrics, readSeconds: readSeconds,
      uploadSeconds: ProcessInfo.processInfo.systemUptime - uploadStart)
  }
}
