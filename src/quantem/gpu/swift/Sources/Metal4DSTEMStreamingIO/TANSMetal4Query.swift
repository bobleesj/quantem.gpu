import Foundation
import Metal

/// Isolated submission experiment. The original entropy bytes and three
/// existing integer kernels remain authoritative; there is no decoded cache.
@available(macOS 26.0, iOS 26.0, *)
final class TANSMetal4Query {
  private let device: MTLDevice
  private let queue: MTL4CommandQueue
  private let sourceSet: MTLResidencySet
  private let sourceBuffers: [MTLBuffer]
  private let homogeneous: MTLComputePipelineState
  private let mixed: MTLComputePipelineState
  private let sparse: MTLComputePipelineState

  init(device: MTLDevice, library: MTLLibrary, sources: [MTLBuffer]) throws {
    self.device = device
    guard device.supportsFamily(.metal4), let queue = device.makeMTL4CommandQueue() else {
      throw TANSArchive.invalid("Explicit entropy submission requires Metal 4")
    }
    self.queue = queue
    let compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
    func pipeline(_ name: String, mixed: Bool) throws -> MTLComputePipelineState {
      let constants = MTLFunctionConstantValues()
      var no = false
      var yes = true
      var one: UInt32 = 1
      var threads: UInt32 = 128
      for index in [2, 3, 7, 8, 9, 10, 13, 14, 16, 17, 19, 20, 22] {
        constants.setConstantValue(&no, type: .bool, index: index)
      }
      constants.setConstantValue(&threads, type: .uint, index: 4)
      for index in [5, 6, 15, 18] { constants.setConstantValue(&one, type: .uint, index: index) }
      var zero: UInt32 = 0
      constants.setConstantValue(&zero, type: .uint, index: 21)
      for index in [11, 12] {
        if mixed {
          constants.setConstantValue(&yes, type: .bool, index: index)
        } else {
          constants.setConstantValue(&no, type: .bool, index: index)
        }
      }
      let function = MTL4LibraryFunctionDescriptor()
      function.library = library
      function.name = name
      let specialized = MTL4SpecializedFunctionDescriptor()
      specialized.functionDescriptor = function
      specialized.constantValues = constants
      let descriptor = MTL4ComputePipelineDescriptor()
      descriptor.computeFunctionDescriptor = specialized
      return try compiler.makeComputePipelineState(descriptor: descriptor, compilerTaskOptions: nil)
    }
    homogeneous = try pipeline("tans_detector_shared_model_batch", mixed: false)
    mixed = try pipeline("tans_detector_shared_model_batch", mixed: true)
    sparse = try pipeline("tans_detector_sparse_batch", mixed: false)
    var seen = Set<ObjectIdentifier>()
    sourceBuffers = sources.filter { seen.insert(ObjectIdentifier($0)).inserted }
    let descriptor = MTLResidencySetDescriptor()
    descriptor.initialCapacity = sourceBuffers.count
    sourceSet = try device.makeResidencySet(descriptor: descriptor)
    for buffer in sourceBuffers { sourceSet.addAllocation(buffer) }
    sourceSet.commit()
    queue.addResidencySet(sourceSet)
  }

  deinit { queue.removeResidencySet(sourceSet) }

  private final class Completion: @unchecked Sendable {
    let ready = DispatchSemaphore(value: 0)
    var start = 0.0
    var end = 0.0
    var error: Error?
  }

  func run(
    table: MTLBuffer, globals: [MTLBuffer], selection: MTLBuffer,
    coefficients: MTLBuffer, modelOffsets: MTLBuffer, grouped: [MTLBuffer],
    images: [MTLBuffer], records: Int, denseCount: Int, sparseCount: Int,
    retainedColumns: UInt32, sparseColumns: UInt32, homogeneousGroups: Int, mixedGroups: Int
  ) throws -> (start: Double, end: Double) {
    guard grouped.count == 8, globals.count == 4, records > 0 else {
      throw TANSArchive.invalid("Metal 4 entropy query requires the frozen grouped layout")
    }
    guard let allocator = device.makeCommandAllocator(), let command = device.makeCommandBuffer()
    else {
      throw TANSArchive.invalid("Cannot allocate explicit entropy command")
    }
    func upload(_ values: [UInt32]) throws -> MTLBuffer {
      try values.withUnsafeBytes {
        guard
          let b = device.makeBuffer(
            bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        else { throw TANSArchive.invalid("Cannot allocate exact query parameters") }
        return b
      }
    }
    let denseParameters = try upload([
      retainedColumns, sparseColumns, 0, UInt32(denseCount), UInt32((denseCount + 31) / 32),
    ])
    let sparseParameters = try upload([UInt32(sparseCount), sparseColumns])
    let bindings =
      [table, selection, coefficients, modelOffsets, denseParameters, sparseParameters] + grouped
      + images
    let descriptor = MTLResidencySetDescriptor()
    descriptor.initialCapacity = bindings.count
    let querySet = try device.makeResidencySet(descriptor: descriptor)
    for buffer in bindings { querySet.addAllocation(buffer) }
    querySet.commit()
    command.beginCommandBuffer(allocator: allocator)
    var ended = false
    defer { if !ended { command.endCommandBuffer() } }
    command.useResidencySet(querySet)
    var tables: [MTL4ArgumentTable] = []
    let argumentDescriptor = MTL4ArgumentTableDescriptor()
    argumentDescriptor.maxBufferBindCount = 15
    argumentDescriptor.initializeBindings = true
    for first in stride(from: 0, to: records, by: 64) {
      try Task.checkCancellation()
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw TANSArchive.invalid("Cannot encode explicit entropy query")
      }
      defer { encoder.endEncoding() }
      let count = min(64, records - first)
      func draw(
        _ pipeline: MTLComputePipelineState, grid: MTLSize, threads: Int,
        buffers: [(Int, MTLBuffer, Int)], threadGrid: Bool = false
      ) throws {
        let arguments = try device.makeArgumentTable(descriptor: argumentDescriptor)
        for (index, buffer, offset) in buffers {
          arguments.setAddress(buffer.gpuAddress + UInt64(offset), index: index)
        }
        tables.append(arguments)
        encoder.setComputePipelineState(pipeline)
        encoder.setArgumentTable(arguments)
        let width = MTLSize(width: threads, height: 1, depth: 1)
        if threadGrid {
          encoder.dispatchThreads(threadsPerGrid: grid, threadsPerThreadgroup: width)
        } else {
          encoder.dispatchThreadgroups(threadgroupsPerGrid: grid, threadsPerThreadgroup: width)
        }
      }
      let base: [(Int, MTLBuffer, Int)] = [
        (0, table, first * 40), (1, globals[0], 0), (2, globals[1], 0), (4, globals[3], 0),
        (7, denseParameters, 0), (8, modelOffsets, first * 4), (9, grouped[0], 0),
        (10, grouped[1], 0), (11, grouped[2], 0),
      ]
      if homogeneousGroups > 0 {
        try draw(
          homogeneous, grid: MTLSize(width: homogeneousGroups, height: 8, depth: count),
          threads: 128, buffers: base + [(12, grouped[3], first * 4), (13, grouped[5], first * 4)])
      }
      if mixedGroups > 0 {
        try draw(
          mixed, grid: MTLSize(width: mixedGroups, height: 8, depth: count),
          threads: 128, buffers: base + [(12, grouped[6], first * 4), (13, grouped[7], first * 4)])
      }
      if sparseCount > 0 {
        try draw(
          sparse, grid: MTLSize(width: sparseCount * 32, height: count, depth: 1),
          threads: 128,
          buffers: [
            (0, table, first * 40), (1, globals[2], 0),
            (2, selection, denseCount * 4), (3, coefficients, denseCount * 4),
            (4, sparseParameters, 0),
          ],
          threadGrid: true)
      }
    }
    command.endCommandBuffer()
    ended = true
    try Task.checkCancellation()
    let completion = Completion()
    let options = MTL4CommitOptions()
    options.addFeedbackHandler {
      completion.start = $0.gpuStartTime
      completion.end = $0.gpuEndTime
      completion.error = $0.error
      completion.ready.signal()
    }
    queue.commit([command], options: options)
    completion.ready.wait()
    withExtendedLifetime((allocator, command, querySet, bindings, globals, tables, sourceBuffers)) {
    }
    if let error = completion.error { throw error }
    try Task.checkCancellation()
    return (completion.start, completion.end)
  }
}
