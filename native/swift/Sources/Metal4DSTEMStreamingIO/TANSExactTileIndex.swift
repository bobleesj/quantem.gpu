import Foundation
import Metal

/// Optional exact query accelerator. Only losslessly packed 2D tile sums live
/// here; original encoded counts remain owned by the source series.
final class TANSExactTileIndex {
  struct Tile {
    let row: Int
    let col: Int
    let side: Int
    var pixels: [Int] {
      (row..<(row + side)).flatMap { r in (col..<(col + side)).map { r * 192 + $0 } }
    }
  }
  struct Field {
    let tile: Tile
    let payload: MTLBuffer
    let width: UInt32
    let words: UInt32
    var blocks: MTLBuffer? = nil
  }
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let maximumPipeline, packPipeline, addPipeline: MTLComputePipelineState
  private let blockPipelines: [MTLComputePipelineState]
  private let imageArguments, fieldArguments: MTLArgumentEncoder
  let blockedPacking: Bool
  private(set) var fields: [Field] = []
  var residentBytes: Int { fields.reduce(0) { $0 + $1.payload.length + ($1.blocks?.length ?? 0) } }

  init(
    device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary,
    blockedPacking: Bool = false
  ) throws {
    self.device = device
    self.queue = queue
    self.blockedPacking = blockedPacking
    blockPipelines = try [
      "tans_index_block_stats", "tans_index_block_offsets",
      "tans_index_acquisition_offsets", "tans_index_block_pack",
    ].map { name in
      guard let function = library.makeFunction(name: name) else {
        throw TANSArchive.invalid("Missing exact block packing kernel: \(name)")
      }
      return try device.makeComputePipelineState(function: function)
    }
    guard let maximum = library.makeFunction(name: "tans_index_max"),
      let pack = library.makeFunction(name: "tans_index_pack"),
      let add = library.makeFunction(name: "tans_index_add")
    else {
      throw TANSArchive.invalid("Missing experimental exact tile-index kernels")
    }
    maximumPipeline = try device.makeComputePipelineState(function: maximum)
    packPipeline = try device.makeComputePipelineState(function: pack)
    addPipeline = try device.makeComputePipelineState(function: add)
    imageArguments = maximum.makeArgumentEncoder(bufferIndex: 0)
    fieldArguments = add.makeArgumentEncoder(bufferIndex: 1)
  }

  private func imageTable(_ images: [MTLBuffer]) throws -> MTLBuffer {
    guard
      let table = device.makeBuffer(
        length: images.count * imageArguments.encodedLength,
        options: .storageModeShared)
    else { throw TANSArchive.invalid("Cannot allocate tile image table") }
    for (i, image) in images.enumerated() {
      imageArguments.setArgumentBuffer(table, offset: i * imageArguments.encodedLength)
      imageArguments.setBuffer(image, offset: 0, index: 0)
    }
    return table
  }

  /// Integer maximum and packing are GPU operations. Host reads one maximum
  /// solely to admit the exact width/allocation. Every packed value is audited.
  func append(tile: Tile, images: [MTLBuffer], maximumBytes: UInt64) throws {
    if blockedPacking {
      try appendBlocked(tile: tile, images: images, maximumBytes: maximumBytes)
      return
    }
    try autoreleasepool {
      let table = try imageTable(images)
      guard let maximum = device.makeBuffer(length: 4, options: .storageModeShared),
        let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
      else {
        throw TANSArchive.invalid("Cannot encode exact tile maximum")
      }
      maximum.contents().storeBytes(of: UInt32(0), as: UInt32.self)
      encoder.setComputePipelineState(maximumPipeline)
      encoder.useResources(images, usage: .read)
      encoder.setBuffer(table, offset: 0, index: 0)
      encoder.setBuffer(maximum, offset: 0, index: 1)
      encoder.dispatchThreadgroups(
        MTLSize(width: images.count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed else { throw TANSArchive.invalid("Tile maximum failed") }
      let width = UInt32(max(1, 32 - maximum.contents().load(as: UInt32.self).leadingZeroBitCount))
      let words = UInt32(262144) * width / 32
      let bytes = Int(words) * 4 * images.count
      guard UInt64(residentBytes + bytes) <= maximumBytes, bytes <= device.maxBufferLength,
        let payload = device.makeBuffer(length: bytes, options: .storageModePrivate),
        let packCommand = queue.makeCommandBuffer(),
        let packEncoder = packCommand.makeComputeCommandEncoder()
      else {
        throw TANSArchive.invalid(
          "Exact tile index exceeds its explicit budget; no precision fallback")
      }
      packEncoder.setComputePipelineState(packPipeline)
      packEncoder.useResources(images, usage: .read)
      packEncoder.setBuffer(table, offset: 0, index: 0)
      packEncoder.setBuffer(payload, offset: 0, index: 1)
      var layout = [width, words]
      packEncoder.setBytes(&layout, length: 8, index: 2)
      packEncoder.dispatchThreads(
        MTLSize(width: Int(words), height: images.count, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      packEncoder.endEncoding()
      packCommand.commit()
      packCommand.waitUntilCompleted()
      guard packCommand.status == .completed else {
        throw TANSArchive.invalid("Exact tile packing failed")
      }
      fields.append(Field(tile: tile, payload: payload, width: width, words: words))
      // Independent roundtrip against complete decoded uint32 source products.
      // This temporary 2D audit is included in preparation, not query timing.
      let copies = try images.map { image -> MTLBuffer in
        guard let result = device.makeBuffer(length: image.length, options: .storageModeShared)
        else {
          throw TANSArchive.invalid("Cannot allocate tile roundtrip audit")
        }
        return result
      }
      guard let audit = queue.makeCommandBuffer(), let zero = audit.makeBlitCommandEncoder() else {
        throw TANSArchive.invalid("Cannot encode tile roundtrip audit")
      }
      for copy in copies { zero.fill(buffer: copy, range: 0..<copy.length, value: 0) }
      zero.endEncoding()
      try encodeAdd(
        command: audit, images: copies, acquisitions: Array(images.indices),
        selected: [(fields.count - 1, 1)])
      audit.commit()
      audit.waitUntilCompleted()
      guard audit.status == .completed else { throw TANSArchive.invalid("Tile roundtrip failed") }
      for (image, copy) in zip(images, copies) {
        guard memcmp(image.contents(), copy.contents(), image.length) == 0 else {
          throw TANSArchive.invalid("Exact packed tile roundtrip changed uint32 counts")
        }
      }
    }
  }

  private func appendBlocked(tile: Tile, images: [MTLBuffer], maximumBytes: UInt64) throws {
    try autoreleasepool {
      let metadataBytes = images.count * 1024 * 16
      guard UInt64(residentBytes + metadataBytes + 4) <= maximumBytes,
        let metadata = device.makeBuffer(length: metadataBytes, options: .storageModePrivate),
        let totals = device.makeBuffer(length: (images.count + 1) * 4, options: .storageModeShared),
        let command = queue.makeCommandBuffer()
      else {
        throw TANSArchive.invalid("Cannot admit exact block index metadata")
      }
      let table = try imageTable(images)
      for stage in 0..<3 {
        guard let encoder = command.makeComputeCommandEncoder() else {
          throw TANSArchive.invalid("Cannot encode exact block layout")
        }
        encoder.setComputePipelineState(blockPipelines[stage])
        if stage == 0 {
          encoder.useResources(images, usage: .read)
          encoder.setBuffer(table, offset: 0, index: 0)
          encoder.setBuffer(metadata, offset: 0, index: 1)
          encoder.dispatchThreadgroups(
            MTLSize(width: 1024, height: images.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else if stage == 1 {
          encoder.setBuffer(metadata, offset: 0, index: 0)
          encoder.setBuffer(totals, offset: 0, index: 1)
          encoder.dispatchThreads(
            MTLSize(width: images.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        } else {
          encoder.setBuffer(totals, offset: 0, index: 0)
          var count = UInt32(images.count)
          encoder.setBytes(&count, length: 4, index: 1)
          encoder.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        }
        encoder.endEncoding()
      }
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed else {
        throw TANSArchive.invalid("Exact block layout failed")
      }
      let words = totals.contents().load(fromByteOffset: images.count * 4, as: UInt32.self)
      let bytes = max(4, Int(words) * 4)
      guard UInt64(residentBytes + metadataBytes + bytes) <= maximumBytes,
        bytes <= device.maxBufferLength,
        let payload = device.makeBuffer(length: bytes, options: .storageModePrivate),
        let pack = queue.makeCommandBuffer(), let encoder = pack.makeComputeCommandEncoder()
      else {
        throw TANSArchive.invalid("Exact block index exceeds budget; no precision fallback")
      }
      encoder.setComputePipelineState(blockPipelines[3])
      encoder.useResources(images, usage: .read)
      for (i, buffer) in [table, metadata, totals, payload].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: i)
      }
      encoder.dispatchThreadgroups(
        MTLSize(width: 1024, height: images.count, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      encoder.endEncoding()
      pack.commit()
      pack.waitUntilCompleted()
      guard pack.status == .completed else {
        throw TANSArchive.invalid("Exact block packing failed")
      }
      fields.append(Field(tile: tile, payload: payload, width: 0, words: words, blocks: metadata))
      do {
        let copies = try images.map { image -> MTLBuffer in
          guard let result = device.makeBuffer(length: image.length, options: .storageModeShared)
          else {
            throw TANSArchive.invalid("Cannot allocate block roundtrip audit")
          }
          return result
        }
        guard let audit = queue.makeCommandBuffer(), let zero = audit.makeBlitCommandEncoder()
        else {
          throw TANSArchive.invalid("Cannot encode block roundtrip audit")
        }
        for copy in copies { zero.fill(buffer: copy, range: 0..<copy.length, value: 0) }
        zero.endEncoding()
        try encodeAdd(
          command: audit, images: copies, acquisitions: Array(images.indices),
          selected: [(fields.count - 1, 1)])
        audit.commit()
        audit.waitUntilCompleted()
        guard audit.status == .completed else {
          throw TANSArchive.invalid("Block roundtrip command failed")
        }
        for (image, copy) in zip(images, copies) {
          guard memcmp(image.contents(), copy.contents(), image.length) == 0 else {
            throw TANSArchive.invalid("Block packing changed exact uint32 counts")
          }
        }
      } catch {
        fields.removeLast()
        throw error
      }
    }
  }

  /// Plan only geometry/coefficients. Opposite signed pixels block a tile
  /// substitution, ensuring every residual stays in -1...1.
  func plan(coefficients: inout [Int32], valid: [UInt8], cost: [Double]) -> [(Int, Int32)] {
    var chosen: [(Int, Int32)] = []
    for (index, field) in fields.enumerated() {
      let pixels = field.tile.pixels.filter { valid[$0] != 0 }
      let oldCost = pixels.reduce(0.0) { $0 + (coefficients[$1] != 0 ? cost[$1] : 0) }
      var best = oldCost
      var selected: Int32 = 0
      for sign in [Int32(-1), Int32(1)] {
        guard pixels.allSatisfy({ abs(coefficients[$0] - sign) <= 1 }) else { continue }
        let value = 0.5 + pixels.reduce(0.0) { $0 + (coefficients[$1] != sign ? cost[$1] : 0) }
        if value < best {
          best = value
          selected = sign
        }
      }
      if selected != 0 {
        for pixel in pixels { coefficients[pixel] -= selected }
        chosen.append((index, selected))
      }
    }
    return chosen
  }

  func encodeAdd(
    command: MTLCommandBuffer, images: [MTLBuffer], acquisitions: [Int],
    selected: [(Int, Int32)]
  ) throws {
    guard !selected.isEmpty else { return }
    let outputs = try imageTable(images)
    guard
      let table = device.makeBuffer(
        length: fieldArguments.encodedLength * fields.count, options: .storageModeShared),
      let selection = device.makeBuffer(length: selected.count * 8, options: .storageModeShared),
      let indices = device.makeBuffer(length: acquisitions.count * 4, options: .storageModeShared),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw TANSArchive.invalid("Cannot allocate exact tile-query metadata")
    }
    for (i, field) in fields.enumerated() {
      fieldArguments.setArgumentBuffer(table, offset: i * fieldArguments.encodedLength)
      fieldArguments.setBuffer(field.payload, offset: 0, index: 0)
      fieldArguments.constantData(at: 1).storeBytes(of: field.width, as: UInt32.self)
      fieldArguments.constantData(at: 2).storeBytes(of: field.words, as: UInt32.self)
      fieldArguments.setBuffer(field.blocks, offset: 0, index: 3)
      fieldArguments.constantData(at: 4).storeBytes(
        of: UInt32(field.blocks == nil ? 0 : 1), as: UInt32.self)
    }
    for (i, item) in selected.enumerated() {
      selection.contents().storeBytes(of: Int32(item.0), toByteOffset: i * 8, as: Int32.self)
      selection.contents().storeBytes(of: item.1, toByteOffset: i * 8 + 4, as: Int32.self)
    }
    for (i, index) in acquisitions.enumerated() {
      indices.contents().storeBytes(of: UInt32(index), toByteOffset: i * 4, as: UInt32.self)
    }
    encoder.setComputePipelineState(addPipeline)
    encoder.useResources(images, usage: [.read, .write])
    encoder.useResources(fields.map(\.payload), usage: .read)
    encoder.useResources(fields.compactMap(\.blocks), usage: .read)
    for (i, buffer) in [outputs, table, selection].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: i)
    }
    var count = UInt32(selected.count)
    encoder.setBytes(&count, length: 4, index: 3)
    encoder.setBuffer(indices, offset: 0, index: 4)
    encoder.dispatchThreads(
      MTLSize(width: 262144, height: images.count, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
  }
}
