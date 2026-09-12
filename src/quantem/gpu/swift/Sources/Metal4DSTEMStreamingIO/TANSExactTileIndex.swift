import Foundation
import Metal

/// Optional exact query accelerator. Only losslessly packed 2D tile sums live
/// here; original encoded counts remain owned by the source series.
final class TANSExactTileIndex {
  enum Layout: String {
    case centerFine
    case uniform16
    case uniform24
    /// 576 exact 8x8 tile sums. A moving annulus boundary crosses many more,
    /// smaller tiles, so a full recompute leaves far fewer residual columns.
    case uniform8
  }

  static func tiles(for layout: Layout) -> [Tile] {
    func grid(_ side: Int) -> [Tile] {
      stride(from: 0, to: 192, by: side).flatMap { row in
        stride(from: 0, to: 192, by: side).map {
          Tile(row: row, col: $0, side: side)
        }
      }
    }
    switch layout {
    case .centerFine:
      return grid(32)
        + stride(from: 64, to: 128, by: 8).flatMap { row in
          stride(from: 64, to: 128, by: 8).map { Tile(row: row, col: $0, side: 8) }
        }
    case .uniform16: return grid(16)
    case .uniform24: return grid(24)
    case .uniform8: return grid(8)
    }
  }

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
  private let maximumPipeline, packPipeline, addPipeline, addBasePipeline: MTLComputePipelineState
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
      let add = library.makeFunction(name: "tans_index_add"),
      let addBase = library.makeFunction(name: "tans_index_add_base")
    else {
      throw TANSArchive.invalid("Missing experimental exact tile-index kernels")
    }
    maximumPipeline = try device.makeComputePipelineState(function: maximum)
    packPipeline = try device.makeComputePipelineState(function: pack)
    addPipeline = try device.makeComputePipelineState(function: add)
    addBasePipeline = try device.makeComputePipelineState(function: addBase)
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

  /// Read one packed field back for an on-disk cache. Payloads live in private
  /// storage, so a blit copies the bytes into shared memory first. Blocked
  /// fields are not cacheable.
  func exportField(_ index: Int) throws -> Data {
    guard fields.indices.contains(index) else { throw TANSArchive.invalid("No such index field") }
    let field = fields[index]
    guard field.blocks == nil else {
      throw TANSArchive.invalid("Blocked exact index fields are not cacheable")
    }
    guard
      let staging = device.makeBuffer(length: field.payload.length, options: .storageModeShared),
      let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
    else { throw TANSArchive.invalid("Cannot read back an exact index field") }
    blit.copy(
      from: field.payload, sourceOffset: 0, to: staging, destinationOffset: 0,
      size: field.payload.length)
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("Exact index field read-back failed")
    }
    return Data(bytes: staging.contents(), count: staging.length)
  }

  /// Restore one packed field from cached bytes without any tile query. The
  /// caller has already verified the bytes against the cache manifest digest;
  /// the layout arithmetic is re-checked here so a mislabeled cache cannot
  /// produce a field that decodes out of bounds.
  func restore(
    tile: Tile, width: UInt32, words: UInt32, payload: Data, imageCount: Int,
    maximumBytes: UInt64
  ) throws {
    guard !blockedPacking, (1...32).contains(width), words == UInt32(262144) * width / 32,
      imageCount > 0, payload.count == Int(words) * 4 * imageCount
    else {
      throw TANSArchive.invalid("Cached exact index field does not match its declared layout")
    }
    guard UInt64(residentBytes + payload.count) <= maximumBytes,
      payload.count <= device.maxBufferLength
    else { throw TANSArchive.invalid("Cached exact tile index exceeds its explicit budget") }
    let staging: MTLBuffer? = payload.withUnsafeBytes { bytes in
      device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
    }
    guard let staging,
      let buffer = device.makeBuffer(length: payload.count, options: .storageModePrivate),
      let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
    else { throw TANSArchive.invalid("Cannot allocate a restored exact index field") }
    blit.copy(from: staging, sourceOffset: 0, to: buffer, destinationOffset: 0, size: payload.count)
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("Restoring an exact index field failed")
    }
    fields.append(Field(tile: tile, payload: buffer, width: width, words: words))
  }

  /// Plan only geometry/coefficients. Opposite signed pixels block a tile
  /// substitution, ensuring every residual stays in -1...1.
  func plan(
    coefficients: inout [Int32], valid: [UInt8], cost: [Double], tileCost: Double = 0.5
  ) -> [(Int, Int32)] {
    var chosen: [(Int, Int32)] = []
    for (index, pixels) in validPixels(valid).enumerated() {
      // Same sums in the same order as before, so every decision is identical.
      var oldCost = 0.0
      for pixel in pixels where coefficients[pixel] != 0 { oldCost += cost[pixel] }
      var best = oldCost
      var selected: Int32 = 0
      for sign in [Int32(-1), Int32(1)] {
        var fits = true
        for pixel in pixels where abs(coefficients[pixel] - sign) > 1 {
          fits = false
          break
        }
        guard fits else { continue }
        var sum = 0.0
        for pixel in pixels where coefficients[pixel] != sign { sum += cost[pixel] }
        let value = tileCost + sum
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

  /// Valid pixels of every field, built once per valid mask: `Tile.pixels` is
  /// computed, and rebuilding 576 lists twice per query cost host time on the
  /// interactive path.
  private var validPixelCache: (valid: [UInt8], fieldCount: Int, pixels: [[Int]])?
  private func validPixels(_ valid: [UInt8]) -> [[Int]] {
    if let cache = validPixelCache, cache.fieldCount == fields.count, cache.valid == valid {
      return cache.pixels
    }
    let pixels = fields.map { field in field.tile.pixels.filter { valid[$0] != 0 } }
    validPixelCache = (valid, fields.count, pixels)
    return pixels
  }

  /// Compare exact algebraic decompositions after tile substitution. Raw mask
  /// area is not the amount of remaining entropy work. Only geometry/metadata
  /// are inspected; neither plan computes or approximates scientific counts.
  func planChoosingBase(
    mask: [UInt8], previous: [UInt8], preferPrevious: Bool,
    valid: [UInt8], cost: [Double], tileCost: Double = 0.5
  ) -> (residual: [Int32], tiles: [(Int, Int32)], usePrevious: Bool) {
    var full = mask.map(Int32.init)
    var delta = mask.indices.map { Int32(mask[$0]) - Int32(previous[$0]) }
    let fullTiles = plan(coefficients: &full, valid: valid, cost: cost, tileCost: tileCost)
    let deltaTiles = plan(coefficients: &delta, valid: valid, cost: cost, tileCost: tileCost)
    func score(_ residual: [Int32], _ tiles: [(Int, Int32)]) -> Double {
      Double(tiles.count) * tileCost
        + residual.indices.reduce(0.0) { $0 + (residual[$1] != 0 ? cost[$1] : 0) }
    }
    let fullScore = score(full, fullTiles)
    let deltaScore = score(delta, deltaTiles)
    let usePrevious = deltaScore < fullScore || (deltaScore == fullScore && preferPrevious)
    return usePrevious ? (delta, deltaTiles, true) : (full, fullTiles, false)
  }

  /// `bindSelectedOnly` binds just the selected fields, so the per-query table
  /// and residency declarations follow the fields used, not the fields held.
  func encodeAdd(
    command: MTLCommandBuffer, images: [MTLBuffer], acquisitions: [Int],
    selected: [(Int, Int32)], bindSelectedOnly: Bool = false
  ) throws {
    guard !selected.isEmpty else { return }
    let bound = bindSelectedOnly ? selected.map { fields[$0.0] } : fields
    let selection =
      bindSelectedOnly ? selected.enumerated().map { ($0.offset, $0.element.1) } : selected
    let outputs = try imageTable(images)
    guard
      let table = device.makeBuffer(
        length: fieldArguments.encodedLength * bound.count, options: .storageModeShared),
      let selectionBuffer = device.makeBuffer(
        length: selection.count * 8, options: .storageModeShared),
      let indices = device.makeBuffer(length: acquisitions.count * 4, options: .storageModeShared),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw TANSArchive.invalid("Cannot allocate exact tile-query metadata")
    }
    for (i, field) in bound.enumerated() {
      fieldArguments.setArgumentBuffer(table, offset: i * fieldArguments.encodedLength)
      fieldArguments.setBuffer(field.payload, offset: 0, index: 0)
      fieldArguments.constantData(at: 1).storeBytes(of: field.width, as: UInt32.self)
      fieldArguments.constantData(at: 2).storeBytes(of: field.words, as: UInt32.self)
      fieldArguments.setBuffer(field.blocks, offset: 0, index: 3)
      fieldArguments.constantData(at: 4).storeBytes(
        of: UInt32(field.blocks == nil ? 0 : 1), as: UInt32.self)
    }
    for (i, item) in selection.enumerated() {
      selectionBuffer.contents().storeBytes(of: Int32(item.0), toByteOffset: i * 8, as: Int32.self)
      selectionBuffer.contents().storeBytes(of: item.1, toByteOffset: i * 8 + 4, as: Int32.self)
    }
    for (i, index) in acquisitions.enumerated() {
      indices.contents().storeBytes(of: UInt32(index), toByteOffset: i * 4, as: UInt32.self)
    }
    encoder.setComputePipelineState(addPipeline)
    encoder.useResources(images, usage: [.read, .write])
    encoder.useResources(bound.map(\.payload), usage: .read)
    encoder.useResources(bound.compactMap(\.blocks), usage: .read)
    for (i, buffer) in [outputs, table, selectionBuffer].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: i)
    }
    var count = UInt32(selection.count)
    encoder.setBytes(&count, length: 4, index: 3)
    encoder.setBuffer(indices, offset: 0, index: 4)
    encoder.dispatchThreads(
      MTLSize(width: 262144, height: images.count, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
  }

  /// Exact initialization plus field sum into an existing compute pass: each
  /// output becomes `base` (the exact seed image, or zero when `bases` is nil)
  /// plus the selected signed fields. Replaces a blit copy/fill followed by
  /// `encodeAdd`; the caller orders later dispatches after it.
  func encodeAddFromBase(
    encoder: MTLComputeCommandEncoder, images: [MTLBuffer], bases: [MTLBuffer]?,
    acquisitions: [Int], selected: [(Int, Int32)], bindSelectedOnly: Bool = false
  ) throws {
    guard !selected.isEmpty, bases == nil || bases!.count == images.count else {
      throw TANSArchive.invalid("Exact base field add needs fields and one base per image")
    }
    let bound = bindSelectedOnly ? selected.map { fields[$0.0] } : fields
    let selection =
      bindSelectedOnly ? selected.enumerated().map { ($0.offset, $0.element.1) } : selected
    let outputs = try imageTable(images)
    let baseTable = try bases.map { try imageTable($0) } ?? outputs
    guard
      let table = device.makeBuffer(
        length: fieldArguments.encodedLength * bound.count, options: .storageModeShared),
      let selectionBuffer = device.makeBuffer(
        length: selection.count * 8, options: .storageModeShared),
      let indices = device.makeBuffer(length: acquisitions.count * 4, options: .storageModeShared)
    else {
      throw TANSArchive.invalid("Cannot allocate exact tile-query metadata")
    }
    for (i, field) in bound.enumerated() {
      fieldArguments.setArgumentBuffer(table, offset: i * fieldArguments.encodedLength)
      fieldArguments.setBuffer(field.payload, offset: 0, index: 0)
      fieldArguments.constantData(at: 1).storeBytes(of: field.width, as: UInt32.self)
      fieldArguments.constantData(at: 2).storeBytes(of: field.words, as: UInt32.self)
      fieldArguments.setBuffer(field.blocks, offset: 0, index: 3)
      fieldArguments.constantData(at: 4).storeBytes(
        of: UInt32(field.blocks == nil ? 0 : 1), as: UInt32.self)
    }
    for (i, item) in selection.enumerated() {
      selectionBuffer.contents().storeBytes(of: Int32(item.0), toByteOffset: i * 8, as: Int32.self)
      selectionBuffer.contents().storeBytes(of: item.1, toByteOffset: i * 8 + 4, as: Int32.self)
    }
    for (i, index) in acquisitions.enumerated() {
      indices.contents().storeBytes(of: UInt32(index), toByteOffset: i * 4, as: UInt32.self)
    }
    encoder.setComputePipelineState(addBasePipeline)
    encoder.useResources(images, usage: [.read, .write])
    if let bases { encoder.useResources(bases, usage: .read) }
    encoder.useResources(bound.map(\.payload), usage: .read)
    encoder.useResources(bound.compactMap(\.blocks), usage: .read)
    for (i, buffer) in [outputs, table, selectionBuffer].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: i)
    }
    var count = UInt32(selection.count)
    encoder.setBytes(&count, length: 4, index: 3)
    encoder.setBuffer(indices, offset: 0, index: 4)
    encoder.setBuffer(baseTable, offset: 0, index: 5)
    var hasBase: UInt32 = bases == nil ? 0 : 1
    encoder.setBytes(&hasBase, length: 4, index: 6)
    encoder.dispatchThreads(
      MTLSize(width: 262144, height: images.count, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  }
}
