import Foundation

/// Validates address-bearing stream metadata before any Metal query can use it.
/// This examines compressed layout, not scientific counts or a decoded volume.
enum TANSStreamValidation {
  static func offsets(_ bytes: UnsafeRawBufferPointer, columns: Int, sparse: Bool, limit: Int)
    throws
  {
    let headerWords = columns + (sparse ? 1 : 0)
    guard columns >= 0, bytes.count == (columns * 9 + (sparse ? 1 : 0)) * 4 else {
      throw TANSArchive.invalid("Entropy offset-table length disagrees with the detector map")
    }
    func word(_ index: Int) -> UInt32 {
      UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self))
    }
    var next = 0
    for group in 0..<columns {
      guard Int(word(group)) == next else {
        throw TANSArchive.invalid("Entropy stream groups are not contiguous")
      }
      for part in 0..<8 {
        let packed = word(headerWords + group * 8 + part)
        next +=
          Int(packed & 255) + Int((packed >> 8) & 255)
          + Int((packed >> 16) & 255) + Int(packed >> 24) + (sparse ? 0 : 4)
      }
      guard next <= limit else {
        throw TANSArchive.invalid("Entropy stream extent exceeds its payload")
      }
    }
    if sparse {
      guard Int(word(columns)) == next, next == limit else {
        throw TANSArchive.invalid("Sparse terminal offset disagrees with the event count")
      }
    }
  }

  static func record(_ data: Data, chunk: TANSArchive.Chunk, cacheMap: Data, models: Data) throws {
    let map: [Int32] = cacheMap.withUnsafeBytes { bytes in
      (0..<36864).map {
        Int32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self))
      }
    }
    let retained = map.enumerated().filter { $0.element < 0 }.map(\.offset)
    let sparseColumns = 36864 - retained.count
    try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
      func component(_ name: String) throws -> UnsafeRawBufferPointer {
        guard let descriptor = chunk.components.first(where: { $0.name == name }) else {
          throw TANSArchive.invalid("Missing stream component \(name)")
        }
        return UnsafeRawBufferPointer(
          rebasing: raw[descriptor.offset..<(descriptor.offset + descriptor.nbytes)])
      }
      let dense = try component("dense")
      let denseOffsets = try component("dense_offsets")
      try offsets(denseOffsets, columns: retained.count, sparse: false, limit: dense.count / 4)
      let modelOffset = (chunk.acquisition * 4 + (chunk.chunk % 16) / 4) * 36864
      for (rank, detector) in retained.enumerated() where models[modelOffset + detector] == 255 {
        for packet in 0..<32 {
          let stream = packet * retained.count + rank
          let index = retained.count * 4 + stream
          guard denseOffsets[index] == 255 else {
            throw TANSArchive.invalid("Literal uint16 stream must contain all512native counts")
          }
        }
      }
      let events = try component("sparse")
      guard events.count >= 16 else { throw TANSArchive.invalid("Truncated sparse header") }
      func eventWord(_ index: Int) -> Int {
        Int(UInt32(littleEndian: events.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)))
      }
      let count = eventWord(0)
      let positions = eventWord(1)
      let flags = eventWord(2)
      let ranks = eventWord(3)
      let prefixWords = 4 + positions + flags + ranks
      guard positions >= (count * 9 + 31) / 32, flags >= (count + 31) / 32,
        ranks >= (count + 255) / 256, prefixWords <= events.count / 4
      else {
        throw TANSArchive.invalid("Sparse position/flag/rank arrays exceed authenticated payload")
      }
      let sparseOffsets = try component("sparse_offsets")
      try offsets(sparseOffsets, columns: sparseColumns, sparse: true, limit: count)
      var prefix = 0
      for first in stride(from: 0, to: count, by: 256) {
        guard eventWord(4 + positions + flags + first / 256) == prefix else {
          throw TANSArchive.invalid("Sparse count-value rank is inconsistent with flags")
        }
        for at in stride(from: first, to: min(count, first + 256), by: 32) {
          let word = UInt32(eventWord(4 + positions + at / 32))
          let length = min(32, count - at)
          let mask = length == 32 ? UInt32.max : (UInt32(1) << length) - 1
          guard word & ~mask == 0 else {
            throw TANSArchive.invalid("Sparse flags extend beyond event count")
          }
          prefix += word.nonzeroBitCount
        }
      }
      guard prefix <= events.count - prefixWords * 4 else {
        throw TANSArchive.invalid("Sparse literal count values are truncated")
      }
    }
  }
}
