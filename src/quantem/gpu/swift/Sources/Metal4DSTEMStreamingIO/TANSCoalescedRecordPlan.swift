import Foundation

/// Concatenates unchanged encoded records, with no padding or decoded storage.
struct TANSCoalescedRecordPlan {
  let allocationIndices: [Int]
  let offsets: [Int]
  let allocationSizes: [Int]
  let totalBytes: Int

  init(lengths: [Int], maxBufferLength: Int, maximumBytes: UInt64) throws {
    guard !lengths.isEmpty, lengths.count % 16 == 0, maxBufferLength > 0 else {
      throw TANSArchive.invalid("Coalesced source requires sixteen records per acquisition")
    }
    var indices: [Int] = []
    var positions: [Int] = []
    var sizes: [Int] = []
    var total = 0
    for first in stride(from: 0, to: lengths.count, by: 16) {
      var size = 0
      for index in first..<(first + 16) {
        let length = lengths[index]
        guard length > 0, length % 4 == 0 else {
          throw TANSArchive.invalid("Encoded slices require positive four-byte-aligned lengths")
        }
        indices.append(sizes.count)
        positions.append(size)
        let next = size.addingReportingOverflow(length)
        guard !next.overflow, next.partialValue <= maxBufferLength else {
          throw TANSArchive.invalid("Coalesced acquisition exceeds Metal buffer limit; no fallback")
        }
        size = next.partialValue
      }
      let next = total.addingReportingOverflow(size)
      guard !next.overflow, UInt64(next.partialValue) <= maximumBytes else {
        throw TANSArchive.invalid("Coalesced encoded source exceeds the caller budget")
      }
      total = next.partialValue
      sizes.append(size)
    }
    allocationIndices = indices
    offsets = positions
    allocationSizes = sizes
    totalBytes = total
  }
}
