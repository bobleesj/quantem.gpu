import Metal

/// Experiment-only, disjoint private backing for the unchanged encoded records.
/// No resource aliases another record, and hazard tracking remains enabled.
struct TANSRecordHeapPlan {
  let heapSizes: [Int]
  let heapIndices: [Int]
  let offsets: [Int]
  let allocationBytes: Int

  init(lengths: [Int], recordsPerHeap: Int, device: MTLDevice) throws {
    guard [16, 64].contains(recordsPerHeap), !lengths.isEmpty,
      lengths.allSatisfy({ $0 > 0 && $0 <= device.maxBufferLength })
    else {
      throw TANSArchive.invalid("Heap experiment requires positive records and groups of16 or64")
    }
    var sizes: [Int] = []
    var indices: [Int] = []
    var positions: [Int] = []
    var total = 0
    func checkedAdd(_ left: Int, _ right: Int) throws -> Int {
      let sum = left.addingReportingOverflow(right)
      guard !sum.overflow else { throw TANSArchive.invalid("Encoded heap size overflow") }
      return sum.partialValue
    }
    func aligned(_ value: Int, _ alignment: Int) throws -> Int {
      guard alignment > 0 else { throw TANSArchive.invalid("Invalid Metal heap alignment") }
      return try checkedAdd(value, (alignment - value % alignment) % alignment)
    }
    for first in stride(from: 0, to: lengths.count, by: recordsPerHeap) {
      var endOffset = 0
      var maximumAlignment = 1
      for index in first..<min(lengths.count, first + recordsPerHeap) {
        let layout = device.heapBufferSizeAndAlign(
          length: lengths[index], options: [.storageModePrivate, .hazardTrackingModeTracked])
        guard layout.size >= lengths[index] else {
          throw TANSArchive.invalid("Metal heap cannot hold the complete encoded record")
        }
        let offset = try aligned(endOffset, layout.align)
        positions.append(offset)
        indices.append(sizes.count)
        endOffset = try checkedAdd(offset, layout.size)
        maximumAlignment = max(maximumAlignment, layout.align)
      }
      let size = try aligned(endOffset, maximumAlignment)
      sizes.append(size)
      total = try checkedAdd(total, size)
    }
    heapSizes = sizes
    heapIndices = indices
    offsets = positions
    allocationBytes = total
  }
}
