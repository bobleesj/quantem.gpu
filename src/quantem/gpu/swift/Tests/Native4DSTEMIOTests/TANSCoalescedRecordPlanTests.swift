import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

final class TANSCoalescedRecordPlanTests: XCTestCase {
  func testEveryByteHasOneOrderedSliceWithNoPadding() throws {
    let lengths = (1...32).map { $0 * 4 }
    let plan = try TANSCoalescedRecordPlan(
      lengths: lengths, maxBufferLength: 4096, maximumBytes: 4096)
    XCTAssertEqual(plan.allocationSizes, [544, 1568])
    XCTAssertEqual(plan.totalBytes, lengths.reduce(0, +))
    for allocation in 0..<2 {
      var position = 0
      for index in (allocation * 16)..<((allocation + 1) * 16) {
        XCTAssertEqual(plan.allocationIndices[index], allocation)
        XCTAssertEqual(plan.offsets[index], position)
        position += lengths[index]
      }
      XCTAssertEqual(position, plan.allocationSizes[allocation])
    }
  }

  func testMalformedLimitsAndAdmissionFailClosed() {
    for lengths in [
      [], [4], [Int](repeating: 3, count: 16),
      [Int](repeating: 0, count: 16), [Int](repeating: Int.max - 3, count: 16),
    ] {
      XCTAssertThrowsError(
        try TANSCoalescedRecordPlan(
          lengths: lengths, maxBufferLength: Int.max, maximumBytes: UInt64.max))
    }
    for (limit, budget) in [(63, UInt64(64)), (64, UInt64(63))] {
      XCTAssertThrowsError(
        try TANSCoalescedRecordPlan(
          lengths: [Int](repeating: 4, count: 16), maxBufferLength: limit,
          maximumBytes: budget))
    }
  }

  func testSharedAllocationRequiresBoundedDisjointUploadSlices() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let buffer = try XCTUnwrap(
      device.makeBuffer(
        length: 16, options: [.storageModePrivate, .hazardTrackingModeTracked]))
    try TANSUploadWindow.validateDestinations([buffer, buffer], offsets: [0, 8], lengths: [8, 8])
    for offsets in [[0, 4], [-4, 8], [0, 12], [0], [0, Int.max - 3]] {
      XCTAssertThrowsError(
        try TANSUploadWindow.validateDestinations(
          [buffer, buffer], offsets: offsets, lengths: [8, 8]))
    }
    let other = try XCTUnwrap(device.makeBuffer(length: 16, options: .storageModePrivate))
    try TANSUploadWindow.validateDestinations(
      [buffer, other], offsets: [0, 0], lengths: [16, 16])
    let shared = try XCTUnwrap(device.makeBuffer(length: 16, options: .storageModeShared))
    XCTAssertThrowsError(
      try TANSUploadWindow.validateDestinations([shared], offsets: [0], lengths: [16]))
    try TANSUploadWindow.validateDestinations(
      [shared, shared], offsets: [0, 8], lengths: [8, 8], expectedStorageMode: .shared)
    XCTAssertThrowsError(
      try TANSUploadWindow.validateDestinations(
        [shared, shared], offsets: [0, 4], lengths: [8, 8], expectedStorageMode: .shared))
    XCTAssertThrowsError(
      try TANSUploadWindow.validateDestinations(
        [buffer], offsets: [0], lengths: [16], expectedStorageMode: .shared))
  }
}
