import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

final class TANSRecordHeapExperimentTests: XCTestCase {
  func testPlacementPlanHasExactNonoverlappingBuffersAndReadback() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    for recordsPerHeap in [16, 64] {
      for count in [1, 17, 67] {
        try autoreleasepool {
          let lengths = (0..<count).map { 1 + ($0 * 751) % 4096 }
          let plan = try TANSRecordHeapPlan(
            lengths: lengths, recordsPerHeap: recordsPerHeap, device: device)
          XCTAssertEqual(plan.heapSizes.count, (count + recordsPerHeap - 1) / recordsPerHeap)
          XCTAssertEqual(plan.allocationBytes, plan.heapSizes.reduce(0, +))
          let heaps = try plan.heapSizes.map { size in
            let descriptor = MTLHeapDescriptor()
            descriptor.type = .placement
            descriptor.storageMode = .private
            descriptor.hazardTrackingMode = .tracked
            descriptor.size = size
            return try XCTUnwrap(device.makeHeap(descriptor: descriptor))
          }
          var buffers: [MTLBuffer] = []
          var stages: [MTLBuffer] = []
          let command = try XCTUnwrap(queue.makeCommandBuffer())
          let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
          for index in lengths.indices {
            let layout = device.heapBufferSizeAndAlign(
              length: lengths[index], options: [.storageModePrivate, .hazardTrackingModeTracked])
            let offset = plan.offsets[index]
            XCTAssertEqual(offset % layout.align, 0)
            XCTAssertLessThanOrEqual(offset + layout.size, plan.heapSizes[plan.heapIndices[index]])
            if index > 0 && plan.heapIndices[index] == plan.heapIndices[index - 1] {
              let previous = device.heapBufferSizeAndAlign(
                length: lengths[index - 1],
                options: [.storageModePrivate, .hazardTrackingModeTracked])
              XCTAssertGreaterThanOrEqual(offset, plan.offsets[index - 1] + previous.size)
            }
            let buffer = try XCTUnwrap(
              heaps[plan.heapIndices[index]].makeBuffer(
                length: lengths[index], options: [.storageModePrivate, .hazardTrackingModeTracked],
                offset: offset))
            let stage = try XCTUnwrap(
              device.makeBuffer(length: lengths[index], options: .storageModeShared))
            stage.contents().initializeMemory(
              as: UInt8.self, repeating: UInt8(index + 1), count: lengths[index])
            blit.copy(
              from: stage, sourceOffset: 0, to: buffer, destinationOffset: 0, size: lengths[index])
            buffers.append(buffer)
            stages.append(stage)
          }
          // Overwrite the shared staging before reading back only after the upload completes.
          blit.endEncoding()
          command.commit()
          command.waitUntilCompleted()
          XCTAssertEqual(command.status, .completed)
          let read = try XCTUnwrap(queue.makeCommandBuffer())
          let back = try XCTUnwrap(read.makeBlitCommandEncoder())
          for index in lengths.indices {
            stages[index].contents().initializeMemory(
              as: UInt8.self, repeating: 0, count: lengths[index])
            back.copy(
              from: buffers[index], sourceOffset: 0, to: stages[index], destinationOffset: 0,
              size: lengths[index])
          }
          back.endEncoding()
          read.commit()
          read.waitUntilCompleted()
          XCTAssertEqual(read.status, .completed)
          for index in lengths.indices {
            XCTAssertEqual(
              Data(bytes: stages[index].contents(), count: lengths[index]),
              Data(repeating: UInt8(index + 1), count: lengths[index]))
          }
        }
      }
    }
  }

  func testHeapPlanRejectsInvalidLengthsAndGrouping() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    for lengths in [[], [0], [-1], [Int.max]] {
      XCTAssertThrowsError(
        try TANSRecordHeapPlan(lengths: lengths, recordsPerHeap: 16, device: device))
    }
    for width in [-1, 0, 1, 15, 17] {
      XCTAssertThrowsError(
        try TANSRecordHeapPlan(lengths: [4096], recordsPerHeap: width, device: device))
    }
  }

  func testHeapUploadFailureJoinsAndExactRetryWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_ARCHIVE_FIXTURE"] else {
      throw XCTSkip("Requires sealed entropy archive; two encoded records only")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let archive = try TANSArchive(directory: URL(fileURLWithPath: path), acquisitions: [0])
    let good = archive.chunks[0]
    let bad = TANSArchive.Chunk(
      chunk: good.chunk, acquisition: good.acquisition, firstScan: good.firstScan,
      scanCount: good.scanCount, shard: good.shard, fileOffset: good.fileOffset,
      recordBytes: good.recordBytes, sha256: String(repeating: "0", count: 64),
      components: good.components)
    let window = try TANSUploadWindow(
      archive: archive, device: device, queue: queue,
      stageBytes: good.recordBytes, concurrency: 2)
    let allocationBefore = device.currentAllocatedSize
    try autoreleasepool {
      let plan = try TANSRecordHeapPlan(
        lengths: [good.recordBytes, good.recordBytes], recordsPerHeap: 16, device: device)
      let descriptor = MTLHeapDescriptor()
      descriptor.type = .placement
      descriptor.storageMode = .private
      descriptor.hazardTrackingMode = .tracked
      descriptor.size = plan.allocationBytes
      let heap = try XCTUnwrap(device.makeHeap(descriptor: descriptor))
      let destinations = try plan.offsets.map { offset in
        try XCTUnwrap(
          heap.makeBuffer(
            length: good.recordBytes,
            options: [.storageModePrivate, .hazardTrackingModeTracked], offset: offset))
      }
      XCTAssertThrowsError(
        try window.load([good, good], destinations: [destinations[0], destinations[0]]))
      XCTAssertThrowsError(try window.load([good, bad], destinations: destinations))
      let loaded = try window.load([good, good], destinations: destinations)
      XCTAssertEqual(loaded.count, 2)
      let readback = try XCTUnwrap(
        device.makeBuffer(length: good.recordBytes, options: .storageModeShared))
      for item in loaded {
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
        blit.copy(
          from: item.resident, sourceOffset: 0, to: readback, destinationOffset: 0,
          size: good.recordBytes)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed)
        try TANSArchive.verify(
          Data(bytesNoCopy: readback.contents(), count: readback.length, deallocator: .none),
          sha256: good.sha256)
      }
    }
    XCTAssertLessThan(device.currentAllocatedSize, allocationBefore + 8 * 1024 * 1024)
  }
}
