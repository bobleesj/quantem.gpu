import Metal
import Metal4DSTEMKernels
import XCTest

@testable import Metal4DSTEMStreamingIO

final class CompactH5MetadataKernelsTests: XCTestCase {
  func testGPUOffsetsMatchScalarOracleAcrossSIMDAndHierarchyBoundaries() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let kernels = try CompactH5MetadataKernels(
      device: device, library: Metal4DSTEMKernels.makeCompactH5Library(device: device))
    for count in [1, 31, 32, 33, 255, 256, 257, 65_535, 65_537, 1_048_579] {
      for kind in [UInt32(0), 1] {
        let input = (0..<count).map { UInt8(($0 * 13 + 7) % (kind == 0 ? 17 : 256)) }
        let buffer = try CompactH5MetadataKernels.buffer(device, bytes: count, shared: true)
        input.withUnsafeBytes { buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: count) }
        let status = try CompactH5MetadataKernels.buffer(device, bytes: 4, shared: true)
        status.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let offsets = try kernels.encodeOffsets(
          input: buffer, count: count, kind: kind, status: status, device: device, command: command)
        let output = try readback(offsets, device: device, command: command)
        try complete(command)
        XCTAssertEqual(status.contents().load(as: UInt32.self), 0)
        let values = output.contents().bindMemory(to: UInt32.self, capacity: count)
        var expected: UInt32 = 0
        for index in 0..<count {
          XCTAssertEqual(values[index], expected, "kind \(kind), count \(count), index \(index)")
          expected += kind == 0 ? UInt32(input[index]) * 4 : UInt32(input[index]) + 1
        }
      }
    }
  }

  func testGPUScanRejectsOverflowIncludingAcrossBlocks() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let kernels = try CompactH5MetadataKernels(
      device: device, library: Metal4DSTEMKernels.makeCompactH5Library(device: device))
    for count in [2, 33, 257, 65_537] {
      var values = [UInt32](repeating: 0, count: count)
      values[0] = .max
      values[count - 1] = 1
      let input = try XCTUnwrap(device.makeBuffer(bytes: values, length: count * 4))
      let status = try CompactH5MetadataKernels.buffer(device, bytes: 4, shared: true)
      status.contents().storeBytes(of: UInt32(0), as: UInt32.self)
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      _ = try kernels.encodeOffsets(
        input: input, count: count, kind: 2, status: status, device: device, command: command)
      try complete(command)
      XCTAssertEqual(status.contents().load(as: UInt32.self) & 1, 1)
    }
  }

  func testGPUStatusReductionFindsTailErrors() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let kernels = try CompactH5MetadataKernels(
      device: device, library: Metal4DSTEMKernels.makeCompactH5Library(device: device))
    for code in [UInt32(0), 1, 7] {
      var values = [UInt32](repeating: 0, count: 65_537)
      values[values.count - 1] = code
      let input = try XCTUnwrap(device.makeBuffer(bytes: values, length: values.count * 4))
      let status = try CompactH5MetadataKernels.buffer(device, bytes: 4, shared: true)
      status.contents().storeBytes(of: UInt32(0), as: UInt32.self)
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      try kernels.encodeDecodeStatus(input: input, count: values.count, status: status, command: command)
      try complete(command)
      XCTAssertEqual(status.contents().load(as: UInt32.self), code == 0 ? 0 : 16)
    }
  }

  private func readback(
    _ input: MTLBuffer, device: MTLDevice, command: MTLCommandBuffer
  ) throws -> MTLBuffer {
    let output = try CompactH5MetadataKernels.buffer(device, bytes: input.length, shared: true)
    let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
    blit.copy(from: input, sourceOffset: 0, to: output, destinationOffset: 0, size: input.length)
    blit.endEncoding()
    return output
  }

  private func complete(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed, command.error?.localizedDescription ?? "GPU failed")
  }
}
