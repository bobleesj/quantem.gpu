import Metal
import XCTest

@testable import Metal4DSTEMKernels

private struct CompactLZ4ChunkForTest {
  var inputOffset: UInt32
  var inputBytes: UInt32
  var outputWord: UInt32
  var outputBytes: UInt32
}

private struct CompactLZ4ParametersForTest {
  var chunkCount: UInt32
  var compressedBytes: UInt32
}

final class CompactH5KernelsTests: XCTestCase {
  func testRawLZ4DecodeWritesExactBytesOnMetal() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeCompactH5Library(device: device)
    let function = try XCTUnwrap(
      library.makeFunction(name: Metal4DSTEMKernels.compactH5DecodeFunction)
    )
    let pipeline = try device.makeComputePipelineState(function: function)

    let expected = (0..<128).map { UInt8(($0 * 37 + 11) & 0xff) }
    var compressed: [UInt8] = [0xf0, 113]
    compressed.append(contentsOf: expected)
    while compressed.count % MemoryLayout<UInt32>.stride != 0 {
      compressed.append(0)
    }
    let compressedBuffer = try XCTUnwrap(
      device.makeBuffer(bytes: compressed, length: compressed.count)
    )
    var chunk = CompactLZ4ChunkForTest(
      inputOffset: 0,
      inputBytes: 130,
      outputWord: 0,
      outputBytes: 128
    )
    let chunkBuffer = try XCTUnwrap(
      device.makeBuffer(bytes: &chunk, length: MemoryLayout.stride(ofValue: chunk))
    )
    let output = try XCTUnwrap(
      device.makeBuffer(length: 128, options: .storageModeShared)
    )
    let status = try XCTUnwrap(
      device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
    )
    memset(output.contents(), 0, output.length)
    memset(status.contents(), 0xff, status.length)
    var parameters = CompactLZ4ParametersForTest(
      chunkCount: 1,
      compressedBytes: 130
    )

    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(compressedBuffer, offset: 0, index: 0)
    encoder.setBuffer(chunkBuffer, offset: 0, index: 1)
    encoder.setBuffer(output, offset: 0, index: 2)
    encoder.setBuffer(status, offset: 0, index: 3)
    encoder.setBytes(
      &parameters,
      length: MemoryLayout.stride(ofValue: parameters),
      index: 4
    )
    encoder.dispatchThreadgroups(
      MTLSize(width: 1, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(
      command.status,
      .completed,
      command.error?.localizedDescription ?? "Metal command failed"
    )

    XCTAssertEqual(status.contents().load(as: UInt32.self), 0)
    let actual = Array(
      UnsafeBufferPointer(
        start: output.contents().bindMemory(to: UInt8.self, capacity: 128),
        count: 128
      )
    )
    XCTAssertEqual(actual, expected)
  }
}
