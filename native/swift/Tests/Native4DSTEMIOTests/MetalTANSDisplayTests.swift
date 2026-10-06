import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMKernels
import XCTest

final class MetalTANSDisplayTests: XCTestCase {
  func testDisplayWideningPreservesNativeUInt16CountsExactly() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    let function = try XCTUnwrap(library.makeFunction(name: "tans_display_counts"))
    let pipeline = try device.makeComputePipelineState(function: function)
    var values = (0..<36864).map { UInt16(truncatingIfNeeded: $0 * 199) }
    values[0] = 0
    values[1] = 65535
    values[2] = 256
    values[3] = 63
    let source = try XCTUnwrap(
      values.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let result = try XCTUnwrap(device.makeBuffer(length: 36864 * 4, options: .storageModeShared))
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(result, offset: 0, index: 1)
    encoder.dispatchThreads(
      MTLSize(width: 36864, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed)
    let actual = result.contents().assumingMemoryBound(to: UInt32.self)
    for index in values.indices { XCTAssertEqual(actual[index], UInt32(values[index])) }
  }
}
