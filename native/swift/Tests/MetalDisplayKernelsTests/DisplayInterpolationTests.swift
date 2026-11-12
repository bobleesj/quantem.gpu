import Metal
import XCTest

@testable import MetalDisplayKernels

final class DisplayInterpolationTests: XCTestCase {
  func testIntegerPeakStaysCenteredAndSourceDoesNotChange() throws {
    let values: [UInt32] = [0, 0, 0, 0, 255, 0]
    var p = MetalDisplayParameters(rows: 2, cols: 3, low: 0, high: 255, scale: .linear)
    let nearest = try render(values, parameters: p, floating: false)
    p.interpolation = .linear
    let smooth = try render(values, parameters: p, floating: false)
    // 5x display: native pixel centers remain exact, including edges.
    for row in 0..<2 {
      for col in 0..<3 {
        let i = ((row * 5 + 2) * 15 + col * 5 + 2) * 4
        XCTAssertEqual(Array(smooth[i..<i + 4]), Array(nearest[i..<i + 4]))
      }
    }
    let peak = (7 * 15 + 7) * 4
    XCTAssertEqual(smooth[peak], 255)
    XCTAssertEqual(smooth[peak - 4], smooth[peak + 4])
    XCTAssertGreaterThan(smooth[peak - 4], 0)
    XCTAssertLessThan(smooth[peak - 4], 255)
    XCTAssertNotEqual(smooth, nearest)
    XCTAssertEqual(smooth[(9 * 15 + 7) * 4], 255)  // clamped bottom edge
  }

  func testFloatSignedDisplayHasNoHalfPixelShiftOrInvalidValueReplacement() throws {
    let values: [Float] = [-1, 0, 1]
    var p = MetalFloatDisplayParameters(rows: 1, cols: 3, low: -1, high: 1, scale: .linear)
    let nearest = try render(values, parameters: p, floating: true)
    p.interpolation = .linear
    let smooth = try render(values, parameters: p, floating: true)
    for col in 0..<3 {
      let i = (col * 5 + 2) * 4
      XCTAssertEqual(Array(smooth[i..<i + 4]), Array(nearest[i..<i + 4]))
    }
    XCTAssertEqual(smooth[0], 0)
    XCTAssertEqual(smooth[14 * 4], 255)
    XCTAssertGreaterThan(smooth[3 * 4], nearest[3 * 4])
    let invalid = try render([Float(-1), .nan, 1], parameters: p, floating: true)
    XCTAssertEqual(Array(invalid[7 * 4..<8 * 4]), [143, 63, 143, 255])
  }

  private func render<T, P>(_ source: [T], parameters: P, floating: Bool) throws -> [UInt8] {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try MetalDisplayKernels.makeLibrary(device: device)
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: MetalDisplayKernels.vertexFunction)
    descriptor.fragmentFunction = library.makeFunction(
      name: floating
        ? MetalDisplayKernels.floatFragmentFunction : MetalDisplayKernels.fragmentFunction)
    descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
    let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
    let height = source.count == 6 ? 10 : 5
    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .bgra8Unorm,
      width: 15, height: height, mipmapped: false)
    textureDescriptor.storageMode = .shared
    textureDescriptor.usage = .renderTarget
    let texture = try XCTUnwrap(device.makeTexture(descriptor: textureDescriptor))
    let original = source.withUnsafeBytes { Data($0) }
    let buffer = try XCTUnwrap(
      original.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    let lut = try MetalDisplayKernels.makeLUTBuffer(device: device, colormap: .gray)
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = texture
    pass.colorAttachments[0].loadAction = .dontCare
    pass.colorAttachments[0].storeAction = .store
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass))
    encoder.setRenderPipelineState(pipeline)
    encoder.setFragmentBuffer(buffer, offset: 0, index: 0)
    var p = parameters
    withUnsafeBytes(of: &p) {
      encoder.setFragmentBytes($0.baseAddress!, length: $0.count, index: 1)
    }
    encoder.setFragmentBuffer(lut, offset: 0, index: 2)
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed)
    XCTAssertEqual(Data(bytes: buffer.contents(), count: original.count), original)
    var pixels = [UInt8](repeating: 0, count: 15 * height * 4)
    texture.getBytes(
      &pixels, bytesPerRow: 60, from: MTLRegionMake2D(0, 0, 15, height), mipmapLevel: 0)
    return pixels
  }
}
