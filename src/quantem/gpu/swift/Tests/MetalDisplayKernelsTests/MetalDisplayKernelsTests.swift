import Metal
import XCTest

@testable import MetalDisplayKernels

final class MetalDisplayKernelsTests: XCTestCase {
  func testDisplayParameterLayoutMatchesMetalABI() {
    XCTAssertEqual(MemoryLayout<MetalDisplayParameters>.size, 32)
    XCTAssertEqual(MemoryLayout<MetalDisplayParameters>.stride, 32)
    XCTAssertEqual(MemoryLayout<MetalDisplayParameters>.alignment, 4)

    var parameters = MetalDisplayParameters(
      rows: 1,
      cols: 2,
      low: 3,
      high: 4,
      scale: .logarithmic,
      lutCount: 5
    )
    let words = withUnsafeBytes(of: &parameters) { bytes in
      Array(bytes.bindMemory(to: UInt32.self))
    }
    XCTAssertEqual(words, [1, 2, 3, 4, 1, 5, 0, 0])
  }

  func testFloatDisplayParameterLayoutMatchesMetalABI() {
    XCTAssertEqual(MemoryLayout<MetalFloatDisplayParameters>.size, 32)
    XCTAssertEqual(MemoryLayout<MetalFloatDisplayParameters>.stride, 32)
    XCTAssertEqual(MemoryLayout<MetalFloatDisplayParameters>.alignment, 4)
  }

  func testColormapLUTsAreValid() throws {
    XCTAssertEqual(MetalColormap.allCases.count, 15)
    for colormap in MetalColormap.allCases {
      let lut = try MetalDisplayKernels.lut(colormap)
      XCTAssertEqual(lut.count, 256)
      for color in lut {
        XCTAssertTrue(color.x.isFinite)
        XCTAssertTrue(color.y.isFinite)
        XCTAssertTrue(color.z.isFinite)
        XCTAssertTrue((0...1).contains(color.x))
        XCTAssertTrue((0...1).contains(color.y))
        XCTAssertTrue((0...1).contains(color.z))
        XCTAssertEqual(color.w, 1)
      }
    }
  }

  func testGrayLUTEndpoints() throws {
    let lut = try MetalDisplayKernels.lut(.gray)
    XCTAssertEqual(lut.first, .init(0, 0, 0, 1))
    XCTAssertEqual(lut.last, .init(1, 1, 1, 1))
  }

  func testMetalFunctionsCompile() throws {
    let device = try metalDevice()
    let library = try MetalDisplayKernels.makeLibrary(device: device)
    let names = [
      MetalDisplayKernels.vertexFunction,
      MetalDisplayKernels.fragmentFunction,
      MetalDisplayKernels.rangeFunction,
      MetalDisplayKernels.histogramFunction,
      MetalDisplayKernels.floatFragmentFunction,
      MetalDisplayKernels.floatHistogramFunction,
    ]
    for name in names {
      XCTAssertNotNil(library.makeFunction(name: name), "Missing Metal function \(name)")
    }
  }

  func testLinearFragmentRendersExactGrayLUTIndices() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try MetalDisplayKernels.makeLibrary(device: device)
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = try XCTUnwrap(
      library.makeFunction(name: MetalDisplayKernels.vertexFunction)
    )
    descriptor.fragmentFunction = try XCTUnwrap(
      library.makeFunction(name: MetalDisplayKernels.fragmentFunction)
    )
    descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
    let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .bgra8Unorm,
      width: 5,
      height: 1,
      mipmapped: false
    )
    textureDescriptor.usage = .renderTarget
    textureDescriptor.storageMode = .shared
    let texture = try XCTUnwrap(device.makeTexture(descriptor: textureDescriptor))
    let values = try makeBuffer(device: device, values: Array(UInt32(0)...4))
    let lut = try MetalDisplayKernels.makeLUTBuffer(device: device, colormap: .gray)
    var parameters = MetalDisplayParameters(
      rows: 1,
      cols: 5,
      low: 0,
      high: 4,
      scale: .linear
    )
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = texture
    pass.colorAttachments[0].loadAction = .dontCare
    pass.colorAttachments[0].storeAction = .store
    let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(
      commandBuffer.makeRenderCommandEncoder(descriptor: pass)
    )
    encoder.setRenderPipelineState(pipeline)
    encoder.setFragmentBuffer(values, offset: 0, index: 0)
    withUnsafeBytes(of: &parameters) { bytes in
      encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: 1)
    }
    encoder.setFragmentBuffer(lut, offset: 0, index: 2)
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertEqual(commandBuffer.status, .completed)

    var pixels = [UInt8](repeating: 0, count: 20)
    texture.getBytes(
      &pixels,
      bytesPerRow: 20,
      from: MTLRegionMake2D(0, 0, 5, 1),
      mipmapLevel: 0
    )
    XCTAssertEqual(
      pixels,
      [
        0, 0, 0, 255,
        63, 63, 63, 255,
        127, 127, 127, 255,
        191, 191, 191, 255,
        255, 255, 255, 255,
      ]
    )
  }

  func testSignedLogFloatFragmentRendersExactGrayLUTIndices() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try MetalDisplayKernels.makeLibrary(device: device)
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = try XCTUnwrap(
      library.makeFunction(name: MetalDisplayKernels.vertexFunction)
    )
    descriptor.fragmentFunction = try XCTUnwrap(
      library.makeFunction(name: MetalDisplayKernels.floatFragmentFunction)
    )
    descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
    let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .bgra8Unorm,
      width: 5,
      height: 1,
      mipmapped: false
    )
    textureDescriptor.usage = .renderTarget
    textureDescriptor.storageMode = .shared
    let texture = try XCTUnwrap(device.makeTexture(descriptor: textureDescriptor))
    let values = try makeBuffer(device: device, values: [Float(-7), -3, 0, 3, 7])
    let lut = try MetalDisplayKernels.makeLUTBuffer(device: device, colormap: .gray)
    var parameters = MetalFloatDisplayParameters(
      rows: 1,
      cols: 5,
      low: -7,
      high: 7,
      scale: .logarithmic
    )
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = texture
    pass.colorAttachments[0].loadAction = .dontCare
    pass.colorAttachments[0].storeAction = .store
    let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(
      commandBuffer.makeRenderCommandEncoder(descriptor: pass)
    )
    encoder.setRenderPipelineState(pipeline)
    encoder.setFragmentBuffer(values, offset: 0, index: 0)
    withUnsafeBytes(of: &parameters) { bytes in
      encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: 1)
    }
    encoder.setFragmentBuffer(lut, offset: 0, index: 2)
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertEqual(commandBuffer.status, .completed)

    var pixels = [UInt8](repeating: 0, count: 20)
    texture.getBytes(
      &pixels,
      bytesPerRow: 20,
      from: MTLRegionMake2D(0, 0, 5, 1),
      mipmapLevel: 0
    )
    XCTAssertEqual(
      pixels,
      [
        0, 0, 0, 255,
        42, 42, 42, 255,
        127, 127, 127, 255,
        212, 212, 212, 255,
        255, 255, 255, 255,
      ]
    )
  }

  func testRangeAndLinearHistogramParity() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try MetalDisplayKernels.makeLibrary(device: device)
    let rangePipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: MetalDisplayKernels.rangeFunction)
      )
    )
    let histogramPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: MetalDisplayKernels.histogramFunction)
      )
    )
    let values = Array(UInt32(0)...UInt32(7))
    let valueBuffer = try makeBuffer(device: device, values: values)
    let rangeBuffer = try XCTUnwrap(
      device.makeBuffer(length: 2 * MemoryLayout<UInt32>.stride)
    )
    let binsBuffer = try XCTUnwrap(
      device.makeBuffer(length: 256 * MemoryLayout<UInt32>.stride)
    )
    let range = rangeBuffer.contents().bindMemory(to: UInt32.self, capacity: 2)
    range[0] = .max
    range[1] = 0
    memset(binsBuffer.contents(), 0, binsBuffer.length)

    let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
    let rangeEncoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
    rangeEncoder.setComputePipelineState(rangePipeline)
    rangeEncoder.setBuffer(valueBuffer, offset: 0, index: 0)
    rangeEncoder.setBuffer(rangeBuffer, offset: 0, index: 1)
    var count = UInt32(values.count)
    withUnsafeBytes(of: &count) { bytes in
      rangeEncoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
    }
    dispatch(rangeEncoder, pipeline: rangePipeline, count: values.count)
    rangeEncoder.endEncoding()

    let histogramEncoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
    histogramEncoder.setComputePipelineState(histogramPipeline)
    histogramEncoder.setBuffer(valueBuffer, offset: 0, index: 0)
    histogramEncoder.setBuffer(binsBuffer, offset: 0, index: 1)
    var parameters = MetalDisplayParameters(
      rows: 1,
      cols: values.count,
      low: 0,
      high: 7,
      scale: .linear
    )
    withUnsafeBytes(of: &parameters) { bytes in
      histogramEncoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
    }
    dispatch(histogramEncoder, pipeline: histogramPipeline, count: values.count)
    histogramEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertEqual(commandBuffer.status, .completed)

    XCTAssertEqual(Array(UnsafeBufferPointer(start: range, count: 2)), [0, 7])
    let bins = binsBuffer.contents().bindMemory(to: UInt32.self, capacity: 256)
    let histogramSum = (0..<256).reduce(UInt64(0)) { total, index in
      total + UInt64(bins[index])
    }
    XCTAssertEqual(histogramSum, UInt64(values.count))
    for index in [0, 36, 73, 109, 146, 182, 219, 255] {
      XCTAssertEqual(bins[index], 1, "Unexpected count in bin \(index)")
    }
  }

  func testLogHistogramParity() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try MetalDisplayKernels.makeLibrary(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: MetalDisplayKernels.histogramFunction)
      )
    )
    let values = Array(UInt32(0)...UInt32(7))
    let valueBuffer = try makeBuffer(device: device, values: values)
    let binsBuffer = try XCTUnwrap(
      device.makeBuffer(length: 256 * MemoryLayout<UInt32>.stride)
    )
    memset(binsBuffer.contents(), 0, binsBuffer.length)
    var parameters = MetalDisplayParameters(
      rows: 1,
      cols: values.count,
      low: 0,
      high: 7,
      scale: .logarithmic
    )
    let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(valueBuffer, offset: 0, index: 0)
    encoder.setBuffer(binsBuffer, offset: 0, index: 1)
    withUnsafeBytes(of: &parameters) { bytes in
      encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
    }
    dispatch(encoder, pipeline: pipeline, count: values.count)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertEqual(commandBuffer.status, .completed)

    let bins = binsBuffer.contents().bindMemory(to: UInt32.self, capacity: 256)
    for index in [0, 85, 135, 170, 198, 220, 239, 255] {
      XCTAssertEqual(bins[index], 1, "Unexpected log count in bin \(index)")
    }
  }

  func testSignedLogFloatHistogramParity() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try MetalDisplayKernels.makeLibrary(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: MetalDisplayKernels.floatHistogramFunction)
      )
    )
    let values: [Float] = [-7, -3, 0, 3, 7]
    let valueBuffer = try makeBuffer(device: device, values: values)
    let binsBuffer = try XCTUnwrap(
      device.makeBuffer(length: 256 * MemoryLayout<UInt32>.stride)
    )
    memset(binsBuffer.contents(), 0, binsBuffer.length)
    var parameters = MetalFloatDisplayParameters(
      rows: 1,
      cols: values.count,
      low: -7,
      high: 7,
      scale: .logarithmic
    )
    let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(valueBuffer, offset: 0, index: 0)
    encoder.setBuffer(binsBuffer, offset: 0, index: 1)
    withUnsafeBytes(of: &parameters) { bytes in
      encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
    }
    dispatch(encoder, pipeline: pipeline, count: values.count)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertEqual(commandBuffer.status, .completed)

    let bins = binsBuffer.contents().bindMemory(to: UInt32.self, capacity: 256)
    for index in [0, 42, 128, 213, 255] {
      XCTAssertEqual(bins[index], 1, "Unexpected signed-log count in bin \(index)")
    }
  }

  func testFloatHistogramConstantAndNonfiniteParity() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try MetalDisplayKernels.makeLibrary(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: MetalDisplayKernels.floatHistogramFunction)
      )
    )

    func histogram(_ values: [Float], low: Float, high: Float) throws -> [UInt32] {
      let valueBuffer = try makeBuffer(device: device, values: values)
      let binsBuffer = try XCTUnwrap(
        device.makeBuffer(length: 256 * MemoryLayout<UInt32>.stride)
      )
      memset(binsBuffer.contents(), 0, binsBuffer.length)
      var parameters = MetalFloatDisplayParameters(
        rows: 1,
        cols: values.count,
        low: low,
        high: high,
        scale: .linear
      )
      let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
      let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(valueBuffer, offset: 0, index: 0)
      encoder.setBuffer(binsBuffer, offset: 0, index: 1)
      withUnsafeBytes(of: &parameters) { bytes in
        encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
      }
      dispatch(encoder, pipeline: pipeline, count: values.count)
      encoder.endEncoding()
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      XCTAssertEqual(commandBuffer.status, .completed)
      let bins = binsBuffer.contents().bindMemory(to: UInt32.self, capacity: 256)
      return Array(UnsafeBufferPointer(start: bins, count: 256))
    }

    let constant = try histogram([-7, 0, 7], low: 3, high: 3)
    XCTAssertEqual(constant.reduce(0, +), 3)
    XCTAssertEqual(constant[128], 3)

    let nonfinite = try histogram(
      [.nan, -.infinity, .infinity, -1, 0, 1],
      low: -1,
      high: 1
    )
    XCTAssertEqual(nonfinite.reduce(0, +), 3)
    XCTAssertEqual(nonfinite[0], 1)
    XCTAssertEqual(nonfinite[128], 1)
    XCTAssertEqual(nonfinite[255], 1)
  }

  func testFloatFragmentRendersNonfiniteWithExplicitInvalidColor() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try MetalDisplayKernels.makeLibrary(device: device)
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = try XCTUnwrap(
      library.makeFunction(name: MetalDisplayKernels.vertexFunction)
    )
    descriptor.fragmentFunction = try XCTUnwrap(
      library.makeFunction(name: MetalDisplayKernels.floatFragmentFunction)
    )
    descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
    let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .bgra8Unorm,
      width: 1,
      height: 1,
      mipmapped: false
    )
    textureDescriptor.usage = .renderTarget
    textureDescriptor.storageMode = .shared
    let texture = try XCTUnwrap(device.makeTexture(descriptor: textureDescriptor))
    let values = try makeBuffer(device: device, values: [Float.nan])
    let lut = try MetalDisplayKernels.makeLUTBuffer(device: device, colormap: .gray)
    var parameters = MetalFloatDisplayParameters(
      rows: 1,
      cols: 1,
      low: 0,
      high: 1,
      scale: .linear
    )
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = texture
    pass.colorAttachments[0].loadAction = .dontCare
    pass.colorAttachments[0].storeAction = .store
    let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(
      commandBuffer.makeRenderCommandEncoder(descriptor: pass)
    )
    encoder.setRenderPipelineState(pipeline)
    encoder.setFragmentBuffer(values, offset: 0, index: 0)
    withUnsafeBytes(of: &parameters) { bytes in
      encoder.setFragmentBytes(bytes.baseAddress!, length: bytes.count, index: 1)
    }
    encoder.setFragmentBuffer(lut, offset: 0, index: 2)
    encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    XCTAssertEqual(commandBuffer.status, .completed)

    var pixels = [UInt8](repeating: 0, count: 4)
    texture.getBytes(
      &pixels,
      bytesPerRow: 4,
      from: MTLRegionMake2D(0, 0, 1, 1),
      mipmapLevel: 0
    )
    XCTAssertEqual(pixels, [143, 63, 143, 255])
  }

  private func metalDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("No Metal device is available on this host.")
    }
    return device
  }

  private func makeBuffer<T>(device: MTLDevice, values: [T]) throws -> MTLBuffer {
    try values.withUnsafeBytes { bytes in
      try XCTUnwrap(
        device.makeBuffer(
          bytes: bytes.baseAddress!,
          length: bytes.count,
          options: .storageModeShared
        )
      )
    }
  }

  private func dispatch(
    _ encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState,
    count: Int
  ) {
    let width = min(pipeline.maxTotalThreadsPerThreadgroup, max(1, count))
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
    )
  }
}
