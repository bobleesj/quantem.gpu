import Metal
import XCTest

@testable import Metal4DSTEMKernels

private struct DetectorParameters {
  var frameCount: UInt32
  var detectorPixels: UInt32
  var globalFrameOffset: UInt32
  var padding: UInt32 = 0
}

final class Metal4DSTEMKernelsTests: XCTestCase {
  func testHDF5FunctionsCompile() throws {
    let device = try metalDevice()
    let library = try Metal4DSTEMKernels.makeHDF5Library(device: device)
    let names = [
      Metal4DSTEMKernels.decodeU8Function,
      Metal4DSTEMKernels.decodeU16Function,
    ]
    for name in names {
      XCTAssertNotNil(library.makeFunction(name: name), "Missing Metal function \(name)")
    }
  }

  func testDetectorFunctionsCompile() throws {
    let device = try metalDevice()
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let names = [
      Metal4DSTEMKernels.detectorProductsU8Function,
      Metal4DSTEMKernels.detectorProductsU16Function,
      Metal4DSTEMKernels.detectorSumU8Function,
      Metal4DSTEMKernels.detectorSumU16Function,
      Metal4DSTEMKernels.transposeScanWordsFunction,
      Metal4DSTEMKernels.fullSumU8Function,
      Metal4DSTEMKernels.signedDeltaU8Function,
      Metal4DSTEMKernels.fullSumU16Function,
      Metal4DSTEMKernels.signedDeltaU16Function,
      Metal4DSTEMKernels.extractU8Function,
      Metal4DSTEMKernels.extractU16Function,
    ]
    for name in names {
      XCTAssertNotNil(library.makeFunction(name: name), "Missing Metal function \(name)")
    }
  }

  func testDetectorProductsU8MatchIntegerReference() throws {
    let values: [UInt8] = (0..<210).map { index in
      UInt8((index * 17 + 11) % 251)
    }
    try assertDetectorProducts(
      values: values,
      functionName: Metal4DSTEMKernels.detectorProductsU8Function
    )
  }

  func testDetectorProductsU16MatchIntegerReference() throws {
    let values: [UInt16] = (0..<210).map { index in
      UInt16((index * 17 + 11) % 1000)
    }
    try assertDetectorProducts(
      values: values,
      functionName: Metal4DSTEMKernels.detectorProductsU16Function
    )
  }

  func testDetectorSumsMatchIntegerReference() throws {
    let valuesU8: [UInt8] = (0..<210).map { index in
      UInt8((index * 13 + 7) % 251)
    }
    let valuesU16: [UInt16] = (0..<210).map { index in
      UInt16((index * 13 + 7) % 2000)
    }
    try assertDetectorSum(
      values: valuesU8,
      functionName: Metal4DSTEMKernels.detectorSumU8Function
    )
    try assertDetectorSum(
      values: valuesU16,
      functionName: Metal4DSTEMKernels.detectorSumU16Function
    )
  }

  private func assertDetectorProducts<Sample: FixedWidthInteger & UnsignedInteger>(
    values: [Sample],
    functionName: String
  ) throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let function = try XCTUnwrap(library.makeFunction(name: functionName))
    let pipeline = try device.makeComputePipelineState(function: function)
    let frameCount = 3
    let detectorPixels = 70
    XCTAssertEqual(values.count, frameCount * detectorPixels)
    let bands = (0..<detectorPixels).map { pixel -> UInt8 in
      (pixel % 2 == 0 ? 1 : 0)
        | (pixel % 3 == 0 ? 2 : 0)
        | (pixel % 5 == 0 ? 4 : 0)
    }
    let source = try makeBuffer(device: device, values: values)
    let bandBuffer = try makeBuffer(device: device, values: bands)
    let bf = try outputBuffer(device: device, count: frameCount)
    let abf = try outputBuffer(device: device, count: frameCount)
    let df = try outputBuffer(device: device, count: frameCount)
    var parameters = DetectorParameters(
      frameCount: UInt32(frameCount),
      detectorPixels: UInt32(detectorPixels),
      globalFrameOffset: 0
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(bf, offset: 0, index: 1)
    encoder.setBuffer(abf, offset: 0, index: 2)
    encoder.setBuffer(df, offset: 0, index: 3)
    withUnsafeBytes(of: &parameters) { bytes in
      encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 4)
    }
    encoder.setBuffer(bandBuffer, offset: 0, index: 5)
    encoder.dispatchThreadgroups(
      MTLSize(width: frameCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    var expected = [[UInt32]](repeating: [0, 0, 0], count: frameCount)
    for frame in 0..<frameCount {
      for pixel in 0..<detectorPixels {
        let value = UInt32(values[frame * detectorPixels + pixel])
        if bands[pixel] & 1 != 0 { expected[frame][0] += value }
        if bands[pixel] & 2 != 0 { expected[frame][1] += value }
        if bands[pixel] & 4 != 0 { expected[frame][2] += value }
      }
    }
    XCTAssertEqual(bufferValues(bf, count: frameCount), expected.map { $0[0] })
    XCTAssertEqual(bufferValues(abf, count: frameCount), expected.map { $0[1] })
    XCTAssertEqual(bufferValues(df, count: frameCount), expected.map { $0[2] })
  }

  private func assertDetectorSum<Sample: FixedWidthInteger & UnsignedInteger>(
    values: [Sample],
    functionName: String
  ) throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let function = try XCTUnwrap(library.makeFunction(name: functionName))
    let pipeline = try device.makeComputePipelineState(function: function)
    let frameCount = 3
    let detectorPixels = 70
    XCTAssertEqual(values.count, frameCount * detectorPixels)
    let source = try makeBuffer(device: device, values: values)
    let output = try outputBuffer(device: device, count: detectorPixels)
    var pixelCount = UInt32(detectorPixels)
    var frames = UInt32(frameCount)
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    withUnsafeBytes(of: &pixelCount) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 2)
    }
    withUnsafeBytes(of: &frames) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 3)
    }
    encoder.dispatchThreads(
      MTLSize(width: detectorPixels, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(detectorPixels, pipeline.maxTotalThreadsPerThreadgroup),
        height: 1,
        depth: 1
      )
    )
    encoder.endEncoding()
    try complete(command)

    let expected = (0..<detectorPixels).map { pixel in
      (0..<frameCount).reduce(UInt32(0)) { sum, frame in
        sum + UInt32(values[frame * detectorPixels + pixel])
      }
    }
    XCTAssertEqual(bufferValues(output, count: detectorPixels), expected)
  }

  func testStreamingTransposeMatchesReferenceWithOffset() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let function = try XCTUnwrap(
      library.makeFunction(name: Metal4DSTEMKernels.transposeScanWordsFunction)
    )
    let pipeline = try device.makeComputePipelineState(function: function)
    let sourceScans = 67
    let detectorWords = 70
    let destinationScans = 131
    let destinationOffset = 19
    let sourceValues = (0..<(sourceScans * detectorWords)).map(UInt32.init)
    let sentinel = UInt32.max
    let destinationValues = [UInt32](
      repeating: sentinel,
      count: destinationScans * detectorWords
    )
    let source = try makeBuffer(device: device, values: sourceValues)
    let destination = try makeBuffer(device: device, values: destinationValues)
    var sourceCount = UInt32(sourceScans)
    var wordCount = UInt32(detectorWords)
    var destinationCount = UInt32(destinationScans)
    var offset = UInt32(destinationOffset)
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    withUnsafeBytes(of: &sourceCount) { encoder.setBytes($0.baseAddress!, length: 4, index: 2) }
    withUnsafeBytes(of: &wordCount) { encoder.setBytes($0.baseAddress!, length: 4, index: 3) }
    withUnsafeBytes(of: &destinationCount) {
      encoder.setBytes($0.baseAddress!, length: 4, index: 4)
    }
    withUnsafeBytes(of: &offset) { encoder.setBytes($0.baseAddress!, length: 4, index: 5) }
    encoder.dispatchThreadgroups(
      MTLSize(
        width: (detectorWords + 31) / 32,
        height: (sourceScans + 31) / 32,
        depth: 1
      ),
      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    let actual = bufferValues(destination, count: destinationValues.count)
    for word in 0..<detectorWords {
      for scan in 0..<destinationScans {
        let value = actual[word * destinationScans + scan]
        if (destinationOffset..<(destinationOffset + sourceScans)).contains(scan) {
          let sourceScan = scan - destinationOffset
          XCTAssertEqual(value, sourceValues[sourceScan * detectorWords + word])
        } else {
          XCTAssertEqual(value, sentinel)
        }
      }
    }
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

  private func outputBuffer(device: MTLDevice, count: Int) throws -> MTLBuffer {
    let buffer = try XCTUnwrap(
      device.makeBuffer(
        length: count * MemoryLayout<UInt32>.stride,
        options: .storageModeShared
      )
    )
    memset(buffer.contents(), 0, buffer.length)
    return buffer
  }

  private func bufferValues(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
    let pointer = buffer.contents().bindMemory(to: UInt32.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
  }

  private func complete(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { throw error }
    XCTAssertEqual(command.status, .completed)
  }
}
