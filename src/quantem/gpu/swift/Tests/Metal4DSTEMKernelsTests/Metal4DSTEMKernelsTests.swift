import Metal
import XCTest

@testable import Metal4DSTEMKernels

private struct DetectorParameters {
  var frameCount: UInt32
  var detectorPixels: UInt32
  var globalFrameOffset: UInt32
  var padding: UInt32 = 0
}

private struct ScanBinParameters {
  var sourceRows: UInt32
  var sourceCols: UInt32
  var detectorPixels: UInt32
  var scanBin: UInt32
  var outputScanCount: UInt32
  var outputCols: UInt32
  var destinationRowOffset: UInt32
  var padding: UInt32 = 0
}

private struct WordMajorDetectorParameters {
  var scanCount: UInt32
  var detectorPixels: UInt32
}

private struct FFT2DParameters {
  var width: UInt32
  var height: UInt32
  var log2Size: UInt32
  var stage: UInt32
  var direction: Float
  var rowAxis: UInt32
}

private struct Bluestein2DParameters {
  var sourceWidth: UInt32
  var sourceHeight: UInt32
  var paddedWidth: UInt32
  var paddedHeight: UInt32
  var direction: Float
  var scale: Float
  var padding0: UInt32 = 0
  var padding1: UInt32 = 0
}

private struct ResidentRebinParameters {
  var sourceRows: UInt32
  var sourceCols: UInt32
  var sourceScanCount: UInt32
  var sourceDetectorRows: UInt32
  var sourceDetectorCols: UInt32
  var sourceRowOffset: UInt32
  var sourceColOffset: UInt32
  var selectedRows: UInt32
  var selectedCols: UInt32
  var scanBin: UInt32
  var detectorBin: UInt32
  var outputRows: UInt32
  var outputCols: UInt32
  var outputScanCount: UInt32
  var outputDetectorRows: UInt32
  var outputDetectorCols: UInt32
}

private struct ScanDetectorBinParameters {
  var sourceRows: UInt32
  var sourceCols: UInt32
  var sourceDetectorRows: UInt32
  var sourceDetectorCols: UInt32
  var scanBin: UInt32
  var detectorBin: UInt32
  var outputScanCount: UInt32
  var outputScanCols: UInt32
  var outputDetectorRows: UInt32
  var outputDetectorCols: UInt32
  var destinationScanRowOffset: UInt32
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
      Metal4DSTEMKernels.scanBinU8Function,
      Metal4DSTEMKernels.scanBinU16Function,
      Metal4DSTEMKernels.scanDetectorBinU8Function,
      Metal4DSTEMKernels.scanDetectorBinU16Function,
      Metal4DSTEMKernels.residentRebinU8Function,
      Metal4DSTEMKernels.residentRebinU16Function,
      Metal4DSTEMKernels.residentRebinU32Function,
      Metal4DSTEMKernels.detectorProductsU32Function,
      Metal4DSTEMKernels.centerOfMassU8Function,
      Metal4DSTEMKernels.centerOfMassU16Function,
      Metal4DSTEMKernels.centerOfMassU32Function,
      Metal4DSTEMKernels.fullSumU8Function,
      Metal4DSTEMKernels.signedDeltaU8Function,
      Metal4DSTEMKernels.fullSumU16Function,
      Metal4DSTEMKernels.signedDeltaU16Function,
      Metal4DSTEMKernels.fullSumU32Function,
      Metal4DSTEMKernels.signedDeltaU32Function,
      Metal4DSTEMKernels.extractU8Function,
      Metal4DSTEMKernels.extractU16Function,
      Metal4DSTEMKernels.extractU32Function,
      Metal4DSTEMKernels.extractU8ToU32Function,
      Metal4DSTEMKernels.extractU16ToU32Function,
      Metal4DSTEMKernels.extractU32ToU32Function,
    ]
    for name in names {
      XCTAssertNotNil(library.makeFunction(name: name), "Missing Metal function \(name)")
    }
  }

  func testDPCFunctionsCompile() throws {
    let device = try metalDevice()
    let library = try Metal4DSTEMKernels.makeDPCLibrary(device: device)
    let names = [
      Metal4DSTEMKernels.dpcPackFunction,
      Metal4DSTEMKernels.fftBitReverseRowsFunction,
      Metal4DSTEMKernels.fftBitReverseColumnsFunction,
      Metal4DSTEMKernels.fftButterflyRowsFunction,
      Metal4DSTEMKernels.fftButterflyColumnsFunction,
      Metal4DSTEMKernels.fftNormalizeFunction,
      Metal4DSTEMKernels.bluesteinPrepareFunction,
      Metal4DSTEMKernels.complexMultiplyFunction,
      Metal4DSTEMKernels.bluesteinExtractFunction,
      Metal4DSTEMKernels.dpcPoissonFunction,
      Metal4DSTEMKernels.dpcExtractPhaseFunction,
    ]
    for name in names {
      XCTAssertNotNil(library.makeFunction(name: name), "Missing Metal function \(name)")
    }
  }

  func testBluesteinFFTMatchesDirectDFTForArbitraryShape() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDPCLibrary(device: device)
    let rows = 3
    let columns = 5
    let paddedRows = 8
    let paddedColumns = 16
    let values = (0..<(rows * columns)).map { index in
      SIMD2<Float>(Float(index % 4) - 1.5, Float((index * 3) % 5) - 2)
    }
    let source = try makeBuffer(device: device, values: values)
    let destination = try XCTUnwrap(
      device.makeBuffer(
        length: values.count * MemoryLayout<SIMD2<Float>>.stride,
        options: .storageModeShared
      )
    )
    let workspaceBytes =
      paddedRows * paddedColumns * MemoryLayout<SIMD2<Float>>.stride
    let signal = try XCTUnwrap(
      device.makeBuffer(length: workspaceBytes, options: .storageModePrivate)
    )
    let chirp = try XCTUnwrap(
      device.makeBuffer(length: workspaceBytes, options: .storageModePrivate)
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    try encodeBluesteinFFT(
      encoder,
      library: library,
      device: device,
      source: source,
      destination: destination,
      rows: rows,
      columns: columns,
      paddedRows: paddedRows,
      paddedColumns: paddedColumns,
      signal: signal,
      chirp: chirp,
      inverse: false
    )
    encoder.endEncoding()
    try complete(command)

    let result = destination.contents().bindMemory(
      to: SIMD2<Float>.self,
      capacity: values.count
    )
    for outputRow in 0..<rows {
      for outputColumn in 0..<columns {
        var expected = SIMD2<Double>(repeating: 0)
        for inputRow in 0..<rows {
          for inputColumn in 0..<columns {
            let value = values[inputRow * columns + inputColumn]
            let angle = -2 * Double.pi * (
              Double(outputRow * inputRow) / Double(rows)
                + Double(outputColumn * inputColumn) / Double(columns)
            )
            let cosine = cos(angle)
            let sine = sin(angle)
            expected.x += Double(value.x) * cosine - Double(value.y) * sine
            expected.y += Double(value.x) * sine + Double(value.y) * cosine
          }
        }
        let actual = result[outputRow * columns + outputColumn]
        XCTAssertEqual(actual.x, Float(expected.x), accuracy: 2e-4)
        XCTAssertEqual(actual.y, Float(expected.y), accuracy: 2e-4)
      }
    }

    let recovered = try XCTUnwrap(
      device.makeBuffer(
        length: values.count * MemoryLayout<SIMD2<Float>>.stride,
        options: .storageModeShared
      )
    )
    let inverseCommand = try XCTUnwrap(queue.makeCommandBuffer())
    let inverseEncoder = try XCTUnwrap(inverseCommand.makeComputeCommandEncoder())
    try encodeBluesteinFFT(
      inverseEncoder,
      library: library,
      device: device,
      source: destination,
      destination: recovered,
      rows: rows,
      columns: columns,
      paddedRows: paddedRows,
      paddedColumns: paddedColumns,
      signal: signal,
      chirp: chirp,
      inverse: true
    )
    inverseEncoder.endEncoding()
    try complete(inverseCommand)
    let recoveredValues = recovered.contents().bindMemory(
      to: SIMD2<Float>.self,
      capacity: values.count
    )
    for index in values.indices {
      XCTAssertEqual(recoveredValues[index].x, values[index].x, accuracy: 2e-4)
      XCTAssertEqual(recoveredValues[index].y, values[index].y, accuracy: 2e-4)
    }
  }

  func testWordMajorU32CenterOfMassMatchesReference() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.centerOfMassU32Function)
      )
    )
    let scanCount = 3
    let detectorRows = 4
    let detectorColumns = 5
    let detectorPixels = detectorRows * detectorColumns
    let values: [UInt32] = (0..<detectorPixels).flatMap { pixel in
      (0..<scanCount).map { scan in UInt32((pixel + 2) * (scan + 1)) }
    }
    let source = try makeBuffer(device: device, values: values)
    let rowOutput = try XCTUnwrap(
      device.makeBuffer(length: scanCount * MemoryLayout<Float>.stride)
    )
    let columnOutput = try XCTUnwrap(
      device.makeBuffer(length: scanCount * MemoryLayout<Float>.stride)
    )
    var parameters = WordMajorDetectorParameters(
      scanCount: UInt32(scanCount),
      detectorPixels: UInt32(detectorPixels)
    )
    var columns = UInt32(detectorColumns)
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(rowOutput, offset: 0, index: 1)
    encoder.setBuffer(columnOutput, offset: 0, index: 2)
    withUnsafeBytes(of: &parameters) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 3)
    }
    withUnsafeBytes(of: &columns) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 4)
    }
    encoder.dispatchThreadgroups(
      MTLSize(width: scanCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    let expected = (0..<scanCount).map { scan -> (Float, Float) in
      var total = 0.0
      var rowMoment = 0.0
      var columnMoment = 0.0
      for pixel in 0..<detectorPixels {
        let value = Double(values[pixel * scanCount + scan])
        total += value
        rowMoment += value * Double(pixel / detectorColumns)
        columnMoment += value * Double(pixel % detectorColumns)
      }
      return (Float(rowMoment / total), Float(columnMoment / total))
    }
    let rows = floatBufferValues(rowOutput, count: scanCount)
    let columnsOutput = floatBufferValues(columnOutput, count: scanCount)
    for scan in 0..<scanCount {
      XCTAssertEqual(rows[scan], expected[scan].0, accuracy: 1e-5)
      XCTAssertEqual(columnsOutput[scan], expected[scan].1, accuracy: 1e-5)
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

  func testLoadPlanPreservesCropAndIncompleteEdgeBins() throws {
    let region = try Metal4DSTEMScanRegion(
      rowStart: 2,
      rowStop: 7,
      columnStart: 3,
      columnStop: 10,
      sourceRows: 9,
      sourceColumns: 12
    )
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 9,
      sourceScanColumns: 12,
      detectorRows: 8,
      detectorColumns: 10,
      sourceBytesPerValue: 2,
      scanRegion: region,
      scanBin: 4
    )
    XCTAssertEqual(plan.outputScanRows, 2)
    XCTAssertEqual(plan.outputScanColumns, 2)
    XCTAssertEqual(plan.outputScanPositions, 4)
    XCTAssertEqual(plan.sourceContributionCount(outputRow: 0, outputColumn: 0), 16)
    XCTAssertEqual(plan.sourceContributionCount(outputRow: 0, outputColumn: 1), 12)
    XCTAssertEqual(plan.sourceContributionCount(outputRow: 1, outputColumn: 0), 4)
    XCTAssertEqual(plan.sourceContributionCount(outputRow: 1, outputColumn: 1), 3)
    XCTAssertEqual(plan.residentVolumeBytes, 4 * 80 * 4)
    XCTAssertFalse(plan.isFullNative)
  }

  func testLoadPlanAccountsForExactDetectorSumBinning() throws {
    let region = try Metal4DSTEMScanRegion.full(sourceRows: 8, sourceColumns: 8)
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 8,
      sourceScanColumns: 8,
      detectorRows: 5,
      detectorColumns: 7,
      sourceBytesPerValue: 2,
      scanRegion: region,
      detectorBin: 2
    )

    XCTAssertEqual(plan.outputDetectorRows, 3)
    XCTAssertEqual(plan.outputDetectorColumns, 4)
    XCTAssertEqual(plan.outputDetectorPixels, 12)
    XCTAssertEqual(plan.residentBytesPerValue, MemoryLayout<UInt32>.stride)
    XCTAssertEqual(plan.residentVolumeBytes, 64 * 12 * 4)
    XCTAssertEqual(plan.detectorContributionCount(outputRow: 0, outputColumn: 0), 4)
    XCTAssertEqual(plan.detectorContributionCount(outputRow: 2, outputColumn: 3), 1)
    XCTAssertTrue(plan.provenanceLabel.contains("detector-sum bin 2×2"))
  }

  func testScanBinU16MatchesIntegerReferenceIncludingEdges() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let function = try XCTUnwrap(
      library.makeFunction(name: Metal4DSTEMKernels.scanBinU16Function)
    )
    let pipeline = try device.makeComputePipelineState(function: function)
    let sourceRows = 3
    let sourceColumns = 5
    let detectorPixels = 3
    let scanBin = 2
    let outputRows = 2
    let outputColumns = 3
    let outputScans = outputRows * outputColumns
    let sourceValues: [UInt16] = (0..<(sourceRows * sourceColumns * detectorPixels)).map {
      UInt16($0 + 1)
    }
    let source = try makeBuffer(device: device, values: sourceValues)
    let destination = try outputBuffer(device: device, count: outputScans * detectorPixels)
    var parameters = ScanBinParameters(
      sourceRows: UInt32(sourceRows),
      sourceCols: UInt32(sourceColumns),
      detectorPixels: UInt32(detectorPixels),
      scanBin: UInt32(scanBin),
      outputScanCount: UInt32(outputScans),
      outputCols: UInt32(outputColumns),
      destinationRowOffset: 0
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    withUnsafeBytes(of: &parameters) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 2)
    }
    encoder.dispatchThreads(
      MTLSize(width: detectorPixels, height: outputScans, depth: 1),
      threadsPerThreadgroup: MTLSize(width: detectorPixels, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    var expected = [UInt32](repeating: 0, count: outputScans * detectorPixels)
    for outputRow in 0..<outputRows {
      for outputColumn in 0..<outputColumns {
        let outputScan = outputRow * outputColumns + outputColumn
        for detectorPixel in 0..<detectorPixels {
          for sourceRow in (outputRow * scanBin)..<min(sourceRows, (outputRow + 1) * scanBin) {
            for sourceColumn
              in (outputColumn * scanBin)..<min(sourceColumns, (outputColumn + 1) * scanBin)
            {
              let sourceScan = sourceRow * sourceColumns + sourceColumn
              expected[detectorPixel * outputScans + outputScan] += UInt32(
                sourceValues[sourceScan * detectorPixels + detectorPixel]
              )
            }
          }
        }
      }
    }
    XCTAssertEqual(
      bufferValues(destination, count: expected.count),
      expected
    )
  }

  func testScanBinU8MatchesIntegerReference() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.scanBinU8Function)
      )
    )
    let source = try makeBuffer(
      device: device,
      values: Array(UInt8(1)...UInt8(12))
    )
    let destination = try outputBuffer(device: device, count: 4)
    var parameters = ScanBinParameters(
      sourceRows: 2,
      sourceCols: 3,
      detectorPixels: 2,
      scanBin: 2,
      outputScanCount: 2,
      outputCols: 2,
      destinationRowOffset: 0
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    withUnsafeBytes(of: &parameters) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 2)
    }
    encoder.dispatchThreads(
      MTLSize(width: 2, height: 2, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 2, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    XCTAssertEqual(bufferValues(destination, count: 4), [20, 16, 24, 18])
  }

  func testCombinedScanAndDetectorBinU16MatchesIntegerReference() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.scanDetectorBinU16Function)
      )
    )
    let scanRows = 3
    let scanColumns = 5
    let detectorRows = 3
    let detectorColumns = 5
    let detectorPixels = detectorRows * detectorColumns
    let sourceValues: [UInt16] = (0..<(scanRows * scanColumns * detectorPixels)).map {
      UInt16($0 + 1)
    }
    let source = try makeBuffer(device: device, values: sourceValues)
    let outputScanRows = 2
    let outputScanColumns = 3
    let outputScans = outputScanRows * outputScanColumns
    let outputDetectorRows = 2
    let outputDetectorColumns = 3
    let outputDetectorPixels = outputDetectorRows * outputDetectorColumns
    let destination = try outputBuffer(
      device: device,
      count: outputScans * outputDetectorPixels
    )
    var parameters = ScanDetectorBinParameters(
      sourceRows: UInt32(scanRows),
      sourceCols: UInt32(scanColumns),
      sourceDetectorRows: UInt32(detectorRows),
      sourceDetectorCols: UInt32(detectorColumns),
      scanBin: 2,
      detectorBin: 2,
      outputScanCount: UInt32(outputScans),
      outputScanCols: UInt32(outputScanColumns),
      outputDetectorRows: UInt32(outputDetectorRows),
      outputDetectorCols: UInt32(outputDetectorColumns),
      destinationScanRowOffset: 0
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    withUnsafeBytes(of: &parameters) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 2)
    }
    encoder.dispatchThreads(
      MTLSize(width: outputDetectorPixels, height: outputScans, depth: 1),
      threadsPerThreadgroup: MTLSize(width: outputDetectorPixels, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    var expected = [UInt32](repeating: 0, count: outputScans * outputDetectorPixels)
    for outputScanRow in 0..<outputScanRows {
      for outputScanColumn in 0..<outputScanColumns {
        let outputScan = outputScanRow * outputScanColumns + outputScanColumn
        for outputDetectorRow in 0..<outputDetectorRows {
          for outputDetectorColumn in 0..<outputDetectorColumns {
            let outputDetector =
              outputDetectorRow * outputDetectorColumns + outputDetectorColumn
            for scanRow
              in (outputScanRow * 2)..<min(scanRows, (outputScanRow + 1) * 2)
            {
              for scanColumn
                in (outputScanColumn * 2)..<min(scanColumns, (outputScanColumn + 1) * 2)
              {
                let sourceScan = scanRow * scanColumns + scanColumn
                for detectorRow
                  in (outputDetectorRow * 2)..<min(detectorRows, (outputDetectorRow + 1) * 2)
                {
                  for detectorColumn
                    in (outputDetectorColumn * 2)..<min(
                      detectorColumns, (outputDetectorColumn + 1) * 2
                    )
                  {
                    let sourceDetector = detectorRow * detectorColumns + detectorColumn
                    expected[outputDetector * outputScans + outputScan] += UInt32(
                      sourceValues[sourceScan * detectorPixels + sourceDetector]
                    )
                  }
                }
              }
            }
          }
        }
      }
    }
    XCTAssertEqual(bufferValues(destination, count: expected.count), expected)
  }

  func testResidentRebinU16MatchesCroppedIntegerReferenceIncludingEdges() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.residentRebinU16Function)
      )
    )
    let sourceRows = 4
    let sourceColumns = 5
    let sourceScans = sourceRows * sourceColumns
    let detectorPixels = 3
    let frameMajor: [UInt16] = (0..<(sourceScans * detectorPixels)).map {
      UInt16($0 + 1)
    }
    var wordMajor = [UInt32](repeating: 0, count: 2 * sourceScans)
    for scan in 0..<sourceScans {
      for pixel in 0..<detectorPixels {
        wordMajor[(pixel / 2) * sourceScans + scan] |=
          UInt32(frameMajor[scan * detectorPixels + pixel]) << UInt32((pixel % 2) * 16)
      }
    }
    let source = try makeBuffer(device: device, values: wordMajor)
    let outputRows = 2
    let outputColumns = 2
    let outputScans = outputRows * outputColumns
    let destination = try outputBuffer(
      device: device,
      count: outputScans * detectorPixels
    )
    var parameters = ResidentRebinParameters(
      sourceRows: UInt32(sourceRows),
      sourceCols: UInt32(sourceColumns),
      sourceScanCount: UInt32(sourceScans),
      sourceDetectorRows: 1,
      sourceDetectorCols: UInt32(detectorPixels),
      sourceRowOffset: 1,
      sourceColOffset: 1,
      selectedRows: 3,
      selectedCols: 4,
      scanBin: 2,
      detectorBin: 1,
      outputRows: UInt32(outputRows),
      outputCols: UInt32(outputColumns),
      outputScanCount: UInt32(outputScans),
      outputDetectorRows: 1,
      outputDetectorCols: UInt32(detectorPixels)
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    withUnsafeBytes(of: &parameters) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 2)
    }
    encoder.dispatchThreads(
      MTLSize(width: detectorPixels, height: outputScans, depth: 1),
      threadsPerThreadgroup: MTLSize(width: detectorPixels, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    var expected = [UInt32](repeating: 0, count: outputScans * detectorPixels)
    for outputRow in 0..<outputRows {
      for outputColumn in 0..<outputColumns {
        let outputScan = outputRow * outputColumns + outputColumn
        for pixel in 0..<detectorPixels {
          for row in (1 + outputRow * 2)..<min(4, 1 + (outputRow + 1) * 2) {
            for column in (1 + outputColumn * 2)..<min(5, 1 + (outputColumn + 1) * 2) {
              expected[pixel * outputScans + outputScan] += UInt32(
                frameMajor[(row * sourceColumns + column) * detectorPixels + pixel]
              )
            }
          }
        }
      }
    }
    XCTAssertEqual(bufferValues(destination, count: expected.count), expected)
  }

  func testResidentRebinU32MatchesCoarserScanAndDetectorReference() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.residentRebinU32Function)
      )
    )
    let sourceRows = 3
    let sourceColumns = 3
    let sourceScans = sourceRows * sourceColumns
    let detectorPixels = 2
    let wordMajor: [UInt32] = (0..<detectorPixels).flatMap { pixel in
      (0..<sourceScans).map { scan in UInt32(100 * pixel + scan + 1) }
    }
    let source = try makeBuffer(device: device, values: wordMajor)
    let destination = try outputBuffer(device: device, count: 4)
    var parameters = ResidentRebinParameters(
      sourceRows: 3,
      sourceCols: 3,
      sourceScanCount: 9,
      sourceDetectorRows: 1,
      sourceDetectorCols: 2,
      sourceRowOffset: 0,
      sourceColOffset: 0,
      selectedRows: 3,
      selectedCols: 3,
      scanBin: 2,
      detectorBin: 2,
      outputRows: 2,
      outputCols: 2,
      outputScanCount: 4,
      outputDetectorRows: 1,
      outputDetectorCols: 1
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    withUnsafeBytes(of: &parameters) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 2)
    }
    encoder.dispatchThreads(
      MTLSize(width: 1, height: 4, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    XCTAssertEqual(
      bufferValues(destination, count: 4),
      [424, 218, 230, 118]
    )
  }

  func testWordMajorU32DetectorProductsAndInteractiveUpdatesMatchReference() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let scanCount = 4
    let detectorPixels = 7
    let values: [UInt32] = (0..<detectorPixels).flatMap { pixel in
      (0..<scanCount).map { scan in UInt32(100 * pixel + scan + 1) }
    }
    let bands: [UInt8] = (0..<detectorPixels).map { pixel in
      (pixel % 2 == 0 ? 1 : 0) | (pixel % 3 == 0 ? 2 : 0) | (pixel >= 4 ? 4 : 0)
    }
    let source = try makeBuffer(device: device, values: values)
    let bandBuffer = try makeBuffer(device: device, values: bands)
    let bf = try outputBuffer(device: device, count: scanCount)
    let abf = try outputBuffer(device: device, count: scanCount)
    let df = try outputBuffer(device: device, count: scanCount)
    var productParameters = WordMajorDetectorParameters(
      scanCount: UInt32(scanCount),
      detectorPixels: UInt32(detectorPixels)
    )
    let productPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.detectorProductsU32Function)
      )
    )
    let productCommand = try XCTUnwrap(queue.makeCommandBuffer())
    let productEncoder = try XCTUnwrap(productCommand.makeComputeCommandEncoder())
    productEncoder.setComputePipelineState(productPipeline)
    productEncoder.setBuffer(source, offset: 0, index: 0)
    productEncoder.setBuffer(bf, offset: 0, index: 1)
    productEncoder.setBuffer(abf, offset: 0, index: 2)
    productEncoder.setBuffer(df, offset: 0, index: 3)
    withUnsafeBytes(of: &productParameters) {
      productEncoder.setBytes($0.baseAddress!, length: $0.count, index: 4)
    }
    productEncoder.setBuffer(bandBuffer, offset: 0, index: 5)
    productEncoder.dispatchThreadgroups(
      MTLSize(width: scanCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    productEncoder.endEncoding()
    try complete(productCommand)

    func reference(_ band: UInt8) -> [UInt32] {
      (0..<scanCount).map { scan in
        (0..<detectorPixels).reduce(UInt32(0)) { sum, pixel in
          sum + (bands[pixel] & band == 0 ? 0 : values[pixel * scanCount + scan])
        }
      }
    }
    XCTAssertEqual(bufferValues(bf, count: scanCount), reference(1))
    XCTAssertEqual(bufferValues(abf, count: scanCount), reference(2))
    XCTAssertEqual(bufferValues(df, count: scanCount), reference(4))

    var scanCountU32 = UInt32(scanCount)
    var entryCount = UInt32(2)
    let initialEntries = try makeBuffer(
      device: device,
      values: [SIMD2<UInt32>(1, 1), SIMD2<UInt32>(3, 1)]
    )
    let output = try outputBuffer(device: device, count: scanCount)
    let fullPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(library.makeFunction(name: Metal4DSTEMKernels.fullSumU32Function))
    )
    let fullCommand = try XCTUnwrap(queue.makeCommandBuffer())
    let fullEncoder = try XCTUnwrap(fullCommand.makeComputeCommandEncoder())
    fullEncoder.setComputePipelineState(fullPipeline)
    fullEncoder.setBuffer(source, offset: 0, index: 0)
    fullEncoder.setBuffer(initialEntries, offset: 0, index: 1)
    fullEncoder.setBuffer(output, offset: 0, index: 2)
    fullEncoder.setBytes(&scanCountU32, length: 4, index: 3)
    fullEncoder.setBytes(&entryCount, length: 4, index: 4)
    fullEncoder.dispatchThreads(
      MTLSize(width: scanCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: scanCount, height: 1, depth: 1)
    )
    fullEncoder.endEncoding()
    try complete(fullCommand)
    let initialExpected = (0..<scanCount).map {
      values[scanCount + $0] + values[3 * scanCount + $0]
    }
    XCTAssertEqual(bufferValues(output, count: scanCount), initialExpected)

    let deltaEntries = try makeBuffer(
      device: device,
      values: [SIMD2<UInt32>(1, 2), SIMD2<UInt32>(2, 1)]
    )
    let deltaPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.signedDeltaU32Function)
      )
    )
    let deltaCommand = try XCTUnwrap(queue.makeCommandBuffer())
    let deltaEncoder = try XCTUnwrap(deltaCommand.makeComputeCommandEncoder())
    deltaEncoder.setComputePipelineState(deltaPipeline)
    deltaEncoder.setBuffer(source, offset: 0, index: 0)
    deltaEncoder.setBuffer(deltaEntries, offset: 0, index: 1)
    deltaEncoder.setBuffer(output, offset: 0, index: 2)
    deltaEncoder.setBytes(&scanCountU32, length: 4, index: 3)
    deltaEncoder.setBytes(&entryCount, length: 4, index: 4)
    deltaEncoder.dispatchThreads(
      MTLSize(width: scanCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: scanCount, height: 1, depth: 1)
    )
    deltaEncoder.endEncoding()
    try complete(deltaCommand)
    let deltaExpected = (0..<scanCount).map {
      values[2 * scanCount + $0] + values[3 * scanCount + $0]
    }
    XCTAssertEqual(bufferValues(output, count: scanCount), deltaExpected)

    let extracted = try outputBuffer(device: device, count: detectorPixels)
    let extractPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(library.makeFunction(name: Metal4DSTEMKernels.extractU32Function))
    )
    var selectedScan = UInt32(2)
    var pixelCount = UInt32(detectorPixels)
    let extractCommand = try XCTUnwrap(queue.makeCommandBuffer())
    let extractEncoder = try XCTUnwrap(extractCommand.makeComputeCommandEncoder())
    extractEncoder.setComputePipelineState(extractPipeline)
    extractEncoder.setBuffer(source, offset: 0, index: 0)
    extractEncoder.setBuffer(extracted, offset: 0, index: 1)
    extractEncoder.setBytes(&selectedScan, length: 4, index: 2)
    extractEncoder.setBytes(&scanCountU32, length: 4, index: 3)
    extractEncoder.setBytes(&pixelCount, length: 4, index: 4)
    extractEncoder.dispatchThreads(
      MTLSize(width: detectorPixels, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: detectorPixels, height: 1, depth: 1)
    )
    extractEncoder.endEncoding()
    try complete(extractCommand)
    XCTAssertEqual(
      bufferValues(extracted, count: detectorPixels),
      (0..<detectorPixels).map { values[$0 * scanCount + 2] }
    )
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

  private func floatBufferValues(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
  }

  private func complete(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { throw error }
    XCTAssertEqual(command.status, .completed)
  }

  private func encodeBluesteinFFT(
    _ encoder: MTLComputeCommandEncoder,
    library: MTLLibrary,
    device: MTLDevice,
    source: MTLBuffer,
    destination: MTLBuffer,
    rows: Int,
    columns: Int,
    paddedRows: Int,
    paddedColumns: Int,
    signal: MTLBuffer,
    chirp: MTLBuffer,
    inverse: Bool
  ) throws {
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      try device.makeComputePipelineState(function: XCTUnwrap(library.makeFunction(name: name)))
    }
    var parameters = Bluestein2DParameters(
      sourceWidth: UInt32(columns),
      sourceHeight: UInt32(rows),
      paddedWidth: UInt32(paddedColumns),
      paddedHeight: UInt32(paddedRows),
      direction: inverse ? 1 : -1,
      scale: inverse ? 1 / Float(rows * columns) : 1
    )
    encoder.setComputePipelineState(try pipeline(Metal4DSTEMKernels.bluesteinPrepareFunction))
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(signal, offset: 0, index: 1)
    encoder.setBuffer(chirp, offset: 0, index: 2)
    encoder.setBytes(&parameters, length: MemoryLayout<Bluestein2DParameters>.stride, index: 3)
    encoder.dispatchThreads(
      MTLSize(width: paddedColumns, height: paddedRows, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
    )
    encoder.memoryBarrier(scope: .buffers)
    try encodeFFT(
      encoder,
      library: library,
      device: device,
      buffer: signal,
      rows: paddedRows,
      columns: paddedColumns,
      inverse: false
    )
    try encodeFFT(
      encoder,
      library: library,
      device: device,
      buffer: chirp,
      rows: paddedRows,
      columns: paddedColumns,
      inverse: false
    )
    var paddedCount = UInt32(paddedRows * paddedColumns)
    encoder.setComputePipelineState(try pipeline(Metal4DSTEMKernels.complexMultiplyFunction))
    encoder.setBuffer(signal, offset: 0, index: 0)
    encoder.setBuffer(chirp, offset: 0, index: 1)
    encoder.setBytes(&paddedCount, length: 4, index: 2)
    encoder.dispatchThreads(
      MTLSize(width: Int(paddedCount), height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.memoryBarrier(scope: .buffers)
    try encodeFFT(
      encoder,
      library: library,
      device: device,
      buffer: signal,
      rows: paddedRows,
      columns: paddedColumns,
      inverse: true
    )
    encoder.setComputePipelineState(try pipeline(Metal4DSTEMKernels.bluesteinExtractFunction))
    encoder.setBuffer(signal, offset: 0, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    encoder.setBytes(&parameters, length: MemoryLayout<Bluestein2DParameters>.stride, index: 2)
    encoder.dispatchThreads(
      MTLSize(width: columns, height: rows, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
    )
    encoder.memoryBarrier(scope: .buffers)
  }

  private func encodeFFT(
    _ encoder: MTLComputeCommandEncoder,
    library: MTLLibrary,
    device: MTLDevice,
    buffer: MTLBuffer,
    rows: Int,
    columns: Int,
    inverse: Bool
  ) throws {
    func dispatch(
      _ function: String,
      width: Int,
      height: Int,
      log2Size: UInt32,
      stage: UInt32,
      rowAxis: Bool
    ) throws {
      let pipeline = try device.makeComputePipelineState(
        function: XCTUnwrap(library.makeFunction(name: function))
      )
      var parameters = FFT2DParameters(
        width: UInt32(columns),
        height: UInt32(rows),
        log2Size: log2Size,
        stage: stage,
        direction: inverse ? 1 : -1,
        rowAxis: rowAxis ? 1 : 0
      )
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(buffer, offset: 0, index: 0)
      encoder.setBytes(&parameters, length: MemoryLayout<FFT2DParameters>.stride, index: 1)
      encoder.dispatchThreads(
        MTLSize(width: width, height: height, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
      )
      encoder.memoryBarrier(scope: .buffers)
    }
    let widthStages = UInt32(columns.trailingZeroBitCount)
    let heightStages = UInt32(rows.trailingZeroBitCount)
    try dispatch(
      Metal4DSTEMKernels.fftBitReverseRowsFunction,
      width: columns,
      height: rows,
      log2Size: widthStages,
      stage: 0,
      rowAxis: true
    )
    for stage in 0..<widthStages {
      try dispatch(
        Metal4DSTEMKernels.fftButterflyRowsFunction,
        width: columns / 2,
        height: rows,
        log2Size: widthStages,
        stage: stage,
        rowAxis: true
      )
    }
    try dispatch(
      Metal4DSTEMKernels.fftBitReverseColumnsFunction,
      width: columns,
      height: rows,
      log2Size: heightStages,
      stage: 0,
      rowAxis: false
    )
    for stage in 0..<heightStages {
      try dispatch(
        Metal4DSTEMKernels.fftButterflyColumnsFunction,
        width: columns,
        height: rows / 2,
        log2Size: heightStages,
        stage: stage,
        rowAxis: false
      )
    }
    if inverse {
      try dispatch(
        Metal4DSTEMKernels.fftNormalizeFunction,
        width: columns,
        height: rows,
        log2Size: heightStages,
        stage: 0,
        rowAxis: false
      )
    }
  }
}
