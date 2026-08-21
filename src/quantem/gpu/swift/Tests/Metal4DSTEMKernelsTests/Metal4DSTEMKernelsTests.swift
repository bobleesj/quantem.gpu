import Foundation
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

private struct ScanRegionSumParameters {
  var scanRows: UInt32
  var scanColumns: UInt32
  var scanCount: UInt32
  var detectorPixels: UInt32
  var centerRow: Float
  var centerColumn: Float
  var radius: Float
  var shape: UInt32
  var reduction: UInt32
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

private struct RealQH5Chunk: Decodable {
  let rangeStart: UInt64
}

private struct RealQH5Metadata: Decodable {
  let sourcePath: String
  let nFrames: Int
  let nBlocksPerFrame: Int
  let chunks: [RealQH5Chunk]
}

private struct RealQH5Block {
  let globalFrame: Int
  let block: Int
  let compressed: Data
}

private func littleEndianUInt32(_ data: Data, offset: Int) -> UInt32 {
  data.withUnsafeBytes { raw in
    raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian
  }
}

private func realQH5Blocks(
  indexDirectory: URL,
  globalFrames: [Int]
) throws -> [RealQH5Block] {
  let indexURLs = try FileManager.default.contentsOfDirectory(
    at: indexDirectory,
    includingPropertiesForKeys: nil
  ).filter { $0.pathExtension == "qh5idx" }.sorted {
    $0.lastPathComponent < $1.lastPathComponent
  }
  var globalStart = 0
  var result: [RealQH5Block] = []
  for indexURL in indexURLs {
    let index = try Data(contentsOf: indexURL)
    guard String(decoding: index.prefix(8), as: UTF8.self) == "QH5IDX01" else {
      XCTFail("Unsupported QH5 index at \(indexURL.path)")
      continue
    }
    let jsonBytes = Int(littleEndianUInt32(index, offset: 8))
    let wordCount = Int(littleEndianUInt32(index, offset: 12))
    let metadata = try JSONDecoder().decode(
      RealQH5Metadata.self,
      from: index.subdata(in: 16..<(16 + jsonBytes))
    )
    let selectedFrames = globalFrames.filter {
      globalStart <= $0 && $0 < globalStart + metadata.nFrames
    }
    if !selectedFrames.isEmpty {
      let wordsOffset = 16 + (jsonBytes + 3) / 4 * 4
      let source = try FileHandle(forReadingFrom: URL(fileURLWithPath: metadata.sourcePath))
      defer { try? source.close() }
      for globalFrame in selectedFrames {
        let localFrame = globalFrame - globalStart
        for block in 0..<metadata.nBlocksPerFrame {
          let pair = localFrame * metadata.nBlocksPerFrame + block
          let offset = littleEndianUInt32(index, offset: wordsOffset + pair * 8)
          let length = littleEndianUInt32(index, offset: wordsOffset + pair * 8 + 4)
          XCTAssertLessThan(pair * 2 + 1, wordCount)
          try source.seek(toOffset: metadata.chunks[0].rangeStart + UInt64(offset))
          let compressed = try XCTUnwrap(source.read(upToCount: Int(length)))
          XCTAssertEqual(compressed.count, Int(length))
          result.append(
            RealQH5Block(
              globalFrame: globalFrame,
              block: block,
              compressed: compressed
            )
          )
        }
      }
    }
    globalStart += metadata.nFrames
  }
  return result
}

private func lz4Prefix(_ compressed: Data, count: Int) -> [UInt8] {
  let source = [UInt8](compressed)
  var output: [UInt8] = []
  output.reserveCapacity(count)
  var inputIndex = 0
  while inputIndex < source.count && output.count < count {
    let token = source[inputIndex]
    inputIndex += 1
    var literalCount = Int(token >> 4)
    if literalCount == 15 {
      var next = 255
      while next == 255 {
        next = Int(source[inputIndex])
        inputIndex += 1
        literalCount += next
      }
    }
    let copiedLiterals = min(literalCount, count - output.count)
    output.append(contentsOf: source[inputIndex..<(inputIndex + copiedLiterals)])
    inputIndex += literalCount
    if inputIndex >= source.count || output.count >= count { break }
    let matchOffset = Int(source[inputIndex]) | Int(source[inputIndex + 1]) << 8
    inputIndex += 2
    var matchCount = 4 + Int(token & 15)
    if token & 15 == 15 {
      var next = 255
      while next == 255 {
        next = Int(source[inputIndex])
        inputIndex += 1
        matchCount += next
      }
    }
    for _ in 0..<min(matchCount, count - output.count) {
      output.append(output[output.count - matchOffset])
    }
  }
  return output
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

private struct QH5DirectDetectorBinParameters {
  var sourceDetectorRows: UInt32
  var sourceDetectorColumns: UInt32
  var outputDetectorColumns: UInt32
  var outputScanCount: UInt32
}

final class Metal4DSTEMKernelsTests: XCTestCase {
  func testRealQH5PrefixStopsAtExactLowPlaneBoundary() throws {
    guard
      let indexDirectory = ProcessInfo.processInfo.environment[
        "QUANTEM_GPU_QH5_REAL_INDEX_DIR"
      ]
    else {
      throw XCTSkip("Set QUANTEM_GPU_QH5_REAL_INDEX_DIR to run real QH5 decode parity")
    }
    let records = try realQH5Blocks(
      indexDirectory: URL(fileURLWithPath: indexDirectory),
      globalFrames: [117_229, 152_528, 155_484, 217_787]
    )
    XCTAssertEqual(records.count, 36)
    for record in records {
      let completeBlock = lz4Prefix(record.compressed, count: 8_192)
      XCTAssertEqual(completeBlock.count, 8_192)
      XCTAssertEqual(
        lz4Prefix(record.compressed, count: 4_096),
        Array(completeBlock.prefix(4_096)),
        "QH5 prefix mismatch at scan frame \(record.globalFrame), block \(record.block)"
      )
    }
  }

  func testRealQH5ScalarLowPlaneDecodeMatchesCPUReference() throws {
    guard
      let indexDirectory = ProcessInfo.processInfo.environment[
        "QUANTEM_GPU_QH5_REAL_INDEX_DIR"
      ]
    else {
      throw XCTSkip("Set QUANTEM_GPU_QH5_REAL_INDEX_DIR to run real QH5 decode parity")
    }
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeHDF5Library(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.decodeU16AuditedLow8ScalarFunction)
      )
    )
    let records = try realQH5Blocks(
      indexDirectory: URL(fileURLWithPath: indexDirectory),
      globalFrames: [117_229, 152_528, 155_484, 217_787]
    )
    XCTAssertEqual(records.count, 36)
    for record in records {
      let expected = lz4Prefix(record.compressed, count: 4_096)
      for repeatIndex in 0..<8 {
        let compressed = try makeBuffer(device: device, values: Array(record.compressed))
        let metadata = try makeBuffer(
          device: device,
          values: [SIMD2<UInt32>(0, UInt32(record.compressed.count))]
        )
        let output = try XCTUnwrap(
          device.makeBuffer(length: expected.count, options: .storageModeShared)
        )
        memset(output.contents(), 0, expected.count)
        var rangeStart: UInt64 = 0
        var blocksPerFrame: UInt32 = 1
        var frameElements: UInt32 = 4_096
        var metadataFrameOffset: UInt32 = 0
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(compressed, offset: 0, index: 0)
        encoder.setBuffer(metadata, offset: 0, index: 1)
        encoder.setBytes(&rangeStart, length: 8, index: 2)
        encoder.setBytes(&blocksPerFrame, length: 4, index: 3)
        encoder.setBytes(&frameElements, length: 4, index: 4)
        encoder.setBuffer(output, offset: 0, index: 5)
        encoder.setBytes(&metadataFrameOffset, length: 4, index: 6)
        encoder.dispatchThreads(
          MTLSize(width: 1, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        encoder.endEncoding()
        try complete(command)
        let actual = Array(
          UnsafeBufferPointer(
            start: output.contents().bindMemory(to: UInt8.self, capacity: expected.count),
            count: expected.count
          )
        )
        XCTAssertEqual(
          actual,
          expected,
          "QH5 low-plane mismatch at scan frame \(record.globalFrame), "
            + "block \(record.block), repeat \(repeatIndex)"
        )
      }
    }
  }

  func testRealQH5ScalarDetectorBin4MatchesCPUReference() throws {
    guard
      let indexDirectory = ProcessInfo.processInfo.environment[
        "QUANTEM_GPU_QH5_REAL_INDEX_DIR"
      ]
    else {
      throw XCTSkip("Set QUANTEM_GPU_QH5_REAL_INDEX_DIR to run real QH5 decode parity")
    }
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeHDF5Library(device: device)
    let decodePipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.decodeU16AuditedLow8ScalarFunction)
      )
    )
    let binPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels.binU16AuditedLow8ScalarU16WordMajorFunction
        )
      )
    )
    let frameOwnedBinPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels
            .binU16AuditedLow8ScalarU16WordMajorFrameOwnedFunction
        )
      )
    )
    let frameOwnedRow8BinPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels
            .binU16AuditedLow8ScalarU16WordMajorFrameOwnedRow8Function
        )
      )
    )
    let frameMajorRow8BinPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels
            .binU16AuditedLow8ScalarU16FrameMajorRow8Function
        )
      )
    )
    let records = try realQH5Blocks(
      indexDirectory: URL(fileURLWithPath: indexDirectory),
      globalFrames: [117_229, 152_528, 155_484, 217_787]
    )
    XCTAssertEqual(records.count, 36)
    let grouped = Dictionary(grouping: records, by: \.globalFrame)
    for frame in grouped.keys.sorted() {
      let frameRecords = try XCTUnwrap(grouped[frame]?.sorted { $0.block < $1.block })
      XCTAssertEqual(frameRecords.map(\.block), Array(0..<9))
      var compressed: [UInt8] = []
      var metadata: [SIMD2<UInt32>] = []
      var decodedBlocks: [[UInt8]] = []
      for record in frameRecords {
        metadata.append(SIMD2(UInt32(compressed.count), UInt32(record.compressed.count)))
        compressed.append(contentsOf: record.compressed)
        decodedBlocks.append(lz4Prefix(record.compressed, count: 4_096))
      }
      let h5 = try makeBuffer(device: device, values: compressed)
      let blockMetadata = try makeBuffer(device: device, values: metadata)
      let scratch = try XCTUnwrap(
        device.makeBuffer(length: 192 * 192, options: .storageModeShared)
      )
      let output = try outputBuffer(device: device, count: 48 * 48 / 2)
      let badMask = try makeBuffer(device: device, values: [UInt8](repeating: 0, count: 192 * 192))
      let countAudit = try outputBuffer(device: device, count: 2)
      let bandValues = (0..<(48 * 48)).map { outputPixel -> UInt8 in
        let row = outputPixel / 48
        let column = outputPixel % 48
        return (row < 24 ? 1 : 0) | (column < 24 ? 2 : 0) | (row >= 24 ? 4 : 0)
      }
      let detectorBands = try makeBuffer(device: device, values: bandValues)
      let bf = try outputBuffer(device: device, count: 1)
      let abf = try outputBuffer(device: device, count: 1)
      let df = try outputBuffer(device: device, count: 1)
      let total = try outputBuffer(device: device, count: 1)
      let rowMoment = try outputBuffer(device: device, count: 1)
      let columnMoment = try outputBuffer(device: device, count: 1)
      var rangeStart: UInt64 = 0
      var blocksPerFrame: UInt32 = 9
      var frameElements: UInt32 = 192 * 192
      var metadataFrameOffset: UInt32 = 0
      var globalFrameOffset: UInt32 = 0
      var parameters = QH5DirectDetectorBinParameters(
        sourceDetectorRows: 192,
        sourceDetectorColumns: 192,
        outputDetectorColumns: 48,
        outputScanCount: 1
      )
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
      encoder.setComputePipelineState(decodePipeline)
      encoder.setBuffer(h5, offset: 0, index: 0)
      encoder.setBuffer(blockMetadata, offset: 0, index: 1)
      encoder.setBytes(&rangeStart, length: 8, index: 2)
      encoder.setBytes(&blocksPerFrame, length: 4, index: 3)
      encoder.setBytes(&frameElements, length: 4, index: 4)
      encoder.setBuffer(scratch, offset: 0, index: 5)
      encoder.setBytes(&metadataFrameOffset, length: 4, index: 6)
      encoder.dispatchThreads(
        MTLSize(width: 9, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
      )
      encoder.memoryBarrier(scope: .buffers)
      encoder.setComputePipelineState(binPipeline)
      encoder.setBuffer(scratch, offset: 0, index: 0)
      encoder.setBuffer(output, offset: 0, index: 1)
      encoder.setBuffer(badMask, offset: 0, index: 2)
      encoder.setBuffer(countAudit, offset: 0, index: 3)
      encoder.setBytes(&globalFrameOffset, length: 4, index: 4)
      encoder.setBytes(&blocksPerFrame, length: 4, index: 5)
      encoder.setBytes(&frameElements, length: 4, index: 6)
      withUnsafeBytes(of: &parameters) {
        encoder.setBytes($0.baseAddress!, length: $0.count, index: 7)
      }
      encoder.setBuffer(detectorBands, offset: 0, index: 8)
      encoder.setBuffer(bf, offset: 0, index: 9)
      encoder.setBuffer(abf, offset: 0, index: 10)
      encoder.setBuffer(df, offset: 0, index: 11)
      encoder.setBuffer(total, offset: 0, index: 12)
      encoder.setBuffer(rowMoment, offset: 0, index: 13)
      encoder.setBuffer(columnMoment, offset: 0, index: 14)
      encoder.dispatchThreadgroups(
        MTLSize(width: 9, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
      )
      encoder.endEncoding()
      try complete(command)

      func sourceValue(row: Int, column: Int) -> UInt32 {
        let pixel = row * 192 + column
        let block = pixel / 4_096
        let local = pixel % 4_096
        let group = local / 32
        let lane = local % 32
        var value: UInt32 = 0
        for bit in 0..<8 {
          let wordOffset = bit * 512 + group * 4
          let bytes = decodedBlocks[block]
          let word =
            UInt32(bytes[wordOffset])
            | UInt32(bytes[wordOffset + 1]) << 8
            | UInt32(bytes[wordOffset + 2]) << 16
            | UInt32(bytes[wordOffset + 3]) << 24
          if word & (1 << lane) != 0 { value |= 1 << bit }
        }
        return value
      }
      var expectedBins = [UInt32](repeating: 0, count: 48 * 48)
      var expectedBF: UInt32 = 0
      var expectedABF: UInt32 = 0
      var expectedDF: UInt32 = 0
      var expectedTotal: UInt32 = 0
      var expectedRowMoment: UInt32 = 0
      var expectedColumnMoment: UInt32 = 0
      for outputRow in 0..<48 {
        for outputColumn in 0..<48 {
          var sum: UInt32 = 0
          for rowOffset in 0..<4 {
            for columnOffset in 0..<4 {
              sum += sourceValue(
                row: outputRow * 4 + rowOffset,
                column: outputColumn * 4 + columnOffset
              )
            }
          }
          expectedBins[outputRow * 48 + outputColumn] = sum
          let bands = bandValues[outputRow * 48 + outputColumn]
          if bands & 1 != 0 { expectedBF += sum }
          if bands & 2 != 0 { expectedABF += sum }
          if bands & 4 != 0 { expectedDF += sum }
          expectedTotal += sum
          expectedRowMoment += sum * UInt32(outputRow)
          expectedColumnMoment += sum * UInt32(outputColumn)
        }
      }
      let actualWords = bufferValues(output, count: 48 * 48 / 2)
      let actualBins = actualWords.flatMap { [$0 & 0xffff, $0 >> 16] }
      XCTAssertEqual(actualBins, expectedBins, "detector-bin mismatch at scan frame \(frame)")
      XCTAssertEqual(bufferValues(total, count: 1), [expectedTotal])
      XCTAssertEqual(bufferValues(bf, count: 1), [expectedBF])
      XCTAssertEqual(bufferValues(abf, count: 1), [expectedABF])
      XCTAssertEqual(bufferValues(df, count: 1), [expectedDF])
      XCTAssertEqual(bufferValues(rowMoment, count: 1), [expectedRowMoment])
      XCTAssertEqual(bufferValues(columnMoment, count: 1), [expectedColumnMoment])

      let frameMajorRow8Output = try outputBuffer(
        device: device,
        count: 48 * 48 / 2
      )
      let frameMajorRow8Audit = try outputBuffer(device: device, count: 1)
      let frameMajorRow8Command = try XCTUnwrap(queue.makeCommandBuffer())
      let frameMajorRow8Encoder = try XCTUnwrap(
        frameMajorRow8Command.makeComputeCommandEncoder()
      )
      frameMajorRow8Encoder.setComputePipelineState(frameMajorRow8BinPipeline)
      frameMajorRow8Encoder.setBuffer(scratch, offset: 0, index: 0)
      frameMajorRow8Encoder.setBuffer(frameMajorRow8Output, offset: 0, index: 1)
      frameMajorRow8Encoder.setBuffer(badMask, offset: 0, index: 2)
      frameMajorRow8Encoder.setBuffer(frameMajorRow8Audit, offset: 0, index: 3)
      frameMajorRow8Encoder.setBytes(&globalFrameOffset, length: 4, index: 4)
      frameMajorRow8Encoder.setBytes(&frameElements, length: 4, index: 5)
      withUnsafeBytes(of: &parameters) {
        frameMajorRow8Encoder.setBytes($0.baseAddress!, length: $0.count, index: 6)
      }
      frameMajorRow8Encoder.dispatchThreadgroups(
        MTLSize(width: 9, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
      )
      frameMajorRow8Encoder.endEncoding()
      try complete(frameMajorRow8Command)

      XCTAssertEqual(
        bufferValues(frameMajorRow8Output, count: 48 * 48 / 2),
        actualWords,
        "frame-major row8 detector-bin mismatch at scan frame \(frame)"
      )
      XCTAssertEqual(
        bufferValues(frameMajorRow8Audit, count: 1),
        [bufferValues(countAudit, count: 2)[0]]
      )

      let frameOwnedOutput = try outputBuffer(device: device, count: 48 * 48 / 2)
      let frameOwnedAudit = try outputBuffer(device: device, count: 2)
      let frameOwnedBF = try outputBuffer(device: device, count: 1)
      let frameOwnedABF = try outputBuffer(device: device, count: 1)
      let frameOwnedDF = try outputBuffer(device: device, count: 1)
      let frameOwnedTotal = try outputBuffer(device: device, count: 1)
      let frameOwnedRowMoment = try outputBuffer(device: device, count: 1)
      let frameOwnedColumnMoment = try outputBuffer(device: device, count: 1)
      let frameOwnedCommand = try XCTUnwrap(queue.makeCommandBuffer())
      let frameOwnedEncoder = try XCTUnwrap(
        frameOwnedCommand.makeComputeCommandEncoder()
      )
      frameOwnedEncoder.setComputePipelineState(frameOwnedBinPipeline)
      frameOwnedEncoder.setBuffer(scratch, offset: 0, index: 0)
      frameOwnedEncoder.setBuffer(frameOwnedOutput, offset: 0, index: 1)
      frameOwnedEncoder.setBuffer(badMask, offset: 0, index: 2)
      frameOwnedEncoder.setBuffer(frameOwnedAudit, offset: 0, index: 3)
      frameOwnedEncoder.setBytes(&globalFrameOffset, length: 4, index: 4)
      frameOwnedEncoder.setBytes(&blocksPerFrame, length: 4, index: 5)
      frameOwnedEncoder.setBytes(&frameElements, length: 4, index: 6)
      withUnsafeBytes(of: &parameters) {
        frameOwnedEncoder.setBytes($0.baseAddress!, length: $0.count, index: 7)
      }
      frameOwnedEncoder.setBuffer(detectorBands, offset: 0, index: 8)
      frameOwnedEncoder.setBuffer(frameOwnedBF, offset: 0, index: 9)
      frameOwnedEncoder.setBuffer(frameOwnedABF, offset: 0, index: 10)
      frameOwnedEncoder.setBuffer(frameOwnedDF, offset: 0, index: 11)
      frameOwnedEncoder.setBuffer(frameOwnedTotal, offset: 0, index: 12)
      frameOwnedEncoder.setBuffer(frameOwnedRowMoment, offset: 0, index: 13)
      frameOwnedEncoder.setBuffer(frameOwnedColumnMoment, offset: 0, index: 14)
      frameOwnedEncoder.dispatchThreadgroups(
        MTLSize(width: 9, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
      )
      frameOwnedEncoder.endEncoding()
      try complete(frameOwnedCommand)

      XCTAssertEqual(
        bufferValues(frameOwnedOutput, count: 48 * 48 / 2),
        actualWords,
        "frame-owned detector-bin mismatch at scan frame \(frame)"
      )
      XCTAssertEqual(
        bufferValues(frameOwnedAudit, count: 2),
        bufferValues(countAudit, count: 2)
      )
      XCTAssertEqual(bufferValues(frameOwnedBF, count: 1), bufferValues(bf, count: 1))
      XCTAssertEqual(bufferValues(frameOwnedABF, count: 1), bufferValues(abf, count: 1))
      XCTAssertEqual(bufferValues(frameOwnedDF, count: 1), bufferValues(df, count: 1))
      XCTAssertEqual(bufferValues(frameOwnedTotal, count: 1), [expectedTotal])
      XCTAssertEqual(bufferValues(frameOwnedRowMoment, count: 1), [expectedRowMoment])
      XCTAssertEqual(
        bufferValues(frameOwnedColumnMoment, count: 1),
        [expectedColumnMoment]
      )

      let frameOwnedRow8Output = try outputBuffer(device: device, count: 48 * 48 / 2)
      let frameOwnedRow8Audit = try outputBuffer(device: device, count: 2)
      let frameOwnedRow8BF = try outputBuffer(device: device, count: 1)
      let frameOwnedRow8ABF = try outputBuffer(device: device, count: 1)
      let frameOwnedRow8DF = try outputBuffer(device: device, count: 1)
      let frameOwnedRow8Total = try outputBuffer(device: device, count: 1)
      let frameOwnedRow8RowMoment = try outputBuffer(device: device, count: 1)
      let frameOwnedRow8ColumnMoment = try outputBuffer(device: device, count: 1)
      let frameOwnedRow8Command = try XCTUnwrap(queue.makeCommandBuffer())
      let frameOwnedRow8Encoder = try XCTUnwrap(
        frameOwnedRow8Command.makeComputeCommandEncoder()
      )
      frameOwnedRow8Encoder.setComputePipelineState(frameOwnedRow8BinPipeline)
      frameOwnedRow8Encoder.setBuffer(scratch, offset: 0, index: 0)
      frameOwnedRow8Encoder.setBuffer(frameOwnedRow8Output, offset: 0, index: 1)
      frameOwnedRow8Encoder.setBuffer(badMask, offset: 0, index: 2)
      frameOwnedRow8Encoder.setBuffer(frameOwnedRow8Audit, offset: 0, index: 3)
      frameOwnedRow8Encoder.setBytes(&globalFrameOffset, length: 4, index: 4)
      frameOwnedRow8Encoder.setBytes(&blocksPerFrame, length: 4, index: 5)
      frameOwnedRow8Encoder.setBytes(&frameElements, length: 4, index: 6)
      withUnsafeBytes(of: &parameters) {
        frameOwnedRow8Encoder.setBytes($0.baseAddress!, length: $0.count, index: 7)
      }
      frameOwnedRow8Encoder.setBuffer(detectorBands, offset: 0, index: 8)
      frameOwnedRow8Encoder.setBuffer(frameOwnedRow8BF, offset: 0, index: 9)
      frameOwnedRow8Encoder.setBuffer(frameOwnedRow8ABF, offset: 0, index: 10)
      frameOwnedRow8Encoder.setBuffer(frameOwnedRow8DF, offset: 0, index: 11)
      frameOwnedRow8Encoder.setBuffer(frameOwnedRow8Total, offset: 0, index: 12)
      frameOwnedRow8Encoder.setBuffer(frameOwnedRow8RowMoment, offset: 0, index: 13)
      frameOwnedRow8Encoder.setBuffer(frameOwnedRow8ColumnMoment, offset: 0, index: 14)
      frameOwnedRow8Encoder.dispatchThreadgroups(
        MTLSize(width: 9, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
      )
      frameOwnedRow8Encoder.endEncoding()
      try complete(frameOwnedRow8Command)

      XCTAssertEqual(
        bufferValues(frameOwnedRow8Output, count: 48 * 48 / 2),
        actualWords,
        "frame-owned row8 detector-bin mismatch at scan frame \(frame)"
      )
      XCTAssertEqual(
        bufferValues(frameOwnedRow8Audit, count: 2),
        bufferValues(countAudit, count: 2)
      )
      XCTAssertEqual(bufferValues(frameOwnedRow8BF, count: 1), [expectedBF])
      XCTAssertEqual(bufferValues(frameOwnedRow8ABF, count: 1), [expectedABF])
      XCTAssertEqual(bufferValues(frameOwnedRow8DF, count: 1), [expectedDF])
      XCTAssertEqual(bufferValues(frameOwnedRow8Total, count: 1), [expectedTotal])
      XCTAssertEqual(
        bufferValues(frameOwnedRow8RowMoment, count: 1),
        [expectedRowMoment]
      )
      XCTAssertEqual(
        bufferValues(frameOwnedRow8ColumnMoment, count: 1),
        [expectedColumnMoment]
      )
    }
  }

  func testFrameOwnedRow8FallsBackForUnalignedDetectorGeometry() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeHDF5Library(device: device)
    let row4Pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels
            .binU16AuditedLow8ScalarU16WordMajorFrameOwnedFunction
        )
      )
    )
    let row8Pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels
            .binU16AuditedLow8ScalarU16WordMajorFrameOwnedRow8Function
        )
      )
    )
    let sourceRows = 20
    let sourceColumns = 140
    let frameElements = 4_096
    let outputRows = sourceRows / 4
    let outputColumns = sourceColumns / 4
    let outputPixels = outputRows * outputColumns
    let outputWords = (outputPixels + 1) / 2
    let values = (0..<(sourceRows * sourceColumns)).map {
      UInt8(($0 * 7 + $0 / sourceColumns * 3) % 16)
    }
    let badPixels = [0, 139, 140, sourceRows * sourceColumns - 1]
    let badSet = Set(badPixels)
    var lowPlaneWords = [UInt32](repeating: 0, count: frameElements / 4)
    for (pixel, value) in values.enumerated() {
      for bit in 0..<8 where value & (1 << bit) != 0 {
        lowPlaneWords[bit * 128 + pixel / 32] |= 1 << (pixel % 32)
      }
    }
    let scratch = try makeBuffer(device: device, values: lowPlaneWords)
    var badMaskValues = [UInt8](repeating: 0, count: frameElements)
    for pixel in badPixels { badMaskValues[pixel] = 1 }
    let badMask = try makeBuffer(device: device, values: badMaskValues)
    let bandValues = (0..<outputPixels).map { pixel -> UInt8 in
      let row = pixel / outputColumns
      let column = pixel % outputColumns
      return (row < 2 ? 1 : 0) | (column < 17 ? 2 : 0) | (row >= 2 ? 4 : 0)
    }
    let detectorBands = try makeBuffer(device: device, values: bandValues)
    var globalFrameOffset: UInt32 = 0
    var blocksPerFrame: UInt32 = 1
    var frameElementCount = UInt32(frameElements)
    var parameters = QH5DirectDetectorBinParameters(
      sourceDetectorRows: UInt32(sourceRows),
      sourceDetectorColumns: UInt32(sourceColumns),
      outputDetectorColumns: UInt32(outputColumns),
      outputScanCount: 1
    )

    func dispatch(_ pipeline: MTLComputePipelineState) throws -> [[UInt32]] {
      let output = try outputBuffer(device: device, count: outputWords)
      let audit = try outputBuffer(device: device, count: 2)
      let products = try (0..<6).map { _ in
        try outputBuffer(device: device, count: 1)
      }
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(scratch, offset: 0, index: 0)
      encoder.setBuffer(output, offset: 0, index: 1)
      encoder.setBuffer(badMask, offset: 0, index: 2)
      encoder.setBuffer(audit, offset: 0, index: 3)
      encoder.setBytes(&globalFrameOffset, length: 4, index: 4)
      encoder.setBytes(&blocksPerFrame, length: 4, index: 5)
      encoder.setBytes(&frameElementCount, length: 4, index: 6)
      withUnsafeBytes(of: &parameters) {
        encoder.setBytes($0.baseAddress!, length: $0.count, index: 7)
      }
      encoder.setBuffer(detectorBands, offset: 0, index: 8)
      for (offset, product) in products.enumerated() {
        encoder.setBuffer(product, offset: 0, index: 9 + offset)
      }
      encoder.dispatchThreadgroups(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
      )
      encoder.endEncoding()
      try complete(command)
      return [
        bufferValues(output, count: outputWords),
        bufferValues(audit, count: 2),
        bufferValues(products[0], count: 1),
        bufferValues(products[1], count: 1),
        bufferValues(products[2], count: 1),
        bufferValues(products[3], count: 1),
        bufferValues(products[4], count: 1),
        bufferValues(products[5], count: 1),
      ]
    }

    let row4 = try dispatch(row4Pipeline)
    let row8 = try dispatch(row8Pipeline)
    XCTAssertEqual(row8, row4)
    var expected = [UInt32](repeating: 0, count: outputPixels)
    for outputRow in 0..<outputRows {
      for outputColumn in 0..<outputColumns {
        for rowOffset in 0..<4 {
          for columnOffset in 0..<4 {
            let pixel =
              (outputRow * 4 + rowOffset) * sourceColumns
              + outputColumn * 4 + columnOffset
            if !badSet.contains(pixel) {
              expected[outputRow * outputColumns + outputColumn] += UInt32(values[pixel])
            }
          }
        }
      }
    }
    let actual = row8[0].flatMap { [$0 & 0xffff, $0 >> 16] }.prefix(outputPixels)
    XCTAssertEqual(Array(actual), expected)
  }

  func testRealQH5ParallelFullBlockDecodeMatchesCPUReference() throws {
    guard
      let indexDirectory = ProcessInfo.processInfo.environment[
        "QUANTEM_GPU_QH5_REAL_INDEX_DIR"
      ]
    else {
      throw XCTSkip("Set QUANTEM_GPU_QH5_REAL_INDEX_DIR to run real QH5 decode parity")
    }
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeHDF5Library(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.decodeU16AuditedLow8Function)
      )
    )
    let records = try realQH5Blocks(
      indexDirectory: URL(fileURLWithPath: indexDirectory),
      globalFrames: [117_229, 152_528, 155_484, 217_787]
    )
    XCTAssertEqual(records.count, 36)
    let grouped = Dictionary(grouping: records, by: \.globalFrame)
    for frame in grouped.keys.sorted() {
      let frameRecords = try XCTUnwrap(grouped[frame]?.sorted { $0.block < $1.block })
      var compressed: [UInt8] = []
      var metadata: [SIMD2<UInt32>] = []
      var decodedBlocks: [[UInt8]] = []
      for record in frameRecords {
        metadata.append(SIMD2(UInt32(compressed.count), UInt32(record.compressed.count)))
        compressed.append(contentsOf: record.compressed)
        decodedBlocks.append(lz4Prefix(record.compressed, count: 8_192))
      }
      let h5 = try makeBuffer(device: device, values: compressed)
      let blockMetadata = try makeBuffer(device: device, values: metadata)
      let output = try XCTUnwrap(
        device.makeBuffer(length: 192 * 192, options: .storageModeShared)
      )
      memset(output.contents(), 0, output.length)
      let badMask = try makeBuffer(
        device: device, values: [UInt8](repeating: 0, count: 192 * 192)
      )
      let countAudit = try outputBuffer(device: device, count: 2)
      var rangeStart: UInt64 = 0
      var blocksPerFrame: UInt32 = 9
      var frameElements: UInt32 = 192 * 192
      var metadataFrameOffset: UInt32 = 0
      var globalFrameOffset: UInt32 = 0
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(h5, offset: 0, index: 0)
      encoder.setBuffer(blockMetadata, offset: 0, index: 1)
      encoder.setBytes(&rangeStart, length: 8, index: 2)
      encoder.setBytes(&blocksPerFrame, length: 4, index: 3)
      encoder.setBytes(&frameElements, length: 4, index: 4)
      encoder.setBuffer(output, offset: 0, index: 5)
      encoder.setBytes(&metadataFrameOffset, length: 4, index: 6)
      encoder.setBuffer(badMask, offset: 0, index: 7)
      encoder.setBuffer(countAudit, offset: 0, index: 8)
      encoder.setBytes(&globalFrameOffset, length: 4, index: 9)
      encoder.dispatchThreadgroups(
        MTLSize(width: 1, height: 1, depth: 9),
        threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
      )
      encoder.endEncoding()
      try complete(command)
      let actual = Array(
        UnsafeBufferPointer(
          start: output.contents().bindMemory(to: UInt8.self, capacity: 192 * 192),
          count: 192 * 192
        )
      )
      var expected = [UInt8](repeating: 0, count: 192 * 192)
      for pixel in 0..<(192 * 192) {
        let block = pixel / 4_096
        let local = pixel % 4_096
        let group = local / 32
        let lane = local % 32
        for bit in 0..<8 {
          let wordOffset = bit * 512 + group * 4
          let bytes = decodedBlocks[block]
          let word =
            UInt32(bytes[wordOffset])
            | UInt32(bytes[wordOffset + 1]) << 8
            | UInt32(bytes[wordOffset + 2]) << 16
            | UInt32(bytes[wordOffset + 3]) << 24
          if word & (1 << lane) != 0 { expected[pixel] |= 1 << bit }
        }
      }
      let mismatch = zip(actual, expected).enumerated().first { $0.element.0 != $0.element.1 }
      XCTAssertNil(
        mismatch,
        "parallel full-block mismatch at scan frame \(frame), detector pixel \(mismatch?.offset ?? -1)"
      )
    }
  }

  func testHDF5FunctionsCompile() throws {
    let device = try metalDevice()
    let library = try Metal4DSTEMKernels.makeHDF5Library(device: device)
    let names = [
      Metal4DSTEMKernels.decodeU8Function,
      Metal4DSTEMKernels.decodeU16Function,
      Metal4DSTEMKernels.decodeU16TwoBlockFunction,
      Metal4DSTEMKernels.decodeU16LosslessU8Function,
      Metal4DSTEMKernels.decodeU16AuditedLow8Function,
      Metal4DSTEMKernels.decodeU16AuditedLow8Bin4U16WordMajorFunction,
      Metal4DSTEMKernels.decodeU16AuditedLow8ScalarFunction,
      Metal4DSTEMKernels.binU16AuditedLow8ScalarU16WordMajorFunction,
      Metal4DSTEMKernels.clearU16WordMajorRangeFunction,
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
      Metal4DSTEMKernels.detectorProductsU8MomentsFunction,
      Metal4DSTEMKernels.detectorProductsU16Function,
      Metal4DSTEMKernels.detectorProductsU16MomentsFunction,
      Metal4DSTEMKernels.detectorSumU8Function,
      Metal4DSTEMKernels.detectorSumU16Function,
      Metal4DSTEMKernels.transposeScanWordsFunction,
      Metal4DSTEMKernels.transposeScanWords32x8Function,
      Metal4DSTEMKernels.scanBinU8Function,
      Metal4DSTEMKernels.scanBinU16Function,
      Metal4DSTEMKernels.scanDetectorBinU8Function,
      Metal4DSTEMKernels.scanDetectorBinU8ToU16Function,
      Metal4DSTEMKernels.scanDetectorBinU16Function,
      Metal4DSTEMKernels.residentRebinU8Function,
      Metal4DSTEMKernels.residentRebinU16Function,
      Metal4DSTEMKernels.residentRebinU32Function,
      Metal4DSTEMKernels.detectorProductsU32Function,
      Metal4DSTEMKernels.detectorProductsU16WordMajorFunction,
      Metal4DSTEMKernels.detectorProductsU16WordMajorMomentsFunction,
      Metal4DSTEMKernels.centerOfMassU8Function,
      Metal4DSTEMKernels.centerOfMassU16Function,
      Metal4DSTEMKernels.centerOfMassU32Function,
      Metal4DSTEMKernels.centerOfMassU32MomentsFunction,
      Metal4DSTEMKernels.centerOfMassU64MomentsFunction,
      Metal4DSTEMKernels.widenU32AccumulatorTripletToU64Function,
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
      Metal4DSTEMKernels.scanRegionSumU8Function,
      Metal4DSTEMKernels.scanRegionSumU16Function,
      Metal4DSTEMKernels.scanRegionSumU32Function,
    ]
    for name in names {
      XCTAssertNotNil(library.makeFunction(name: name), "Missing Metal function \(name)")
    }
  }

  func testScanRegionU16MatchesCircleSquareAndReductionReferences() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let scanRows = 5
    let scanColumns = 5
    let scanCount = scanRows * scanColumns
    let detectorPixels = 3
    let frameMajor: [UInt16] = (0..<(scanCount * detectorPixels)).map {
      UInt16($0 + 1)
    }
    var wordMajor = [UInt32](repeating: 0, count: 2 * scanCount)
    for scan in 0..<scanCount {
      for pixel in 0..<detectorPixels {
        wordMajor[(pixel / 2) * scanCount + scan] |=
          UInt32(frameMajor[scan * detectorPixels + pixel]) << UInt32((pixel % 2) * 16)
      }
    }
    let source = try makeBuffer(device: device, values: wordMajor)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.scanRegionSumU16Function)
      )
    )

    for shape in UInt32(0)...UInt32(1) {
      for reduction in UInt32(0)...UInt32(1) {
        let destination = try outputBuffer(device: device, count: detectorPixels)
        var parameters = ScanRegionSumParameters(
          scanRows: UInt32(scanRows),
          scanColumns: UInt32(scanColumns),
          scanCount: UInt32(scanCount),
          detectorPixels: UInt32(detectorPixels),
          centerRow: 2,
          centerColumn: 2,
          radius: 1.1,
          shape: shape,
          reduction: reduction
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
          MTLSize(width: detectorPixels, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: detectorPixels, height: 1, depth: 1)
        )
        encoder.endEncoding()
        try complete(command)

        var expected = [UInt32](repeating: 0, count: detectorPixels)
        for row in 0..<scanRows {
          for column in 0..<scanColumns {
            let rowOffset = Float(row) - 2
            let columnOffset = Float(column) - 2
            let selected =
              shape == 0
              ? rowOffset * rowOffset + columnOffset * columnOffset <= 1.1 * 1.1
              : abs(rowOffset) <= 1.1 && abs(columnOffset) <= 1.1
            guard selected else { continue }
            let scan = row * scanColumns + column
            for pixel in 0..<detectorPixels {
              let value = UInt32(frameMajor[scan * detectorPixels + pixel])
              expected[pixel] =
                reduction == 1
                ? max(expected[pixel], value)
                : expected[pixel] + value
            }
          }
        }
        XCTAssertEqual(bufferValues(destination, count: detectorPixels), expected)
      }
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
            let angle =
              -2 * Double.pi
              * (Double(outputRow * inputRow) / Double(rows)
                + Double(outputColumn * inputColumn) / Double(columns))
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

  func testExactBinnerU16ToPackedU16MatchesReferenceAndRejectsExtraRows() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 3,
      sourceScanColumns: 2,
      detectorRows: 3,
      detectorColumns: 5,
      sourceBytesPerValue: 2,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 3, sourceColumns: 2),
      detectorBin: 2
    )
    let frameCount = plan.scanRegion.scanPositions
    let values = (0..<(frameCount * plan.detectorPixels)).map {
      UInt16(($0 % 97) + 1)
    }
    let audit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "e", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: UInt32(values.max()!),
      pixelsAbove255: 0
    )
    let source = try makeBuffer(device: device, values: values)
    let outputWordsPerScan = (plan.outputDetectorPixels + 1) / 2
    let destination = try outputBuffer(
      device: device,
      count: outputWordsPerScan * plan.outputScanPositions
    )
    let binner = try Metal4DSTEMExactBinner(device: device)
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    _ = try binner.encodeBatch(
      commandBuffer: command,
      stagedSource: source,
      destination: destination,
      plan: plan,
      sourceBatchRows: 3,
      destinationScanRowOffset: 0,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )
    try complete(command)

    var expected = [UInt32](
      repeating: 0,
      count: outputWordsPerScan * plan.outputScanPositions
    )
    for scan in 0..<plan.outputScanPositions {
      var detectorSums = [UInt32](repeating: 0, count: plan.outputDetectorPixels)
      for outputRow in 0..<plan.outputDetectorRows {
        for outputColumn in 0..<plan.outputDetectorColumns {
          let outputPixel = outputRow * plan.outputDetectorColumns + outputColumn
          for row in (outputRow * 2)..<min((outputRow + 1) * 2, plan.detectorRows) {
            for column in (outputColumn * 2)..<min((outputColumn + 1) * 2, plan.detectorColumns) {
              let sourcePixel = row * plan.detectorColumns + column
              detectorSums[outputPixel] += UInt32(
                values[scan * plan.detectorPixels + sourcePixel]
              )
            }
          }
        }
      }
      for word in 0..<outputWordsPerScan {
        let low = detectorSums[word * 2]
        let high =
          word * 2 + 1 < detectorSums.count
          ? detectorSums[word * 2 + 1] : 0
        expected[word * plan.outputScanPositions + scan] = low | (high << 16)
      }
    }
    XCTAssertEqual(bufferValues(destination, count: expected.count), expected)

    let provenance = try Metal4DSTEMExactBinner.provenance(
      plan: plan,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )
    let bytesPerOutputScanRow =
      outputWordsPerScan * plan.outputScanColumns
      * MemoryLayout<UInt32>.stride
    let shardPlan = try Metal4DSTEMExactBinningShardPlan(
      provenance: provenance,
      maximumShardBytes: UInt64(bytesPerOutputScanRow)
    )
    XCTAssertEqual(shardPlan.shards.count, plan.outputScanRows)
    let sourceValuesPerScanRow = plan.scanRegion.columns * plan.detectorPixels
    for shard in shardPlan.shards {
      let shardDestination = try outputBuffer(
        device: device,
        count: outputWordsPerScan * shard.outputScanPositionCount
      )
      let shardCommand = try XCTUnwrap(queue.makeCommandBuffer())
      _ = try binner.encodeBatch(
        commandBuffer: shardCommand,
        stagedSource: source,
        stagedSourceOffset:
          shard.outputScanRowStart * sourceValuesPerScanRow
          * MemoryLayout<UInt16>.stride,
        destination: shardDestination,
        destinationView: .scanRowShard(plan: shardPlan, index: shard.index),
        plan: plan,
        sourceBatchRows: shard.outputScanRowStop - shard.outputScanRowStart,
        destinationScanRowOffset: shard.outputScanRowStart,
        sourceAudit: audit,
        stagingDtype: .uint16,
        outputDtype: .uint16
      )
      try complete(shardCommand)

      var expectedShard: [UInt32] = []
      for word in 0..<outputWordsPerScan {
        let fullWordStart =
          word * plan.outputScanPositions
          + shard.outputScanPositionStart
        let fullWordStop = fullWordStart + shard.outputScanPositionCount
        expectedShard.append(contentsOf: expected[fullWordStart..<fullWordStop])
      }
      XCTAssertEqual(
        bufferValues(shardDestination, count: expectedShard.count),
        expectedShard
      )
    }

    let crossingCommand = try XCTUnwrap(queue.makeCommandBuffer())
    XCTAssertThrowsError(
      try binner.encodeBatch(
        commandBuffer: crossingCommand,
        stagedSource: source,
        destination: destination,
        destinationView: .scanRowShard(plan: shardPlan, index: 0),
        plan: plan,
        sourceBatchRows: 2,
        destinationScanRowOffset: 0,
        sourceAudit: audit,
        stagingDtype: .uint16,
        outputDtype: .uint16
      )
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMExactBinnerError,
        .batchCrossesDestinationShard(
          batchStart: 0, batchStop: 2, shardStart: 0, shardStop: 1
        )
      )
    }

    let invalidShardCommand = try XCTUnwrap(queue.makeCommandBuffer())
    XCTAssertThrowsError(
      try binner.encodeBatch(
        commandBuffer: invalidShardCommand,
        stagedSource: source,
        destination: destination,
        destinationView: .scanRowShard(
          plan: shardPlan,
          index: shardPlan.shards.count
        ),
        plan: plan,
        sourceBatchRows: 1,
        destinationScanRowOffset: 0,
        sourceAudit: audit,
        stagingDtype: .uint16,
        outputDtype: .uint16
      )
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMExactBinnerError,
        .invalidDestinationShard(shardPlan.shards.count)
      )
    }

    let undersizedShardDestination = try outputBuffer(
      device: device,
      count: outputWordsPerScan * plan.outputScanColumns - 1
    )
    let undersizedShardCommand = try XCTUnwrap(queue.makeCommandBuffer())
    XCTAssertThrowsError(
      try binner.encodeBatch(
        commandBuffer: undersizedShardCommand,
        stagedSource: source,
        destination: undersizedShardDestination,
        destinationView: .scanRowShard(plan: shardPlan, index: 0),
        plan: plan,
        sourceBatchRows: 1,
        destinationScanRowOffset: 0,
        sourceAudit: audit,
        stagingDtype: .uint16,
        outputDtype: .uint16
      )
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMExactBinnerError,
        .destinationBufferTooSmall(
          expected: UInt64(bytesPerOutputScanRow),
          actual: UInt64(bytesPerOutputScanRow - MemoryLayout<UInt32>.stride)
        )
      )
    }

    let invalidCommand = try XCTUnwrap(queue.makeCommandBuffer())
    XCTAssertThrowsError(
      try binner.encodeBatch(
        commandBuffer: invalidCommand,
        stagedSource: source,
        destination: destination,
        plan: plan,
        sourceBatchRows: 2,
        destinationScanRowOffset: 2,
        sourceAudit: audit,
        stagingDtype: .uint16,
        outputDtype: .uint16
      )
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMExactBinnerError,
        .invalidBatchCoverage(rows: 2, remaining: 1)
      )
    }
  }

  func testExactAccumulatorBoundsAdmitAuditedLow8DetectorBin4Load() throws {
    let region = try Metal4DSTEMScanRegion.full(sourceRows: 512, sourceColumns: 512)
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 512,
      sourceScanColumns: 512,
      detectorRows: 192,
      detectorColumns: 192,
      sourceBytesPerValue: 2,
      scanRegion: region,
      detectorBin: 4
    )

    let bounds = try plan.exactAccumulatorBounds(maxSourceCount: 255)

    XCTAssertEqual(bounds.maximumScanContributions, 1)
    XCTAssertEqual(bounds.maximumDetectorSum, 9_400_320)
    XCTAssertEqual(bounds.maximumDetectorRowMoment, 441_815_040)
    XCTAssertEqual(bounds.maximumDetectorColumnMoment, 441_815_040)
    XCTAssertTrue(bounds.fitsUInt32Accumulators)
  }

  func testExactAccumulatorBoundsIncludeScanBinAndIncompleteDetectorEdges() throws {
    let region = try Metal4DSTEMScanRegion(
      rowStart: 1,
      rowStop: 4,
      columnStart: 2,
      columnStop: 7,
      sourceRows: 6,
      sourceColumns: 8
    )
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 6,
      sourceScanColumns: 8,
      detectorRows: 5,
      detectorColumns: 7,
      sourceBytesPerValue: 2,
      scanRegion: region,
      scanBin: 4,
      detectorBin: 2
    )

    let bounds = try plan.exactAccumulatorBounds(maxSourceCount: 10)

    XCTAssertEqual(bounds.maximumScanContributions, 12)
    XCTAssertEqual(bounds.maximumDetectorSum, 4_200)
    XCTAssertEqual(bounds.maximumDetectorRowMoment, 8_400)
    XCTAssertEqual(bounds.maximumDetectorColumnMoment, 12_600)
    XCTAssertTrue(bounds.fitsUInt32Accumulators)
  }

  func testExactAccumulatorBoundsRejectHighDynamicRangeU32Moments() throws {
    let region = try Metal4DSTEMScanRegion.full(sourceRows: 2, sourceColumns: 2)
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 2,
      sourceScanColumns: 2,
      detectorRows: 256,
      detectorColumns: 256,
      sourceBytesPerValue: 2,
      scanRegion: region
    )

    let bounds = try plan.exactAccumulatorBounds(maxSourceCount: UInt32(UInt16.max))

    XCTAssertEqual(bounds.maximumDetectorSum, 4_294_901_760)
    XCTAssertGreaterThan(bounds.maximumDetectorRowMoment, UInt64(UInt32.max))
    XCTAssertGreaterThan(bounds.maximumDetectorColumnMoment, UInt64(UInt32.max))
    XCTAssertFalse(bounds.fitsUInt32Accumulators)
  }

  func testExactAccumulatorBoundsFailClosedOnUInt64Overflow() throws {
    let region = try Metal4DSTEMScanRegion.full(sourceRows: 16, sourceColumns: 16)
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 16,
      sourceScanColumns: 16,
      detectorRows: Int.max / 2,
      detectorColumns: 2,
      sourceBytesPerValue: 2,
      scanRegion: region,
      scanBin: 16
    )

    XCTAssertThrowsError(
      try plan.exactAccumulatorBounds(maxSourceCount: UInt32.max)
    ) { error in
      XCTAssertEqual(
        error as? Metal4DSTEMExactAccumulatorBoundsError,
        .arithmeticOverflow
      )
    }
  }

  func testRecommendedStreamingPlanSplitsSelectedLoadWithinScratchBudget() throws {
    let region = try Metal4DSTEMScanRegion.full(sourceRows: 512, sourceColumns: 512)
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 512,
      sourceScanColumns: 512,
      detectorRows: 192,
      detectorColumns: 192,
      sourceBytesPerValue: 2,
      scanRegion: region,
      detectorBin: 4
    )
    let rowBytes = UInt64(512 * 192 * 192 * 2)
    let scratchBudget = rowBytes * 32
    let streaming = try Metal4DSTEMStreamingPlan(
      loadPlan: plan,
      scratchBudgetBytes: scratchBudget,
      preferredDepth: Metal4DSTEMStreamingPlan.recommendedDepth
    )

    XCTAssertEqual(streaming.depth, 4)
    XCTAssertEqual(streaming.rowsPerBatch, 8)
    XCTAssertEqual(streaming.framesPerBuffer, 8 * 512)
    XCTAssertEqual(streaming.bytesPerBuffer, rowBytes * 8)
    XCTAssertEqual(streaming.totalScratchBytes, scratchBudget)
    XCTAssertEqual(streaming.batchCount, 64)
  }

  func testRecommendedStreamingDepthProtectsMemoryConstrainedMacs() {
    XCTAssertEqual(
      Metal4DSTEMStreamingPlan.recommendedDepth(physicalMemoryBytes: UInt64(8) << 30),
      2
    )
    XCTAssertEqual(
      Metal4DSTEMStreamingPlan.recommendedDepth(physicalMemoryBytes: UInt64(16) << 30),
      2
    )
    XCTAssertEqual(
      Metal4DSTEMStreamingPlan.recommendedDepth(physicalMemoryBytes: UInt64(128) << 30),
      4
    )
  }

  func testCompactStagingHalvesScratchWithoutChangingBatchGeometry() throws {
    let region = try Metal4DSTEMScanRegion.full(sourceRows: 512, sourceColumns: 512)
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 512,
      sourceScanColumns: 512,
      detectorRows: 192,
      detectorColumns: 192,
      sourceBytesPerValue: 2,
      scanRegion: region,
      detectorBin: 4
    )
    let nativeRowBytes = UInt64(512 * 192 * 192 * 2)
    let native = try Metal4DSTEMStreamingPlan(
      loadPlan: plan,
      scratchBudgetBytes: nativeRowBytes * 32,
      preferredDepth: 4
    )
    let compact = try Metal4DSTEMStreamingPlan(
      loadPlan: plan,
      scratchBudgetBytes: nativeRowBytes * 16,
      preferredDepth: 4,
      stagingBytesPerValue: 1
    )

    XCTAssertEqual(compact.depth, native.depth)
    XCTAssertEqual(compact.rowsPerBatch, native.rowsPerBatch)
    XCTAssertEqual(compact.framesPerBuffer, native.framesPerBuffer)
    XCTAssertEqual(compact.bytesPerBuffer * 2, native.bytesPerBuffer)
    XCTAssertEqual(compact.totalScratchBytes * 2, native.totalScratchBytes)
  }

  func testStreamingPlanAlignsBatchesToScanBin() throws {
    let region = try Metal4DSTEMScanRegion.full(sourceRows: 33, sourceColumns: 10)
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 33,
      sourceScanColumns: 10,
      detectorRows: 2,
      detectorColumns: 2,
      sourceBytesPerValue: 2,
      scanRegion: region,
      scanBin: 4,
      detectorBin: 2
    )
    let rowBytes = UInt64(10 * 2 * 2 * 2)
    let streaming = try Metal4DSTEMStreamingPlan(
      loadPlan: plan,
      scratchBudgetBytes: rowBytes * 10 * 2,
      preferredDepth: 2
    )

    XCTAssertEqual(streaming.depth, 2)
    XCTAssertEqual(streaming.rowsPerBatch, 8)
    XCTAssertEqual(streaming.batchCount, 5)
    XCTAssertLessThanOrEqual(streaming.totalScratchBytes, rowBytes * 10 * 2)
  }

  func testStreamingPlanRejectsInvalidOrTooSmallBudget() throws {
    let region = try Metal4DSTEMScanRegion.full(sourceRows: 2, sourceColumns: 2)
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 2,
      sourceScanColumns: 2,
      detectorRows: 2,
      detectorColumns: 2,
      sourceBytesPerValue: 2,
      scanRegion: region
    )

    XCTAssertThrowsError(
      try Metal4DSTEMStreamingPlan(loadPlan: plan, scratchBudgetBytes: 16, preferredDepth: 0)
    ) { error in
      XCTAssertEqual(error as? Metal4DSTEMLoadPlanError, .invalidStreamingDepth(0))
    }
    XCTAssertThrowsError(
      try Metal4DSTEMStreamingPlan(loadPlan: plan, scratchBudgetBytes: 15, preferredDepth: 1)
    ) { error in
      XCTAssertEqual(error as? Metal4DSTEMLoadPlanError, .insufficientStreamingScratchBudget)
    }
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

  func testDetectorBinU8ToPackedU16MatchesIntegerReference() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(name: Metal4DSTEMKernels.scanDetectorBinU8ToU16Function)
      )
    )
    let scanRows = 2
    let scanColumns = 3
    let detectorRows = 3
    let detectorColumns = 5
    let detectorPixels = detectorRows * detectorColumns
    let sourceValues: [UInt8] = (0..<(scanRows * scanColumns * detectorPixels)).map {
      UInt8(($0 * 7 + 3) % 251)
    }
    let source = try makeBuffer(device: device, values: sourceValues)
    let outputDetectorRows = 2
    let outputDetectorColumns = 3
    let outputDetectorPixels = outputDetectorRows * outputDetectorColumns
    let outputScans = scanRows * scanColumns
    let outputWords = (outputDetectorPixels + 1) / 2
    let destination = try outputBuffer(device: device, count: outputWords * outputScans)
    var parameters = ScanDetectorBinParameters(
      sourceRows: UInt32(scanRows),
      sourceCols: UInt32(scanColumns),
      sourceDetectorRows: UInt32(detectorRows),
      sourceDetectorCols: UInt32(detectorColumns),
      scanBin: 1,
      detectorBin: 2,
      outputScanCount: UInt32(outputScans),
      outputScanCols: UInt32(scanColumns),
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
      MTLSize(width: outputWords, height: outputScans, depth: 1),
      threadsPerThreadgroup: MTLSize(width: outputWords, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    var expected = [UInt32](repeating: 0, count: outputWords * outputScans)
    for scan in 0..<outputScans {
      for outputPixel in 0..<outputDetectorPixels {
        let outputRow = outputPixel / outputDetectorColumns
        let outputColumn = outputPixel % outputDetectorColumns
        var sum: UInt32 = 0
        for row in (outputRow * 2)..<min(detectorRows, outputRow * 2 + 2) {
          for column in (outputColumn * 2)..<min(detectorColumns, outputColumn * 2 + 2) {
            sum += UInt32(sourceValues[scan * detectorPixels + row * detectorColumns + column])
          }
        }
        expected[(outputPixel / 2) * outputScans + scan] |=
          sum << UInt32((outputPixel % 2) * 16)
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

  func testWordMajorPackedU16DetectorProductsMatchReference() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let scanCount = 5
    let detectorPixels = 7
    let values: [UInt16] = (0..<detectorPixels).flatMap { pixel in
      (0..<scanCount).map { scan in UInt16(100 * pixel + scan + 1) }
    }
    var packed = [UInt32](
      repeating: 0,
      count: ((detectorPixels + 1) / 2) * scanCount
    )
    for pixel in 0..<detectorPixels {
      for scan in 0..<scanCount {
        packed[(pixel / 2) * scanCount + scan] |=
          UInt32(values[pixel * scanCount + scan]) << UInt32((pixel % 2) * 16)
      }
    }
    let bands: [UInt8] = (0..<detectorPixels).map { pixel in
      (pixel % 2 == 0 ? 1 : 0) | (pixel % 3 == 0 ? 2 : 0) | (pixel >= 4 ? 4 : 0)
    }
    let source = try makeBuffer(device: device, values: packed)
    let bandBuffer = try makeBuffer(device: device, values: bands)
    let bf = try outputBuffer(device: device, count: scanCount)
    let abf = try outputBuffer(device: device, count: scanCount)
    let df = try outputBuffer(device: device, count: scanCount)
    var parameters = WordMajorDetectorParameters(
      scanCount: UInt32(scanCount),
      detectorPixels: UInt32(detectorPixels)
    )
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels.detectorProductsU16WordMajorFunction
        )
      )
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(bf, offset: 0, index: 1)
    encoder.setBuffer(abf, offset: 0, index: 2)
    encoder.setBuffer(df, offset: 0, index: 3)
    withUnsafeBytes(of: &parameters) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 4)
    }
    encoder.setBuffer(bandBuffer, offset: 0, index: 5)
    encoder.dispatchThreadgroups(
      MTLSize(width: scanCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    func reference(_ band: UInt8) -> [UInt32] {
      (0..<scanCount).map { scan in
        (0..<detectorPixels).reduce(UInt32(0)) { sum, pixel in
          sum
            + (bands[pixel] & band == 0
              ? 0 : UInt32(values[pixel * scanCount + scan]))
        }
      }
    }
    XCTAssertEqual(bufferValues(bf, count: scanCount), reference(1))
    XCTAssertEqual(bufferValues(abf, count: scanCount), reference(2))
    XCTAssertEqual(bufferValues(df, count: scanCount), reference(4))
  }

  func testExactU64WindowProductsAndDetectorAccumulationDoNotOverflowUInt32() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let productsPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels.detectorProductsU16ExactU64Function
        )
      )
    )
    let detectorSumPipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels.detectorAccumulateU16U64Function
        )
      )
    )
    let detectorRows = 257
    let detectorColumns = 256
    let detectorPixels = detectorRows * detectorColumns
    let sourceValues = [UInt16](repeating: .max, count: detectorPixels)
    let bands = [UInt8](repeating: 7, count: detectorPixels)
    let source = try makeBuffer(device: device, values: sourceValues)
    let bandBuffer = try makeBuffer(device: device, values: bands)
    let band1 = try outputBuffer64(device: device, count: 2)
    let band2 = try outputBuffer64(device: device, count: 2)
    let band4 = try outputBuffer64(device: device, count: 2)
    let total = try outputBuffer64(device: device, count: 2)
    let rowMoment = try outputBuffer64(device: device, count: 2)
    let columnMoment = try outputBuffer64(device: device, count: 2)
    let detectorSum = try outputBuffer64(device: device, count: detectorPixels)

    for globalFrame in 0..<2 {
      var parameters = DetectorParameters(
        frameCount: 1,
        detectorPixels: UInt32(detectorPixels),
        globalFrameOffset: UInt32(globalFrame)
      )
      var detectorColumnCount = UInt32(detectorColumns)
      var detectorPixelCount = UInt32(detectorPixels)
      var frameCount: UInt32 = 1
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
      encoder.setComputePipelineState(productsPipeline)
      encoder.setBuffer(source, offset: 0, index: 0)
      encoder.setBuffer(band1, offset: 0, index: 1)
      encoder.setBuffer(band2, offset: 0, index: 2)
      encoder.setBuffer(band4, offset: 0, index: 3)
      withUnsafeBytes(of: &parameters) {
        encoder.setBytes($0.baseAddress!, length: $0.count, index: 4)
      }
      encoder.setBuffer(bandBuffer, offset: 0, index: 5)
      encoder.setBuffer(total, offset: 0, index: 6)
      encoder.setBuffer(rowMoment, offset: 0, index: 7)
      encoder.setBuffer(columnMoment, offset: 0, index: 8)
      encoder.setBytes(&detectorColumnCount, length: 4, index: 9)
      encoder.dispatchThreadgroups(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
      )
      encoder.memoryBarrier(scope: .buffers)
      encoder.setComputePipelineState(detectorSumPipeline)
      encoder.setBuffer(source, offset: 0, index: 0)
      encoder.setBuffer(detectorSum, offset: 0, index: 1)
      encoder.setBytes(&detectorPixelCount, length: 4, index: 2)
      encoder.setBytes(&frameCount, length: 4, index: 3)
      encoder.dispatchThreads(
        MTLSize(width: detectorPixels, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
      )
      encoder.endEncoding()
      try complete(command)
    }

    let expectedBand = UInt64(UInt16.max) * UInt64(detectorPixels)
    XCTAssertGreaterThan(expectedBand, UInt64(UInt32.max))
    let rowCoordinateSum = UInt64((detectorRows - 1) * detectorRows / 2)
    let columnCoordinateSum = UInt64(
      (detectorColumns - 1) * detectorColumns / 2
    )
    let expectedRowMoment =
      UInt64(UInt16.max) * UInt64(detectorColumns) * rowCoordinateSum
    let expectedColumnMoment =
      UInt64(UInt16.max) * UInt64(detectorRows) * columnCoordinateSum
    XCTAssertEqual(bufferValues64(band1, count: 2), [expectedBand, expectedBand])
    XCTAssertEqual(bufferValues64(band2, count: 2), [expectedBand, expectedBand])
    XCTAssertEqual(bufferValues64(band4, count: 2), [expectedBand, expectedBand])
    XCTAssertEqual(bufferValues64(total, count: 2), [expectedBand, expectedBand])
    XCTAssertEqual(
      bufferValues64(rowMoment, count: 2),
      [expectedRowMoment, expectedRowMoment]
    )
    XCTAssertEqual(
      bufferValues64(columnMoment, count: 2),
      [expectedColumnMoment, expectedColumnMoment]
    )
    XCTAssertEqual(
      bufferValues64(detectorSum, count: detectorPixels),
      [UInt64](repeating: UInt64(UInt16.max) * 2, count: detectorPixels)
    )
  }

  func testWordMajorPackedU16SummaryMomentsMatchReference() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let scanCount = 5
    let detectorRows = 2
    let detectorColumns = 4
    let detectorPixels = detectorRows * detectorColumns
    let values: [UInt16] = (0..<detectorPixels).flatMap { pixel in
      (0..<scanCount).map { scan in UInt16(100 * pixel + scan + 1) }
    }
    var packed = [UInt32](repeating: 0, count: (detectorPixels / 2) * scanCount)
    for pixel in 0..<detectorPixels {
      for scan in 0..<scanCount {
        packed[(pixel / 2) * scanCount + scan] |=
          UInt32(values[pixel * scanCount + scan]) << UInt32((pixel % 2) * 16)
      }
    }
    let bands: [UInt8] = (0..<detectorPixels).map { pixel in
      (pixel % 2 == 0 ? 1 : 0) | (pixel % 3 == 0 ? 2 : 0) | (pixel >= 4 ? 4 : 0)
    }
    let source = try makeBuffer(device: device, values: packed)
    let bandBuffer = try makeBuffer(device: device, values: bands)
    let bf = try outputBuffer(device: device, count: scanCount)
    let abf = try outputBuffer(device: device, count: scanCount)
    let df = try outputBuffer(device: device, count: scanCount)
    let total = try outputBuffer64(device: device, count: scanCount)
    let rowMoment = try outputBuffer64(device: device, count: scanCount)
    let columnMoment = try outputBuffer64(device: device, count: scanCount)
    var parameters = WordMajorDetectorParameters(
      scanCount: UInt32(scanCount),
      detectorPixels: UInt32(detectorPixels)
    )
    var detectorColumnCount = UInt32(detectorColumns)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels.detectorProductsU16WordMajorMomentsFunction
        )
      )
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(bf, offset: 0, index: 1)
    encoder.setBuffer(abf, offset: 0, index: 2)
    encoder.setBuffer(df, offset: 0, index: 3)
    withUnsafeBytes(of: &parameters) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 4)
    }
    encoder.setBuffer(bandBuffer, offset: 0, index: 5)
    encoder.setBuffer(total, offset: 0, index: 6)
    encoder.setBuffer(rowMoment, offset: 0, index: 7)
    encoder.setBuffer(columnMoment, offset: 0, index: 8)
    encoder.setBytes(&detectorColumnCount, length: 4, index: 9)
    encoder.dispatchThreadgroups(
      MTLSize(width: scanCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    func bandReference(_ band: UInt8) -> [UInt32] {
      (0..<scanCount).map { scan in
        (0..<detectorPixels).reduce(UInt32(0)) { sum, pixel in
          sum
            + (bands[pixel] & band == 0
              ? 0 : UInt32(values[pixel * scanCount + scan]))
        }
      }
    }
    let totalReference: [UInt64] = (0..<scanCount).map { scan in
      (0..<detectorPixels).reduce(UInt64(0)) { sum, pixel in
        sum + UInt64(values[pixel * scanCount + scan])
      }
    }
    let rowReference: [UInt64] = (0..<scanCount).map { scan in
      (0..<detectorPixels).reduce(UInt64(0)) { sum, pixel in
        sum + UInt64(values[pixel * scanCount + scan]) * UInt64(pixel / detectorColumns)
      }
    }
    let columnReference: [UInt64] = (0..<scanCount).map { scan in
      (0..<detectorPixels).reduce(UInt64(0)) { sum, pixel in
        sum + UInt64(values[pixel * scanCount + scan]) * UInt64(pixel % detectorColumns)
      }
    }
    XCTAssertEqual(bufferValues(bf, count: scanCount), bandReference(1))
    XCTAssertEqual(bufferValues(abf, count: scanCount), bandReference(2))
    XCTAssertEqual(bufferValues(df, count: scanCount), bandReference(4))
    XCTAssertEqual(bufferValues64(total, count: scanCount), totalReference)
    XCTAssertEqual(bufferValues64(rowMoment, count: scanCount), rowReference)
    XCTAssertEqual(bufferValues64(columnMoment, count: scanCount), columnReference)
  }

  func testWordMajorSummaryMomentsPreserveFullUInt16DynamicRange() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let detectorRows = 256
    let detectorColumns = 256
    let detectorPixels = detectorRows * detectorColumns
    let packed = [UInt32](
      repeating: UInt32(UInt16.max) | UInt32(UInt16.max) << 16,
      count: detectorPixels / 2
    )
    let source = try makeBuffer(device: device, values: packed)
    let bands = try makeBuffer(
      device: device,
      values: [UInt8](repeating: 0, count: detectorPixels)
    )
    let bf = try outputBuffer(device: device, count: 1)
    let abf = try outputBuffer(device: device, count: 1)
    let df = try outputBuffer(device: device, count: 1)
    let total = try outputBuffer64(device: device, count: 1)
    let rowMoment = try outputBuffer64(device: device, count: 1)
    let columnMoment = try outputBuffer64(device: device, count: 1)
    var parameters = WordMajorDetectorParameters(
      scanCount: 1,
      detectorPixels: UInt32(detectorPixels)
    )
    var detectorColumnCount = UInt32(detectorColumns)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels.detectorProductsU16WordMajorMomentsFunction
        )
      )
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(bf, offset: 0, index: 1)
    encoder.setBuffer(abf, offset: 0, index: 2)
    encoder.setBuffer(df, offset: 0, index: 3)
    withUnsafeBytes(of: &parameters) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 4)
    }
    encoder.setBuffer(bands, offset: 0, index: 5)
    encoder.setBuffer(total, offset: 0, index: 6)
    encoder.setBuffer(rowMoment, offset: 0, index: 7)
    encoder.setBuffer(columnMoment, offset: 0, index: 8)
    encoder.setBytes(&detectorColumnCount, length: 4, index: 9)
    encoder.dispatchThreadgroups(
      MTLSize(width: 1, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    let value = UInt64(UInt16.max)
    let expectedTotal = value * UInt64(detectorPixels)
    let coordinateSum = UInt64(detectorRows * (detectorRows - 1) / 2)
    let expectedRowMoment = value * UInt64(detectorColumns) * coordinateSum
    let expectedColumnMoment = value * UInt64(detectorRows) * coordinateSum
    XCTAssertGreaterThan(expectedRowMoment, UInt64(UInt32.max))
    XCTAssertEqual(bufferValues64(total, count: 1), [expectedTotal])
    XCTAssertEqual(bufferValues64(rowMoment, count: 1), [expectedRowMoment])
    XCTAssertEqual(bufferValues64(columnMoment, count: 1), [expectedColumnMoment])
    XCTAssertEqual(bufferValues(bf, count: 1), [0])
    XCTAssertEqual(bufferValues(abf, count: 1), [0])
    XCTAssertEqual(bufferValues(df, count: 1), [0])
  }

  func testWidenU32AccumulatorTripletToU64IsExact() throws {
    let device = try metalDevice()
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    let firstValues: [UInt32] = [0, 1, UInt32.max, 19]
    let secondValues: [UInt32] = [7, UInt32.max, 2, 0]
    let thirdValues: [UInt32] = [UInt32.max - 1, 9, 0, UInt32.max]
    let firstInput = try makeBuffer(device: device, values: firstValues)
    let secondInput = try makeBuffer(device: device, values: secondValues)
    let thirdInput = try makeBuffer(device: device, values: thirdValues)
    let firstOutput = try outputBuffer64(device: device, count: firstValues.count)
    let secondOutput = try outputBuffer64(device: device, count: firstValues.count)
    let thirdOutput = try outputBuffer64(device: device, count: firstValues.count)
    var count = UInt32(firstValues.count)
    let pipeline = try device.makeComputePipelineState(
      function: XCTUnwrap(
        library.makeFunction(
          name: Metal4DSTEMKernels.widenU32AccumulatorTripletToU64Function
        )
      )
    )
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(firstInput, offset: 0, index: 0)
    encoder.setBuffer(secondInput, offset: 0, index: 1)
    encoder.setBuffer(thirdInput, offset: 0, index: 2)
    encoder.setBuffer(firstOutput, offset: 0, index: 3)
    encoder.setBuffer(secondOutput, offset: 0, index: 4)
    encoder.setBuffer(thirdOutput, offset: 0, index: 5)
    encoder.setBytes(&count, length: 4, index: 6)
    encoder.dispatchThreads(
      MTLSize(width: firstValues.count + 4, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1)
    )
    encoder.endEncoding()
    try complete(command)

    XCTAssertEqual(
      bufferValues64(firstOutput, count: firstValues.count),
      firstValues.map(UInt64.init)
    )
    XCTAssertEqual(
      bufferValues64(secondOutput, count: secondValues.count),
      secondValues.map(UInt64.init)
    )
    XCTAssertEqual(
      bufferValues64(thirdOutput, count: thirdValues.count),
      thirdValues.map(UInt64.init)
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
    let deltaExpected: [UInt32] = (0..<scanCount).map { scanIndex in
      let firstOffset = 2 * scanCount + scanIndex
      let secondOffset = 3 * scanCount + scanIndex
      return values[firstOffset] + values[secondOffset]
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

  private func outputBuffer64(device: MTLDevice, count: Int) throws -> MTLBuffer {
    let buffer = try XCTUnwrap(
      device.makeBuffer(
        length: count * MemoryLayout<UInt64>.stride,
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

  private func bufferValues64(_ buffer: MTLBuffer, count: Int) -> [UInt64] {
    let pointer = buffer.contents().bindMemory(to: UInt64.self, capacity: count)
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
