import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMKernels

private struct ScanRotationFixture: Decodable {
  struct Source: Decodable {
    let shape: [Int]
    let dtype: String
    let values: [UInt16]
  }

  struct Case: Decodable {
    let id: String
    let quarterTurnsCounterclockwise: UInt32
    let outputShape: [Int]
    let expectedValues: [UInt16]
  }

  let id: String
  let source: Source
  let cases: [Case]
}

final class ScanRotationParityTests: XCTestCase {
  func testMetalQuarterTurnsMatchSharedGoldFixture() throws {
    let fixture = try loadFixture()
    XCTAssertEqual(fixture.id, "quantem-scan-rotation-v1")
    XCTAssertEqual(fixture.source.dtype, "uint16")
    XCTAssertEqual(fixture.source.shape.count, 4)
    let scanRows = fixture.source.shape[0]
    let scanColumns = fixture.source.shape[1]
    let detectorPixels = fixture.source.shape[2] * fixture.source.shape[3]
    XCTAssertEqual(detectorPixels % 2, 0)
    let wordsPerScan = detectorPixels / 2
    let sourceWords = packUInt16(fixture.source.values)

    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let rotator = try MetalScanRotator(device: device)
    let source = try XCTUnwrap(
      device.makeBuffer(
        bytes: sourceWords,
        length: sourceWords.count * MemoryLayout<UInt32>.stride,
        options: .storageModeShared
      )
    )
    let destination = try XCTUnwrap(
      device.makeBuffer(
        length: source.length,
        options: .storageModeShared
      )
    )

    for testCase in fixture.cases {
      let quarterTurn = try XCTUnwrap(
        ScanQuarterTurn(rawValue: testCase.quarterTurnsCounterclockwise),
        "Unsupported quarter turn in \(testCase.id)"
      )
      let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
      let shape = try rotator.encode(
        source: source,
        destination: destination,
        scanRows: scanRows,
        scanColumns: scanColumns,
        wordsPerScan: wordsPerScan,
        quarterTurn: quarterTurn,
        commandBuffer: commandBuffer
      )
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      XCTAssertEqual(commandBuffer.status, .completed, testCase.id)
      XCTAssertEqual(
        shape,
        RotatedScanShape(
          rows: testCase.outputShape[0],
          columns: testCase.outputShape[1]
        ),
        testCase.id
      )
      let outputWords = Array(
        UnsafeBufferPointer(
          start: destination.contents().assumingMemoryBound(to: UInt32.self),
          count: sourceWords.count
        )
      )
      XCTAssertEqual(unpackUInt16(outputWords), testCase.expectedValues, testCase.id)
    }
  }

  private func loadFixture() throws -> ScanRotationFixture {
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<4 {
      root.deleteLastPathComponent()
    }
    let url = root
      .appendingPathComponent("parity", isDirectory: true)
      .appendingPathComponent("scan_rotation_v1.json")
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(
      ScanRotationFixture.self,
      from: Data(contentsOf: url)
    )
  }

  private func packUInt16(_ values: [UInt16]) -> [UInt32] {
    stride(from: 0, to: values.count, by: 2).map { index in
      UInt32(values[index]) | UInt32(values[index + 1]) << 16
    }
  }

  private func unpackUInt16(_ words: [UInt32]) -> [UInt16] {
    words.flatMap { word in
      [UInt16(word & 0xffff), UInt16(word >> 16)]
    }
  }
}
