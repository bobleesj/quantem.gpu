import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels
@_spi(PairedRuntimeTANSPrototype) @testable import Metal4DSTEMStreamingIO
import XCTest

final class PairedRuntimeDetectorColumnsTests: XCTestCase {
  func testExactColumnsAcrossPacketsModesAndOffsetLayouts() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("A physical Metal device is required")
    }
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let pixels = 8, packets = 8, scans = packets * 512
    let streams: [[UInt16]] = (0..<pixels * packets).map { stream in
      (0..<512).map { scan in
        switch stream % pixels {
        case 0: return 0
        case 1: return UInt16(65_535 - stream)
        case 2: return scan == 511 ? 128 : 0
        case 3: return UInt16((scan * 37 + stream) % 65_536)
        case 4: return scan == 251 ? 65_535 : UInt16((scan * 17 + scan / 7) % 11)
        default: return UInt16((scan * 13 + stream) % 5)
        }
      }
    }
    let codec = try MetalPairedRuntimeTANSSyntheticCodec(device: device)
    let encoded = try codec.roundTrip(streams: streams, logicalDtype: .uint16)
    XCTAssertEqual(encoded.decodedStreams, streams)
    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
      try values.withUnsafeBytes {
        try XCTUnwrap(device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared))
      }
    }
    let payload = try buffer(encoded.payload), modes = try buffer(encoded.modes)
    let table = try buffer(PairedRuntimeTANSTables.build().packedDecoding)
    for compact in [false, true] {
      let offsets: MTLBuffer
      if compact {
        let bases = stride(from: 0, through: pixels * packets, by: 32).map { encoded.offsets[$0] }
        let starts = encoded.offsets.indices.map { UInt16(encoded.offsets[$0] - encoded.offsets[$0 & ~31]) }
        var bytes = bases.withUnsafeBytes { Data($0) }
        bytes.append(starts.withUnsafeBytes { Data($0) })
        offsets = try buffer(Array(bytes))
      } else {
        offsets = try buffer(encoded.offsets)
      }
      let reader = try PairedRuntimeDetectorColumns(device: device, library: library, compactOffsets: compact)
      for selected in [[UInt32(7), 0, 4, 7, 2, 1, 3, 6, 5], (0..<32).map { UInt32($0 % pixels) }] {
        let output = try XCTUnwrap(device.makeBuffer(length: selected.count * scans * 4 + 4, options: .storageModeShared))
        memset(output.contents(), 0xCD, output.length)
        try reader.read(queue: queue, payload: payload, offsets: offsets, modes: modes,
          table: table, detectorPixels: pixels, scanCount: scans, selectedRanks: selected, into: output)
        let actual = output.contents().assumingMemoryBound(to: UInt32.self)
        for (plane, pixel) in selected.enumerated() {
          let expected = (0..<packets).flatMap { streams[$0 * pixels + Int(pixel)].map(UInt32.init) }
          XCTAssertEqual(Array(UnsafeBufferPointer(start: actual + plane * scans, count: scans)), expected)
        }
        XCTAssertEqual(actual[selected.count * scans], 0xCDCDCDCD, "No write beyond the requested columns")
        XCTAssertThrowsError(try reader.read(queue: queue, payload: payload, offsets: offsets, modes: modes,
          table: table, detectorPixels: pixels, scanCount: scans, selectedRanks: [], into: output))
        XCTAssertThrowsError(try reader.read(queue: queue, payload: payload, offsets: offsets, modes: modes,
          table: table, detectorPixels: pixels, scanCount: scans, selectedRanks: [UInt32(pixels)], into: output))
      }
      let short = try XCTUnwrap(device.makeBuffer(length: scans * 4 - 4, options: .storageModeShared))
      XCTAssertThrowsError(try reader.read(queue: queue, payload: payload, offsets: offsets, modes: modes,
        table: table, detectorPixels: pixels, scanCount: scans, selectedRanks: [0], into: short))
      modes.contents().assumingMemoryBound(to: UInt8.self)[7 * pixels + 4] = 250
      let output = try XCTUnwrap(device.makeBuffer(length: scans * 4, options: .storageModeShared))
      XCTAssertThrowsError(try reader.read(queue: queue, payload: payload, offsets: offsets, modes: modes,
        table: table, detectorPixels: pixels, scanCount: scans, selectedRanks: [4], into: output))
      modes.contents().assumingMemoryBound(to: UInt8.self)[7 * pixels + 4] = encoded.modes[7 * pixels + 4]
    }
  }
}
