import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMKernels
import XCTest

@testable import Metal4DSTEMStreamingIO

final class TANSTileLayoutTests: XCTestCase {
  func testBlockedIndexExactHighCountsAndFailedAdmissionPreservePriorFields() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    let index = try TANSExactTileIndex(
      device: device, queue: queue, library: library, blockedPacking: true)
    let image = try XCTUnwrap(
      device.makeBuffer(length: 262144 * 4, options: .storageModeShared))
    let values = image.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<262144 {
      let patterns: [UInt32] = [0, 63, 65535, 65536, 2_415_882_240, UInt32.max]
      values[i] = i < 256 ? UInt32.max : patterns[(i * 7) % patterns.count]
    }
    let tile = TANSExactTileIndex.Tile(row: 0, col: 0, side: 16)
    // append includes a complete independent decode/roundtrip comparison.
    try index.append(tile: tile, images: [image], maximumBytes: 8 << 20)
    let prior = index.residentBytes
    var target = [UInt8](repeating: 0, count: 36864)
    for pixel in tile.pixels { target[pixel] = 1 }
    var previous = [UInt8](repeating: 0, count: 36864)
    previous[0] = 1
    let valid = [UInt8](repeating: 1, count: 36864)
    let plan = index.planChoosingBase(
      mask: target, previous: previous, preferPrevious: true,
      valid: valid, cost: [Double](repeating: 1, count: 36864))
    XCTAssertFalse(plan.usePrevious, "Fewer changed pixels is not fewer decoded pixels")
    XCTAssertTrue(plan.residual.allSatisfy { $0 == 0 })
    XCTAssertEqual(plan.tiles.count, 1)
    var reconstructed = plan.residual
    for (field, sign) in plan.tiles {
      for pixel in index.fields[field].tile.pixels { reconstructed[pixel] += sign }
    }
    XCTAssertEqual(reconstructed, target.map(Int32.init))
    let same = index.planChoosingBase(
      mask: target, previous: target, preferPrevious: true,
      valid: valid, cost: [Double](repeating: 1, count: 36864))
    XCTAssertTrue(same.usePrevious)
    XCTAssertTrue(same.residual.allSatisfy { $0 == 0 })
    XCTAssertTrue(same.tiles.isEmpty)
    let payload = index.fields[0].payload
    XCTAssertThrowsError(try index.append(tile: tile, images: [image], maximumBytes: 0))
    XCTAssertEqual(index.fields.count, 1)
    XCTAssertEqual(index.residentBytes, prior)
    XCTAssertTrue(index.fields[0].payload === payload)
    try index.append(tile: tile, images: [image], maximumBytes: 8 << 20)
    XCTAssertEqual(index.fields.count, 2)
    XCTAssertEqual(index.residentBytes, prior * 2)
  }

  func testUniformLayoutsPartitionEveryDetectorPixelExactlyOnce() {
    for layout in [TANSExactTileIndex.Layout.uniform16, .uniform24] {
      let tiles = TANSExactTileIndex.tiles(for: layout)
      var coverage = [Int](repeating: 0, count: 192 * 192)
      for tile in tiles {
        XCTAssertGreaterThanOrEqual(tile.row, 0)
        XCTAssertLessThanOrEqual(tile.row + tile.side, 192)
        XCTAssertGreaterThanOrEqual(tile.col, 0)
        XCTAssertLessThanOrEqual(tile.col + tile.side, 192)
        for pixel in tile.pixels { coverage[pixel] += 1 }
      }
      XCTAssertTrue(coverage.allSatisfy { $0 == 1 })
      XCTAssertEqual(tiles.count, layout == .uniform16 ? 144 : 64)
    }
  }

  func testDefaultDictionaryIsUnchanged() {
    let tiles = TANSExactTileIndex.tiles(for: .centerFine)
    XCTAssertEqual(tiles.count, 100)
    XCTAssertTrue(tiles.prefix(36).allSatisfy { $0.side == 32 })
    XCTAssertTrue(tiles.suffix(64).allSatisfy { $0.side == 8 })
    XCTAssertEqual(tiles.first?.row, 0)
    XCTAssertEqual(tiles.last?.row, 120)
    XCTAssertEqual(tiles.last?.col, 120)
  }
}
