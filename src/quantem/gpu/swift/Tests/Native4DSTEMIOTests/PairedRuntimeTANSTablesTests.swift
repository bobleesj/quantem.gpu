import XCTest

@testable import Metal4DSTEMStreamingIO

final class PairedRuntimeTANSTablesTests: XCTestCase {
  func testFrozenOracleHashesAndTransitionInversion() throws {
    let tables = try PairedRuntimeTANSTables.build()

    XCTAssertEqual(tables.frequencies.count, 32 * 1_089)
    XCTAssertEqual(tables.encoding.count, 32 * 1_024)
    XCTAssertEqual(tables.packedDecoding.count, 32 * 1_024)
    XCTAssertEqual(
      tables.frequencySHA256,
      "74a39b914a3d563cec122294bf3d29c4972e089ae64ae42c6500eaaafde4145f")
    XCTAssertEqual(
      tables.encodingSHA256,
      "b76355bf9e4d4c9bf4949faf412671d7ec881b232df7a303250a5071342253d8")
    XCTAssertEqual(
      tables.packedDecodingSHA256,
      "44966937a0352082f194c121b37cbd0dd6883cf65919f2d73f94a893c303becd")
    XCTAssertEqual(try tables.validateTransitionInversion(), 2_572_288)
  }

  func testAllModelsContainExactlyOneStateTable() throws {
    let tables = try PairedRuntimeTANSTables.build()
    for model in 0..<PairedRuntimeTANSTables.modelCount {
      let lower = model * PairedRuntimeTANSTables.symbolCount
      let upper = lower + PairedRuntimeTANSTables.symbolCount
      XCTAssertEqual(
        tables.frequencies[lower..<upper].reduce(UInt32(0), +),
        UInt32(PairedRuntimeTANSTables.stateCount))
    }
  }

  func testTwoAndFourBitMacroTablesMatchEveryScalarEntry() throws {
    let decoding = try PairedRuntimeTANSTables.build().packedDecoding

    for lookaheadBits in [2, 4] {
      let lookaheadCount = 1 << lookaheadBits
      let wordsPerModel = 1_024 + 1_024 * lookaheadCount * 2
      let table = try PairedRuntimeTANSMacroTable.build(
        decoding: decoding, lookaheadBits: lookaheadBits)
      XCTAssertEqual(table.count, 32 * wordsPerModel)
      XCTAssertEqual(
        try PairedRuntimeTANSMacroTable.validate(
          interleaved: table, decoding: decoding, lookaheadBits: lookaheadBits),
        32 * 1_024 * lookaheadCount)
    }
  }

  func testMacroTableRejectsUnsupportedLookaheadWidth() throws {
    let decoding = try PairedRuntimeTANSTables.build().packedDecoding
    XCTAssertThrowsError(
      try PairedRuntimeTANSMacroTable.build(decoding: decoding, lookaheadBits: 3))
  }
}
