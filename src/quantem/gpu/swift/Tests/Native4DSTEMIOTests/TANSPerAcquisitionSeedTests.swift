import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Exactness of independent per-acquisition delta seeds and output rings while
/// single-acquisition, subset and full queries interleave. Every image is
/// compared with an unseeded recompute or the frozen full-series reference.
final class TANSPerAcquisitionSeedTests: XCTestCase {
  func testInterleavedSingleSubsetAndFullQueriesStayExactWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_SEED_FIXTURE"] else {
      throw XCTSkip("Requires the complete sealed 66-acquisition archive and a 128 GB-class device")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.releaseResidentStorage() }
    try source.prepareExperimentalTileIndex(maximumIndexBytes: 2 << 30)
    source.experimentalUseTileIndex = true
    source.experimentalDetectorStreamsPerLane = 32
    let valid = source.validDetectorMask
    func mask(_ row: Double, _ col: Double, _ inner: Double, _ outer: Double) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - col
        let radius2 = r * r + c * c
        return valid[q] != 0 && radius2 >= inner * inner && radius2 <= outer * outer ? 1 : 0
      }
    }
    func hash(_ buffer: MTLBuffer) -> String {
      SHA256.hash(
        data: Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none)
      )
      .map { String(format: "%02x", $0) }.joined()
    }
    func unseeded(_ selection: [UInt8], _ acquisition: Int) throws -> String {
      hash(
        try source.detectorImages(
          mask: selection, maximumAdditionalBytes: 1 << 30, rebase: true,
          selectedAcquisitions: [acquisition])[0])
    }
    let budget: UInt64 = 1 << 30
    // Geometry G0..G4: an ADF annulus moving one pixel per step.
    let geometry = (0..<5).map { step in mask(95.5 + Double(step), 95.5, 40, 80) }

    // Start: every acquisition at G0 through one full-series query.
    let start = try source.detectorImages(
      mask: geometry[0], maximumAdditionalBytes: budget, rebase: true)
    XCTAssertEqual(start.count, 66)
    let startHashes = start.map(hash)

    // Interactive tier: acquisition 7 walks G1..G4 alone; each step seeds from its own image.
    var walked: [MTLBuffer] = []
    for step in 1...4 {
      let image = try source.detectorImages(
        mask: geometry[step], maximumAdditionalBytes: budget, selectedAcquisitions: [7])
      XCTAssertEqual(image.count, 1)
      XCTAssertTrue(
        source.lastDetectorUsedPrevious, "step \(step) must reuse acquisition 7's own seed")
      XCTAssertEqual(source.lastDetectorCommandTiming["seed_groups"], 1)
      walked.append(image[0])
    }
    // Ring contract: the buffer returned two queries ago is still intact and distinct.
    XCTAssertFalse(walked[3] === walked[2])
    XCTAssertFalse(walked[3] === walked[1])
    XCTAssertEqual(hash(walked[2]), try unseeded(geometry[3], 7), "previous ring slot untouched")
    XCTAssertEqual(hash(walked[3]), try unseeded(geometry[4], 7), "delta chain on one acquisition")

    // Catch-up tier: a subset of the others jumps straight from G0 to G4 in one seeded group.
    let subset = [0, 1, 2, 3, 65]
    let caught = try source.detectorImages(
      mask: geometry[4], maximumAdditionalBytes: budget, selectedAcquisitions: subset)
    XCTAssertTrue(source.lastDetectorUsedPrevious)
    XCTAssertEqual(source.lastDetectorCommandTiming["seed_groups"], 1)
    for (position, acquisition) in subset.enumerated() {
      XCTAssertEqual(hash(caught[position]), try unseeded(geometry[4], acquisition))
    }

    // Mixed seeds in one request: acquisition 7 (at G4), the subset (at G4) and the
    // untouched remainder (at G0) form separate exact groups but one complete result.
    let everything = try source.detectorImages(mask: geometry[4], maximumAdditionalBytes: budget)
    XCTAssertEqual(everything.count, 66)
    XCTAssertGreaterThanOrEqual(source.lastDetectorCommandTiming["seed_groups"] ?? 0, 2)
    let referenceG4 = try source.detectorImages(
      mask: geometry[4], maximumAdditionalBytes: budget, rebase: true)
    XCTAssertEqual(everything.map(hash), referenceG4.map(hash))

    // Hover switch: 7 -> 33 -> 7 with an unchanged mask must return exact
    // images and must not create work for acquisitions that already match.
    for acquisition in [33, 7, 33] {
      let image = try source.detectorImages(
        mask: geometry[4], maximumAdditionalBytes: budget, selectedAcquisitions: [acquisition])
      XCTAssertEqual(hash(image[0]), referenceG4.map(hash)[acquisition])
      XCTAssertEqual(source.lastDetectorDecodedColumns, 0, "identical mask decodes nothing")
    }

    // Returning to G0 for acquisition 7 alone is a seeded backward step.
    let back = try source.detectorImages(
      mask: geometry[0], maximumAdditionalBytes: budget, selectedAcquisitions: [7])
    XCTAssertEqual(hash(back[0]), startHashes[7])

    // Diffraction subset equals the corresponding full-series diffraction images.
    let fullDP = try source.diffractionImages(scanRow: 300, scanColumn: 17)
    let subsetDP = try source.diffractionImages(
      scanRow: 300, scanColumn: 17, selectedAcquisitions: [65, 7, 0])
    XCTAssertEqual(subsetDP.map(hash), [65, 7, 0].map { fullDP.map(hash)[$0] })
    XCTAssertThrowsError(
      try source.diffractionImages(scanRow: 0, scanColumn: 0, selectedAcquisitions: [1, 1]))
    XCTAssertThrowsError(
      try source.diffractionImages(scanRow: 0, scanColumn: 0, selectedAcquisitions: [99]))

    // A failed query preserves seeds: an over-budget request throws and the
    // next seeded step is still exact.
    XCTAssertThrowsError(
      try source.detectorImages(
        mask: geometry[1], maximumAdditionalBytes: 1, selectedAcquisitions: [7]))
    let after = try source.detectorImages(
      mask: geometry[1], maximumAdditionalBytes: budget, selectedAcquisitions: [7])
    XCTAssertEqual(hash(after[0]), try unseeded(geometry[1], 7))
    print(
      "PER_ACQUISITION_SEED_TEST exact=true ring_slots=\(MetalTANSResidentSeries.detectorImageRingSlots)"
    )
  }
}
