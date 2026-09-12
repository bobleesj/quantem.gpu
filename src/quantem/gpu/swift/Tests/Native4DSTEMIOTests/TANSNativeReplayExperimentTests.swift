import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Same masks as retained native gestures; package diagnostics, not native FPS.
final class TANSNativeReplayExperimentTests: XCTestCase {
  private struct Request: Decodable {
    let geometry: [String: Double]
    let columns: Int
    let previous: Bool
    let probes: String
    let maskSha256: String
  }
  private struct Replay: Decodable { let rows: [Request] }

  func testAll66NativeMasksActiveIdleActiveWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let path = environment["QUANTEM_TANS_REPLAY_FIXTURE"],
      let replayPath = environment["QUANTEM_TANS_REPLAY_MASKS"]
    else { throw XCTSkip("Requires complete all66 fixture and recorded native mask metadata") }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let replay = try decoder.decode(
      Replay.self, from: Data(contentsOf: URL(fileURLWithPath: replayPath)))
    XCTAssertEqual(replay.rows.count, 9)
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.releaseResidentStorage() }
    let sourceBytes = source.residentBytes
    let prepareStart = ProcessInfo.processInfo.systemUptime
    try source.prepareExperimentalTileIndex(maximumIndexBytes: 2 << 30)
    source.experimentalUseTileIndex = true
    source.experimentalDetectorStreamsPerLane = 32
    print(
      "NATIVE_REPLAY_LOAD acquisitions=66 source_bytes=\(sourceBytes) index_bytes=\(source.experimentalTileIndexBytes) load_s=\(source.loadSeconds) index_s=\(ProcessInfo.processInfo.systemUptime-prepareStart) source_pages=unspecified route=package_not_headed"
    )
    fflush(stdout)
    func hash(_ data: Data) -> String {
      SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    func hashes(_ images: [MTLBuffer]) -> [String] {
      images.map { hash(Data(bytesNoCopy: $0.contents(), count: $0.length, deallocator: .none)) }
    }
    func mask(row: Double, col: Double, inner: Double, outer: Double) -> [UInt8] {
      (0..<36864).map { q in
        let radius2 = pow(Double(q / 192) - row, 2) + pow(Double(q % 192) - col, 2)
        return source.validDetectorMask[q] != 0 && radius2 >= inner * inner
          && radius2 <= outer * outer ? 1 : 0
      }
    }
    let masks = try replay.rows.map { request in
      let geometry = request.geometry
      let values = try mask(
        row: XCTUnwrap(geometry["center-row"]), col: XCTUnwrap(geometry["center-column"]),
        inner: XCTUnwrap(geometry["inner"]), outer: XCTUnwrap(geometry["outer"]))
      XCTAssertEqual(hash(Data(values)), request.maskSha256)
      return values
    }
    let frozen = try Data(
      contentsOf: URL(fileURLWithPath: path).appendingPathComponent(
        "series-products/shared-resident-images.npy"), options: .mappedIfSafe)
    XCTAssertEqual(hash(frozen), "06b96168b62d89ba5a893416dcf7b0000d34647310da91db243ceab945cf89ac")
    let offset = 10 + Int(frozen[8]) + (Int(frozen[9]) << 8)
    XCTAssertEqual(frozen.count - offset, 66 * 4 * 512 * 512 * 4)
    for (product, inner, outer) in [(0, 0.0, 28.0), (1, 40.0, 80.0), (2, 14.0, 28.0)] {
      try autoreleasepool {
        let images = try source.detectorImages(
          mask: mask(row: 95.5, col: 95.5, inner: inner, outer: outer),
          maximumAdditionalBytes: 1 << 30, rebase: true)
        for acquisition in 0..<66 {
          let start = offset + (acquisition * 4 + product) * 512 * 512 * 4
          let image = images[acquisition]
          XCTAssertEqual(
            Data(bytesNoCopy: image.contents(), count: image.length, deallocator: .none),
            frozen.subdata(in: start..<(start + image.length)))
        }
      }
    }
    let referenceHashes = try masks.enumerated().map { index, mask in
      try autoreleasepool {
        let images = try source.detectorImages(
          mask: mask, maximumAdditionalBytes: 1 << 30, rebase: true)
        let nativeProbes = replay.rows[index].probes.split(separator: ";")
        XCTAssertEqual(nativeProbes.count, 66)
        for acquisition in 0..<66 {
          let entry = nativeProbes[acquisition].split(separator: ":")
          XCTAssertEqual(Int(entry[0]), acquisition)
          let expected = entry[1].split(separator: ",").map { UInt32($0)! }
          for (point, scan) in [0, 131328, 262143].enumerated() {
            XCTAssertEqual(
              images[acquisition].contents().load(fromByteOffset: scan * 4, as: UInt32.self),
              expected[point])
          }
        }
        return hashes(images)
      }
    }
    for (arm, idleSeconds) in [("A1", 0.0), ("B_idle", 4.0), ("A2", 0.0)] {
      for cycle in 0..<3 {
        for index in masks.indices {
          try autoreleasepool {
            if idleSeconds > 0 { Thread.sleep(forTimeInterval: idleSeconds) }
            let started = ProcessInfo.processInfo.systemUptime
            let images = try source.detectorImages(
              mask: masks[index], maximumAdditionalBytes: 1 << 30, rebase: index == 0)
            let wall = (ProcessInfo.processInfo.systemUptime - started) * 1000
            let gpu = source.lastDetectorGPUSeconds * 1000
            XCTAssertEqual(images.count, 66)
            XCTAssertEqual(hashes(images), referenceHashes[index])
            XCTAssertEqual(source.lastDetectorDecodedColumns, replay.rows[index].columns)
            XCTAssertEqual(source.lastDetectorUsedPrevious, replay.rows[index].previous)
            print(
              "NATIVE_REPLAY_SAMPLE arm=\(arm) cycle=\(cycle) case=\(index) idle_s=\(idleSeconds) wall_ms=\(wall) gpu_ms=\(gpu) columns=\(source.lastDetectorDecodedColumns) previous=\(source.lastDetectorUsedPrevious) allocated_bytes=\(device.currentAllocatedSize) source_bytes=\(sourceBytes) index_bytes=\(source.experimentalTileIndexBytes) images=66 exact=true"
            )
            fflush(stdout)
          }
        }
      }
    }
  }
}
