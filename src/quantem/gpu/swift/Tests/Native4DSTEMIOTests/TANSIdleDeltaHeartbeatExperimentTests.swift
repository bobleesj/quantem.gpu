import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Second idle experiment: does touching the inspected acquisition's encoded
/// records (a real one-pixel delta) or holding its records resident keep the
/// first frame after an idle pause fast? Package diagnostic only.
final class TANSIdleDeltaHeartbeatExperimentTests: XCTestCase {
  func testIdleDeltaHeartbeatAndResidencyArmsWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_HEARTBEAT2_FIXTURE"] else {
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
    var col = 95.5
    let acquisition = 5
    func query(_ c: Double) throws -> Double {
      let started = ProcessInfo.processInfo.systemUptime
      _ = try source.detectorImages(
        mask: mask(95.5, c, 40, 80), maximumAdditionalBytes: 1 << 30,
        selectedAcquisitions: [acquisition])
      return (ProcessInfo.processInfo.systemUptime - started) * 1000
    }
    _ = try source.detectorImages(
      mask: mask(95.5, col, 40, 80), maximumAdditionalBytes: 1 << 30, rebase: true,
      selectedAcquisitions: [acquisition])
    for _ in 0..<5 {
      col += 1
      _ = try query(col)
    }
    let arms = ["none", "delta-heartbeat-250ms", "residency-hold", "none-again"]
    for trial in 0..<3 {
      for arm in arms {
        if arm == "residency-hold" {
          try source.beginExperimentalAcquisitionResidencyHold(acquisitions: [acquisition])
        }
        let idleUntil = ProcessInfo.processInfo.systemUptime + 3
        var beats = 0
        var beatWall: [Double] = []
        if arm.hasPrefix("delta-heartbeat") {
          while ProcessInfo.processInfo.systemUptime < idleUntil {
            Thread.sleep(forTimeInterval: 0.25)
            // Real work: alternate one pixel so the encoded records are read.
            let offset = beats % 2 == 0 ? 1.0 : -1.0
            beatWall.append(try query(col + offset))
            beats += 1
          }
          if beats % 2 == 1 { _ = try query(col) }
        } else {
          Thread.sleep(forTimeInterval: 3)
        }
        col += 1
        let first = try query(col)
        col += 1
        let second = try query(col)
        col += 1
        let third = try query(col)
        if arm == "residency-hold" { source.endExperimentalResidencyHold() }
        print(
          "HEARTBEAT2_SAMPLE trial=\(trial) arm=\(arm) beats=\(beats) beat_wall_ms_median=\(beatWall.isEmpty ? -1 : beatWall.sorted()[beatWall.count / 2]) first_ms=\(first) second_ms=\(second) third_ms=\(third) columns=\(source.lastDetectorDecodedColumns) residency_bytes=\(source.experimentalResidencySetBytes) load=\(ProcessInfo.processInfo.systemUptime)"
        )
        fflush(stdout)
      }
    }
  }
}
