import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Third idle experiment: how small can the keep-warm touch be? A one-pixel
/// mask toggle (one or two residual columns) declares and reads the inspected
/// acquisition's encoded records with almost no decode work.
final class TANSIdleTinyHeartbeatExperimentTests: XCTestCase {
  func testTinyDeltaHeartbeatWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_HEARTBEAT3_FIXTURE"] else {
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
    func mask(_ row: Double, _ col: Double, _ inner: Double, _ outer: Double, toggle: Int? = nil)
      -> [UInt8]
    {
      var values = (0..<36864).map { q -> UInt8 in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - col
        let radius2 = r * r + c * c
        return valid[q] != 0 && radius2 >= inner * inner && radius2 <= outer * outer ? 1 : 0
      }
      if let toggle { values[toggle] = valid[toggle] != 0 ? 1 : 0 }
      return values
    }
    var col = 95.5
    let acquisition = 5
    func query(_ values: [UInt8]) throws -> Double {
      let started = ProcessInfo.processInfo.systemUptime
      _ = try source.detectorImages(
        mask: values, maximumAdditionalBytes: 1 << 30, selectedAcquisitions: [acquisition])
      return (ProcessInfo.processInfo.systemUptime - started) * 1000
    }
    _ = try source.detectorImages(
      mask: mask(95.5, col, 40, 80), maximumAdditionalBytes: 1 << 30, rebase: true,
      selectedAcquisitions: [acquisition])
    for _ in 0..<5 {
      col += 1
      _ = try query(mask(95.5, col, 40, 80))
    }
    // Far pixel (row 10, column 10) outside every annulus: toggling it adds one residual column.
    let farPixel = 10 * 192 + 10
    let arms = ["none", "tiny-heartbeat-250ms", "tiny-heartbeat-500ms", "none-again"]
    for trial in 0..<3 {
      for arm in arms {
        let idleUntil = ProcessInfo.processInfo.systemUptime + 3
        var beats = 0
        var beatWall: [Double] = []
        var beatColumns: [Int] = []
        if arm.hasPrefix("tiny") {
          let interval = arm.contains("500") ? 0.5 : 0.25
          while ProcessInfo.processInfo.systemUptime < idleUntil {
            Thread.sleep(forTimeInterval: interval)
            let toggled = beats % 2 == 0
            beatWall.append(try query(mask(95.5, col, 40, 80, toggle: toggled ? farPixel : nil)))
            beatColumns.append(source.lastDetectorDecodedColumns)
            beats += 1
          }
          if beats % 2 == 1 { _ = try query(mask(95.5, col, 40, 80)) }
        } else {
          Thread.sleep(forTimeInterval: 3)
        }
        col += 1
        let first = try query(mask(95.5, col, 40, 80))
        let firstColumns = source.lastDetectorDecodedColumns
        col += 1
        let second = try query(mask(95.5, col, 40, 80))
        col += 1
        let third = try query(mask(95.5, col, 40, 80))
        print(
          "HEARTBEAT3_SAMPLE trial=\(trial) arm=\(arm) beats=\(beats) beat_wall_ms_median=\(beatWall.isEmpty ? -1 : beatWall.sorted()[beatWall.count / 2]) beat_columns=\(beatColumns.first ?? -1) first_ms=\(first) first_columns=\(firstColumns) second_ms=\(second) third_ms=\(third)"
        )
        fflush(stdout)
      }
    }
  }
}
