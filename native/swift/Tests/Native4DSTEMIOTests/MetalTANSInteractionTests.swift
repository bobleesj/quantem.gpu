import CryptoKit
import Foundation
import Metal
import MetalDisplayKernels
import MetalImageRuntime
import XCTest

@testable import Metal4DSTEMStreamingIO

final class MetalTANSInteractionTests: XCTestCase {
  func testRealAll66SustainedDetectorGesturesWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_GESTURE_FIXTURE"] else {
      throw XCTSkip("Requires all 66 exact entropy acquisitions; sustained GPU gesture benchmark")
    }
    let root = URL(fileURLWithPath: path)
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let archive = try TANSArchive(directory: root, acquisitions: Array(0..<66))
    let valid = try XCTUnwrap(archive.arrays["planner__valid"])
    let source = try MetalTANSResidentSeries(
      directory: root, acquisitions: Array(0..<66),
      device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.releaseResidentStorage() }
    func hashes(_ images: [MTLBuffer]) -> [String] {
      images.map { buffer in
        SHA256.hash(
          data: Data(
            bytesNoCopy: buffer.contents(), count: buffer.length,
            deallocator: .none)
        ).map { String(format: "%02x", $0) }.joined()
      }
    }
    func percentile(_ values: [Double], _ p: Double) -> Double {
      let sorted = values.sorted()
      return sorted[min(sorted.count - 1, max(0, Int(ceil(p * Double(sorted.count))) - 1))]
    }
    print(
      "GESTURE_RESIDENT acquisitions=66 shape=66x512x512x192x192 dtype=uint16 representation=entropy scan_bin=1 detector_bin=1 crop=none resident_bytes=\(source.residentBytes) source_pages=unspecified load_seconds=\(source.loadSeconds) route=package_not_headed"
    )
    fflush(stdout)
    for (name, inner, outer) in [("BF", 0.0, 28.0), ("ABF", 14.0, 28.0), ("ADF", 40.0, 80.0)] {
      // A continuous loop with direction reversals and simultaneous center and
      // outer-radius movement. Every timed step publishes all 66 full images.
      let masks: [[UInt8]] = (0..<32).map { step in
        let angle = Double(step) * 2 * .pi / 31
        let row = 95.5 + 6 * sin(angle)
        let col = 95.5 + 6 * cos(angle)
        let radius = outer + 3 * sin(2 * angle)
        return (0..<36864).map { q in
          let r = Double(q / 192) - row
          let c = Double(q % 192) - col
          let distance = r * r + c * c
          return valid[q] != 0 && distance >= inner * inner && distance <= radius * radius ? 1 : 0
        }
      }
      let references = try masks.map { mask in
        try autoreleasepool {
          hashes(
            try source.detectorImages(mask: mask, maximumAdditionalBytes: 1 << 30, rebase: true))
        }
      }
      var walls: [Double] = []
      var gpu: [Double] = []
      for cycle in 0..<3 {
        try autoreleasepool {
          _ = try source.detectorImages(
            mask: masks[0], maximumAdditionalBytes: 1 << 30, rebase: true)
        }
        for step in 1..<masks.count {
          try autoreleasepool {
            let started = ProcessInfo.processInfo.systemUptime
            let images = try source.detectorImages(
              mask: masks[step], maximumAdditionalBytes: 1 << 30)
            let wall = (ProcessInfo.processInfo.systemUptime - started) * 1000
            walls.append(wall)
            gpu.append(source.lastDetectorGPUSeconds * 1000)
            XCTAssertEqual(images.count, 66)
            XCTAssertEqual(
              hashes(images), references[step],
              "\(name) cycle \(cycle) step \(step): all 66 exact full images")
            XCTAssertTrue(source.lastDetectorUsedPrevious)
            print(
              "GESTURE_SAMPLE product=\(name) cycle=\(cycle) step=\(step) wall_ms=\(wall) gpu_ms=\(source.lastDetectorGPUSeconds * 1000) changed_columns=\(source.lastDetectorDecodedColumns) images=66 exact=true"
            )
            fflush(stdout)
          }
        }
      }
      print(
        "GESTURE_SUMMARY product=\(name) n=\(walls.count) wall_p50_ms=\(percentile(walls, 0.5)) wall_p95_ms=\(percentile(walls, 0.95)) wall_max_ms=\(walls.max()!) gpu_p50_ms=\(percentile(gpu, 0.5)) gpu_p95_ms=\(percentile(gpu, 0.95)) images_per_update=66 exact=true allocated_bytes=\(device.currentAllocatedSize) route=package_not_headed"
      )
      fflush(stdout)
    }
  }

  func testRealDetectorImagesMatchFrozenLinuxProductsWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_DETECTOR_FIXTURE"] else {
      throw XCTSkip("Requires complete source and frozen Linux detector products")
    }
    let root = URL(fileURLWithPath: path)
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let metadata = try TANSArchive(directory: root, acquisitions: Array(0..<66))
    let valid = try XCTUnwrap(metadata.arrays["planner__valid"])
    let source = try MetalTANSResidentSeries(
      directory: root, acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.releaseResidentStorage() }
    let frozen = try Data(
      contentsOf: root.appendingPathComponent("series-products/shared-resident-images.npy"),
      options: .mappedIfSafe)
    XCTAssertEqual(
      SHA256.hash(data: frozen).map { String(format: "%02x", $0) }.joined(),
      "06b96168b62d89ba5a893416dcf7b0000d34647310da91db243ceab945cf89ac")
    let headerLength = Int(frozen[8]) | (Int(frozen[9]) << 8)
    let offset = 10 + headerLength
    XCTAssertEqual(frozen.count - offset, 66 * 4 * 512 * 512 * 4)
    XCTAssertThrowsError(try source.detectorImages(mask: [], maximumAdditionalBytes: 1 << 30))
    XCTAssertThrowsError(
      try source.detectorImages(
        mask: Array(valid), maximumAdditionalBytes: 1 << 30,
        rebase: false, selectedAcquisitions: [0, 0]))
    XCTAssertThrowsError(
      try source.detectorImages(
        mask: Array(valid), maximumAdditionalBytes: 1 << 30,
        rebase: false, selectedAcquisitions: []))
    XCTAssertThrowsError(
      try source.detectorImages(
        mask: [UInt8](repeating: 1, count: 36864), maximumAdditionalBytes: 1))
    for (product, name, inner, outer) in [
      (0, "BF", 0.0, 28.0), (1, "ADF", 40.0, 80.0), (2, "ABF", 14.0, 28.0),
    ] {
      let mask: [UInt8] = (0..<36864).map { q in
        let row = Double(q / 192) - 95.5
        let col = Double(q % 192) - 95.5
        let r2 = row * row + col * col
        return valid[q] != 0 && r2 >= inner * inner && r2 <= outer * outer ? 1 : 0
      }
      let started = ProcessInfo.processInfo.systemUptime
      let images = try source.detectorImages(
        mask: mask, maximumAdditionalBytes: 1 << 30, rebase: true)
      let wall = ProcessInfo.processInfo.systemUptime - started
      for index in 0..<66 {
        let start = offset + (index * 4 + product) * 512 * 512 * 4
        let actual = Data(
          bytesNoCopy: images[index].contents(), count: images[index].length, deallocator: .none)
        XCTAssertTrue(
          actual == frozen.subdata(in: start..<(start + actual.count)),
          "\(name) acquisition \(index) complete image mismatch")
      }
      print(
        "SPEED_DETECTOR product=\(name) wall_ms=\(wall*1000) gpu_ms=\(source.lastDetectorGPUSeconds*1000) columns=\(mask.filter{$0 != 0}.count) scratch_bytes=\(source.lastDetectorScratchBytes) images=66 source=entropy full_scan=true bin=1 crop=none exact=true"
      )
      fflush(stdout)
    }
    func digest(_ images: [MTLBuffer]) -> [String] {
      images.map { buffer in
        SHA256.hash(
          data: Data(
            bytesNoCopy: buffer.contents(),
            count: buffer.length, deallocator: .none)
        ).map { String(format: "%02x", $0) }.joined()
      }
    }
    func aperture(_ row: Double, _ col: Double, _ inner: Double, _ outer: Double) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - col
        let d = r * r + c * c
        return valid[q] != 0 && d >= inner * inner && d <= outer * outer ? 1 : 0
      }
    }
    for (name, inner, outer) in [("BF", 0.0, 28.0), ("ADF", 40.0, 80.0)] {
      let base = aperture(95.5, 95.5, inner, outer)
      for (motion, row, col, dr) in [
        ("fractional", 95.625, 95.5, 0.0),
        ("center", 96.5, 96.5, 0.0), ("radius", 95.5, 95.5, 1.0),
      ] {
        let moved = aperture(row, col, inner, outer + dr)
        let reference = try source.detectorImages(
          mask: moved, maximumAdditionalBytes: 1 << 30, rebase: true)
        let expected = digest(reference)
        for cycle in 0..<3 {
          let previous = try source.detectorImages(mask: base, maximumAdditionalBytes: 1 << 30)
          let priorHash = digest(previous)
          XCTAssertThrowsError(try source.detectorImages(mask: moved, maximumAdditionalBytes: 1))
          let started = ProcessInfo.processInfo.systemUptime
          let changed = try source.detectorImages(mask: moved, maximumAdditionalBytes: 1 << 30)
          let wall = ProcessInfo.processInfo.systemUptime - started
          XCTAssertEqual(
            digest(changed), expected, "Exact \(name) \(motion) delta, complete 66-image parity")
          XCTAssertEqual(
            digest(previous), priorHash, "Prior published images must remain immutable")
          print(
            "SPEED_DELTA product=\(name) motion=\(motion) cycle=\(cycle) wall_ms=\(wall*1000) gpu_ms=\(source.lastDetectorGPUSeconds*1000) columns=\(source.lastDetectorDecodedColumns) previous=\(source.lastDetectorUsedPrevious) scratch_bytes=\(source.lastDetectorScratchBytes) exact=66"
          )
          fflush(stdout)
        }
      }
    }
    let zero = try source.detectorImages(
      mask: [UInt8](repeating: 0, count: 36864), maximumAdditionalBytes: 1 << 30)
    XCTAssertTrue(
      zero.allSatisfy { buffer in
        UnsafeBufferPointer(
          start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: 512 * 512
        ).allSatisfy { $0 == 0 }
      })
  }

  func testRealSeriesQueryABBAAndBatchStatisticsWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_INTERACTION_FIXTURE"] else {
      throw XCTSkip("Requires the complete exact entropy series; opt-in memory benchmark")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path),
      acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.releaseResidentStorage() }
    print(
      "SPEED_LOAD seconds=\(source.loadSeconds) bytes=\(source.residentBytes) allocated=\(device.currentAllocatedSize) source_pages=unspecified source=prepared_entropy"
    )
    let locations = [(0, 0), (17, 63), (256, 128), (392, 255), (511, 404), (511, 511)]
    func hashes(_ buffers: [MTLBuffer]) -> [String] {
      buffers.map { buffer in
        SHA256.hash(
          data: Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none)
        )
        .map { String(format: "%02x", $0) }.joined()
      }
    }
    var references: [[String]] = []
    for (row, column) in locations {
      let images = try source.diffractionImages(scanRow: row, scanColumn: column)
      references.append(hashes(images))
    }
    for (arm, word32, width) in [
      ("A1", false, 128), ("B32", true, 32), ("B64", true, 64),
      ("B128", true, 128), ("B256", true, 256), ("A2", false, 128),
    ] {
      source.useWord32Query = word32
      source.queryThreadgroupWidth = width
      for cycle in 0..<3 {
        for (index, location) in locations.enumerated() {
          let started = ProcessInfo.processInfo.systemUptime
          let images = try source.diffractionImages(scanRow: location.0, scanColumn: location.1)
          let wall = ProcessInfo.processInfo.systemUptime - started
          XCTAssertEqual(
            hashes(images), references[index], "\(arm) \(location) exact full DP hashes")
          print(
            "SPEED_QUERY arm=\(arm) cycle=\(cycle) row=\(location.0) col=\(location.1) wall_ms=\(wall*1000) gpu_ms=\(source.lastQueryGPUSeconds*1000) exact=66"
          )
        }
      }
    }
    let statistics = try MetalDisplayStatistics(device: device)
    let images = try source.diffractionImages(scanRow: 256, scanColumn: 256)
    let scales: [MetalDisplayScale] = [.linear, .logarithmic]
    var serialReference: [[MetalUInt32Statistics]] = []
    for arm in ["serialA1", "batchB", "serialA2"] {
      for cycle in 0..<5 {
        let started = ProcessInfo.processInfo.systemUptime
        let result =
          arm == "batchB"
          ? try statistics.analyzeUInt32Batch(
            values: images, rows: 192, columns: 192, scales: scales)
          : try images.map { image in
            try scales.map { scale in
              try statistics.analyzeUInt32(values: image, rows: 192, columns: 192, scale: scale)
            }
          }
        let wall = ProcessInfo.processInfo.systemUptime - started
        if serialReference.isEmpty { serialReference = result }
        for index in images.indices {
          for j in scales.indices {
            XCTAssertEqual(result[index][j].minimum, serialReference[index][j].minimum)
            XCTAssertEqual(result[index][j].maximum, serialReference[index][j].maximum)
            XCTAssertEqual(result[index][j].bins, serialReference[index][j].bins)
          }
        }
        print(
          "SPEED_STATS arm=\(arm) cycle=\(cycle) wall_ms=\(wall*1000) images=66 scales=2 exact=true"
        )
      }
    }
    source.releaseResidentStorage()
    print("SPEED_RELEASE bytes=\(source.residentBytes) allocated=\(device.currentAllocatedSize)")
  }

  func testRealSelectedDetectorImageMatchesFrozenProductWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_DETECTOR_FIXTURE"] else {
      throw XCTSkip("Requires complete source and frozen Linux detector products")
    }
    let root = URL(fileURLWithPath: path)
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let archive = try TANSArchive(directory: root, acquisitions: Array(0..<66))
    let valid = try XCTUnwrap(archive.arrays["planner__valid"])
    let source = try MetalTANSResidentSeries(
      directory: root, acquisitions: Array(0..<66),
      device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.releaseResidentStorage() }
    let frozen = try Data(
      contentsOf: root.appendingPathComponent("series-products/shared-resident-images.npy"),
      options: .mappedIfSafe)
    let headerLength = Int(frozen[8]) | (Int(frozen[9]) << 8)
    let offset = 10 + headerLength
    for (product, name, inner, outer) in [
      (0, "BF", 0.0, 28.0),
      (1, "ADF", 40.0, 80.0), (2, "ABF", 14.0, 28.0),
    ] {
      let mask: [UInt8] = (0..<36864).map { q in
        let row = Double(q / 192) - 95.5
        let col = Double(q % 192) - 95.5
        let radius2 = row * row + col * col
        return valid[q] != 0 && radius2 >= inner * inner && radius2 <= outer * outer ? 1 : 0
      }
      let started = ProcessInfo.processInfo.systemUptime
      let images = try source.detectorImages(
        mask: mask, maximumAdditionalBytes: 1 << 30,
        rebase: true, selectedAcquisitions: [0])
      let wall = ProcessInfo.processInfo.systemUptime - started
      XCTAssertEqual(images.count, 1)
      let expectedStart = offset + product * 512 * 512 * 4
      let expected = frozen.subdata(in: expectedStart..<(expectedStart + 512 * 512 * 4))
      XCTAssertEqual(
        Data(bytesNoCopy: images[0].contents(), count: images[0].length, deallocator: .none),
        expected)
      print(
        "SPEED_SELECTED product=\(name) wall_ms=\(wall * 1000) gpu_ms=\(source.lastDetectorGPUSeconds * 1000) columns=\(source.lastDetectorDecodedColumns) images=1 exact=true"
      )
    }
  }
}
