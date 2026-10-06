import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

/// Independent raw-array oracle: every detector pixel, with direct scan membership.
@main struct RegionMeanDiffractionCheck {
  static func main() throws {
    setbuf(stdout, nil)
    let args = CommandLine.arguments
    if args.count > 1 { try run(args); return }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    for width in [1, 2] {
      let url = folder.appendingPathComponent("counts\(width).npy")
      var header = "{'descr': '\(width == 1 ? "|u1" : "<u2")', 'fortran_order': False, 'shape': (17, 33, 4, 5), }"
      header += String(repeating: " ", count: (64 - (10 + header.utf8.count + 1) % 64) % 64) + "\n"
      var data = Data([0x93, 78, 85, 77, 80, 89, 1, 0, UInt8(header.utf8.count & 255), UInt8(header.utf8.count >> 8)])
      data.append(contentsOf: header.utf8)
      for index in 0..<(17 * 33 * 4 * 5) {
        let pixel = index % 20, position = (index / 20) % 512
        let value: Int
        switch pixel {
        case 0: value = 0
        case 1: value = width == 1 ? 255 : 65535
        case 2: value = position == 17 ? 5 : 0
        case 3: value = position == 10 || position == 27 ? 128 : 0
        default: value = (index * 37) % (width == 1 ? 256 : 65536)
        }
        data.append(UInt8(value & 255))
        if width == 2 { data.append(UInt8(value >> 8)) }
      }
      try data.write(to: url)
      try run([args[0], url.path])
    }
  }

  static func run(_ args: [String]) throws {
    guard args.count >= 2, let device = MTLCreateSystemDefaultDevice() else {
      fatalError("Supply original.dm4 or counts.npy and optionally a compressed copy")
    }
    let url = URL(fileURLWithPath: args[1])
    let original: any NativeCountArray = url.pathExtension == "dm4"
      ? try NativeDM4Source(url: url) : try NativeNPYSource(url: url)
    let source = args.count > 2
      ? try MetalRuntimeANSResidentSource.load(snapshot: NativeANSSnapshot(url: URL(fileURLWithPath: args[2])), device: device)
      : try MetalRuntimeANSResidentSource.load(array: original, device: device)
    defer { source.releaseResidentStorage() }
    let shape = original.shape, pixels = shape[2] * shape[3]
    precondition(shape == source.shape && shape[0] >= 8 && shape[1] >= 8)
    let raw = try Data(contentsOf: url, options: .alwaysMapped)
    let itemBytes = original.dataset.sourceDtype == "uint8" ? 1 : 2
    var cases: [(Range<Int>, Range<Int>, MetalScanRegionShape)] = [
      (0..<1, 0..<1, .rectangle), (1..<3, 2..<7, .rectangle),
      (2..<8, 2..<8, .circle),
      (shape[0]-5..<shape[0], shape[1]-5..<shape[1], .circle),
      (2..<8, 2..<8, .rectangle), (0..<1, 0..<1, .circle),
    ]
    if pixels <= 1024 { cases.append((0..<shape[0], 0..<shape[1], .rectangle)) }
    if shape[0] >= 40 && shape[1] >= 40 {
      cases += [(10..<22, 11..<23, .circle), (12..<38, 13..<39, .rectangle),
        (12..<38, 13..<39, .circle)]
    }
    let originalResidentBytes = source.residentBytes
    print("device=\(device.name), shape=\(shape), dtype=\(original.dataset.sourceDtype) resident_bytes=\(originalResidentBytes)")
    for (rows, columns, regionShape) in cases {
      var expected = [UInt64](repeating: 0, count: pixels), count = 0
      raw.withUnsafeBytes { bytes in
        for row in rows { for column in columns {
          let rowOffset = Double(row) + 0.5 - Double(rows.lowerBound + rows.upperBound) / 2
          let colOffset = Double(column) + 0.5 - Double(columns.lowerBound + columns.upperBound) / 2
          if regionShape == .circle && rowOffset * rowOffset + colOffset * colOffset > pow(Double(rows.count) / 2, 2) { continue }
          count += 1
          let start = original.dataOffset + (row * shape[1] + column) * pixels * itemBytes
          for pixel in 0..<pixels {
            let offset = start + pixel * itemBytes
            expected[pixel] += itemBytes == 1 ? UInt64(bytes[offset])
              : UInt64(UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
          }
        } }
      }
      for repeatIndex in 0..<2 {
        let result = try source.meanDiffractionPattern(rows: rows, columns: columns, shape: regionShape)
        precondition(result.detectorSum == expected, "Exact raw-count regional sums differ")
        precondition(result.mean == expected.map { Float(Double($0) / Double(count)) }, "Float32 means differ")
        print("PASS \(regionShape) rows=\(rows) columns=\(columns) samples=\(count) repeat=\(repeatIndex) gpu_ms=\(result.gpuMilliseconds) wall_ms=\(result.wallMilliseconds)")
      }
    }
    let caps = try Metal4DSTEMResidentCapabilities.runtimeANS(source)
    print("PASS exact sums and correctly rounded means for every detector pixel; capabilities=\(caps.products.count)")
    if ProcessInfo.processInfo.environment["QGPU_REGION_BENCHMARK"] == "1" {
      for diameter in [6, 12, 26] where diameter < min(shape[0], shape[1]) {
        for selection in [MetalScanRegionShape.rectangle, .circle] {
          var wall: [Double] = [], gpu: [Double] = []
          for step in 0..<22 {
            let offset = 10 + step % 5
            let result = try source.meanDiffractionPattern(
              rows: offset..<offset + diameter, columns: offset..<offset + diameter,
              shape: selection)
            if step >= 2 { wall.append(result.wallMilliseconds); gpu.append(result.gpuMilliseconds) }
          }
          print("BENCH shape=\(selection) diameter=\(diameter) wall_ms=\(wall) gpu_ms=\(gpu)")
        }
      }
    }
    print("MEMORY query_cache_bytes=\(source.residentBytes - originalResidentBytes) allocated_bytes=\(device.currentAllocatedSize)")
    source.releaseResidentStorage()
    precondition(source.residentBytes == 0)
    print("RELEASE_PASS")
  }
}
