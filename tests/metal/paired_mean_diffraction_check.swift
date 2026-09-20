import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels
@_spi(PairedRuntimeTANSPrototype) @testable import Metal4DSTEMStreamingIO
import Native4DSTEMIO

@main
enum PairedMeanCheck {
  static func main() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      fatalError("A Metal device is required")
    }
    let codec = try MetalPairedRuntimeTANSSyntheticCodec(device: device)
    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    func buffer<T>(_ values: [T]) -> MTLBuffer {
      values.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
      }
    }
    // Tiny independent count oracle exercises zero, constants, sparse events,
    // entropy, literal data and uint16 escape values without a full CPU cube.
    let streams: [[UInt16]] = (0..<32).map { pixel in
      (0..<512).map { scan in
        switch pixel % 6 {
        case 0: return 0
        case 1: return 65_535
        case 2: return scan == 511 ? 128 : 0
        case 3: return UInt16((scan * 17 + scan / 7) % 11)
        case 4: return UInt16(truncatingIfNeeded: scan * 197 + pixel * 719)
        default: return scan % 97 == 0 ? 32_768 : UInt16(scan % 3)
        }
      }
    }
    let encoded = try codec.roundTrip(streams: streams, logicalDtype: .uint16)
    let decoding = buffer(try PairedRuntimeTANSTables.build().packedDecoding)
    for compact in [false, true] {
      for packetCount in [2, 512] {
        var offsets: [UInt32] = []
        var payload: [UInt8] = []
        for _ in 0..<packetCount {
          offsets += encoded.offsets.dropLast().map { $0 + UInt32(payload.count) }
          payload += encoded.payload
        }
        offsets.append(UInt32(payload.count))
        // The decoder's bounded word reader permits eight padding bytes.
        payload += [UInt8](repeating: 0, count: 8)
        let directory: MTLBuffer
        if compact {
          let bases = stride(from: 0, through: offsets.count - 1, by: 32).map { offsets[$0] }
          let deltas = offsets.indices.map { UInt16(offsets[$0] - offsets[($0 / 32) * 32]) }
          var bytes = bases.withUnsafeBytes { Array($0) }
          bytes += deltas.withUnsafeBytes { Array($0) }
          directory = buffer(bytes)
        } else {
          directory = buffer(offsets)
        }
        let payloadBuffer = buffer(payload)
        let modes = buffer(Array(repeating: encoded.modes, count: packetCount).flatMap { $0 })
        let reducer = try PairedRuntimeMeanDiffraction(
          device: device, library: library, compactOffsets: compact)
        let scanRows = packetCount * 2
        let cases: [(Range<Int>, Range<Int>, MetalScanRegionShape)] =
          packetCount == 2
          ? [
            (0..<4, 0..<256, .rectangle), (1..<3, 2..<255, .rectangle),
            (0..<4, 0..<4, .circle), (3..<4, 255..<256, .rectangle),
          ]
          : [(0..<scanRows, 0..<256, .rectangle)]
        let permutation = (0..<32).reversed().map(UInt32.init)
        for (rows, columns, selection) in cases {
          let result = try reducer.mean(
            queue: queue, payload: payloadBuffer, offsets: directory,
            modes: modes, table: decoding, shape: [scanRows, 256, 4, 8],
            pixelOfStreamRank: permutation, rows: rows, columns: columns, regionShape: selection)
          var expected = [UInt64](repeating: 0, count: 32)
          for rank in 0..<32 {
            if packetCount == 512 {
              expected[Int(permutation[rank])] =
                streams[rank].reduce(UInt64(0)) { $0 + UInt64($1) } * 512
            } else {
              for row in rows {
                for column in columns {
                  let dr = 2 * row + 1 - rows.lowerBound - rows.upperBound
                  let dc = 2 * column + 1 - columns.lowerBound - columns.upperBound
                  if selection == .rectangle || dr * dr + dc * dc <= rows.count * rows.count {
                    expected[Int(permutation[rank])] += UInt64(
                      streams[rank][(row * 256 + column) % 512])
                  }
                }
              }
            }
          }
          let count = selection.sampleCount(rowCount: rows.count, columnCount: columns.count)
          precondition(result.detectorSum == expected, "Exact region sums differ")
          precondition(
            result.mean == expected.map { Float(Double($0) / Double(count)) }, "Means differ")
          print(
            "PASS paired mean compact=\(compact) packets=\(packetCount) region=\(selection) gpu_ms=\(result.gpuMilliseconds)"
          )
        }
      }
    }
    if CommandLine.arguments.count == 3 {
      let catalog = try Native4DSTEMCatalogBuilder(
        cacheDirectory: URL(fileURLWithPath: CommandLine.arguments[2])
      )
      .prepare(input: URL(fileURLWithPath: CommandLine.arguments[1]))
      precondition(catalog.datasets.count == 1, "Choose one acquisition")
      let indexed = try Native4DSTEMIndexedSource.open(dataset: catalog.datasets[0])
      let actual = try MetalPairedRuntimeTANSResidentSource.load(source: indexed, device: device)
      let reference = try MetalRuntimeANSResidentSource.load(source: indexed, device: device)
      defer {
        actual.releaseResidentStorage()
        reference.releaseResidentStorage()
      }
      for (rows, columns, region) in [
        (0..<512, 0..<512, MetalScanRegionShape.rectangle),
        (249..<265, 249..<265, .circle), (0..<7, 501..<512, .rectangle),
      ] {
        let expected = try reference.meanDiffractionPattern(
          rows: rows, columns: columns, shape: region)
        let result = try actual.meanDiffractionPattern(rows: rows, columns: columns, shape: region)
        precondition(
          result.detectorSum == expected.detectorSum && result.mean == expected.mean,
          "Paired mean differs from independent runtime-ANS GPU reference")
        print(
          "PASS real paired mean rows=\(rows.count) columns=\(columns.count) shape=\(region) wall_ms=\(result.wallMilliseconds) gpu_ms=\(result.gpuMilliseconds)"
        )
      }
    }
  }
}
