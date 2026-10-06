import Foundation
import Metal
import Metal4DSTEMStreamingIO
import MetalScientificNumerics

/// Masked exact reductions on a calibrated scaled-uint16 result versus the full
/// restore-and-weight traversal. Usage: scaled-masked-sums <result.qem> [dump-dir]
/// A dump directory receives the BF r48 image and five restored frames as raw
/// little-endian float32 for cross-language parity checks.
@main
struct ScaledMaskedSums {
  static func main() throws {
    guard CommandLine.arguments.count > 1, let device = MTLCreateSystemDefaultDevice() else {
      print("usage: scaled-masked-sums result.qem")
      return
    }
    var started = Date()
    let source = try MetalPackedSource.loadQEM(
      url: URL(fileURLWithPath: CommandLine.arguments[1]), device: device)
    print(
      String(
        format: "loadQEM %.2f s, %.2f GiB resident", Date().timeIntervalSince(started),
        Double(source.residentBytes) / 1_073_741_824))
    let operations = try MetalImageOperations()
    let rows = source.shape[2]
    let columns = source.shape[3]
    func mask(_ centerRow: Double, _ centerColumn: Double, _ inner: Double, _ outer: Double)
      -> [Int]
    {
      (0..<(rows * columns)).filter { pixel in
        let dr = Double(pixel / columns) - centerRow
        let dc = Double(pixel % columns) - centerColumn
        let d = dr * dr + dc * dc
        return d >= inner * inner && d < outer * outer
      }
    }
    let cases: [(String, [Int], Bool)] = [
      ("off-axis disk r18", mask(70, 120, 0, 18), true),
      ("BF disk r48", mask(95.5, 95.5, 0, 48), true),
      ("ADF annulus 48-96", mask(95.5, 95.5, 48, 96), false),
      ("full detector", Array(0..<(rows * columns)), false),
    ]
    let frames = source.shape[0] * source.shape[1]
    if CommandLine.arguments.count > 2 {
      let directory = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
      func write(_ buffer: MTLBuffer, _ count: Int, _ name: String) throws {
        try Data(bytes: buffer.contents(), count: count * 4).write(
          to: directory.appendingPathComponent(name), options: .withoutOverwriting)
      }
      try write(try source.maskedVirtualImage(detectorPixels: cases[1].1), frames, "bf_r48.f32")
      var patterns = Data()
      for index in [0, 4095, 4096, frames / 2 + source.shape[1] / 2, frames - 1] {
        let frame = try source.read(index..<(index + 1))
        patterns.append(Data(bytes: frame.contents(), count: rows * columns * 4))
      }
      try patterns.write(
        to: directory.appendingPathComponent("frames.f32"), options: .withoutOverwriting)
      print("dumped BF r48 and 5 frames to \(directory.path)")
    }
    for (name, pixels, compare) in cases {
      var times = [Double]()
      var image: MTLBuffer?
      for _ in 0..<3 {
        started = Date()
        image = try source.maskedVirtualImage(detectorPixels: pixels)
        times.append(Date().timeIntervalSince(started))
      }
      var line = String(
        format: "%-18@ %6d px  masked %.3f/%.3f/%.3f s", name as NSString, pixels.count,
        times[0], times[1], times[2])
      if compare {
        started = Date()
        var weights = [Float](repeating: 0, count: rows * columns)
        for pixel in pixels { weights[pixel] = 1 }
        let reference = try operations.virtualImage(source: source, weights: weights)
        let full = Date().timeIntervalSince(started)
        let a = image!.contents().assumingMemoryBound(to: Float.self)
        let b = reference.buffer.contents().assumingMemoryBound(to: Float.self)
        var worst = 0.0
        var worstAbs = 0.0
        for frame in 0..<frames {
          let diff = abs(Double(a[frame]) - Double(b[frame]))
          worstAbs = max(worstAbs, diff)
          worst = max(worst, diff / max(abs(Double(b[frame])), 1e-30))
          precondition(
            diff <= max(2e-5, abs(Double(b[frame])) * 3e-6),
            "\(name): frame \(frame) masked \(a[frame]) versus full \(b[frame])")
        }
        line += String(
          format: "  | full traversal %.2f s, max rel %.2e, max abs %.3g  PASS", full, worst,
          worstAbs)
      }
      print(line)
    }
  }
}
