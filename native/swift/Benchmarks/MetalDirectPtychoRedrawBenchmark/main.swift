import Foundation
import Metal
import MetalDirectPtychoKernels

// Interactive SSB redraw timing: the phase re-rendered while one control changes, after a fit.
//
// The geometry is that of a 512 x 512 (or, with --scans, 256 x 256) x 192 x 192 acquisition at 300 kV, 30 mrad and a 0.373 A step with about
// 8,800 bright-field pixels. Counts are deterministic synthetic values: the reconstruction's work depends on the
// geometry, the grid and the fitted terms, not on the count values. Every BF pixel, all 14 aberration terms, tilt
// and depth are retained. Each observation changes C10 and synchronizes the GPU (reconstruct returns a finished
// phase); preparation and warmup are excluded.
//
//   swift run -c release metal-direct-ptycho-redraw-benchmark [--scans 256,512] [--factors 1,2,3,4] [--orders low,fifth] [--runs 20]
//     [--warmup 3] [--method ssb|phase|complex] [--dump <folder>] [--json <file>]
//
// `--method phase|complex` times the regularized direct reconstructions (default regularization), which prepare their
// normal equations again at every C10 change.
//
// `--dump` writes the phase of the first (unchanged-C10) observation per order and factor as raw float32 for
// before/after parity checks.

struct Options {
  var factors = [1, 2, 3, 4]
  var orders = ["low", "fifth"]
  var scans = [512]
  var runs = 20
  var warmup = 3
  var dump: URL?
  var json: URL?
  var method = DirectPtychoReconstructionMethod.ssb

  init(_ arguments: [String]) {
    var iterator = arguments.dropFirst().makeIterator()
    while let argument = iterator.next() {
      let value = iterator.next() ?? ""
      switch argument {
      case "--factors": factors = value.split(separator: ",").compactMap { Int($0) }
      case "--orders": orders = value.split(separator: ",").map(String.init)
      case "--scans": scans = value.split(separator: ",").compactMap { Int($0) }
      case "--runs": runs = Int(value) ?? runs
      case "--warmup": warmup = Int(value) ?? warmup
      case "--dump": dump = URL(fileURLWithPath: value)
      case "--json": json = URL(fileURLWithPath: value)
      case "--method": method = DirectPtychoReconstructionMethod(rawValue: value) ?? method
      default: fatalError("Unknown option \(argument)")
      }
    }
  }
}

let options = Options(CommandLine.arguments)
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal is unavailable") }
// Only a physical Apple GPU qualifies these timings.
guard device.name.hasPrefix("Apple"), !device.isLowPower, device.hasUnifiedMemory else {
  fatalError("Not a physical Apple-silicon GPU: \(device.name)")
}
print("device \(device.name), working set \(device.recommendedMaxWorkingSetSize >> 30) GiB")

let detector = 192
let radius = 52.95
let calibration = MetalDirectPtychoCalibration(
  voltageKV: 300, semiangleMrad: 30, scanSamplingRowAngstroms: 0.373,
  scanSamplingColumnAngstroms: 0.373, detectorSamplingRowMrad: 30 / radius,
  detectorSamplingColumnMrad: 30 / radius, brightfieldCenterRowPixels: 95.5,
  brightfieldCenterColumnPixels: 95.5, brightfieldRadiusPixels: radius)
var records: [[String: Any]] = []
for scan in options.scans {
  let setup = try calibration.geometry(
    detectorRows: detector, detectorColumns: detector,
    detectorSum: [UInt64](repeating: 40 * 512 * 512, count: detector * detector),
    scanRows: scan, scanColumns: scan)
  let count = setup.pixels.count
  let plane = scan * scan
  print(
    "scan \(scan)^2, bright-field pixels \(count), auto output \(try calibration.ssbUpsamplingFactor())x"
  )

  guard let counts = device.makeBuffer(length: count * plane, options: .storageModeShared) else {
    fatalError("Cannot allocate \(count * plane) count bytes")
  }
  // Deterministic counts around 40 with a weak scan-dependent pattern.
  nonisolated(unsafe) let countBytes = counts.contents().assumingMemoryBound(to: UInt8.self)
  DispatchQueue.concurrentPerform(iterations: count) { k in
    let base = countBytes + k * plane
    var state = UInt64(k &+ 1) &* 0x9E37_79B9_7F4A_7C15
    for i in 0..<plane {
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      let pattern = (i / 512 + i % 512 + k) % 7
      base[i] = UInt8(30 + Int(state >> 61) + pattern)
    }
  }
  let engine = try MetalDirectPtychoEngine(device: device, geometry: setup.geometry)
  var started = Date()
  try engine.prepare(brightfield: counts, countType: .uint8)
  print(String(format: "prepare %.2f s", Date().timeIntervalSince(started)))
  let model = try engine.jointModel(scanDetectorRotationDegrees: 17)

  func fifthOrderTerms() -> [DirectPtychoModelHigherOrder] {
    // Plausible fitted magnitudes (Angstrom) and azimuths; every higher-order term is nonzero.
    let magnitudes: [Double] = [300, 200, 1e5, 3e4, 2e4, 5e5, 3e5, 2e5, 1e7, 5e6, 4e6, 3e6]
    return zip(DirectPtychoModelHigherOrder.supported, magnitudes).enumerated().map { index, pair in
      var term = pair.0
      term.magnitudeAngstroms = pair.1
      term.angleRadians = 0.3 + 0.41 * Double(index)
      return term
    }
  }

  func point(_ order: String, c10: Double) -> DirectPtychoModelPoint {
    DirectPtychoModelPoint(
      c10Angstroms: c10, c12Angstroms: 23.5, phi12Radians: 0.62, sampleTiltMrad: [1.2, -0.8],
      depthSpreadAngstroms: 150, higherOrder: order == "fifth" ? fifthOrderTerms() : [])
  }

  func percentile(_ values: [Double], _ p: Double) -> Double {
    let sorted = values.sorted()
    let position = p * Double(sorted.count - 1)
    let low = Int(position.rounded(.down))
    let high = Int(position.rounded(.up))
    return sorted[low] + (sorted[high] - sorted[low]) * (position - Double(low))
  }

  for order in options.orders {
    for factor in options.factors {
      var times: [Double] = []
      var lastKernel: MetalKernelIdentity?
      for run in 0..<(options.warmup + options.runs) {
        let c10 = -80.0 + 1.5 * Double(run)
        started = Date()
        let result = try model.reconstruct(
          point(order, c10: c10), upsamplingFactor: factor, method: options.method)
        let elapsed = Date().timeIntervalSince(started) * 1000
        if run == 0, let folder = options.dump {
          try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
          let bytes = Data(bytes: result.phase.contents(), count: result.rows * result.columns * 4)
          try bytes.write(to: folder.appendingPathComponent("\(scan)_\(order)_\(factor)x.f32"))
        }
        if run >= options.warmup { times.append(elapsed) }
        lastKernel = result.kernel
      }
      let median = percentile(times, 0.5)
      let p90 = percentile(times, 0.9)
      print("  kernel \(lastKernel?.label ?? "unrecorded")")
      print(
        String(
          format: "%d %@ %dx (%d^2): median %.2f ms, p90 %.2f ms, min %.2f ms, %.1f FPS at median",
          scan, order, factor, scan * factor, median, p90, times.min()!, 1000 / median))
      records.append([
        "scan": scan, "brightfield_pixels": count, "order": order, "factor": factor,
        "output": scan * factor, "median_ms": median,
        "p90_ms": p90, "min_ms": times.min()!, "runs": times.count,
        "kernel": lastKernel?.label ?? "unrecorded",
      ])
    }
  }
}
if let json = options.json {
  let payload: [String: Any] = [
    "device": device.name, "method": options.method.rawValue,
    "records": records,
  ]
  try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    .write(to: json)
}
