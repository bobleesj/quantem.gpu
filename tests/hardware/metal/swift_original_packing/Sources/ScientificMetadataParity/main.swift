import Foundation
import Metal
import MetalImageRuntime
import Native4DSTEMIO

let folder = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let metadata = NativeMicroscopeMetadata(metadata: [
  "electron_microscope/electron_source/accelerating_voltage": "300000 V",
  "electron_microscope/illumination_system/semi_convergence_angle": "0 mrad",
  "electron_microscope/scan_controller/regular_scan/dwell_time": "0.00005 s",
  "electron_microscope/imaging_system/reciprocal_pixel_size_x": "0.182 mrad",
  "electron_microscope/imaging_system/reciprocal_pixel_size_y": "0.000182 rad",
])
precondition(metadata.beamEnergyKeV == 300 && metadata.semiConvergenceAngleMrad == nil)
precondition(abs(metadata.dwellTimeMicroseconds! - 50) < 1e-12)
precondition(abs(metadata.angularRowMrad! - metadata.angularColumnMrad!) < 1e-12)
let unknown = NativeMicroscopeMetadata(metadata: ["electron_microscope/electron_source/accelerating_voltage": "300000"])
precondition(unknown.beamEnergyKeV == nil)
let histogram = MetalHistogramDisplayContract.reference(values: (0...7).map(Double.init), scale: .logarithmic)
precondition(histogram.bins.enumerated().filter { $0.element > 0 }.map(\.offset) == [0, 85, 135, 170, 198, 220, 239, 255])
let device = MTLCreateSystemDefaultDevice()!
let statistics = try MetalDisplayStatistics(device: device)
for values: [UInt32] in [[0, 1], [0, 1, 2, 3], Array(0...7), [10, 15, 27, 50], [65535], [UInt32.max]] {
  let buffer = values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
  for logarithmic in [false, true] {
    let gpu = try statistics.analyzeUInt32(values: buffer, rows: 1, columns: values.count,
      scale: logarithmic ? .logarithmic : .linear)
    let cpu = MetalHistogramDisplayContract.reference(values: values.map(Double.init),
      scale: logarithmic ? .logarithmic : .linear)
    precondition(gpu.minimum == values.min() && gpu.maximum == values.max())
    precondition(gpu.bins == cpu.bins, "GPU histogram differs from CPU oracle: \(values), log=\(logarithmic)")
  }
}
for logarithmic in [false, true] {
  let values: [Float] = [-7, -1.5, 0, 0.5, 2, 13, .nan, .infinity, -.infinity]
  let buffer = values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
  let gpu = try statistics.analyzeFloat32(values: buffer, rows: 1, columns: values.count,
    scale: logarithmic ? .logarithmic : .linear)
  let cpu = MetalHistogramDisplayContract.reference(values: values.map(Double.init),
    scale: logarithmic ? .logarithmic : .linear)
  precondition(gpu.bins == cpu.bins && cpu.invalidCount == 3)
  for value in values.filter(\.isFinite) {
    let fraction = MetalHistogramDisplayContract.normalizedFraction(value: Double(value), minimum: -7, maximum: 13,
      scale: logarithmic ? .logarithmic : .linear)
    let restored = MetalHistogramDisplayContract.rawValue(fraction: fraction, minimum: -7, maximum: 13,
      scale: logarithmic ? .logarithmic : .linear)!
    precondition(abs(restored - Double(value)) < 1e-12)
  }
}
let unsigned: [UInt32] = [0, 1, 65535, 16777217, 4294967294, 4294967295]
let floatBits: [UInt32] = [0x80000000, 0x3f000000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc01234]
let document = Data("{\"schema\":\"test\",\"notes\":\"Crystalline area; 20 µs dwell.\",\"calibration\":0.4153}".utf8)
let images = [
  NativeScientificImage(name: "counts", rows: 2, columns: 3, scalarType: .uint32, values: unsigned.withUnsafeBytes { Data($0) }),
  NativeScientificImage(name: "measurements", rows: 3, columns: 2, scalarType: .float32, values: floatBits.withUnsafeBytes { Data($0) }),
]
let destination = folder.appendingPathComponent("scientific.h5")
try NativeScientificExport.write(images: images, metadata: document, to: destination)
let restoredMetadata = try NativeScientificExport.metadata(at: destination)
precondition(restoredMetadata == document)
let original = try Data(contentsOf: destination)
do {
  try NativeScientificExport.write(images: images, metadata: document, to: destination)
  fatalError("Existing output must not be overwritten")
} catch {}
let unchanged = try Data(contentsOf: destination)
precondition(unchanged == original)
print("PASS: metadata units, absent invalid values, histogram bins, export and protected destination")
