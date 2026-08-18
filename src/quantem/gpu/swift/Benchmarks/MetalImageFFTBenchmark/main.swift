import Foundation
import Metal
import MetalImageFFT

private func percentile(_ sorted: [Double], fraction: Double) -> Double {
  let index = min(sorted.count - 1, Int((Double(sorted.count - 1) * fraction).rounded()))
  return sorted[index]
}

private func milliseconds<T>(_ operation: () throws -> T) rethrows -> (T, Double) {
  let start = DispatchTime.now().uptimeNanoseconds
  let value = try operation()
  let end = DispatchTime.now().uptimeNanoseconds
  return (value, Double(end - start) / 1_000_000)
}

let arguments = CommandLine.arguments.dropFirst()
let rows = arguments.first.flatMap(Int.init) ?? 2048
let columns = arguments.dropFirst().first.flatMap(Int.init) ?? rows
let iterations = arguments.dropFirst(2).first.flatMap(Int.init) ?? 12

guard rows > 0, columns > 0, iterations > 0 else {
  fatalError("Usage: metal-image-fft-benchmark [rows] [columns] [iterations]")
}
guard let device = MTLCreateSystemDefaultDevice() else {
  fatalError("Metal is unavailable")
}
let count = rows * columns
let values = (0..<count).map { index -> Float in
  let row = index / columns
  let column = index - row * columns
  return sin(Float(row) * 0.017) + cos(Float(column) * 0.013)
}
let source = values.withUnsafeBytes { bytes in
  device.makeBuffer(
    bytes: bytes.baseAddress!,
    length: bytes.count,
    options: .storageModeShared
  )!
}

let (fft, initializationMS) = try milliseconds { try MetalImageFFT(device: device) }
try fft.prewarm(rows: rows, columns: columns)
let output = device.makeBuffer(
  length: count * MemoryLayout<Float>.stride,
  options: .storageModeShared
)!
var measurements: [Double] = []
var maximum: Float = 0
for _ in 0..<iterations {
  let (result, elapsed) = try milliseconds {
    try fft.logMagnitude(
      source: source,
      rows: rows,
      columns: columns,
      scalarType: .float32,
      output: output
    )
  }
  measurements.append(elapsed)
  maximum = result.maximum
}
let first = measurements[0]
let warm = Array(measurements.dropFirst()).sorted()
let p50 = warm.isEmpty ? first : percentile(warm, fraction: 0.50)
let p95 = warm.isEmpty ? first : percentile(warm, fraction: 0.95)

print("device=\(device.name)")
print("shape=\(rows)x\(columns) dtype=float32 full_resolution=true backend=mpsgraph")
print(String(format: "pipeline_initialization_ms=%.3f", initializationMS))
print(String(format: "first_transform_ms=%.3f", first))
print(String(format: "warm_transform_p50_ms=%.3f warm_p50_fps=%.1f", p50, 1_000 / max(0.001, p50)))
print(String(format: "warm_transform_p95_ms=%.3f warm_p95_fps=%.1f", p95, 1_000 / max(0.001, p95)))
print(String(format: "output_maximum=%.6f iterations=%d", maximum, iterations))
