import Foundation
import Metal
import MetalImageFFT
import MetalImageRuntime

private func elapsedMilliseconds<T>(_ operation: () throws -> T) rethrows -> (T, Double) {
  let started = DispatchTime.now().uptimeNanoseconds
  let value = try operation()
  let ended = DispatchTime.now().uptimeNanoseconds
  return (value, Double(ended - started) / 1_000_000)
}

private func percentile(_ sorted: [Double], fraction: Double) -> Double {
  sorted[min(sorted.count - 1, Int((Double(sorted.count - 1) * fraction).rounded()))]
}

let arguments = CommandLine.arguments.dropFirst()
let side = arguments.first.flatMap(Int.init) ?? 512
let iterations = arguments.dropFirst().first.flatMap(Int.init) ?? 20
guard side > 0, iterations > 1 else {
  fatalError("Usage: metal-image-runtime-benchmark [side] [iterations]")
}
guard let device = MTLCreateSystemDefaultDevice() else {
  fatalError("Metal is unavailable")
}
let count = side * side
let values = (0..<count).map { UInt32(($0 * 17) % 4096) }
let source = values.withUnsafeBytes { bytes in
  device.makeBuffer(
    bytes: bytes.baseAddress!,
    length: bytes.count,
    options: .storageModeShared
  )!
}

let (statistics, statisticsInitializationMS) = try elapsedMilliseconds {
  try MetalDisplayStatistics(device: device)
}
let (fft, fftInitializationMS) = try elapsedMilliseconds {
  try MetalImageFFT(device: device)
}
var statisticsMeasurements: [Double] = []
var fftMeasurements: [Double] = []
for _ in 0..<iterations {
  let (_, statisticsMS) = try elapsedMilliseconds {
    try statistics.analyzeUInt32(
      values: source,
      rows: side,
      columns: side,
      scale: .linear
    )
  }
  statisticsMeasurements.append(statisticsMS)
  let (_, fftMS) = try elapsedMilliseconds {
    try fft.logMagnitude(
      source: source,
      rows: side,
      columns: side,
      scalarType: .uint32
    )
  }
  fftMeasurements.append(fftMS)
}

let statisticsWarm = Array(statisticsMeasurements.dropFirst()).sorted()
let fftWarm = Array(fftMeasurements.dropFirst()).sorted()
print("device=\(device.name) shape=\(side)x\(side) dtype=uint32 iterations=\(iterations)")
print(
  String(
    format: "initialization_ms statistics=%.3f fft=%.3f",
    statisticsInitializationMS,
    fftInitializationMS
  )
)
print(
  String(
    format: "cold_ms statistics=%.3f fft=%.3f",
    statisticsMeasurements[0],
    fftMeasurements[0]
  )
)
print(
  String(
    format: "warm_p50_ms statistics=%.3f fft=%.3f",
    percentile(statisticsWarm, fraction: 0.50),
    percentile(fftWarm, fraction: 0.50)
  )
)
print(
  String(
    format: "warm_p95_ms statistics=%.3f fft=%.3f",
    percentile(statisticsWarm, fraction: 0.95),
    percentile(fftWarm, fraction: 0.95)
  )
)
