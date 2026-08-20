import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMKernels

private struct Configuration {
  var batchRows = 4
  var scanColumns = 512
  var detectorRows = 192
  var detectorColumns = 192
  var detectorBin = 2
  var warmups = 3
  var iterations = 15

  init(arguments: [String]) throws {
    var index = 0
    while index < arguments.count {
      guard index + 1 < arguments.count else {
        throw BenchmarkError.invalidArguments("Missing value after \(arguments[index]).")
      }
      let value = try Self.positiveInt(arguments[index + 1], name: arguments[index])
      switch arguments[index] {
      case "--batch-rows": batchRows = value
      case "--scan-columns": scanColumns = value
      case "--detector-rows": detectorRows = value
      case "--detector-columns": detectorColumns = value
      case "--detector-bin": detectorBin = value
      case "--warmups": warmups = value
      case "--iterations": iterations = value
      default:
        throw BenchmarkError.invalidArguments("Unknown option \(arguments[index]).")
      }
      index += 2
    }
  }

  private static func positiveInt(_ value: String, name: String) throws -> Int {
    guard let parsed = Int(value), parsed > 0 else {
      throw BenchmarkError.invalidArguments("\(name) requires a positive integer.")
    }
    return parsed
  }
}

private enum BenchmarkError: LocalizedError {
  case invalidArguments(String)
  case noMetalDevice
  case commandQueueUnavailable
  case commandBufferUnavailable
  case bufferAllocationFailed(String, UInt64)
  case commandFailed(String)
  case parityMismatch(expected: String, actual: String)

  var errorDescription: String? {
    switch self {
    case .invalidArguments(let message): message
    case .noMetalDevice: "No Metal device is available."
    case .commandQueueUnavailable: "Metal could not create a command queue."
    case .commandBufferUnavailable: "Metal could not create a command buffer."
    case .bufferAllocationFailed(let role, let bytes):
      "Metal could not allocate the \(role) buffer (\(bytes) bytes)."
    case .commandFailed(let message): "Metal command failed: \(message)"
    case .parityMismatch(let expected, let actual):
      "Exact detector-bin output differs from the CPU reference: "
        + "expected SHA-256 \(expected), received \(actual)."
    }
  }
}

private struct TimingSummary: Codable {
  let samples: Int
  let p50Milliseconds: Double
  let p95Milliseconds: Double
  let maxMilliseconds: Double
}

private struct Report: Codable {
  let schema: String
  let benchmarkDefinition: String
  let timestampUTC: String
  let device: String
  let hasUnifiedMemory: Bool
  let maxBufferLength: UInt64
  let recommendedMaxWorkingSetSize: UInt64
  let batchRows: Int
  let scanColumns: Int
  let sourceDetectorRows: Int
  let sourceDetectorColumns: Int
  let outputDetectorRows: Int
  let outputDetectorColumns: Int
  let scanRegionRowStart: Int
  let scanRegionRowStop: Int
  let scanRegionColumnStart: Int
  let scanRegionColumnStop: Int
  let scanBin: Int
  let detectorBin: Int
  let sourceDtype: String
  let stagingDtype: String
  let outputDtype: String
  let sourceMaximum: UInt32
  let maximumOutputCount: UInt64
  let stagedSourceBytes: UInt64
  let batchOutputBytes: UInt64
  let full512OutputBytes: UInt64
  let warmups: Int
  let iterations: Int
  let wall: TimingSummary
  let gpu: TimingSummary?
  let stagedInputGigabytesPerSecond: Double
  let outputSHA256: String
  let referenceOutputSHA256: String
  let parity: String
}

private func checkedProduct(_ values: [Int]) throws -> Int {
  var product = 1
  for value in values {
    let next = product.multipliedReportingOverflow(by: value)
    guard !next.overflow else { throw BenchmarkError.invalidArguments("Shape overflows Int.") }
    product = next.partialValue
  }
  return product
}

private func summary(_ values: [Double]) -> TimingSummary {
  let sorted = values.sorted()
  func percentile(_ probability: Double) -> Double {
    let rank = max(1, Int(ceil(probability * Double(sorted.count))))
    return sorted[min(rank - 1, sorted.count - 1)]
  }
  return TimingSummary(
    samples: sorted.count,
    p50Milliseconds: percentile(0.50),
    p95Milliseconds: percentile(0.95),
    maxMilliseconds: sorted.last!
  )
}

private func makeSharedBuffer(
  device: MTLDevice,
  length: Int,
  role: String
) throws -> MTLBuffer {
  guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
    throw BenchmarkError.bufferAllocationFailed(role, UInt64(length))
  }
  return buffer
}

private func availableMetalDevice() -> MTLDevice? {
  if let device = MTLCreateSystemDefaultDevice() { return device }
  #if os(macOS)
    return MTLCopyAllDevices().first
  #else
    return nil
  #endif
}

private func sha256(_ data: Data) -> String {
  SHA256.hash(data: data)
    .map { String(format: "%02x", $0) }
    .joined()
}

private func exactPackedReference(
  source: UnsafePointer<UInt16>,
  plan: Metal4DSTEMLoadPlan
) -> Data {
  let wordsPerScan = (plan.outputDetectorPixels + 1) / 2
  var words = [UInt32](
    repeating: 0,
    count: wordsPerScan * plan.outputScanPositions
  )
  for outputScanRow in 0..<plan.outputScanRows {
    let sourceScanRowStart = outputScanRow * plan.scanBin
    let sourceScanRowStop = min(
      sourceScanRowStart + plan.scanBin,
      plan.scanRegion.rows
    )
    for outputScanColumn in 0..<plan.outputScanColumns {
      let sourceScanColumnStart = outputScanColumn * plan.scanBin
      let sourceScanColumnStop = min(
        sourceScanColumnStart + plan.scanBin,
        plan.scanRegion.columns
      )
      let outputScan = outputScanRow * plan.outputScanColumns + outputScanColumn
      for outputWord in 0..<wordsPerScan {
        var packed: UInt32 = 0
        for lane in 0..<2 {
          let outputDetectorPixel = outputWord * 2 + lane
          guard outputDetectorPixel < plan.outputDetectorPixels else { continue }
          let outputDetectorRow = outputDetectorPixel / plan.outputDetectorColumns
          let outputDetectorColumn =
            outputDetectorPixel - outputDetectorRow * plan.outputDetectorColumns
          let sourceDetectorRowStart = outputDetectorRow * plan.detectorBin
          let sourceDetectorRowStop = min(
            sourceDetectorRowStart + plan.detectorBin,
            plan.detectorRows
          )
          let sourceDetectorColumnStart = outputDetectorColumn * plan.detectorBin
          let sourceDetectorColumnStop = min(
            sourceDetectorColumnStart + plan.detectorBin,
            plan.detectorColumns
          )
          var sum: UInt32 = 0
          for scanRow in sourceScanRowStart..<sourceScanRowStop {
            for scanColumn in sourceScanColumnStart..<sourceScanColumnStop {
              let sourceScan = scanRow * plan.scanRegion.columns + scanColumn
              let frameOffset = sourceScan * plan.detectorPixels
              for detectorRow in sourceDetectorRowStart..<sourceDetectorRowStop {
                let detectorOffset = detectorRow * plan.detectorColumns
                for detectorColumn in sourceDetectorColumnStart..<sourceDetectorColumnStop {
                  sum += UInt32(source[frameOffset + detectorOffset + detectorColumn])
                }
              }
            }
          }
          packed |= sum << UInt32(lane * 16)
        }
        words[outputWord * plan.outputScanPositions + outputScan] = packed
      }
    }
  }
  return words.withUnsafeBytes { Data($0) }
}

private func run() throws {
  let configuration = try Configuration(arguments: Array(CommandLine.arguments.dropFirst()))
  guard let device = availableMetalDevice() else { throw BenchmarkError.noMetalDevice }
  guard let queue = device.makeCommandQueue() else {
    throw BenchmarkError.commandQueueUnavailable
  }
  let sourceIdentity = String(repeating: "a", count: 64)
  let sourceMaximum: UInt32 = 53
  let audit = try Metal4DSTEMExactSourceAudit(
    sourceIdentitySHA256: sourceIdentity,
    sourceDtype: .uint16,
    badPixelIndices: [],
    maximumSourceCount: sourceMaximum,
    pixelsAbove255: 0
  )
  let batchPlan = try Metal4DSTEMLoadPlan(
    sourceScanRows: configuration.batchRows,
    sourceScanColumns: configuration.scanColumns,
    detectorRows: configuration.detectorRows,
    detectorColumns: configuration.detectorColumns,
    sourceBytesPerValue: MemoryLayout<UInt16>.stride,
    scanRegion: Metal4DSTEMScanRegion.full(
      sourceRows: configuration.batchRows,
      sourceColumns: configuration.scanColumns
    ),
    scanBin: 1,
    detectorBin: configuration.detectorBin
  )
  let batchProvenance = try Metal4DSTEMExactBinner.provenance(
    plan: batchPlan,
    sourceAudit: audit,
    stagingDtype: .uint16,
    outputDtype: .uint16
  )
  let fullPlan = try Metal4DSTEMLoadPlan(
    sourceScanRows: 512,
    sourceScanColumns: 512,
    detectorRows: configuration.detectorRows,
    detectorColumns: configuration.detectorColumns,
    sourceBytesPerValue: MemoryLayout<UInt16>.stride,
    scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 512, sourceColumns: 512),
    scanBin: 1,
    detectorBin: configuration.detectorBin
  )
  let fullProvenance = try Metal4DSTEMExactBinner.provenance(
    plan: fullPlan,
    sourceAudit: audit,
    stagingDtype: .uint16,
    outputDtype: .uint16
  )
  let sourceValueCount = try checkedProduct([
    configuration.batchRows,
    configuration.scanColumns,
    configuration.detectorRows,
    configuration.detectorColumns,
  ])
  let sourceBytes = try checkedProduct([sourceValueCount, MemoryLayout<UInt16>.stride])
  guard batchProvenance.outputPayloadBytes <= UInt64(Int.max) else {
    throw BenchmarkError.bufferAllocationFailed("destination", batchProvenance.outputPayloadBytes)
  }
  let source = try makeSharedBuffer(device: device, length: sourceBytes, role: "source")
  let sourceValues = source.contents().bindMemory(to: UInt16.self, capacity: sourceValueCount)
  for index in 0..<sourceValueCount {
    sourceValues[index] = UInt16(index % (Int(sourceMaximum) + 1))
  }
  let destination = try makeSharedBuffer(
    device: device,
    length: Int(batchProvenance.outputPayloadBytes),
    role: "destination"
  )
  memset(destination.contents(), 0, destination.length)
  let binner = try Metal4DSTEMExactBinner(device: device)
  var wallSamples: [Double] = []
  var gpuSamples: [Double] = []
  for iteration in 0..<(configuration.warmups + configuration.iterations) {
    guard let command = queue.makeCommandBuffer() else {
      throw BenchmarkError.commandBufferUnavailable
    }
    let started = DispatchTime.now().uptimeNanoseconds
    _ = try binner.encodeBatch(
      commandBuffer: command,
      stagedSource: source,
      destination: destination,
      plan: batchPlan,
      sourceBatchRows: configuration.batchRows,
      destinationScanRowOffset: 0,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw BenchmarkError.commandFailed(command.error?.localizedDescription ?? "unknown error")
    }
    let finished = DispatchTime.now().uptimeNanoseconds
    if iteration >= configuration.warmups {
      wallSamples.append(Double(finished - started) / 1_000_000)
      if command.gpuEndTime > command.gpuStartTime {
        gpuSamples.append((command.gpuEndTime - command.gpuStartTime) * 1_000)
      }
    }
  }
  let wall = summary(wallSamples)
  let gpu = gpuSamples.count == wallSamples.count ? summary(gpuSamples) : nil
  let outputData = Data(bytes: destination.contents(), count: destination.length)
  let referenceData = exactPackedReference(source: sourceValues, plan: batchPlan)
  let outputDigest = sha256(outputData)
  let referenceDigest = sha256(referenceData)
  guard outputData == referenceData else {
    throw BenchmarkError.parityMismatch(
      expected: referenceDigest,
      actual: outputDigest
    )
  }
  let report = Report(
    schema: "quantem.gpu.metal-4dstem-binning-benchmark/v1",
    benchmarkDefinition:
      "Exact detector-bin kernel only; source is already staged in unified memory. "
      + "Excludes file IO, HDF5 discovery, compressed reads, decode, cache, products, and UI.",
    timestampUTC: ISO8601DateFormatter().string(from: Date()),
    device: device.name,
    hasUnifiedMemory: device.hasUnifiedMemory,
    maxBufferLength: UInt64(device.maxBufferLength),
    recommendedMaxWorkingSetSize: device.recommendedMaxWorkingSetSize,
    batchRows: configuration.batchRows,
    scanColumns: configuration.scanColumns,
    sourceDetectorRows: configuration.detectorRows,
    sourceDetectorColumns: configuration.detectorColumns,
    outputDetectorRows: batchPlan.outputDetectorRows,
    outputDetectorColumns: batchPlan.outputDetectorColumns,
    scanRegionRowStart: batchPlan.scanRegion.rowStart,
    scanRegionRowStop: batchPlan.scanRegion.rowStop,
    scanRegionColumnStart: batchPlan.scanRegion.columnStart,
    scanRegionColumnStop: batchPlan.scanRegion.columnStop,
    scanBin: batchPlan.scanBin,
    detectorBin: batchPlan.detectorBin,
    sourceDtype: batchProvenance.sourceDtype.rawValue,
    stagingDtype: batchProvenance.stagingDtype.rawValue,
    outputDtype: batchProvenance.outputDtype.rawValue,
    sourceMaximum: sourceMaximum,
    maximumOutputCount: batchProvenance.maximumOutputCount,
    stagedSourceBytes: UInt64(sourceBytes),
    batchOutputBytes: batchProvenance.outputPayloadBytes,
    full512OutputBytes: fullProvenance.outputPayloadBytes,
    warmups: configuration.warmups,
    iterations: configuration.iterations,
    wall: wall,
    gpu: gpu,
    stagedInputGigabytesPerSecond: (Double(sourceBytes) / 1_000_000_000)
      / (wall.p50Milliseconds / 1_000),
    outputSHA256: outputDigest,
    referenceOutputSHA256: referenceDigest,
    parity: "byte_exact"
  )
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
  FileHandle.standardOutput.write(try encoder.encode(report))
  FileHandle.standardOutput.write(Data([0x0A]))
}

do {
  try run()
} catch {
  FileHandle.standardError.write(Data("ERROR: \(error.localizedDescription)\n".utf8))
  exit(2)
}
