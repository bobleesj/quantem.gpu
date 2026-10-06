import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMKernels

private struct Options {
  let rowInput: URL
  let columnInput: URL
  let outputDirectory: URL
  let revision: String
  let rows: Int
  let columns: Int
  let rotationDegrees: Double
  let useTranspose: Bool
  let warmups: Int
  let iterations: Int

  static let usage = """
    Usage: metal-4dstem-dpc-benchmark \
      --row-input PATH --column-input PATH --output-dir EMPTY_PATH \
      --revision 40_CHARACTER_COMMIT_SHA --rows N --columns N \
      --rotation-degrees VALUE --transpose true|false \
      [--warmups N] [--iterations N]

    Measures the existing UI-free Metal CoM rotation and Fourier iDPC kernels.
    Inputs and outputs are row-major float32 fields. The scan shape must be
    power-of-two for the current radix-2 path. This benchmark does not select a
    rotation policy, load 4D-STEM data, or present an application UI.
    """

  static func parse(_ arguments: [String]) throws -> Self {
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
      guard arguments[index].hasPrefix("--"), index + 1 < arguments.count else {
        throw BenchmarkError.usage(Self.usage)
      }
      values[arguments[index]] = arguments[index + 1]
      index += 2
    }
    guard let rowInput = values["--row-input"],
      let columnInput = values["--column-input"],
      let outputDirectory = values["--output-dir"],
      let revision = values["--revision"], revision.utf8.count == 40,
      revision.utf8.allSatisfy({
        (48...57).contains($0) || (97...102).contains($0)
      }),
      let rows = values["--rows"].flatMap(Int.init), rows > 0,
      let columns = values["--columns"].flatMap(Int.init), columns > 0,
      let rotationDegrees = values["--rotation-degrees"].flatMap(Double.init),
      let transposeRaw = values["--transpose"],
      let useTranspose = ["true": true, "false": false][transposeRaw],
      let warmups = Int(values["--warmups"] ?? "3"), warmups >= 0,
      let iterations = Int(values["--iterations"] ?? "15"), iterations > 0
    else {
      throw BenchmarkError.usage(Self.usage)
    }
    guard isPowerOfTwo(rows), isPowerOfTwo(columns) else {
      throw BenchmarkError.usage(
        "The current radix-2 iDPC path requires power-of-two scan rows and columns."
      )
    }
    return Self(
      rowInput: URL(fileURLWithPath: rowInput),
      columnInput: URL(fileURLWithPath: columnInput),
      outputDirectory: URL(fileURLWithPath: outputDirectory, isDirectory: true),
      revision: revision,
      rows: rows,
      columns: columns,
      rotationDegrees: rotationDegrees,
      useTranspose: useTranspose,
      warmups: warmups,
      iterations: iterations
    )
  }
}

private enum BenchmarkError: LocalizedError {
  case usage(String)
  case invalid(String)

  var errorDescription: String? {
    switch self {
    case .usage(let message), .invalid(let message): message
    }
  }
}

private struct FFT2DParameters {
  var width: UInt32
  var height: UInt32
  var log2Size: UInt32
  var stage: UInt32
  var direction: Float
  var rowAxis: UInt32
}

private struct DPCPackParameters {
  var count: UInt32
  var flags: UInt32
  var padding0: UInt32 = 0
  var padding1: UInt32 = 0
  var rotation: SIMD4<Float>
}

private struct TimingSummary: Codable {
  let samples: Int
  let p50Milliseconds: Double
  let p95Milliseconds: Double
  let maximumMilliseconds: Double
}

private struct Report: Codable {
  let schema: String
  let benchmarkDefinition: String
  let timestampUTC: String
  let revision: String
  let host: String
  let os: String
  let device: String
  let rows: Int
  let columns: Int
  let inputDtype: String
  let outputDtype: String
  let rotationDegrees: Double
  let useTranspose: Bool
  let warmups: Int
  let iterations: Int
  let pipelineCompilationSeconds: Double
  let wallSamplesMilliseconds: [Double]
  let gpuSamplesMilliseconds: [Double]
  let wall: TimingSummary
  let gpu: TimingSummary
  let metalAllocatedBytesBeforeBuffers: UInt64
  let metalAllocatedBytesAfterBuffers: UInt64
  let peakSampledMetalAllocatedBytes: UInt64
  let metalAllocatedBytesAfterBenchmark: UInt64
  let rowInputSHA256: String
  let columnInputSHA256: String
  let rotatedRowSHA256: String
  let rotatedColumnSHA256: String
  let phaseSHA256: String
}

private struct Pipelines {
  let pack: MTLComputePipelineState
  let bitReverseRows: MTLComputePipelineState
  let bitReverseColumns: MTLComputePipelineState
  let butterflyRows: MTLComputePipelineState
  let butterflyColumns: MTLComputePipelineState
  let normalize: MTLComputePipelineState
  let poisson: MTLComputePipelineState
  let extract: MTLComputePipelineState
}

private func isPowerOfTwo(_ value: Int) -> Bool {
  value > 0 && value & (value - 1) == 0
}

private func sha256(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
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
    maximumMilliseconds: sorted.last!
  )
}

private func pipeline(
  _ library: MTLLibrary,
  _ name: String,
  _ device: MTLDevice
) throws -> MTLComputePipelineState {
  guard let function = library.makeFunction(name: name) else {
    throw BenchmarkError.invalid("The Metal library is missing function \(name).")
  }
  return try device.makeComputePipelineState(function: function)
}

private func makePipelines(device: MTLDevice) throws -> Pipelines {
  let library = try Metal4DSTEMKernels.makeDPCLibrary(device: device)
  return try Pipelines(
    pack: pipeline(library, Metal4DSTEMKernels.dpcPackFunction, device),
    bitReverseRows: pipeline(
      library, Metal4DSTEMKernels.fftBitReverseRowsFunction, device
    ),
    bitReverseColumns: pipeline(
      library, Metal4DSTEMKernels.fftBitReverseColumnsFunction, device
    ),
    butterflyRows: pipeline(
      library, Metal4DSTEMKernels.fftButterflyRowsFunction, device
    ),
    butterflyColumns: pipeline(
      library, Metal4DSTEMKernels.fftButterflyColumnsFunction, device
    ),
    normalize: pipeline(library, Metal4DSTEMKernels.fftNormalizeFunction, device),
    poisson: pipeline(library, Metal4DSTEMKernels.dpcPoissonFunction, device),
    extract: pipeline(library, Metal4DSTEMKernels.dpcExtractPhaseFunction, device)
  )
}

private func encodeFFT(
  encoder: MTLComputeCommandEncoder,
  buffer: MTLBuffer,
  rows: Int,
  columns: Int,
  inverse: Bool,
  pipelines: Pipelines
) {
  let widthStages = UInt32(columns.trailingZeroBitCount)
  let heightStages = UInt32(rows.trailingZeroBitCount)
  func dispatch(
    _ pipeline: MTLComputePipelineState,
    width: Int,
    height: Int,
    log2Size: UInt32,
    stage: UInt32,
    rowAxis: Bool
  ) {
    var parameters = FFT2DParameters(
      width: UInt32(columns),
      height: UInt32(rows),
      log2Size: log2Size,
      stage: stage,
      direction: inverse ? 1 : -1,
      rowAxis: rowAxis ? 1 : 0
    )
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBytes(&parameters, length: MemoryLayout<FFT2DParameters>.stride, index: 1)
    encoder.dispatchThreads(
      MTLSize(width: width, height: height, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
    )
    encoder.memoryBarrier(scope: .buffers)
  }
  dispatch(
    pipelines.bitReverseRows,
    width: columns,
    height: rows,
    log2Size: widthStages,
    stage: 0,
    rowAxis: true
  )
  for stage in 0..<widthStages {
    dispatch(
      pipelines.butterflyRows,
      width: columns / 2,
      height: rows,
      log2Size: widthStages,
      stage: stage,
      rowAxis: true
    )
  }
  dispatch(
    pipelines.bitReverseColumns,
    width: columns,
    height: rows,
    log2Size: heightStages,
    stage: 0,
    rowAxis: false
  )
  for stage in 0..<heightStages {
    dispatch(
      pipelines.butterflyColumns,
      width: columns,
      height: rows / 2,
      log2Size: heightStages,
      stage: stage,
      rowAxis: false
    )
  }
  if inverse {
    dispatch(
      pipelines.normalize,
      width: columns,
      height: rows,
      log2Size: heightStages,
      stage: 0,
      rowAxis: false
    )
  }
}

private func encodePack(
  encoder: MTLComputeCommandEncoder,
  pipelines: Pipelines,
  row: MTLBuffer,
  column: MTLBuffer,
  gradient: MTLBuffer,
  options: Options
) {
  let angle = options.rotationDegrees * .pi / 180
  var parameters = DPCPackParameters(
    count: UInt32(options.rows * options.columns),
    flags: options.useTranspose ? 1 : 0,
    rotation: SIMD4(Float(cos(angle)), Float(sin(angle)), 0, 0)
  )
  encoder.setComputePipelineState(pipelines.pack)
  encoder.setBuffer(row, offset: 0, index: 0)
  encoder.setBuffer(column, offset: 0, index: 1)
  encoder.setBuffer(gradient, offset: 0, index: 2)
  encoder.setBytes(
    &parameters,
    length: MemoryLayout<DPCPackParameters>.stride,
    index: 3
  )
  encoder.dispatchThreads(
    MTLSize(width: options.rows * options.columns, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
  )
}

private func encodeIDPC(
  encoder: MTLComputeCommandEncoder,
  pipelines: Pipelines,
  row: MTLBuffer,
  column: MTLBuffer,
  gradient: MTLBuffer,
  phaseFFT: MTLBuffer,
  phase: MTLBuffer,
  options: Options
) {
  let count = options.rows * options.columns
  encodePack(
    encoder: encoder,
    pipelines: pipelines,
    row: row,
    column: column,
    gradient: gradient,
    options: options
  )
  encoder.memoryBarrier(scope: .buffers)
  encodeFFT(
    encoder: encoder,
    buffer: gradient,
    rows: options.rows,
    columns: options.columns,
    inverse: false,
    pipelines: pipelines
  )
  var shape = SIMD4<UInt32>(UInt32(options.columns), UInt32(options.rows), UInt32(count), 0)
  encoder.setComputePipelineState(pipelines.poisson)
  encoder.setBuffer(gradient, offset: 0, index: 0)
  encoder.setBuffer(phaseFFT, offset: 0, index: 1)
  encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 2)
  encoder.dispatchThreads(
    MTLSize(width: count, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
  )
  encoder.memoryBarrier(scope: .buffers)
  encodeFFT(
    encoder: encoder,
    buffer: phaseFFT,
    rows: options.rows,
    columns: options.columns,
    inverse: true,
    pipelines: pipelines
  )
  encoder.setComputePipelineState(pipelines.extract)
  encoder.setBuffer(phaseFFT, offset: 0, index: 0)
  encoder.setBuffer(phase, offset: 0, index: 1)
  var countU32 = UInt32(count)
  encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.stride, index: 2)
  encoder.dispatchThreads(
    MTLSize(width: count, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
  )
}

private func makeBuffer(
  device: MTLDevice,
  data: Data,
  options: MTLResourceOptions,
  role: String
) throws -> MTLBuffer {
  guard let buffer = device.makeBuffer(bytes: [UInt8](data), length: data.count, options: options)
  else {
    throw BenchmarkError.invalid("Metal could not allocate the \(role) buffer.")
  }
  return buffer
}

private func makeBuffer(
  device: MTLDevice,
  length: Int,
  options: MTLResourceOptions,
  role: String
) throws -> MTLBuffer {
  guard let buffer = device.makeBuffer(length: length, options: options) else {
    throw BenchmarkError.invalid("Metal could not allocate the \(role) buffer.")
  }
  return buffer
}

private func run() throws {
  let arguments = Array(CommandLine.arguments.dropFirst())
  if arguments.contains("--help") || arguments.contains("-h") {
    print(Options.usage)
    return
  }
  let options = try Options.parse(arguments)
  let count = options.rows * options.columns
  let scalarBytes = count * MemoryLayout<Float>.stride
  let complexBytes = count * MemoryLayout<SIMD2<Float>>.stride
  let rowData = try Data(contentsOf: options.rowInput)
  let columnData = try Data(contentsOf: options.columnInput)
  guard rowData.count == scalarBytes, columnData.count == scalarBytes else {
    throw BenchmarkError.invalid(
      "Each float32 input must contain exactly \(scalarBytes) bytes for "
        + "\(options.rows)×\(options.columns) values."
    )
  }
  if FileManager.default.fileExists(atPath: options.outputDirectory.path) {
    let contents = try FileManager.default.contentsOfDirectory(
      at: options.outputDirectory,
      includingPropertiesForKeys: nil
    )
    guard contents.isEmpty else {
      throw BenchmarkError.invalid("Choose a new empty output directory.")
    }
  } else {
    try FileManager.default.createDirectory(
      at: options.outputDirectory,
      withIntermediateDirectories: true
    )
  }
  guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    throw BenchmarkError.invalid("No usable Metal device or command queue is available.")
  }
  let compileStarted = CFAbsoluteTimeGetCurrent()
  let pipelines = try makePipelines(device: device)
  let compileSeconds = CFAbsoluteTimeGetCurrent() - compileStarted
  let metalAllocatedBytesBeforeBuffers = UInt64(device.currentAllocatedSize)
  let row = try makeBuffer(
    device: device,
    data: rowData,
    options: .storageModeShared,
    role: "row CoM"
  )
  let column = try makeBuffer(
    device: device,
    data: columnData,
    options: .storageModeShared,
    role: "column CoM"
  )
  let gradient = try makeBuffer(
    device: device,
    length: complexBytes,
    options: .storageModePrivate,
    role: "gradient"
  )
  let phaseFFT = try makeBuffer(
    device: device,
    length: complexBytes,
    options: .storageModePrivate,
    role: "phase FFT"
  )
  let phase = try makeBuffer(
    device: device,
    length: scalarBytes,
    options: .storageModeShared,
    role: "phase"
  )
  let validationGradient = try makeBuffer(
    device: device,
    length: complexBytes,
    options: .storageModeShared,
    role: "validation gradient"
  )
  let metalAllocatedBytesAfterBuffers = UInt64(device.currentAllocatedSize)
  var peakSampledMetalAllocatedBytes = metalAllocatedBytesAfterBuffers

  guard let validationCommand = queue.makeCommandBuffer(),
    let validationEncoder = validationCommand.makeComputeCommandEncoder()
  else {
    throw BenchmarkError.invalid("Metal could not create the rotation validation command.")
  }
  encodePack(
    encoder: validationEncoder,
    pipelines: pipelines,
    row: row,
    column: column,
    gradient: validationGradient,
    options: options
  )
  validationEncoder.endEncoding()
  validationCommand.commit()
  validationCommand.waitUntilCompleted()
  guard validationCommand.status == .completed else {
    throw BenchmarkError.invalid(
      validationCommand.error?.localizedDescription ?? "Rotation validation failed."
    )
  }

  let rotated = validationGradient.contents().bindMemory(
    to: SIMD2<Float>.self,
    capacity: count
  )
  var rotatedRows = [Float](repeating: 0, count: count)
  var rotatedColumns = [Float](repeating: 0, count: count)
  for index in 0..<count {
    rotatedRows[index] = rotated[index].x
    rotatedColumns[index] = rotated[index].y
  }
  let rotatedRowData = rotatedRows.withUnsafeBytes { Data($0) }
  let rotatedColumnData = rotatedColumns.withUnsafeBytes { Data($0) }
  try rotatedRowData.write(
    to: options.outputDirectory.appendingPathComponent("rotated_row.f32")
  )
  try rotatedColumnData.write(
    to: options.outputDirectory.appendingPathComponent("rotated_column.f32")
  )

  var wallSamples: [Double] = []
  var gpuSamples: [Double] = []
  for iteration in 0..<(options.warmups + options.iterations) {
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw BenchmarkError.invalid("Metal could not create the iDPC command.")
    }
    let started = DispatchTime.now().uptimeNanoseconds
    encodeIDPC(
      encoder: encoder,
      pipelines: pipelines,
      row: row,
      column: column,
      gradient: gradient,
      phaseFFT: phaseFFT,
      phase: phase,
      options: options
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    let finished = DispatchTime.now().uptimeNanoseconds
    guard command.status == .completed else {
      throw BenchmarkError.invalid(
        command.error?.localizedDescription ?? "The iDPC command failed."
      )
    }
    peakSampledMetalAllocatedBytes = max(
      peakSampledMetalAllocatedBytes,
      UInt64(device.currentAllocatedSize)
    )
    if iteration >= options.warmups {
      wallSamples.append(Double(finished - started) / 1_000_000)
      guard command.gpuEndTime > command.gpuStartTime else {
        throw BenchmarkError.invalid("Metal did not report synchronized GPU timing.")
      }
      gpuSamples.append((command.gpuEndTime - command.gpuStartTime) * 1_000)
    }
  }
  let phaseData = Data(bytes: phase.contents(), count: scalarBytes)
  try phaseData.write(to: options.outputDirectory.appendingPathComponent("idpc.f32"))
  let report = Report(
    schema: "quantem.gpu.metal-4dstem-dpc-benchmark/v1",
    benchmarkDefinition:
      "Explicit CoM rotation plus radix-2 Fourier iDPC on already resident float32 maps. "
      + "Excludes 4D-STEM load, automatic rotation selection, display, and application UI.",
    timestampUTC: ISO8601DateFormatter().string(from: Date()),
    revision: options.revision,
    host: ProcessInfo.processInfo.hostName,
    os: ProcessInfo.processInfo.operatingSystemVersionString,
    device: device.name,
    rows: options.rows,
    columns: options.columns,
    inputDtype: "float32",
    outputDtype: "float32",
    rotationDegrees: options.rotationDegrees,
    useTranspose: options.useTranspose,
    warmups: options.warmups,
    iterations: options.iterations,
    pipelineCompilationSeconds: compileSeconds,
    wallSamplesMilliseconds: wallSamples,
    gpuSamplesMilliseconds: gpuSamples,
    wall: summary(wallSamples),
    gpu: summary(gpuSamples),
    metalAllocatedBytesBeforeBuffers: metalAllocatedBytesBeforeBuffers,
    metalAllocatedBytesAfterBuffers: metalAllocatedBytesAfterBuffers,
    peakSampledMetalAllocatedBytes: peakSampledMetalAllocatedBytes,
    metalAllocatedBytesAfterBenchmark: UInt64(device.currentAllocatedSize),
    rowInputSHA256: sha256(rowData),
    columnInputSHA256: sha256(columnData),
    rotatedRowSHA256: sha256(rotatedRowData),
    rotatedColumnSHA256: sha256(rotatedColumnData),
    phaseSHA256: sha256(phaseData)
  )
  let reportData = try JSONEncoder.sorted.encode(report)
  try reportData.write(to: options.outputDirectory.appendingPathComponent("summary.json"))
  FileHandle.standardOutput.write(reportData)
  FileHandle.standardOutput.write(Data([0x0A]))
}

extension JSONEncoder {
  fileprivate static var sorted: JSONEncoder {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    return encoder
  }
}

do {
  try run()
} catch {
  FileHandle.standardError.write(Data("ERROR: \(error.localizedDescription)\n".utf8))
  exit(2)
}
