import Darwin
import Foundation
import Metal
import MetalSSBKernels

let scanSize = 512
let scanPixels = scanSize * scanSize

struct SourceMetadata: Decodable {
  let expectedFilename: String
  let bytes: Int
  let sha256: String
  let layout: String

  enum CodingKeys: String, CodingKey {
    case expectedFilename = "expected_filename"
    case bytes
    case sha256
    case layout
  }
}

struct ReferenceMetadata: Decodable {
  let filename: String
  let sha256: String
  let backend: String
}

struct AberrationMetadata: Decodable {
  let c10: Float
  let c12: Float
  let phi12: Float

  enum CodingKeys: String, CodingKey {
    case c10 = "C10"
    case c12 = "C12"
    case phi12
  }
}

struct BenchmarkMetadata: Decodable {
  let schema: String
  let dataset: String
  let detectorBin: Int
  let scanCrop: String?
  let storageDType: String
  let logicalBrightfieldCount: Int
  let apertureActiveBrightfieldCount: Int
  let wavelengthAngstroms: Float
  let semiangleRadians: Float
  let rotationAngleDegrees: Float
  let source: SourceMetadata
  let reference: ReferenceMetadata
  let aberrations: AberrationMetadata
  let detectorRows: [Int]
  let detectorColumns: [Int]
  let brightfieldKX: [Float]
  let brightfieldKY: [Float]
  let brightfieldAlphaSquared: [Float]
  let brightfieldCos2Phi: [Float]
  let brightfieldSin2Phi: [Float]
  let brightfieldAperture: [Float]
  let qxByRow: [Float]
  let qyByColumn: [Float]
  let angularSamplingRadians: [Float]
  let dcValue: [Float]

  enum CodingKeys: String, CodingKey {
    case schema
    case dataset
    case detectorBin = "detector_bin"
    case scanCrop = "scan_crop"
    case storageDType = "storage_dtype"
    case logicalBrightfieldCount = "logical_bf_count"
    case apertureActiveBrightfieldCount = "aperture_active_bf_count"
    case wavelengthAngstroms = "wavelength_A"
    case semiangleRadians = "semiangle_rad"
    case rotationAngleDegrees = "rotation_angle_deg"
    case source
    case reference
    case aberrations
    case detectorRows = "detector_rows"
    case detectorColumns = "detector_cols"
    case brightfieldKX = "kx"
    case brightfieldKY = "ky"
    case brightfieldAlphaSquared = "alpha_k2"
    case brightfieldCos2Phi = "cos2phi_k"
    case brightfieldSin2Phi = "sin2phi_k"
    case brightfieldAperture = "aperture_k"
    case qxByRow = "qx_1d"
    case qyByColumn = "qy_1d"
    case angularSamplingRadians = "angular_sampling_rad"
    case dcValue = "dc_value"
  }
}

final class ReadOnlyMappedFile {
  let pointer: UnsafeMutableRawPointer
  let length: Int
  private var ownsMapping = true

  init(path: String) throws {
    let descriptor = Darwin.open(path, O_RDONLY)
    guard descriptor >= 0 else {
      throw CocoaError(.fileReadNoSuchFile)
    }
    defer { Darwin.close(descriptor) }
    var status = stat()
    guard fstat(descriptor, &status) == 0 else {
      throw CocoaError(.fileReadUnknown)
    }
    length = Int(status.st_size)
    guard
      let mapping = mmap(
        nil,
        length,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE,
        descriptor,
        0
      ), mapping != MAP_FAILED
    else {
      throw CocoaError(.fileReadUnknown)
    }
    pointer = mapping
  }

  deinit {
    if ownsMapping {
      munmap(pointer, length)
    }
  }

  func relinquishOwnership() {
    ownsMapping = false
  }
}

struct PhaseMetrics {
  let relativeL2: Double
  let maximumAbsoluteRadians: Double
  let meanAbsoluteRadians: Double
}

func elapsedMilliseconds<T>(
  _ operation: () throws -> T
) rethrows -> (T, Double) {
  let start = DispatchTime.now().uptimeNanoseconds
  let value = try operation()
  let end = DispatchTime.now().uptimeNanoseconds
  return (value, Double(end - start) / 1_000_000)
}

func percentile(_ sorted: [Double], fraction: Double) -> Double {
  let index = min(
    sorted.count - 1,
    Int((Double(sorted.count - 1) * fraction).rounded())
  )
  return sorted[index]
}

func phaseMetrics(
  object: MTLBuffer,
  reference: Data
) throws -> PhaseMetrics {
  let expectedBytes = scanPixels * MemoryLayout<Float>.stride
  guard reference.count == expectedBytes else {
    throw CocoaError(.fileReadCorruptFile)
  }
  let objectValues = object.contents().bindMemory(
    to: SIMD2<Float>.self,
    capacity: scanPixels
  )
  return reference.withUnsafeBytes { bytes in
    let referenceValues = bytes.bindMemory(to: Float.self)
    var differenceSquared = 0.0
    var referenceSquared = 0.0
    var absoluteSum = 0.0
    var maximum = 0.0
    for index in 0..<scanPixels {
      let value = objectValues[index]
      let phase = atan2(Double(value.y), Double(value.x))
      let target = Double(referenceValues[index])
      let difference = atan2(sin(phase - target), cos(phase - target))
      let absolute = abs(difference)
      differenceSquared += difference * difference
      referenceSquared += target * target
      absoluteSum += absolute
      maximum = max(maximum, absolute)
    }
    return PhaseMetrics(
      relativeL2: sqrt(differenceSquared / max(referenceSquared, 1e-30)),
      maximumAbsoluteRadians: maximum,
      meanAbsoluteRadians: absoluteSum / Double(scanPixels)
    )
  }
}

let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.count >= 3 else {
  fatalError(
    "Usage: metal-ssb-benchmark METADATA_JSON FULL_BF_U8 REFERENCE_PHASE_F32 [warm_iterations] [cache_budget_bytes|full] [fit_trials] [fit_repeats]"
  )
}
let metadataPath = arguments[0]
let sourcePath = arguments[1]
let referencePath = arguments[2]
let warmIterations = arguments.count > 3 ? Int(arguments[3]) ?? 7 : 7
let cacheBudgetBytes: Int?
if arguments.count > 4, arguments[4] != "full" {
  guard let parsed = Int(arguments[4]), parsed >= 0 else {
    fatalError("cache_budget_bytes must be a nonnegative integer or 'full'")
  }
  cacheBudgetBytes = parsed
} else {
  cacheBudgetBytes = nil
}
let fitTrials = arguments.count > 5 ? Int(arguments[5]) ?? 0 : 0
let fitRepeats = arguments.count > 6 ? Int(arguments[6]) ?? 1 : 1
guard warmIterations > 0 else {
  fatalError("warm_iterations must be positive")
}
guard fitTrials >= 0, fitRepeats > 0 else {
  fatalError("fit_trials must be nonnegative and fit_repeats must be positive")
}
guard let device = MTLCreateSystemDefaultDevice() else {
  fatalError("Metal is unavailable")
}

let metadataData = try Data(contentsOf: URL(fileURLWithPath: metadataPath))
let metadata = try JSONDecoder().decode(BenchmarkMetadata.self, from: metadataData)
print("stage=metadata-decoded")
fflush(stdout)
guard metadata.detectorBin == 1, metadata.scanCrop == nil,
  metadata.storageDType == "uint8"
else {
  fatalError("The benchmark requires native detector bin 1, no scan crop, and uint8 BF input")
}
guard metadata.angularSamplingRadians.count == 2,
  metadata.dcValue.count == 2
else {
  fatalError("The calibration metadata is incomplete")
}

let (mappedSource, mapMilliseconds) = try elapsedMilliseconds {
  try ReadOnlyMappedFile(path: sourcePath)
}
print("stage=source-mapped")
fflush(stdout)
guard mappedSource.length == metadata.source.bytes else {
  fatalError(
    "Source byte count \(mappedSource.length) does not match metadata \(metadata.source.bytes)"
  )
}
guard
  let sourceBuffer = device.makeBuffer(
    bytesNoCopy: mappedSource.pointer,
    length: mappedSource.length,
    options: .storageModeShared,
    deallocator: { address, length in
      munmap(address, length)
    }
  )
else {
  fatalError("Metal could not map the full-BF source without a copy")
}
mappedSource.relinquishOwnership()
print("stage=metal-source-buffer-created")
fflush(stdout)
let reference = try Data(
  contentsOf: URL(fileURLWithPath: referencePath),
  options: .mappedIfSafe
)

let geometry = MetalSSBGeometry(
  brightfieldKX: metadata.brightfieldKX,
  brightfieldKY: metadata.brightfieldKY,
  brightfieldAlphaSquared: metadata.brightfieldAlphaSquared,
  brightfieldAperture: metadata.brightfieldAperture,
  brightfieldCos2Phi: metadata.brightfieldCos2Phi,
  brightfieldSin2Phi: metadata.brightfieldSin2Phi,
  qxByRow: metadata.qxByRow,
  qyByColumn: metadata.qyByColumn,
  wavelengthAngstroms: metadata.wavelengthAngstroms,
  semiangleRadians: metadata.semiangleRadians,
  angularSamplingYRadians: metadata.angularSamplingRadians[0],
  angularSamplingXRadians: metadata.angularSamplingRadians[1],
  dcValue: SIMD2<Float>(metadata.dcValue[0], metadata.dcValue[1]),
  referenceRotationDegrees: metadata.rotationAngleDegrees
)
let aberrations = MetalSSBAberrations(
  c10Nanometers: metadata.aberrations.c10,
  c12Nanometers: metadata.aberrations.c12,
  phi12Radians: metadata.aberrations.phi12
)
print("stage=geometry-created")
fflush(stdout)

let engine: MetalSSBEngine
let initializationMilliseconds: Double
do {
  (engine, initializationMilliseconds) = try elapsedMilliseconds {
    try MetalSSBEngine(
      device: device,
      geometry: geometry,
      cacheBudgetBytes: cacheBudgetBytes
    )
  }
} catch {
  if case MetalSSBError.libraryCompilation(let message) = error {
    fputs("stage=engine-initialization-error detail=\(message)\n", stderr)
  } else {
    fputs("stage=engine-initialization-error\n", stderr)
  }
  exit(2)
}
print("stage=engine-initialized")
fflush(stdout)
guard engine.logicalBrightfieldCount == metadata.logicalBrightfieldCount,
  engine.executedBrightfieldCount == metadata.apertureActiveBrightfieldCount
else {
  fatalError(
    "Active BF mismatch: engine \(engine.executedBrightfieldCount), metadata \(metadata.apertureActiveBrightfieldCount)"
  )
}
let (_, prepareMilliseconds) = try elapsedMilliseconds {
  try engine.prepare(brightfield: sourceBuffer)
}
print("stage=source-prepared")
fflush(stdout)

let first = try engine.reconstruct(aberrations: aberrations)
let firstMetrics = try phaseMetrics(object: first.object, reference: reference)
var warmWallMilliseconds: [Double] = []
var warmGPUMilliseconds: [Double] = []
var finalMetrics = firstMetrics
for _ in 0..<warmIterations {
  let result = try engine.reconstruct(aberrations: aberrations)
  warmWallMilliseconds.append(result.wallSeconds * 1_000)
  warmGPUMilliseconds.append(result.gpuSeconds * 1_000)
  finalMetrics = try phaseMetrics(object: result.object, reference: reference)
}
let (firstLoss, firstLossCallMilliseconds) = try elapsedMilliseconds {
  try engine.phaseVariance(aberrations: aberrations)
}
var warmLossWallMilliseconds: [Double] = []
var warmLossGPUMilliseconds: [Double] = []
for _ in 0..<warmIterations {
  let (measurement, callMilliseconds) = try elapsedMilliseconds {
    try engine.phaseVariance(aberrations: aberrations)
  }
  warmLossWallMilliseconds.append(callMilliseconds)
  warmLossGPUMilliseconds.append(measurement.gpuSeconds * 1_000)
}
let sortedWall = warmWallMilliseconds.sorted()
let sortedGPU = warmGPUMilliseconds.sorted()
let sortedLossWall = warmLossWallMilliseconds.sorted()
let sortedLossGPU = warmLossGPUMilliseconds.sorted()
var fitResults: [SSBOptimizationResult] = []
if fitTrials > 0 {
  for _ in 0..<fitRepeats {
    fitResults.append(
      try engine.optimize(
        start: aberrations,
        globalTrials: fitTrials,
        seed: 42
      ))
  }
}

print("device=\(device.name)")
print("metadata_schema=\(metadata.schema) dataset=\(metadata.dataset)")
print("source_path=\(sourcePath)")
print("source_expected_filename=\(metadata.source.expectedFilename)")
print("source_bytes=\(metadata.source.bytes) source_sha256=\(metadata.source.sha256)")
print("source_layout=\(metadata.source.layout) source_dtype=uint8")
print("reference_path=\(referencePath)")
print("reference_backend=\(metadata.reference.backend)")
print("reference_sha256=\(metadata.reference.sha256)")
print("scan_shape=512x512 scan_bin=1 scan_crop=none detector_bin=1")
print(
  "logical_bf=\(engine.logicalBrightfieldCount) executed_bf=\(engine.executedBrightfieldCount) zero_aperture_bf=\(engine.logicalBrightfieldCount - engine.executedBrightfieldCount)"
)
print(
  "cached_bf=\(first.provenance.cachedBrightfieldCount) streamed_bf=\(first.provenance.streamedBrightfieldCount) cache_bytes=\(first.provenance.cacheBytes)"
)
print("cache_budget_bytes=\(cacheBudgetBytes.map(String.init) ?? "full")")
print(String(format: "file_map_ms=%.3f", mapMilliseconds))
print(String(format: "engine_initialization_ms=%.3f", initializationMilliseconds))
print(String(format: "prepare_wall_ms=%.3f", prepareMilliseconds))
print(String(format: "first_reconstruct_wall_ms=%.3f", first.wallSeconds * 1_000))
print(String(format: "first_reconstruct_gpu_ms=%.3f", first.gpuSeconds * 1_000))
print(String(format: "warm_reconstruct_wall_p50_ms=%.3f", percentile(sortedWall, fraction: 0.50)))
print(String(format: "warm_reconstruct_wall_p95_ms=%.3f", percentile(sortedWall, fraction: 0.95)))
print(String(format: "warm_reconstruct_wall_max_ms=%.3f", sortedWall.last!))
print(String(format: "warm_reconstruct_gpu_p50_ms=%.3f", percentile(sortedGPU, fraction: 0.50)))
print(String(format: "warm_reconstruct_gpu_p95_ms=%.3f", percentile(sortedGPU, fraction: 0.95)))
print(String(format: "warm_reconstruct_gpu_max_ms=%.3f", sortedGPU.last!))
print(String(format: "first_phase_relative_l2=%.9g", firstMetrics.relativeL2))
print(String(format: "first_phase_max_abs_rad=%.9g", firstMetrics.maximumAbsoluteRadians))
print(String(format: "warm_phase_relative_l2=%.9g", finalMetrics.relativeL2))
print(String(format: "warm_phase_max_abs_rad=%.9g", finalMetrics.maximumAbsoluteRadians))
print(String(format: "warm_phase_mean_abs_rad=%.9g", finalMetrics.meanAbsoluteRadians))
print(String(format: "phase_variance=%.9g", firstLoss.loss))
print(String(format: "first_phase_variance_call_wall_ms=%.3f", firstLossCallMilliseconds))
print(String(format: "first_phase_variance_kernel_wall_ms=%.3f", firstLoss.wallSeconds * 1_000))
print(String(format: "first_phase_variance_gpu_ms=%.3f", firstLoss.gpuSeconds * 1_000))
print(
  String(format: "warm_phase_variance_wall_p50_ms=%.3f", percentile(sortedLossWall, fraction: 0.50))
)
print(
  String(format: "warm_phase_variance_wall_p95_ms=%.3f", percentile(sortedLossWall, fraction: 0.95))
)
print(String(format: "warm_phase_variance_wall_max_ms=%.3f", sortedLossWall.last!))
print(
  String(format: "warm_phase_variance_gpu_p50_ms=%.3f", percentile(sortedLossGPU, fraction: 0.50)))
print(
  String(format: "warm_phase_variance_gpu_p95_ms=%.3f", percentile(sortedLossGPU, fraction: 0.95)))
print(String(format: "warm_phase_variance_gpu_max_ms=%.3f", sortedLossGPU.last!))
if let firstFit = fitResults.first {
  let fitSeconds = fitResults.map(\.elapsedSeconds).sorted()
  let deterministic = fitResults.dropFirst().allSatisfy {
    $0.best == firstFit.best && $0.loss == firstFit.loss
  }
  print("fit_trials=\(fitTrials) fit_repeats=\(fitRepeats) fit_seed=42")
  print(String(format: "fit_elapsed_p50_s=%.6f", percentile(fitSeconds, fraction: 0.50)))
  print(String(format: "fit_elapsed_p95_s=%.6f", percentile(fitSeconds, fraction: 0.95)))
  print(String(format: "fit_elapsed_max_s=%.6f", fitSeconds.last!))
  print(String(format: "fit_loss=%.12g", firstFit.loss))
  print(String(format: "fit_c10_nm=%.9g", firstFit.best.c10Nanometers))
  print(String(format: "fit_c12_nm=%.9g", firstFit.best.c12Nanometers))
  print(String(format: "fit_phi12_rad=%.9g", firstFit.best.phi12Radians))
  print("fit_refinement_evaluations=\(firstFit.refinementEvaluations)")
  print("fit_deterministic=\(deterministic)")
}
print("warm_iterations=\(warmIterations)")
