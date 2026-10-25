import Foundation
import Metal
import Metal4DSTEMStreamingIO
import MetalSSBKernels
import Native4DSTEMIO

// End-to-end native SSB fit timing on one measured acquisition: low-order branch fits, the branch test, and the
// fifth-order refinement, with the objective and every fitted parameter. The first run reads the original HDF5
// through the resident loader and writes the exact selected BF counts to --cache; later runs map that cache.
//
//   metal-ssb-fit-benchmark --input MASTER.h5 --cache DIR --scan-step A [--kv 300] [--semiangle 30]
//     [--index DIR] [--max-order 5] [--trials 200] [--repeats 1] [--output RESULT.json]
//     [--search newton|gradient] [--fifth-only BASELINE.json [--gradient-check]]
// --fifth-only starts the fifth-order stage at the kept low-order fit and rotation of an earlier result;
// --gradient-check compares the exact gradient with central differences there and at the earlier optimum.
//     [--compare-standard] [--dark-field DARK.f32] [--verify-branches] [--check-maximum]
//   metal-ssb-fit-benchmark --input MASTER.h5 --cache DIR --export-dark-field DARK.f32
//
// --max-order 1 stops after the low-order branch fits and the branch test. --dark-field passes a float32
// scan-shaped dark-field image to the branch test (written by --export-dark-field from the original HDF5 with
// the app's mask: valid detector pixels at or beyond 1.1 BF radii). --verify-branches also searches the
// branch that was mapped from the fitted one, and reports both points' agreements and phases and the branch
// decision taken with the searched fit instead. --check-maximum evaluates the kept low-order fit's neighbours
// one stencil-Newton tolerance away along each Newton coordinate (C10, C12 x/y, tilt row/column, depth), and
// with --baseline RESULT.json the same for that run's kept low-order fit.
//   metal-ssb-fit-benchmark --cache DIR --direct-parity LIMITS.json --cuda-dir DIR [--limits 2,10] [--factors 1,3]
//     [--output PARITY.json] [--phase-out DIR]
//     ssb, phase and complex of the limits' optics against CUDA's arrays, with timings (DirectParity.swift).
//   metal-ssb-fit-benchmark --cache FIXTURE --fixture-parity FIXTURE [--oracle-dir DIR] [--optics NAME]
//     [--methods ssb,phase,complex] [--output PARITY.json] [--phase-out DIR]
//     every reconstruction of a fixture's references.json natively, with per-setting parity and timing.
//   metal-ssb-fit-benchmark --cache DIR --objective-bench RUN.json [--repeats 3] [--output BENCH.json]
//     [--search-kernels-only]
//     times the agreement objectives at fixed candidates instead of fitting (ObjectiveBenchmark.swift).

struct Options {
  var input: URL?
  var cache: URL
  var index: URL?
  var scanStep = 0.0
  var voltageKV = 300.0
  var semiangleMrad = 30.0
  var maximumOrder = 5
  var trials = 200
  var repeats = 1
  var output: URL?
  var dump: URL?
  var search = SSBFitOptions().higherOrderSearch
  var fifthOnly: URL?
  var gradientCheck = false
  var gradientSettings = SSBFifthOrderGradientSearch.Settings()
  var verbose = false
  var curvature = false
  var score: URL?
  var detectorSampling: Double?
  var center: (Double, Double)?
  var radius: Double?
  /// Use the case's exported `dark_field.f32` for the branch test unless `--no-dark-field`.
  var caseDarkField = true
  var compareStandard = false
  var darkField: URL?
  var exportDarkField: URL?
  var verifyBranches = false
  var checkMaximum = false
  var baseline: URL?
  var objectiveBench: URL?
  var phaseFrom: URL?
  var phaseOut: URL?
  var phaseLabel = "run"
  var limitSweep: URL?
  var limits: [Double] = [2, 3, 4, 6, 10]
  /// `--objective-bench` times only the exact-gradient search's calls.
  var searchKernelsOnly = false
  var directParity: URL?
  var cudaDirectory: URL?
  var directFactors: [Int]?
  var fixtureParity: URL?
  var oracleDirectory: URL?
  var opticsName: String?
  /// `--fixture-parity` only: the reconstruction methods to run (default all).
  var methods: Set<String>?

  init() throws {
    var args = Array(CommandLine.arguments.dropFirst())
    var cache: URL?
    func value(_ flag: String) throws -> String {
      guard !args.isEmpty else { throw failure("Missing value for \(flag)") }
      return args.removeFirst()
    }
    while !args.isEmpty {
      let flag = args.removeFirst()
      switch flag {
      case "--input": input = URL(fileURLWithPath: try value(flag))
      case "--cache": cache = URL(fileURLWithPath: try value(flag))
      case "--index": index = URL(fileURLWithPath: try value(flag))
      case "--scan-step": scanStep = Double(try value(flag)) ?? 0
      case "--kv": voltageKV = Double(try value(flag)) ?? 0
      case "--semiangle": semiangleMrad = Double(try value(flag)) ?? 0
      case "--max-order": maximumOrder = Int(try value(flag)) ?? 5
      case "--trials": trials = Int(try value(flag)) ?? 200
      case "--repeats": repeats = Int(try value(flag)) ?? 1
      case "--output": output = URL(fileURLWithPath: try value(flag))
      case "--dump": dump = URL(fileURLWithPath: try value(flag))
      case "--search":
        switch try value(flag) {
        case "newton": search = .finiteDifferenceNewton
        case "gradient": search = .exactGradient
        case "gradient-coarse": search = .exactGradientCoarseToFine
        case let other:
          throw failure("Unknown search \(other); use newton, gradient or gradient-coarse")
        }
      case "--fifth-only": fifthOnly = URL(fileURLWithPath: try value(flag))
      case "--gradient-check": gradientCheck = true
      case "--hessian-stride": gradientSettings.hessianBFStride = Int(try value(flag)) ?? 64
      case "--initial-radius": gradientSettings.initialRadius = Double(try value(flag)) ?? 1
      case "--max-radius": gradientSettings.maxRadius = Double(try value(flag)) ?? 64
      case "--gain-tolerance": gradientSettings.gainTolerance = Double(try value(flag)) ?? 1e-8
      case "--coarse-stride": gradientSettings.coarseBFStride = Int(try value(flag)) ?? 1
      case "--coarse-tolerance": gradientSettings.coarseTolerance = Double(try value(flag)) ?? 1e-7
      case "--fine-radius": gradientSettings.fineRadius = Double(try value(flag)) ?? 0.1
      case "--update":
        gradientSettings.update = SSBFifthOrderGradientSearch.Update(rawValue: try value(flag)) ?? .bfgs
      case "--verbose": verbose = true
      case "--curvature": curvature = true
      case "--score": score = URL(fileURLWithPath: try value(flag))
      case "--detector-sampling": detectorSampling = Double(try value(flag))
      case "--center":
        let parts = try value(flag).split(separator: ",").compactMap { Double($0) }
        guard parts.count == 2 else { throw failure("--center takes ROW,COLUMN") }
        center = (parts[0], parts[1])
      case "--radius": radius = Double(try value(flag))
      case "--no-dark-field": caseDarkField = false
      case "--compare-standard": compareStandard = true
      case "--dark-field": darkField = URL(fileURLWithPath: try value(flag))
      case "--export-dark-field": exportDarkField = URL(fileURLWithPath: try value(flag))
      case "--verify-branches": verifyBranches = true
      case "--check-maximum": checkMaximum = true
      case "--baseline": baseline = URL(fileURLWithPath: try value(flag))
      case "--objective-bench": objectiveBench = URL(fileURLWithPath: try value(flag))
      case "--phase-from": phaseFrom = URL(fileURLWithPath: try value(flag))
      case "--phase-out": phaseOut = URL(fileURLWithPath: try value(flag))
      case "--phase-label": phaseLabel = try value(flag)
      case "--limit-sweep": limitSweep = URL(fileURLWithPath: try value(flag))
      case "--limits": limits = try value(flag).split(separator: ",").compactMap { Double($0) }
      case "--search-kernels-only": searchKernelsOnly = true
      case "--direct-parity": directParity = URL(fileURLWithPath: try value(flag))
      case "--cuda-dir": cudaDirectory = URL(fileURLWithPath: try value(flag))
      case "--fixture-parity": fixtureParity = URL(fileURLWithPath: try value(flag))
      case "--oracle-dir": oracleDirectory = URL(fileURLWithPath: try value(flag))
      case "--optics": opticsName = try value(flag)
      case "--methods": methods = Set(try value(flag).split(separator: ",").map(String.init))
      case "--factors": directFactors = try value(flag).split(separator: ",").compactMap { Int($0) }
      default: throw failure("Unknown argument \(flag)")
      }
    }
    guard let cache else {
      throw failure(
        "Usage: metal-ssb-fit-benchmark --input MASTER.h5 --cache DIR --scan-step A [--kv 300] [--semiangle 30] [--max-order 5] [--trials 200] [--repeats 1] [--output JSON]"
      )
    }
    self.cache = cache
  }
}

struct Failure: Error, CustomStringConvertible { let description: String }
func failure(_ message: String) -> Failure { Failure(description: message) }

/// The exported, exact selected-BF evidence of one acquisition.
struct CaseRecord: Codable {
  var source: String
  var scanRows: Int
  var scanColumns: Int
  var detectorRows: Int
  var detectorColumns: Int
  var calibration: MetalSSBCalibration
  var excludedPixels: [Int]
  var pixels: [Int]
  var countType: Int
  var rotationEstimateDegrees: Double
  var exportSeconds: Double
  /// Median-filled camera-flagged pixels; nil for the first export format (flagged pixels excluded).
  var flaggedPixels: [Int]? = nil
  var evidence: String? = nil
}

func log(_ message: String) {
  FileHandle.standardError.write(Data((message + "\n").utf8))
}

/// The app's flagged-pixel handling (Live4DSTEM `SSBFlaggedPixels`, Python `io.load`): each flagged pixel takes,
/// at every scan position, the median of the valid pixels in the smallest square around it that has any.
struct FlaggedPixels {
  var pixels: [Int] = []
  var neighbors: [[Int]] = []

  init(validity: [UInt8], rows: Int, columns: Int) {
    for pixel in validity.indices where validity[pixel] == 0 {
      let row = pixel / columns
      let column = pixel % columns
      var radius = 1
      while true {
        var near: [Int] = []
        for r in max(0, row - radius)...min(rows - 1, row + radius) {
          for c in max(0, column - radius)...min(columns - 1, column + radius)
          where validity[r * columns + c] != 0 {
            near.append(r * columns + c)
          }
        }
        if !near.isEmpty {
          neighbors.append(near)
          break
        }
        radius += 1
      }
      pixels.append(pixel)
    }
  }

  static func median(_ values: inout [UInt32]) -> UInt32 {
    values.sort()
    let total = UInt64(values[(values.count - 1) / 2]) + UInt64(values[values.count / 2])
    let half = total >> 1
    return UInt32(half + (total & half & 1))
  }
}

/// Exact detector columns (uint32, plane-major) of the given pixels.
func readColumns(
  _ pixels: [Int], resident: MetalCompactH5ResidentSource, device: MTLDevice, plane: Int
) throws -> [[UInt32]] {
  guard let queue = device.makeCommandQueue(),
    let staging = device.makeBuffer(length: 32 * plane * 4, options: .storageModeShared)
  else { throw failure("Metal staging unavailable") }
  var columns: [[UInt32]] = []
  columns.reserveCapacity(pixels.count)
  for first in stride(from: 0, to: pixels.count, by: 32) {
    let batch = Array(pixels[first..<min(first + 32, pixels.count)])
    guard let commands = queue.makeCommandBuffer() else { throw failure("Metal command unavailable") }
    try resident.encodeDetectorColumns(pixels: batch, into: staging, commands: commands)
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error { throw error }
    let values = staging.contents().assumingMemoryBound(to: UInt32.self)
    for index in batch.indices {
      columns.append(Array(UnsafeBufferPointer(start: values + index * plane, count: plane)))
    }
  }
  return columns
}

/// Read the original acquisition once and keep its exact selected BF columns beside a case record, prepared as
/// the app prepares SSB: flagged pixels median-filled and kept, the BF disk measured on the filled totals, the
/// rotation from the marker-free filled moments, and the dark field of the branch test.
func export(options: Options, device: MTLDevice) throws {
  guard let input = options.input else {
    throw failure("No cache at \(options.cache.path); pass --input MASTER.h5 to create it")
  }
  guard options.scanStep > 0 else { throw failure("Pass --scan-step in Angstrom") }
  let started = Date()
  let index =
    options.index ?? options.cache.appendingPathComponent("index", isDirectory: true)
  try FileManager.default.createDirectory(at: index, withIntermediateDirectories: true)
  let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: index).prepare(input: input)
  guard let dataset = catalog.datasets.first else { throw failure("No acquisition in \(input.path)") }
  let indexed = try Native4DSTEMIndexedSource.open(dataset: dataset)
  let resident = try MetalCompactH5Loader.load(source: indexed, device: device)
  defer { resident.releaseResidentStorage() }
  let metadata = resident.metadata
  let plane = metadata.scanCount
  let rows = metadata.detectorRows
  let columns = metadata.detectorColumns
  log("loaded \(metadata.scanRows)x\(metadata.scanColumns) scan, \(rows)x\(columns) detector in \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
  let raw = try resident.meanDiffractionPattern().detectorSum
  // Camera-flagged pixels hold the dtype's maximum in every frame.
  let markers: [UInt64] = [255, 65535, 4_294_967_295]
  let validity = raw.map { total in markers.contains { $0 * UInt64(plane) == total } ? UInt8(0) : 1 }
  let flagged = FlaggedPixels(validity: validity, rows: rows, columns: columns)
  let neighborPixels = Array(Set(flagged.neighbors.flatMap { $0 })).sorted()
  let neighborColumns = Dictionary(
    uniqueKeysWithValues: zip(
      neighborPixels,
      try readColumns(neighborPixels, resident: resident, device: device, plane: plane)))
  let filled: [[UInt32]] = flagged.neighbors.map { near in
    var values = [UInt32](repeating: 0, count: near.count)
    return (0..<plane).map { position in
      for (slot, pixel) in near.enumerated() { values[slot] = neighborColumns[pixel]![position] }
      return FlaggedPixels.median(&values)
    }
  }
  var detectorSum = raw
  for (pixel, column) in zip(flagged.pixels, filled) {
    detectorSum[pixel] = column.reduce(UInt64(0)) { $0 + UInt64($1) }
  }
  log("median-filled \(flagged.pixels.count) flagged detector pixels \(flagged.pixels)")
  let excluded = Set(metadata.excludedDetectorPixels)
  let disk = try MetalSSBBrightfieldEstimate.measure(
    device: device, detectorSum: detectorSum, rows: rows, columns: columns, excluded: excluded)
  let sampling = try options.detectorSampling ?? disk.detectorSampling(semiangle: options.semiangleMrad)
  let calibration = MetalSSBCalibration(
    voltageKV: options.voltageKV, semiangleMrad: options.semiangleMrad,
    scanSamplingRowAngstroms: options.scanStep, scanSamplingColumnAngstroms: options.scanStep,
    detectorSamplingRowMrad: sampling, detectorSamplingColumnMrad: sampling,
    brightfieldCenterRowPixels: options.center?.0 ?? disk.centerRow,
    brightfieldCenterColumnPixels: options.center?.1 ?? disk.centerColumn,
    brightfieldRadiusPixels: options.radius ?? disk.selectionRadius)
  let setup = try calibration.geometry(
    detectorRows: rows, detectorColumns: columns, detectorSum: detectorSum, excludedPixels: excluded,
    scanRows: metadata.scanRows, scanColumns: metadata.scanColumns)
  // Rotation from exact moments without the markers, with each flagged pixel's filled column
  // (Live4DSTEM `SSBMeasuredEvidence.rotation`).
  guard let moments = try resident.preparedDPCMomentValues() else {
    throw failure("The resident has no exact DPC moments for the rotation estimate")
  }
  var total = moments.total
  var rowMoment = moments.detectorRowMoment
  var columnMoment = moments.detectorColumnMoment
  for (slot, pixel) in flagged.pixels.enumerated() {
    let marker = raw[pixel] / UInt64(plane)
    let r = UInt64(pixel / columns)
    let c = UInt64(pixel % columns)
    for position in 0..<plane {
      let value = UInt64(filled[slot][position])
      total[position] = total[position] - marker + value
      rowMoment[position] = rowMoment[position] - marker * r + value * r
      columnMoment[position] = columnMoment[position] - marker * c + value * c
    }
  }
  let com = SSBRotationBranches.centerOfMass(
    total: total, rowMoment: rowMoment, columnMoment: columnMoment)
  let rotation = SSBRotationBranches.curlRotationDegrees(
    row: com.row, column: com.column, rows: metadata.scanRows, columns: metadata.scanColumns)
  // Dark field: every count at or beyond 1.1 BF radii (float32 distances, Python `_dark_field_of`).
  let inner = Float(
    SSBRotationBranches.darkFieldInnerRadius * calibration.semiangleMrad
      / min(calibration.detectorSamplingRowMrad, calibration.detectorSamplingColumnMrad))
  // Summed directly over the dark-field pixels (flagged ones as filled), independent of the DPC moments.
  let unfilled = excluded.subtracting(flagged.pixels)
  let darkPixels = (0..<(rows * columns)).filter { pixel in
    let dr = Float(pixel / columns) - Float(calibration.brightfieldCenterRowPixels)
    let dc = Float(pixel % columns) - Float(calibration.brightfieldCenterColumnPixels)
    return (dr * dr + dc * dc).squareRoot() >= inner && !unfilled.contains(pixel)
  }
  let filledByPixel = Dictionary(uniqueKeysWithValues: zip(flagged.pixels, filled))
  var dark = [Double](repeating: 0, count: plane)
  for first in stride(from: 0, to: darkPixels.count, by: 1024) {
    let batch = Array(darkPixels[first..<min(first + 1024, darkPixels.count)])
    let read = try readColumns(batch, resident: resident, device: device, plane: plane)
    for (pixel, column) in zip(batch, read) {
      let values = filledByPixel[pixel] ?? column
      for position in 0..<plane { dark[position] += Double(values[position]) }
    }
  }
  // Exact BF columns (flagged pixels as filled), narrowed only when every count fits.
  let selected = try readColumns(setup.pixels, resident: resident, device: device, plane: plane)
  var wide = [UInt32]()
  wide.reserveCapacity(setup.pixels.count * plane)
  for (pixel, column) in zip(setup.pixels, selected) { wide += filledByPixel[pixel] ?? column }
  let maximum = wide.max() ?? 0
  try FileManager.default.createDirectory(at: options.cache, withIntermediateDirectories: true)
  let countType: MetalSSBCountType = maximum <= 255 ? .uint8 : (maximum <= 65535 ? .uint16 : .uint32)
  let countsURL = options.cache.appendingPathComponent("counts.bin")
  switch countType {
  case .uint8: try Data(wide.map { UInt8($0) }.withUnsafeBytes { Data($0) }).write(to: countsURL)
  case .uint16: try Data(wide.map { UInt16($0) }.withUnsafeBytes { Data($0) }).write(to: countsURL)
  case .uint32: try wide.withUnsafeBytes { Data($0) }.write(to: countsURL)
  }
  try dark.map(Float.init).withUnsafeBytes { Data($0) }.write(
    to: options.cache.appendingPathComponent("dark_field.f32"))
  let record = CaseRecord(
    source: input.path, scanRows: metadata.scanRows, scanColumns: metadata.scanColumns,
    detectorRows: rows, detectorColumns: columns,
    calibration: calibration, excludedPixels: excluded.sorted(), pixels: setup.pixels,
    countType: countType.rawValue, rotationEstimateDegrees: rotation,
    exportSeconds: Date().timeIntervalSince(started), flaggedPixels: flagged.pixels,
    evidence: "app-v2: flagged median fill, marker-free moments, dark field")
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
  try encoder.encode(record).write(to: options.cache.appendingPathComponent("case.json"))
  try detectorSum.withUnsafeBytes { Data($0) }.write(
    to: options.cache.appendingPathComponent("detector_sum.u64"))
  log(
    "exported \(setup.pixels.count) BF columns (max count \(maximum), \(countType)) rotation \(rotation) deg in \(String(format: "%.1f", record.exportSeconds)) s"
  )
}

/// The app's branch-test dark field (`SSBMeasuredEvidence.darkFieldMask`): exact counts of the valid detector
/// pixels at or beyond 1.1 BF radii from the calibrated centre, per scan position, as float32.
func exportDarkField(options: Options, record: CaseRecord, to url: URL, device: MTLDevice) throws {
  guard let input = options.input else { throw failure("--export-dark-field needs --input MASTER.h5") }
  let started = Date()
  let index = options.index ?? options.cache.appendingPathComponent("index", isDirectory: true)
  let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: index).prepare(input: input)
  guard let dataset = catalog.datasets.first else { throw failure("No acquisition in \(input.path)") }
  let resident = try MetalCompactH5Loader.load(
    source: try Native4DSTEMIndexedSource.open(dataset: dataset), device: device)
  defer { resident.releaseResidentStorage() }
  let metadata = resident.metadata
  guard metadata.scanRows == record.scanRows, metadata.scanColumns == record.scanColumns,
    metadata.detectorRows == record.detectorRows, metadata.detectorColumns == record.detectorColumns
  else { throw failure("\(input.path) is not the acquisition of \(options.cache.path)") }
  let calibration = record.calibration
  let radius =
    calibration.semiangleMrad
    / min(calibration.detectorSamplingRowMrad, calibration.detectorSamplingColumnMrad)
  let inner = Float(SSBRotationBranches.darkFieldInnerRadius * radius)
  let excluded = Set(record.excludedPixels)
  let mask = (0..<(record.detectorRows * record.detectorColumns)).map { pixel -> UInt8 in
    let dr = Float(pixel / record.detectorColumns) - Float(calibration.brightfieldCenterRowPixels)
    let dc = Float(pixel % record.detectorColumns) - Float(calibration.brightfieldCenterColumnPixels)
    return (dr * dr + dc * dc).squareRoot() >= inner && !excluded.contains(pixel) ? 1 : 0
  }
  try resident.updateVirtualDetector(mask: mask)
  let values = try resident.virtualDetectorValues64().map { Float($0) }
  try values.withUnsafeBytes { Data($0) }.write(to: url)
  log(
    "exported dark field (\(mask.reduce(0) { $0 + Int($1) }) detector pixels, inner radius \(inner) px) in \(String(format: "%.1f", Date().timeIntervalSince(started))) s"
  )
}

func mapped(_ url: URL, device: MTLDevice) throws -> MTLBuffer {
  let handle = open(url.path, O_RDONLY)
  guard handle >= 0 else { throw failure("Cannot open \(url.path)") }
  defer { close(handle) }
  var info = stat()
  guard fstat(handle, &info) == 0 else { throw failure("Cannot stat \(url.path)") }
  let page = Int(getpagesize())
  let length = (Int(info.st_size) + page - 1) / page * page
  guard let pointer = mmap(nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE, handle, 0),
    pointer != MAP_FAILED
  else { throw failure("Cannot map \(url.path)") }
  guard
    let buffer = device.makeBuffer(
      bytesNoCopy: pointer, length: length, options: .storageModeShared,
      deallocator: { pointer, length in munmap(pointer, length) })
  else { throw failure("Metal cannot wrap \(url.path)") }
  return buffer
}

func record(_ fit: SSBFit) -> [String: Any] {
  let p = fit.point
  var values: [String: Any] = [
    "rotation_deg": fit.scanDetectorRotationDegrees, "C10_nm": p.c10Nanometers,
    "C12_nm": p.c12Nanometers, "phi12_deg": p.phi12Degrees, "sample_tilt_mrad": p.sampleTiltMrad,
    "depth_spread_nm": p.depthSpreadNanometers, "timings": fit.timings,
    "higher_order": p.higherOrder.map {
      [
        "order": $0.order, "symmetry": $0.symmetry, "magnitude_A": $0.magnitudeAngstroms,
        "angle_rad": $0.angleRadians,
      ] as [String: Any]
    },
  ]
  if let agreement = fit.sampleAgreement { values["sample_agreement"] = agreement }
  if let starts = fit.newtonStarts {
    values["newton_starts"] = starts.map {
      ["iterations": $0.iterations, "evaluations": $0.evaluations, "converged": $0.converged]
        as [String: Any]
    }
    values["newton_evaluations"] = starts.reduce(0) { $0 + $1.evaluations }
  }
  if let source = fit.mappedFromBranch { values["mapped_from_branch"] = source }
  if let normalization = fit.higherOrderNormalization {
    values["higher_order_normalization"] = normalization
  }
  if let refinement = fit.higherOrderRefinement {
    values["refinement"] = [
      "evaluations": refinement.evaluations, "executed_objectives": refinement.executedObjectives,
      "iterations": refinement.iterations, "stop": refinement.stop,
      "normalized_agreement": refinement.agreement,
      "coordinates": refinement.coordinates,
    ] as [String: Any]
  }
  return values
}

func branchRecord(_ branch: SSBRotationBranches.Branch) -> [String: Any] {
  var values: [String: Any] = [
    "rotation_deg": branch.scanDetectorRotationDegrees, "phase_skewness": branch.phaseSkewness,
  ]
  if let column = branch.column {
    values["column_count"] = column.count
    if let z = column.z { values["column_z"] = z }
    if let effect = column.effect { values["column_effect"] = effect }
    if let positive = column.positiveFraction { values["column_positive_fraction"] = positive }
  }
  return values
}

func phaseValues(_ phase: MetalSSBJointPhase) -> [Float] {
  Array(
    UnsafeBufferPointer(
      start: phase.phase.contents().assumingMemoryBound(to: Float.self),
      count: phase.rows * phase.columns))
}

/// Search every mapped branch at its own rotation with the same primary options and compare: the agreement of
/// both points there, the largest phase difference, and the branch decision made with the searched fit. The
/// model is left at the kept rotation.
func verifyBranches(
  _ fitted: SSBBranchedFit, model: MetalSSBJointModel, engine: MetalSSBEngine, darkField: [Float]?,
  sampling: (Double, Double), trials: Int
) throws -> [String: Any] {
  var report: [String: Any] = [:]
  var phases: [[Float]] = []
  var size = 0
  var searchedFits = fitted.fits
  for (index, fit) in fitted.fits.enumerated() {
    try model.retarget(scanDetectorRotationDegrees: fit.scanDetectorRotationDegrees)
    let phase = try model.reconstruct(fit.point, upsamplingFactor: 1)
    size = phase.rows
    guard let source = fit.mappedFromBranch else {
      phases.append(phaseValues(phase))
      continue
    }
    let started = Date()
    let searched = try SSBModelFit.fit(
      model: model, engine: engine,
      options: SSBFitOptions(trials: trials, compareStandard: false, maximumOrder: 1))
    let seconds = Date().timeIntervalSince(started)
    searchedFits[index] = searched
    let agreements = try model.agreements([fit.point, searched.point])
    let searchedPhase = phaseValues(try model.reconstruct(searched.point, upsamplingFactor: 1))
    let mappedPhase = phaseValues(phase)
    let largest = zip(mappedPhase, searchedPhase).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
    phases.append(searchedPhase)
    // The source point's agreement at its own rotation, against its image here.
    try model.retarget(scanDetectorRotationDegrees: fitted.fits[source].scanDetectorRotationDegrees)
    let sourceAgreement = try model.agreements([fitted.fits[source].point])[0]
    report["branch_\(index)"] = [
      "mapped_from": source, "mapped": record(fit), "searched": record(searched),
      "searched_s": seconds, "mapped_agreement": agreements[0], "searched_agreement": agreements[1],
      "source_agreement_at_source_rotation": sourceAgreement,
      "mapped_minus_source_relative": (agreements[0] - sourceAgreement) / sourceAgreement,
      "searched_minus_mapped_relative": (agreements[1] - agreements[0]) / agreements[0],
      "max_abs_phase_difference_rad": Double(largest),
    ] as [String: Any]
  }
  let decision = SSBRotationBranches.compare(
    phases: phases, rotations: fitted.fits.map(\.scanDetectorRotationDegrees), rows: size,
    columns: size, darkField: darkField, scanSampling: sampling)
  report["searched_decision"] = [
    "kept": decision.kept, "chosen": decision.chosen, "decided_by": decision.decidedBy.rawValue,
    "columns": decision.columns, "branch_tests": decision.branches.map(branchRecord),
  ] as [String: Any]
  report["decision_unchanged"] =
    decision.kept == fitted.kept && decision.decidedBy == fitted.decidedBy
  try model.retarget(scanDetectorRotationDegrees: fitted.fits[fitted.kept].scanDetectorRotationDegrees)
  return report
}

/// The exact objective inputs at the kept rotation, with the baseline start and result, for an independent
/// reference implementation (float32 little-endian arrays).
func dumpObjective(model: MetalSSBJointModel, fit: SSBBranchedFit, to directory: URL) throws {
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  let arrays = try model.objectiveArrays()
  func write<T>(_ values: [T], _ name: String) throws {
    try values.withUnsafeBytes { Data($0) }.write(to: directory.appendingPathComponent(name))
  }
  try write(arrays.k, "k_f32x2.bin")
  try write(arrays.q, "q_f32x2.bin")
  try write(arrays.weights, "weights_f32.bin")
  try write(arrays.probe, "probe_f32x4.bin")
  try write(arrays.evidence, "evidence_c64.bin")
  var summary: [String: Any] = ["constants": arrays.constants]
  summary["initial"] = record(fit.fits[fit.kept])
  summary["fit"] = record(fit.fit)
  let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
  try data.write(to: directory.appendingPathComponent("objective.json"))
  log("dumped objective arrays to \(directory.path)")
}

let options = try Options()
guard let device = MTLCreateSystemDefaultDevice() else { throw failure("Metal is unavailable") }
let caseURL = options.cache.appendingPathComponent("case.json")
if !FileManager.default.fileExists(atPath: caseURL.path) {
  try export(options: options, device: device)
}
let loadStarted = Date()
let caseRecord = try JSONDecoder().decode(CaseRecord.self, from: Data(contentsOf: caseURL))
if let target = options.exportDarkField {
  try exportDarkField(options: options, record: caseRecord, to: target, device: device)
  exit(0)
}
let caseDarkFieldURL = options.cache.appendingPathComponent("dark_field.f32")
let darkFieldURL =
  options.darkField
  ?? (options.caseDarkField && FileManager.default.fileExists(atPath: caseDarkFieldURL.path)
    ? caseDarkFieldURL : nil)
let darkField: [Float]? = try darkFieldURL.map { url in
  let values = try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  guard values.count == caseRecord.scanRows * caseRecord.scanColumns else {
    throw failure("\(url.path) is not a \(caseRecord.scanRows)x\(caseRecord.scanColumns) float32 image")
  }
  return values
}
let detectorSum = try Data(contentsOf: options.cache.appendingPathComponent("detector_sum.u64"))
  .withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
let setup = try caseRecord.calibration.geometry(
  detectorRows: caseRecord.detectorRows, detectorColumns: caseRecord.detectorColumns,
  detectorSum: detectorSum, excludedPixels: Set(caseRecord.excludedPixels),
  scanRows: caseRecord.scanRows, scanColumns: caseRecord.scanColumns)
guard setup.pixels == caseRecord.pixels else { throw failure("BF selection differs from the cache") }
let counts = try mapped(options.cache.appendingPathComponent("counts.bin"), device: device)
guard let countType = MetalSSBCountType(rawValue: caseRecord.countType) else {
  throw failure("Unknown count type")
}
if let fixture = options.fixtureParity {
  try fixtureParity(
    directory: fixture, oracleDirectory: options.oracleDirectory, output: options.output, phaseOut: options.phaseOut,
    device: device, calibration: caseRecord.calibration, geometry: setup.geometry, counts: counts,
    countType: countType, rotationEstimate: caseRecord.rotationEstimateDegrees, only: options.opticsName,
    methods: options.methods)
  exit(0)
}
if let sweep = options.directParity {
  try directParity(
    sweep: sweep, limits: options.limits, cudaDirectory: options.cudaDirectory ?? sweep.deletingLastPathComponent(),
    output: options.output, phaseOut: options.phaseOut, device: device, calibration: caseRecord.calibration,
    geometry: setup.geometry, counts: counts, countType: countType,
    rotationEstimate: caseRecord.rotationEstimateDegrees, factors: options.directFactors)
  exit(0)
}
if let run = options.limitSweep, let directory = options.phaseOut {
  try limitSweep(
    run: run, limits: options.limits, label: options.phaseLabel, directory: directory, device: device,
    calibration: caseRecord.calibration, geometry: setup.geometry, counts: counts,
    countType: countType, rotationEstimate: caseRecord.rotationEstimateDegrees, search: options.search)
  exit(0)
}
if let run = options.phaseFrom, let directory = options.phaseOut {
  try exportPhases(
    run: run, label: options.phaseLabel, directory: directory, device: device,
    calibration: caseRecord.calibration, geometry: setup.geometry, counts: counts,
    countType: countType, rotationEstimate: caseRecord.rotationEstimateDegrees)
  exit(0)
}
if let run = options.objectiveBench {
  try objectiveBenchmark(
    run: run, device: device, geometry: setup.geometry, counts: counts, countType: countType,
    repeats: options.repeats, searchKernelsOnly: options.searchKernelsOnly, output: options.output)
  exit(0)
}
var results: [[String: Any]] = []
if let baselineURL = options.fifthOnly {
  results = try fifthOnly(
    baselineURL: baselineURL, options: options, device: device, setup: setup, counts: counts,
    countType: countType, rotationEstimate: caseRecord.rotationEstimateDegrees)
}
for repeatIndex in 0..<(options.fifthOnly == nil ? options.repeats : 0) {
  try autoreleasepool {
    let prepareStarted = Date()
    let engine = try MetalSSBEngine(device: device, geometry: setup.geometry)
    try engine.prepare(brightfield: counts, countType: countType)
    let prepareSeconds = Date().timeIntervalSince(prepareStarted)
    let model = try engine.jointModel(
      scanDetectorRotationDegrees: caseRecord.rotationEstimateDegrees)
    let modelSeconds = Date().timeIntervalSince(prepareStarted) - prepareSeconds
    var stages: [(String, Double)] = []
    var lastStage = ""
    let fitStarted = Date()
    var initialSeconds = 0.0
    var options5 = SSBFitOptions(
      trials: options.trials, compareStandard: options.compareStandard,
      maximumOrder: options.maximumOrder)
    options5.higherOrderSearch = options.search
    let sampling = (
      caseRecord.calibration.scanSamplingRowAngstroms,
      caseRecord.calibration.scanSamplingColumnAngstroms
    )
    let fitted = try SSBModelFit.findAberrations(
      model: model, engine: engine, rotationDegrees: caseRecord.rotationEstimateDegrees,
      darkField: darkField, scanSampling: sampling, options: options5,
      progress: { stage, _, _ in
        if stage != lastStage {
          stages.append((stage, Date().timeIntervalSince(fitStarted)))
          lastStage = stage
        }
      },
      onInitialFit: { _ in initialSeconds = Date().timeIntervalSince(fitStarted) })
    let totalSeconds = Date().timeIntervalSince(fitStarted)
    var result: [String: Any] = [
      "repeat": repeatIndex, "device": device.name, "source": caseRecord.source,
      "bf_pixels": setup.pixels.count, "band_count": model.bandCount,
      "rotation_estimate_deg": caseRecord.rotationEstimateDegrees,
      "dark_field": darkField != nil, "evidence": caseRecord.evidence ?? "v1: flagged excluded",
      "prepare_s": prepareSeconds, "joint_model_s": modelSeconds,
      "initial_fit_s": initialSeconds, "fit_total_s": totalSeconds,
      "stages": stages.map { ["stage": $0.0, "start_s": $0.1] as [String: Any] },
      "kept": fitted.kept, "decided_by": fitted.decidedBy.rawValue,
      "fit": record(fitted.fit), "branches": fitted.fits.map(record),
    ]
    if let initial = fitted.refinedFit == nil ? nil : fitted.fits[fitted.kept] as SSBFit? {
      result["initial"] = record(initial)
    }
    result["branch_tests"] = fitted.branches.map(branchRecord)
    result["column_count"] = fitted.columnCount
    func neighbours(_ p: SSBModelPoint, rotation: Double) throws -> [String: Any] {
      try model.retarget(scanDetectorRotationDegrees: rotation)
      let z = [
        p.c10Angstroms, p.c12Angstroms * cos(2 * p.phi12Radians),
        p.c12Angstroms * sin(2 * p.phi12Radians), p.sampleTiltMrad[0], p.sampleTiltMrad[1],
        p.depthSpreadAngstroms,
      ]
      let delta = [0.05, 0.05, 0.05, 0.005, 0.005, 0.5]
      var points = [p]
      for d in 0..<6 {
        for sign in [1.0, -1.0] {
          var shifted = z
          shifted[d] += sign * delta[d]
          points.append(
            SSBModelPoint(
              c10Angstroms: shifted[0], c12Angstroms: hypot(shifted[1], shifted[2]),
              phi12Radians: atan2(shifted[2], shifted[1]) / 2,
              sampleTiltMrad: [shifted[3], shifted[4]], depthSpreadAngstroms: shifted[5]))
        }
      }
      let values = try model.agreements(points)
      return [
        "agreement": values[0], "delta": delta, "rotation_deg": rotation,
        "neighbour_relative": values.dropFirst().map { ($0 - values[0]) / values[0] },
      ]
    }
    if options.checkMaximum {
      let kept = fitted.fits[fitted.kept]
      result["maximum_check"] = try neighbours(
        kept.point, rotation: kept.scanDetectorRotationDegrees)
      if let baseline = options.baseline {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: baseline)) as! [String: Any]
        let run = (json["results"] as! [[String: Any]])[0]
        let branch = (run["branches"] as! [[String: Any]])[run["kept"] as! Int]
        let point = SSBModelPoint(
          c10Nanometers: branch["C10_nm"] as! Double, c12Nanometers: branch["C12_nm"] as! Double,
          phi12Degrees: branch["phi12_deg"] as! Double,
          sampleTiltMrad: branch["sample_tilt_mrad"] as! [Double],
          depthSpreadNanometers: branch["depth_spread_nm"] as! Double)
        result["baseline_maximum_check"] = try neighbours(
          point, rotation: branch["rotation_deg"] as! Double)
      }
      try model.retarget(scanDetectorRotationDegrees: kept.scanDetectorRotationDegrees)
    }
    if options.verifyBranches {
      result["verification"] = try verifyBranches(
        fitted, model: model, engine: engine, darkField: darkField, sampling: sampling,
        trials: options.trials)
    }
    results.append(result)
    if repeatIndex == 0, let dump = options.dump {
      try dumpObjective(model: model, fit: fitted, to: dump)
    }
    let refinement = fitted.fit.higherOrderRefinement
    log(
      String(
        format:
          "repeat %d: prepare %.2f s, initial fit %.2f s, total %.2f s, agreement %.9g, fifth-order evals %d (%@)",
        repeatIndex, prepareSeconds, initialSeconds, totalSeconds,
        fitted.fit.sampleAgreement ?? .nan, refinement?.evaluations ?? 0,
        refinement?.stop ?? "none"))
  }
}
let summary: [String: Any] = [
  "schema": "quantem-gpu-ssb-fit-benchmark/v1", "cache": options.cache.path,
  "search": options.search.rawValue,
  "load_s": Date().timeIntervalSince(loadStarted), "results": results,
]
let data = try JSONSerialization.data(
  withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
if let output = options.output {
  try data.write(to: output)
} else {
  FileHandle.standardOutput.write(data)
}
