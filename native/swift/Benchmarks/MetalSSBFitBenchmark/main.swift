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
  var detectorSampling: Double?
  var center: (Double, Double)?
  var radius: Double?
  var darkField = true

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
      case "--detector-sampling": detectorSampling = Double(try value(flag))
      case "--center":
        let parts = try value(flag).split(separator: ",").compactMap { Double($0) }
        guard parts.count == 2 else { throw failure("--center takes ROW,COLUMN") }
        center = (parts[0], parts[1])
      case "--radius": radius = Double(try value(flag))
      case "--no-dark-field": darkField = false
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
  let insidePixels = (0..<(rows * columns)).filter { pixel in
    let dr = Float(pixel / columns) - Float(calibration.brightfieldCenterRowPixels)
    let dc = Float(pixel % columns) - Float(calibration.brightfieldCenterColumnPixels)
    return (dr * dr + dc * dc).squareRoot() < inner
  }
  let filledByPixel = Dictionary(uniqueKeysWithValues: zip(flagged.pixels, filled))
  var dark = total.map(Double.init)
  for first in stride(from: 0, to: insidePixels.count, by: 1024) {
    let batch = Array(insidePixels[first..<min(first + 1024, insidePixels.count)])
    let read = try readColumns(batch, resident: resident, device: device, plane: plane)
    for (pixel, column) in zip(batch, read) {
      let values = filledByPixel[pixel] ?? column
      for position in 0..<plane { dark[position] -= Double(values[position]) }
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
let darkFieldURL = options.cache.appendingPathComponent("dark_field.f32")
let darkField: [Float]? =
  options.darkField && FileManager.default.fileExists(atPath: darkFieldURL.path)
  ? try Data(contentsOf: darkFieldURL).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } : nil
var results: [[String: Any]] = []
for repeatIndex in 0..<options.repeats {
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
    let options5 = SSBFitOptions(trials: options.trials, maximumOrder: options.maximumOrder)
    let fitted = try SSBModelFit.findAberrations(
      model: model, engine: engine, rotationDegrees: caseRecord.rotationEstimateDegrees,
      darkField: darkField,
      scanSampling: (
        caseRecord.calibration.scanSamplingRowAngstroms,
        caseRecord.calibration.scanSamplingColumnAngstroms
      ), options: options5,
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
  "load_s": Date().timeIntervalSince(loadStarted), "results": results,
]
let data = try JSONSerialization.data(
  withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
if let output = options.output {
  try data.write(to: output)
} else {
  FileHandle.standardOutput.write(data)
}
