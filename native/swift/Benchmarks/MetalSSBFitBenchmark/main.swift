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
//     [--compare-standard] [--dark-field DARK.f32] [--verify-branches]
//   metal-ssb-fit-benchmark --input MASTER.h5 --cache DIR --export-dark-field DARK.f32
//
// --max-order 1 stops after the low-order branch fits and the branch test. --dark-field passes a float32
// scan-shaped dark-field image to the branch test (written by --export-dark-field from the original HDF5 with
// the app's mask: valid detector pixels at or beyond 1.1 BF radii). --verify-branches also searches the
// branch that was mapped from the fitted one, and reports both points' agreements and phases and the branch
// decision taken with the searched fit instead.

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
  var compareStandard = false
  var darkField: URL?
  var exportDarkField: URL?
  var verifyBranches = false

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
      case "--compare-standard": compareStandard = true
      case "--dark-field": darkField = URL(fileURLWithPath: try value(flag))
      case "--export-dark-field": exportDarkField = URL(fileURLWithPath: try value(flag))
      case "--verify-branches": verifyBranches = true
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
}

func log(_ message: String) {
  FileHandle.standardError.write(Data((message + "\n").utf8))
}

/// Python `dpc.workflow.find_optimal_rotation` (unflipped): the exact minimum of the mean squared curl.
func curlRotation(rowCOM: [Double], columnCOM: [Double], rows: Int, columns: Int) -> Double {
  var curl = 0.0
  var divergence = 0.0
  var cross = 0.0
  var count = 0.0
  for r in 1..<(rows - 1) {
    for c in 1..<(columns - 1) {
      let vcDown = Double(Float(columnCOM[(r + 1) * columns + c]))
      let vcUp = Double(Float(columnCOM[(r - 1) * columns + c]))
      let vrRight = Double(Float(rowCOM[r * columns + c + 1]))
      let vrLeft = Double(Float(rowCOM[r * columns + c - 1]))
      let vrDown = Double(Float(rowCOM[(r + 1) * columns + c]))
      let vrUp = Double(Float(rowCOM[(r - 1) * columns + c]))
      let vcRight = Double(Float(columnCOM[r * columns + c + 1]))
      let vcLeft = Double(Float(columnCOM[r * columns + c - 1]))
      let k = 0.5 * (vcDown - vcUp) - 0.5 * (vrRight - vrLeft)
      let d = 0.5 * (vrDown - vrUp) + 0.5 * (vcRight - vcLeft)
      curl += k * k
      divergence += d * d
      cross += k * d
      count += 1
    }
  }
  curl /= count
  divergence /= count
  cross /= count
  let half = 0.5 * (curl - divergence)
  guard hypot(half, cross) > 0 else { return 0 }
  let angle = (0.5 * (atan2(cross, half) + Double.pi)).truncatingRemainder(dividingBy: Double.pi)
  let degrees = (angle * 180 / Double.pi).truncatingRemainder(dividingBy: 180)
  return degrees == 180 ? 0 : degrees
}

/// Read the original acquisition once and keep its exact selected BF columns beside a case record.
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
  log("loaded \(metadata.scanRows)x\(metadata.scanColumns) scan, \(metadata.detectorRows)x\(metadata.detectorColumns) detector in \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
  let detectorSum = try resident.meanDiffractionPattern().detectorSum
  // Camera-flagged pixels hold the dtype's maximum in every frame; the app's validity mask excludes them.
  let markers: [UInt64] = [255, 65535, 4_294_967_295].map { $0 * UInt64(metadata.scanCount) }
  let flagged = detectorSum.indices.filter { markers.contains(detectorSum[$0]) }
  let excluded = Set(metadata.excludedDetectorPixels).union(flagged)
  log("excluded \(excluded.count) flagged detector pixels")
  let disk = try MetalSSBBrightfieldEstimate.measure(
    device: device, detectorSum: detectorSum, rows: metadata.detectorRows,
    columns: metadata.detectorColumns, excluded: excluded)
  let sampling = try disk.detectorSampling(semiangle: options.semiangleMrad)
  let calibration = MetalSSBCalibration(
    voltageKV: options.voltageKV, semiangleMrad: options.semiangleMrad,
    scanSamplingRowAngstroms: options.scanStep, scanSamplingColumnAngstroms: options.scanStep,
    detectorSamplingRowMrad: sampling, detectorSamplingColumnMrad: sampling,
    brightfieldCenterRowPixels: disk.centerRow, brightfieldCenterColumnPixels: disk.centerColumn,
    brightfieldRadiusPixels: disk.selectionRadius)
  let setup = try calibration.geometry(
    detectorRows: metadata.detectorRows, detectorColumns: metadata.detectorColumns,
    detectorSum: detectorSum, excludedPixels: excluded,
    scanRows: metadata.scanRows, scanColumns: metadata.scanColumns)
  guard let moments = try resident.preparedDPCMomentValues() else {
    throw failure("The resident has no exact DPC moments for the rotation estimate")
  }
  let rowCOM = zip(moments.detectorRowMoment, moments.total).map {
    $1 > 0 ? Double($0) / Double($1) : 0
  }
  let columnCOM = zip(moments.detectorColumnMoment, moments.total).map {
    $1 > 0 ? Double($0) / Double($1) : 0
  }
  let rotation = curlRotation(
    rowCOM: rowCOM, columnCOM: columnCOM, rows: metadata.scanRows, columns: metadata.scanColumns)
  // Exact columns, narrowed only when every count fits.
  let plane = metadata.scanCount
  guard let queue = device.makeCommandQueue(),
    let staging = device.makeBuffer(length: 32 * plane * 4, options: .storageModeShared)
  else { throw failure("Metal staging unavailable") }
  var wide = [UInt32](repeating: 0, count: setup.pixels.count * plane)
  for first in stride(from: 0, to: setup.pixels.count, by: 32) {
    let pixels = Array(setup.pixels[first..<min(first + 32, setup.pixels.count)])
    guard let commands = queue.makeCommandBuffer() else { throw failure("Metal command unavailable") }
    try resident.encodeDetectorColumns(pixels: pixels, into: staging, commands: commands)
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error { throw error }
    let values = staging.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<(pixels.count * plane) { wide[first * plane + i] = values[i] }
  }
  let maximum = wide.max() ?? 0
  try FileManager.default.createDirectory(at: options.cache, withIntermediateDirectories: true)
  let countType: MetalSSBCountType = maximum <= 255 ? .uint8 : (maximum <= 65535 ? .uint16 : .uint32)
  let countsURL = options.cache.appendingPathComponent("counts.bin")
  switch countType {
  case .uint8: try Data(wide.map { UInt8($0) }.withUnsafeBytes { Data($0) }).write(to: countsURL)
  case .uint16: try Data(wide.map { UInt16($0) }.withUnsafeBytes { Data($0) }).write(to: countsURL)
  case .uint32: try wide.withUnsafeBytes { Data($0) }.write(to: countsURL)
  }
  let record = CaseRecord(
    source: input.path, scanRows: metadata.scanRows, scanColumns: metadata.scanColumns,
    detectorRows: metadata.detectorRows, detectorColumns: metadata.detectorColumns,
    calibration: calibration, excludedPixels: excluded.sorted(), pixels: setup.pixels,
    countType: countType.rawValue, rotationEstimateDegrees: rotation,
    exportSeconds: Date().timeIntervalSince(started))
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
let darkField: [Float]? = try options.darkField.map { url in
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
    let options5 = SSBFitOptions(
      trials: options.trials, compareStandard: options.compareStandard,
      maximumOrder: options.maximumOrder)
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
  "load_s": Date().timeIntervalSince(loadStarted), "results": results,
]
let data = try JSONSerialization.data(
  withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
if let output = options.output {
  try data.write(to: output)
} else {
  FileHandle.standardOutput.write(data)
}
