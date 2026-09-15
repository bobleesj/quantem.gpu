import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMStreamingIO
import Native4DSTEMIO

func emit(_ value: [String: Any]) throws {
  print(String(decoding: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]), as: UTF8.self))
  fflush(stdout)
}
func require(_ condition: Bool, _ message: String) throws {
  if !condition { throw NSError(domain: "resident-index-parity", code: 1,
    userInfo: [NSLocalizedDescriptionKey: message]) }
}
func masks(_ source: MetalPairedRuntimeTANSResidentSource) -> [[UInt8]] {
  [(0.0, 32.0, 0.0), (16.0, 64.0, 0.0), (45.0, 90.0, 0.0),
   (45.0, 90.0, 8.0), (45.0, 90.0, -8.0)].map { inner, outer, offset in
    (0..<192*192).map { pixel in
      let row = Double(pixel / 192) - 95.5
      let column = Double(pixel % 192) - 95.5 - offset
      let squared = row*row + column*column
      return squared >= inner*inner && squared <= outer*outer
        && source.detectorValidityMask[pixel] != 0 ? UInt8(1) : UInt8(0)
    }
  }
}

let arguments = Array(CommandLine.arguments.dropFirst())
try require(arguments.count == 2, "Usage: resident-index-probe INPUT_FOLDER INDEX_DIRECTORY")
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Physical Metal device required") }
let budget = min(UInt64(17_162_698_752), device.recommendedMaxWorkingSetSize)
let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: URL(fileURLWithPath: arguments[1]))
  .prepare(input: URL(fileURLWithPath: arguments[0]))
try require(catalog.datasets.count == 7, "Expected exactly seven acquisitions")
var sources: [MetalPairedRuntimeTANSResidentSource] = []
defer { sources.forEach { $0.releaseResidentStorage() } }
for dataset in catalog.datasets {
  let held = UInt64(device.currentAllocatedSize)
  try require(held < budget, "Resident budget exhausted before load")
  let indexed = try Native4DSTEMIndexedSource.open(dataset: dataset)
  let source = try MetalPairedRuntimeTANSResidentSource.load(source: indexed,
    device: device, maximumAdditionalBytes: budget-held, interaction: .normal)
  sources.append(source)
  try emit(["phase":"load", "ordinal":sources.count, "identity":source.sourceIdentitySHA256,
    "resident_bytes":source.residentBytes, "seconds":source.loadMetrics.totalSeconds])
}
var references: [[[UInt32]]] = []
var diffraction: [[UInt32]] = []
for source in sources {
  references.append(try masks(source).map { try source.updateVirtualDetector(mask:$0).values })
  diffraction.append(try source.extractRawDiffraction(scanRow:256, scanColumn:256))
}
let baselineBytes = sources.reduce(UInt64(0)) { $0 + $1.residentBytes }
var totalSeconds = 0.0
var fullMapChecks = 0
var maximumSampledAllocation = UInt64(device.currentAllocatedSize)
for (index, source) in sources.enumerated() {
  let originalBytes = source.residentBytes
  do {
    try source.prepareResidentDetectorIndex(maximumAdditionalBytes:0, shouldCancel:{true})
    try require(false, "Canceled index preparation unexpectedly succeeded")
  } catch Metal4DSTEMStreamingIOError.cancelled {}
  try require(source.residentBytes == originalBytes, "Cancellation altered residency")
  let originalLoadSeconds = source.loadMetrics.totalSeconds
  let held = UInt64(device.currentAllocatedSize)
  try require(held < budget, "Resident budget exhausted before indexing")
  let started = CFAbsoluteTimeGetCurrent()
  try source.prepareResidentDetectorIndex(maximumAdditionalBytes:budget-held)
  let seconds = CFAbsoluteTimeGetCurrent()-started
  totalSeconds += seconds
  maximumSampledAllocation = max(maximumSampledAllocation, UInt64(device.currentAllocatedSize))
  try require(source.loadMetrics.totalSeconds == originalLoadSeconds, "Upgrade changed original load metrics")
  let indexedBytes = source.residentBytes
  try source.prepareResidentDetectorIndex(maximumAdditionalBytes:0)
  try require(source.residentBytes == indexedBytes, "Repeated upgrade reallocated")
  for (maskIndex, mask) in masks(source).enumerated() {
    let actual = try source.updateVirtualDetector(mask:mask).values
    try require(actual == references[index][maskIndex], "Full map differs after index: source \(index), mask \(maskIndex)")
    fullMapChecks += 1
  }
  let dp = try source.extractRawDiffraction(scanRow:256, scanColumn:256)
  try require(dp == diffraction[index], "Selected raw DP differs after indexing")
  try emit(["phase":"index", "ordinal":index+1, "seconds":seconds,
    "index_bytes":source.polarIndexBytes, "resident_bytes":source.residentBytes,
    "allocated_bytes":device.currentAllocatedSize, "full_map_checks":fullMapChecks,
    "index_only":true, "interaction_mode":source.interactionMode!.rawValue])
}
let indexedBytes = sources.reduce(UInt64(0)) { $0 + $1.residentBytes }
for (index, source) in sources.enumerated() {
  source.releaseResidentDetectorIndex()
  for (maskIndex, mask) in masks(source).enumerated() {
    let actual = try source.updateVirtualDetector(mask:mask).values
    try require(actual == references[index][maskIndex], "Full map differs after removing index")
    fullMapChecks += 1
  }
}
try require(sources.reduce(UInt64(0)){$0+$1.residentBytes} == baselineBytes, "Removing indexes did not restore memory")
try emit(["phase":"result", "pass":true, "sources":sources.count,
  "baseline_resident_bytes":baselineBytes, "indexed_resident_bytes":indexedBytes,
  "index_only_seconds":totalSeconds, "sampled_peak_metal_bytes":maximumSampledAllocation,
  "full_map_checks":fullMapChecks, "selected_dp_checks":sources.count,
  "reloads_during_upgrade":0, "reference":"same resident before index, not independent HDF5 oracle",
  "full_fast_profile":false])
