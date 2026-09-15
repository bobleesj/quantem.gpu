import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMStreamingIO
import Native4DSTEMIO

func emit(_ value: [String: Any]) throws {
  print(String(decoding: try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]), as: UTF8.self))
  fflush(stdout)
}
func require(_ value: Bool, _ reason: String) throws {
  if !value { throw NSError(domain: "conversion-parity", code: 1, userInfo: [NSLocalizedDescriptionKey: reason]) }
}
let args = CommandLine.arguments
try require(args.count == 5, "Usage: probe FOLDER INDEX_DIRECTORY COUNT WORKERS")
let count = Int(args[3])!
let workers = Int(args[4])!
try require((1...4).contains(workers), "Use one to four workers within the qualified fixture budget")
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Physical Metal required") }
let budget = min(UInt64(17_162_698_752), device.recommendedMaxWorkingSetSize)
let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: URL(fileURLWithPath: args[2]))
  .prepare(input: URL(fileURLWithPath: args[1]))
try require(count > 0 && catalog.datasets.count >= count, "Not enough distinct acquisitions in fixture folder")
var sources: [MetalPairedRuntimeTANSResidentSource] = []
var packedSources: [MetalCompactH5ResidentSource] = []
defer {
  sources.forEach { $0.releaseResidentStorage() }
  packedSources.forEach { $0.releaseResidentStorage() }
}
do {
  for dataset in catalog.datasets.prefix(count) {
    let source = try MetalPairedRuntimeTANSResidentSource.load(
      source: Native4DSTEMIndexedSource.open(dataset: dataset), device: device,
      maximumAdditionalBytes: budget - UInt64(device.currentAllocatedSize), interaction: .normal)
    sources.append(source)
    try emit(["phase":"load", "identity":source.sourceIdentitySHA256,
      "resident_bytes":source.residentBytes, "seconds":source.loadMetrics.totalSeconds])
  }
  try require(Set(sources.map(\.sourceIdentitySHA256)).count == count, "Acquisition identities are not distinct")
  let masks: [[UInt8]] = [(0.0,32.0,0.0), (16.0,64.0,0.0), (48.0,92.0,6.0)].map { inner, outer, offset in
      (0..<192*192).map { pixel in
        let row = Double(pixel/192)-95.5, column = Double(pixel%192)-95.5-offset
        let squared = row*row+column*column
        return squared >= inner*inner && squared <= outer*outer ? 1 : 0
      }
  }
  let positions = [0,1,31,32,511,512,4095,4096,65535,131328,262143]
  var mapReferences = [[[UInt32]]](), dpReferences = [[[UInt32]]]()
  for source in sources {
    mapReferences.append(try masks.map { try source.updateVirtualDetector(mask:$0).values })
    dpReferences.append(try positions.map { try source.extractRawDiffraction(scanRow:$0/512,scanColumn:$0%512) })
    do {
      _ = try source.makePackedResident(maximumAdditionalBytes:0,shouldCancel:{true})
      try require(false,"Canceled conversion unexpectedly succeeded")
    } catch Metal4DSTEMStreamingIOError.cancelled {}
  }
  let first = sources[0]
  do {
    _ = try first.makePackedResident(maximumAdditionalBytes:0)
    try require(false,"Zero-budget conversion unexpectedly succeeded")
  } catch let error as NSError where error.domain == "paired-resident-conversion" {}
  var checkpoints = 0
  do {
    _ = try autoreleasepool {
      try first.makePackedResident(maximumAdditionalBytes:budget-UInt64(device.currentAllocatedSize),
        shouldCancel:{ checkpoints += 1; return checkpoints >= 30 })
    }
    try require(false,"Mid-conversion cancellation unexpectedly succeeded")
  } catch Metal4DSTEMStreamingIOError.cancelled {}
  try require(try first.extractRawDiffraction(scanRow:0,scanColumn:0) == dpReferences[0][0],"Cancellation invalidated original")
  try emit(["phase":"failure-safety","early_cancel":true,"mid_cancel":true,"zero_budget":true])
  var total = 0.0
  let resultLock = NSLock()
  var packedResults = [MetalCompactH5ResidentSource?](repeating:nil,count:sources.count)
  var failures = [Error]()
  var peakSampledBytes = UInt64(device.currentAllocatedSize)
  let operations = OperationQueue()
  operations.maxConcurrentOperationCount = workers
  operations.qualityOfService = .userInitiated
  let seriesStarted = CFAbsoluteTimeGetCurrent()
  for (index, source) in sources.enumerated() {
    operations.addOperation {
      do { try autoreleasepool {
    let started = CFAbsoluteTimeGetCurrent()
    let held = UInt64(device.currentAllocatedSize)
    try require(held < budget, "Device budget exhausted before conversion")
    let packed = try source.makePackedResident(maximumAdditionalBytes:budget-held,shouldCancel:{
      let allocated = UInt64(device.currentAllocatedSize)
      resultLock.lock(); peakSampledBytes = max(peakSampledBytes,allocated); resultLock.unlock()
      return allocated > budget
    })
    let seconds = CFAbsoluteTimeGetCurrent()-started
    resultLock.lock()
    total += seconds
    packedResults[index] = packed
    resultLock.unlock()
    try emit(["phase":"converted", "ordinal":index+1, "seconds":seconds,
      "resident_bytes":packed.loadMetrics.totalResidentBytes,"allocated_bytes":device.currentAllocatedSize,
      "decode_seconds":packed.loadMetrics.gpuDecodeMilliseconds/1000,
      "write_seconds":packed.loadMetrics.gpuPreparationMilliseconds/1000,
      "source_read_ms":packed.loadMetrics.sourceReadMilliseconds])
    try require(try source.extractRawDiffraction(scanRow:0,scanColumn:0) == dpReferences[index][0],"Original source invalidated")
    source.releaseResidentStorage()
      }} catch { resultLock.lock(); failures.append(error); resultLock.unlock() }
    }
  }
  operations.waitUntilAllOperationsAreFinished()
  let seriesSeconds = CFAbsoluteTimeGetCurrent()-seriesStarted
  packedSources = packedResults.compactMap { $0 }
  if let failure = failures.first { throw failure }
  try require(packedSources.count == count, "Some conversions did not return a resident")
  for (index,packed) in packedSources.enumerated() {
    let references = mapReferences[index], dps = dpReferences[index]
    for (maskIndex, mask) in masks.enumerated() {
      try packed.updateVirtualDetector(mask:mask,forceRebase:true)
      let actual = try packed.virtualDetectorValues()
      try require(actual.count == references[maskIndex].count,"Detector extent changed")
      let mismatches = zip(actual,references[maskIndex]).filter { $0 != $1 }.count
      try emit(["phase":"map-parity", "ordinal":index+1,"mask":maskIndex,"mismatches":mismatches,
        "actual_first":Array(actual.prefix(8)),"expected_first":Array(references[maskIndex].prefix(8))])
      try require(mismatches==0,"Full detector map mismatch")
    }
    for (positionIndex, scan) in positions.enumerated() {
      let actual = try packed.extractDiffraction(scanRow:scan/512,scanColumn:scan%512)
      try require(actual == dps[positionIndex],"Raw DP mismatch at scan \(scan)")
    }
    try emit(["phase":"parity-pass", "ordinal":index+1,"full_maps":masks.count,"full_dps":positions.count,
      "allocated_after_release":device.currentAllocatedSize])
  }
  try emit(["phase":"result","pass":true,"sources":sources.count,"conversion_seconds":total,
    "series_wall_seconds":seriesSeconds,"workers":workers,"peak_sampled_device_bytes":peakSampledBytes,
    "thermal_state":ProcessInfo.processInfo.thermalState.rawValue,
    "packed_resident_bytes":packedSources.reduce(UInt64(0)){$0+$1.loadMetrics.totalResidentBytes}])
  var reverseSeconds = 0.0
  var reversePeak = UInt64(device.currentAllocatedSize)
  var returned = [MetalPairedRuntimeTANSResidentSource]()
  defer { returned.forEach { $0.releaseResidentStorage() } }
  for (index,packed) in packedSources.enumerated() {
    do {
      _ = try packed.makeANSResident(maximumAdditionalBytes:0,shouldCancel:{true})
      try require(false,"Reverse cancellation unexpectedly succeeded")
    } catch Metal4DSTEMStreamingIOError.cancelled {}
    let reverseStarted = CFAbsoluteTimeGetCurrent()
    let ans = try packed.makeANSResident(maximumAdditionalBytes:budget-UInt64(device.currentAllocatedSize),shouldCancel:{
      reversePeak = max(reversePeak,UInt64(device.currentAllocatedSize)); return false
    })
    let seconds = CFAbsoluteTimeGetCurrent()-reverseStarted
    reverseSeconds += seconds; returned.append(ans)
    try emit(["phase":"reverse-converted","ordinal":index+1,"seconds":seconds,
      "resident_bytes":ans.residentBytes,"allocated_bytes":device.currentAllocatedSize,
      "unpack_and_encode_seconds":ans.loadMetrics.fusedDecodeAndSizeSeconds,
      "prefix_seconds":ans.loadMetrics.provisionalCPUPrefixSeconds,
      "compact_seconds":ans.loadMetrics.compactSeconds,
      "consolidation_seconds":ans.loadMetrics.consolidationSeconds])
    for (maskIndex,mask) in masks.enumerated() {
      let actual = try ans.updateVirtualDetector(mask:mask).values
      try require(actual == mapReferences[index][maskIndex],"Reverse full-map mismatch")
    }
    for (positionIndex,scan) in positions.enumerated() {
      try require(try ans.extractRawDiffraction(scanRow:scan/512,scanColumn:scan%512) == dpReferences[index][positionIndex],
        "Reverse raw-DP mismatch")
    }
    try require(try packed.extractDiffraction(scanRow:0,scanColumn:0) == dpReferences[index][0],"Reverse invalidated packed original")
    packed.releaseResidentStorage()
    try emit(["phase":"reverse-parity","ordinal":index+1,"pass":true,"maps":masks.count,"dps":positions.count])
  }
  try emit(["phase":"reverse-result","pass":true,"sources":returned.count,"summed_conversion_seconds":reverseSeconds,
    "peak_sampled_device_bytes":reversePeak,"resident_bytes":returned.reduce(UInt64(0)){$0+$1.residentBytes}])
} catch {
  try emit(["phase":"failure","error":error.localizedDescription])
  exit(1)
}
