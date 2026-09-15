import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMStreamingIO

private let interval = 512

private enum ProbeError: Error, CustomStringConvertible {
  case failed(String)
  var description: String { if case .failed(let message) = self { return message }; return "" }
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
  if !condition() { throw ProbeError.failed(message) }
}

private func makeBuffer<T>(_ values: [T], on device: MTLDevice) throws -> MTLBuffer {
  guard let buffer = values.withUnsafeBytes({ bytes in
    device.makeBuffer(bytes: bytes.baseAddress!, length: max(bytes.count, 1), options: .storageModeShared)
  }) else { throw ProbeError.failed("Metal buffer allocation failed") }
  return buffer
}

private func decodingBuffer(from codec: MetalPairedRuntimeTANSSyntheticCodec) throws -> MTLBuffer {
  // The SPI result intentionally exposes only the portable compact bytes. Reuse the
  // exact immutable table already owned by the codec rather than cloning its builder.
  guard let table = Mirror(reflecting: codec).children.first(where: { $0.label == "decoding" })?.value
    as? MTLBuffer else {
    throw ProbeError.failed("Could not locate the synthetic codec's decoding table")
  }
  return table
}

private func decodingValues(from buffer: MTLBuffer) -> [UInt32] {
  Array(UnsafeBufferPointer(
    start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: 32 * 1024))
}

private func streams() -> [[UInt16]] {
  var result = [[UInt16]]()
  result.append([UInt16](repeating: 0, count: interval))                         // 253
  result.append([UInt16](repeating: 65_535, count: interval))                    // 255
  var sparse = [UInt16](repeating: 0, count: interval); sparse[13] = 128; sparse[511] = 1
  result.append(sparse)                                                           // 252, boundaries
  result.append((0..<interval).map { UInt16(($0 * 17 + $0 / 7) % 31) })          // entropy
  result.append((0..<interval).map { UInt16(truncatingIfNeeded: $0 &* 40503 &+ 7919) }) // 254
  var escapes = (0..<interval).map { UInt16(($0 * 11 + 3) % 29) }
  for position in stride(from: 5, to: interval, by: 47) { escapes[position] = UInt16(400 + position) }
  result.append(escapes)                                                          // entropy escapes
  // Low-mean entropy modes dominate real ADF changes. These have enough events
  // to avoid sparse mode while varying the terminal partial byte.
  for spacing in [19, 17, 16, 15, 13, 11, 9, 8] {
    var lowMean = [UInt16](repeating: 0, count: interval)
    for scan in 0..<interval where scan % spacing == spacing / 2 {
      lowMean[scan] = UInt16((scan / spacing) % 3 + 1)
    }
    result.append(lowMean)
  }
  // 67 selected pixels is deliberately not a multiple of a SIMD width. Cycle all
  // representations so adjacent lanes see different modes and coefficient signs.
  while result.count < 67 {
    let index = result.count
    switch index % 6 {
    case 0: result.append([UInt16](repeating: UInt16(1000 + index), count: interval))
    case 1:
      var value = [UInt16](repeating: 0, count: interval)
      value[(index * 13) % 511] = UInt16(index % 128 + 1); result.append(value)
    case 2: result.append((0..<interval).map { UInt16(($0 + index * 3) % 33) })
    case 3: result.append((0..<interval).map { UInt16(truncatingIfNeeded: $0 * (index * 997 + 1)) })
    case 4: result.append([UInt16](repeating: 0, count: interval))
    default:
      var value = (0..<interval).map { UInt16(($0 * 5 + index) % 27) }
      value[511] = UInt16(600 + index); result.append(value)
    }
  }
  // Twin the UInt16 maximum lane so one copy can carry a negative coefficient
  // while the independent UInt64 reference remains nonnegative.
  result[66] = [UInt16](repeating: 65_535, count: interval)
  return result
}


private func malformedEntropyCheck(
  name: String, device: MTLDevice, queue: MTLCommandQueue,
  pipeline: MTLComputePipelineState, compact: MetalPairedRuntimeTANSSyntheticResult,
  decoding: MTLBuffer, stream: Int, includeSparse: Bool = true,
  mutate: (inout [UInt8], inout [UInt32]) -> Void
) throws {
  var payloadBytes = compact.payload
  var offsetValues = compact.offsets
  mutate(&payloadBytes, &offsetValues)
  let payload = try makeBuffer(payloadBytes + [UInt8](repeating: 0, count: 8), on: device)
  let offsets = try makeBuffer(offsetValues, on: device)
  let modes = try makeBuffer(compact.modes, on: device)
  let selected = try makeBuffer([UInt32(stream)], on: device)
  let coefficients = try makeBuffer([Int32(1)], on: device)
  let output = try makeBuffer([UInt32](repeating: 0, count: interval), on: device)
  let failure = try makeBuffer([UInt32(0)], on: device)
  guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("Could not create \(name) command")
  }
  encoder.setComputePipelineState(pipeline)
  for (index, buffer) in [payload, offsets, modes, decoding, selected, coefficients, output, failure].enumerated() {
    encoder.setBuffer(buffer, offset: 0, index: index)
  }
  var parameters: [UInt32] = [UInt32(compact.modes.count), 1, 1, 0,
    UInt32(payloadBytes.count), includeSparse ? 1 : 0]
  encoder.setBytes(&parameters, length: parameters.count * 4, index: 8)
  encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
  try require(command.status == .completed, "\(name) dispatch failed unexpectedly")
  try require(failure.contents().load(as: UInt32.self) != 0, "\(name) was not rejected")
}

private func run(
  name: String, pipeline: MTLComputePipelineState, sparsePipeline: MTLComputePipelineState?,
  device: MTLDevice, queue: MTLCommandQueue, compact: MetalPairedRuntimeTANSSyntheticResult,
  decoding: MTLBuffer, selected: [UInt32], coefficients: [Int32], packetSplits: Int = 1
) throws -> ([UInt32], UInt32) {
  // Match the resident's safe prefetch padding; declared extents remain exact.
  let payload = try makeBuffer(compact.payload + [UInt8](repeating: 0, count: 8), on: device)
  let offsets = try makeBuffer(compact.offsets, on: device)
  let modes = try makeBuffer(compact.modes, on: device)
  let selectedBuffer = try makeBuffer(selected, on: device)
  let coefficientBuffer = try makeBuffer(coefficients, on: device)
  let output = try makeBuffer([UInt32](repeating: 0, count: interval), on: device)
  let failure = try makeBuffer([UInt32(0)], on: device)
  guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("Could not create \(name) command")
  }
  encoder.setComputePipelineState(pipeline)
  let common = [payload, offsets, modes, decoding, selectedBuffer, coefficientBuffer, output, failure]
  for (index, buffer) in common.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
  var parameters: [UInt32] = [UInt32(selected.count), 1, UInt32(selected.count), 0,
    UInt32(compact.payload.count), sparsePipeline == nil ? 1 : 0]
  encoder.setBytes(&parameters, length: parameters.count * 4, index: 8)
  encoder.dispatchThreadgroups(MTLSize(width: packetSplits, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  encoder.endEncoding()
  if let sparsePipeline {
    guard let sparseEncoder = command.makeComputeCommandEncoder() else {
      throw ProbeError.failed("Could not create \(name) sparse command")
    }
    sparseEncoder.setComputePipelineState(sparsePipeline)
    for (index, buffer) in [payload, offsets, modes, selectedBuffer, coefficientBuffer, output, failure].enumerated() {
      sparseEncoder.setBuffer(buffer, offset: 0, index: index)
    }
    var sparseParameters: [UInt32] = [UInt32(selected.count), 1, UInt32(selected.count), UInt32(compact.payload.count)]
    sparseEncoder.setBytes(&sparseParameters, length: sparseParameters.count * 4, index: 7)
    sparseEncoder.dispatchThreads(MTLSize(width: selected.count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: min(256, selected.count), height: 1, depth: 1))
    sparseEncoder.endEncoding()
  }
  command.commit(); command.waitUntilCompleted()
  try require(command.status == .completed, "\(name) command failed: \(command.error?.localizedDescription ?? "unknown")")
  let values = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt32.self), count: interval))
  return (values, failure.contents().load(as: UInt32.self))
}

private func runPartials(
  name: String, partialsPipeline: MTLComputePipelineState, finishPipeline: MTLComputePipelineState,
  device: MTLDevice, queue: MTLCommandQueue, compact: MetalPairedRuntimeTANSSyntheticResult,
  decoding: MTLBuffer, selected: [UInt32], coefficients: [Int32]
) throws -> ([UInt32], UInt32) {
  let groups = (selected.count + 63) / 64
  let payload = try makeBuffer(compact.payload + [UInt8](repeating: 0, count: 8), on: device)
  let offsets = try makeBuffer(compact.offsets, on: device)
  let modes = try makeBuffer(compact.modes, on: device)
  let selectedBuffer = try makeBuffer(selected, on: device)
  let coefficientBuffer = try makeBuffer(coefficients, on: device)
  let partials = try makeBuffer([UInt32](repeating: 0, count: groups * interval), on: device)
  let output = try makeBuffer([UInt32](repeating: 0, count: interval), on: device)
  let failure = try makeBuffer([UInt32(0)], on: device)
  guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("Could not create \(name) partial command")
  }
  encoder.setComputePipelineState(partialsPipeline)
  for (index, buffer) in [payload, offsets, modes, decoding, selectedBuffer,
    coefficientBuffer, partials, failure].enumerated() {
    encoder.setBuffer(buffer, offset: 0, index: index)
  }
  var parameters: [UInt32] = [UInt32(selected.count), 1, UInt32(selected.count),
    UInt32(groups), UInt32(compact.payload.count)]
  encoder.setBytes(&parameters, length: parameters.count * 4, index: 8)
  encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
  encoder.endEncoding()
  guard let finish = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("Could not create \(name) finish command")
  }
  finish.setComputePipelineState(finishPipeline)
  finish.setBuffer(partials, offset: 0, index: 0)
  finish.setBuffer(output, offset: 0, index: 1)
  var finishParameters: [UInt32] = [1, UInt32(groups)]
  finish.setBytes(&finishParameters, length: finishParameters.count * 4, index: 2)
  finish.dispatchThreads(MTLSize(width: interval, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  finish.endEncoding(); command.commit(); command.waitUntilCompleted()
  try require(command.status == .completed,
    "\(name) command failed: \(command.error?.localizedDescription ?? "unknown")")
  return (Array(UnsafeBufferPointer(
    start: output.contents().assumingMemoryBound(to: UInt32.self), count: interval)),
    failure.contents().load(as: UInt32.self))
}

private func malformedPartialsCheck(
  device: MTLDevice, queue: MTLCommandQueue, pipeline: MTLComputePipelineState,
  compact: MetalPairedRuntimeTANSSyntheticResult, decoding: MTLBuffer, stream: Int
) throws {
  var offsetValues = compact.offsets
  offsetValues[stream + 1] -= 1
  let payload = try makeBuffer(compact.payload + [UInt8](repeating: 0, count: 8), on: device)
  let offsets = try makeBuffer(offsetValues, on: device)
  let modes = try makeBuffer(compact.modes, on: device)
  let selected = try makeBuffer([UInt32(stream)], on: device)
  let coefficients = try makeBuffer([Int32(1)], on: device)
  let partials = try makeBuffer([UInt32](repeating: 0, count: interval), on: device)
  let failure = try makeBuffer([UInt32(0)], on: device)
  guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("Could not create malformed partials command")
  }
  encoder.setComputePipelineState(pipeline)
  for (index, buffer) in [payload, offsets, modes, decoding, selected,
    coefficients, partials, failure].enumerated() {
    encoder.setBuffer(buffer, offset: 0, index: index)
  }
  var parameters: [UInt32] = [UInt32(compact.modes.count), 1, 1, 1,
    UInt32(compact.payload.count)]
  encoder.setBytes(&parameters, length: parameters.count * 4, index: 8)
  encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
  encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
  try require(command.status == .completed, "Malformed partials dispatch failed unexpectedly")
  try require(failure.contents().load(as: UInt32.self) != 0,
    "Truncated dense partial stream was not rejected")
}

private func malformedSparseCheck(
  device: MTLDevice, queue: MTLCommandQueue, pipeline: MTLComputePipelineState,
  compact: MetalPairedRuntimeTANSSyntheticResult
) throws {
  guard let stream = compact.modes.firstIndex(of: 252),
    compact.offsets[stream + 1] - compact.offsets[stream] >= 4 else {
    throw ProbeError.failed("Fixture did not produce a two-event sparse stream")
  }
  var corrupt = compact.payload
  let first = Int(compact.offsets[stream])
  corrupt[first + 2] = corrupt[first]; corrupt[first + 3] = corrupt[first + 1] // duplicate position
  let payload = try makeBuffer(corrupt, on: device)
  let offsets = try makeBuffer(compact.offsets, on: device)
  let modes = try makeBuffer(compact.modes, on: device)
  let selected = try makeBuffer([UInt32(stream)], on: device)
  let coefficients = try makeBuffer([Int32(1)], on: device)
  let output = try makeBuffer([UInt32](repeating: 0, count: interval), on: device)
  let failure = try makeBuffer([UInt32(0)], on: device)
  guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("Could not create malformed sparse command")
  }
  encoder.setComputePipelineState(pipeline)
  for (index, buffer) in [payload, offsets, modes, selected, coefficients, output, failure].enumerated() {
    encoder.setBuffer(buffer, offset: 0, index: index)
  }
  var parameters: [UInt32] = [UInt32(compact.modes.count), 1, 1, UInt32(corrupt.count)]
  encoder.setBytes(&parameters, length: parameters.count * 4, index: 7)
  encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
  encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
  try require(command.status == .completed, "Malformed sparse dispatch itself failed")
  try require(failure.contents().load(as: UInt32.self) != 0, "Malformed sparse stream was not rejected")
}

@main
struct DetectorCorrectnessProbe {
  static func main() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw ProbeError.failed("Metal is unavailable")
    }
    let values = streams()
    let codec = try MetalPairedRuntimeTANSSyntheticCodec(device: device)
    let compact = try codec.roundTrip(streams: values, logicalDtype: .uint16)
    try require(compact.decodedStreams == values, "Synthetic codec round trip disagrees with fixture")
    for mode in [UInt8(252), 253, 254, 255] {
      try require(compact.modes.contains(mode), "Fixture did not exercise mode \(mode)")
    }
    try require(compact.modes.contains(where: { $0 < 252 }), "Fixture did not exercise entropy mode")

    let selected = values.indices.map(UInt32.init)
    var coefficients = values.indices.map {
      $0 > 1 && ($0 % 6 == 1 || $0 % 6 == 2) ? Int32(-1) : Int32(1)
    }
    coefficients[1] = -1 // cancelled by the positive 65,535 twin at lane 66
    coefficients[5] = -1 // an escape-bearing negative delta
    var reference = [UInt64](repeating: 0, count: interval)
    for scan in 0..<interval {
      let signed = zip(values, coefficients).reduce(Int64(0)) { $0 + Int64($1.0[scan]) * Int64($1.1) }
      try require(signed >= 0 && signed <= Int64(UInt32.max), "CPU fixture escaped UInt32 result range at scan \(scan)")
      reference[scan] = UInt64(signed)
    }

    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      let function = try library.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
      return try device.makeComputePipelineState(function: function)
    }
    let owner2 = try pipeline(Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function)
    let dense = try pipeline(Metal4DSTEMKernels.pairedRuntimeTANSDetectorDenseCompactionFunction)
    let constants = MTLFunctionConstantValues()
    var plainScratch = true
    constants.setConstantValue(&plainScratch, type: .bool, index: 1)
    let plainFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorDenseCompactionFunction,
      constantValues: constants)
    let plain = try device.makeComputePipelineState(function: plainFunction)
    let sparse = try pipeline(Metal4DSTEMKernels.pairedRuntimeTANSDetectorSparseScatterFunction)
    let cooperative = try pipeline("paired_runtime_tans_detector_cooperative")
    let partialsPipeline = try pipeline(Metal4DSTEMKernels.pairedRuntimeTANSDetectorPartialsFunction)
    let finishPipeline = try pipeline(Metal4DSTEMKernels.pairedRuntimeTANSDetectorFinishFunction)
    let table = try decodingBuffer(from: codec)
    for splitCount in [2, 4, 8] {
      let splitConstants = MTLFunctionConstantValues()
      var splitValue = UInt32(splitCount)
      var lanes = UInt32(2)
      splitConstants.setConstantValue(&splitValue, type: .uint, index: 9)
      splitConstants.setConstantValue(&lanes, type: .uint, index: 0)
      let splitFunction = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: splitConstants)
      let splitPipeline = try device.makeComputePipelineState(function: splitFunction)
      let (actual, failure) = try run(
        name: "packet split \(splitCount)", pipeline: splitPipeline, sparsePipeline: nil,
        device: device, queue: queue, compact: compact, decoding: table,
        selected: selected, coefficients: coefficients, packetSplits: splitCount)
      try require(failure == 0, "packet split \(splitCount) signaled failure")
      try require(actual.map(UInt64.init) == reference,
        "packet split \(splitCount) differs from independent UInt64 sums")
      print("PASS packet split \(splitCount): 512 exact signed-delta outputs, mixed uint16 modes")
    }
    let macroWords = try PairedRuntimeTANSMacroTable.build(decoding: decodingValues(from: table))
    let macroTable = try makeBuffer(macroWords, on: device)
    let macroConstants = MTLFunctionConstantValues()
    var twoStreams = UInt32(2)
    var macroEnabled = true
    macroConstants.setConstantValue(&twoStreams, type: .uint, index: 0)
    macroConstants.setConstantValue(&macroEnabled, type: .bool, index: 2)
    let macroFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: macroConstants)
    let macroOwner2 = try device.makeComputePipelineState(function: macroFunction)
    let reader32Constants = MTLFunctionConstantValues()
    var reader32TwoStreams = UInt32(2)
    var reader32Enabled = true
    reader32Constants.setConstantValue(&reader32TwoStreams, type: .uint, index: 0)
    reader32Constants.setConstantValue(&reader32Enabled, type: .bool, index: 3)
    let reader32Function = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: reader32Constants)
    let reader32Owner2 = try device.makeComputePipelineState(function: reader32Function)
    let pendingReuseConstants = MTLFunctionConstantValues()
    var pendingReuseTwoStreams = UInt32(2)
    var pendingReuseEnabled = true
    pendingReuseConstants.setConstantValue(&pendingReuseTwoStreams, type: .uint, index: 0)
    pendingReuseConstants.setConstantValue(&pendingReuseEnabled, type: .bool, index: 10)
    let pendingReuseFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: pendingReuseConstants)
    let pendingReuseOwner2 = try device.makeComputePipelineState(function: pendingReuseFunction)
    let registerReuseConstants = MTLFunctionConstantValues()
    var registerReuseTwoStreams = UInt32(2)
    var registerReuseEnabled = true
    registerReuseConstants.setConstantValue(&registerReuseTwoStreams, type: .uint, index: 0)
    registerReuseConstants.setConstantValue(&registerReuseEnabled, type: .bool, index: 11)
    let registerReuseFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: registerReuseConstants)
    let registerReuseOwner2 = try device.makeComputePipelineState(function: registerReuseFunction)
    let denseBatchConstants = MTLFunctionConstantValues()
    var denseBatchTwoStreams = UInt32(2)
    var denseBatchEnabled = true
    denseBatchConstants.setConstantValue(&denseBatchTwoStreams, type: .uint, index: 0)
    denseBatchConstants.setConstantValue(&denseBatchEnabled, type: .bool, index: 12)
    let denseBatchFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: denseBatchConstants)
    let denseBatchOwner2 = try device.makeComputePipelineState(function: denseBatchFunction)
    let trustedTableConstants = MTLFunctionConstantValues()
    var trustedTableTwoStreams = UInt32(2)
    var trustedTableEnabled = true
    trustedTableConstants.setConstantValue(&trustedTableTwoStreams, type: .uint, index: 0)
    trustedTableConstants.setConstantValue(&trustedTableEnabled, type: .bool, index: 13)
    let trustedTableFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: trustedTableConstants)
    let trustedTableOwner2 = try device.makeComputePipelineState(function: trustedTableFunction)
    let lazyRefillConstants = MTLFunctionConstantValues()
    var lazyRefillTwoStreams = UInt32(2)
    var lazyRefillEnabled = true
    lazyRefillConstants.setConstantValue(&lazyRefillTwoStreams, type: .uint, index: 0)
    lazyRefillConstants.setConstantValue(&lazyRefillEnabled, type: .bool, index: 15)
    let lazyRefillFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: lazyRefillConstants)
    let lazyRefillOwner2 = try device.makeComputePipelineState(function: lazyRefillFunction)
    let popConstants = MTLFunctionConstantValues()
    popConstants.setConstantValue(&lazyRefillTwoStreams, type: .uint, index: 0)
    popConstants.setConstantValue(&lazyRefillEnabled, type: .bool, index: 16)
    let popFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: popConstants)
    let popPipeline = try device.makeComputePipelineState(function: popFunction)
    var readerCandidates: [(String, MTLComputePipelineState)] = [("branchless_pop", popPipeline)]
    for threshold in [UInt32(16), UInt32(24)] {
      let refillConstants = MTLFunctionConstantValues()
      var streamsPerLane = UInt32(2)
      var refillThreshold = threshold
      refillConstants.setConstantValue(&streamsPerLane, type: .uint, index: 0)
      refillConstants.setConstantValue(&refillThreshold, type: .uint, index: 17)
      let refillFunction = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: refillConstants)
      readerCandidates.append(("eager_refill_\(threshold)",
        try device.makeComputePipelineState(function: refillFunction)))
    }
    let phasedConstants = MTLFunctionConstantValues()
    var phasedStreamsPerLane = UInt32(2)
    var phasedEnabled = true
    phasedConstants.setConstantValue(&phasedStreamsPerLane, type: .uint, index: 0)
    phasedConstants.setConstantValue(&phasedEnabled, type: .bool, index: 18)
    let phasedFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: phasedConstants)
    readerCandidates.append(("phased_readers",
      try device.makeComputePipelineState(function: phasedFunction)))
    for factor in [UInt32(2), UInt32(4), UInt32(8)] {
      let unrollConstants = MTLFunctionConstantValues()
      var streamsPerLane = UInt32(2)
      var unrollFactor = factor
      unrollConstants.setConstantValue(&streamsPerLane, type: .uint, index: 0)
      unrollConstants.setConstantValue(&unrollFactor, type: .uint, index: 19)
      let unrollFunction = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: unrollConstants)
      readerCandidates.append(("unroll_\(factor)",
        try device.makeComputePipelineState(function: unrollFunction)))
    }
    for (candidateName, candidate) in readerCandidates {
      for split in [Optional<MTLComputePipelineState>.none, Optional(sparse)] {
        let name = candidateName + (split == nil ? "" : "+sparse")
        let (actual, failure) = try run(name: name, pipeline: candidate, sparsePipeline: split,
          device: device, queue: queue, compact: compact, decoding: table,
          selected: selected, coefficients: coefficients)
        try require(failure == 0, "\(name) reported failure \(failure)")
        try require(actual.map(UInt64.init) == reference,
          "\(name) differs from independent signed high-count UInt64 reference")
        print("PASS \(name): 512 exact signed-delta outputs")
      }
    }
    for (name, primary, split) in [("owner2", owner2, nil), ("owner2+sparse", owner2, sparse),
      ("dense_compaction+sparse", dense, sparse), ("dense_plain+sparse", plain, sparse)] {
      let (actual, failure) = try run(name: name, pipeline: primary, sparsePipeline: split,
        device: device, queue: queue, compact: compact, decoding: table,
        selected: selected, coefficients: coefficients)
      try require(failure == 0, "\(name) reported failure \(failure)")
      let widened = actual.map(UInt64.init)
      try require(widened == reference, "\(name) differs from UInt64 CPU reference")
      print("PASS \(name): 512 exact signed-delta outputs")
    }
    for (name, split) in [("macro_owner2", Optional<MTLComputePipelineState>.none),
      ("macro_owner2+sparse", Optional(sparse))] {
      let (actual, failure) = try run(name: name, pipeline: macroOwner2, sparsePipeline: split,
        device: device, queue: queue, compact: compact, decoding: macroTable,
        selected: selected, coefficients: coefficients)
      try require(failure == 0, "\(name) reported failure \(failure)")
      try require(actual.map(UInt64.init) == reference, "\(name) differs from UInt64 CPU reference")
      print("PASS \(name): 512 exact signed-delta outputs")
    }
    let (partialsActual, partialsFailure) = try runPartials(
      name: "partials+finish", partialsPipeline: partialsPipeline,
      finishPipeline: finishPipeline, device: device, queue: queue, compact: compact,
      decoding: table, selected: selected, coefficients: coefficients)
    try require(partialsFailure == 0, "partials+finish reported failure \(partialsFailure)")
    try require(partialsActual.map(UInt64.init) == reference,
      "partials+finish differs from UInt64 CPU reference")
    print("PASS partials+finish: two groups, 512 exact signed-delta outputs")
    for (name, split) in [("reader32_owner2", Optional<MTLComputePipelineState>.none),
      ("reader32_owner2+sparse", Optional(sparse))] {
      let (actual, failure) = try run(name: name, pipeline: reader32Owner2,
        sparsePipeline: split, device: device, queue: queue, compact: compact,
        decoding: table, selected: selected, coefficients: coefficients)
      try require(failure == 0, "\(name) reported failure \(failure)")
      try require(actual.map(UInt64.init) == reference,
        "\(name) differs from UInt64 CPU reference")
      print("PASS \(name): 512 exact signed-delta outputs")
    }
    for (name, split) in [("pending_reuse_owner2", Optional<MTLComputePipelineState>.none),
      ("pending_reuse_owner2+sparse", Optional(sparse))] {
      let (actual, failure) = try run(name: name, pipeline: pendingReuseOwner2,
        sparsePipeline: split, device: device, queue: queue, compact: compact,
        decoding: table, selected: selected, coefficients: coefficients)
      try require(failure == 0, "\(name) reported failure \(failure)")
      try require(actual.map(UInt64.init) == reference,
        "\(name) differs from UInt64 CPU reference")
      print("PASS \(name): 512 exact signed-delta outputs")
    }
    for (name, split) in [("register_reuse_owner2", Optional<MTLComputePipelineState>.none),
      ("register_reuse_owner2+sparse", Optional(sparse))] {
      let (actual, failure) = try run(name: name, pipeline: registerReuseOwner2,
        sparsePipeline: split, device: device, queue: queue, compact: compact,
        decoding: table, selected: selected, coefficients: coefficients)
      try require(failure == 0, "\(name) reported failure \(failure)")
      try require(actual.map(UInt64.init) == reference,
        "\(name) differs from UInt64 CPU reference")
      print("PASS \(name): 512 exact signed-delta outputs")
    }
    for (name, split) in [("dense_batch_owner2", Optional<MTLComputePipelineState>.none),
      ("dense_batch_owner2+sparse", Optional(sparse))] {
      let (actual, failure) = try run(name: name, pipeline: denseBatchOwner2,
        sparsePipeline: split, device: device, queue: queue, compact: compact,
        decoding: table, selected: selected, coefficients: coefficients)
      try require(failure == 0, "\(name) reported failure \(failure)")
      try require(actual.map(UInt64.init) == reference,
        "\(name) differs from UInt64 CPU reference")
      print("PASS \(name): 512 exact signed-delta outputs")
    }
    for (name, split) in [("trusted_table_owner2", Optional<MTLComputePipelineState>.none),
      ("trusted_table_owner2+sparse", Optional(sparse))] {
      let (actual, failure) = try run(name: name, pipeline: trustedTableOwner2,
        sparsePipeline: split, device: device, queue: queue, compact: compact,
        decoding: table, selected: selected, coefficients: coefficients)
      try require(failure == 0, "\(name) reported failure \(failure)")
      try require(actual.map(UInt64.init) == reference,
        "\(name) differs from UInt64 CPU reference")
      print("PASS \(name): 512 exact signed-delta outputs")
    }
    for (name, split) in [("lazy_refill_owner2", Optional<MTLComputePipelineState>.none),
      ("lazy_refill_owner2+sparse", Optional(sparse))] {
      let (actual, failure) = try run(name: name, pipeline: lazyRefillOwner2,
        sparsePipeline: split, device: device, queue: queue, compact: compact,
        decoding: table, selected: selected, coefficients: coefficients)
      try require(failure == 0, "\(name) reported failure \(failure)")
      try require(actual.map(UInt64.init) == reference,
        "\(name) differs from UInt64 CPU reference")
      print("PASS \(name): 512 exact signed-delta outputs")
    }
    for splitCount in [2, 4] {
      let splitConstants = MTLFunctionConstantValues()
      var splitValue = UInt32(splitCount)
      var lanes = UInt32(2)
      var enabled = true
      splitConstants.setConstantValue(&lanes, type: .uint, index: 0)
      splitConstants.setConstantValue(&splitValue, type: .uint, index: 9)
      splitConstants.setConstantValue(&enabled, type: .bool, index: 12)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: splitConstants)
      let splitPipeline = try device.makeComputePipelineState(function: function)
      let (actual, failure) = try run(
        name: "dense batch packet split \(splitCount)", pipeline: splitPipeline,
        sparsePipeline: sparse, device: device, queue: queue, compact: compact,
        decoding: table, selected: selected, coefficients: coefficients,
        packetSplits: splitCount)
      try require(failure == 0, "dense batch packet split \(splitCount) reported failure")
      try require(actual.map(UInt64.init) == reference,
        "dense batch packet split \(splitCount) differs from UInt64 CPU reference")
      print("PASS dense batch packet split \(splitCount): 67 mixed streams")
    }
    let (cooperativeActual, cooperativeFailure) = try run(
      name: "cooperative+sparse", pipeline: cooperative, sparsePipeline: sparse,
      device: device, queue: queue, compact: compact, decoding: table,
      selected: selected, coefficients: coefficients)
    try require(cooperativeFailure == 0,
      "cooperative+sparse reported failure \(cooperativeFailure)")
    try require(cooperativeActual.map(UInt64.init) == reference,
      "cooperative+sparse differs from UInt64 CPU reference")
    print("PASS cooperative+sparse: 512 exact signed-delta outputs")
    let lowMean = compact.modes.indices.filter { (75...80).contains(Int(compact.modes[$0])) }
    try require(!lowMean.isEmpty, "Fixture did not produce low-mean entropy modes 75...80")
    let tailed = lowMean.first { stream in
      let first = Int(compact.offsets[stream]); return (compact.payload[first] & 7) != 0
    }
    try require(tailed != nil, "Low-mean fixture did not produce a partial-byte entropy tail")
    let entropy = tailed!
    try malformedPartialsCheck(device: device, queue: queue, pipeline: partialsPipeline,
      compact: compact, decoding: table, stream: entropy)
    print("PASS partials rejected malformed dense entropy stream")
    let entropyStreams = compact.modes.indices.filter { (64..<96).contains(compact.modes[$0]) }
    let byteAlignments = Set(entropyStreams.map { Int(compact.offsets[$0 + 1] & 3) })
    try require(byteAlignments == Set(0..<4),
      "Entropy fixtures did not cover trailing 0/8/16/24-bit word alignments: \(byteAlignments)")
    guard let sparseStream = compact.modes.firstIndex(of: 252),
      compact.offsets[sparseStream + 1] - compact.offsets[sparseStream] >= 4 else {
      throw ProbeError.failed("Fixture did not produce a two-event sparse stream")
    }
    // Use every entropy stream, so malformed checks cover all four word
    // alignments, escape-bearing modes, and each available partial-byte tail.
    for (name, candidate) in readerCandidates {
      for stream in entropyStreams {
        try malformedEntropyCheck(name: "\(name) stream \(stream) truncated entropy",
          device: device, queue: queue, pipeline: candidate, compact: compact,
          decoding: table, stream: stream) { _, offsets in
            offsets[stream + 1] -= 1
          }
        try malformedEntropyCheck(name: "\(name) stream \(stream) trailing entropy byte",
          device: device, queue: queue, pipeline: candidate, compact: compact,
          decoding: table, stream: stream) { payload, offsets in
            payload.insert(0, at: Int(offsets[stream + 1]))
            for index in (stream + 1)..<offsets.count { offsets[index] += 1 }
          }
        let tail = compact.payload[Int(compact.offsets[stream])] & 7
        if tail != 0 {
          try malformedEntropyCheck(name: "\(name) stream \(stream) nonzero tail padding",
            device: device, queue: queue, pipeline: candidate, compact: compact,
            decoding: table, stream: stream) { payload, offsets in
              payload[Int(offsets[stream + 1]) - 1] |= UInt8(1 << tail)
            }
        }
      }
      try malformedEntropyCheck(name: "\(name) duplicate sparse position", device: device,
        queue: queue, pipeline: candidate, compact: compact, decoding: table,
        stream: sparseStream) { payload, offsets in
          let first = Int(offsets[sparseStream])
          payload[first + 2] = payload[first]
          payload[first + 3] = payload[first + 1]
        }
      try malformedEntropyCheck(name: "\(name) truncated sparse terminal event", device: device,
        queue: queue, pipeline: candidate, compact: compact, decoding: table,
        stream: sparseStream) { _, offsets in
          offsets[sparseStream + 1] -= 1
        }
      print("PASS \(name): malformed entropy word boundaries/tails and sparse terminal checks")
    }
    try malformedEntropyCheck(name: "macro truncated entropy", device: device, queue: queue,
      pipeline: macroOwner2, compact: compact, decoding: macroTable, stream: entropy) { _, offsets in
        offsets[entropy + 1] -= 1
      }
    try malformedEntropyCheck(name: "macro nonzero tail padding", device: device, queue: queue,
      pipeline: macroOwner2, compact: compact, decoding: macroTable, stream: entropy) { payload, offsets in
        let tail = payload[Int(offsets[entropy])] & 7
        payload[Int(offsets[entropy + 1]) - 1] |= UInt8(1 << tail)
      }
    try malformedEntropyCheck(name: "macro trailing entropy byte", device: device, queue: queue,
      pipeline: macroOwner2, compact: compact, decoding: macroTable, stream: entropy) { payload, offsets in
        payload.insert(0, at: Int(offsets[entropy + 1]))
        for index in (entropy + 1)..<offsets.count { offsets[index] += 1 }
      }
    print("PASS macro malformed entropy truncation/tail/terminal checks")
    try malformedEntropyCheck(name: "reader32 truncated entropy", device: device,
      queue: queue, pipeline: reader32Owner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { _, offsets in
        offsets[entropy + 1] -= 1
      }
    try malformedEntropyCheck(name: "reader32 nonzero tail padding", device: device,
      queue: queue, pipeline: reader32Owner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        let tail = payload[Int(offsets[entropy])] & 7
        payload[Int(offsets[entropy + 1]) - 1] |= UInt8(1 << tail)
      }
    try malformedEntropyCheck(name: "reader32 trailing entropy byte", device: device,
      queue: queue, pipeline: reader32Owner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        payload.insert(0, at: Int(offsets[entropy + 1]))
        for index in (entropy + 1)..<offsets.count { offsets[index] += 1 }
      }
    print("PASS reader32 cross-32-bit alignment and malformed tail checks")
    try malformedEntropyCheck(name: "pending reuse truncated entropy", device: device,
      queue: queue, pipeline: pendingReuseOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { _, offsets in
        offsets[entropy + 1] -= 1
      }
    try malformedEntropyCheck(name: "pending reuse nonzero tail padding", device: device,
      queue: queue, pipeline: pendingReuseOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        let tail = payload[Int(offsets[entropy])] & 7
        payload[Int(offsets[entropy + 1]) - 1] |= UInt8(1 << tail)
      }
    try malformedEntropyCheck(name: "pending reuse trailing entropy byte", device: device,
      queue: queue, pipeline: pendingReuseOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        payload.insert(0, at: Int(offsets[entropy + 1]))
        for index in (entropy + 1)..<offsets.count { offsets[index] += 1 }
      }
    print("PASS pending reuse cross-word alignment and malformed tail checks")
    try malformedEntropyCheck(name: "register reuse truncated entropy", device: device,
      queue: queue, pipeline: registerReuseOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { _, offsets in
        offsets[entropy + 1] -= 1
      }
    try malformedEntropyCheck(name: "register reuse nonzero tail padding", device: device,
      queue: queue, pipeline: registerReuseOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        let tail = payload[Int(offsets[entropy])] & 7
        payload[Int(offsets[entropy + 1]) - 1] |= UInt8(1 << tail)
      }
    try malformedEntropyCheck(name: "register reuse trailing entropy byte", device: device,
      queue: queue, pipeline: registerReuseOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        payload.insert(0, at: Int(offsets[entropy + 1]))
        for index in (entropy + 1)..<offsets.count { offsets[index] += 1 }
      }
    print("PASS register reuse cross-word alignment and malformed tail checks")
    try malformedEntropyCheck(name: "dense batch truncated entropy", device: device,
      queue: queue, pipeline: denseBatchOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { _, offsets in
        offsets[entropy + 1] -= 1
      }
    try malformedEntropyCheck(name: "dense batch nonzero tail padding", device: device,
      queue: queue, pipeline: denseBatchOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        let tail = payload[Int(offsets[entropy])] & 7
        payload[Int(offsets[entropy + 1]) - 1] |= UInt8(1 << tail)
      }
    try malformedEntropyCheck(name: "dense batch trailing entropy byte", device: device,
      queue: queue, pipeline: denseBatchOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        payload.insert(0, at: Int(offsets[entropy + 1]))
        for index in (entropy + 1)..<offsets.count { offsets[index] += 1 }
      }
    print("PASS dense batch malformed entropy terminal checks")
    try malformedEntropyCheck(name: "trusted table truncated entropy", device: device,
      queue: queue, pipeline: trustedTableOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { _, offsets in
        offsets[entropy + 1] -= 1
      }
    try malformedEntropyCheck(name: "trusted table nonzero tail padding", device: device,
      queue: queue, pipeline: trustedTableOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        let tail = payload[Int(offsets[entropy])] & 7
        payload[Int(offsets[entropy + 1]) - 1] |= UInt8(1 << tail)
      }
    try malformedEntropyCheck(name: "trusted table trailing entropy byte", device: device,
      queue: queue, pipeline: trustedTableOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        payload.insert(0, at: Int(offsets[entropy + 1]))
        for index in (entropy + 1)..<offsets.count { offsets[index] += 1 }
      }
    print("PASS trusted table malformed entropy terminal checks")
    try malformedEntropyCheck(name: "lazy refill truncated entropy", device: device,
      queue: queue, pipeline: lazyRefillOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { _, offsets in
        offsets[entropy + 1] -= 1
      }
    try malformedEntropyCheck(name: "lazy refill nonzero tail padding", device: device,
      queue: queue, pipeline: lazyRefillOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        let tail = payload[Int(offsets[entropy])] & 7
        payload[Int(offsets[entropy + 1]) - 1] |= UInt8(1 << tail)
      }
    try malformedEntropyCheck(name: "lazy refill trailing entropy byte", device: device,
      queue: queue, pipeline: lazyRefillOwner2, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { payload, offsets in
        payload.insert(0, at: Int(offsets[entropy + 1]))
        for index in (entropy + 1)..<offsets.count { offsets[index] += 1 }
      }
    print("PASS lazy refill malformed entropy terminal checks")
    try malformedEntropyCheck(name: "cooperative truncated entropy", device: device,
      queue: queue, pipeline: cooperative, compact: compact, decoding: table,
      stream: entropy, includeSparse: false) { _, offsets in
        offsets[entropy + 1] -= 1
      }
    print("PASS cooperative malformed entropy truncation check")
    try malformedSparseCheck(device: device, queue: queue, pipeline: sparse, compact: compact)
    print("PASS malformed sparse stream sets failure")
    print("PASS modes=\(Set(compact.modes).sorted()) streams=\(values.count) selected=\(selected.count)")
  }
}
