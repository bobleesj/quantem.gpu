import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMStreamingIO

private let interval = 512
private let pixels = 32
private let packets = 2

private enum ProbeError: Error, CustomStringConvertible {
  case failed(String)

  var description: String {
    if case .failed(let message) = self { return message }
    return "compact-offset probe failed"
  }
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
  guard condition() else { throw ProbeError.failed(message) }
}

private func makeBuffer<T>(_ values: [T], on device: MTLDevice, label: String) throws -> MTLBuffer {
  let result = values.withUnsafeBytes { bytes in
    device.makeBuffer(
      bytes: bytes.baseAddress!, length: max(bytes.count, 1), options: .storageModeShared)
  }
  guard let result else { throw ProbeError.failed("Metal could not allocate \(label)") }
  result.label = label
  return result
}

private func makeBuffer(bytes: Int, on device: MTLDevice, label: String) throws -> MTLBuffer {
  guard let result = device.makeBuffer(length: max(bytes, 1), options: .storageModeShared) else {
    throw ProbeError.failed("Metal could not allocate \(label)")
  }
  result.label = label
  memset(result.contents(), 0, result.length)
  return result
}

private func decodingBuffer(from codec: MetalPairedRuntimeTANSSyntheticCodec) throws -> MTLBuffer {
  guard let table = Mirror(reflecting: codec).children.first(where: { $0.label == "decoding" })?.value
    as? MTLBuffer else {
    throw ProbeError.failed("Could not locate the synthetic codec's decoding table")
  }
  return table
}

private func rawStream(_ seed: UInt32) -> [UInt16] {
  var values = (0..<interval).map { scan in
    UInt16(truncatingIfNeeded: UInt32(scan &* 40_503 &+ 7_919) &+ seed)
  }
  // Preserve the genuine high-count boundary while keeping the stream otherwise
  // full-width and incompressible enough for raw mode.
  values[interval - 1] = 65_535
  return values
}

private func sparseStream() -> [UInt16] {
  var values = [UInt16](repeating: 0, count: interval)
  values[13] = 128
  values[511] = 1
  return values
}

private func entropyStream(_ phase: Int) -> [UInt16] {
  (0..<interval).map { (scan: Int) -> UInt16 in
    let value = scan * 17 + scan / 7 + phase
    return UInt16(value % 31)
  }
}

private func streams() -> [[UInt16]] {
  var result = [[UInt16]]()
  result.reserveCapacity(64)

  // A maximally sized first 32-stream block: 32 x 1024 raw bytes. The terminal
  // offset at stream 32 must become the next block base, not a wrapped delta.
  for stream in 0..<32 {
    result.append(rawStream(0x1357_9bdf ^ UInt32(stream &* 0x9e37)))
  }

  // The next block starts exactly on the compact-offset anchor. It contains all
  // special modes, a terminal-sentinel raw stream, and escape-bearing entropy.
  result.append([UInt16](repeating: 0, count: interval))
  result.append([UInt16](repeating: 65_535, count: interval))
  result.append(sparseStream())
  result.append(entropyStream(0))
  result.append(rawStream(0x2468_ace1))
  var escapes = entropyStream(3)
  for position in stride(from: 5, to: interval, by: 47) {
    escapes[position] = UInt16(400 + position)
  }
  result.append(escapes)
  while result.count < 64 {
    switch result.count % 5 {
    case 0: result.append([UInt16](repeating: 0, count: interval))
    case 1: result.append([UInt16](repeating: 65_535, count: interval))
    case 2: result.append(sparseStream())
    case 3: result.append(entropyStream(result.count))
    default: result.append(rawStream(0xa5a5_0000 ^ UInt32(result.count)))
    }
  }
  return result
}

private func compactOffsets(
  library: MTLLibrary, device: MTLDevice, queue: MTLCommandQueue,
  source: [UInt32], sourceStreams: Int, destinationFirst: Int,
  payloadFirst: Int, destinationStreams: Int, expectedRecordPayloadBytes: Int,
  destination: MTLBuffer
) throws -> UInt32 {
  guard let function = library.makeFunction(name: "paired_runtime_tans_rebase_block32_offsets") else {
    throw ProbeError.failed("Block32 offset conversion kernel is missing")
  }
  let pipeline = try device.makeComputePipelineState(function: function)
  let sourceBuffer = try makeBuffer(source, on: device, label: "full record offsets")
  let failure = try makeBuffer([UInt32(0)], on: device, label: "offset conversion status")
  var parameters: [UInt32] = [
    UInt32(sourceStreams), UInt32(destinationFirst), UInt32(payloadFirst),
    UInt32(destinationStreams), UInt32(expectedRecordPayloadBytes),
  ]
  guard let command = queue.makeCommandBuffer(),
    let encoder = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("Could not encode block32 offset conversion")
  }
  encoder.label = "block32 offset conversion"
  encoder.setComputePipelineState(pipeline)
  encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
  encoder.setBuffer(destination, offset: 0, index: 1)
  encoder.setBuffer(failure, offset: 0, index: 2)
  encoder.setBytes(&parameters, length: parameters.count * 4, index: 3)
  encoder.dispatchThreads(
    MTLSize(width: sourceStreams + 1, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  encoder.endEncoding()
  command.commit()
  command.waitUntilCompleted()
  try require(command.status == .completed,
    "Offset conversion command failed: \(command.error?.localizedDescription ?? "unknown")")
  return failure.contents().load(as: UInt32.self)
}

private func packedOffsetBytes(streamCount: Int) -> Int {
  let groupBases = streamCount / 32 + 1
  let startsOffset = (groupBases * MemoryLayout<UInt32>.stride + 3) & ~3
  return startsOffset + (streamCount + 1) * MemoryLayout<UInt16>.stride
}

private func verifyPackedOffsets(
  _ buffer: MTLBuffer, fullOffsets: [UInt32], streamCount: Int
) throws {
  let baseCount = streamCount / 32 + 1
  let startByteOffset = (baseCount * MemoryLayout<UInt32>.stride + 3) & ~3
  let bases = Array(UnsafeBufferPointer(
    start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: baseCount))
  let starts = Array(UnsafeBufferPointer(
    start: buffer.contents().advanced(by: startByteOffset)
      .assumingMemoryBound(to: UInt16.self), count: streamCount + 1))
  for group in 0..<baseCount {
    let stream = group * 32
    try require(bases[group] == fullOffsets[stream],
      "Block base \(group) mismatch: \(bases[group]) != \(fullOffsets[stream])")
  }
  for index in 0...streamCount {
    let groupStart = (index / 32) * 32
    let expected = fullOffsets[index] - fullOffsets[groupStart]
    try require(starts[index] == UInt16(expected),
      "Block-relative start \(index) mismatch: \(starts[index]) != \(expected)")
  }
  try require(starts[streamCount] == 0, "Final sentinel is not a zero-relative block start")
}

private func expectConversionFailure(
  name: String, library: MTLLibrary, device: MTLDevice, queue: MTLCommandQueue,
  source: [UInt32], sourceStreams: Int, destinationStreams: Int,
  expectedRecordPayloadBytes: Int, expectedFailureMask: UInt32
) throws {
  let destination = try makeBuffer(
    bytes: packedOffsetBytes(streamCount: destinationStreams), on: device,
    label: "malformed compact offsets \(name)")
  let status = try compactOffsets(
    library: library, device: device, queue: queue, source: source,
    sourceStreams: sourceStreams, destinationFirst: 0, payloadFirst: 0,
    destinationStreams: destinationStreams,
    expectedRecordPayloadBytes: expectedRecordPayloadBytes, destination: destination)
  try require(status != 0, "Malformed block32 offsets were accepted: \(name)")
  try require(status & expectedFailureMask != 0,
    "Malformed block32 offsets set unexpected status \(status): \(name)")
  print("PASS rejected malformed block32 offsets: \(name), status=\(status)")
}

private func selectedDP(
  name: String, pipeline: MTLComputePipelineState, device: MTLDevice,
  queue: MTLCommandQueue, payload: MTLBuffer, offsets: MTLBuffer,
  modes: MTLBuffer, decoding: MTLBuffer, payloadBytes: Int,
  source: [[UInt16]], scan: Int
) throws {
  let output = try makeBuffer([UInt32](repeating: 0, count: pixels),
    on: device, label: "selected DP output")
  let failure = try makeBuffer([UInt32(0)], on: device, label: "selected DP status")
  var parameters: [UInt32] = [UInt32(pixels), UInt32(packets), UInt32(scan), UInt32(payloadBytes)]
  guard let command = queue.makeCommandBuffer(),
    let encoder = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("Could not encode selected DP \(name)")
  }
  encoder.setComputePipelineState(pipeline)
  for (index, buffer) in [payload, offsets, modes, decoding, output, failure].enumerated() {
    encoder.setBuffer(buffer, offset: 0, index: index)
  }
  encoder.setBytes(&parameters, length: parameters.count * 4, index: 6)
  encoder.dispatchThreads(
    MTLSize(width: pixels, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
  encoder.endEncoding()
  command.commit()
  command.waitUntilCompleted()
  try require(command.status == .completed,
    "Selected DP \(name) command failed: \(command.error?.localizedDescription ?? "unknown")")
  try require(failure.contents().load(as: UInt32.self) == 0,
    "Selected DP \(name) signaled failure")
  let actual = Array(UnsafeBufferPointer(
    start: output.contents().assumingMemoryBound(to: UInt32.self), count: pixels))
  let packet = scan / interval
  let local = scan % interval
  let expected = (0..<pixels).map { UInt32(source[packet * pixels + $0][local]) }
  try require(actual == expected, "Selected DP \(name) differs from exact source counts")
}

private func decodeAll(
  pipeline: MTLComputePipelineState, device: MTLDevice, queue: MTLCommandQueue,
  payload: MTLBuffer, offsets: MTLBuffer, modes: MTLBuffer,
  decoding: MTLBuffer, payloadBytes: Int, source: [[UInt16]]
) throws {
  let streamCount = source.count
  let output = try makeBuffer([UInt16](repeating: 0, count: interval * streamCount),
    on: device, label: "compact full-decode output")
  let failure = try makeBuffer([UInt32(0)], on: device, label: "compact full-decode status")
  var parameters: [UInt32] = [
    UInt32(interval), UInt32(streamCount), UInt32(streamCount), 2, UInt32(payloadBytes),
  ]
  guard let command = queue.makeCommandBuffer(),
    let encoder = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("Could not encode compact full decode")
  }
  encoder.setComputePipelineState(pipeline)
  for (index, buffer) in [payload, offsets, modes, decoding, output, failure].enumerated() {
    encoder.setBuffer(buffer, offset: 0, index: index)
  }
  encoder.setBytes(&parameters, length: parameters.count * 4, index: 6)
  encoder.dispatchThreads(
    MTLSize(width: streamCount, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  encoder.endEncoding()
  command.commit()
  command.waitUntilCompleted()
  try require(command.status == .completed,
    "Compact full decode failed: \(command.error?.localizedDescription ?? "unknown")")
  try require(failure.contents().load(as: UInt32.self) == 0,
    "Compact full decode signaled failure")
  let actual = Array(UnsafeBufferPointer(
    start: output.contents().assumingMemoryBound(to: UInt16.self),
    count: interval * streamCount))
  let expected = (0..<interval).flatMap { scan in source.map { $0[scan] } }
  try require(actual == expected,
    "Compact full decode differs from exact signed-high-u16 source samples")
}

private func detectorDelta(
  name: String, pipeline: MTLComputePipelineState, parametersIndex: Int,
  partials: Bool, device: MTLDevice, queue: MTLCommandQueue,
  payload: MTLBuffer, offsets: MTLBuffer, modes: MTLBuffer,
  decoding: MTLBuffer, payloadBytes: Int, source: [[UInt16]],
  selected: MTLBuffer, coefficients: [Int32]
) throws {
  let coefficientBuffer = try makeBuffer(coefficients, on: device, label: "signed coefficients")
  let groups = 1
  let output = try makeBuffer([UInt32](repeating: 0, count: packets * interval),
    on: device, label: partials ? "polar index partial fields" : "detector output")
  let failure = try makeBuffer([UInt32(0)], on: device, label: "detector status")
  guard let command = queue.makeCommandBuffer(),
    let encoder = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("Could not encode \(name)")
  }
  encoder.setComputePipelineState(pipeline)
  for (index, buffer) in [payload, offsets, modes, decoding, selected,
    coefficientBuffer, output, failure].enumerated() {
    encoder.setBuffer(buffer, offset: 0, index: index)
  }
  var parameters: [UInt32] = partials
    ? [UInt32(pixels), UInt32(packets), UInt32(pixels), UInt32(groups), UInt32(payloadBytes)]
    : [UInt32(pixels), UInt32(packets), UInt32(pixels), 0, UInt32(payloadBytes), 1]
  encoder.setBytes(&parameters, length: parameters.count * 4, index: parametersIndex)
  encoder.dispatchThreadgroups(
    MTLSize(width: partials ? groups : 1, height: partials ? packets : 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: partials ? 32 : 128, height: 1, depth: 1))
  encoder.endEncoding()
  command.commit()
  command.waitUntilCompleted()
  try require(command.status == .completed,
    "\(name) command failed: \(command.error?.localizedDescription ?? "unknown")")
  try require(failure.contents().load(as: UInt32.self) == 0,
    "\(name) signaled failure \(failure.contents().load(as: UInt32.self))")
  let actual = Array(UnsafeBufferPointer(
    start: output.contents().assumingMemoryBound(to: UInt32.self), count: packets * interval))
  var expected = [UInt32](repeating: 0, count: packets * interval)
  for packet in 0..<packets {
    for scan in 0..<interval {
      let sum = (0..<pixels).reduce(Int64(0)) { partial, pixel in
        partial + Int64(source[packet * pixels + pixel][scan]) * Int64(coefficients[pixel])
      }
      expected[packet * interval + scan] = UInt32(bitPattern: Int32(sum))
    }
  }
  try require(actual == expected,
    "\(name) differs from exact signed UInt16 reference (including modulo UInt32 sums)")
}

@main
struct CompactOffsetProbe {
  static func main() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw ProbeError.failed("Metal is unavailable")
    }
    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    let codec = try MetalPairedRuntimeTANSSyntheticCodec(device: device)
    let source = streams()
    let encoded = try codec.roundTrip(streams: source, logicalDtype: .uint16)
    try require(encoded.decodedStreams == source, "Fixture did not round trip exactly")
    try require(encoded.modes[0..<32].allSatisfy({ $0 == 254 }),
      "First block must contain exactly 32 raw 1024-byte streams")
    let secondBlockModes = Array(encoded.modes[32..<64])
    for mode in [UInt8(252), 253, 254, 255] {
      try require(secondBlockModes.contains(mode), "Second block missed mode \(mode)")
    }
    try require(secondBlockModes.contains(where: { $0 < 252 }),
      "Second block missed entropy mode")
    let rawBlockBytes = encoded.offsets[32] - encoded.offsets[0]
    try require(rawBlockBytes == 32 * 1024,
      "First raw block is \(rawBlockBytes) bytes, expected exactly 32768")

    // Match the resident's payload rebasing and two adjacent 32-stream record
    // conversions. Each record carries its own UInt32 offset sentinel.
    let payloadPrefix = 16
    let fullOffsets = encoded.offsets.map { $0 + UInt32(payloadPrefix) }
    let destination = try makeBuffer(
      bytes: packedOffsetBytes(streamCount: source.count), on: device,
      label: "resident block32 offsets")
    let firstRecord = Array(encoded.offsets[0...32])
    let firstStatus = try compactOffsets(
      library: library, device: device, queue: queue, source: firstRecord,
      sourceStreams: 32, destinationFirst: 0, payloadFirst: payloadPrefix,
      destinationStreams: source.count,
      expectedRecordPayloadBytes: Int(encoded.offsets[32]), destination: destination)
    try require(firstStatus == 0, "First record conversion failed with \(firstStatus)")
    let secondRecord = (32...64).map { encoded.offsets[$0] - encoded.offsets[32] }
    let secondStatus = try compactOffsets(
      library: library, device: device, queue: queue, source: secondRecord,
      sourceStreams: 32, destinationFirst: 32,
      payloadFirst: payloadPrefix + Int(encoded.offsets[32]),
      destinationStreams: source.count,
      expectedRecordPayloadBytes: Int(encoded.offsets[64] - encoded.offsets[32]),
      destination: destination)
    try require(secondStatus == 0, "Second record conversion failed with \(secondStatus)")
    try verifyPackedOffsets(destination, fullOffsets: fullOffsets, streamCount: source.count)
    try require(destination.length == packedOffsetBytes(streamCount: source.count),
      "Packed resident allocation contains unexpected padding")
    print("PASS GPU block32 conversion: two records, 64 streams, all starts and final sentinel exact")

    var overwideBlock = [UInt32](repeating: 0, count: 33)
    for index in 1...31 { overwideBlock[index] = UInt32(index * 1024) }
    overwideBlock[32] = 32_769 // one byte beyond 32 x 1024
    try expectConversionFailure(
      name: "block span 32769 bytes", library: library, device: device, queue: queue,
      source: overwideBlock, sourceStreams: 32, destinationStreams: 32,
      expectedRecordPayloadBytes: 32_769, expectedFailureMask: 4)
    let wrongSentinel = firstRecord
    try expectConversionFailure(
      name: "record terminal disagrees with receipt", library: library,
      device: device, queue: queue, source: wrongSentinel, sourceStreams: 32,
      destinationStreams: 32, expectedRecordPayloadBytes: Int(encoded.offsets[32]) + 1,
      expectedFailureMask: 2)

    let payloadBytes = payloadPrefix + encoded.payload.count
    let payload = try makeBuffer(
      [UInt8](repeating: 0, count: payloadPrefix) + encoded.payload + [UInt8](repeating: 0, count: 8),
      on: device, label: "rebased synthetic payload")
    let modeBuffer = try makeBuffer(encoded.modes, on: device, label: "synthetic modes")
    let table = try decodingBuffer(from: codec)

    let decodeConstants = MTLFunctionConstantValues()
    var compactOffsetsEnabled = true
    decodeConstants.setConstantValue(&compactOffsetsEnabled, type: .bool, index: 20)
    let decodeFunction = try library.makeFunction(
      name: "paired_runtime_tans_decode", constantValues: decodeConstants)
    let decodePipeline = try device.makeComputePipelineState(function: decodeFunction)
    try decodeAll(pipeline: decodePipeline, device: device, queue: queue,
      payload: payload, offsets: destination, modes: modeBuffer, decoding: table,
      payloadBytes: payloadBytes, source: source)
    print("PASS generic decode compact accessor: all 32768 uint16 values and terminal sentinel exact")

    let dpConstants = MTLFunctionConstantValues()
    dpConstants.setConstantValue(&compactOffsetsEnabled, type: .bool, index: 20)
    let dpFunction = try library.makeFunction(
      name: "paired_runtime_tans_selected_dp", constantValues: dpConstants)
    let dpPipeline = try device.makeComputePipelineState(function: dpFunction)
    for scan in [0, 511, 512, 525, 1023] {
      try selectedDP(name: "scan \(scan)", pipeline: dpPipeline, device: device,
        queue: queue, payload: payload, offsets: destination, modes: modeBuffer,
        decoding: table, payloadBytes: payloadBytes, source: source, scan: scan)
    }
    print("PASS selected-DP compact accessor: signed high-u16 and all modes at packet boundaries")

    let detectorConstants = MTLFunctionConstantValues()
    var streamsPerLane = UInt32(2)
    detectorConstants.setConstantValue(&streamsPerLane, type: .uint, index: 0)
    detectorConstants.setConstantValue(&compactOffsetsEnabled, type: .bool, index: 20)
    let detectorFunction = try library.makeFunction(
      name: "paired_runtime_tans_detector_packet_owner2", constantValues: detectorConstants)
    let detectorPipeline = try device.makeComputePipelineState(function: detectorFunction)
    let selected = try makeBuffer((0..<pixels).map(UInt32.init), on: device,
      label: "detector selected pixels")
    let coefficients = (0..<pixels).map { $0.isMultiple(of: 2) ? Int32(-1) : Int32(1) }
    try detectorDelta(name: "packet-owner2", pipeline: detectorPipeline,
      parametersIndex: 8, partials: false, device: device, queue: queue,
      payload: payload, offsets: destination, modes: modeBuffer, decoding: table,
      payloadBytes: payloadBytes, source: source, selected: selected,
      coefficients: coefficients)

    let indexConstants = MTLFunctionConstantValues()
    var leafWidth = UInt32(pixels)
    var outputFields = UInt32(1)
    indexConstants.setConstantValue(&leafWidth, type: .uint, index: 5)
    indexConstants.setConstantValue(&outputFields, type: .uint, index: 6)
    indexConstants.setConstantValue(&compactOffsetsEnabled, type: .bool, index: 20)
    let indexFunction = try library.makeFunction(
      name: "paired_runtime_tans_detector_partials", constantValues: indexConstants)
    let indexPipeline = try device.makeComputePipelineState(function: indexFunction)
    try detectorDelta(name: "polar-index partial fields", pipeline: indexPipeline,
      parametersIndex: 8, partials: true, device: device, queue: queue,
      payload: payload, offsets: destination, modes: modeBuffer, decoding: table,
      payloadBytes: payloadBytes, source: source, selected: selected,
      coefficients: coefficients)
    print("PASS detector and polar-index field kernels: exact signed sums for both packets")
  }
}
