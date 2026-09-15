import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMStreamingIO

private enum ProbeError: Error { case failed(String) }
private func check(_ value: @autoclosure () -> Bool, _ message: String) throws {
  if !value() { throw ProbeError.failed(message) }
}
private func buffer<T>(_ values: [T], _ device: MTLDevice) throws -> MTLBuffer {
  guard let result = values.withUnsafeBytes({ device.makeBuffer(
    bytes: $0.baseAddress!, length: max(1, $0.count), options: .storageModeShared) }) else {
    throw ProbeError.failed("buffer allocation")
  }
  return result
}
private func bytes(_ buffer: MTLBuffer, _ count: Int) -> [UInt8] {
  Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt8.self), count: count))
}
private func words(_ buffer: MTLBuffer, _ count: Int) -> [UInt32] {
  Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: count))
}
private func reflected(_ name: String, from object: Any) throws -> MTLBuffer {
  guard let value = Mirror(reflecting: object).children.first(where: { $0.label == name })?.value as? MTLBuffer else {
    throw ProbeError.failed("missing codec table \(name)")
  }
  return value
}
private func pipeline(_ name: String, _ library: MTLLibrary, _ device: MTLDevice) throws -> MTLComputePipelineState {
  guard let function = library.makeFunction(name: name) else { throw ProbeError.failed("missing \(name)") }
  return try device.makeComputePipelineState(function: function)
}
private func encodePipeline(_ scratchless: Bool, _ library: MTLLibrary, _ device: MTLDevice) throws -> MTLComputePipelineState {
  let constants = MTLFunctionConstantValues()
  var enabled = scratchless
  constants.setConstantValue(&enabled, type: .bool, index: 7)
  return try device.makeComputePipelineState(function: library.makeFunction(
    name: Metal4DSTEMKernels.pairedRuntimeTANSEncodeFunction, constantValues: constants))
}
private func submit(
  _ queue: MTLCommandQueue, _ pso: MTLComputePipelineState, buffers: [Int: MTLBuffer],
  parameters: [UInt32], width: Int
) throws {
  guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
    throw ProbeError.failed("command allocation")
  }
  encoder.setComputePipelineState(pso)
  for (index, value) in buffers { encoder.setBuffer(value, offset: 0, index: index) }
  var p = parameters
  encoder.setBytes(&p, length: p.count * 4, index: buffers.keys.contains(8) ? 7 : (p.count == 3 ? 5 : 6))
  encoder.dispatchThreads(MTLSize(width: width, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: min(width, 128), height: 1, depth: 1))
  encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
  try check(command.status == .completed, "Metal command failed: \(command.error?.localizedDescription ?? "unknown")")
}

private let scans = 389                 // partial packet, odd and not a multiple of 256
private let streamCount = 67            // adjacent mixed lanes and incomplete SIMD group
private func fixtures() -> [[UInt16]] {
  var streams = [[UInt16]]()
  streams.append([UInt16](repeating: 0, count: scans))
  streams.append([UInt16](repeating: .max, count: scans))
  for eventCount in [8, 32] {           // entropy is attempted before sparse wins
    var stream = [UInt16](repeating: 0, count: scans)
    for event in 0..<eventCount { stream[(event * 37 + 13) % scans] = 128 }
    streams.append(stream)
  }
  streams.append((0..<scans).map { UInt16(($0 * 17 + $0 / 7) % 31) })
  streams.append((0..<scans).map { UInt16(truncatingIfNeeded: $0 &* 40_503 &+ 7_919) })
  var two = [UInt16](repeating: 0, count: scans); two[13] = 128; two[scans - 1] = 1
  streams.append(two)
  while streams.count < streamCount {
    let lane = streams.count
    switch lane % 6 {
    case 0: streams.append([UInt16](repeating: UInt16(1_000 + lane), count: scans))
    case 1: streams.append((0..<scans).map { UInt16(($0 + lane * 3) % 33) })
    case 2:
      var value = [UInt16](repeating: 0, count: scans)
      for event in 0..<(lane % 25 + 3) { value[(event * 29 + lane) % scans] = UInt16(event % 100 + 1) }
      streams.append(value)
    case 3: streams.append((0..<scans).map { UInt16(truncatingIfNeeded: $0 * (lane * 997 + 1)) })
    case 4: streams.append([UInt16](repeating: 0, count: scans))
    default:
      var value = (0..<scans).map { UInt16(($0 * 5 + lane) % 27) }
      value[scans - 1] = UInt16(600 + lane); streams.append(value)
    }
  }
  return streams
}

@main enum ScratchlessHDF5EncoderProbe {
  static func main() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw ProbeError.failed("Metal unavailable")
    }
    let codec = try MetalPairedRuntimeTANSSyntheticCodec(device: device)
    let frequency = try reflected("frequencyStarts", from: codec)
    let encoding = try reflected("encoding", from: codec)
    let decoding = try reflected("decoding", from: codec)
    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    let originalEncode = try encodePipeline(false, library, device)
    let directEncode = try encodePipeline(true, library, device)
    let compact = try pipeline(Metal4DSTEMKernels.pairedRuntimeTANSCompactFunction, library, device)
    let decode = try pipeline(Metal4DSTEMKernels.pairedRuntimeTANSDecodeFunction, library, device)
    let source = fixtures()
    let rawValues = (0..<scans).flatMap { scan in source.map { $0[scan] } }
    let raw = try buffer(rawValues, device)
    let stride = 2 * scans

    func encode(_ pso: MTLComputePipelineState, direct: Bool, output: MTLBuffer, offsets: MTLBuffer) throws -> (MTLBuffer, MTLBuffer) {
      let sizes = try buffer([UInt32](repeating: 0, count: streamCount), device)
      let modes = try buffer([UInt8](repeating: 0, count: streamCount), device)
      let failure = try buffer([UInt32(0)], device)
      try submit(queue, pso, buffers: [0: raw, 1: frequency, 2: encoding, 3: output,
        4: sizes, 5: modes, 6: failure, 8: offsets],
        parameters: [UInt32(scans), UInt32(streamCount), UInt32(streamCount), 2, UInt32(stride), direct ? 1 : 0],
        width: streamCount)
      try check(words(failure, 1)[0] == 0, "encode failure \(words(failure, 1)[0])")
      return (sizes, modes)
    }
    func prefix(_ sizeBuffer: MTLBuffer) throws -> [UInt32] {
      var result = [UInt32](repeating: 0, count: streamCount + 1)
      for i in 0..<streamCount {
        let next = result[i].addingReportingOverflow(words(sizeBuffer, streamCount)[i])
        try check(!next.overflow, "offset overflow"); result[i + 1] = next.partialValue
      }
      return result
    }

    let dummyOffsets = try buffer([UInt32](repeating: 0, count: streamCount + 1), device)
    let scratch = try buffer([UInt8](repeating: 0, count: stride * streamCount), device)
    let (oldSizes, oldModes) = try encode(originalEncode, direct: false, output: scratch, offsets: dummyOffsets)
    let oldOffsetValues = try prefix(oldSizes)
    let oldOffsets = try buffer(oldOffsetValues, device)
    let oldPayload = try buffer([UInt8](repeating: 0, count: max(1, Int(oldOffsetValues.last!))), device)
    let compactFailure = try buffer([UInt32(0)], device)
    try submit(queue, compact, buffers: [0: scratch, 1: oldSizes, 2: oldOffsets, 3: oldPayload, 4: compactFailure],
      parameters: [UInt32(streamCount), UInt32(stride), oldOffsetValues.last!], width: streamCount)
    try check(words(compactFailure, 1)[0] == 0, "compact failure")

    let sizeDummy = try buffer([UInt8](repeating: 0, count: 1), device)
    let (newSizes, newModes) = try encode(directEncode, direct: false, output: sizeDummy, offsets: dummyOffsets)
    let newOffsetValues = try prefix(newSizes)
    try check(words(oldSizes, streamCount) == words(newSizes, streamCount), "size pass differs")
    try check(bytes(oldModes, streamCount) == bytes(newModes, streamCount), "mode pass differs")
    try check(oldOffsetValues == newOffsetValues, "offsets differ")
    let canaryCount = 256
    let canary: UInt8 = 0xA5
    let guardedPayload = try buffer([UInt8](repeating: 0, count: Int(newOffsetValues.last!))
      + [UInt8](repeating: canary, count: canaryCount), device)
    let newOffsets = try buffer(newOffsetValues, device)
    _ = try encode(directEncode, direct: true, output: guardedPayload, offsets: newOffsets)
    let payloadCount = Int(newOffsetValues.last!)
    try check(bytes(oldPayload, payloadCount) == bytes(guardedPayload, payloadCount), "payload bytes differ")
    try check(bytes(guardedPayload, guardedPayload.length).suffix(canaryCount).allSatisfy { $0 == canary },
      "scratchless encoder overwrote payload canary")

    let modesSeen = Set(bytes(newModes, streamCount))
    try check(modesSeen.contains(252) && modesSeen.contains(253) && modesSeen.contains(254)
      && modesSeen.contains(255) && modesSeen.contains(where: { $0 < 252 }), "fixture missed an encoding mode")

    func decoded(_ payload: MTLBuffer, offsets: MTLBuffer) throws -> [UInt16] {
      let output = try buffer([UInt16](repeating: 0, count: rawValues.count), device)
      let failure = try buffer([UInt32(0)], device)
      try submit(queue, decode, buffers: [0: payload, 1: offsets, 2: newModes, 3: decoding, 4: output, 5: failure],
        parameters: [UInt32(scans), UInt32(streamCount), UInt32(streamCount), 2, UInt32(payloadCount)], width: streamCount)
      try check(words(failure, 1)[0] == 0, "decode failure")
      return Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt16.self), count: rawValues.count))
    }
    let originalDecoded = try decoded(oldPayload, offsets: oldOffsets)
    let scratchlessDecoded = try decoded(guardedPayload, offsets: newOffsets)
    try check(originalDecoded == rawValues, "original decode mismatch")
    try check(scratchlessDecoded == rawValues, "scratchless decode mismatch")

    var malformed = newOffsetValues; malformed[streamCount] += 1
    let malformedOffsets = try buffer(malformed, device)
    let malformedOutput = try buffer([UInt16](repeating: 0, count: rawValues.count), device)
    let malformedFailure = try buffer([UInt32(0)], device)
    try submit(queue, decode, buffers: [0: guardedPayload, 1: malformedOffsets, 2: newModes,
      3: decoding, 4: malformedOutput, 5: malformedFailure],
      parameters: [UInt32(scans), UInt32(streamCount), UInt32(streamCount), 2, UInt32(payloadCount)], width: streamCount)
    try check(words(malformedFailure, 1)[0] != 0, "malformed offset was accepted")
    print("scratchless HDF5 encoder parity passed: \(streamCount) streams, \(scans) scans, \(payloadCount) bytes")
  }
}
