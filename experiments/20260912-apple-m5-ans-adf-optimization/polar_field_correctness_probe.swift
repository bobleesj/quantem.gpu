import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels

private let scans = 512
private let leaves = 576
private let fields = 612
private let maximumLeaf = UInt32(65_535 * 64)

private enum PolarProbeError: Error { case failed(String) }

private func require(_ value: @autoclosure () -> Bool, _ message: String) throws {
  if !value() { throw PolarProbeError.failed(message) }
}

private func buffer<T>(_ values: [T], device: MTLDevice) throws -> MTLBuffer {
  guard let result = values.withUnsafeBytes({ bytes in
    device.makeBuffer(bytes: bytes.baseAddress!, length: max(1, bytes.count), options: .storageModeShared)
  }) else { throw PolarProbeError.failed("Metal allocation failed") }
  return result
}

private func values<T>(from buffer: MTLBuffer, count: Int, as: T.Type) -> [T] {
  Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: T.self), count: count))
}

private func fixtureLeaves() -> [UInt32] {
  var result = [UInt32](repeating: 0, count: leaves * scans)
  for leaf in 0..<leaves {
    for scan in 0..<scans {
      let value: UInt32
      switch leaf % 9 {
      case 0: value = 0
      case 1: value = UInt32(37 + leaf) // constant nonzero, width-zero delta form
      case 2: value = scan.isMultiple(of: 64) ? maximumLeaf : 0
      case 3: value = UInt32(1_000 + leaf) + UInt32(scan & 1)
      case 4: value = UInt32((scan * 257 + leaf * 13) & 65_535)
      case 5: value = UInt32((scan * 65_537 + leaf * 8191) % Int(maximumLeaf + 1))
      case 6: value = UInt32((scan * 31 + leaf) & 255)
      case 7: value = scan == 511 ? maximumLeaf : UInt32(200 + leaf)
      default: value = UInt32((scan * 1_048_573 + leaf * 97) % Int(maximumLeaf + 1))
      }
      result[leaf * scans + scan] = value
    }
  }
  // Force the first root to its exact uint16 detector-sum ceiling at scan 64.
  for leaf in 0..<16 { result[leaf * scans + 64] = maximumLeaf }
  return result
}

private func referenceFields(_ leafValues: [UInt32]) throws -> [UInt32] {
  var result = [UInt32](repeating: 0, count: fields * scans)
  for leaf in 0..<leaves {
    result.replaceSubrange((leaf * scans)..<((leaf + 1) * scans),
      with: leafValues[(leaf * scans)..<((leaf + 1) * scans)])
  }
  for root in 0..<36 {
    for scan in 0..<scans {
      var sum = UInt64(0)
      for child in 0..<16 { sum += UInt64(leafValues[(root * 16 + child) * scans + scan]) }
      try require(sum <= UInt64(UInt32.max), "Root \(root) overflowed UInt32")
      result[(leaves + root) * scans + scan] = UInt32(sum)
    }
  }
  try require(result[leaves * scans + 64] == 67_107_840, "Wide-root sentinel is missing")
  return result
}

private func unpack(payload: [UInt32], offsets: [UInt32], tags: [UInt8]) throws -> [UInt32] {
  var result = [UInt32](repeating: 0, count: fields * scans)
  for field in 0..<fields {
    var first = Int(offsets[field]), end = Int(offsets[field + 1])
    let tag = UInt32(tags[field]), width = Int(tag & 63)
    try require(width <= 32 && first <= end && end <= payload.count, "CPU unpack metadata invalid")
    var base = UInt32(0)
    if tag & 128 != 0 { try require(first < end, "CPU unpack missing base"); base = payload[first]; first += 1 }
    try require(end - first == (scans * width + 31) / 32, "CPU unpack extent mismatch")
    for scan in 0..<scans {
      if width == 0 { result[field * scans + scan] = base; continue }
      let bit = scan * width, at = first + bit / 32, shift = bit & 31
      var word = UInt64(payload[at])
      if shift + width > 32 { word |= UInt64(payload[at + 1]) << 32 }
      let mask = width == 32 ? UInt64(UInt32.max) : (UInt64(1) << width) - 1
      result[field * scans + scan] = base + UInt32((word >> shift) & mask)
    }
  }
  return result
}

private func encode(
  _ pipeline: MTLComputePipelineState, command: MTLCommandBuffer,
  buffers: [MTLBuffer], parameters: inout [UInt32], parameterIndex: Int,
  threads: Int, threadgroup: Int
) throws {
  guard let encoder = command.makeComputeCommandEncoder() else {
    throw PolarProbeError.failed("Could not create compute encoder")
  }
  encoder.setComputePipelineState(pipeline)
  for (index, item) in buffers.enumerated() { encoder.setBuffer(item, offset: 0, index: index) }
  encoder.setBytes(&parameters, length: parameters.count * 4, index: parameterIndex)
  encoder.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: threadgroup, height: 1, depth: 1))
  encoder.endEncoding()
}

private func malformedQuery(
  name: String, pipeline: MTLComputePipelineState, device: MTLDevice, queue: MTLCommandQueue,
  payloadValues: [UInt32], offsetValues: [UInt32], tagValues: [UInt8],
  mutate: (inout [UInt32], inout [UInt8]) -> Void
) throws {
  var offsets = offsetValues, tags = tagValues; mutate(&offsets, &tags)
  let payload = try buffer(payloadValues, device: device), offset = try buffer(offsets, device: device)
  let tag = try buffer(tags, device: device), selected = try buffer([UInt32(0)], device: device)
  let coefficients = try buffer([Int32(1)], device: device)
  let output = try buffer([UInt32](repeating: 0, count: scans), device: device)
  let failure = try buffer([UInt32(0)], device: device)
  guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
    throw PolarProbeError.failed("Could not create malformed query")
  }
  encoder.setComputePipelineState(pipeline)
  for (index, item) in [payload, offset, tag, selected, coefficients, output, failure].enumerated() {
    encoder.setBuffer(item, offset: 0, index: index)
  }
  var parameters: [UInt32] = [1, UInt32(fields), 1, UInt32(payloadValues.count), 0]
  encoder.setBytes(&parameters, length: parameters.count * 4, index: 7)
  encoder.setThreadgroupMemoryLength(16, index: 0)
  encoder.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
  try require(command.status == .completed, "\(name) command failed unexpectedly")
  try require(failure.contents().load(as: UInt32.self) != 0, "\(name) was not rejected")
}

@main
struct PolarFieldCorrectnessProbe {
  static func main() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw PolarProbeError.failed("Metal unavailable")
    }
    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else { throw PolarProbeError.failed("Missing \(name)") }
      return try device.makeComputePipelineState(function: function)
    }
    let roots = try pipeline("paired_runtime_tans_polar_roots")
    let sizes = try pipeline("paired_runtime_tans_polar_sizes")
    let pack = try pipeline("paired_runtime_tans_polar_pack")
    let query = try pipeline("paired_runtime_tans_polar_query")
    let leafValues = fixtureLeaves(), expected = try referenceFields(leafValues)
    let leavesBuffer = try buffer(leafValues, device: device)
    let fieldBuffer = try buffer([UInt32](repeating: 0, count: fields * scans), device: device)
    let tagsBuffer = try buffer([UInt8](repeating: 0, count: fields), device: device)
    let sizesBuffer = try buffer([UInt32](repeating: 0, count: fields), device: device)
    let failure = try buffer([UInt32(0)], device: device)
    guard let command = queue.makeCommandBuffer() else { throw PolarProbeError.failed("No command buffer") }
    var rootParameters = [UInt32(1)]
    try encode(roots, command: command, buffers: [leavesBuffer, fieldBuffer, failure],
      parameters: &rootParameters, parameterIndex: 3, threads: fields * scans, threadgroup: 256)
    var sizeParameters: [UInt32] = [1, UInt32(fields)]
    try encode(sizes, command: command, buffers: [fieldBuffer, tagsBuffer, sizesBuffer, failure],
      parameters: &sizeParameters, parameterIndex: 4, threads: fields, threadgroup: 128)
    command.commit(); command.waitUntilCompleted()
    try require(command.status == .completed && failure.contents().load(as: UInt32.self) == 0,
      "Root/size kernels failed")
    let actualFields = values(from: fieldBuffer, count: fields * scans, as: UInt32.self)
    try require(actualFields == expected, "Polar roots differ from independent UInt64 reference")
    let sizeValues = values(from: sizesBuffer, count: fields, as: UInt32.self)
    var offsets = [UInt32](repeating: 0, count: fields + 1)
    for index in 0..<fields { offsets[index + 1] = offsets[index] + sizeValues[index] }
    let offsetsBuffer = try buffer(offsets, device: device)
    let payload = try buffer([UInt32](repeating: 0, count: Int(offsets.last!)), device: device)
    memset(failure.contents(), 0, failure.length)
    guard let packCommand = queue.makeCommandBuffer() else { throw PolarProbeError.failed("No pack command") }
    var packParameters: [UInt32] = [1, UInt32(fields), offsets.last!]
    try encode(pack, command: packCommand,
      buffers: [fieldBuffer, tagsBuffer, offsetsBuffer, payload, failure],
      parameters: &packParameters, parameterIndex: 5, threads: fields, threadgroup: 128)
    packCommand.commit(); packCommand.waitUntilCompleted()
    try require(packCommand.status == .completed && failure.contents().load(as: UInt32.self) == 0,
      "Polar pack failed")
    let payloadValues = values(from: payload, count: Int(offsets.last!), as: UInt32.self)
    let tagValues = values(from: tagsBuffer, count: fields, as: UInt8.self)
    let unpacked = try unpack(payload: payloadValues, offsets: offsets, tags: tagValues)
    try require(unpacked == expected,
      "Full CPU unpack differs from source fields")

    let selectedValues: [UInt32] = [576, 0, 1, 577, 16, 611, 560]
    let coefficientValues: [Int32] = [1, -1, -1, 1, -1, 1, -1]
    var reference = [UInt64](repeating: 0, count: scans)
    for scan in 0..<scans {
      let signed = zip(selectedValues, coefficientValues).reduce(Int64(0)) {
        $0 + Int64(expected[Int($1.0) * scans + scan]) * Int64($1.1)
      }
      try require(signed >= 0 && signed <= Int64(UInt32.max), "Signed query reference overflow")
      reference[scan] = UInt64(signed)
    }
    let selected = try buffer(selectedValues, device: device)
    let coefficients = try buffer(coefficientValues, device: device)
    let output = try buffer([UInt32](repeating: 0, count: scans), device: device)
    memset(failure.contents(), 0, failure.length)
    guard let queryCommand = queue.makeCommandBuffer(), let encoder = queryCommand.makeComputeCommandEncoder() else {
      throw PolarProbeError.failed("No query command")
    }
    encoder.setComputePipelineState(query)
    for (index, item) in [payload, offsetsBuffer, tagsBuffer, selected, coefficients, output, failure].enumerated() {
      encoder.setBuffer(item, offset: 0, index: index)
    }
    var queryParameters: [UInt32] = [1, UInt32(fields), UInt32(selectedValues.count), offsets.last!, 0]
    encoder.setBytes(&queryParameters, length: queryParameters.count * 4, index: 7)
    encoder.setThreadgroupMemoryLength(selectedValues.count * 16, index: 0)
    encoder.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding(); queryCommand.commit(); queryCommand.waitUntilCompleted()
    try require(queryCommand.status == .completed && failure.contents().load(as: UInt32.self) == 0,
      "Polar query failed")
    try require(values(from: output, count: scans, as: UInt32.self).map(UInt64.init) == reference,
      "Signed polar query differs from UInt64 reference")

    // The physical polar hierarchy is bounded to 67,107,840. Exercise the
    // generic UInt32 pack/query ABI separately without feeding impossible leaf
    // sums through the root constructor.
    var directFields = [UInt32](repeating: 0, count: fields * scans)
    for scan in 0..<scans {
      directFields[scan] = scan.isMultiple(of: 17)
        ? UInt32.max : UInt32(truncatingIfNeeded: scan &* 2_654_435_761)
      directFields[scans + scan] = UInt32.max - 17
    }
    let directFieldBuffer = try buffer(directFields, device: device)
    let directTagsBuffer = try buffer([UInt8](repeating: 0, count: fields), device: device)
    let directSizesBuffer = try buffer([UInt32](repeating: 0, count: fields), device: device)
    memset(failure.contents(), 0, failure.length)
    guard let directSizeCommand = queue.makeCommandBuffer() else {
      throw PolarProbeError.failed("No direct-field size command")
    }
    var directSizeParameters: [UInt32] = [1, UInt32(fields)]
    try encode(sizes, command: directSizeCommand,
      buffers: [directFieldBuffer, directTagsBuffer, directSizesBuffer, failure],
      parameters: &directSizeParameters, parameterIndex: 4, threads: fields, threadgroup: 128)
    directSizeCommand.commit(); directSizeCommand.waitUntilCompleted()
    try require(directSizeCommand.status == .completed
      && failure.contents().load(as: UInt32.self) == 0, "Direct-field sizes failed")
    let directSizes = values(from: directSizesBuffer, count: fields, as: UInt32.self)
    var directOffsets = [UInt32](repeating: 0, count: fields + 1)
    for field in 0..<fields { directOffsets[field + 1] = directOffsets[field] + directSizes[field] }
    let directOffsetsBuffer = try buffer(directOffsets, device: device)
    let directPayload = try buffer(
      [UInt32](repeating: 0, count: Int(directOffsets.last!)), device: device)
    memset(failure.contents(), 0, failure.length)
    guard let directPackCommand = queue.makeCommandBuffer() else {
      throw PolarProbeError.failed("No direct-field pack command")
    }
    var directPackParameters: [UInt32] = [1, UInt32(fields), directOffsets.last!]
    try encode(pack, command: directPackCommand,
      buffers: [directFieldBuffer, directTagsBuffer, directOffsetsBuffer, directPayload, failure],
      parameters: &directPackParameters, parameterIndex: 5, threads: fields, threadgroup: 128)
    directPackCommand.commit(); directPackCommand.waitUntilCompleted()
    try require(directPackCommand.status == .completed
      && failure.contents().load(as: UInt32.self) == 0, "Direct-field pack failed")
    let directPayloadValues = values(
      from: directPayload, count: Int(directOffsets.last!), as: UInt32.self)
    let directTags = values(from: directTagsBuffer, count: fields, as: UInt8.self)
    try require(directTags[0] & 63 == 32 && directTags[0] & 128 == 0,
      "UInt32.max fixture did not select raw width 32")
    try require(directTags[1] == 128,
      "Constant UInt32 fixture did not select minimum-plus-zero-delta")
    let directUnpacked = try unpack(
      payload: directPayloadValues, offsets: directOffsets, tags: directTags)
    try require(directUnpacked == directFields, "UInt32-limit full pack round trip failed")

    let directSelected = try buffer([UInt32(0), UInt32(1)], device: device)
    let directCoefficients = try buffer([Int32(-1), Int32(1)], device: device)
    let directOutput = try buffer([UInt32](repeating: 0, count: scans), device: device)
    memset(failure.contents(), 0, failure.length)
    guard let directQueryCommand = queue.makeCommandBuffer(),
      let directEncoder = directQueryCommand.makeComputeCommandEncoder()
    else { throw PolarProbeError.failed("No direct-field query command") }
    directEncoder.setComputePipelineState(query)
    for (index, item) in [directPayload, directOffsetsBuffer, directTagsBuffer,
      directSelected, directCoefficients, directOutput, failure].enumerated() {
      directEncoder.setBuffer(item, offset: 0, index: index)
    }
    var directQueryParameters: [UInt32] = [1, UInt32(fields), 2, directOffsets.last!, 0]
    directEncoder.setBytes(&directQueryParameters,
      length: directQueryParameters.count * 4, index: 7)
    directEncoder.setThreadgroupMemoryLength(2 * 16, index: 0)
    directEncoder.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    directEncoder.endEncoding(); directQueryCommand.commit(); directQueryCommand.waitUntilCompleted()
    try require(directQueryCommand.status == .completed
      && failure.contents().load(as: UInt32.self) == 0, "Direct-field query failed")
    let modulus = UInt64(UInt32.max) + 1
    let directReference = (0..<scans).map { scan in
      UInt32((UInt64(directFields[scans + scan]) + modulus
        - UInt64(directFields[scan])) % modulus)
    }
    try require(values(from: directOutput, count: scans, as: UInt32.self) == directReference,
      "Modulo signed UInt32 query differs from independent UInt64 reference")

    try malformedQuery(name: "invalid polar low-bit width 33", pipeline: query,
      device: device, queue: queue, payloadValues: directPayloadValues,
      offsetValues: directOffsets, tagValues: directTags) { _, tags in tags[0] = 33 }

    try malformedQuery(name: "invalid polar width", pipeline: query, device: device, queue: queue,
      payloadValues: payloadValues, offsetValues: offsets, tagValues: tagValues) { _, tags in tags[0] |= 64 }
    try malformedQuery(name: "invalid polar offset", pipeline: query, device: device, queue: queue,
      payloadValues: payloadValues, offsetValues: offsets, tagValues: tagValues) { offsets, _ in
        offsets[1] = UInt32(payloadValues.count + 1)
      }
    print("PASS polar roots, physical and UInt32-limit pack round trips, signed queries, and malformed metadata")
  }
}
