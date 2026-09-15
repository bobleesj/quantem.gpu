import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels

/// Exact synthetic output used to promote the paired runtime tANS Metal ABI.
@_spi(PairedRuntimeTANSPrototype)
public struct MetalPairedRuntimeTANSSyntheticResult: Sendable {
  public let modes: [UInt8]
  public let offsets: [UInt32]
  public let payload: [UInt8]
  public let decodedStreams: [[UInt16]]
  public let encodeMilliseconds: Double
  public let compactMilliseconds: Double
  public let decodeMilliseconds: Double
}

/// Bounded Metal harness for the exact paired runtime tANS kernels.
///
/// The harness deliberately accepts synthetic streams rather than HDF5 data.
/// It proves the codec ABI before any resident-source integration is allowed.
/// The opt-in interleaved-state mode is a separate codec experiment and is not
/// a deployable resident format; resident consumers do not recognize its modes.
@_spi(PairedRuntimeTANSPrototype)
public final class MetalPairedRuntimeTANSSyntheticCodec {
  private static let interleavedStatesFunctionConstantIndex = 24

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let encodePipeline: MTLComputePipelineState
  private let interleavedEncodePipeline: MTLComputePipelineState
  private let compactPipeline: MTLComputePipelineState
  private let decodePipeline: MTLComputePipelineState
  private let frequencyStarts: MTLBuffer
  private let encoding: MTLBuffer
  private let decoding: MTLBuffer

  public init(device: MTLDevice) throws {
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      throw Self.invalid("Metal could not create the paired tANS validation queue")
    }
    self.queue = queue
    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    let sparseSlackValue =
      ProcessInfo.processInfo.environment[
        "QGPU_PAIRED_RUNTIME_SPARSE_SLACK"] ?? "0"
    guard sparseSlackValue == "0" || sparseSlackValue == "4" || sparseSlackValue == "8",
      let sparseSlack = UInt32(sparseSlackValue)
    else { throw Self.invalid("QGPU_PAIRED_RUNTIME_SPARSE_SLACK must be 0, 4, or 8") }
    let encodeConstants = MTLFunctionConstantValues()
    var specializedSparseSlack = sparseSlack
    encodeConstants.setConstantValue(&specializedSparseSlack, type: .uint, index: 4)
    var interleavedStatesDisabled = false
    encodeConstants.setConstantValue(
      &interleavedStatesDisabled, type: .bool,
      index: Self.interleavedStatesFunctionConstantIndex)
    let encodeFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSEncodeFunction,
      constantValues: encodeConstants)
    encodePipeline = try device.makeComputePipelineState(function: encodeFunction)
    let interleavedEncodeConstants = MTLFunctionConstantValues()
    var interleavedSparseSlack = sparseSlack
    var interleavedStatesEnabled = true
    interleavedEncodeConstants.setConstantValue(&interleavedSparseSlack, type: .uint, index: 4)
    interleavedEncodeConstants.setConstantValue(
      &interleavedStatesEnabled, type: .bool,
      index: Self.interleavedStatesFunctionConstantIndex)
    let interleavedEncodeFunction = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSEncodeFunction,
      constantValues: interleavedEncodeConstants)
    interleavedEncodePipeline = try device.makeComputePipelineState(
      function: interleavedEncodeFunction)
    compactPipeline = try Self.pipeline(
      library: library, device: device, name: Metal4DSTEMKernels.pairedRuntimeTANSCompactFunction)
    decodePipeline = try Self.pipeline(
      library: library, device: device, name: Metal4DSTEMKernels.pairedRuntimeTANSDecodeFunction)

    let tables = try PairedRuntimeTANSTables.build()
    var packed = [UInt32](repeating: 0, count: tables.frequencies.count)
    for model in 0..<PairedRuntimeTANSTables.modelCount {
      var start = 0
      let base = model * PairedRuntimeTANSTables.symbolCount
      for symbol in 0..<PairedRuntimeTANSTables.symbolCount {
        let frequency = Int(tables.frequencies[base + symbol])
        packed[base + symbol] = UInt32(start) | (UInt32(frequency) << 16)
        start += frequency
      }
    }
    frequencyStarts = try Self.buffer(packed, device: device, label: "paired tANS frequencies")
    encoding = try Self.buffer(tables.encoding, device: device, label: "paired tANS encoding")
    decoding = try Self.buffer(
      tables.packedDecoding, device: device, label: "paired tANS decoding")
  }

  /// Encode, compact, and decode independent streams through Metal.
  ///
  /// `useInterleavedStates` opts into the experimental two-state entropy mode.
  /// `corruptInterleavedHeaderForTesting` flips its reserved header bit so the
  /// decoder's fail-closed validation can be exercised.
  public func roundTrip(
    streams: [[UInt16]], logicalDtype: Metal4DSTEMIntegerDType,
    useInterleavedStates: Bool = false,
    corruptInterleavedHeaderForTesting: Bool = false
  ) throws -> MetalPairedRuntimeTANSSyntheticResult {
    guard !streams.isEmpty, let count = streams.first?.count,
      (1...PairedRuntimeTANSTables.stateCount / 2).contains(count),
      streams.allSatisfy({ $0.count == count })
    else {
      throw Self.invalid(
        "Paired tANS validation needs equal nonempty streams of at most 512 counts")
    }
    guard logicalDtype == .uint8 || logicalDtype == .uint16 else {
      throw Self.invalid("Paired tANS validation supports exact uint8 or uint16 counts")
    }
    guard !corruptInterleavedHeaderForTesting || useInterleavedStates else {
      throw Self.invalid(
        "Interleaved header corruption requires the experimental two-state format")
    }
    if logicalDtype == .uint8, streams.joined().contains(where: { $0 > UInt8.max }) {
      throw Self.invalid("A uint8 paired tANS validation stream contains a count above 255")
    }
    let streamCount = streams.count
    let scratchStride = 2 * count
    let raw: MTLBuffer
    if logicalDtype == .uint8 {
      let values = (0..<count).flatMap { scan in streams.map { UInt8($0[scan]) } }
      raw = try Self.buffer(values, device: device, label: "paired tANS uint8 input")
    } else {
      let values = (0..<count).flatMap { scan in streams.map { $0[scan] } }
      raw = try Self.buffer(values, device: device, label: "paired tANS uint16 input")
    }
    let scratch = try Self.buffer(
      device: device, bytes: scratchStride * streamCount, options: .storageModePrivate,
      label: "paired tANS scratch")
    let sizes = try Self.buffer(
      device: device, bytes: streamCount * MemoryLayout<UInt32>.stride,
      label: "paired tANS sizes")
    let modes = try Self.buffer(
      device: device, bytes: streamCount, label: "paired tANS modes")
    let failure = try Self.buffer(device: device, bytes: 4, label: "paired tANS failure")
    memset(failure.contents(), 0, failure.length)
    var encodeParameters: [UInt32] = [
      UInt32(count), UInt32(streamCount), UInt32(streamCount),
      UInt32(logicalDtype.bytesPerValue), UInt32(scratchStride),
    ]
    let encodeCommand = try command(label: "paired tANS encode")
    guard let encode = encodeCommand.makeComputeCommandEncoder() else {
      throw Self.invalid("Metal could not encode paired tANS streams")
    }
    encode.setComputePipelineState(
      useInterleavedStates ? interleavedEncodePipeline : encodePipeline)
    for (index, buffer) in [
      raw, frequencyStarts, encoding, scratch, sizes, modes, failure,
    ]
    .enumerated() {
      encode.setBuffer(buffer, offset: 0, index: index)
    }
    encode.setBytes(&encodeParameters, length: encodeParameters.count * 4, index: 7)
    encode.dispatchThreads(
      MTLSize(width: streamCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: min(128, streamCount), height: 1, depth: 1))
    encode.endEncoding()
    try finish(encodeCommand, failure: failure, stage: "encode")

    let sizeValues = Self.values(UInt32.self, from: sizes, count: streamCount)
    var offsetValues = [UInt32](repeating: 0, count: streamCount + 1)
    for stream in 0..<streamCount {
      let next = offsetValues[stream].addingReportingOverflow(sizeValues[stream])
      guard !next.overflow else { throw Self.invalid("Paired tANS payload exceeds uint32 offsets") }
      offsetValues[stream + 1] = next.partialValue
    }
    let payloadBytes = Int(offsetValues.last!)
    let offsets = try Self.buffer(offsetValues, device: device, label: "paired tANS offsets")
    let payload = try Self.buffer(
      device: device, bytes: max(payloadBytes, 1), label: "paired tANS payload")
    var compactParameters: [UInt32] = [
      UInt32(streamCount), UInt32(scratchStride), UInt32(payloadBytes),
    ]
    let compactCommand = try command(label: "paired tANS compact")
    guard let compact = compactCommand.makeComputeCommandEncoder() else {
      throw Self.invalid("Metal could not compact paired tANS streams")
    }
    compact.setComputePipelineState(compactPipeline)
    for (index, buffer) in [scratch, sizes, offsets, payload, failure].enumerated() {
      compact.setBuffer(buffer, offset: 0, index: index)
    }
    compact.setBytes(&compactParameters, length: compactParameters.count * 4, index: 5)
    compact.dispatchThreads(
      MTLSize(width: streamCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: min(128, streamCount), height: 1, depth: 1))
    compact.endEncoding()
    try finish(compactCommand, failure: failure, stage: "compact")

    if corruptInterleavedHeaderForTesting {
      let modeValues = Self.values(UInt8.self, from: modes, count: streamCount)
      guard let candidateStream = modeValues.firstIndex(where: { (96..<128).contains(Int($0)) })
      else {
        throw Self.invalid("Interleaved header corruption needs an entropy-coded stream")
      }
      let headerByte = Int(offsetValues[candidateStream]) + 2
      payload.contents().assumingMemoryBound(to: UInt8.self)[headerByte] |= 0x80
    }

    let output = try Self.buffer(
      device: device, bytes: count * streamCount * MemoryLayout<UInt16>.stride,
      label: "paired tANS decoded counts")
    var decodeParameters: [UInt32] = [
      UInt32(count), UInt32(streamCount), UInt32(streamCount),
      UInt32(logicalDtype.bytesPerValue), UInt32(payloadBytes),
    ]
    let decodeCommand = try command(label: "paired tANS decode")
    guard let decode = decodeCommand.makeComputeCommandEncoder() else {
      throw Self.invalid("Metal could not decode paired tANS streams")
    }
    decode.setComputePipelineState(decodePipeline)
    for (index, buffer) in [payload, offsets, modes, decoding, output, failure].enumerated() {
      decode.setBuffer(buffer, offset: 0, index: index)
    }
    decode.setBytes(&decodeParameters, length: decodeParameters.count * 4, index: 6)
    decode.dispatchThreads(
      MTLSize(width: streamCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: min(128, streamCount), height: 1, depth: 1))
    decode.endEncoding()
    try finish(decodeCommand, failure: failure, stage: "decode")

    let decoded = Self.values(UInt16.self, from: output, count: count * streamCount)
    let decodedStreams = (0..<streamCount).map { stream in
      (0..<count).map { decoded[$0 * streamCount + stream] }
    }
    return MetalPairedRuntimeTANSSyntheticResult(
      modes: Self.values(UInt8.self, from: modes, count: streamCount),
      offsets: offsetValues,
      payload: Self.values(UInt8.self, from: payload, count: payloadBytes),
      decodedStreams: decodedStreams,
      encodeMilliseconds: Self.gpuMilliseconds(encodeCommand),
      compactMilliseconds: Self.gpuMilliseconds(compactCommand),
      decodeMilliseconds: Self.gpuMilliseconds(decodeCommand))
  }

  private func command(label: String) throws -> MTLCommandBuffer {
    guard let command = queue.makeCommandBuffer() else {
      throw Self.invalid("Metal could not create the \(label) command")
    }
    command.label = label
    return command
  }

  private func finish(
    _ command: MTLCommandBuffer, failure: MTLBuffer, stage: String
  ) throws {
    command.commit()
    command.waitUntilCompleted()
    let code = failure.contents().load(as: UInt32.self)
    guard command.status == .completed, code == 0 else {
      throw Self.invalid(
        "Paired tANS Metal \(stage) failed with code \(code): "
          + (command.error?.localizedDescription ?? "invalid stream"))
    }
  }

  private static func pipeline(
    library: MTLLibrary, device: MTLDevice, name: String
  ) throws -> MTLComputePipelineState {
    let function = try library.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
    return try device.makeComputePipelineState(function: function)
  }

  private static func buffer<T>(
    _ values: [T], device: MTLDevice, label: String
  ) throws -> MTLBuffer {
    let result = values.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
    }
    guard let result else { throw invalid("Metal could not allocate \(label)") }
    result.label = label
    return result
  }

  private static func buffer(
    device: MTLDevice, bytes: Int, options: MTLResourceOptions = .storageModeShared,
    label: String
  ) throws -> MTLBuffer {
    guard let result = device.makeBuffer(length: bytes, options: options) else {
      throw invalid("Metal could not allocate \(label)")
    }
    result.label = label
    return result
  }

  private static func values<T>(_ type: T.Type, from buffer: MTLBuffer, count: Int) -> [T] {
    guard count > 0 else { return [] }
    return Array(
      UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: T.self), count: count))
  }

  private static func gpuMilliseconds(_ command: MTLCommandBuffer) -> Double {
    max(0, command.gpuEndTime - command.gpuStartTime) * 1_000
  }

  private static func invalid(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest(message)
  }
}
