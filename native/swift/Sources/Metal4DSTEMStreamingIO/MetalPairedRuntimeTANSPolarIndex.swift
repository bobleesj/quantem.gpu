import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels

/// Exact packed polar-field accelerator for paired-runtime detector deltas.
final class MetalPairedRuntimeTANSPolarIndex {
  private static let buildLock = NSLock()
  private static let scansPerPacket = 512

  private let device: MTLDevice
  private let packets: Int
  let leafPixels: Int
  let layoutKind: String
  private let leaves: Int
  private let fields: Int
  private let packedPayload: MTLBuffer
  private let packedOffsets: MTLBuffer
  private let packedTags: MTLBuffer
  private let queryPipeline: MTLComputePipelineState
  private let queryScan512Pipelines: [QueryVariant: MTLComputePipelineState]

  let residentBytes: UInt64
  let buildMilliseconds: Double
  var scan512QueryPipelinePrepared: Bool { queryScan512Pipelines[.scan512] != nil }
  func queryPipelinePrepared(for variant: QueryVariant) -> Bool {
    queryScan512Pipelines[variant] != nil
  }

  enum QueryVariant: String, Equatable, Hashable {
    case packetGroups = "packet-groups"
    case scan512 = "scan512"
    case scan512Stripe2 = "scan512-stripe2"
    case scan512Stripe4 = "scan512-stripe4"
    case scan512Stripe8 = "scan512-stripe8"
    case scan512Field4 = "scan512-field4"
    case scan512ContiguousQuad = "scan512-contiguous-quad"
    case packetMajor = "packet-major"

    var requiresPreparedQueryPipeline: Bool {
      scan512StripeCount != nil || self == .scan512Field4 || self == .packetMajor
    }

    var threadgroupsPerPacket: Int {
      switch self {
      case .packetGroups: 4
      case .scan512Field4: 16
      default: 1
      }
    }

    var additionalThreadgroupBytes: Int {
      self == .scan512Field4 ? 128 * MemoryLayout<UInt32>.size : 0
    }

    var scan512StripeCount: UInt32? {
      switch self {
      case .packetGroups: nil
      case .scan512: 1
      case .scan512Stripe2: 2
      case .scan512Stripe4: 4
      case .scan512Stripe8: 8
      case .scan512Field4: nil
      case .scan512ContiguousQuad: 1
      case .packetMajor: nil
      }
    }

    var usesContiguousQuad: Bool { self == .scan512ContiguousQuad }
  }

  struct PreparedInputs {
    struct Batch {
      let selected: MTLBuffer
      let coefficients: MTLBuffer
      let count: Int
    }
    let batches: [Batch]
    var byteCount: Int {
      batches.reduce(0) { $0 + $1.selected.length + $1.coefficients.length }
    }
  }

  init(
    device: MTLDevice, library: MTLLibrary, queue: MTLCommandQueue,
    payload: MTLBuffer, offsets: MTLBuffer, modes: MTLBuffer, decoding: MTLBuffer,
    validPixels: [UInt8], packets: Int = 512, leafPixels: Int = 64,
    layoutKind: String = "polar", streamRankOfPixel: [UInt32]? = nil,
    compactOffsetsEnabled: Bool = false,
    prepareScan512QueryPipeline: Bool = false,
    allocationLimit: UInt64? = nil,
    shouldCancel: () -> Bool = { false }
  ) throws {
    let started = CFAbsoluteTimeGetCurrent()
    let workingSetLimit = min(allocationLimit ?? UInt64.max, device.recommendedMaxWorkingSetSize)
    Self.buildLock.lock()
    defer { Self.buildLock.unlock() }
    guard packets > 0, [16, 32, 64].contains(leafPixels), validPixels.count == 192 * 192,
      validPixels.allSatisfy({ $0 == 0 || $0 == 1 }),
      payload.device.registryID == device.registryID,
      offsets.device.registryID == device.registryID,
      modes.device.registryID == device.registryID,
      decoding.device.registryID == device.registryID
    else { throw Self.invalid("Polar-index inputs do not match the paired-runtime ABI") }
    self.device = device
    self.packets = packets
    self.leafPixels = leafPixels
    self.layoutKind = layoutKind
    guard
      let layoutForSize = PairedRuntimeTANSPolarPlan.indexLayout(
        leafPixels: leafPixels, layoutKind: layoutKind)
    else {
      throw Self.invalid("Polar-index leaf width is unsupported")
    }
    let leaves = layoutForSize.leaves
    let fields = leaves + leaves / 16
    self.leaves = leaves
    self.fields = fields
    let retainedCap =
      (leafPixels == 64
        ? 256
        : leafPixels == 32
          ? 384
          : PairedRuntimeTANSPolarPlan.isPaddedLayout(layoutKind) ? 768 : 512) * 1024 * 1024

    let partialConstants = MTLFunctionConstantValues()
    var leafWidth = UInt32(leafPixels)
    var fieldStride = UInt32(fields)
    var compactOffsets = compactOffsetsEnabled
    partialConstants.setConstantValue(&leafWidth, type: .uint, index: 5)
    partialConstants.setConstantValue(&fieldStride, type: .uint, index: 6)
    var packetBatches = true
    partialConstants.setConstantValue(&packetBatches, type: .bool, index: 51)
    partialConstants.setConstantValue(
      &compactOffsets, type: .bool,
      index: Metal4DSTEMKernels.pairedRuntimeTANSCompactOffsetsFunctionConstantIndex)
    let partialFunction = try library.makeFunction(
      name: "paired_runtime_tans_detector_partials", constantValues: partialConstants)
    let partialsPipeline = try device.makeComputePipelineState(function: partialFunction)
    let rootsPipeline = try Self.pipeline(
      library: library, device: device, name: "paired_runtime_tans_polar_roots_in_place")
    let sizesPipeline = try Self.pipeline(
      library: library, device: device, name: "paired_runtime_tans_polar_sizes")
    let packPipeline = try Self.pipeline(
      library: library, device: device, name: "paired_runtime_tans_polar_pack")
    queryPipeline = try Self.pipeline(
      library: library, device: device, name: "paired_runtime_tans_polar_query")
    var scan512Pipelines: [QueryVariant: MTLComputePipelineState] = [:]
    if prepareScan512QueryPipeline {
      for variant in [
        QueryVariant.scan512, .scan512Stripe2, .scan512Stripe4, .scan512Stripe8,
        .scan512Field4, .scan512ContiguousQuad, .packetMajor,
      ] {
        if variant == .packetMajor {
          guard
            let function = library.makeFunction(
              name: Metal4DSTEMKernels.pairedRuntimeTANSPolarQueryPacketMajorFunction)
          else {
            throw Self.invalid("Metal could not load the packet-major polar-query kernel")
          }
          scan512Pipelines[variant] = try device.makeComputePipelineState(function: function)
        } else if variant == .scan512Field4 {
          guard
            let function = library.makeFunction(
              name: Metal4DSTEMKernels.pairedRuntimeTANSPolarQueryField4Function)
          else {
            throw Self.invalid("Metal could not load the field-parallel polar-query kernel")
          }
          scan512Pipelines[variant] = try device.makeComputePipelineState(function: function)
        } else {
          guard let stripeCount = variant.scan512StripeCount else { continue }
          let constants = MTLFunctionConstantValues()
          var configuredStripeCount = stripeCount
          var configuredContiguousQuad = variant.usesContiguousQuad
          constants.setConstantValue(
            &configuredStripeCount, type: .uint,
            index: Metal4DSTEMKernels.pairedRuntimeTANSPolarScan512StripesFunctionConstantIndex)
          constants.setConstantValue(
            &configuredContiguousQuad, type: .bool,
            index: Metal4DSTEMKernels
              .pairedRuntimeTANSPolarScan512ContiguousQuadFunctionConstantIndex)
          let function = try library.makeFunction(
            name: Metal4DSTEMKernels.pairedRuntimeTANSPolarQueryScan512Function,
            constantValues: constants)
          scan512Pipelines[variant] = try device.makeComputePipelineState(function: function)
        }
      }
    }
    queryScan512Pipelines = scan512Pipelines

    // Fields are built, sized and packed one packet batch at a time, so the transient
    // field buffer holds `batchPackets` packets instead of the whole source.
    let batchSetting =
      Int(
        pairedRuntimeEnvironment("QGPU_PAIRED_RUNTIME_POLAR_BUILD_BATCH_PACKETS") ?? "64") ?? 64
    let batchPackets = max(1, min(packets, batchSetting))
    let fieldBytes = try Self.byteProduct([batchPackets, fields, Self.scansPerPacket, 4])
    let streamCount = try Self.byteProduct([packets, fields])
    let streamMetadataBytes = try Self.byteProduct([streamCount, 5])
    let selectionBytes = try Self.byteProduct([validPixels.count, 8])
    let metadataBytes = streamMetadataBytes + selectionBytes + 4
    let transientBytes = UInt64(fieldBytes) + UInt64(metadataBytes)
    let allocated = UInt64(device.currentAllocatedSize)
    guard allocated <= workingSetLimit,
      transientBytes <= workingSetLimit - allocated
    else {
      throw Self.invalid(
        "Polar-index construction needs \(transientBytes) additional bytes; release residents")
    }
    var fieldBuffer: MTLBuffer! = try Self.buffer(
      device: device, bytes: fieldBytes, options: .storageModePrivate,
      label: "paired-runtime polar fields")
    let sizes = try Self.buffer(
      device: device, bytes: streamCount * 4, options: .storageModeShared,
      label: "paired-runtime polar sizes")
    let tags = try Self.buffer(
      device: device, bytes: streamCount, options: .storageModePrivate,
      label: "paired-runtime polar tags")
    let failure = try Self.buffer(
      device: device, bytes: 4, options: .storageModeShared,
      label: "paired-runtime polar failure")

    guard
      let layout = PairedRuntimeTANSPolarPlan.indexLayout(
        leafPixels: leafPixels, layoutKind: layoutKind)
    else {
      throw Self.invalid("Polar-index leaf width is unsupported")
    }
    let permutation = layout.permutation
    guard permutation.count == leaves * leafPixels,
      permutation.count == validPixels.count
        || PairedRuntimeTANSPolarPlan.isPaddedLayout(layoutKind),
      Set(permutation.filter { $0 >= 0 }).count == validPixels.count
    else {
      throw Self.invalid("Polar-index permutation is incomplete")
    }
    // Stream addresses follow the resident's stream order; coefficients stay per pixel.
    // Empty slots of padded leaves read pixel 0 with coefficient 0.
    let selected = permutation.map { pixel in
      let real = UInt32(max(pixel, 0))
      return streamRankOfPixel.map { $0[Int(real)] } ?? real
    }
    let coefficients = permutation.map { $0 >= 0 ? Int32(validPixels[Int($0)]) : 0 }
    let selectedBuffer = try Self.upload(
      selected, device: device, label: "paired-runtime polar permutation")
    let coefficientBuffer = try Self.upload(
      coefficients, device: device, label: "paired-runtime polar validity")

    var prefix = [UInt32](repeating: 0, count: streamCount + 1)
    var words = UInt64(0)
    var batchPayloads: [(buffer: MTLBuffer, baseWord: UInt64, words: UInt64)] = []
    for firstPacket in stride(from: 0, to: packets, by: batchPackets) {
      if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
      let batchCount = min(batchPackets, packets - firstPacket)
      let firstStream = firstPacket * fields
      let batchStreams = batchCount * fields
      memset(failure.contents(), 0, 4)
      guard let fieldCommand = queue.makeCommandBuffer(),
        let clear = fieldCommand.makeBlitCommandEncoder()
      else { throw Self.invalid("Metal could not begin polar-field construction") }
      clear.fill(
        buffer: fieldBuffer, range: 0..<(batchStreams * Self.scansPerPacket * 4), value: 0)
      clear.endEncoding()
      guard let fieldEncoder = fieldCommand.makeComputeCommandEncoder() else {
        throw Self.invalid("Metal could not encode polar-field construction")
      }
      var partialParameters: [UInt32] = [
        UInt32(validPixels.count), UInt32(packets), UInt32(selected.count), UInt32(leaves),
        UInt32(payload.length), UInt32(firstPacket), UInt32(batchCount),
      ]
      fieldEncoder.setComputePipelineState(partialsPipeline)
      for (index, buffer) in [
        payload, offsets, modes, decoding, selectedBuffer, coefficientBuffer,
        fieldBuffer, failure,
      ].enumerated() {
        fieldEncoder.setBuffer(buffer, offset: 0, index: index)
      }
      fieldEncoder.setBytes(&partialParameters, length: partialParameters.count * 4, index: 8)
      fieldEncoder.dispatchThreadgroups(
        MTLSize(width: leaves, height: batchCount, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      var rootParameters: [UInt32] = [UInt32(batchCount), UInt32(leaves), UInt32(fields)]
      fieldEncoder.setComputePipelineState(rootsPipeline)
      fieldEncoder.setBuffer(fieldBuffer, offset: 0, index: 0)
      fieldEncoder.setBuffer(failure, offset: 0, index: 1)
      fieldEncoder.setBytes(&rootParameters, length: rootParameters.count * 4, index: 2)
      fieldEncoder.dispatchThreads(
        MTLSize(width: batchCount * (leaves / 16) * Self.scansPerPacket, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      var sizeParameters: [UInt32] = [UInt32(batchCount), UInt32(fields)]
      fieldEncoder.setComputePipelineState(sizesPipeline)
      fieldEncoder.setBuffer(fieldBuffer, offset: 0, index: 0)
      fieldEncoder.setBuffer(tags, offset: firstStream, index: 1)
      fieldEncoder.setBuffer(sizes, offset: firstStream * 4, index: 2)
      fieldEncoder.setBuffer(failure, offset: 0, index: 3)
      fieldEncoder.setBytes(&sizeParameters, length: sizeParameters.count * 4, index: 4)
      fieldEncoder.dispatchThreads(
        MTLSize(width: batchStreams, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      fieldEncoder.endEncoding()
      try Self.finish(fieldCommand, failure: failure, operation: "polar fields")

      let sizeValues = sizes.contents().assumingMemoryBound(to: UInt32.self)
      var local = [UInt32](repeating: 0, count: batchStreams + 1)
      var localWords = UInt64(0)
      for stream in 0..<batchStreams {
        localWords += UInt64(sizeValues[firstStream + stream])
        guard localWords <= UInt64(UInt32.max), words + localWords <= UInt64(UInt32.max) else {
          throw Self.invalid("Polar-index payload exceeds its UInt32 offset ABI")
        }
        local[stream + 1] = UInt32(localWords)
        prefix[firstStream + stream + 1] = UInt32(words + localWords)
      }
      // Check the retained index and device headroom before holding another batch.
      let retainedSoFar = (words + localWords) * 4 + UInt64((streamCount + 1) * 4 + streamCount)
      guard retainedSoFar <= UInt64(retainedCap) else {
        throw Self.invalid(
          "Polar index needs more than its \(retainedCap)-byte prototype cap")
      }
      let allocatedForBatch = UInt64(device.currentAllocatedSize)
      guard allocatedForBatch <= workingSetLimit,
        localWords * 4 + UInt64((batchStreams + 1) * 4) <= workingSetLimit - allocatedForBatch
      else {
        throw Self.invalid(
          "Polar-index packing needs \(localWords * 4) more bytes; release other residents")
      }
      let batchPayload = try Self.buffer(
        device: device, bytes: max(4, Int(localWords * 4)), options: .storageModePrivate,
        label: "paired-runtime polar batch payload")
      let localOffsets = try Self.upload(
        local, device: device, label: "paired-runtime polar batch offsets")
      memset(failure.contents(), 0, 4)
      guard let packCommand = queue.makeCommandBuffer(),
        let packEncoder = packCommand.makeComputeCommandEncoder()
      else { throw Self.invalid("Metal could not encode polar-field packing") }
      var packParameters: [UInt32] = [UInt32(batchCount), UInt32(fields), UInt32(localWords)]
      packEncoder.setComputePipelineState(packPipeline)
      packEncoder.setBuffer(fieldBuffer, offset: 0, index: 0)
      packEncoder.setBuffer(tags, offset: firstStream, index: 1)
      packEncoder.setBuffer(localOffsets, offset: 0, index: 2)
      packEncoder.setBuffer(batchPayload, offset: 0, index: 3)
      packEncoder.setBuffer(failure, offset: 0, index: 4)
      packEncoder.setBytes(&packParameters, length: packParameters.count * 4, index: 5)
      packEncoder.dispatchThreads(
        MTLSize(width: batchStreams, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      packEncoder.endEncoding()
      try Self.finish(packCommand, failure: failure, operation: "polar packing")
      batchPayloads.append((batchPayload, words, localWords))
      words += localWords
    }

    fieldBuffer = nil
    let retained = words * 4 + UInt64(prefix.count * 4 + streamCount)
    guard retained <= UInt64(retainedCap) else {
      throw Self.invalid(
        "Polar index needs \(retained) retained bytes, above its \(retainedCap)-byte prototype cap")
    }
    let allocatedForPacking = UInt64(device.currentAllocatedSize)
    guard allocatedForPacking <= workingSetLimit,
      retained <= workingSetLimit - allocatedForPacking
    else {
      throw Self.invalid(
        "Polar-index packing needs \(retained) more bytes; release other residents")
    }
    let packedPayload = try Self.buffer(
      device: device, bytes: max(4, Int(words * 4)), options: .storageModePrivate,
      label: "paired-runtime packed polar payload")
    guard let assemble = queue.makeCommandBuffer(), let blit = assemble.makeBlitCommandEncoder()
    else { throw Self.invalid("Metal could not assemble the polar payload") }
    for batch in batchPayloads where batch.words > 0 {
      blit.copy(
        from: batch.buffer, sourceOffset: 0, to: packedPayload,
        destinationOffset: Int(batch.baseWord * 4), size: Int(batch.words * 4))
    }
    blit.endEncoding()
    memset(failure.contents(), 0, 4)
    try Self.finish(assemble, failure: failure, operation: "polar payload assembly")
    batchPayloads.removeAll()
    let packedOffsets = try Self.privateUpload(
      prefix, device: device, queue: queue, label: "paired-runtime polar offsets")

    self.packedPayload = packedPayload
    self.packedOffsets = packedOffsets
    packedTags = tags
    residentBytes = UInt64(packedPayload.length + packedOffsets.length + tags.length)
    buildMilliseconds = (CFAbsoluteTimeGetCurrent() - started) * 1_000
  }

  /// Encode the indexed part of one exact plan without committing or synchronizing.
  func encode(
    plan: PairedRuntimeTANSPolarPlan, output: MTLBuffer, failure: MTLBuffer,
    command: MTLCommandBuffer,
    variant: QueryVariant = .packetGroups,
    profiler: PairedRuntimeDetectorProfiler.Session? = nil,
    packetStride: Int = 1, packetPhase: Int = 0
  ) throws {
    guard command.retainedReferences,
      output.device.registryID == device.registryID,
      failure.device.registryID == device.registryID
    else { throw Self.invalid("Polar query requires retaining commands and same-device buffers") }
    let inputs = try prepareInputs(plan: plan)
    try encode(
      inputs: inputs, output: output, failure: failure,
      command: command, variant: variant, profiler: profiler,
      packetStride: packetStride, packetPhase: packetPhase)
  }

  /// Upload immutable query batches once for repeated diagnostic submissions.
  func prepareInputs(plan: PairedRuntimeTANSPolarPlan) throws -> PreparedInputs {
    guard plan.usedIndex, plan.layoutKind == layoutKind,
      plan.leafPixelCount == leafPixels,
      plan.selectedFields.count == plan.fieldCoefficients.count,
      plan.selectedFields.allSatisfy({ Int($0) < fields })
    else { throw Self.invalid("Polar query requires one valid coefficient per selected field") }
    guard !plan.selectedFields.isEmpty else { return PreparedInputs(batches: []) }
    let fixedQueryBytes = 128 * MemoryLayout<UInt32>.size
    let availableFieldBytes = device.maxThreadgroupMemoryLength - fixedQueryBytes
    let batchCapacity = min(1_024, availableFieldBytes / 16)
    guard batchCapacity > 0 else {
      throw Self.invalid(
        "Polar queries need more than \(fixedQueryBytes) bytes of threadgroup memory")
    }
    var first = 0
    var batches: [PreparedInputs.Batch] = []
    while first < plan.selectedFields.count {
      let end = min(plan.selectedFields.count, first + batchCapacity)
      let batchFields = Array(plan.selectedFields[first..<end])
      let batchCoefficients = Array(plan.fieldCoefficients[first..<end])
      let selected = try Self.upload(
        batchFields, device: device, label: "paired-runtime selected polar fields")
      let coefficients = try Self.upload(
        batchCoefficients, device: device, label: "paired-runtime polar coefficients")
      batches.append(
        .init(
          selected: selected, coefficients: coefficients,
          count: batchFields.count))
      first = end
    }
    return PreparedInputs(batches: batches)
  }

  /// Encode prepared batches without allocating query input buffers.
  func encode(
    inputs: PreparedInputs, output: MTLBuffer, failure: MTLBuffer,
    command: MTLCommandBuffer,
    variant: QueryVariant = .packetGroups,
    profiler: PairedRuntimeDetectorProfiler.Session? = nil,
    packetStride: Int = 1, packetPhase: Int = 0
  ) throws {
    guard command.retainedReferences,
      output.device.registryID == device.registryID,
      failure.device.registryID == device.registryID
    else { throw Self.invalid("Polar query requires retaining commands and same-device buffers") }
    guard [1, 2, 4, 8].contains(packetStride), packetPhase >= 0, packetPhase < packetStride,
      packetStride == 1 || variant.scan512StripeCount != nil
    else {
      throw Self.invalid("Block-stride polar queries require a scan512 variant and a valid phase")
    }
    let logicalPackets = (packets - packetPhase + packetStride - 1) / packetStride
    let pipeline: MTLComputePipelineState
    if variant == .packetGroups {
      pipeline = queryPipeline
    } else if let prepared = queryScan512Pipelines[variant] {
      pipeline = prepared
    } else {
      throw Self.invalid(
        "Prepare the scan512 polar-query pipelines before selecting "
          + "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT=\(variant.rawValue)")
    }
    for batch in inputs.batches {
      guard
        let encoder = profiler?.makeComputeEncoder(
          commandBuffer: command, stage: "polar")
          ?? command.makeComputeCommandEncoder()
      else {
        throw Self.invalid("Metal could not encode the polar query")
      }
      var parameters: [UInt32] = [
        UInt32(packets), UInt32(fields), UInt32(batch.count),
        UInt32(packedPayload.length / 4), 0, UInt32(packetStride), UInt32(packetPhase),
      ]
      encoder.setComputePipelineState(pipeline)
      for (index, buffer) in [
        packedPayload, packedOffsets, packedTags, batch.selected, batch.coefficients, output,
        failure,
      ].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBytes(&parameters, length: parameters.count * 4, index: 7)
      if variant == .packetMajor {
        let width = min(pipeline.threadExecutionWidth, packets)
        encoder.dispatchThreads(
          MTLSize(width: packets, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        continue
      }
      encoder.setThreadgroupMemoryLength(
        batch.count * 16 + variant.additionalThreadgroupBytes, index: 0)
      encoder.dispatchThreadgroups(
        MTLSize(width: logicalPackets * variant.threadgroupsPerPacket, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.endEncoding()
    }
    // A retaining command buffer owns the bound buffers through completion.
  }

  private static func pipeline(
    library: MTLLibrary, device: MTLDevice, name: String
  ) throws -> MTLComputePipelineState {
    let constants = MTLFunctionConstantValues()
    let function = try library.makeFunction(name: name, constantValues: constants)
    return try device.makeComputePipelineState(function: function)
  }

  private static func finish(
    _ command: MTLCommandBuffer, failure: MTLBuffer, operation: String
  ) throws {
    command.commit()
    command.waitUntilCompleted()
    let code = failure.contents().load(as: UInt32.self)
    guard command.status == .completed, command.error == nil, code == 0 else {
      throw invalid(
        "Paired-runtime \(operation) failed with code \(code): "
          + (command.error?.localizedDescription ?? "invalid field index"))
    }
  }

  private static func upload<T>(
    _ values: [T], device: MTLDevice, label: String
  ) throws -> MTLBuffer {
    let result = values.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
    }
    guard let result else { throw invalid("Metal could not allocate \(label)") }
    result.label = label
    return result
  }

  private static func privateUpload<T>(
    _ values: [T], device: MTLDevice, queue: MTLCommandQueue, label: String
  ) throws -> MTLBuffer {
    let staging = try upload(values, device: device, label: label + " staging")
    let result = try buffer(
      device: device, bytes: staging.length, options: .storageModePrivate, label: label)
    guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
    else { throw invalid("Metal could not upload \(label)") }
    blit.copy(
      from: staging, sourceOffset: 0, to: result, destinationOffset: 0,
      size: staging.length)
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed, command.error == nil else {
      throw invalid("Metal could not complete \(label) upload")
    }
    return result
  }

  private static func buffer(
    device: MTLDevice, bytes: Int, options: MTLResourceOptions, label: String
  ) throws -> MTLBuffer {
    guard bytes > 0, bytes <= device.maxBufferLength,
      let result = device.makeBuffer(length: bytes, options: options)
    else { throw invalid("Metal could not allocate \(label) (\(bytes) bytes)") }
    result.label = label
    return result
  }

  private static func byteProduct(_ factors: [Int]) throws -> Int {
    var result = 1
    for factor in factors {
      let next = result.multipliedReportingOverflow(by: factor)
      guard factor >= 0, !next.overflow else { throw invalid("Polar-index size overflow") }
      result = next.partialValue
    }
    return result
  }

  private static func invalid(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest(message)
  }
}
