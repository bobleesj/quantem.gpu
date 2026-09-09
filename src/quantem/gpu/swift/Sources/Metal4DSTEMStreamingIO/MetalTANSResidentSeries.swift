import CryptoKit
import Foundation
import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMKernels

/// Unqualified archive-compatibility prototype, intentionally package-internal.
///
/// Selected acquisitions retain exact entropy streams, not a dense 4D tensor.
/// Authentication is sealed-record integrity, not independent scientific parity.
/// Do not expose through a consumer receipt until full validation gates pass.
final class MetalTANSResidentSeries {
  let acquisitionIndices: [Int]
  let shape: [Int]
  private(set) var readAndAuthenticationSeconds = 0.0
  private(set) var privateUploadSeconds = 0.0
  private(set) var loadSeconds = 0.0
  private(set) var readMetrics = TANSArchive.ReadMetrics()
  let transferConcurrency: Int
  let sourceReadPolicy: TANSArchive.SourceReadPolicy
  private(set) var stagingBytes = 0
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let pipeline: MTLComputePipelineState
  private let word32Pipeline: MTLComputePipelineState
  var useWord32Query = false
  var queryThreadgroupWidth = 128
  private let auditPipeline: MTLComputePipelineState
  private let displayPipeline: MTLComputePipelineState
  private let detectorPipeline: MTLComputePipelineState
  private let detectorFinishPipeline: MTLComputePipelineState
  private let detectorBatchFunction: MTLFunction
  private let detectorBatchPipeline: MTLComputePipelineState
  private var interleavedDetectorPipelines: [Int: MTLComputePipelineState] = [:]
  private let tansLibrary: MTLLibrary
  private var sharedModelDetectorPipelines: [Int: MTLComputePipelineState] = [:]
  // Explicit experiment only. Zero retains the frozen ordinary kernel.
  var experimentalDetectorStreamsPerLane = 0
  var experimentalUseTileIndex = false
  private var exactTileIndex: TANSExactTileIndex?
  private let detectorColumnCost: [Double]?
  var experimentalTileIndexBytes: Int { exactTileIndex?.residentBytes ?? 0 }
  private(set) var lastDetectorTileFields = 0
  private let detectorPartialBatchPipeline: MTLComputePipelineState
  private let detectorFinishBatchPipeline: MTLComputePipelineState
  private let detectorSparsePipeline: MTLComputePipelineState
  private let cacheMapValues: [Int32]
  let validDetectorMask: [UInt8]
  var useBatchedDetector = true
  // Benchmark-only alternative. Exact, but measured slower for live deltas
  // because its partial write/read traffic outweighs avoided atomics.
  var usePartialDetector = false
  private(set) var lastDetectorGPUSeconds = 0.0
  private(set) var lastDetectorScratchBytes = 0
  private(set) var lastDetectorDecodedColumns = 0
  private(set) var lastDetectorUsedPrevious = false
  private var previousDetectorMask: [UInt8]?
  private var previousDetectorImages: [MTLBuffer] = []
  private var previousDetectorAcquisitionIndices: [Int] = []
  private let chunks: [TANSArchive.Chunk]
  private var records: [MTLBuffer] = []
  private var globals: [MTLBuffer] = []
  private var output: MTLBuffer?
  private let retainedColumns: UInt32
  private let sparseColumns: UInt32
  private(set) var isReleased = false
  private(set) var lastQueryGPUSeconds = 0.0

  var residentBytes: Int {
    records.reduce(0) { $0 + $1.length } + globals.reduce(0) { $0 + $1.length }
      + (output?.length ?? 0) + previousDetectorImages.reduce(0) { $0 + $1.length }
      + experimentalTileIndexBytes
  }

  init(
    directory: URL, acquisitions: [Int], device: MTLDevice, maximumAdditionalBytes: UInt64,
    transferConcurrency: Int = 4,
    sourceReadPolicy: TANSArchive.SourceReadPolicy = .avoidCaching
  )
    throws
  {
    guard (1...4).contains(transferConcurrency) else {
      throw TANSArchive.invalid("Use one through four bounded encoded-record transfer workers")
    }
    self.transferConcurrency = transferConcurrency
    self.sourceReadPolicy = sourceReadPolicy
    let start = ProcessInfo.processInfo.systemUptime
    let archive = try TANSArchive(directory: directory, acquisitions: acquisitions)
    self.device = device
    acquisitionIndices = acquisitions
    shape = [acquisitions.count, 512, 512, 192, 192]
    chunks = archive.chunks
    guard let queue = device.makeCommandQueue() else {
      throw TANSArchive.invalid("Cannot create a Metal command queue")
    }
    self.queue = queue
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    tansLibrary = library
    guard let function = library.makeFunction(name: "tans_diffraction") else {
      throw TANSArchive.invalid("Missing tANS kernel")
    }
    pipeline = try device.makeComputePipelineState(function: function)
    guard let word32 = library.makeFunction(name: "tans_diffraction_word32") else {
      throw TANSArchive.invalid("Missing exact word32 tANS query")
    }
    word32Pipeline = try device.makeComputePipelineState(function: word32)
    guard let auditFunction = library.makeFunction(name: "tans_audit_packet") else {
      throw TANSArchive.invalid("Missing bounded tANS audit kernel")
    }
    auditPipeline = try device.makeComputePipelineState(function: auditFunction)
    guard let displayFunction = library.makeFunction(name: "tans_display_counts") else {
      throw TANSArchive.invalid("Missing exact diffraction display kernel")
    }
    displayPipeline = try device.makeComputePipelineState(function: displayFunction)
    guard let detectorFunction = library.makeFunction(name: "tans_detector_partials"),
      let finishFunction = library.makeFunction(name: "tans_detector_finish")
    else {
      throw TANSArchive.invalid("Missing exact tANS detector kernels")
    }
    detectorPipeline = try device.makeComputePipelineState(function: detectorFunction)
    detectorFinishPipeline = try device.makeComputePipelineState(function: finishFunction)
    guard let batchFunction = library.makeFunction(name: "tans_detector_batch") else {
      throw TANSArchive.invalid("Missing batched exact detector kernel")
    }
    detectorBatchFunction = batchFunction
    detectorBatchPipeline = try device.makeComputePipelineState(function: batchFunction)
    guard let partialBatchFunction = library.makeFunction(name: "tans_detector_partial_batch"),
      let finishBatchFunction = library.makeFunction(name: "tans_detector_finish_batch")
    else {
      throw TANSArchive.invalid("Missing non-atomic batched detector kernels")
    }
    detectorPartialBatchPipeline = try device.makeComputePipelineState(
      function: partialBatchFunction)
    detectorFinishBatchPipeline = try device.makeComputePipelineState(function: finishBatchFunction)
    guard let sparseFunction = library.makeFunction(name: "tans_detector_sparse_batch") else {
      throw TANSArchive.invalid("Missing exact sparse detector kernel")
    }
    detectorSparsePipeline = try device.makeComputePipelineState(function: sparseFunction)
    let mapData = archive.arrays["planner__cache_map"]!
    validDetectorMask = Array(archive.arrays["planner__valid"]!)
    detectorColumnCost = archive.arrays["planner__column_cost"]?.withUnsafeBytes { raw in
      (0..<36864).map { raw.loadUnaligned(fromByteOffset: $0 * 8, as: Double.self) }
    }
    let cacheMap: [Int32] = mapData.withUnsafeBytes { bytes in
      (0..<36864).map {
        Int32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self))
      }
    }
    let selectedSparse = cacheMap.filter { $0 >= 0 }.sorted()
    cacheMapValues = cacheMap
    guard cacheMap.allSatisfy({ $0 >= -1 }),
      selectedSparse == (0..<selectedSparse.count).map(Int32.init)
    else {
      throw TANSArchive.invalid("Sparse detector mapping is not a permutation")
    }
    sparseColumns = UInt32(selectedSparse.count)
    retainedColumns = UInt32(36864 - selectedSparse.count)
    var rank = UInt32(0)
    let ranks: [UInt32] = cacheMap.map { value in
      if value >= 0 { return UInt32.max }
      defer { rank += 1 }
      return rank
    }
    let modelCount = archive.arrays["codec__decoding"]!.count / 4096
    try archive.arrays["codec__decoding"]!.withUnsafeBytes { bytes in
      for offset in stride(from: 0, to: bytes.count, by: 4) {
        let code = UInt32(
          littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        let bits = (code >> 12) & 15
        guard bits <= 10, (code >> 16) + ((1 << bits) - 1) < 1024 else {
          throw TANSArchive.invalid("tANS transition leaves its1024-state model")
        }
      }
    }
    guard archive.arrays["model_ids"]!.allSatisfy({ $0 == 255 || Int($0) < modelCount }) else {
      throw TANSArchive.invalid("Detector references an absent tANS model")
    }
    let lengths = chunks.map(\.recordBytes)
    let globalBytes = archive.arrays.values.reduce(0) { $0 + $1.count } + ranks.count * 4
    let outputBytes = acquisitions.count * 36864 * 2
    let stageBytes = lengths.max() ?? 0
    stagingBytes = transferConcurrency * stageBytes
    // One reusable encoded-record stage per worker; read/authenticate in place.
    // No intermediate Data copy or decoded 4D storage is allocated.
    var required = UInt64(
      globalBytes + outputBytes + transferConcurrency * stageBytes)
    for length in lengths {
      let next = required.addingReportingOverflow(UInt64(length))
      guard !next.overflow, length <= device.maxBufferLength else {
        throw TANSArchive.invalid("Encoded record exceeds Metal limits")
      }
      required = next.partialValue
    }
    guard required <= maximumAdditionalBytes else {
      throw TANSArchive.invalid(
        "Encoded series needs \(required) additional bytes, budget is \(maximumAdditionalBytes); no fallback or binning"
      )
    }
    func shared(_ data: Data, label: String) throws -> MTLBuffer {
      let buffer = data.withUnsafeBytes { bytes in
        device.makeBuffer(
          bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
      }
      guard let buffer else { throw TANSArchive.invalid("Cannot allocate \(label)") }
      buffer.label = label
      return buffer
    }
    globals = try ["codec__decoding", "model_ids", "planner__cache_map"].map {
      try shared(archive.arrays[$0]!, label: $0)
    }
    globals.append(
      try ranks.withUnsafeBytes { try shared(Data($0), label: "retained detector ranks") })
    guard let output = device.makeBuffer(length: outputBytes, options: .storageModeShared)
    else {
      throw TANSArchive.invalid("Cannot allocate bounded transfer and diffraction buffers")
    }
    self.output = output
    output.label = "Exact tANS selected diffraction series"
    let window = try TANSUploadWindow(
      archive: archive, device: device, queue: queue, stageBytes: stageBytes,
      concurrency: transferConcurrency, readPolicy: sourceReadPolicy)
    var readSeconds = 0.0
    var uploadSeconds = 0.0
    for first in stride(from: 0, to: chunks.count, by: transferConcurrency) {
      let end = min(chunks.count, first + transferConcurrency)
      // Join every worker/GPU command before failure; retain source record order.
      for loaded in try window.load(Array(chunks[first..<end])) {
        readSeconds += loaded.readSeconds
        uploadSeconds += loaded.uploadSeconds
        readMetrics.ioSeconds += loaded.metrics.ioSeconds
        readMetrics.hashSeconds += loaded.metrics.hashSeconds
        readMetrics.validationSeconds += loaded.metrics.validationSeconds
        readMetrics.sourceBytesRead += loaded.metrics.sourceBytesRead
        readMetrics.authenticatedRecords += loaded.metrics.authenticatedRecords
        records.append(loaded.resident)
      }
    }
    readAndAuthenticationSeconds = readSeconds
    privateUploadSeconds = uploadSeconds
    loadSeconds = ProcessInfo.processInfo.systemUptime - start
  }

  /// Full uint16 detector counts at one native scan location, in selected acquisition order.
  /// The returned array is an independent small 2D-per-acquisition snapshot.
  private func diffractionBuffer(scanRow: Int, scanColumn: Int) throws -> MTLBuffer {
    guard !isReleased, let output else {
      throw TANSArchive.invalid("tANS series has been released")
    }
    guard (0..<512).contains(scanRow), (0..<512).contains(scanColumn) else {
      throw TANSArchive.invalid("Scan row and column must be in 0..<512")
    }
    let position = scanRow * 512 + scanColumn
    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
    else {
      throw TANSArchive.invalid("Cannot encode exact tANS diffraction")
    }
    encoder.setComputePipelineState(useWord32Query ? word32Pipeline : pipeline)
    for (index, acquisition) in acquisitionIndices.enumerated() {
      let localChunk = position / 16384
      let recordIndex = index * 16 + localChunk
      let chunk = chunks[recordIndex]
      for (binding, name) in ["dense", "dense_offsets", "sparse", "sparse_offsets"].enumerated() {
        guard let component = chunk.components.first(where: { $0.name == name }),
          component.dtype == "<u4"
        else {
          encoder.endEncoding()
          throw TANSArchive.invalid("Missing authenticated entropy component \(name)")
        }
        encoder.setBuffer(records[recordIndex], offset: component.offset, index: binding)
      }
      for (index, buffer) in globals.enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index + 4)
      }
      encoder.setBuffer(output, offset: index * 36864 * 2, index: 8)
      var parameters: [UInt32] = [
        UInt32(position % 16384), retainedColumns, sparseColumns,
        UInt32((acquisition * 4 + localChunk / 4) * 36864),
      ]
      encoder.setBytes(&parameters, length: 16, index: 9)
      encoder.dispatchThreads(
        MTLSize(width: 36864, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: queryThreadgroupWidth, height: 1, depth: 1))
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("Exact tANS query failed: \(String(describing: command.error))")
    }
    lastQueryGPUSeconds = command.gpuEndTime - command.gpuStartTime
    return output
  }

  func extractDiffraction(scanRow: Int, scanColumn: Int) throws -> [UInt16] {
    let output = try diffractionBuffer(scanRow: scanRow, scanColumn: scanColumn)
    return Array(
      UnsafeBufferPointer(
        start: output.contents().bindMemory(
          to: UInt16.self, capacity: acquisitionIndices.count * 36864),
        count: acquisitionIndices.count * 36864))
  }

  /// Exact widening for existing UInt32 Metal image renderers. No image download.
  func diffractionImages(scanRow: Int, scanColumn: Int) throws -> [MTLBuffer] {
    let source = try diffractionBuffer(scanRow: scanRow, scanColumn: scanColumn)
    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
    else { throw TANSArchive.invalid("Cannot prepare exact diffraction display") }
    encoder.setComputePipelineState(displayPipeline)
    var images: [MTLBuffer] = []
    for index in acquisitionIndices.indices {
      guard let image = device.makeBuffer(length: 36864 * 4, options: .storageModeShared) else {
        encoder.endEncoding()
        throw TANSArchive.invalid("Cannot allocate a diffraction image")
      }
      images.append(image)
      encoder.setBuffer(source, offset: index * 36864 * 2, index: 0)
      encoder.setBuffer(image, offset: 0, index: 1)
      encoder.dispatchThreads(
        MTLSize(width: 36864, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("Diffraction display failed")
    }
    return images
  }

  /// Exact binary-mask sum for every scan in every retained acquisition.
  /// The caller supplies any validity mask explicitly. Raw source is unchanged.
  /// Scratch contains only 32-column reduction partials, not raw detector data.
  /// Returned count buffers are immutable. Reuse only a completed result from
  /// this source instance; failures never advance the retained mask or image.
  func detectorImages(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool = false
  ) throws -> [MTLBuffer] {
    try detectorImages(
      mask: mask, maximumAdditionalBytes: maximumAdditionalBytes,
      rebase: rebase, selectedAcquisitions: nil)
  }

  /// Exact detector images for either the full resident series or a selected
  /// acquisition. Every requested image is complete at return. Drain temporary
  /// Objective-C command objects per request so a sustained worker does not
  /// retain old GPU publications until its thread's outer autorelease pool.
  func detectorImages(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool = false, selectedAcquisitions: [Int]?
  ) throws -> [MTLBuffer] {
    try autoreleasepool {
      try detectorImagesNow(
        mask: mask, maximumAdditionalBytes: maximumAdditionalBytes,
        rebase: rebase, selectedAcquisitions: selectedAcquisitions)
    }
  }

  private func detectorImagesNow(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool, selectedAcquisitions: [Int]?
  ) throws -> [MTLBuffer] {
    guard !isReleased, mask.count == 36864, mask.allSatisfy({ $0 <= 1 }) else {
      throw TANSArchive.invalid("Provide a binary 192x192 detector mask on a live entropy series")
    }
    let outputAcquisitionIndices = selectedAcquisitions ?? acquisitionIndices
    guard !outputAcquisitionIndices.isEmpty,
      Set(outputAcquisitionIndices).count == outputAcquisitionIndices.count,
      outputAcquisitionIndices.allSatisfy({ acquisitionIndices.contains($0) })
    else {
      throw TANSArchive.invalid("Select unique retained detector acquisition indices")
    }
    let full = mask.indices.filter { mask[$0] != 0 }
    let difference =
      previousDetectorMask.map { old in mask.indices.filter { mask[$0] != old[$0] } } ?? []
    let sameAcquisitions = previousDetectorAcquisitionIndices == outputAcquisitionIndices
    let usePrevious =
      !rebase && sameAcquisitions && previousDetectorMask != nil && difference.count < full.count
    var residual = mask.indices.map {
      usePrevious
        ? Int32(mask[$0]) - Int32(previousDetectorMask![$0]) : Int32(mask[$0])
    }
    var selectedTiles: [(Int, Int32)] = []
    if experimentalUseTileIndex {
      guard useBatchedDetector, let index = exactTileIndex, let cost = detectorColumnCost else {
        throw TANSArchive.invalid("Prepare the exact tile index before enabling its batched query")
      }
      selectedTiles = index.plan(coefficients: &residual, valid: validDetectorMask, cost: cost)
    }
    lastDetectorTileFields = selectedTiles.count
    let unsorted = mask.indices.filter { residual[$0] != 0 }
    let dense = unsorted.filter { cacheMapValues[$0] < 0 }
    let sparse = unsorted.filter { cacheMapValues[$0] >= 0 }
    let selected = (useBatchedDetector ? dense + sparse : unsorted).map(UInt32.init)
    let signs: [Int32] = selected.map { residual[Int($0)] }
    let groups = (selected.count + 31) / 32
    // The batched kernel is also valid for a selected acquisition. Its record
    // table is narrowed to the requested acquisition's 16 entropy chunks, so
    // one dispatch can cover the complete selected image without changing the
    // exact integer/delta contract.
    let useBatchForRequest = useBatchedDetector
    // A full-series delta has one disjoint partial range per chunk/group. It
    // avoids atomics when that exact scratch fits the caller's budget; large
    // masks (notably full ADF) retain the bounded atomic fallback.
    let outputBytes = outputAcquisitionIndices.count * 512 * 512 * 4
    let selectionBytes = max(4, selected.count * 4)
    let indirectBytes =
      (useBatchForRequest ? chunks.count * 44 : 0)
      + (selectedTiles.isEmpty
        ? 0
        : (exactTileIndex!.fields.count * 64 + selectedTiles.count * 8 + outputAcquisitionIndices
          .count * 12))
    let overheadBytes = outputBytes + selectionBytes * 2 + indirectBytes
    let batchRecordCount = outputAcquisitionIndices.count * 16
    let partialGroups = (dense.count + 31) / 32
    let partialBytesPerRecord = partialGroups * 32 * 512 * 4
    let availableScratchBytes =
      maximumAdditionalBytes > UInt64(overheadBytes)
      ? maximumAdditionalBytes - UInt64(overheadBytes) : 0
    let partialRecordBudget =
      partialBytesPerRecord > 0
      ? Int(min(availableScratchBytes, UInt64(device.maxBufferLength))) / partialBytesPerRecord : 0
    let usePartialBatch =
      usePartialDetector && useBatchForRequest
      && outputAcquisitionIndices.count == acquisitionIndices.count
      && dense.count > 0 && partialRecordBudget > 0
    let partialBatchBytes =
      usePartialBatch
      ? min(batchRecordCount, partialRecordBudget) * partialBytesPerRecord : 0
    let scratchBytes =
      usePartialBatch
      ? partialBatchBytes
      : (useBatchForRequest ? 4 : max(4, groups * 32 * 512 * 4))
    guard
      UInt64(scratchBytes + outputBytes + selectionBytes * 2 + indirectBytes)
        <= maximumAdditionalBytes,
      scratchBytes <= device.maxBufferLength
    else {
      throw TANSArchive.invalid(
        "Insufficient budget for exact detector partials and 2D outputs; prior output preserved")
    }
    guard let scratch = device.makeBuffer(length: scratchBytes, options: .storageModePrivate),
      let selection = device.makeBuffer(length: selectionBytes, options: .storageModeShared),
      let coefficients = device.makeBuffer(length: selectionBytes, options: .storageModeShared)
    else {
      throw TANSArchive.invalid("Cannot allocate exact detector partials")
    }
    selected.withUnsafeBytes {
      if !$0.isEmpty { memcpy(selection.contents(), $0.baseAddress!, $0.count) }
    }
    signs.withUnsafeBytes {
      if !$0.isEmpty { memcpy(coefficients.contents(), $0.baseAddress!, $0.count) }
    }
    let images = try outputAcquisitionIndices.map { _ -> MTLBuffer in
      guard let image = device.makeBuffer(length: 512 * 512 * 4, options: .storageModeShared) else {
        throw TANSArchive.invalid("Cannot allocate complete exact virtual images")
      }
      return image
    }
    lastDetectorScratchBytes = scratchBytes + selectionBytes * 2 + indirectBytes
    lastDetectorDecodedColumns = selected.count
    lastDetectorUsedPrevious = usePrevious
    lastDetectorGPUSeconds = 0
    if useBatchForRequest {
      if outputAcquisitionIndices.count == acquisitionIndices.count {
        lastDetectorGPUSeconds = try detectorBatch(
          images: images, selection: selection,
          coefficients: coefficients, selectedCount: selected.count,
          denseCount: dense.count, seed: usePrevious, partials: usePartialBatch ? scratch : nil,
          outputAcquisitionIndices: outputAcquisitionIndices, selectedTiles: selectedTiles,
          metadataBudget: maximumAdditionalBytes - UInt64(scratchBytes + overheadBytes))
      } else {
        // Keep chunk scan offsets distinct for selected and multi-selection
        // callers as well; a fused sum across chunks is not a valid image.
        lastDetectorGPUSeconds = try detectorBatch(
          images: images, selection: selection,
          coefficients: coefficients, selectedCount: selected.count,
          denseCount: dense.count, seed: usePrevious, partials: nil,
          outputAcquisitionIndices: outputAcquisitionIndices, selectedTiles: selectedTiles,
          metadataBudget: maximumAdditionalBytes - UInt64(scratchBytes + overheadBytes))
      }
      try Task.checkCancellation()
      previousDetectorMask = mask
      previousDetectorImages = images
      previousDetectorAcquisitionIndices = outputAcquisitionIndices
      return images
    }
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw TANSArchive.invalid("Cannot encode exact detector query")
    }
    // One serial compute encoder replaces 2,112 encoders and 66 CPU/GPU waits.
    // Explicit barriers protect the bounded scratch between producer/consumer
    // dispatches; every final 2D output is separate from the prior publication.
    defer { if command.status == .notEnqueued { encoder.endEncoding() } }
    for (index, acquisition) in outputAcquisitionIndices.enumerated() {
      try Task.checkCancellation()
      guard let retainedIndex = acquisitionIndices.firstIndex(of: acquisition) else {
        throw TANSArchive.invalid("Selected detector acquisition is not retained")
      }
      for localChunk in 0..<16 {
        let recordIndex = retainedIndex * 16 + localChunk
        let chunk = chunks[recordIndex]
        if groups > 0 {
          encoder.setComputePipelineState(detectorPipeline)
          for (binding, name) in ["dense", "dense_offsets", "sparse", "sparse_offsets"].enumerated()
          {
            guard let component = chunk.components.first(where: { $0.name == name }) else {
              throw TANSArchive.invalid("Missing authenticated detector source")
            }
            encoder.setBuffer(records[recordIndex], offset: component.offset, index: binding)
          }
          for (binding, buffer) in globals.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: binding + 4)
          }
          encoder.setBuffer(scratch, offset: 0, index: 8)
          var parameters: [UInt32] = [
            retainedColumns, sparseColumns,
            UInt32((acquisition * 4 + localChunk / 4) * 36864), UInt32(selected.count),
            UInt32(groups),
          ]
          encoder.setBytes(&parameters, length: 20, index: 9)
          encoder.setBuffer(selection, offset: 0, index: 10)
          encoder.setBuffer(coefficients, offset: 0, index: 11)
          encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 32, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
          encoder.memoryBarrier(scope: .buffers)
        }
        encoder.setComputePipelineState(detectorFinishPipeline)
        encoder.setBuffer(scratch, offset: 0, index: 0)
        encoder.setBuffer(images[index], offset: localChunk * 16384 * 4, index: 1)
        var count = UInt32(groups)
        encoder.setBytes(&count, length: 4, index: 2)
        if usePrevious {
          guard let previousIndex = previousDetectorAcquisitionIndices.firstIndex(of: acquisition),
            previousIndex < previousDetectorImages.count
          else {
            throw TANSArchive.invalid("Previous detector image set does not match selection")
          }
          encoder.setBuffer(
            previousDetectorImages[previousIndex], offset: localChunk * 16384 * 4,
            index: 3)
        } else {
          encoder.setBuffer(images[index], offset: localChunk * 16384 * 4, index: 3)
        }
        var seed: UInt32 = usePrevious ? 1 : 0
        encoder.setBytes(&seed, length: 4, index: 4)
        encoder.dispatchThreads(
          MTLSize(width: 16384, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
      }
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("Exact detector query failed: \(String(describing:command.error))")
    }
    lastDetectorGPUSeconds = command.gpuEndTime - command.gpuStartTime
    try Task.checkCancellation()
    previousDetectorMask = mask
    previousDetectorImages = images
    previousDetectorAcquisitionIndices = outputAcquisitionIndices
    return images
  }

  private func detectorBatch(
    images: [MTLBuffer], selection: MTLBuffer, coefficients: MTLBuffer,
    selectedCount: Int, denseCount: Int, seed: Bool,
    partials: MTLBuffer?,
    outputAcquisitionIndices: [Int], selectedTiles: [(Int, Int32)], metadataBudget: UInt64
  ) throws -> Double {
    guard device.argumentBuffersSupport == .tier2 else {
      throw TANSArchive.invalid("Batched entropy query requires Metal argument-buffer tier 2")
    }
    guard
      [0, 1, 2, 4, 8, 32, 64, 65, 66, 67, 68, 69, 70].contains(experimentalDetectorStreamsPerLane)
    else {
      throw TANSArchive.invalid(
        "Experimental detector mode must be 0, 1, 2, 4, 8, 32 or a registered CUDA-followup mode 64...70"
      )
    }
    let streams = experimentalDetectorStreamsPerLane
    let useShared = streams >= 32
    let pairReduction = [64, 66, 67, 68, 69, 70].contains(streams)
    let sharedThreads = [67, 70].contains(streams) ? 512 : 128
    if useShared && sharedModelDetectorPipelines[streams] == nil {
      let constants = MTLFunctionConstantValues()
      var pair = pairReduction
      var word32 = [65, 66, 67].contains(streams)
      var threads = UInt32(sharedThreads)
      constants.setConstantValue(&pair, type: .bool, index: 2)
      constants.setConstantValue(&word32, type: .bool, index: 3)
      constants.setConstantValue(&threads, type: .uint, index: 4)
      var pairs: UInt32 = streams >= 69 ? 4 : 1
      constants.setConstantValue(&pairs, type: .uint, index: 6)
      let name =
        streams >= 68 ? "tans_detector_cuda_funnel_batch" : "tans_detector_shared_model_batch"
      let function = try tansLibrary.makeFunction(name: name, constantValues: constants)
      sharedModelDetectorPipelines[streams] = try device.makeComputePipelineState(
        function: function)
    }
    if streams > 0 && !useShared && interleavedDetectorPipelines[streams] == nil {
      let values = MTLFunctionConstantValues()
      var count = UInt32(streams == 8 ? 1 : streams)
      var coalesced = streams == 8
      values.setConstantValue(&count, type: .uint, index: 0)
      values.setConstantValue(&coalesced, type: .bool, index: 1)
      let function = try tansLibrary.makeFunction(
        name: "tans_detector_interleaved_batch", constantValues: values)
      interleavedDetectorPipelines[streams] = try device.makeComputePipelineState(
        function: function)
    }
    let arguments = detectorBatchFunction.makeArgumentEncoder(bufferIndex: 0)
    let recordIndices: [Int] = outputAcquisitionIndices.flatMap { acquisition in
      guard let retainedIndex = acquisitionIndices.firstIndex(of: acquisition) else {
        return [] as [Int]
      }
      return (0..<16).map { retainedIndex * 16 + $0 }
    }
    guard recordIndices.count == outputAcquisitionIndices.count * 16,
      arguments.encodedLength == 40,
      let table = device.makeBuffer(length: recordIndices.count * 40, options: .storageModeShared),
      let modelOffsets = device.makeBuffer(
        length: recordIndices.count * 4, options: .storageModeShared),
      let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
    else {
      throw TANSArchive.invalid("Cannot allocate exact detector dispatch table")
    }
    for (index, image) in images.enumerated() {
      if seed {
        blit.copy(
          from: previousDetectorImages[index], sourceOffset: 0,
          to: image, destinationOffset: 0, size: image.length)
      } else {
        blit.fill(buffer: image, range: 0..<image.length, value: 0)
      }
    }
    blit.endEncoding()
    if !selectedTiles.isEmpty {
      try exactTileIndex!.encodeAdd(
        command: command, images: images,
        acquisitions: outputAcquisitionIndices.map { acquisitionIndices.firstIndex(of: $0)! },
        selected: selectedTiles)
    }
    let offsets = modelOffsets.contents().assumingMemoryBound(to: UInt32.self)
    for (index, recordIndex) in recordIndices.enumerated() {
      let chunk = chunks[recordIndex]
      arguments.setArgumentBuffer(table, offset: index * 40)
      for (binding, name) in ["dense", "dense_offsets", "sparse", "sparse_offsets"].enumerated() {
        guard let component = chunk.components.first(where: { $0.name == name }) else {
          throw TANSArchive.invalid("Missing exact batched entropy component")
        }
        arguments.setBuffer(records[recordIndex], offset: component.offset, index: binding)
      }
      guard let imageIndex = outputAcquisitionIndices.firstIndex(of: chunk.acquisition) else {
        throw TANSArchive.invalid("Batched detector record is not in the requested output set")
      }
      arguments.setBuffer(images[imageIndex], offset: (index % 16) * 16384 * 4, index: 4)
      offsets[index] = UInt32((chunk.acquisition * 4 + (index % 16) / 4) * 36864)
    }
    var groupedBuffers: [MTLBuffer] = []
    var maximumModelGroups = 0
    if useShared && denseCount > 0 {
      // Group only immutable metadata on CPU. Scientific counts stay encoded
      // on GPU; no decoding, output computation, or representation duplication.
      let models = globals[1].contents().assumingMemoryBound(to: UInt8.self)
      let input = selection.contents().assumingMemoryBound(to: UInt32.self)
      let signs = coefficients.contents().assumingMemoryBound(to: Int32.self)
      var descriptors: [UInt32] = []
      var groupedSelection: [UInt32] = []
      var groupedSigns: [Int32] = []
      var starts: [UInt32] = []
      var counts: [UInt32] = []
      var contexts: [UInt32: (UInt32, UInt32)] = [:]
      for index in recordIndices.indices {
        let context = offsets[index]
        if contexts[context] == nil {
          var byModel: [[Int]] = Array(repeating: [], count: pairReduction ? 512 : 256)
          for i in 0..<denseCount {
            let model = Int(models[Int(context) + Int(input[i])])
            let key = pairReduction ? model * 2 + (signs[i] < 0 ? 1 : 0) : model
            byModel[key].append(i)
          }
          let start = UInt32(descriptors.count / 2)
          for key in byModel.indices {
            let model = pairReduction ? key / 2 : key
            let members = byModel[key]
            for begin in stride(from: 0, to: members.count, by: 32) {
              descriptors += [UInt32(model), UInt32(groupedSelection.count)]
              for lane in 0..<32 {
                if begin + lane < members.count {
                  let i = members[begin + lane]
                  groupedSelection.append(input[i])
                  groupedSigns.append(signs[i])
                } else {
                  groupedSelection.append(UInt32.max)
                  groupedSigns.append(0)
                }
              }
            }
          }
          let count = UInt32(descriptors.count / 2) - start
          contexts[context] = (start, count)
        }
        let range = contexts[context]!
        starts.append(range.0)
        counts.append(range.1)
        maximumModelGroups = max(maximumModelGroups, Int(range.1))
      }
      let bytes =
        (descriptors.count + groupedSelection.count + groupedSigns.count + starts.count
          + counts.count) * 4
      guard UInt64(bytes) <= metadataBudget else {
        throw TANSArchive.invalid("Exact model-group metadata exceeds caller budget; no fallback")
      }
      func upload<T>(_ values: [T]) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
          guard
            let buffer = device.makeBuffer(
              bytes: bytes.baseAddress!, length: bytes.count,
              options: .storageModeShared)
          else {
            throw TANSArchive.invalid("Cannot allocate exact model-group metadata")
          }
          return buffer
        }
      }
      groupedBuffers = try [
        upload(descriptors), upload(groupedSelection), upload(groupedSigns),
        upload(starts), upload(counts),
      ]
      lastDetectorScratchBytes += bytes
    }
    if selectedCount > 0 {
      let denseGroups = (denseCount + 31) / 32
      let sparseCount = selectedCount - denseCount
      if let partials, denseCount > 0 {
        // Apple GPU maxBufferLength is commonly much smaller than the full
        // 66-acquisition partial volume. Reuse one bounded scratch shard and
        // order producer/finish encoders per shard; no atomic accumulation is
        // needed, and every source record still participates exactly once.
        let bytesPerRecord = max(1, denseGroups * 32 * 512 * 4)
        let recordsPerShard = max(1, min(recordIndices.count, partials.length / bytesPerRecord))
        var shardStart = 0
        while shardStart < recordIndices.count {
          let shardCount = min(recordsPerShard, recordIndices.count - shardStart)
          guard let encoder = command.makeComputeCommandEncoder() else {
            throw TANSArchive.invalid("Cannot encode sharded detector partials")
          }
          encoder.useResources(
            recordIndices[shardStart..<(shardStart + shardCount)].map { records[$0] }, usage: .read)
          encoder.useResources(images, usage: [.read, .write])
          encoder.setComputePipelineState(detectorPartialBatchPipeline)
          encoder.setBuffer(table, offset: shardStart * 40, index: 0)
          for (binding, buffer) in globals.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: binding + 1)
          }
          encoder.setBuffer(selection, offset: 0, index: 5)
          encoder.setBuffer(coefficients, offset: 0, index: 6)
          var parameters: [UInt32] = [
            retainedColumns, sparseColumns, 0,
            UInt32(denseCount), UInt32(denseGroups),
          ]
          encoder.setBytes(&parameters, length: 20, index: 7)
          encoder.setBuffer(modelOffsets, offset: shardStart * 4, index: 8)
          encoder.setBuffer(partials, offset: 0, index: 9)
          encoder.dispatchThreadgroups(
            MTLSize(width: denseGroups, height: 32, depth: shardCount),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
          if sparseCount > 0 {
            encoder.setComputePipelineState(detectorSparsePipeline)
            encoder.setBuffer(table, offset: shardStart * 40, index: 0)
            encoder.setBuffer(globals[2], offset: 0, index: 1)
            encoder.setBuffer(selection, offset: denseCount * 4, index: 2)
            encoder.setBuffer(coefficients, offset: denseCount * 4, index: 3)
            var sparseParameters: [UInt32] = [UInt32(sparseCount), sparseColumns]
            encoder.setBytes(&sparseParameters, length: 8, index: 4)
            encoder.dispatchThreads(
              MTLSize(width: sparseCount * 32, height: shardCount, depth: 1),
              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          }
          encoder.endEncoding()
          guard let finish = command.makeComputeCommandEncoder() else {
            throw TANSArchive.invalid("Cannot encode sharded detector finish")
          }
          finish.useResources(images, usage: [.read, .write])
          finish.setComputePipelineState(detectorFinishBatchPipeline)
          finish.setBuffer(partials, offset: 0, index: 0)
          finish.setBuffer(table, offset: shardStart * 40, index: 1)
          var groupCount = UInt32(denseGroups)
          finish.setBytes(&groupCount, length: 4, index: 2)
          finish.dispatchThreads(
            MTLSize(width: 16384, height: 1, depth: shardCount),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
          finish.endEncoding()
          shardStart += shardCount
        }
      } else {
        guard let encoder = command.makeComputeCommandEncoder() else {
          throw TANSArchive.invalid("Cannot encode batched entropy detector")
        }
        encoder.useResources(recordIndices.map { records[$0] }, usage: .read)
        encoder.useResources(images, usage: [.read, .write])
        if denseCount > 0 {
          let selectedPipeline =
            useShared
            ? sharedModelDetectorPipelines[streams]!
            : (streams > 0 ? interleavedDetectorPipelines[streams]! : detectorBatchPipeline)
          let streamsPerLane = streams == 8 ? 1 : max(1, streams)
          let dispatchGroups = (denseCount + 32 * streamsPerLane - 1) / (32 * streamsPerLane)
          encoder.setComputePipelineState(selectedPipeline)
          encoder.setBuffer(table, offset: 0, index: 0)
          for (binding, buffer) in globals.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: binding + 1)
          }
          encoder.setBuffer(selection, offset: 0, index: 5)
          encoder.setBuffer(coefficients, offset: 0, index: 6)
          var parameters: [UInt32] = [
            retainedColumns, sparseColumns, 0, UInt32(denseCount), UInt32(denseGroups),
          ]
          encoder.setBytes(&parameters, length: 20, index: 7)
          encoder.setBuffer(modelOffsets, offset: 0, index: 8)
          if useShared {
            for (index, buffer) in groupedBuffers.enumerated() {
              encoder.setBuffer(buffer, offset: 0, index: index + 9)
            }
            encoder.dispatchThreadgroups(
              MTLSize(
                width: maximumModelGroups, height: 1024 / sharedThreads, depth: recordIndices.count),
              threadsPerThreadgroup: MTLSize(width: sharedThreads, height: 1, depth: 1))
          } else {
            encoder.dispatchThreadgroups(
              MTLSize(width: dispatchGroups, height: 32, depth: recordIndices.count),
              threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
          }
        }
        if sparseCount > 0 {
          encoder.setComputePipelineState(detectorSparsePipeline)
          encoder.setBuffer(table, offset: 0, index: 0)
          encoder.setBuffer(globals[2], offset: 0, index: 1)
          encoder.setBuffer(selection, offset: denseCount * 4, index: 2)
          encoder.setBuffer(coefficients, offset: denseCount * 4, index: 3)
          var parameters: [UInt32] = [UInt32(sparseCount), sparseColumns]
          encoder.setBytes(&parameters, length: 8, index: 4)
          encoder.dispatchThreads(
            MTLSize(width: sparseCount * 32, height: recordIndices.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        encoder.endEncoding()
      }
    }
    try Task.checkCancellation()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid(
        "Batched exact detector query failed: \(String(describing:command.error))")
    }
    return command.gpuEndTime - command.gpuStartTime
  }

  /// Explicit experiment preparation. The original source is never expanded;
  /// each temporary full 2D tile product is packed and independently checked.
  func prepareExperimentalTileIndex(maximumIndexBytes: UInt64, blockedPacking: Bool = false) throws
  {
    guard detectorColumnCost != nil else {
      throw TANSArchive.invalid(
        "Exact tile-index experiment requires authenticated detector planning costs")
    }
    guard !isReleased, maximumIndexBytes <= 2 << 30,
      UInt64(device.currentAllocatedSize) + maximumIndexBytes + (1 << 30)
        <= device.recommendedMaxWorkingSetSize
    else {
      throw TANSArchive.invalid(
        "Exact tile index requires its explicit <=2GiB budget and safe headroom")
    }
    let priorMask = previousDetectorMask
    let priorImages = previousDetectorImages
    let priorAcquisitions = previousDetectorAcquisitionIndices
    let priorEnabled = experimentalUseTileIndex
    let priorMode = experimentalDetectorStreamsPerLane
    defer {
      previousDetectorMask = priorMask
      previousDetectorImages = priorImages
      previousDetectorAcquisitionIndices = priorAcquisitions
      experimentalUseTileIndex = priorEnabled
      experimentalDetectorStreamsPerLane = priorMode
    }
    experimentalUseTileIndex = false
    experimentalDetectorStreamsPerLane = 32
    let index = try TANSExactTileIndex(
      device: device, queue: queue, library: tansLibrary,
      blockedPacking: blockedPacking)
    let coarse = stride(from: 0, to: 192, by: 32).flatMap { row in
      stride(from: 0, to: 192, by: 32).map { TANSExactTileIndex.Tile(row: row, col: $0, side: 32) }
    }
    let fine = stride(from: 64, to: 128, by: 8).flatMap { row in
      stride(from: 64, to: 128, by: 8).map { TANSExactTileIndex.Tile(row: row, col: $0, side: 8) }
    }
    for (ordinal, tile) in (coarse + fine).enumerated() {
      try Task.checkCancellation()
      try autoreleasepool {
        var mask = [UInt8](repeating: 0, count: 36864)
        for q in tile.pixels { mask[q] = validDetectorMask[q] }
        let images = try detectorImages(mask: mask, maximumAdditionalBytes: 1 << 30, rebase: true)
        try index.append(tile: tile, images: images, maximumBytes: maximumIndexBytes)
      }
      if ordinal % 10 == 9 {
        print(
          "TILE_INDEX_BUILD fields=\(ordinal+1) bytes=\(index.residentBytes) roundtrip_exact=true")
        fflush(stdout)
      }
    }
    exactTileIndex = index
  }

  /// Caller serializes queries and release; no in-flight command remains after a query returns.
  func releaseResidentStorage() {
    records.removeAll()
    globals.removeAll()
    exactTileIndex = nil
    output = nil
    previousDetectorMask = nil
    previousDetectorImages.removeAll()
    previousDetectorAcquisitionIndices.removeAll()
    isReleased = true
  }

  /// Expensive independent-parity diagnostic, not a rendering or resident path.
  /// Streams one36MiB decoded packet through SHA256, then reuses the same scratch.
  /// No full decoded acquisition is retained and audit time is not load timing.
  func auditFullCountSHA256(acquisitionIndex: Int) throws -> String {
    guard !isReleased, let selection = acquisitionIndices.firstIndex(of: acquisitionIndex) else {
      throw TANSArchive.invalid("Select a retained acquisition before auditing")
    }
    guard let scratch = device.makeBuffer(length: 512 * 36864 * 2, options: .storageModeShared)
    else {
      throw TANSArchive.invalid("Cannot allocate bounded36MiB parity scratch")
    }
    scratch.label = "Bounded tANS full-count audit scratch, not resident source"
    var digest = SHA256()
    for packet in 0..<512 {
      let localChunk = packet / 32
      let recordIndex = selection * 16 + localChunk
      let chunk = chunks[recordIndex]
      guard let command = queue.makeCommandBuffer(),
        let encoder = command.makeComputeCommandEncoder()
      else {
        throw TANSArchive.invalid("Cannot encode bounded tANS audit")
      }
      encoder.setComputePipelineState(auditPipeline)
      for (binding, name) in ["dense", "dense_offsets", "sparse", "sparse_offsets"].enumerated() {
        guard let component = chunk.components.first(where: { $0.name == name }),
          component.dtype == "<u4"
        else {
          encoder.endEncoding()
          throw TANSArchive.invalid("Missing authenticated entropy component \(name)")
        }
        encoder.setBuffer(records[recordIndex], offset: component.offset, index: binding)
      }
      for (binding, buffer) in globals.enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: binding + 4)
      }
      encoder.setBuffer(scratch, offset: 0, index: 8)
      var parameters: [UInt32] = [
        UInt32((packet % 32) * 512), retainedColumns, sparseColumns,
        UInt32((acquisitionIndex * 4 + localChunk / 4) * 36864),
      ]
      encoder.setBytes(&parameters, length: 16, index: 9)
      encoder.dispatchThreads(
        MTLSize(width: 36864, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed else {
        throw TANSArchive.invalid("Bounded tANS audit failed")
      }
      digest.update(
        bufferPointer: UnsafeRawBufferPointer(start: scratch.contents(), count: scratch.length))
    }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
