import Metal

struct Metal4DSTEMScanDetectorBinParameters {
  var sourceRows: UInt32
  var sourceColumns: UInt32
  var sourceDetectorRows: UInt32
  var sourceDetectorColumns: UInt32
  var scanBin: UInt32
  var detectorBin: UInt32
  var outputScanCount: UInt32
  var outputScanColumns: UInt32
  var outputDetectorRows: UInt32
  var outputDetectorColumns: UInt32
  var destinationScanRowOffset: UInt32
  var padding: UInt32 = 0
}

struct Metal4DSTEMContiguousDetectorBinParameters {
  var frameCount: UInt32
  var sourceDetectorRows: UInt32
  var sourceDetectorColumns: UInt32
  var detectorBin: UInt32
  var destinationScanCount: UInt32
  var destinationScanOffset: UInt32
  var outputDetectorRows: UInt32
  var outputDetectorColumns: UInt32
}

extension Metal4DSTEMExactBinner {
  /// Encode one aligned batch into a complete or scan-row-sharded destination.
  ///
  /// `stagedSource` is frame-major over the selected scan columns. Values at
  /// `sourceAudit.badPixelIndices` must already be zero in every staged frame;
  /// the kernel does not apply the bad-pixel policy a second time. The caller
  /// retains ownership of both buffers and of command-buffer commit and
  /// synchronization.
  @discardableResult
  public func encodeBatch(
    commandBuffer: MTLCommandBuffer,
    stagedSource: MTLBuffer,
    stagedSourceOffset: Int = 0,
    destination: MTLBuffer,
    destinationView: Metal4DSTEMExactBinningDestination = .complete,
    plan: Metal4DSTEMLoadPlan,
    sourceBatchRows: Int,
    destinationScanRowOffset: Int,
    sourceAudit: Metal4DSTEMExactSourceAudit,
    stagingDtype: Metal4DSTEMIntegerDType,
    outputDtype: Metal4DSTEMIntegerDType
  ) throws -> Metal4DSTEMExactBinningProvenance {
    let provenance = try Self.provenance(
      plan: plan,
      sourceAudit: sourceAudit,
      stagingDtype: stagingDtype,
      outputDtype: outputDtype
    )
    guard sourceBatchRows > 0, sourceBatchRows <= plan.scanRegion.rows else {
      throw Metal4DSTEMExactBinnerError.invalidBatchRows(sourceBatchRows)
    }
    let sourceStart = destinationScanRowOffset.multipliedReportingOverflow(
      by: plan.scanBin
    )
    guard destinationScanRowOffset >= 0, !sourceStart.overflow,
      sourceStart.partialValue < plan.scanRegion.rows
    else {
      throw Metal4DSTEMExactBinnerError.invalidDestinationScanRowOffset(
        destinationScanRowOffset
      )
    }
    let remainingSourceRows = plan.scanRegion.rows - sourceStart.partialValue
    guard sourceBatchRows <= remainingSourceRows else {
      throw Metal4DSTEMExactBinnerError.invalidBatchCoverage(
        rows: sourceBatchRows, remaining: remainingSourceRows
      )
    }
    let adjustedRows = sourceBatchRows.addingReportingOverflow(plan.scanBin - 1)
    guard !adjustedRows.overflow else {
      throw Metal4DSTEMExactBinnerError.arithmeticOverflow
    }
    let localOutputRows = adjustedRows.partialValue / plan.scanBin
    let destinationStop = destinationScanRowOffset.addingReportingOverflow(localOutputRows)
    guard destinationScanRowOffset >= 0, !destinationStop.overflow,
      destinationStop.partialValue <= plan.outputScanRows
    else {
      throw Metal4DSTEMExactBinnerError.invalidDestinationScanRowOffset(
        destinationScanRowOffset
      )
    }
    let isFinalBatch = sourceBatchRows == remainingSourceRows
    guard (destinationStop.partialValue == plan.outputScanRows) == isFinalBatch else {
      throw Metal4DSTEMExactBinnerError.invalidBatchCoverage(
        rows: sourceBatchRows, remaining: remainingSourceRows
      )
    }
    guard isFinalBatch || sourceBatchRows % plan.scanBin == 0 else {
      throw Metal4DSTEMExactBinnerError.misalignedNonfinalBatch(
        rows: sourceBatchRows, scanBin: plan.scanBin
      )
    }
    guard stagedSourceOffset >= 0,
      stagedSourceOffset % stagingDtype.bytesPerValue == 0
    else {
      throw Metal4DSTEMExactBinnerError.invalidSourceOffset(stagedSourceOffset)
    }
    guard
      let sourceValues = Self.checkedProduct(
        UInt64(sourceBatchRows), UInt64(plan.scanRegion.columns)
      ).flatMap({ Self.checkedProduct($0, UInt64(plan.detectorPixels)) }),
      let sourceBytes = Self.checkedProduct(
        sourceValues, UInt64(stagingDtype.bytesPerValue)
      ), let sourceOffsetBytes = UInt64(exactly: stagedSourceOffset),
      let requiredSourceBytes = Self.checkedSum(sourceOffsetBytes, sourceBytes)
    else { throw Metal4DSTEMExactBinnerError.arithmeticOverflow }
    guard requiredSourceBytes <= UInt64(stagedSource.length) else {
      throw Metal4DSTEMExactBinnerError.sourceBufferTooSmall(
        expected: requiredSourceBytes, actual: UInt64(stagedSource.length)
      )
    }
    let destinationScanCount: Int
    let localDestinationScanRowOffset: Int
    let requiredDestinationBytes: UInt64
    switch destinationView {
    case .complete:
      destinationScanCount = plan.outputScanPositions
      localDestinationScanRowOffset = destinationScanRowOffset
      requiredDestinationBytes = provenance.outputPayloadBytes
    case .scanRowShard(let shardPlan, let index):
      try shardPlan.validate(provenance: provenance)
      guard shardPlan.shards.indices.contains(index) else {
        throw Metal4DSTEMExactBinnerError.invalidDestinationShard(index)
      }
      let shard = shardPlan.shards[index]
      guard destinationScanRowOffset >= shard.outputScanRowStart,
        destinationStop.partialValue <= shard.outputScanRowStop
      else {
        throw Metal4DSTEMExactBinnerError.batchCrossesDestinationShard(
          batchStart: destinationScanRowOffset,
          batchStop: destinationStop.partialValue,
          shardStart: shard.outputScanRowStart,
          shardStop: shard.outputScanRowStop
        )
      }
      destinationScanCount = shard.outputScanPositionCount
      localDestinationScanRowOffset =
        destinationScanRowOffset - shard.outputScanRowStart
      requiredDestinationBytes = shard.payloadBytes
    }
    guard requiredDestinationBytes <= UInt64(destination.length) else {
      throw Metal4DSTEMExactBinnerError.destinationBufferTooSmall(
        expected: requiredDestinationBytes, actual: UInt64(destination.length)
      )
    }
    guard let parameterSourceRows = UInt32(exactly: sourceBatchRows),
      let parameterSourceColumns = UInt32(exactly: plan.scanRegion.columns),
      let parameterDetectorRows = UInt32(exactly: plan.detectorRows),
      let parameterDetectorColumns = UInt32(exactly: plan.detectorColumns),
      let parameterScanBin = UInt32(exactly: plan.scanBin),
      let parameterDetectorBin = UInt32(exactly: plan.detectorBin),
      let parameterOutputScanCount = UInt32(exactly: destinationScanCount),
      let parameterOutputScanColumns = UInt32(exactly: plan.outputScanColumns),
      let parameterOutputDetectorRows = UInt32(exactly: plan.outputDetectorRows),
      let parameterOutputDetectorColumns = UInt32(exactly: plan.outputDetectorColumns),
      let parameterDestinationRowOffset = UInt32(
        exactly: localDestinationScanRowOffset
      )
    else { throw Metal4DSTEMExactBinnerError.arithmeticOverflow }
    guard Self.fitsMetalUIntProduct(sourceBatchRows, plan.scanRegion.columns),
      Self.fitsMetalUIntProduct(plan.detectorRows, plan.detectorColumns),
      Self.fitsMetalUIntProduct(localOutputRows, plan.outputScanColumns),
      Self.fitsMetalUIntProduct(plan.outputDetectorRows, plan.outputDetectorColumns),
      UInt64(destinationScanCount) <= UInt64(UInt32.max)
    else { throw Metal4DSTEMExactBinnerError.arithmeticOverflow }
    var parameters = Metal4DSTEMScanDetectorBinParameters(
      sourceRows: parameterSourceRows,
      sourceColumns: parameterSourceColumns,
      sourceDetectorRows: parameterDetectorRows,
      sourceDetectorColumns: parameterDetectorColumns,
      scanBin: parameterScanBin,
      detectorBin: parameterDetectorBin,
      outputScanCount: parameterOutputScanCount,
      outputScanColumns: parameterOutputScanColumns,
      outputDetectorRows: parameterOutputDetectorRows,
      outputDetectorColumns: parameterOutputDetectorColumns,
      destinationScanRowOffset: parameterDestinationRowOffset
    )
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw Metal4DSTEMExactBinnerError.commandEncoderUnavailable
    }
    encoder.setComputePipelineState(
      pipeline(stagingDtype: stagingDtype, outputDtype: outputDtype)
    )
    encoder.setBuffer(stagedSource, offset: stagedSourceOffset, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    encoder.setBytes(
      &parameters,
      length: MemoryLayout<Metal4DSTEMScanDetectorBinParameters>.stride,
      index: 2
    )
    let gridWidth =
      outputDtype == .uint16
      ? (plan.outputDetectorPixels + 1) / 2
      : plan.outputDetectorPixels
    encoder.dispatchThreads(
      MTLSize(
        width: gridWidth,
        height: localOutputRows * plan.outputScanColumns,
        depth: 1
      ),
      threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
    )
    encoder.endEncoding()
    return provenance
  }

  /// Encode a contiguous full-scan frame interval into exact packed uint16 shards.
  ///
  /// Prepared QH5 slices may begin or end inside a scan row. This method keeps
  /// those source intervals contiguous instead of manufacturing a rectangular
  /// row batch. It supports only a complete scan with `scanBin == 1`; the
  /// global scan-position offset therefore identifies both source coverage and
  /// destination placement. Values at audited bad-pixel indices must already
  /// be zero. The caller owns command-buffer commit and synchronization.
  @discardableResult
  public func encodeContiguousUInt16Frames(
    commandBuffer: MTLCommandBuffer,
    stagedSource: MTLBuffer,
    stagedSourceOffset: Int = 0,
    destination: MTLBuffer,
    destinationView: Metal4DSTEMExactBinningDestination = .complete,
    plan: Metal4DSTEMLoadPlan,
    sourceFrameCount: Int,
    globalScanPositionOffset: Int,
    sourceAudit: Metal4DSTEMExactSourceAudit
  ) throws -> Metal4DSTEMExactBinningProvenance {
    try encodeContiguousFrames(
      commandBuffer: commandBuffer,
      existingEncoder: nil,
      stagedSource: stagedSource,
      stagedSourceOffset: stagedSourceOffset,
      destination: destination,
      destinationView: destinationView,
      plan: plan,
      sourceFrameCount: sourceFrameCount,
      globalScanPositionOffset: globalScanPositionOffset,
      sourceAudit: sourceAudit,
      stagingDtype: .uint16,
      pipeline: contiguousU16ToU16
    )
  }

  /// Encode identity-audited lossless uint8 staging into exact packed uint16 shards.
  ///
  /// This is the compact-staging counterpart of `encodeContiguousUInt16Frames`.
  /// The source audit must prove every valid uint16 source value fits uint8;
  /// otherwise validation fails before a command encoder is created.
  @discardableResult
  public func encodeContiguousAuditedUInt8Frames(
    commandBuffer: MTLCommandBuffer,
    stagedSource: MTLBuffer,
    stagedSourceOffset: Int = 0,
    destination: MTLBuffer,
    destinationView: Metal4DSTEMExactBinningDestination = .complete,
    plan: Metal4DSTEMLoadPlan,
    sourceFrameCount: Int,
    globalScanPositionOffset: Int,
    sourceAudit: Metal4DSTEMExactSourceAudit
  ) throws -> Metal4DSTEMExactBinningProvenance {
    try encodeContiguousFrames(
      commandBuffer: commandBuffer,
      existingEncoder: nil,
      stagedSource: stagedSource,
      stagedSourceOffset: stagedSourceOffset,
      destination: destination,
      destinationView: destinationView,
      plan: plan,
      sourceFrameCount: sourceFrameCount,
      globalScanPositionOffset: globalScanPositionOffset,
      sourceAudit: sourceAudit,
      stagingDtype: .uint8,
      pipeline: contiguousU8ToU16
    )
  }

  /// Package-internal exact uint16 staging path for an existing encoder.
  package func encodeContiguousUInt16Frames(
    encoder: MTLComputeCommandEncoder,
    stagedSource: MTLBuffer,
    stagedSourceOffset: Int = 0,
    destination: MTLBuffer,
    destinationView: Metal4DSTEMExactBinningDestination = .complete,
    plan: Metal4DSTEMLoadPlan,
    sourceFrameCount: Int,
    globalScanPositionOffset: Int,
    sourceAudit: Metal4DSTEMExactSourceAudit
  ) throws -> Metal4DSTEMExactBinningProvenance {
    try encodeContiguousFrames(
      commandBuffer: nil,
      existingEncoder: encoder,
      stagedSource: stagedSource,
      stagedSourceOffset: stagedSourceOffset,
      destination: destination,
      destinationView: destinationView,
      plan: plan,
      sourceFrameCount: sourceFrameCount,
      globalScanPositionOffset: globalScanPositionOffset,
      sourceAudit: sourceAudit,
      stagingDtype: .uint16,
      pipeline: contiguousU16ToU16
    )
  }

  /// Package-internal audited uint8 staging path for an existing encoder.
  package func encodeContiguousAuditedUInt8Frames(
    encoder: MTLComputeCommandEncoder,
    stagedSource: MTLBuffer,
    stagedSourceOffset: Int = 0,
    destination: MTLBuffer,
    destinationView: Metal4DSTEMExactBinningDestination = .complete,
    plan: Metal4DSTEMLoadPlan,
    sourceFrameCount: Int,
    globalScanPositionOffset: Int,
    sourceAudit: Metal4DSTEMExactSourceAudit
  ) throws -> Metal4DSTEMExactBinningProvenance {
    try encodeContiguousFrames(
      commandBuffer: nil,
      existingEncoder: encoder,
      stagedSource: stagedSource,
      stagedSourceOffset: stagedSourceOffset,
      destination: destination,
      destinationView: destinationView,
      plan: plan,
      sourceFrameCount: sourceFrameCount,
      globalScanPositionOffset: globalScanPositionOffset,
      sourceAudit: sourceAudit,
      stagingDtype: .uint8,
      pipeline: contiguousU8ToU16
    )
  }

  private func encodeContiguousFrames(
    commandBuffer: MTLCommandBuffer?,
    existingEncoder: MTLComputeCommandEncoder?,
    stagedSource: MTLBuffer,
    stagedSourceOffset: Int,
    destination: MTLBuffer,
    destinationView: Metal4DSTEMExactBinningDestination,
    plan: Metal4DSTEMLoadPlan,
    sourceFrameCount: Int,
    globalScanPositionOffset: Int,
    sourceAudit: Metal4DSTEMExactSourceAudit,
    stagingDtype: Metal4DSTEMIntegerDType,
    pipeline: MTLComputePipelineState
  ) throws -> Metal4DSTEMExactBinningProvenance {
    let provenance = try Self.provenance(
      plan: plan,
      sourceAudit: sourceAudit,
      stagingDtype: stagingDtype,
      outputDtype: .uint16
    )
    guard plan.scanBin == 1,
      plan.scanRegion.rowStart == 0,
      plan.scanRegion.rowStop == plan.sourceScanRows,
      plan.scanRegion.columnStart == 0,
      plan.scanRegion.columnStop == plan.sourceScanColumns
    else {
      throw Metal4DSTEMExactBinnerError.contiguousFramesRequireFullScanBinOne
    }
    let frameStop = globalScanPositionOffset.addingReportingOverflow(sourceFrameCount)
    guard sourceFrameCount > 0, globalScanPositionOffset >= 0,
      !frameStop.overflow, frameStop.partialValue <= plan.outputScanPositions
    else {
      throw Metal4DSTEMExactBinnerError.invalidGlobalScanPositionRange(
        offset: globalScanPositionOffset,
        count: sourceFrameCount
      )
    }
    guard stagedSourceOffset >= 0,
      stagedSourceOffset % stagingDtype.bytesPerValue == 0
    else {
      throw Metal4DSTEMExactBinnerError.invalidSourceOffset(stagedSourceOffset)
    }
    guard
      let sourceValues = Self.checkedProduct(
        UInt64(sourceFrameCount), UInt64(plan.detectorPixels)
      ),
      let sourceBytes = Self.checkedProduct(
        sourceValues, UInt64(stagingDtype.bytesPerValue)
      ),
      let sourceOffsetBytes = UInt64(exactly: stagedSourceOffset),
      let requiredSourceBytes = Self.checkedSum(sourceOffsetBytes, sourceBytes)
    else { throw Metal4DSTEMExactBinnerError.arithmeticOverflow }
    guard requiredSourceBytes <= UInt64(stagedSource.length) else {
      throw Metal4DSTEMExactBinnerError.sourceBufferTooSmall(
        expected: requiredSourceBytes, actual: UInt64(stagedSource.length)
      )
    }

    let destinationScanCount: Int
    let localDestinationOffset: Int
    let requiredDestinationBytes: UInt64
    switch destinationView {
    case .complete:
      destinationScanCount = plan.outputScanPositions
      localDestinationOffset = globalScanPositionOffset
      requiredDestinationBytes = provenance.outputPayloadBytes
    case .scanRowShard(let shardPlan, let index):
      try shardPlan.validate(provenance: provenance)
      guard shardPlan.shards.indices.contains(index) else {
        throw Metal4DSTEMExactBinnerError.invalidDestinationShard(index)
      }
      let shard = shardPlan.shards[index]
      let shardStart = shard.outputScanPositionStart
      let shardStopResult = shardStart.addingReportingOverflow(
        shard.outputScanPositionCount
      )
      guard !shardStopResult.overflow else {
        throw Metal4DSTEMExactBinnerError.arithmeticOverflow
      }
      let shardStop = shardStopResult.partialValue
      guard globalScanPositionOffset >= shardStart,
        frameStop.partialValue <= shardStop
      else {
        throw Metal4DSTEMExactBinnerError.frameRangeCrossesDestinationShard(
          frameStart: globalScanPositionOffset,
          frameStop: frameStop.partialValue,
          shardStart: shardStart,
          shardStop: shardStop
        )
      }
      destinationScanCount = shard.outputScanPositionCount
      localDestinationOffset = globalScanPositionOffset - shardStart
      requiredDestinationBytes = shard.payloadBytes
    }
    guard requiredDestinationBytes <= UInt64(destination.length) else {
      throw Metal4DSTEMExactBinnerError.destinationBufferTooSmall(
        expected: requiredDestinationBytes, actual: UInt64(destination.length)
      )
    }
    guard let parameterFrameCount = UInt32(exactly: sourceFrameCount),
      let parameterDetectorRows = UInt32(exactly: plan.detectorRows),
      let parameterDetectorColumns = UInt32(exactly: plan.detectorColumns),
      let parameterDetectorBin = UInt32(exactly: plan.detectorBin),
      let parameterDestinationScanCount = UInt32(exactly: destinationScanCount),
      let parameterDestinationOffset = UInt32(exactly: localDestinationOffset),
      let parameterOutputDetectorRows = UInt32(exactly: plan.outputDetectorRows),
      let parameterOutputDetectorColumns = UInt32(
        exactly: plan.outputDetectorColumns
      ),
      Self.fitsMetalUIntProduct(plan.detectorRows, plan.detectorColumns),
      Self.fitsMetalUIntProduct(
        plan.outputDetectorRows, plan.outputDetectorColumns
      )
    else { throw Metal4DSTEMExactBinnerError.arithmeticOverflow }
    var parameters = Metal4DSTEMContiguousDetectorBinParameters(
      frameCount: parameterFrameCount,
      sourceDetectorRows: parameterDetectorRows,
      sourceDetectorColumns: parameterDetectorColumns,
      detectorBin: parameterDetectorBin,
      destinationScanCount: parameterDestinationScanCount,
      destinationScanOffset: parameterDestinationOffset,
      outputDetectorRows: parameterOutputDetectorRows,
      outputDetectorColumns: parameterOutputDetectorColumns
    )
    let gridWidth = (plan.outputDetectorPixels + 1) / 2
    let gridHeight = sourceFrameCount
    guard Self.fitsMetalUIntProduct(gridWidth, gridHeight) else {
      throw Metal4DSTEMExactBinnerError.arithmeticOverflow
    }
    guard let encoder = existingEncoder ?? commandBuffer?.makeComputeCommandEncoder()
    else {
      throw Metal4DSTEMExactBinnerError.commandEncoderUnavailable
    }
    let usesDetector192Bin1Tiled32 =
      stagingDtype == .uint8
      && plan.detectorRows == 192
      && plan.detectorColumns == 192
      && plan.detectorBin == 1
      && plan.outputDetectorRows == 192
      && plan.outputDetectorColumns == 192
    let usesDetector192Bin2Tiled32 =
      stagingDtype == .uint8
      && plan.detectorRows == 192
      && plan.detectorColumns == 192
      && plan.detectorBin == 2
      && plan.outputDetectorRows == 96
      && plan.outputDetectorColumns == 96
    encoder.setComputePipelineState(
      usesDetector192Bin1Tiled32
        ? contiguousBin1U8ToU16Tiled32
        : (usesDetector192Bin2Tiled32 ? contiguousBin2U8ToU16Tiled32 : pipeline)
    )
    encoder.setBuffer(stagedSource, offset: stagedSourceOffset, index: 0)
    encoder.setBuffer(destination, offset: 0, index: 1)
    encoder.setBytes(
      &parameters,
      length: MemoryLayout<Metal4DSTEMContiguousDetectorBinParameters>.stride,
      index: 2
    )
    if usesDetector192Bin1Tiled32 || usesDetector192Bin2Tiled32 {
      encoder.dispatchThreadgroups(
        MTLSize(
          width: (gridWidth + 31) / 32,
          height: (gridHeight + 31) / 32,
          depth: 1
        ),
        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
      )
    } else {
      encoder.dispatchThreads(
        MTLSize(width: gridWidth, height: gridHeight, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
      )
    }
    if existingEncoder == nil {
      encoder.endEncoding()
    }
    return provenance
  }
}
