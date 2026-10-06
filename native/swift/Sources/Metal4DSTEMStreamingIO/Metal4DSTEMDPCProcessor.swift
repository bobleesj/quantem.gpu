import Foundation
import Metal
import Metal4DSTEMKernels

/// Fixed, consumer-selected DPC alignment for one complete scan plane.
public struct Metal4DSTEMDPCConfiguration: Equatable, Sendable {
  public let scanRows: Int
  public let scanColumns: Int
  public let rotationDegrees: Double
  public let transposeComponents: Bool

  public init(
    scanRows: Int,
    scanColumns: Int,
    rotationDegrees: Double,
    transposeComponents: Bool
  ) {
    self.scanRows = scanRows
    self.scanColumns = scanColumns
    self.rotationDegrees = rotationDegrees
    self.transposeComponents = transposeComponents
  }
}

/// Synchronized attribution for one already-resident DPC/iDPC publication.
public struct Metal4DSTEMDPCMetrics: Equatable, Sendable {
  public let wallMilliseconds: Double
  public let gpuMilliseconds: Double
  public let fftDispatchCount: Int
  public let totalDispatchCount: Int
  public let uploadBytes: UInt64
  public let readbackBytes: UInt64
  public let synchronizationCount: UInt64
  public let deviceAllocatedBytesBefore: UInt64
  public let deviceAllocatedBytesAfter: UInt64
}

/// GPU-resident iDPC and Fourier products for one complete scan plane.
///
/// The phase is a row-major float32 buffer. `gradientFFTBuffer` and
/// `phaseFFTBuffer` retain row-major complex-float products so a consumer can
/// compose an FFT view without a CPU round trip. The consumer owns their
/// lifetime and presentation policy.
public struct Metal4DSTEMDPCResidentResult {
  public let scanRows: Int
  public let scanColumns: Int
  public let phaseBuffer: MTLBuffer
  public let gradientFFTBuffer: MTLBuffer
  public let phaseFFTBuffer: MTLBuffer
  public let metrics: Metal4DSTEMDPCMetrics
}

private struct ConsumerFFT2DParameters {
  var width: UInt32
  var height: UInt32
  var log2Size: UInt32
  var stage: UInt32
  var direction: Float
  var rowAxis: UInt32
}

private struct ConsumerDPCPackParameters {
  var count: UInt32
  var flags: UInt32
  var padding0: UInt32 = 0
  var padding1: UInt32 = 0
  var rotation: SIMD4<Float>
}

private struct ConsumerDPCPipelines {
  let pack: MTLComputePipelineState
  let bitReverseRows: MTLComputePipelineState
  let bitReverseColumns: MTLComputePipelineState
  let butterflyRows: MTLComputePipelineState
  let butterflyColumns: MTLComputePipelineState
  let normalize: MTLComputePipelineState
  let poisson: MTLComputePipelineState
  let extract: MTLComputePipelineState
}

/// UI-free Metal DPC/iDPC processor for resident float32 CoM maps.
///
/// QuantEM.GPU owns the fixed rotation, FFT, Poisson integration, and resource
/// accounting. A consumer owns rotation selection, generation cancellation,
/// presentation, and publication acknowledgement.
public final class Metal4DSTEMDPCProcessor {
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let pipelines: ConsumerDPCPipelines

  public init(device: MTLDevice) throws {
    guard let queue = device.makeCommandQueue() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not create the DPC command queue."
      )
    }
    do {
      let library = try Metal4DSTEMKernels.makeDPCLibrary(device: device)
      pipelines = try ConsumerDPCPipelines(
        pack: Self.pipeline(
          library,
          Metal4DSTEMKernels.dpcPackFunction,
          device
        ),
        bitReverseRows: Self.pipeline(
          library,
          Metal4DSTEMKernels.fftBitReverseRowsFunction,
          device
        ),
        bitReverseColumns: Self.pipeline(
          library,
          Metal4DSTEMKernels.fftBitReverseColumnsFunction,
          device
        ),
        butterflyRows: Self.pipeline(
          library,
          Metal4DSTEMKernels.fftButterflyRowsFunction,
          device
        ),
        butterflyColumns: Self.pipeline(
          library,
          Metal4DSTEMKernels.fftButterflyColumnsFunction,
          device
        ),
        normalize: Self.pipeline(
          library,
          Metal4DSTEMKernels.fftNormalizeFunction,
          device
        ),
        poisson: Self.pipeline(
          library,
          Metal4DSTEMKernels.dpcPoissonFunction,
          device
        ),
        extract: Self.pipeline(
          library,
          Metal4DSTEMKernels.dpcExtractPhaseFunction,
          device
        )
      )
    } catch let error as Metal4DSTEMStreamingIOError {
      throw error
    } catch {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(error.localizedDescription)
    }
    self.device = device
    self.queue = queue
  }

  /// Upload centered float32 maps once and publish GPU-resident DPC/FFT outputs.
  public func process(
    centeredDPC: Metal4DSTEMCenteredDPC,
    configuration: Metal4DSTEMDPCConfiguration
  ) throws -> Metal4DSTEMDPCResidentResult {
    let count = try Self.validatedCount(configuration)
    guard centeredDPC.row.count == count, centeredDPC.column.count == count else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "DPC row and column maps must contain one value per complete scan position."
      )
    }
    guard centeredDPC.row.allSatisfy(\.isFinite),
      centeredDPC.column.allSatisfy(\.isFinite)
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "DPC row and column maps must contain only finite float32 values."
      )
    }
    let row = try makeBuffer(values: centeredDPC.row, role: "row DPC upload")
    let column = try makeBuffer(values: centeredDPC.column, role: "column DPC upload")
    return try process(
      centeredRowBuffer: row,
      centeredColumnBuffer: column,
      configuration: configuration,
      uploadBytes: UInt64(count * MemoryLayout<Float>.stride * 2)
    )
  }

  /// Publish GPU-resident DPC/FFT outputs from caller-owned resident maps.
  ///
  /// Both inputs must contain row-major float32 values on this processor's
  /// Metal device. They remain caller-owned and are not modified.
  public func process(
    centeredRowBuffer: MTLBuffer,
    centeredColumnBuffer: MTLBuffer,
    configuration: Metal4DSTEMDPCConfiguration
  ) throws -> Metal4DSTEMDPCResidentResult {
    try process(
      centeredRowBuffer: centeredRowBuffer,
      centeredColumnBuffer: centeredColumnBuffer,
      configuration: configuration,
      uploadBytes: 0
    )
  }

  private func process(
    centeredRowBuffer: MTLBuffer,
    centeredColumnBuffer: MTLBuffer,
    configuration: Metal4DSTEMDPCConfiguration,
    uploadBytes: UInt64
  ) throws -> Metal4DSTEMDPCResidentResult {
    let count = try Self.validatedCount(configuration)
    let scalarBytes = try Self.byteCount(
      count: count,
      stride: MemoryLayout<Float>.stride,
      role: "DPC scalar map"
    )
    let complexBytes = try Self.byteCount(
      count: count,
      stride: MemoryLayout<SIMD2<Float>>.stride,
      role: "DPC complex map"
    )
    guard centeredRowBuffer.device.registryID == device.registryID,
      centeredColumnBuffer.device.registryID == device.registryID
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "DPC input buffers must belong to the processor's Metal device."
      )
    }
    guard centeredRowBuffer.length >= scalarBytes,
      centeredColumnBuffer.length >= scalarBytes
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "DPC input buffers must contain one row-major float32 value per scan position."
      )
    }

    let allocatedBefore = UInt64(device.currentAllocatedSize)
    let gradient = try makeBuffer(
      length: complexBytes,
      options: .storageModePrivate,
      role: "DPC gradient FFT"
    )
    let phaseFFT = try makeBuffer(
      length: complexBytes,
      options: .storageModePrivate,
      role: "iDPC phase FFT"
    )
    let phase = try makeBuffer(
      length: scalarBytes,
      options: .storageModeShared,
      role: "iDPC phase"
    )
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.commandFailed(
        "Metal could not create the DPC/iDPC command."
      )
    }

    let started = DispatchTime.now().uptimeNanoseconds
    encodePack(
      encoder: encoder,
      row: centeredRowBuffer,
      column: centeredColumnBuffer,
      gradient: gradient,
      configuration: configuration,
      count: count
    )
    encoder.memoryBarrier(scope: .buffers)
    encodeFFT(
      encoder: encoder,
      buffer: gradient,
      rows: configuration.scanRows,
      columns: configuration.scanColumns,
      inverse: false
    )
    var shape = SIMD4<UInt32>(
      UInt32(configuration.scanColumns),
      UInt32(configuration.scanRows),
      UInt32(count),
      0
    )
    encoder.setComputePipelineState(pipelines.poisson)
    encoder.setBuffer(gradient, offset: 0, index: 0)
    encoder.setBuffer(phaseFFT, offset: 0, index: 1)
    encoder.setBytes(&shape, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 2)
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.memoryBarrier(scope: .buffers)
    encodeFFT(
      encoder: encoder,
      buffer: phaseFFT,
      rows: configuration.scanRows,
      columns: configuration.scanColumns,
      inverse: true
    )
    encoder.setComputePipelineState(pipelines.extract)
    encoder.setBuffer(phaseFFT, offset: 0, index: 0)
    encoder.setBuffer(phase, offset: 0, index: 1)
    var countU32 = UInt32(count)
    encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.stride, index: 2)
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    let finished = DispatchTime.now().uptimeNanoseconds
    guard command.status == .completed else {
      throw Metal4DSTEMStreamingIOError.commandFailed(
        command.error?.localizedDescription ?? "The DPC/iDPC command did not complete."
      )
    }
    let fftDispatches =
      2
      * (2
        + configuration.scanRows.trailingZeroBitCount
        + configuration.scanColumns.trailingZeroBitCount) + 1
    let metrics = Metal4DSTEMDPCMetrics(
      wallMilliseconds: Double(finished - started) / 1_000_000,
      gpuMilliseconds: command.gpuEndTime > command.gpuStartTime
        ? (command.gpuEndTime - command.gpuStartTime) * 1_000 : 0,
      fftDispatchCount: fftDispatches,
      totalDispatchCount: fftDispatches + 3,
      uploadBytes: uploadBytes,
      readbackBytes: 0,
      synchronizationCount: 1,
      deviceAllocatedBytesBefore: allocatedBefore,
      deviceAllocatedBytesAfter: UInt64(device.currentAllocatedSize)
    )
    return Metal4DSTEMDPCResidentResult(
      scanRows: configuration.scanRows,
      scanColumns: configuration.scanColumns,
      phaseBuffer: phase,
      gradientFFTBuffer: gradient,
      phaseFFTBuffer: phaseFFT,
      metrics: metrics
    )
  }

  private func encodePack(
    encoder: MTLComputeCommandEncoder,
    row: MTLBuffer,
    column: MTLBuffer,
    gradient: MTLBuffer,
    configuration: Metal4DSTEMDPCConfiguration,
    count: Int
  ) {
    let angle = configuration.rotationDegrees * .pi / 180
    var parameters = ConsumerDPCPackParameters(
      count: UInt32(count),
      flags: configuration.transposeComponents ? 1 : 0,
      rotation: SIMD4(Float(cos(angle)), Float(sin(angle)), 0, 0)
    )
    encoder.setComputePipelineState(pipelines.pack)
    encoder.setBuffer(row, offset: 0, index: 0)
    encoder.setBuffer(column, offset: 0, index: 1)
    encoder.setBuffer(gradient, offset: 0, index: 2)
    encoder.setBytes(
      &parameters,
      length: MemoryLayout<ConsumerDPCPackParameters>.stride,
      index: 3
    )
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
  }

  private func encodeFFT(
    encoder: MTLComputeCommandEncoder,
    buffer: MTLBuffer,
    rows: Int,
    columns: Int,
    inverse: Bool
  ) {
    let widthStages = UInt32(columns.trailingZeroBitCount)
    let heightStages = UInt32(rows.trailingZeroBitCount)
    func dispatch(
      _ pipeline: MTLComputePipelineState,
      width: Int,
      height: Int,
      log2Size: UInt32,
      stage: UInt32,
      rowAxis: Bool
    ) {
      var parameters = ConsumerFFT2DParameters(
        width: UInt32(columns),
        height: UInt32(rows),
        log2Size: log2Size,
        stage: stage,
        direction: inverse ? 1 : -1,
        rowAxis: rowAxis ? 1 : 0
      )
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(buffer, offset: 0, index: 0)
      encoder.setBytes(
        &parameters,
        length: MemoryLayout<ConsumerFFT2DParameters>.stride,
        index: 1
      )
      encoder.dispatchThreads(
        MTLSize(width: width, height: height, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
      )
      encoder.memoryBarrier(scope: .buffers)
    }
    dispatch(
      pipelines.bitReverseRows,
      width: columns,
      height: rows,
      log2Size: widthStages,
      stage: 0,
      rowAxis: true
    )
    for stage in 0..<widthStages {
      dispatch(
        pipelines.butterflyRows,
        width: columns / 2,
        height: rows,
        log2Size: widthStages,
        stage: stage,
        rowAxis: true
      )
    }
    dispatch(
      pipelines.bitReverseColumns,
      width: columns,
      height: rows,
      log2Size: heightStages,
      stage: 0,
      rowAxis: false
    )
    for stage in 0..<heightStages {
      dispatch(
        pipelines.butterflyColumns,
        width: columns,
        height: rows / 2,
        log2Size: heightStages,
        stage: stage,
        rowAxis: false
      )
    }
    if inverse {
      dispatch(
        pipelines.normalize,
        width: columns,
        height: rows,
        log2Size: heightStages,
        stage: 0,
        rowAxis: false
      )
    }
  }

  private static func validatedCount(
    _ configuration: Metal4DSTEMDPCConfiguration
  ) throws -> Int {
    let result = configuration.scanRows.multipliedReportingOverflow(
      by: configuration.scanColumns
    )
    guard configuration.scanRows > 0, configuration.scanColumns > 0,
      configuration.scanRows.isPowerOfTwo,
      configuration.scanColumns.isPowerOfTwo,
      !result.overflow,
      UInt32(exactly: result.partialValue) != nil,
      configuration.rotationDegrees.isFinite
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "DPC/iDPC requires finite rotation and positive power-of-two scan rows and columns within UInt32."
      )
    }
    return result.partialValue
  }

  private static func byteCount(
    count: Int,
    stride: Int,
    role: String
  ) throws -> Int {
    let result = count.multipliedReportingOverflow(by: stride)
    guard !result.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The \(role) byte count exceeds this process."
      )
    }
    return result.partialValue
  }

  private static func pipeline(
    _ library: MTLLibrary,
    _ name: String,
    _ device: MTLDevice
  ) throws -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "The packaged DPC library is missing \(name)."
      )
    }
    return try device.makeComputePipelineState(function: function)
  }

  private func makeBuffer(
    values: [Float],
    role: String
  ) throws -> MTLBuffer {
    guard
      let buffer = values.withUnsafeBytes({ raw in
        device.makeBuffer(
          bytes: raw.baseAddress!,
          length: raw.count,
          options: .storageModeShared
        )
      })
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: role,
        bytes: UInt64(values.count * MemoryLayout<Float>.stride)
      )
    }
    return buffer
  }

  private func makeBuffer(
    length: Int,
    options: MTLResourceOptions,
    role: String
  ) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: length, options: options) else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: role,
        bytes: UInt64(length)
      )
    }
    return buffer
  }
}

extension Int {
  fileprivate var isPowerOfTwo: Bool {
    self > 0 && self & (self - 1) == 0
  }
}
