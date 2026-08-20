import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMKernels

/// Opt-in physical-Metal timing fallback for environments that sandbox package executables.
final class Metal4DSTEMExactBinnerBenchmarkTests: XCTestCase {
  func testOptInExactBin2KernelTiming() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["QUANTEM_GPU_METAL_BINNING_BENCHMARK"] == "1" else {
      throw XCTSkip("Set QUANTEM_GPU_METAL_BINNING_BENCHMARK=1 to collect kernel timing.")
    }
    let batchRows = try positiveEnvironmentInt("QUANTEM_GPU_BINNING_BATCH_ROWS", default: 1)
    let scanColumns = try positiveEnvironmentInt(
      "QUANTEM_GPU_BINNING_SCAN_COLUMNS", default: 64
    )
    let warmups = try positiveEnvironmentInt("QUANTEM_GPU_BINNING_WARMUPS", default: 3)
    let iterations = try positiveEnvironmentInt("QUANTEM_GPU_BINNING_ITERATIONS", default: 15)
    let device = try XCTUnwrap(
      availableMetalDevice(),
      "The opt-in benchmark requires a physical Metal device."
    )
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let sourceMaximum: UInt32 = 53
    let audit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: String(repeating: "a", count: 64),
      sourceDtype: .uint16,
      badPixelIndices: [],
      maximumSourceCount: sourceMaximum,
      pixelsAbove255: 0
    )
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: batchRows,
      sourceScanColumns: scanColumns,
      detectorRows: 192,
      detectorColumns: 192,
      sourceBytesPerValue: MemoryLayout<UInt16>.stride,
      scanRegion: Metal4DSTEMScanRegion.full(
        sourceRows: batchRows,
        sourceColumns: scanColumns
      ),
      detectorBin: 2
    )
    let provenance = try Metal4DSTEMExactBinner.provenance(
      plan: plan,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )
    let fullPlan = try Metal4DSTEMLoadPlan(
      sourceScanRows: 512,
      sourceScanColumns: 512,
      detectorRows: 192,
      detectorColumns: 192,
      sourceBytesPerValue: MemoryLayout<UInt16>.stride,
      scanRegion: Metal4DSTEMScanRegion.full(sourceRows: 512, sourceColumns: 512),
      detectorBin: 2
    )
    let fullProvenance = try Metal4DSTEMExactBinner.provenance(
      plan: fullPlan,
      sourceAudit: audit,
      stagingDtype: .uint16,
      outputDtype: .uint16
    )
    let sourceValueCount = try checkedProduct([
      batchRows, scanColumns, plan.detectorPixels,
    ])
    let sourceBytes = try checkedProduct([
      sourceValueCount, MemoryLayout<UInt16>.stride,
    ])
    let source = try XCTUnwrap(
      device.makeBuffer(length: sourceBytes, options: .storageModeShared)
    )
    let sourceValues = source.contents().bindMemory(
      to: UInt16.self,
      capacity: sourceValueCount
    )
    for index in 0..<sourceValueCount {
      sourceValues[index] = UInt16(index % (Int(sourceMaximum) + 1))
    }
    let destinationBytes = try XCTUnwrap(Int(exactly: provenance.outputPayloadBytes))
    let destination = try XCTUnwrap(
      device.makeBuffer(
        length: destinationBytes,
        options: .storageModeShared
      )
    )
    memset(destination.contents(), 0, destination.length)
    let binner = try Metal4DSTEMExactBinner(device: device)
    var wallSamples: [Double] = []
    var gpuSamples: [Double] = []
    for iteration in 0..<(warmups + iterations) {
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      let started = DispatchTime.now().uptimeNanoseconds
      _ = try binner.encodeBatch(
        commandBuffer: command,
        stagedSource: source,
        destination: destination,
        plan: plan,
        sourceBatchRows: batchRows,
        destinationScanRowOffset: 0,
        sourceAudit: audit,
        stagingDtype: .uint16,
        outputDtype: .uint16
      )
      command.commit()
      command.waitUntilCompleted()
      XCTAssertEqual(command.status, .completed, command.error?.localizedDescription ?? "")
      let finished = DispatchTime.now().uptimeNanoseconds
      if iteration >= warmups {
        wallSamples.append(Double(finished - started) / 1_000_000)
        if command.gpuEndTime > command.gpuStartTime {
          gpuSamples.append((command.gpuEndTime - command.gpuStartTime) * 1_000)
        }
      }
    }
    let output = Data(bytes: destination.contents(), count: destination.length)
    let digest = SHA256.hash(data: output)
      .map { String(format: "%02x", $0) }
      .joined()
    let wall = timingSummary(wallSamples)
    let gpu = gpuSamples.count == wallSamples.count ? timingSummary(gpuSamples) : nil
    let result: [String: Any] = [
      "schema": "quantem.gpu.metal-4dstem-binning-xctest/v1",
      "definition":
        "kernel only; source already staged; excludes IO, decode, cache, products, and UI",
      "device": device.name,
      "batch_rows": batchRows,
      "scan_columns": scanColumns,
      "source_detector_shape": [192, 192],
      "output_detector_shape": [96, 96],
      "scan_bin": 1,
      "detector_bin": 2,
      "source_dtype": "uint16",
      "staging_dtype": "uint16",
      "output_dtype": "uint16",
      "source_maximum": sourceMaximum,
      "staged_source_bytes": sourceBytes,
      "batch_output_bytes": provenance.outputPayloadBytes,
      "full_512_output_bytes": fullProvenance.outputPayloadBytes,
      "max_buffer_length": device.maxBufferLength,
      "recommended_max_working_set_size": device.recommendedMaxWorkingSetSize,
      "warmups": warmups,
      "iterations": iterations,
      "wall_ms": wall,
      "gpu_ms": gpu ?? NSNull(),
      "output_sha256": digest,
    ]
    let json = try JSONSerialization.data(
      withJSONObject: result,
      options: [.sortedKeys, .withoutEscapingSlashes]
    )
    print("QUANTEM_GPU_METAL_BINNING_RESULT=" + String(decoding: json, as: UTF8.self))
  }

  private func positiveEnvironmentInt(_ name: String, default value: Int) throws -> Int {
    guard let text = ProcessInfo.processInfo.environment[name] else { return value }
    guard let parsed = Int(text), parsed > 0 else {
      throw NSError(
        domain: "Metal4DSTEMExactBinnerBenchmarkTests",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "\(name) must be a positive integer."]
      )
    }
    return parsed
  }

  private func availableMetalDevice() -> MTLDevice? {
    if let device = MTLCreateSystemDefaultDevice() { return device }
    #if os(macOS)
      return MTLCopyAllDevices().first
    #else
      return nil
    #endif
  }

  private func timingSummary(_ values: [Double]) -> [String: Any] {
    let sorted = values.sorted()
    func percentile(_ probability: Double) -> Double {
      let rank = max(1, Int(ceil(probability * Double(sorted.count))))
      return sorted[min(rank - 1, sorted.count - 1)]
    }
    return [
      "samples": sorted.count,
      "p50": percentile(0.50),
      "p95": percentile(0.95),
      "max": sorted.last!,
    ]
  }

  private func checkedProduct(_ values: [Int]) throws -> Int {
    var product = 1
    for value in values {
      let next = product.multipliedReportingOverflow(by: value)
      guard !next.overflow else {
        throw NSError(
          domain: "Metal4DSTEMExactBinnerBenchmarkTests",
          code: 2,
          userInfo: [NSLocalizedDescriptionKey: "Benchmark shape overflows Int."]
        )
      }
      product = next.partialValue
    }
    return product
  }
}
