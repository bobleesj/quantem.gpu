import Metal
import MetalDisplayKernels
import XCTest

@testable import MetalImageRuntime

final class MetalImageRuntimeTests: XCTestCase {
  func testUInt32StatisticsAndPercentileWindow() throws {
    let device = try metalDevice()
    let runtime = try MetalDisplayStatistics(device: device)
    let values = Array(UInt32(0)...UInt32(7))
    let buffer = try makeBuffer(device: device, values: values)

    let statistics = try runtime.analyzeUInt32(
      values: buffer,
      rows: 1,
      columns: values.count,
      scale: .linear
    )

    XCTAssertEqual(statistics.minimum, 0)
    XCTAssertEqual(statistics.maximum, 7)
    XCTAssertEqual(statistics.bins.reduce(UInt32(0), +), UInt32(values.count))
    let window = MetalHistogramContrast.percentileWindow(
      bins: statistics.bins,
      lowerPercentile: 0,
      upperPercentile: 1
    )
    XCTAssertEqual(window.low, 0)
    XCTAssertEqual(window.high, 1)
  }

  func testFloatStatisticsExcludeNonfiniteAndPreserveSignedRange() throws {
    let device = try metalDevice()
    let runtime = try MetalDisplayStatistics(device: device)
    let values: [Float] = [-2, -1, 0, 2, .nan, -.infinity, .infinity]
    let buffer = try makeBuffer(device: device, values: values)

    let statistics = try runtime.analyzeFloat32(
      values: buffer,
      rows: 1,
      columns: values.count,
      scale: .linear
    )

    XCTAssertEqual(statistics.minimum, -2)
    XCTAssertEqual(statistics.maximum, 2)
    XCTAssertEqual(statistics.bins.reduce(UInt32(0), +), 4)
    XCTAssertEqual(statistics.bins[0], 1)
    XCTAssertEqual(statistics.bins[64], 1)
    XCTAssertEqual(statistics.bins[128], 1)
    XCTAssertEqual(statistics.bins[255], 1)
  }

  func testHistogramReferenceAndSurfaceDisplayParameters() throws {
    let reference = MetalHistogramDisplayContract.reference(
      values: [-2, -1, 0, 2, .nan, .infinity],
      scale: .linear
    )
    XCTAssertEqual(reference.invalidCount, 2)
    XCTAssertEqual(reference.finiteMinimum, -2)
    XCTAssertEqual(reference.finiteMaximum, 2)
    XCTAssertEqual(reference.bins.reduce(UInt32(0), +), 4)

    let device = try metalDevice()
    let runtime = try MetalDisplayStatistics(device: device)
    let values = Array(UInt32(0)...UInt32(7))
    let buffer = try makeBuffer(device: device, values: values)
    let statistics = try runtime.analyzeUInt32(
      values: buffer,
      rows: 1,
      columns: values.count,
      scale: .linear
    )
    var state = MetalUInt32SurfaceState(
      values: buffer,
      statistics: statistics,
      rows: 1,
      columns: values.count,
      scale: .linear,
      colormap: .viridis
    )
    XCTAssertFalse(
      state.configure(
        scale: .linear,
        colormap: .viridis,
        contrastLow: 0,
        contrastHigh: 1
      )
    )
    XCTAssertTrue(
      state.configure(
        scale: .linear,
        colormap: .plasma,
        contrastLow: 0.25,
        contrastHigh: 0.75
      )
    )
    let parameters = state.displayParameters()
    XCTAssertEqual(parameters.low, 2)
    XCTAssertEqual(parameters.high, 5)
  }

  private func metalDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal is unavailable on this runner")
    }
    return device
  }

  private func makeBuffer<T>(device: MTLDevice, values: [T]) throws -> MTLBuffer {
    try values.withUnsafeBytes { bytes in
      try XCTUnwrap(
        device.makeBuffer(
          bytes: bytes.baseAddress!,
          length: bytes.count,
          options: .storageModeShared
        )
      )
    }
  }
}
