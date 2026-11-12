import Foundation
import Metal
import XCTest

@testable import MetalDirectPtychoKernels

final class DirectPtychoBluesteinFFTTests: XCTestCase {
  /// A few complex impulses give an independent analytic DFT without a dense CPU FFT.
  func testBatchedComplexImpulseTransformsOnExactRectangularGrid() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    for (rows, columns) in [
      (3, 5), (90, 68), (256, 257), (511, 509), (509, 511), (457, 456), (464, 510),
    ] {
      let plan = try MetalDirectPtychoBluesteinFFT(
        device: device, queue: queue, rows: rows, columns: columns)
      XCTAssertLessThanOrEqual(plan.tableBytes, 40_960)
      let batch = 3
      let plane = rows * columns
      let input = try XCTUnwrap(
        device.makeBuffer(length: batch * plane * 8, options: .storageModeShared))
      let temporary = try XCTUnwrap(
        device.makeBuffer(length: input.length, options: .storageModePrivate))
      memset(input.contents(), 0, input.length)
      let data = input.contents().assumingMemoryBound(to: SIMD2<Float>.self)
      let locations = [(0, 0), (1, 2), (rows - 1, columns - 1), (rows / 2, columns / 2)]
      let values: [SIMD2<Float>] = [
        SIMD2(0.5, 0.125), SIMD2(-0.125, 0.25), SIMD2(0.0625, -0.25), SIMD2(-0.03125, -0.125),
      ]
      for b in 0..<batch {
        for (location, value) in zip(locations, values) {
          data[b * plane + location.0 * columns + location.1] += value * Float(b + 1)
        }
      }
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      plan.encode(command: command, input: input, temporary: temporary, batch: batch)
      command.commit()
      command.waitUntilCompleted()
      XCTAssertNil(command.error)
      var largestError = 0.0
      for b in 0..<batch {
        for row in 0..<rows {
          for column in 0..<columns {
            var real = 0.0
            var imaginary = 0.0
            for (location, value) in zip(locations, values) {
              let angle =
                2 * Double.pi
                * (Double(row * location.0) / Double(rows)
                  + Double(column * location.1) / Double(columns))
              real += Double(value.x) * cos(angle) + Double(value.y) * sin(angle)
              imaginary += Double(value.y) * cos(angle) - Double(value.x) * sin(angle)
            }
            let actual = data[b * plane + row * columns + column] / Float(b + 1)
            largestError = max(
              largestError, abs(Double(actual.x) - real), abs(Double(actual.y) - imaginary))
          }
        }
      }
      XCTAssertLessThan(
        largestError, 2e-6, "\(rows)x\(columns), every batch and complex output bin")
    }
  }

  /// Strong positive/negative backgrounds must not leak into weak nonzero frequencies.
  func testBackgroundDoesNotHideWeakFourierEvidence() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    for (rows, columns) in [(511, 509), (457, 456)] {
      let plan = try MetalDirectPtychoBluesteinFFT(
        device: device, queue: queue, rows: rows, columns: columns)
      let plane = rows * columns
      let input = try XCTUnwrap(device.makeBuffer(length: plane * 8, options: .storageModeShared))
      let temporary = try XCTUnwrap(
        device.makeBuffer(length: input.length, options: .storageModePrivate))
      let data = input.contents().assumingMemoryBound(to: SIMD2<Float>.self)
      for background: Float in [-100, -1, 0, 1, 100] {
        for i in 0..<plane { data[i] = SIMD2(background, 0) }
        data[0].x += 0.5
        data[columns + 2].x -= 0.125
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        plan.encode(command: command, input: input, temporary: temporary, batch: 1)
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)
        let expectedDC = Float(Double(background) * Double(plane) + 0.375)
        XCTAssertLessThanOrEqual(abs(data[0].x - expectedDC), expectedDC.ulp)
        XCTAssertEqual(data[0].y, 0)
        var error = 0.0
        for i in 1..<plane {
          let angle =
            2 * Double.pi
            * (Double(i / columns) / Double(rows) + 2 * Double(i % columns) / Double(columns))
          error = max(
            error, abs(Double(data[i].x) - (0.5 - 0.125 * cos(angle))),
            abs(Double(data[i].y) - 0.125 * sin(angle)))
        }
        XCTAssertLessThan(error, 2e-6, "\(rows)x\(columns), background \(background)")
      }
    }
  }
}
