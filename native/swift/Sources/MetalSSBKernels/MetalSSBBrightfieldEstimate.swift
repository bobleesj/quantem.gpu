import Foundation
import Metal

/// Geometry measured from the complete detector sum, without an angular assumption.
/// The selection uses mean + standard deviation; the aperture edge uses half the
/// median intensity inside that selection, matching the Python SSB convention.
public struct MetalSSBBrightfieldEstimate: Sendable {
  public let centerRow: Double
  public let centerColumn: Double
  public let selectionRadius: Double
  public let edgeRadius: Double
  public let selectionCount: Int
  public let edgeCount: Int

  /// Infer a semi-angle using the equal-area radius and calibrated pixel area.
  public func semiangle(rowSampling: Double, columnSampling: Double) throws -> Double {
    guard rowSampling.isFinite, rowSampling > 0,
      columnSampling.isFinite, columnSampling > 0
    else {
      throw MetalSSBError.invalidGeometry("Enter positive detector spacing in mrad/px.")
    }
    return edgeRadius * sqrt(rowSampling) * sqrt(columnSampling)
  }

  /// Infer isotropic detector spacing from a calibrated convergence semi-angle.
  public func detectorSampling(semiangle: Double) throws -> Double {
    guard semiangle.isFinite, semiangle > 0 else {
      throw MetalSSBError.invalidGeometry("Enter a positive convergence semi-angle in mrad.")
    }
    return semiangle / edgeRadius
  }

  /// Measure on Metal. The input is the exact small detector reduction, not a
  /// decoded scan cube. Excluded pixels contribute zero, as in SSB preparation.
  /// Empty, flat and clipped disks fail rather than inventing a calibration.
  public static func measure(
    device: MTLDevice, detectorSum: [UInt64], rows: Int, columns: Int,
    excluded: Set<Int> = []
  ) throws -> Self {
    guard rows > 1, columns > 1, rows <= 4096, columns <= 4096,
      detectorSum.count == rows * columns,
      excluded.allSatisfy({ detectorSum.indices.contains($0) })
    else {
      throw MetalSSBError.invalidGeometry(
        "Detector sums and exclusions must match the full detector shape.")
    }
    guard
      let url = Bundle.module.url(
        forResource: "ssb_calibration", withExtension: "metal", subdirectory: "Resources")
    else {
      throw MetalSSBError.missingResource("ssb_calibration.metal")
    }
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try device.makeLibrary(
      source: String(contentsOf: url, encoding: .utf8), options: options)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalSSBError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    let seed = try pipeline("ssb_disk_seed")
    let sort = try pipeline("ssb_disk_sort")
    let measure = try pipeline("ssb_disk_measure")
    var count = 1
    while count < detectorSum.count { count *= 2 }
    var input = detectorSum
    for index in excluded { input[index] = 0 }
    guard let raw = device.makeBuffer(bytes: input, length: input.count * 8),
      let ordered = device.makeBuffer(length: count * 8, options: .storageModePrivate),
      let result = device.makeBuffer(length: 5 * 8, options: .storageModeShared),
      let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw MetalSSBError.allocation("BF calibration buffers")
    }
    var shape = SIMD4<UInt32>(UInt32(rows), UInt32(columns), UInt32(count), 0)
    encoder.setBuffer(raw, offset: 0, index: 0)
    encoder.setBuffer(ordered, offset: 0, index: 1)
    encoder.setBuffer(result, offset: 0, index: 2)
    encoder.setBytes(&shape, length: MemoryLayout.size(ofValue: shape), index: 3)
    let group = MTLSize(width: 256, height: 1, depth: 1)
    encoder.setComputePipelineState(seed)
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: group)
    encoder.memoryBarrier(scope: .buffers)
    encoder.setComputePipelineState(sort)
    var width = 2
    while width <= count {
      var stride = width / 2
      while stride > 0 {
        var step = SIMD2<UInt32>(UInt32(width), UInt32(stride))
        encoder.setBytes(&step, length: MemoryLayout.size(ofValue: step), index: 4)
        encoder.dispatchThreads(
          MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: group)
        encoder.memoryBarrier(scope: .buffers)
        stride /= 2
      }
      width *= 2
    }
    encoder.setComputePipelineState(measure)
    encoder.dispatchThreadgroups(
      MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: group)
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error {
      throw MetalSSBError.commandExecution(error.localizedDescription)
    }
    let values = result.contents().bindMemory(to: UInt64.self, capacity: 5)
    guard values[0] > 0, values[3] > 0, values[4] == 0 else {
      throw MetalSSBError.invalidGeometry(
        "No complete BF disk was found. Verify the diffraction pattern and detector mask, or enter the disk geometry and angular calibration manually."
      )
    }
    let selected = Double(values[0])
    return Self(
      centerRow: Double(values[1]) / selected, centerColumn: Double(values[2]) / selected,
      selectionRadius: sqrt(selected / .pi), edgeRadius: sqrt(Double(values[3]) / .pi),
      selectionCount: Int(values[0]), edgeCount: Int(values[3]))
  }
}
