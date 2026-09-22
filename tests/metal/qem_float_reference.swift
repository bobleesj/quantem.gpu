import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

@main struct QEMFloatReference {
  static func main() throws {
    guard CommandLine.arguments.count == 3, let device = MTLCreateSystemDefaultDevice() else {
      fatalError("Usage: qem-float-reference float32.npy acquisition.qem")
    }
    let numpy = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    precondition(numpy.prefix(8) == Data([147, 78, 85, 77, 80, 89, 1, 0]))
    let start = 10 + Int(numpy[8]) + Int(numpy[9]) * 256
    let source = try NativeEMPADSource.open(URL(fileURLWithPath: CommandLine.arguments[2]))
    let resident = try MetalEMPADResidentSource.load(source, device: device,
      memoryBudgetBytes: UInt64(device.recommendedMaxWorkingSetSize))
    defer { resident.releaseResidentStorage() }
    let frameBytes = source.detectorPixelCount * 4
    precondition(numpy.count - start == source.frameCount * frameBytes)
    let output = device.makeBuffer(length: frameBytes, options: .storageModeShared)!
    let queue = device.makeCommandQueue()!
    for frame in 0..<source.frameCount {
      let command = queue.makeCommandBuffer()!
      try resident.encodeDiffraction(scanRow: frame / source.scanColumns,
        scanColumn: frame % source.scanColumns, into: output, command: command)
      command.commit()
      command.waitUntilCompleted()
      precondition(command.status == .completed)
      let actual = Data(bytes: output.contents(), count: frameBytes)
      precondition(actual == numpy.subdata(in: start + frame * frameBytes..<start + (frame + 1) * frameBytes))
    }
    // The same public native reader ingests the original, saves it, and restores
    // it without using a full dense or packed acquisition on the device.
    let original = try NativeEMPADSource.open(URL(fileURLWithPath: CommandLine.arguments[1]))
    let encoded = try MetalEMPADResidentSource.load(original, device: device,
      memoryBudgetBytes: UInt64(device.recommendedMaxWorkingSetSize))
    defer { encoded.releaseResidentStorage() }
    let copy = URL(fileURLWithPath: CommandLine.arguments[2]).deletingLastPathComponent()
      .appendingPathComponent("native-\(UUID().uuidString).qem")
    defer { try? FileManager.default.removeItem(at: copy) }
    try encoded.saveQEM(to: copy)
    let reopened = try MetalEMPADResidentSource.load(NativeEMPADSource.open(copy), device: device,
      memoryBudgetBytes: UInt64(device.recommendedMaxWorkingSetSize))
    defer { reopened.releaseResidentStorage() }
    precondition(reopened.source.detectorShape == original.detectorShape)
    let capabilities = try Metal4DSTEMResidentCapabilities.empad(reopened)
    precondition(capabilities.detectorRows == original.detectorShape.row)
    precondition(capabilities.detectorColumns == original.detectorShape.column)
    precondition(capabilities.logicalTensorBytes == UInt64(original.frameCount * frameBytes))
    let mask = device.makeBuffer(length: source.detectorPixelCount, options: .storageModeShared)!
    let virtual = device.makeBuffer(length: source.frameCount * 4, options: .storageModeShared)!
    let selected = Array(Set([0, source.detectorPixelCount / 2, source.detectorPixelCount - 1])).sorted()
    func reference(_ frame: Int, _ pixel: Int) -> Float {
      numpy.withUnsafeBytes {
        Float(bitPattern: UInt32(littleEndian: $0.loadUnaligned(
          fromByteOffset: start + frame * frameBytes + pixel * 4, as: UInt32.self)))
      }
    }
    func agrees(_ actual: Float, _ expected: Double) -> Bool {
      if expected.isNaN { return actual.isNaN }
      if expected.isInfinite { return Double(actual) == expected }
      return abs(Double(actual) - expected) <= 2e-5 * max(1, abs(expected))
    }
    // Two masks exercise the incremental path after a completed first command.
    for pixels in [selected, Array(selected.prefix(1))] {
      memset(mask.contents(), 0, mask.length)
      for pixel in pixels { mask.contents().storeBytes(of: UInt8(1), toByteOffset: pixel, as: UInt8.self) }
      let command = queue.makeCommandBuffer()!
      try reopened.encodeVirtualImage(mask: mask, into: virtual, command: command)
      command.commit()
      command.waitUntilCompleted()
      precondition(command.status == .completed)
      for frame in 0..<source.frameCount {
        let expected = pixels.reduce(0.0) { $0 + Double(reference(frame, $1)) }
        precondition(agrees(virtual.contents().load(fromByteOffset: frame * 4, as: Float.self), expected))
      }
    }
    if source.scanRows >= 2 && source.scanColumns >= 2 {
      let side = min(4, source.scanRows, source.scanColumns)
      for regionShape in [MetalScanRegionShape.rectangle, .circle] {
        let command = queue.makeCommandBuffer()!
        try reopened.encodeMeanDiffraction(into: output, command: command,
          rows: 0..<side, columns: 0..<side, shape: regionShape)
        command.commit()
        command.waitUntilCompleted()
        precondition(command.status == .completed)
        for pixel in selected {
          let frames = (0..<side).flatMap { row in
            (0..<side).compactMap { column -> Int? in
              let dr = Double(row) + 0.5 - Double(side) / 2
              let dc = Double(column) + 0.5 - Double(side) / 2
              if regionShape == .circle && dr * dr + dc * dc > Double(side * side) / 4 {
                return nil
              }
              return row * source.scanColumns + column
            }
          }
          if side == 4 { precondition(frames.count == (regionShape == .circle ? 12 : 16)) }
          let expected = frames.reduce(0.0) { $0 + Double(reference($1, pixel)) / Double(frames.count) }
          precondition(agrees(output.contents().load(fromByteOffset: pixel * 4, as: Float.self), expected))
        }
      }
    }
    for frame in 0..<source.frameCount {
      let command = queue.makeCommandBuffer()!
      try reopened.encodeDiffraction(scanRow: frame / source.scanColumns,
        scanColumn: frame % source.scanColumns, into: output, command: command)
      command.commit()
      command.waitUntilCompleted()
      precondition(command.status == .completed)
      precondition(Data(bytes: output.contents(), count: frameBytes)
        == numpy.subdata(in: start + frame * frameBytes..<start + (frame + 1) * frameBytes))
    }
    print("PASS native Metal: every float32 bit, including signed zero, NaN payload and infinity")
  }
}
