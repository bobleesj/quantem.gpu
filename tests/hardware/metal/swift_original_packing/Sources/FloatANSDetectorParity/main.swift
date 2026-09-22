import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Same source and ordered masks in separate processes. Compare the resulting
// binary files with cmp to audit every float32 virtual-image word bit-for-bit.
let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.count == 2 else {
  fatalError("usage: FloatANSDetectorParity source.qem output.f32")
}
let source = try NativeEMPADSource.open(URL(fileURLWithPath: arguments[0]))
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let resident = try MetalEMPADResidentSource.load(
  source, device: device, memoryBudgetBytes: 12 << 30)
defer { resident.releaseResidentStorage() }
let pixels = source.detectorPixelCount
let columns = source.detectorShape.column
let rows = source.detectorShape.row
let mask = device.makeBuffer(length: pixels, options: .storageModeShared)!
let image = device.makeBuffer(length: source.frameCount * 4, options: .storageModeShared)!
let destination = URL(fileURLWithPath: arguments[1])
guard FileManager.default.createFile(atPath: destination.path, contents: nil) else {
  fatalError("Choose a new output path for parity evidence.")
}
let stream = try FileHandle(forWritingTo: destination)
defer { try? stream.close() }
let centerRow = Double(rows - 1) / 2
let centerColumn = Double(columns - 1) / 2
let scale = Double(min(rows, columns)) / 256
for step in 0..<180 {
  let phase = Double(step % 60) / 59 * .pi * 4
  let adf = step < 90
  let shift = adf ? 6.0 : 8.0
  let row = centerRow + shift * scale * sin(phase)
  let column = centerColumn + shift * scale * cos(phase)
  let inner = (adf ? 64.0 : 21.0) * scale
  let outer = (adf ? 124.0 : 85.0) * scale
  let bytes = mask.contents().assumingMemoryBound(to: UInt8.self)
  for pixel in 0..<pixels {
    let dy = Double(pixel / columns) - row
    let dx = Double(pixel % columns) - column
    let squared = dy * dy + dx * dx
    bytes[pixel] = squared >= inner * inner && squared <= outer * outer ? 1 : 0
  }
  let command = queue.makeCommandBuffer()!
  try resident.encodeVirtualImage(mask: mask, into: image, command: command)
  command.commit()
  command.waitUntilCompleted()
  guard command.status == .completed else {
    fatalError("Detector update failed: \(String(describing: command.error))")
  }
  try stream.write(contentsOf: Data(bytes: image.contents(), count: source.frameCount * 4))
}
print("FLOAT_ANS_DETECTOR_PARITY images=180 scan=\(source.scanRows)x\(source.scanColumns) detector=\(rows)x\(columns)")
