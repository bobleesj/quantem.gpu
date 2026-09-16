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
    precondition(numpy.count - start == source.frameCount * 65536)
    let output = device.makeBuffer(length: 65536, options: .storageModeShared)!
    let queue = device.makeCommandQueue()!
    for frame in 0..<source.frameCount {
      let command = queue.makeCommandBuffer()!
      try resident.encodeDiffraction(scanRow: frame / source.scanColumns,
        scanColumn: frame % source.scanColumns, into: output, command: command)
      command.commit()
      command.waitUntilCompleted()
      precondition(command.status == .completed)
      let actual = Data(bytes: output.contents(), count: 65536)
      precondition(actual == numpy.subdata(in: start + frame * 65536..<start + (frame + 1) * 65536))
    }
    print("PASS native Metal: every float32 bit, including signed zero, NaN payload and infinity")
  }
}
