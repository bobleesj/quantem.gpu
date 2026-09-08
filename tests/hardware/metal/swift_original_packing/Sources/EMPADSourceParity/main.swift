import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Emit source bits, not decimal float strings, for an independent file oracle.
let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.count >= 3 else {
  fatalError("usage: EMPADSourceParity input output all|frame[,frame] [scan_rows scan_cols]")
}
let shape: (row: Int, col: Int)? =
  arguments.count == 5
  ? (Int(arguments[3])!, Int(arguments[4])!) : nil
let source = try NativeEMPADSource.open(URL(fileURLWithPath: arguments[0]), scanShape: shape)
let metadata: [String: Any] = [
  "scanRowAngstrom": source.scanCalibration?.rowSamplingAngstrom as Any? ?? NSNull(),
  "scanColumnAngstrom": source.scanCalibration?.columnSamplingAngstrom as Any? ?? NSNull(),
  "diffractionInverseNanometers": source.diffractionSamplingInverseNanometers as Any? ?? NSNull(),
  "acquisitionDate": source.acquisitionDate as Any? ?? NSNull(),
]
try JSONSerialization.data(withJSONObject: metadata).write(
  to: URL(fileURLWithPath: arguments[1] + ".metadata.json"))
if let mutation = ProcessInfo.processInfo.environment["EMPAD_TEST_MUTATE"] {
  let target = mutation == "xml" ? source.metadataURL! : source.rawURL
  let attributes = try FileManager.default.attributesOfItem(atPath: target.path)
  let handle = try FileHandle(forWritingTo: target)
  try handle.write(contentsOf: Data([0]))
  try handle.close()
  // A same-size edit with restored mtime must still invalidate the snapshot.
  try FileManager.default.setAttributes(
    [.modificationDate: attributes[.modificationDate]!], ofItemAtPath: target.path)
}
let indices = arguments[2] == "all" ? Array(0..<source.frameCount)
  : arguments[2].split(separator: ",").map { Int($0)! }
let outputURL = URL(fileURLWithPath: arguments[1])
FileManager.default.createFile(atPath: outputURL.path, contents: nil)
let outputHandle = try FileHandle(forWritingTo: outputURL)
defer { try? outputHandle.close() }
if ProcessInfo.processInfo.environment["EMPAD_TEST_METAL"] == "1" {
  let device = MTLCreateSystemDefaultDevice()!
  let queue = device.makeCommandQueue()!
  let budget = UInt64(ProcessInfo.processInfo.environment["EMPAD_TEST_BUDGET"] ?? "268435456")!
  if let rawPoint = ProcessInfo.processInfo.environment["EMPAD_TEST_CANCEL_AT"],
    let point = Int(rawPoint)
  {
    var checks = 0
    do {
      _ = try MetalEMPADResidentSource.load(
        source, device: device, memoryBudgetBytes: budget,
        shouldCancel: {
          checks += 1
          return checks == point
        })
      fatalError("Cancelled EMPAD load returned a resident")
    } catch is CancellationError {
      print("EMPAD_CANCELLED checks=\(checks)")
    }
    // Complete the same load and parity checks immediately after cancellation.
  }
  let resident = try MetalEMPADResidentSource.load(
    source, device: device, memoryBudgetBytes: budget,
    sourceHashCacheURL: ProcessInfo.processInfo.environment["EMPAD_TEST_HASH_CACHE"].map { URL(fileURLWithPath: $0) })
  print("EMPAD_SOURCE_HASH cached=\(resident.reusedSourceHash ? 1 : 0)")
  let capabilities = try Metal4DSTEMResidentCapabilities.empad(resident)
  try capabilities.residentReceipt.validate()
  guard capabilities.fullInteractiveResident, capabilities.completeSourceResident else {
    fatalError("Complete EMPAD resident prerequisites were not advertised")
  }
  try JSONEncoder().encode(capabilities).write(
    to: URL(fileURLWithPath: arguments[1] + ".capabilities.json"))
  let buffer = device.makeBuffer(length: 16384 * 4, options: .storageModeShared)!
  for index in indices {
    let command = queue.makeCommandBuffer()!
    try resident.encodeDiffraction(
      scanRow: index / source.scanColumns,
      scanColumn: index % source.scanColumns, into: buffer, command: command)
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      fatalError("Diffraction command failed: \(String(describing: command.error))")
    }
    // Stream one exact pattern at a time: auditing a large acquisition must
    // not materialize a second dense 4D tensor in host memory.
    try outputHandle.write(contentsOf: Data(bytes: buffer.contents(), count: 16384 * 4))
  }
  var products = Data()
  let alias = device.makeBuffer(
    length: max(16384, source.frameCount * 4), options: .storageModeShared)!
  do {
    try resident.encodeVirtualImage(mask: alias, into: alias, command: queue.makeCommandBuffer()!)
    fatalError("Mask/output alias accepted")
  } catch {}
  for kind in 0..<4 {
    let mask = device.makeBuffer(length: 16384, options: .storageModeShared)!
    let bytes = mask.contents().assumingMemoryBound(to: UInt8.self)
    for pixel in 0..<16384 {
      let row = pixel / 128 - 64
      let col = pixel % 128 - 64
      let radiusSquared = row * row + col * col
      let included: Bool
      switch kind {
      case 0: included = radiusSquared <= 16 * 16
      case 1: included = radiusSquared >= 8 * 8 && radiusSquared <= 16 * 16
      case 2: included = radiusSquared >= 32 * 32 && radiusSquared <= 63 * 63
      default: included = true
      }
      bytes[pixel] = included ? 1 : 0
    }
    let output = device.makeBuffer(length: source.frameCount * 4, options: .storageModeShared)!
    let command = queue.makeCommandBuffer()!
    try resident.encodeVirtualImage(mask: mask, into: output, command: command)
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      fatalError("Detector command failed: \(String(describing: command.error))")
    }
    products.append(Data(bytes: output.contents(), count: output.length))
  }
  try products.write(to: URL(fileURLWithPath: arguments[1] + ".products"))
  let mean = device.makeBuffer(length: 16384 * 4, options: .storageModeShared)!
  let comRow = device.makeBuffer(length: source.frameCount * 4, options: .storageModeShared)!
  let comColumn = device.makeBuffer(length: source.frameCount * 4, options: .storageModeShared)!
  let derivedCommand = queue.makeCommandBuffer()!
  try resident.encodeMeanDiffraction(into: mean, command: derivedCommand)
  try resident.encodeCenterOfMass(intoRow: comRow, intoColumn: comColumn, command: derivedCommand)
  derivedCommand.commit()
  derivedCommand.waitUntilCompleted()
  guard derivedCommand.status == .completed else { fatalError("EMPAD derived command failed") }
  for (name, buffer) in [("mean", mean), ("com-row", comRow), ("com-column", comColumn)] {
    try Data(bytes: buffer.contents(), count: buffer.length).write(
      to: URL(fileURLWithPath: arguments[1] + "." + name))
  }
  if ProcessInfo.processInfo.environment["EMPAD_TEST_APERTURE_SEQUENCE"] == "1" {
    let mask = device.makeBuffer(length: 16384, options: .storageModeShared)!
    let image = device.makeBuffer(length: source.frameCount * 4, options: .storageModeShared)!
    let destination = URL(fileURLWithPath: arguments[1] + ".apertures")
    FileManager.default.createFile(atPath: destination.path, contents: nil)
    let stream = try FileHandle(forWritingTo: destination)
    defer { try? stream.close() }
    var masks = Data()
    for step in 0..<96 {
      let values = mask.contents().assumingMemoryBound(to: UInt8.self)
      let centerRow = 64.0 + 4.0 * sin(Double(step) * 0.2)
      let centerColumn = 64.0 + 4.0 * cos(Double(step) * 0.2)
      let inner = 24.0 + 2.0 * sin(Double(step) * 0.3)
      let outer = 60.0 + 2.0 * cos(Double(step) * 0.3)
      for pixel in 0..<16384 {
        let row = Double(pixel / 128) - centerRow
        let col = Double(pixel % 128) - centerColumn
        let square = row * row + col * col
        values[pixel] = step == 0 ? 1 : step == 1 ? (pixel < 8 ? 0 : 1)
          : (square >= inner * inner && square <= outer * outer ? 1 : 0)
      }
      let command = queue.makeCommandBuffer()!
      try resident.encodeVirtualImage(mask: mask, into: image, command: command)
      command.commit(); command.waitUntilCompleted()
      guard command.status == .completed else { fatalError("EMPAD aperture sequence command failed") }
      try stream.write(contentsOf: Data(bytes: image.contents(), count: image.length))
      masks.append(Data(bytes: mask.contents(), count: 16384))
    }
    try masks.write(to: URL(fileURLWithPath: arguments[1] + ".aperture-masks"))
    print("EMPAD_APERTURE_SEQUENCE steps=96")
  }
  print("EMPAD_METAL_PARITY device=\(device.name) resident_bytes=\(resident.residentBytes)")
  resident.releaseResidentStorage()
  do {
    _ = try Metal4DSTEMResidentCapabilities.empad(resident)
    fatalError("Released resident advertised source availability")
  } catch {}
  do {
    try resident.encodeDiffraction(
      scanRow: 0, scanColumn: 0, into: buffer,
      command: queue.makeCommandBuffer()!)
    fatalError("Released resident accepted an interaction")
  } catch {}
} else {
  let values = try source.readFrames(indices)
  var output = Data(capacity: values.count * 4)
  for value in values {
    var bits = value.bitPattern.littleEndian
    withUnsafeBytes(of: &bits) { output.append(contentsOf: $0) }
  }
  try outputHandle.write(contentsOf: output)
}
print(
  "EMPAD_SOURCE_PARITY scan=\(source.scanRows)x\(source.scanColumns) detector=128x128 dtype=float32 frames=\(indices.count) bytes=\(indices.count * 16384 * 4)"
)
