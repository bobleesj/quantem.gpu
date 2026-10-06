import CryptoKit
import Foundation
import Metal
@testable import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// This executable uses the production decoder with test visibility, not the
// diagnostic compilation flag. The caller compares independent GPU modes.
let input = URL(fileURLWithPath: CommandLine.arguments[1])
let cache = URL(fileURLWithPath: CommandLine.arguments[2])
let dataset = try Native4DSTEMCatalogBuilder(cacheDirectory: cache).prepare(input: input).datasets[0]
let source = try Native4DSTEMIndexedSource.open(dataset: dataset)
let device = MTLCreateSystemDefaultDevice()!
let packing = try OriginalHDF5Packing(device: device)
let frames = source.logicalFrameCount
let windows = try source.windows(
  maximumDecodedBytes: UInt64(frames) * source.decodedBytesPerFrame, alignToScanRows: false)
precondition(windows.count == 1)
precondition(windows[0].slices.map { $0.globalFrameRange.count } == [2048, 97])
let sampleFrames = [0, 1, 2047, 2048, frames - 2, frames - 1]
var sampleHashes: [String] = []
var momentsDigest = ""
var denseDigest = ""
try packing.forEachExactDecodedWindow(
  source: source, maximumFrames: frames, includeDPCMoments: true,
  shouldCancel: { false }, progress: { _, _ in }
) { dense, moments, range, command in
  let bytes = range.count * Int(source.decodedBytesPerFrame)
  let staging = device.makeBuffer(length: bytes, options: .storageModeShared)!
  let blit = command.makeBlitCommandEncoder()!
  blit.copy(from: dense, sourceOffset: 0, to: staging, destinationOffset: 0, size: bytes)
  blit.endEncoding()
  command.commit()
  command.waitUntilCompleted()
  precondition(command.status == .completed, "GPU window failed")
  func digest(_ buffer: MTLBuffer, offset: Int = 0, count: Int) -> String {
    SHA256.hash(data: Data(bytes: buffer.contents() + offset, count: count))
      .map { String(format: "%02x", $0) }.joined()
  }
  denseDigest = digest(staging, count: bytes)
  momentsDigest = digest(moments!, count: range.count * 32)
  sampleHashes = sampleFrames.map {
    digest(staging, offset: $0 * Int(source.decodedBytesPerFrame),
           count: Int(source.decodedBytesPerFrame))
  }
}
let output: [String: Any] = [
  "sample_frames": sampleFrames, "sample_hashes_u16_le": sampleHashes,
  "dense_sha256": denseDigest, "moments_sha256": momentsDigest,
  "slice_frames": windows[0].slices.map { $0.globalFrameRange.count },
]
print(String(data: try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]),
             encoding: .utf8)!)
