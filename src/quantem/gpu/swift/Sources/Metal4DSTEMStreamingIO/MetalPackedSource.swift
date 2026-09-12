import CNativeHDF5
import Foundation
import Metal
import Native4DSTEMIO

/// Complete scaled uint16 output held in losslessly packed GPU streams.
/// Bounded reads restore float32 intensities using the persisted global scale.
public final class MetalPackedSource {
  public let shape: [Int]
  public let precision: MetalPrecision
  public private(set) var readyFrames = 0
  public private(set) var isReleased = false
  public private(set) var peakAllocatedBytes = 0
  public var metadata: [String: Any] { precision.report }
  private struct Chunk {
    let first, count: Int
    let words, offsets, widths: MTLBuffer
  }
  private var chunks: [Chunk] = []
  public var residentBytes: Int {
    chunks.reduce(0) { $0 + $1.words.length + $1.offsets.length + $1.widths.length }
  }

  public init(shape: [Int], precision: MetalPrecision) throws {
    guard shape.count == 4, shape.allSatisfy({ $0 > 0 }) else {
      throw MetalPrecision.invalid("A positive 4D shape is required.")
    }
    self.shape = shape
    self.precision = precision
  }
  public static func load(path: URL, device: MTLDevice, indexDirectory: URL) throws
    -> MetalPackedSource
  {
    guard let text = qh5_read_root_attribute(path.path, "quantem_precision_v1") else {
      throw MetalPrecision.invalid(
        "This file has no saved precision report; choose a complete scaled uint16 export.")
    }
    let json = String(cString: text)
    qh5_free_error(text)
    let bytes = Data(json.utf8)
    guard var metadata = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
      throw MetalPrecision.invalid("Cannot read the saved precision report.")
    }
    // JSONSerialization bridges decimal numbers through NSNumber, which can
    // introduce double rounding. Decode scientific coefficients as Double.
    struct Scalars: Decodable {
      let scale, offset, intensity_min, intensity_max, rmse, max_abs_error: Double
    }
    let scalars = try JSONDecoder().decode(Scalars.self, from: bytes)
    metadata["scale"] = scalars.scale
    metadata["offset"] = scalars.offset
    metadata["intensity_min"] = scalars.intensity_min
    metadata["intensity_max"] = scalars.intensity_max
    metadata["rmse"] = scalars.rmse
    metadata["max_abs_error"] = scalars.max_abs_error
    let precision = try MetalPrecision(device: device)
    try precision.useSavedReport(metadata)
    let prepared = try Native4DSTEMCatalogBuilder(cacheDirectory: indexDirectory).prepare(
      input: path)
    guard prepared.datasets.count == 1 else {
      throw MetalPrecision.invalid("Choose one 4D export.")
    }
    let source = try Native4DSTEMIndexedSource.open(dataset: prepared.datasets[0])
    let d = source.dataset
    let shape = [d.scanRows, d.scanCols, d.detectorRows, d.detectorCols]
    guard source.sourceBytesPerValue == 2, metadata["source_shape"] as? [Int] == shape else {
      throw MetalPrecision.invalid(
        "The saved precision shape or uint16 storage does not match its metadata.")
    }
    let result = try MetalPackedSource(shape: shape, precision: precision)
    try MetalHDF5Reader.read(source: source, device: device) { codes, frames in
      try result.append(codes, frames: frames.count)
    }
    return result
  }
  /// Pack consecutive uint16 codes exactly, without any further scaling.
  public func append(_ codes: MTLBuffer, frames: Int) throws {
    guard !isReleased, readyFrames + frames <= shape[0] * shape[1] else {
      throw MetalPrecision.invalid("Append within an open packed source.")
    }
    let pixels = shape[2] * shape[3]
    let streams = ((frames + 127) / 128) * pixels
    try precision.validate(codes, count: frames * pixels, bytes: 2)
    let widths = try precision.buffer(streams)
    let lengths = try precision.buffer(streams * 8)
    let offsets = try precision.buffer((streams + 1) * 8)
    let groups = (streams + 255) / 256
    let totals = try precision.buffer(groups * 8)
    var p = precision.parameters(streams)
    p[1] = UInt64(pixels)
    p[2] = UInt64(frames)
    try precision.precision("precision_widths", [codes, widths, lengths], p: p, count: streams)
    let command = try precision.command()
    let local = try precision.encoder(command, "native_prefix64", [lengths, offsets, totals])
    var count = UInt32(streams)
    local.setBytes(&count, length: 4, index: 3)
    precision.dispatch(local, count: streams, groups: true)
    let top = try precision.encoder(command, "native_prefix64_totals", [totals, offsets])
    var groupCounts = SIMD2<UInt32>(UInt32(groups), UInt32(streams))
    top.setBytes(&groupCounts, length: 8, index: 2)
    precision.dispatch(top, count: 1)
    let add = try precision.encoder(command, "native_prefix64_add", [offsets, totals])
    add.setBytes(&count, length: 4, index: 2)
    precision.dispatch(add, count: streams)
    try precision.complete(command)
    let wordCount = Int(offsets.contents().load(fromByteOffset: streams * 8, as: UInt64.self))
    let words = try precision.buffer(max(4, wordCount * 4))
    try precision.precision(
      "precision_pack", [codes, widths, offsets, words], p: p, count: streams)
    chunks.append(
      Chunk(first: readyFrames, count: frames, words: words, offsets: offsets, widths: widths))
    readyFrames += frames
    peakAllocatedBytes = max(peakAllocatedBytes, precision.device.currentAllocatedSize)
  }
  /// Restore a bounded consecutive frame region directly into a float32 GPU buffer.
  public func read(_ frames: Range<Int>) throws -> MTLBuffer {
    guard !isReleased, !frames.isEmpty, frames.lowerBound >= 0, frames.upperBound <= readyFrames,
      frames.count <= 4096
    else {
      throw MetalPrecision.invalid("Read at most 4096 available frames from an open packed source.")
    }
    let pixels = shape[2] * shape[3]
    let result = try precision.buffer(frames.count * pixels * 4)
    for chunk in chunks {
      let first = max(frames.lowerBound, chunk.first)
      let stop = min(frames.upperBound, chunk.first + chunk.count)
      if first >= stop { continue }
      var p = precision.parameters((stop - first) * pixels)
      p[1] = UInt64(pixels)
      p[8] = UInt64(first - chunk.first)
      p[9] = UInt64(first - frames.lowerBound)
      try precision.precision(
        "native_packed_read", [chunk.words, chunk.offsets, chunk.widths, result], p: p,
        count: (stop - first) * pixels)
    }
    return result
  }
  public func releaseResidentStorage() {
    chunks.removeAll()
    isReleased = true
  }
}
