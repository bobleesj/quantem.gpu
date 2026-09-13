import CNativeHDF5
import Foundation
import Metal
import Native4DSTEMIO

/// Scaled uint16 output held in lossless GPU streams.
/// Calibrated appends use ANS; bounded reads restore the saved intensity scale.
public final class MetalPackedSource {
  public let shape: [Int]
  public let precision: MetalPrecision
  public private(set) var readyFrames = 0
  public private(set) var isReleased = false
  public private(set) var peakAllocatedBytes = 0
  /// Application provenance copied into exports; precision metadata is managed internally.
  public var attributes: [String: String] = [:]
  private var savedMetadata: [String: Any]?
  public var metadata: [String: Any] {
    if let savedMetadata { return savedMetadata }
    guard !calibrated.isEmpty else { return precision.report }
    let reports = calibrated.map { $0.precision.report }
    let count = reports.reduce(0.0) { $0 + ($1["values"] as! NSNumber).doubleValue }
    let squared = reports.reduce(0.0) {
      $0 + pow($1["rmse"] as! Double, 2) * ($1["values"] as! NSNumber).doubleValue
    }
    var result: [String: Any] = [
      "version": 2, "storage": "scaled_uint16", "source_dtype": "float32",
      "source_shape": shape,
      "regions": calibrated.map { part -> [String: Any] in
        var report = part.precision.report
        report["first_frame"] = part.first
        report["stop_frame"] = part.first + part.source.readyFrames
        return report
      }, "values": count, "rmse": sqrt(squared / max(1, count)),
      "max_abs_error": reports.map { $0["max_abs_error"] as! Double }.max() ?? 0,
      "range_scope": "automatic regions", "complete": readyFrames == shape[0] * shape[1],
    ]
    for key in ["changed", "positive_to_zero", "overflow", "clipped"] {
      result[key] = reports.reduce(UInt64(0)) { $0 + (($1[key] as? NSNumber)?.uint64Value ?? 0) }
    }
    result["intensity_min"] = reports.map { ($0["intensity_min"] as! NSNumber).doubleValue }.min()!
    result["intensity_max"] = reports.map { ($0["intensity_max"] as! NSNumber).doubleValue }.max()!
    return result
  }
  private struct CalibratedPart {
    let first: Int
    let source: MetalEncodedSource
    let precision: MetalPrecision
  }
  private var calibrated: [CalibratedPart] = []
  private struct Chunk {
    let first, count: Int
    let words, offsets, widths: MTLBuffer
  }
  private var chunks: [Chunk] = []
  public var residentBytes: Int {
    calibrated.reduce(0) { $0 + $1.source.residentBytes }
      + chunks.reduce(0) { $0 + $1.words.length + $1.offsets.length + $1.widths.length }
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
    if (metadata["version"] as? Int) == 2 {
      guard let shape = metadata["source_shape"] as? [Int],
        let regions = metadata["regions"] as? [[String: Any]], !regions.isEmpty
      else {
        throw MetalPrecision.invalid("Saved regional calibration is incomplete.")
      }
      let runtime = try MetalPrecision(device: device)
      let output = try MetalPackedSource(shape: shape, precision: runtime)
      let prepared = try Native4DSTEMCatalogBuilder(cacheDirectory: indexDirectory).prepare(
        input: path)
      guard prepared.datasets.count == 1 else {
        throw MetalPrecision.invalid("Choose one 4D export.")
      }
      let input = try Native4DSTEMIndexedSource.open(dataset: prepared.datasets[0])
      guard input.sourceBytesPerValue == 2,
        [
          input.dataset.scanRows, input.dataset.scanCols, input.dataset.detectorRows,
          input.dataset.detectorCols,
        ] == shape
      else {
        throw MetalPrecision.invalid("Saved regional shape and storage disagree.")
      }
      struct Coefficients: Decodable {
        let scale, offset, intensity_min, intensity_max, rmse, max_abs_error: Double
      }
      struct Envelope: Decodable { let regions: [Coefficients] }
      let coefficients = try JSONDecoder().decode(Envelope.self, from: bytes).regions
      var index = 0
      var first = 0
      try MetalHDF5Reader.read(source: input, device: device) { buffer, frames in
        var cursor = 0
        while cursor < frames.count {
          guard index < regions.count,
            let begin = regions[index]["first_frame"] as? Int,
            let end = regions[index]["stop_frame"] as? Int,
            begin <= first, first < end
          else {
            throw MetalPrecision.invalid("Saved regions do not cover the requested frames.")
          }
          let count = min(frames.count - cursor, end - first)
          let codes = try runtime.buffer(count * shape[2] * shape[3] * 2)
          let command = try runtime.command()
          let copy = command.makeBlitCommandEncoder()!
          copy.copy(
            from: buffer, sourceOffset: cursor * shape[2] * shape[3] * 2,
            to: codes, destinationOffset: 0, size: codes.length)
          copy.endEncoding()
          try runtime.complete(command)
          let calibration = try MetalPrecision(device: device)
          var saved = regions[index]
          let scalars = coefficients[index]
          saved["scale"] = scalars.scale
          saved["offset"] = scalars.offset
          saved["intensity_min"] = scalars.intensity_min
          saved["intensity_max"] = scalars.intensity_max
          saved["rmse"] = scalars.rmse
          saved["max_abs_error"] = scalars.max_abs_error
          try calibration.useSavedReport(saved)
          try output.append(codes, frames: count, calibration: calibration)
          cursor += count
          first += count
          if first == end { index += 1 }
        }
      }
      guard first == shape[0] * shape[1], index == regions.count else {
        throw MetalPrecision.invalid("Saved regional calibration is incomplete.")
      }
      output.savedMetadata = metadata
      return output
    }
    // JSONSerialization bridges decimal numbers through NSNumber, which can
    // introduce double rounding. Decode scientific coefficients as Double.
    struct Scalars: Decodable {
      // Retain the existing precision-report JSON keys.
      // swift-format-ignore: AlwaysUseLowerCamelCase
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
  /// Retain calibrated uint16 codes in ANS storage without changing their values.
  public func append(_ codes: MTLBuffer, frames: Int, calibration: MetalPrecision) throws {
    guard !isReleased, chunks.isEmpty, frames > 0,
      readyFrames + frames <= shape[0] * shape[1],
      calibration.device.registryID == precision.device.registryID
    else {
      throw MetalPrecision.invalid(
        "Append calibrated regions on the same device within the source shape.")
    }
    let source = try MetalEncodedSource(
      shape: [1, frames, shape[2], shape[3]], device: precision.device)
    try source.append(codes, frames: frames)
    calibrated.append(CalibratedPart(first: readyFrames, source: source, precision: calibration))
    readyFrames += frames
    peakAllocatedBytes = max(peakAllocatedBytes, precision.device.currentAllocatedSize)
  }

  /// Save already calibrated codes without reconstructing or rescaling them.
  public func save(to path: URL, metadata attributes: [String: String] = [:]) throws {
    guard !isReleased, !calibrated.isEmpty, readyFrames == shape[0] * shape[1] else {
      throw MetalPrecision.invalid("Save a complete open calibrated source.")
    }
    let writer = try MetalHDF5Writer(path: path, shape: shape, runtime: precision)
    for part in calibrated {
      try autoreleasepool {
        let codes = try part.source.read(0..<part.source.readyFrames)
        try writer.append(codes, frames: part.source.readyFrames)
      }
    }
    var attributes = self.attributes.merging(attributes) { _, supplied in supplied }
    attributes["quantem_precision_v1"] = String(
      data: try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys]),
      encoding: .utf8)!
    try writer.finish(metadata: attributes)
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
    for part in calibrated {
      let first = max(frames.lowerBound, part.first)
      let stop = min(frames.upperBound, part.first + part.source.readyFrames)
      if first >= stop { continue }
      let codes = try part.source.read((first - part.first)..<(stop - part.first))
      let restored = try part.precision.restore(codes, count: (stop - first) * pixels)
      let command = try precision.command()
      guard let copy = command.makeBlitCommandEncoder() else {
        throw MetalPrecision.invalid("Cannot create a calibrated region copy.")
      }
      copy.copy(
        from: restored, sourceOffset: 0, to: result,
        destinationOffset: (first - frames.lowerBound) * pixels * 4,
        size: (stop - first) * pixels * 4)
      copy.endEncoding()
      try precision.complete(command)
    }
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
    for part in calibrated { part.source.releaseResidentStorage() }
    calibrated.removeAll()
    chunks.removeAll()
    isReleased = true
  }
}
