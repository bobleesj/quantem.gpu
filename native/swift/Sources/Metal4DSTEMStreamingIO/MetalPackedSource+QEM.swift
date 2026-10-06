import CryptoKit
import Foundation
import Metal
import Native4DSTEMIO

/// Portable `.qem` storage for calibrated scaled-uint16 results, such as merged tilts.
///
/// The resident ANS code streams are written byte for byte, with the regional intensity
/// calibration, so saving and reopening never re-encode, recalibrate or round again.
/// Example: `try merged.saveQEM(to: url, scientificMetadata: metadata)`, then
/// `try MetalPackedSource.loadQEM(url: url, device: device)`.
extension MetalPackedSource {
  /// Codec name. Readers without this codec refuse the file instead of reading codes as counts.
  public static let qemCodec = "scaled-uint16-column-rans-v1"
  /// The `processing` operation every scaled `.qem` must declare.
  public static let quantizationOperation = "scaled_uint16_quantization"

  /// True for a checksummed QEM envelope that declares the scaled codec.
  public static func isScaledQEM(_ url: URL) -> Bool {
    guard NativeQEMFile.matches(url) else { return false }
    return (try? NativeQEMFile(url: url))?.codec == qemCodec
  }

  /// Save the complete calibrated codes as `.qem`; existing files are never replaced.
  ///
  /// `scientificMetadata` is the public schema-2 record (axes, sampling, processing and
  /// source metadata). It must declare `scaled_uint16_quantization`, which changes
  /// measurements. Application provenance in `attributes` is stored unchanged.
  /// Checksums are computed from the resident codes first, so the file is written once,
  /// without a temporary body copy, and published atomically.
  public func saveQEM(
    to destination: URL, scientificMetadata: [String: Any],
    shouldCancel: () -> Bool = { false }, progress: (Int, Int) -> Void = { _, _ in }
  ) throws {
    let total = shape[0] * shape[1]
    guard !isReleased, !calibrated.isEmpty, !containsPackedChunks, readyFrames == total else {
      throw MetalPrecision.invalid("Save a complete open calibrated result as .qem.")
    }
    guard destination.pathExtension.lowercased() == "qem" else {
      throw MetalPrecision.invalid("Scaled results are written to .qem files only.")
    }
    let manager = FileManager.default
    guard !manager.fileExists(atPath: destination.path) else {
      throw MetalPrecision.invalid(
        "\(destination.lastPathComponent) already exists; choose another name.")
    }
    let report = metadata
    let regions = try Self.regionBounds(report, total: total)
    var header: [String: Any] = [
      "version": 1, "codec": Self.qemCodec, "profile": Self.qemCodec, "interval": 512,
      "shape": shape, "dtype": "uint16", "intensity_calibration": report,
      "attributes": attributes, "scientific_metadata": scientificMetadata,
      "container": NativeQEMMetadata.container, "container_version": 1,
    ]
    // Refuse invalid public metadata before writing any bytes.
    try NativeQEMMetadata.validate(header, shape: shape)
    try Self.requireQuantizationRecord(scientificMetadata)

    // Lay out the body: each array starts on an 8-byte boundary, as in the integer codec.
    let pixels = shape[2] * shape[3]
    let pieces = calibrated.sorted { $0.first < $1.first }.flatMap { part in
      part.source.encodedChunks.map { (first: part.first + $0.first, chunk: $0) }
    }
    var segments = [BodySegment]()
    var table = [[String: Any]]()
    var cursor = 0
    var expected = 0
    for piece in pieces {
      let chunk = piece.chunk
      guard piece.first == expected,
        let region = regions.firstIndex(where: {
          $0.lowerBound <= piece.first && piece.first + chunk.count <= $0.upperBound
        })
      else {
        throw MetalPrecision.invalid("Calibrated regions do not cover the saved frames in order.")
      }
      let streams = ((chunk.count + 511) / 512) * pixels
      let payloadBytes = Int(
        chunk.offsets.contents().load(fromByteOffset: streams * 4, as: UInt32.self))
      var arrays = [[String: Int]]()
      for (buffer, count, itemBytes) in [
        (chunk.payload, payloadBytes, 1), (chunk.offsets, streams + 1, 4),
        (chunk.models, streams, 1),
      ] {
        guard count * itemBytes <= buffer.length else {
          throw MetalPrecision.invalid("Incomplete resident codes; merge or reopen again.")
        }
        let padding = (8 - cursor % 8) % 8
        if padding > 0 {
          segments.append(BodySegment(start: cursor, bytes: nil, count: padding))
          cursor += padding
        }
        arrays.append(["offset": cursor, "count": count])
        if count > 0 {
          segments.append(
            BodySegment(
              start: cursor, bytes: UnsafeRawPointer(buffer.contents()), count: count * itemBytes))
          cursor += count * itemBytes
        }
      }
      table.append(["first": piece.first, "scans": chunk.count, "region": region, "arrays": arrays])
      expected += chunk.count
    }
    guard expected == total, cursor > 0 else {
      throw MetalPrecision.invalid("Calibrated codes do not cover every frame.")
    }
    header["chunks"] = table
    header["bytes"] = cursor
    header["sha256"] = try Self.blockChecksums(segments, bytes: cursor, shouldCancel: shouldCancel)
    let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    guard json.count <= 16 << 20 else {
      throw MetalPrecision.invalid("QEM metadata exceeds 16 MiB; no file was written.")
    }
    var prefix = NativeQEMMetadata.magic
    for number in [json.count, json.count + 56] {
      var value = UInt64(number).littleEndian
      withUnsafeBytes(of: &value) { prefix.append(contentsOf: $0) }
    }
    prefix.append(contentsOf: SHA256.hash(data: json))
    prefix.append(json)

    let directory = destination.deletingLastPathComponent()
    let required = Int64(prefix.count + cursor)
    if let available = try? directory.resourceValues(
      forKeys: [.volumeAvailableCapacityForImportantUsageKey]
    ).volumeAvailableCapacityForImportantUsage, available < required + (256 << 20) {
      throw MetalPrecision.invalid(
        String(
          format: "Not enough free disk space: this .qem needs %.1f GB and %.1f GB is available.",
          Double(required) / 1e9, Double(available) / 1e9))
    }
    let partial = directory.appendingPathComponent(".\(UUID().uuidString).qem-partial")
    defer { try? manager.removeItem(at: partial) }
    guard manager.createFile(atPath: partial.path, contents: nil) else {
      throw MetalPrecision.invalid("Cannot write in \(directory.path); choose a writable folder.")
    }
    let output = try FileHandle(forWritingTo: partial)
    defer { try? output.close() }
    try output.write(contentsOf: prefix)
    let zeros = Data(count: 8)
    for (number, segment) in segments.enumerated() {
      if shouldCancel() { throw MetalPrecision.invalid("Saving cancelled; no file was published.") }
      if let bytes = segment.bytes {
        try output.write(
          contentsOf: Data(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: bytes), count: segment.count,
            deallocator: .none))
      } else {
        try output.write(contentsOf: zeros.prefix(segment.count))
      }
      progress(number + 1, segments.count)
    }
    try output.synchronize()
    if shouldCancel() { throw MetalPrecision.invalid("Saving cancelled; no file was published.") }
    guard link(partial.path, destination.path) == 0 else {
      throw MetalPrecision.invalid(
        "Cannot publish \(destination.lastPathComponent): \(String(cString: strerror(errno))). Existing files were kept."
      )
    }
  }

  /// One contiguous body range: resident bytes, or zero alignment padding when `bytes` is nil.
  struct BodySegment {
    let start: Int
    let bytes: UnsafeRawPointer?
    let count: Int
  }

  /// SHA-256 of each 64 MiB body block, hashed in parallel straight from resident memory.
  static func blockChecksums(
    _ segments: [BodySegment], bytes: Int, shouldCancel: () -> Bool
  ) throws -> [String] {
    let blockBytes = NativeQEMFile.blockBytes
    let blocks = (bytes - 1) / blockBytes + 1
    let zeros = [UInt8](repeating: 0, count: 8)
    nonisolated(unsafe) var digests = [String](repeating: "", count: blocks)
    nonisolated(unsafe) var cancelled = false
    let lock = NSLock()
    let starts = segments.map(\.start)
    // Segments point into resident buffers that stay alive and unchanged while hashing.
    nonisolated(unsafe) let ranges = segments
    let cancel = shouldCancel
    withoutActuallyEscaping(cancel) { cancel in
      nonisolated(unsafe) let check = cancel
      DispatchQueue.concurrentPerform(iterations: blocks) { block in
        if check() {
          lock.lock()
          cancelled = true
          lock.unlock()
          return
        }
        let lower = block * blockBytes
        let upper = min(bytes, lower + blockBytes)
        var hasher = SHA256()
        // First segment that ends after `lower`.
        var index = max(0, (starts.lastIndex { $0 <= lower }) ?? 0)
        while index < ranges.count, ranges[index].start < upper {
          let segment = ranges[index]
          let from = max(lower, segment.start) - segment.start
          let to = min(upper, segment.start + segment.count) - segment.start
          if to > from {
            if let base = segment.bytes {
              hasher.update(
                bufferPointer: UnsafeRawBufferPointer(start: base + from, count: to - from))
            } else {
              zeros.withUnsafeBytes {
                hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: $0[from..<to]))
              }
            }
          }
          index += 1
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        lock.lock()
        digests[block] = digest
        lock.unlock()
      }
    }
    if cancelled { throw MetalPrecision.invalid("Saving cancelled; no file was published.") }
    return digests
  }

  /// Open a scaled `.qem` after verifying every body checksum; codes are not re-encoded.
  public static func loadQEM(
    url: URL, device: MTLDevice, shouldCancel: () -> Bool = { false },
    progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> MetalPackedSource {
    let file = try NativeQEMFile(url: url)
    guard file.codec == qemCodec else {
      throw MetalPrecision.invalid(
        "\(url.lastPathComponent) is not a scaled uint16 result; open it as an acquisition.")
    }
    let header = file.header
    guard header["version"] as? Int == 1, header["interval"] as? Int == 512,
      header["dtype"] as? String == "uint16", let shape = header["shape"] as? [Int],
      shape.count == 4, let table = header["chunks"] as? [[String: Any]], !table.isEmpty,
      var report = header["intensity_calibration"] as? [String: Any],
      var regions = report["regions"] as? [[String: Any]]
    else {
      throw MetalPrecision.invalid("Incomplete scaled .qem header; save the result again.")
    }
    try requireQuantizationRecord(header["scientific_metadata"])
    let total = shape[0] * shape[1]
    let bounds = try regionBounds(report, total: total)
    guard report["source_shape"] as? [Int] == shape else {
      throw MetalPrecision.invalid("Saved calibration and code shapes disagree.")
    }
    let json = try file.headerJSON()
    // JSONSerialization can round decimal text through NSNumber. Decode the
    // scientific coefficients as Double, as the HDF5 reader does.
    struct Coefficients: Decodable {
      // Preserve the persisted precision-report JSON keys.
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let scale, offset, intensity_min, intensity_max, rmse, max_abs_error: Double
    }
    struct Calibration: Decodable { let regions: [Coefficients] }
    struct Envelope: Decodable {
      // swift-format-ignore: AlwaysUseLowerCamelCase
      let intensity_calibration: Calibration
    }
    let coefficients = try JSONDecoder().decode(Envelope.self, from: json)
      .intensity_calibration.regions
    guard coefficients.count == regions.count else {
      throw MetalPrecision.invalid("Saved regional calibration is incomplete.")
    }
    for (index, value) in coefficients.enumerated() {
      regions[index]["scale"] = value.scale
      regions[index]["offset"] = value.offset
      regions[index]["intensity_min"] = value.intensity_min
      regions[index]["intensity_max"] = value.intensity_max
      regions[index]["rmse"] = value.rmse
      regions[index]["max_abs_error"] = value.max_abs_error
    }
    report["regions"] = regions
    let output = try MetalPackedSource(shape: shape, precision: try MetalPrecision(device: device))
    let pixels = shape[2] * shape[3]
    var calibrations = [Int: MetalPrecision]()
    var expected = 0
    struct Planned {
      let first, scans, region, payloadBytes: Int
      let payload, offsets, models: MTLBuffer
    }
    var planned = [Planned]()
    var spans = [NativeQEMFile.BodySpan]()
    for (number, entry) in table.enumerated() {
      if shouldCancel() { throw MetalPrecision.invalid("Opening cancelled.") }
      guard let first = entry["first"] as? Int, let scans = entry["scans"] as? Int,
        let region = entry["region"] as? Int, let arrays = entry["arrays"] as? [[String: Any]],
        arrays.count == 3, first == expected, scans > 0, first + scans <= total,
        bounds.indices.contains(region), bounds[region].lowerBound <= first,
        first + scans <= bounds[region].upperBound
      else {
        throw MetalPrecision.invalid(
          "Scaled .qem chunk \(number) is inconsistent; save the result again.")
      }
      let streams = (scans + 511) / 512 * pixels
      var buffers = [MTLBuffer]()
      var payloadBytes = 0
      for (index, (itemBytes, required)) in [(1, nil), (4, streams + 1), (1, streams)].enumerated()
      {
        guard let offset = arrays[index]["offset"] as? Int,
          let count = arrays[index]["count"] as? Int, offset >= 0, offset % 8 == 0, count >= 0,
          required == nil || count == required, count <= (file.bodyBytes - offset) / itemBytes,
          let buffer = device.makeBuffer(
            length: max(4, count * itemBytes), options: .storageModeShared)
        else {
          throw MetalPrecision.invalid(
            "Scaled .qem chunk \(number) does not fit the file or Metal memory.")
        }
        if count > 0 {
          spans.append(
            NativeQEMFile.BodySpan(
              offset: offset,
              destination: UnsafeMutableRawBufferPointer(
                start: buffer.contents(), count: count * itemBytes)))
        }
        if index == 0 { payloadBytes = count }
        buffers.append(buffer)
      }
      if calibrations[region] == nil {
        let calibration = try MetalPrecision(device: device)
        try calibration.useSavedReport(regions[region])
        calibrations[region] = calibration
      }
      planned.append(
        Planned(
          first: first, scans: scans, region: region, payloadBytes: payloadBytes,
          payload: buffers[0], offsets: buffers[1], models: buffers[2]))
      expected += scans
    }
    // Read every array straight into its Metal buffer while checking all block checksums.
    try file.readVerified(into: spans.sorted { $0.offset < $1.offset }, shouldCancel: shouldCancel)
    // Chunks are independent: validate their stream tables in parallel, then adopt in order.
    nonisolated(unsafe) let work = planned
    nonisolated(unsafe) var sources = [MetalEncodedSource?](repeating: nil, count: work.count)
    nonisolated(unsafe) var failure: Error?
    let lock = NSLock()
    DispatchQueue.concurrentPerform(iterations: work.count) { index in
      let item = work[index]
      do {
        let source = try MetalEncodedSource(
          shape: [1, item.scans, shape[2], shape[3]], device: device)
        try source.adoptEncodedChunk(
          frames: item.scans, payload: item.payload, payloadBytes: item.payloadBytes,
          offsets: item.offsets, models: item.models)
        lock.lock()
        sources[index] = source
        lock.unlock()
      } catch {
        lock.lock()
        failure = failure ?? error
        lock.unlock()
      }
    }
    if let failure { throw failure }
    for (item, source) in zip(work, sources) {
      guard let source, let calibration = calibrations[item.region] else {
        throw MetalPrecision.invalid("Scaled .qem chunks could not be opened.")
      }
      output.calibrated.append(
        CalibratedPart(first: item.first, source: source, precision: calibration))
      output.readyFrames = item.first + item.scans
    }
    output.peakAllocatedBytes = max(output.peakAllocatedBytes, device.currentAllocatedSize)
    progress(table.count, table.count)
    guard expected == total, pixels > 0 else {
      throw MetalPrecision.invalid("Scaled .qem chunks do not cover every frame.")
    }
    output.savedMetadata = report
    output.attributes = header["attributes"] as? [String: String] ?? [:]
    output.qemScientificMetadata = header["scientific_metadata"] as? [String: Any]
    return output
  }

  /// Contiguous regions covering every frame, from a complete version-2 report.
  static func regionBounds(_ report: [String: Any], total: Int) throws -> [Range<Int>] {
    guard report["storage"] as? String == "scaled_uint16", report["version"] as? Int == 2,
      report["complete"] as? Bool == true, let regions = report["regions"] as? [[String: Any]],
      !regions.isEmpty
    else {
      throw MetalPrecision.invalid("A complete regional scaled_uint16 report is required.")
    }
    var bounds = [Range<Int>]()
    for region in regions {
      guard let first = region["first_frame"] as? Int, let stop = region["stop_frame"] as? Int,
        first == (bounds.last?.upperBound ?? 0), stop > first
      else {
        throw MetalPrecision.invalid("Saved calibration regions must be contiguous.")
      }
      bounds.append(first..<stop)
    }
    guard bounds.last?.upperBound == total else {
      throw MetalPrecision.invalid("Saved calibration regions must cover every frame.")
    }
    return bounds
  }

  static func requireQuantizationRecord(_ scientific: Any?) throws {
    guard let processing = (scientific as? [String: Any])?["processing"] as? [[String: Any]],
      processing.contains(where: {
        $0["operation"] as? String == quantizationOperation
          && ($0["changes_measurements"] as? NSNumber)?.boolValue == true
      })
    else {
      throw MetalPrecision.invalid(
        "Scaled .qem metadata must declare \(quantizationOperation), which changes measurements.")
    }
  }
}
