import Foundation

extension SSBPhaseArtifact {
  /// Stable identity of the complete result, independent of JSON whitespace.
  public func contentDigest() throws -> String {
    var record =
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(self)) as! [String: Any]
    record["runMetadata"] = try JSONSerialization.jsonObject(with: runMetadata)
    return Self.digest(try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]))
  }
  /// Write ordinary NumPy phase data followed by its readable JSON manifest.
  /// Publishing JSON last makes an incomplete copy detectable by either viewer.
  public func savePair(to manifest: URL) throws {
    try validate()
    guard manifest.pathExtension == "json" else {
      throw PairError.invalid("Choose a .json result filename.")
    }
    var record =
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(self)) as! [String: Any]
    let arrayURL = manifest.deletingPathExtension().appendingPathExtension("npy")
    record.removeValue(forKey: "phase")
    record["format"] = "live.ssb"
    record["phaseFile"] = arrayURL.lastPathComponent
    record["runMetadata"] = try JSONSerialization.jsonObject(with: runMetadata)
    let json = try JSONSerialization.data(
      withJSONObject: record, options: [.sortedKeys, .prettyPrinted])
    var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(rows), \(columns)), }"
    header += String(repeating: " ", count: (64 - (10 + header.utf8.count + 1) % 64) % 64) + "\n"
    var array = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0])
    array.append(UInt8(header.utf8.count & 255))
    array.append(UInt8(header.utf8.count >> 8))
    array.append(Data(header.utf8))
    array.append(phase)
    try FileManager.default.createDirectory(
      at: manifest.deletingLastPathComponent(), withIntermediateDirectories: true)
    // Never replace a different scientific result, even if a caller reuses a name.
    for (url, bytes) in [(arrayURL, array), (manifest, json)] {
      if FileManager.default.fileExists(atPath: url.path) {
        guard try Data(contentsOf: url) == bytes else {
          throw PairError.invalid(
            "A different result already uses this name. Choose another result name.")
        }
      } else {
        let temporary = url.deletingLastPathComponent().appendingPathComponent(
          ".ssb-\(UUID().uuidString)")
        try bytes.write(to: temporary, options: .withoutOverwriting)
        defer { try? FileManager.default.removeItem(at: temporary) }
        // Link publishes complete bytes without a check-then-replace race.
        do { try FileManager.default.linkItem(at: temporary, to: url) } catch {
          guard (try? Data(contentsOf: url)) == bytes else { throw error }
        }
      }
    }
  }

  static func loadPair(from manifest: URL) throws -> Self {
    guard
      var record = try JSONSerialization.jsonObject(with: Data(contentsOf: manifest))
        as? [String: Any],
      record["format"] as? String == "live.ssb", record["schemaVersion"] as? Int == 1,
      let filename = record["phaseFile"] as? String, filename.hasSuffix(".npy"),
      !filename.contains("/"), !filename.contains("\\"), !filename.hasPrefix("."),
      let rows = record["rows"] as? Int, let columns = record["columns"] as? Int,
      rows > 0, columns > 0, rows <= 4096, columns <= 4096,
      let metadata = record["runMetadata"] as? [String: Any]
    else {
      throw PairError.invalid("Invalid SSB JSON. Keep the original JSON and NumPy files together.")
    }
    let folder = manifest.deletingLastPathComponent().resolvingSymlinksInPath()
    let arrayURL = folder.appendingPathComponent(filename).resolvingSymlinksInPath()
    guard arrayURL.deletingLastPathComponent() == folder,
      let size = try arrayURL.resourceValues(forKeys: [.fileSizeKey]).fileSize,
      size <= rows * columns * 4 + 65536
    else { throw PairError.invalid("Missing or invalid companion NumPy phase file.") }
    let data = try Data(contentsOf: arrayURL)
    guard data.count >= 10, Array(data.prefix(6)) == [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59],
      data[6] == 1, data[7] == 0
    else { throw PairError.invalid("Expected NumPy v1 float32 phase data.") }
    let start = 10 + Int(data[8]) + 256 * Int(data[9])
    guard start <= data.count, data.count - start == rows * columns * 4,
      let header = String(data: data[10..<start], encoding: .ascii)
    else { throw PairError.invalid("Truncated NumPy phase file.") }
    func matches(_ pattern: String) -> Bool {
      header.range(of: pattern, options: .regularExpression) != nil
    }
    guard matches("['\"]descr['\"]\\s*:\\s*['\"]<f4['\"]"),
      matches("['\"]fortran_order['\"]\\s*:\\s*False"),
      matches("['\"]shape['\"]\\s*:\\s*\\(\\s*\(rows)\\s*,\\s*\(columns)\\s*,?\\s*\\)")
    else {
      throw PairError.invalid(
        "SSB phase must be row-major little-endian float32 with the recorded shape.")
    }
    record["phase"] = data.suffix(from: start).base64EncodedString()
    record["runMetadata"] = try JSONSerialization.data(
      withJSONObject: metadata, options: [.sortedKeys]
    ).base64EncodedString()
    return try JSONDecoder().decode(Self.self, from: JSONSerialization.data(withJSONObject: record))
  }
}

extension MetalSSBSavedRun {
  /// Numerical run metadata shared by native bookmarks and portable phase pairs.
  /// Reconstruction buffers remain private to the native saved run.
  public func portableRunMetadata() throws -> Data {
    var metadata =
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(self)) as! [String: Any]
    for key in ["object", "fourierSum", "objectSHA256", "fourierSHA256"] {
      metadata.removeValue(forKey: key)
    }
    return try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys])
  }

  /// Export the displayed scientific phase, with calibration and fit provenance.
  /// The two private reconstruction buffers are not part of the portable result.
  public func savePhasePair(to url: URL, phase: Data) throws {
    guard let calibration else { throw PairError.invalid("Cannot export SSB without calibration.") }
    let padded = calibrationProvenance?["scanPadding"] == "zero-bottom-right-v1"
    let rows =
      padded ? Int(calibrationProvenance?["sourceScanRows"] ?? "") ?? 0 : provenance.scanRows
    let columns =
      padded ? Int(calibrationProvenance?["sourceScanColumns"] ?? "") ?? 0 : provenance.scanColumns
    guard rows > 0, columns > 0, rows <= provenance.scanRows, columns <= provenance.scanColumns
    else {
      throw PairError.invalid("The saved padding description lacks the original scan dimensions.")
    }
    let artifact = SSBPhaseArtifact(
      schemaVersion: 1, sourceIdentity: sourceIdentity,
      rows: rows, columns: columns,
      phaseEncoding: "float32-le-row-major", phaseUnits: "rad", phase: phase,
      phaseSHA256: SSBPhaseArtifact.digest(phase), calibration: calibration,
      c10Nanometers: Double(aberrations.c10Nanometers),
      c12Nanometers: Double(aberrations.c12Nanometers),
      phi12Radians: Double(aberrations.phi12Radians), rotationDegrees: Double(rotationDegrees),
      provenance: ["producer": "Live4DSTEM", "backendRevision": backendRevision],
      runMetadata: try portableRunMetadata())
    try artifact.savePair(to: url)
  }
}

private enum PairError: LocalizedError {
  case invalid(String)
  var errorDescription: String? {
    if case .invalid(let message) = self { return message }
    return nil
  }
}
