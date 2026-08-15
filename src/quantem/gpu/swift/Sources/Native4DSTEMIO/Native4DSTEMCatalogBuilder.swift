import Foundation

public struct Native4DSTEMCatalogBuilder: Sendable {
  public let cacheDirectory: URL

  public init(cacheDirectory: URL) {
    self.cacheDirectory = cacheDirectory
  }

  public func prepare(
    inputs: [URL],
    mode: Native4DSTEMCatalogMode = .indexed
  ) throws -> Native4DSTEMCatalog {
    guard !inputs.isEmpty else { throw Native4DSTEMIOError.noDatasets }
    return try Native4DSTEMCatalog.merging(
      inputs.map { try prepare(input: $0, mode: mode) },
      inputs: inputs
    )
  }

  public func prepare(
    input: URL,
    mode: Native4DSTEMCatalogMode = .indexed
  ) throws -> Native4DSTEMCatalog {
    let source = nativeCanonicalURL(input)
    let datasets = try masters(for: source).map { try prepareDataset(source: $0, mode: mode) }
    guard !datasets.isEmpty else { throw Native4DSTEMIOError.noDatasets }
    return Native4DSTEMCatalog(input: source.path, datasets: datasets)
  }

  private func prepareDataset(
    source: URL,
    mode: Native4DSTEMCatalogMode
  ) throws -> Native4DSTEMDataset {
    let dataFiles = try dataFiles(for: source)
    let missing = dataFiles.filter { !FileManager.default.fileExists(atPath: $0.path) }
    guard missing.isEmpty else {
      throw Native4DSTEMIOError.invalidData(
        "Missing HDF5 shard(s): \(missing.map(\.path).joined(separator: ", "))"
      )
    }
    let signatureFiles = (FileManager.default.fileExists(atPath: source.path) ? [source] : [])
      + dataFiles
    let signature = try nativeDatasetSignature(for: signatureFiles)
    let indexRoot = cacheDirectory.appendingPathComponent(signature, isDirectory: true)
    var indexFiles: [String] = []
    var stacks: [NativeHDF5Stack] = []
    stacks.reserveCapacity(dataFiles.count)
    indexFiles.reserveCapacity(dataFiles.count)

    for dataFile in dataFiles {
      let indexFile = indexRoot.appendingPathComponent(
        dataFile.deletingPathExtension().lastPathComponent + ".qh5idx"
      )
      let stack: NativeHDF5Stack
      if mode == .indexed {
        let metadata = try QH5IndexWriter.prepare(
          source: dataFile,
          destination: indexFile
        )
        stack = NativeHDF5Stack(
          frameCount: metadata.nFrames,
          detectorRows: metadata.detRows,
          detectorColumns: metadata.detCols,
          sourceBytes: metadata.srcDtype == "uint8" ? 1 : 2,
          chunks: []
        )
      } else {
        stack = try NativeHDF5Bridge.inspectStack(
          at: dataFile,
          includeChunks: false
        )
      }
      stacks.append(stack)
      indexFiles.append(indexFile.path)
    }

    guard let first = stacks.first else { throw Native4DSTEMIOError.noDatasets }
    guard stacks.dropFirst().allSatisfy({
      $0.detectorRows == first.detectorRows
        && $0.detectorColumns == first.detectorColumns
        && $0.sourceBytes == first.sourceBytes
    }) else {
      throw Native4DSTEMIOError.invalidData(
        "\(source.lastPathComponent) has inconsistent detector shards"
      )
    }
    let totalFrames = try stacks.reduce(0) { total, stack in
      let (sum, overflow) = total.addingReportingOverflow(stack.frameCount)
      guard !overflow else {
        throw Native4DSTEMIOError.invalidData("The total HDF5 frame count is too large")
      }
      return sum
    }
    let metadataSource = FileManager.default.fileExists(atPath: source.path)
      ? source
      : dataFiles[0]
    let master = try NativeHDF5Bridge.inspectMaster(
      at: metadataSource,
      detectorRows: first.detectorRows,
      detectorColumns: first.detectorColumns
    )
    let scanShape = try scanShape(master: master, totalFrames: totalFrames)
    let sourceBytes = try dataFiles.reduce(0) { total, file in
      let bytes = try nativeFileIdentity(for: file).bytes
      guard let exactBytes = Int(exactly: bytes) else {
        throw Native4DSTEMIOError.invalidData("\(file.lastPathComponent) is too large")
      }
      let (sum, overflow) = total.addingReportingOverflow(exactBytes)
      guard !overflow else {
        throw Native4DSTEMIOError.invalidData("The HDF5 source size is too large")
      }
      return sum
    }
    return Native4DSTEMDataset(
      id: signature,
      label: label(for: source),
      masterPath: isMaster(source) ? source.path : nil,
      dataFiles: dataFiles.map(\.path),
      indexFiles: indexFiles,
      scanRows: scanShape.rows,
      scanCols: scanShape.columns,
      detectorRows: first.detectorRows,
      detectorCols: first.detectorColumns,
      sourceDtype: first.sourceDtype,
      sourceBytes: sourceBytes,
      badPixelIndices: master.badPixelIndices,
      kPixelSizeRow: master.reciprocalSampling?.row,
      kPixelSizeCol: master.reciprocalSampling?.column,
      kPixelUnit: master.reciprocalSampling == nil ? nil : "mrad",
      acquisitionDate: master.acquisitionDate,
      metadata: master.metadata
    )
  }

  private func masters(for input: URL) throws -> [URL] {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: input.path, isDirectory: &isDirectory) else {
      throw Native4DSTEMIOError.invalidData("HDF5 input does not exist: \(input.path)")
    }
    if isDirectory.boolValue {
      guard let enumerator = FileManager.default.enumerator(
        at: input,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
      ) else {
        throw Native4DSTEMIOError.invalidData("Could not inspect folder \(input.path)")
      }
      let masters = enumerator.compactMap { $0 as? URL }
        .filter { isMaster($0) }
        .map(nativeCanonicalURL)
        .sorted { $0.path < $1.path }
      guard !masters.isEmpty else {
        throw Native4DSTEMIOError.invalidData(
          "No recognized *_master.h5 files were found under \(input.path)"
        )
      }
      return masters
    }
    if let stem = shardStem(input.lastPathComponent) {
      let master = input.deletingLastPathComponent()
        .appendingPathComponent(stem + "_master.h5")
      if FileManager.default.fileExists(atPath: master.path) {
        return [nativeCanonicalURL(master)]
      }
    }
    return [input]
  }

  private func dataFiles(for source: URL) throws -> [URL] {
    let directory = source.deletingLastPathComponent()
    if isMaster(source) {
      let suffixLength = "_master.h5".count
      let stem = String(source.lastPathComponent.dropLast(suffixLength))
      let globbed = try FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
      ).filter {
        let name = $0.lastPathComponent.lowercased()
        return name.hasPrefix((stem + "_data_").lowercased()) && name.hasSuffix(".h5")
          && shardStem($0.lastPathComponent) == stem
      }.sorted { $0.lastPathComponent < $1.lastPathComponent }
      if !globbed.isEmpty {
        return globbed.map(nativeCanonicalURL)
      }
      let links = try NativeHDF5Bridge.inspectMaster(
        at: source,
        detectorRows: 0,
        detectorColumns: 0
      ).externalFiles
      if !links.isEmpty {
        return links.map {
          nativeCanonicalURL(URL(fileURLWithPath: $0, relativeTo: directory))
        }
      }
    }
    if let stem = shardStem(source.lastPathComponent) {
      let siblings = try FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
      ).filter { shardStem($0.lastPathComponent) == stem }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
      if !siblings.isEmpty {
        return siblings.map(nativeCanonicalURL)
      }
    }
    return [nativeCanonicalURL(source)]
  }

  private func scanShape(
    master: NativeHDF5Master,
    totalFrames: Int
  ) throws -> (rows: Int, columns: Int) {
    if let shape = master.scanShape,
      shape.rows.multipliedReportingOverflow(by: shape.columns) == (totalFrames, false)
    {
      return shape
    }
    if let expected = master.expectedFrames, expected != totalFrames {
      throw Native4DSTEMIOError.invalidData(
        "Master expects \(expected) scan positions but its shards expose \(totalFrames) frames"
      )
    }
    let side = integerSquareRoot(totalFrames)
    guard side * side == totalFrames else {
      throw Native4DSTEMIOError.invalidData(
        "The scan grid is not square and no exact HDF5 scan_shape=(row, col) attribute is present"
      )
    }
    return (side, side)
  }

  private func integerSquareRoot(_ value: Int) -> Int {
    guard value > 1 else { return value }
    var low = 1
    var high = min(value, 3_037_000_499)
    while low <= high {
      let middle = low + (high - low) / 2
      if middle <= value / middle {
        low = middle + 1
      } else {
        high = middle - 1
      }
    }
    return high
  }

  private func isMaster(_ url: URL) -> Bool {
    url.lastPathComponent.lowercased().hasSuffix("_master.h5")
  }

  private func shardStem(_ filename: String) -> String? {
    let expression = try! NSRegularExpression(
      pattern: #"^(.+)_data_\d{6}\.h5$"#,
      options: [.caseInsensitive]
    )
    let range = NSRange(filename.startIndex..<filename.endIndex, in: filename)
    guard let match = expression.firstMatch(in: filename, range: range),
      let stemRange = Range(match.range(at: 1), in: filename)
    else { return nil }
    return String(filename[stemRange])
  }

  private func label(for source: URL) -> String {
    let suffixLength = isMaster(source) ? "_master.h5".count : 0
    let stem = suffixLength == 0
      ? source.deletingPathExtension().lastPathComponent
      : String(source.lastPathComponent.dropLast(suffixLength))
    let expression = try! NSRegularExpression(
      pattern: #"([+-]?\d+(?:\.\d+)?)x_([+-]?\d+(?:\.\d+)?)y"#
    )
    let range = NSRange(stem.startIndex..<stem.endIndex, in: stem)
    guard let match = expression.firstMatch(in: stem, range: range),
      let rowRange = Range(match.range(at: 1), in: stem),
      let columnRange = Range(match.range(at: 2), in: stem),
      let row = Double(stem[rowRange]),
      let column = Double(stem[columnRange])
    else { return stem }
    return "\(row.formatted(.number.precision(.fractionLength(0...8))))°, \(column.formatted(.number.precision(.fractionLength(0...8))))°"
  }
}
