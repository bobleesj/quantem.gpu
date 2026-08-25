import Darwin
import Foundation

public struct Native4DSTEMCatalogBuilder: Sendable {
  public let cacheDirectory: URL

  public init(cacheDirectory: URL) {
    self.cacheDirectory = cacheDirectory
  }

  public func resolvedAcquisitionInput(_ input: URL) throws -> URL {
    let source = nativeCanonicalURL(input)
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: source.path, isDirectory: &isDirectory) else {
      throw Native4DSTEMIOError.invalidData("HDF5 input does not exist: \(source.path)")
    }
    guard !isDirectory.boolValue else { return source }
    return try masters(for: source).first ?? source
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
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: source.path, isDirectory: &isDirectory) else {
      throw Native4DSTEMIOError.invalidData("HDF5 input does not exist: \(source.path)")
    }
    let candidates = try masters(for: source)
    var datasets: [Native4DSTEMDataset] = []
    var issues: [Native4DSTEMCatalogIssue] = []
    datasets.reserveCapacity(candidates.count)
    for candidate in candidates {
      do {
        datasets.append(try prepareDataset(source: candidate, mode: mode))
      } catch {
        guard isDirectory.boolValue else { throw error }
        issues.append(
          Native4DSTEMCatalogIssue(
            input: candidate.path,
            message: error.localizedDescription
          ))
      }
    }
    guard !datasets.isEmpty else { throw Native4DSTEMIOError.noDatasets }
    return Native4DSTEMCatalog(input: source.path, datasets: datasets, issues: issues)
  }

  private func prepareDataset(
    source: URL,
    mode: Native4DSTEMCatalogMode
  ) throws -> Native4DSTEMDataset {
    if isEMD(source) {
      return try prepareVeloxDataset(source: source, mode: mode)
    }
    let dataFiles = try dataFiles(for: source)
    let missing = dataFiles.filter { !FileManager.default.fileExists(atPath: $0.path) }
    guard missing.isEmpty else {
      throw Native4DSTEMIOError.invalidData(
        "Missing HDF5 shard(s): \(missing.map(\.path).joined(separator: ", "))"
      )
    }
    let signatureFiles =
      (FileManager.default.fileExists(atPath: source.path) ? [source] : [])
      + dataFiles
    let signature = try nativeDatasetSignature(for: signatureFiles)
    let indexRoot = cacheDirectory.appendingPathComponent(signature, isDirectory: true)
    let datasetCache = indexRoot.appendingPathComponent("dataset.json")
    if let cached = try cachedDataset(
      at: datasetCache,
      signature: signature,
      source: source,
      dataFiles: dataFiles
    ) {
      if mode == .catalogOnly {
        return cached
      }
      let indexesCurrent = try zip(dataFiles, cached.indexFiles).allSatisfy {
        try QH5IndexWriter.currentMetadata(
          source: $0.0,
          destination: URL(fileURLWithPath: $0.1)
        ) != nil
      }
      if cached.sourceIdentitySHA256 != nil,
        cached.indexFiles.count == dataFiles.count,
        indexesCurrent
      {
        return cached
      }
    }
    let indexURLs = dataFiles.map {
      indexRoot.appendingPathComponent(
        $0.deletingPathExtension().lastPathComponent + ".qh5idx"
      )
    }
    var hashes: NativeSourceHashes?
    var indexFiles: [String] = []
    var stacks: [NativeHDF5Stack] = []
    stacks.reserveCapacity(dataFiles.count)
    indexFiles.reserveCapacity(dataFiles.count)

    if mode == .indexed {
      let evidence = try prepareIndexedEvidence(
        master: isMaster(source) ? source : nil,
        dataFiles: dataFiles,
        indexRoot: indexRoot,
        indexFiles: indexURLs
      )
      hashes = evidence.hashes
      stacks = evidence.stacks
      indexFiles = indexURLs.map(\.path)
    } else {
      for (dataFile, indexFile) in zip(dataFiles, indexURLs) {
        if let metadata = try QH5IndexWriter.currentMetadata(
          source: dataFile,
          destination: indexFile
        ) {
          stacks.append(
            NativeHDF5Stack(
              frameCount: metadata.nFrames,
              detectorRows: metadata.detRows,
              detectorColumns: metadata.detCols,
              sourceBytes: metadata.srcDtype == "uint8" ? 1 : 2,
              chunks: []
            )
          )
        } else {
          stacks.append(
            try NativeHDF5Bridge.inspectStack(
              at: dataFile,
              includeChunks: false
            )
          )
        }
        indexFiles.append(indexFile.path)
      }
    }

    guard let first = stacks.first else { throw Native4DSTEMIOError.noDatasets }
    guard
      stacks.dropFirst().allSatisfy({
        $0.detectorRows == first.detectorRows
          && $0.detectorColumns == first.detectorColumns
          && $0.sourceBytes == first.sourceBytes
      })
    else {
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
    let metadataSource =
      FileManager.default.fileExists(atPath: source.path)
      ? source
      : dataFiles[0]
    let master = try NativeHDF5Bridge.inspectMaster(
      at: metadataSource,
      detectorRows: first.detectorRows,
      detectorColumns: first.detectorColumns
    )
    let scanShape = try scanShape(master: master, totalFrames: totalFrames)
    let spatialCalibration = spatialCalibration(
      source: source,
      master: master,
      scanShape: scanShape
    )
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
    let dataset = Native4DSTEMDataset(
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
      scanPixelSizeRowNanometer: spatialCalibration.sampling?.row,
      scanPixelSizeColNanometer: spatialCalibration.sampling?.column,
      kPixelSizeRow: master.reciprocalSampling?.row,
      kPixelSizeCol: master.reciprocalSampling?.column,
      kPixelUnit: master.reciprocalSampling == nil ? nil : "mrad",
      acquisitionDate: master.acquisitionDate,
      metadata: master.metadata.merging(spatialCalibration.metadata) { _, new in new },
      schemaIdentity: "live4dstem.dataset/v0.1",
      sourceIdentitySHA256: hashes?.aggregate,
      masterSHA256: hashes?.master,
      orderedMemberSHA256: hashes?.members,
      sourceScanCalibration: nil,
      scalarImageRawPath: nil
    )
    try cacheDataset(dataset, at: datasetCache)
    return dataset
  }

  private func cachedDataset(
    at cache: URL,
    signature: String,
    source: URL,
    dataFiles: [URL]
  ) throws -> Native4DSTEMDataset? {
    guard let data = try? Data(contentsOf: cache),
      let dataset = try? JSONDecoder().decode(Native4DSTEMDataset.self, from: data),
      dataset.id == signature,
      dataset.masterPath == (isMaster(source) ? source.path : nil),
      dataset.dataFiles == dataFiles.map(\.path)
    else { return nil }
    return dataset
  }

  private func cacheDataset(_ dataset: Native4DSTEMDataset, at cache: URL) throws {
    try FileManager.default.createDirectory(
      at: cache.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    try JSONEncoder().encode(dataset).write(to: cache, options: .atomic)
  }

  private func prepareVeloxDataset(
    source: URL,
    mode: Native4DSTEMCatalogMode
  ) throws -> Native4DSTEMDataset {
    let signature = try nativeDatasetSignature(for: [source])
    let indexRoot = cacheDirectory.appendingPathComponent(signature, isDirectory: true)
    let temporary = indexRoot.appendingPathComponent(
      ".\(source.deletingPathExtension().lastPathComponent).\(UUID().uuidString).tmp"
    )
    let before = try nativeFileIdentity(for: source)
    if mode == .indexed {
      try FileManager.default.createDirectory(at: indexRoot, withIntermediateDirectories: true)
    }
    defer { try? FileManager.default.removeItem(at: temporary) }
    let image = try NativeHDF5Bridge.prepareVeloxImage(
      at: source,
      rawOutput: mode == .indexed ? temporary : nil
    )
    let typedRawPath = indexRoot.appendingPathComponent(
      "\(source.deletingPathExtension().lastPathComponent).\(image.sourceDtype).raw"
    )
    if mode == .indexed {
      if FileManager.default.fileExists(atPath: typedRawPath.path) {
        _ = try FileManager.default.replaceItemAt(typedRawPath, withItemAt: temporary)
      } else {
        try FileManager.default.moveItem(at: temporary, to: typedRawPath)
      }
    }
    let calibration = try veloxCalibration(
      metadataJSON: image.metadataJSON,
      rows: image.rows,
      columns: image.columns,
      evidencePath: image.metadataPath
    )
    let hashes =
      mode == .indexed
      ? try nativeSourceHashes(master: nil, dataFiles: [source]) : nil
    let after = try nativeFileIdentity(for: source)
    guard before.device == after.device,
      before.inode == after.inode,
      before.bytes == after.bytes,
      before.modificationNanoseconds == after.modificationNanoseconds
    else {
      throw Native4DSTEMIOError.invalidData(
        "The EMD source changed while it was being prepared; retry"
      )
    }
    guard let sourceBytes = Int(exactly: after.bytes) else {
      throw Native4DSTEMIOError.invalidData("\(source.lastPathComponent) is too large")
    }
    return Native4DSTEMDataset(
      id: signature,
      label: source.deletingPathExtension().lastPathComponent,
      masterPath: nil,
      dataFiles: [source.path],
      indexFiles: [],
      scanRows: image.rows,
      scanCols: image.columns,
      detectorRows: 1,
      detectorCols: 1,
      sourceDtype: image.sourceDtype,
      sourceBytes: sourceBytes,
      badPixelIndices: [],
      kPixelSizeRow: nil,
      kPixelSizeCol: nil,
      kPixelUnit: nil,
      acquisitionDate: nil,
      metadata: [
        "sourceFormat": "Velox EMD scalar image",
        "metadataEvidence": image.metadataPath,
      ],
      schemaIdentity: "live4dstem.dataset/v0.1",
      sourceIdentitySHA256: hashes?.aggregate,
      masterSHA256: nil,
      orderedMemberSHA256: hashes?.members,
      sourceScanCalibration: calibration,
      scalarImageRawPath: typedRawPath.path
    )
  }

  private func veloxCalibration(
    metadataJSON: String,
    rows: Int,
    columns: Int,
    evidencePath: String
  ) throws -> Native4DSTEMScanCalibration {
    guard let data = metadataJSON.data(using: .utf8),
      let document = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
      throw Native4DSTEMIOError.invalidData("Velox Metadata JSON is invalid")
    }
    let unitX = try stringValue(document, path: ["BinaryResult", "PixelUnitX"])
      .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    let unitY = try stringValue(document, path: ["BinaryResult", "PixelUnitY"])
      .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard unitX == "m", unitY == "m" else {
      throw Native4DSTEMIOError.invalidData(
        "Velox PixelSize needs explicit metre PixelUnitX and PixelUnitY values"
      )
    }
    let rowMeters = try positiveDouble(
      document,
      path: ["BinaryResult", "PixelSize", "height"]
    )
    let columnMeters = try positiveDouble(
      document,
      path: ["BinaryResult", "PixelSize", "width"]
    )
    let scanRows = try integerValue(document, path: ["Scan", "ScanSize", "height"])
    let scanColumns = try integerValue(document, path: ["Scan", "ScanSize", "width"])
    guard scanRows == rows, scanColumns == columns else {
      throw Native4DSTEMIOError.invalidData(
        "Velox Scan.ScanSize does not match the image dimensions"
      )
    }
    let rowField = try positiveDouble(
      document,
      path: ["Optics", "FullScanFieldOfView", "y"]
    )
    let columnField = try positiveDouble(
      document,
      path: ["Optics", "FullScanFieldOfView", "x"]
    )
    guard approximatelyEqual(rowMeters * Double(rows), rowField) else {
      throw Native4DSTEMIOError.invalidData(
        "Velox row PixelSize and FullScanFieldOfView disagree"
      )
    }
    guard approximatelyEqual(columnMeters * Double(columns), columnField) else {
      throw Native4DSTEMIOError.invalidData(
        "Velox column PixelSize and FullScanFieldOfView disagree"
      )
    }
    return Native4DSTEMScanCalibration(
      rowSamplingAngstrom: rowMeters * 1e10,
      columnSamplingAngstrom: columnMeters * 1e10,
      origin: .sourceMetadata,
      evidence:
        "\(evidencePath) · BinaryResult.PixelSize + PixelUnitX/Y; cross-checked against Optics.FullScanFieldOfView"
    )
  }

  private func value(_ document: [String: Any], path: [String]) -> Any? {
    var current: Any = document
    for key in path {
      guard let object = current as? [String: Any], let next = object[key] else {
        return nil
      }
      current = next
    }
    return current
  }

  private func stringValue(_ document: [String: Any], path: [String]) throws -> String {
    guard let result = value(document, path: path) as? String else {
      throw Native4DSTEMIOError.invalidData(
        "Velox \(path.joined(separator: ".")) is missing"
      )
    }
    return result
  }

  private func positiveDouble(_ document: [String: Any], path: [String]) throws -> Double {
    let raw = value(document, path: path)
    let result = (raw as? NSNumber)?.doubleValue ?? (raw as? String).flatMap(Double.init)
    guard let result else {
      throw Native4DSTEMIOError.invalidData(
        "Velox \(path.joined(separator: ".")) is missing or not numeric"
      )
    }
    guard result.isFinite, result > 0 else {
      throw Native4DSTEMIOError.invalidData(
        "Velox \(path.joined(separator: ".")) must be finite and positive"
      )
    }
    return result
  }

  private func integerValue(_ document: [String: Any], path: [String]) throws -> Int {
    let raw = value(document, path: path)
    let result = (raw as? NSNumber)?.intValue ?? (raw as? String).flatMap(Int.init)
    guard let result else {
      throw Native4DSTEMIOError.invalidData(
        "Velox \(path.joined(separator: ".")) is missing"
      )
    }
    return result
  }

  private func approximatelyEqual(_ left: Double, _ right: Double) -> Bool {
    abs(left - right) <= max(1e-15, 1e-6 * max(abs(left), abs(right)))
  }

  private func spatialCalibration(
    source: URL,
    master: NativeHDF5Master,
    scanShape: (rows: Int, columns: Int)
  ) -> (
    sampling: (row: Double, column: Double)?,
    metadata: [String: String]
  ) {
    if let sampling = master.scanPixelSizeNanometer {
      return (sampling, ["spatial_calibration_source": "HDF5 metadata"])
    }
    guard let emd = unambiguousVeloxSibling(for: source),
      let fieldOfView = try? NativeHDF5Bridge.veloxFieldOfViewNanometer(at: emd),
      scanShape.rows > 0,
      scanShape.columns > 0
    else { return (nil, [:]) }
    let row = fieldOfView.row / Double(scanShape.rows)
    let column = fieldOfView.column / Double(scanShape.columns)
    guard row.isFinite, column.isFinite, row > 0, column > 0 else {
      return (nil, [:])
    }
    return (
      (row: row, column: column),
      ["spatial_calibration_source": "Velox EMD · \(emd.lastPathComponent)"]
    )
  }

  private func unambiguousVeloxSibling(for source: URL) -> URL? {
    let directory = source.deletingLastPathComponent()
    guard
      let files = try? FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
      )
    else { return nil }
    let emdFiles = files.filter { $0.pathExtension.lowercased() == "emd" }
    guard !emdFiles.isEmpty else { return nil }
    let sourceName = source.lastPathComponent
    let stem =
      sourceName.lowercased().hasSuffix("_master.h5")
      ? String(sourceName.dropLast("_master.h5".count))
      : source.deletingPathExtension().lastPathComponent
    let exact = emdFiles.filter {
      $0.deletingPathExtension().lastPathComponent.caseInsensitiveCompare(stem) == .orderedSame
    }
    if exact.count == 1 { return exact[0] }
    let prefixed = emdFiles.filter {
      $0.deletingPathExtension().lastPathComponent.lowercased().hasPrefix(stem.lowercased())
    }
    if prefixed.count == 1 { return prefixed[0] }
    return emdFiles.count == 1 ? emdFiles[0] : nil
  }

  private func prepareIndexedStacks(
    dataFiles: [URL],
    indexFiles: [URL]
  ) throws -> [NativeHDF5Stack] {
    let workerCount = min(8, dataFiles.count)
    let resultLock = NSLock()
    nonisolated(unsafe) var results = [Result<NativeHDF5Stack, Error>?](
      repeating: nil,
      count: dataFiles.count
    )

    DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
      for index in stride(from: worker, to: dataFiles.count, by: workerCount) {
        let result = Result {
          let metadata = try QH5IndexWriter.prepare(
            source: dataFiles[index],
            destination: indexFiles[index]
          )
          return NativeHDF5Stack(
            frameCount: metadata.nFrames,
            detectorRows: metadata.detRows,
            detectorColumns: metadata.detCols,
            sourceBytes: metadata.srcDtype == "uint8" ? 1 : 2,
            chunks: []
          )
        }
        resultLock.lock()
        results[index] = result
        resultLock.unlock()
      }
    }
    return try results.enumerated().map { index, result in
      guard let result else {
        throw Native4DSTEMIOError.invalidData(
          "Could not prepare \(dataFiles[index].lastPathComponent)"
        )
      }
      return try result.get()
    }
  }

  private func prepareIndexedEvidence(
    master: URL?,
    dataFiles: [URL],
    indexRoot: URL,
    indexFiles: [URL]
  ) throws -> (hashes: NativeSourceHashes, stacks: [NativeHDF5Stack]) {
    let resultLock = NSLock()
    nonisolated(unsafe) var hashResult: Result<NativeSourceHashes, Error>?
    nonisolated(unsafe) var stackResult: Result<[NativeHDF5Stack], Error>?

    DispatchQueue.concurrentPerform(iterations: 2) { task in
      if task == 0 {
        let result = Result {
          try nativeSourceHashes(
            master: master,
            dataFiles: dataFiles,
            cacheFile: indexRoot.appendingPathComponent("source-hashes.json")
          )
        }
        resultLock.lock()
        hashResult = result
        resultLock.unlock()
      } else {
        let result = Result {
          try prepareIndexedStacks(dataFiles: dataFiles, indexFiles: indexFiles)
        }
        resultLock.lock()
        stackResult = result
        resultLock.unlock()
      }
    }
    guard let hashResult, let stackResult else {
      throw Native4DSTEMIOError.invalidData(
        "Could not prepare the exact source identity and QH5 indexes"
      )
    }
    return (try hashResult.get(), try stackResult.get())
  }

  private func masters(for input: URL) throws -> [URL] {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: input.path, isDirectory: &isDirectory) else {
      throw Native4DSTEMIOError.invalidData("HDF5 input does not exist: \(input.path)")
    }
    if isDirectory.boolValue {
      var pending = [input]
      var masters: [URL] = []
      while let directory = pending.popLast() {
        let names = try nativeDirectoryNames(at: directory)
        for name in names where !name.hasPrefix(".") {
          let child = directory.appendingPathComponent(name)
          if try nativePathIsDirectory(child) {
            pending.append(child)
          } else if isMaster(child) || isEMD(child) {
            masters.append(nativeCanonicalURL(child))
          }
        }
      }
      masters.sort { $0.path < $1.path }
      guard !masters.isEmpty else {
        throw Native4DSTEMIOError.invalidData(
          "No recognized *_master.h5 or Velox .emd files were found under \(input.path)"
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
    if input.pathExtension.caseInsensitiveCompare("h5") == .orderedSame {
      let companionMaster = input.deletingLastPathComponent()
        .appendingPathComponent(
          input.deletingPathExtension().lastPathComponent + "_master.h5"
        )
      if FileManager.default.fileExists(atPath: companionMaster.path) {
        return [nativeCanonicalURL(companionMaster)]
      }
    }
    return [input]
  }

  private func dataFiles(for source: URL) throws -> [URL] {
    let directory = source.deletingLastPathComponent()
    if isMaster(source) {
      let suffixLength = "_master.h5".count
      let stem = String(source.lastPathComponent.dropLast(suffixLength))
      let globbed = try nativeDirectoryNames(at: directory)
        .filter { !$0.hasPrefix(".") }
        .map { directory.appendingPathComponent($0) }
        .filter {
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
      let siblings = try nativeDirectoryNames(at: directory)
        .filter { !$0.hasPrefix(".") }
        .map { directory.appendingPathComponent($0) }
        .filter { shardStem($0.lastPathComponent) == stem }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
      if !siblings.isEmpty {
        return siblings.map(nativeCanonicalURL)
      }
    }
    return [nativeCanonicalURL(source)]
  }

  private func nativePathIsDirectory(_ url: URL) throws -> Bool {
    var status = stat()
    let result = url.path.withCString { Darwin.lstat($0, &status) }
    guard result == 0 else {
      throw Native4DSTEMIOError.invalidData("Could not inspect \(url.path)")
    }
    return (status.st_mode & S_IFMT) == S_IFDIR
  }

  private func nativeDirectoryNames(at directory: URL) throws -> [String] {
    guard let stream = directory.path.withCString({ Darwin.opendir($0) }) else {
      throw Native4DSTEMIOError.invalidData("Could not inspect folder \(directory.path)")
    }
    defer { Darwin.closedir(stream) }
    var names: [String] = []
    while let entry = Darwin.readdir(stream) {
      let name = withUnsafePointer(to: entry.pointee.d_name) { pointer in
        pointer.withMemoryRebound(
          to: CChar.self,
          capacity: Int(MAXNAMLEN) + 1
        ) { String(cString: $0) }
      }
      if name != "." && name != ".." {
        names.append(name)
      }
    }
    return names
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

  private func isEMD(_ url: URL) -> Bool {
    url.pathExtension.lowercased() == "emd"
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
    let stem =
      suffixLength == 0
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
    return
      "\(row.formatted(.number.precision(.fractionLength(0...8))))°, \(column.formatted(.number.precision(.fractionLength(0...8))))°"
  }
}
