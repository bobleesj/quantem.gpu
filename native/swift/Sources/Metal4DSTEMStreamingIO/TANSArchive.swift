import CryptoKit
import Darwin
import Foundation

// Experimental sealed-checkpoint reader. No scientific decoding is done here.
struct TANSArchive {
  enum Profile: Equatable {
    case source, prepared
  }

  // Published names describe representation, not the private source identity.
  // Fingerprints retain two immutable prototype archives without rewriting them.
  static func profile(format: String) -> Profile? {
    switch format {
    case "metal-entropy-source-v1": return .source
    case "metal-entropy-prepared-v1": return .prepared
    default:
      let fingerprint = SHA256.hash(data: Data(format.utf8))
        .map { String(format: "%02x", $0) }.joined()
      switch fingerprint {
      case "80a639124c783fce9b809464b7d58557c9e53ce14b58c060e978ac586ac24e49":
        return .source
      case "db5e0e20da8d2d45374f29f99b1c02f955335444a0a26869a1302285e5b834ca":
        return .prepared
      default: return nil
      }
    }
  }

  enum SourceReadPolicy: String, Sendable {
    case systemCache
    case avoidCaching
  }
  struct ArrayDescriptor: Decodable {
    let shape: [Int]
    let dtype: String
    let nbytes: Int
    let sha256: String
  }
  struct Component: Decodable, Sendable {
    let name: String
    let shape: [Int]
    let dtype: String
    let nbytes: Int
    let offset: Int
  }
  struct Chunk: Decodable, Sendable {
    let chunk: Int
    let acquisition: Int
    let firstScan: Int
    let scanCount: Int
    let shard: Int
    let fileOffset: UInt64
    let recordBytes: Int
    let sha256: String
    let components: [Component]
  }
  struct Layout: Decodable {
    struct File: Decodable {
      let name: String
      let nbytes: UInt64
    }
    let shape: [Int]
    let sourceDtype: String
    let byteOrder: String
    let chunkScans: Int
    let streamScans: Int
    let chunks: [Chunk]
    let files: [File]
  }
  struct Metadata: Decodable {
    struct Semantics: Decodable {
      let codec: String
      let sparse: String
      let nativeDtype: String
      let applyMaskToSource: Bool
    }
    struct IndexRebuild: Decodable {
      let sourceCodec: String
      let sparseCodec: String
      let nativeShape: [Int]
      let nativeDtype: String
      let sourceMaskApplied: Bool
      let hardwareCountsPreserved: Bool
    }
    let complete: Bool
    let format: String
    let semantics: Semantics?
    let indexRebuild: IndexRebuild?
    let layout: Layout
    let globalStateFile: String
    let globalStateFileSha256: String
    let globalState: [String: ArrayDescriptor]
  }
  let root: URL
  let metadata: Metadata
  let acquisitions: [Int]
  let chunks: [Chunk]
  let arrays: [String: Data]

  init(directory: URL, acquisitions: [Int]) throws {
    root = directory
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let checkpoint = try Self.readBounded(
      try Self.child("checkpoint.json", under: directory), maximumBytes: 64 * 1024 * 1024)
    let metadata = try decoder.decode(Metadata.self, from: checkpoint)
    self.metadata = metadata
    if let semantics = metadata.semantics {
      guard semantics.codec == "source112-tans1024-pair-v1",
        semantics.sparse == "position9-flag1-count8-rank256-v1",
        semantics.nativeDtype == "<u2", !semantics.applyMaskToSource
      else { throw Self.invalid("Unsupported exact tANS semantics") }
    } else {
      // The source-only writer declares the same count contract in its index
      // rebuild metadata. Require that declaration; do not infer semantics or
      // rewrite the immutable source checkpoint to fit the reader.
      guard Self.profile(format: metadata.format) == .source,
        let contract = metadata.indexRebuild,
        contract.sourceCodec == "source112-tans1024-pair-v1",
        contract.sparseCodec == "position9-flag1-count8-rank256-v1",
        contract.nativeDtype == "<u2", contract.nativeShape == metadata.layout.shape,
        !contract.sourceMaskApplied, contract.hardwareCountsPreserved
      else {
        throw Self.invalid("Missing exact tANS semantics")
      }
    }
    guard metadata.complete,
      Self.profile(format: metadata.format) != nil,
      metadata.layout.shape.count == 5,
      metadata.layout.shape[0] > 0,
      metadata.layout.shape[0] <= Int(UInt32.max) / (4 * 36864),
      Array(metadata.layout.shape.suffix(4)) == [512, 512, 192, 192],
      metadata.layout.sourceDtype == "<u2",
      metadata.layout.byteOrder == "little",
      metadata.layout.chunkScans == 16384, metadata.layout.streamScans == 512
    else { throw Self.invalid("Unsupported or incomplete exact tANS checkpoint contract") }
    guard !acquisitions.isEmpty, Set(acquisitions).count == acquisitions.count,
      acquisitions.allSatisfy({ (0..<metadata.layout.shape[0]).contains($0) })
    else { throw Self.invalid("Select unique in-range acquisition indices") }
    self.acquisitions = acquisitions
    var selected: [Chunk] = []
    for acquisition in acquisitions {
      let records = metadata.layout.chunks.filter { $0.acquisition == acquisition }.sorted {
        $0.firstScan < $1.firstScan
      }
      guard records.count == 16 else {
        throw Self.invalid("Acquisition is missing a native scan chunk")
      }
      for (index, record) in records.enumerated() {
        guard record.firstScan == index * 16384, record.scanCount == 16384,
          record.chunk == acquisition * 16 + index,
          record.recordBytes > 0, record.recordBytes <= 512 * 1024 * 1024,
          metadata.layout.files.indices.contains(record.shard),
          record.fileOffset <= metadata.layout.files[record.shard].nbytes,
          UInt64(record.recordBytes) <= metadata.layout.files[record.shard].nbytes
            - record.fileOffset,
          Set(record.components.map(\.name)).count == record.components.count
        else { throw Self.invalid("Invalid native chunk extent or duplicate component") }
        for component in record.components {
          guard component.offset >= 0, component.nbytes > 0,
            component.offset <= record.recordBytes,
            component.nbytes <= record.recordBytes - component.offset,
            component.offset % 4 == 0,
            try Self.byteCount(shape: component.shape, dtype: component.dtype) == component.nbytes
          else { throw Self.invalid("Component exceeds its authenticated chunk") }
        }
        let ordered = record.components.sorted { $0.offset < $1.offset }
        for pair in zip(ordered, ordered.dropFirst()) {
          guard pair.0.offset + pair.0.nbytes <= pair.1.offset else {
            throw Self.invalid("Authenticated components overlap")
          }
        }
        for name in ["dense", "dense_offsets", "sparse", "sparse_offsets"] {
          guard let component = record.components.first(where: { $0.name == name }),
            component.dtype == "<u4", component.shape.count == 1
          else {
            throw Self.invalid("Missing native entropy component \(name)")
          }
        }
      }
      selected += records
    }
    chunks = selected
    let globalURL = try Self.child(metadata.globalStateFile, under: directory)
    let global = try Self.readBounded(globalURL, maximumBytes: 128 * 1024 * 1024)
    try Self.verify(global, sha256: metadata.globalStateFileSha256)
    let members = try StoredNPZ.members(global)
    var extracted: [String: Data] = [:]
    for name in ["codec__decoding", "model_ids", "planner__cache_map", "planner__valid"] {
      guard let descriptor = metadata.globalState[name], let member = members[name + ".npy"] else {
        throw Self.invalid("Missing exact tANS array \(name)")
      }
      let values = try StoredNPZ.values(member, descriptor: descriptor)
      guard values.count == descriptor.nbytes,
        try Self.byteCount(shape: descriptor.shape, dtype: descriptor.dtype) == descriptor.nbytes
      else {
        throw Self.invalid("Array byte count disagrees with metadata")
      }
      try Self.verify(values, sha256: descriptor.sha256)
      extracted[name] = values
    }
    guard metadata.globalState["codec__decoding"]?.dtype == "<u4",
      metadata.globalState["codec__decoding"]?.shape.count == 2,
      metadata.globalState["codec__decoding"]?.shape.last == 1024,
      (1...255).contains(metadata.globalState["codec__decoding"]!.shape[0]),
      metadata.globalState["model_ids"]?.dtype == "|u1",
      metadata.globalState["model_ids"]?.shape == [metadata.layout.shape[0] * 4, 36864],
      metadata.globalState["planner__cache_map"]?.dtype == "<i4",
      metadata.globalState["planner__cache_map"]?.shape == [36864],
      metadata.globalState["planner__valid"]?.dtype == "|i1",
      metadata.globalState["planner__valid"]?.shape == [36864],
      extracted["codec__decoding"]!.count % 4096 == 0,
      extracted["model_ids"]!.count == metadata.layout.shape[0] * 4 * 36864,
      extracted["planner__cache_map"]!.count == 36864 * 4,
      extracted["planner__valid"]!.count == 36864
    else { throw Self.invalid("tANS model or detector-array geometry mismatch") }
    // Optional planning weights are metadata, not scientific counts. Older
    // source-only archives remain usable without the experimental tile index.
    if let descriptor = metadata.globalState["planner__column_cost"] {
      guard descriptor.dtype == "<f8", descriptor.shape == [36864],
        descriptor.nbytes == 36864 * 8, let member = members["planner__column_cost.npy"]
      else {
        throw Self.invalid("Invalid optional detector planning-cost metadata")
      }
      let values = try StoredNPZ.values(member, descriptor: descriptor)
      try Self.verify(values, sha256: descriptor.sha256)
      guard values.count == 36864 * 8 else {
        throw Self.invalid("Truncated detector planning costs")
      }
      let valid = values.withUnsafeBytes { raw in
        (0..<36864).allSatisfy {
          let value = raw.loadUnaligned(fromByteOffset: $0 * 8, as: Double.self)
          return value.isFinite && value >= 0
        }
      }
      guard valid else {
        throw Self.invalid("Detector planning costs must be finite and nonnegative")
      }
      extracted["planner__column_cost"] = values
    }
    arrays = extracted
  }

  func read(_ chunk: Chunk) throws -> Data {
    var metrics = ReadMetrics()
    return try read(chunk, metrics: &metrics)
  }

  struct ReadMetrics {
    var ioSeconds = 0.0
    var hashSeconds = 0.0
    var validationSeconds = 0.0
    var sourceBytesRead: UInt64 = 0
    var authenticatedRecords = 0
  }

  func read(
    _ chunk: Chunk, metrics: inout ReadMetrics, policy: SourceReadPolicy = .systemCache
  ) throws -> Data {
    guard (1...(512 * 1024 * 1024)).contains(chunk.recordBytes) else {
      throw Self.invalid("Encoded record exceeds the bounded reader")
    }
    var bytes = Data(count: chunk.recordBytes)
    try bytes.withUnsafeMutableBytes {
      try read(chunk, into: $0, metrics: &metrics, policy: policy)
    }
    return bytes
  }

  /// Fill bounded caller-owned compressed staging, then authenticate those exact
  /// bytes before upload. No archive mapping or decoded 4D allocation is retained.
  func read(
    _ chunk: Chunk, into destination: UnsafeMutableRawBufferPointer,
    metrics: inout ReadMetrics, policy: SourceReadPolicy = .systemCache
  ) throws {
    guard (1...(512 * 1024 * 1024)).contains(chunk.recordBytes),
      chunk.fileOffset <= UInt64(Int64.max) - UInt64(chunk.recordBytes),
      destination.count >= chunk.recordBytes, let address = destination.baseAddress
    else {
      throw Self.invalid("Encoded staging is smaller than the authenticated record")
    }
    let ioStart = ProcessInfo.processInfo.systemUptime
    let url = try Self.child(metadata.layout.files[chunk.shard].name, under: root)
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    if policy == .avoidCaching, fcntl(handle.fileDescriptor, F_NOCACHE, 1) != 0 {
      throw Self.invalid("Cannot apply F_NOCACHE to encoded payload descriptor")
    }
    var received = 0
    while received < chunk.recordBytes {
      let count = pread(
        handle.fileDescriptor, address.advanced(by: received), chunk.recordBytes - received,
        off_t(chunk.fileOffset) + off_t(received))
      if count < 0 && errno == EINTR { continue }
      guard count > 0 else {
        throw Self.invalid(
          "Truncated or unreadable tANS record; no partial resident may be published")
      }
      received += count
      metrics.sourceBytesRead += UInt64(count)
    }
    metrics.ioSeconds += ProcessInfo.processInfo.systemUptime - ioStart
    let bytes = Data(bytesNoCopy: address, count: chunk.recordBytes, deallocator: .none)
    let hashStart = ProcessInfo.processInfo.systemUptime
    try Self.verify(bytes, sha256: chunk.sha256)
    metrics.authenticatedRecords += 1
    metrics.hashSeconds += ProcessInfo.processInfo.systemUptime - hashStart
    let validationStart = ProcessInfo.processInfo.systemUptime
    try TANSStreamValidation.record(
      bytes, chunk: chunk, cacheMap: arrays["planner__cache_map"]!, models: arrays["model_ids"]!)
    metrics.validationSeconds += ProcessInfo.processInfo.systemUptime - validationStart
  }

  static func child(_ name: String, under directory: URL) throws -> URL {
    guard !name.isEmpty, name == URL(fileURLWithPath: name).lastPathComponent,
      name != ".", name != ".."
    else { throw invalid("Archive member must be a relative filename") }
    let url = directory.appendingPathComponent(name)
    guard
      url.resolvingSymlinksInPath().deletingLastPathComponent().standardizedFileURL
        == directory.resolvingSymlinksInPath().standardizedFileURL
    else {
      throw invalid("Archive member escapes the selected directory through a symlink")
    }
    return url
  }
  static func readBounded(_ url: URL, maximumBytes: Int) throws -> Data {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    guard attributes[.type] as? FileAttributeType == .typeRegular,
      let size = attributes[.size] as? NSNumber, size.uint64Value <= UInt64(maximumBytes)
    else {
      throw invalid("Archive metadata exceeds the bounded reader or is not a regular file")
    }
    let data = try Data(contentsOf: url)
    guard data.count <= maximumBytes else { throw invalid("Archive metadata grew during read") }
    return data
  }
  static func byteCount(shape: [Int], dtype: String) throws -> Int {
    let width: Int
    switch dtype {
    case "|u1", "|i1", "|b1": width = 1
    case "<u2": width = 2
    case "<u4", "<i4": width = 4
    case "<u8", "<i8", "<f8": width = 8
    default: throw invalid("Unsupported archive scalar dtype \(dtype)")
    }
    var count = width
    for dimension in shape {
      let product = count.multipliedReportingOverflow(by: dimension)
      guard dimension > 0, !product.overflow else {
        throw invalid("Invalid or overflowing array shape")
      }
      count = product.partialValue
    }
    return count
  }
  static func verify(_ data: Data, sha256: String) throws {
    guard SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == sha256 else {
      throw invalid("tANS archive SHA256 mismatch; no result was published")
    }
  }
  static func invalid(_ message: String) -> NSError {
    NSError(domain: "QuantEM.MetalTANS", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
  }
}

// The sealed producer uses uncompressed NPZ members. Unsupported ZIP methods
// fail closed rather than launching Python or a shell inside the application.
private enum StoredNPZ {
  static func members(_ bytes: Data) throws -> [String: Data] {
    var result: [String: Data] = [:]
    var cursor = 0
    func integer(_ offset: Int, _ width: Int) throws -> UInt64 {
      guard offset >= 0, offset <= bytes.count, width <= bytes.count - offset else {
        throw TANSArchive.invalid("Truncated NPZ header")
      }
      return (0..<width).reduce(0) { $0 | (UInt64(bytes[offset + $1]) << ($1 * 8)) }
    }
    while cursor + 4 <= bytes.count {
      let signature = try integer(cursor, 4)
      if signature == 0x0201_4b50 { break }
      guard signature == 0x0403_4b50, try integer(cursor + 8, 2) == 0,
        try integer(cursor + 6, 2) & 9 == 0
      else {
        throw TANSArchive.invalid("Only stored, nonencrypted, sized NPZ members are supported")
      }
      let nameLength = Int(try integer(cursor + 26, 2))
      let extraLength = Int(try integer(cursor + 28, 2))
      let begin = cursor + 30 + nameLength + extraLength
      guard begin <= bytes.count else { throw TANSArchive.invalid("Truncated NPZ name") }
      var size = try integer(cursor + 18, 4)
      if size == UInt32.max {
        let extra = cursor + 30 + nameLength
        guard extraLength >= 20, try integer(extra, 2) == 1,
          try integer(extra + 2, 2) >= 16
        else { throw TANSArchive.invalid("Invalid ZIP64 member size") }
        size = try integer(extra + 12, 8)
        guard size == (try integer(extra + 4, 8)) else {
          throw TANSArchive.invalid("Compressed ZIP64 member is unsupported")
        }
      } else {
        guard size == (try integer(cursor + 22, 4)) else {
          throw TANSArchive.invalid("Compressed NPZ member is unsupported")
        }
      }
      guard size <= UInt64(bytes.count - begin),
        let name = String(data: bytes[(cursor + 30)..<(cursor + 30 + nameLength)], encoding: .utf8),
        result[name] == nil
      else { throw TANSArchive.invalid("Invalid NPZ member extent or duplicate name") }
      result[name] = bytes.subdata(in: begin..<(begin + Int(size)))
      cursor = begin + Int(size)
    }
    return result
  }
  static func values(_ bytes: Data, descriptor: TANSArchive.ArrayDescriptor) throws -> Data {
    guard bytes.count >= 10, Array(bytes.prefix(6)) == [147, 78, 85, 77, 80, 89] else {
      throw TANSArchive.invalid("Invalid NPY member")
    }
    let width: Int
    switch bytes[6] {
    case 1: width = 2
    case 2, 3: width = 4
    default: throw TANSArchive.invalid("Unsupported NPY version")
    }
    guard bytes.count >= 8 + width else { throw TANSArchive.invalid("Truncated NPY header") }
    let size = (0..<width).reduce(0) { $0 | (Int(bytes[8 + $1]) << (8 * $1)) }
    let start = 8 + width + size
    guard start <= bytes.count,
      let header = String(data: bytes[(8 + width)..<start], encoding: .utf8),
      header.contains("'fortran_order': False")
    else { throw TANSArchive.invalid("Unsupported NPY array order") }
    let compact = header.filter { !$0.isWhitespace }
    guard compact.contains("'descr':'\(descriptor.dtype)'"),
      let shapeStart = compact.range(of: "'shape':("),
      let shapeEnd = compact[shapeStart.upperBound...].firstIndex(of: ")")
    else {
      throw TANSArchive.invalid("NPY dtype or shape declaration disagrees with metadata")
    }
    let dimensions = compact[shapeStart.upperBound..<shapeEnd].split(separator: ",")
    guard dimensions.count == descriptor.shape.count,
      zip(dimensions, descriptor.shape).allSatisfy({ Int($0.0) == $0.1 })
    else {
      throw TANSArchive.invalid("NPY dimensions disagree with metadata")
    }
    return bytes.subdata(in: start..<bytes.count)
  }
}
