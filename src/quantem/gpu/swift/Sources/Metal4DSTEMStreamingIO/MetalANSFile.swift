import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMKernels

/// The on-disk section layout of a validated QuantEM count-ANS file.
struct MetalANSFileSection {
  let name: String
  let offset: UInt64
  let byteCount: Int
  let elementCount: Int
  let elementStride: Int
  let dtype: String
  let sha256: String
}

/// Metadata needed to bind a QGANS file to the native Metal count-ANS ABI.
struct MetalANSFileIndex {
  static let magic = Data([0x51, 0x47, 0x41, 0x4e, 0x53, 0x00, 0x01, 0x00])
  static let headerBytes = 24
  static let dataStart: UInt64 = 65_536
  static let sectionNames = [
    "payload", "offsets", "model_ids", "context_offsets", "symbols", "cumulative",
    "frequencies", "literal",
  ]

  let sourceURL: URL
  let shape: [Int]
  let detectorPixelCount: Int
  let logicalDtype: Metal4DSTEMIntegerDType
  let blockFrames: Int
  let scale: Int
  let logicalSHA256: String
  let fileBytes: UInt64
  let sections: [MetalANSFileSection]

  init(sourceURL: URL) throws {
    self.sourceURL = sourceURL.standardizedFileURL
    let attributes = try FileManager.default.attributesOfItem(atPath: sourceURL.path)
    guard let fileSize = (attributes[.size] as? NSNumber)?.uint64Value else {
      throw MetalANSFileError.invalid("could not determine the file size")
    }
    fileBytes = fileSize
    guard fileBytes >= Self.dataStart else {
      throw MetalANSFileError.invalid("file is shorter than the QGANS manifest area")
    }
    let descriptor = sourceURL.path.withCString { Darwin.open($0, O_RDONLY | O_CLOEXEC) }
    guard descriptor >= 0 else {
      throw MetalANSFileError.io("could not open \(sourceURL.lastPathComponent)")
    }
    defer { Darwin.close(descriptor) }
    let header = try Self.read(descriptor: descriptor, offset: 0, count: Self.headerBytes)
    guard header.prefix(8) == Self.magic else {
      throw MetalANSFileError.invalid("missing QGANS\\0\\1\\0 magic")
    }
    let manifestBytes = try Self.readUInt64(header, offset: 8)
    let dataStart = try Self.readUInt64(header, offset: 16)
    guard dataStart == Self.dataStart,
      manifestBytes > 0,
      manifestBytes <= Self.dataStart - UInt64(Self.headerBytes)
    else {
      throw MetalANSFileError.invalid("unsupported QGANS header")
    }
    let manifest = try Self.read(
      descriptor: descriptor, offset: UInt64(Self.headerBytes), count: Int(manifestBytes))
    let decoder = JSONDecoder()
    let document: Manifest
    do {
      document = try decoder.decode(Manifest.self, from: manifest)
    } catch {
      throw MetalANSFileError.invalid("QGANS manifest is malformed: \(error.localizedDescription)")
    }
    guard document.schema == "quantem.gpu.count-ans.v1",
      document.codec == "block-column-rans-byte-v1",
      document.order == "scan_row,scan_column,detector_row,detector_column",
      document.shape.count == 4,
      document.shape.allSatisfy({ $0 > 0 }),
      document.logicalSHA256.count == 64,
      document.logicalSHA256.allSatisfy({ $0.isHexDigit })
    else {
      throw MetalANSFileError.invalid("QGANS scientific contract is unsupported")
    }
    guard let dtype = Metal4DSTEMIntegerDType(rawValue: document.dtype),
      dtype == .uint8 || dtype == .uint16,
      document.blockFrames > 0,
      document.scale >= 1, document.scale <= 15
    else {
      throw MetalANSFileError.invalid("QGANS requires uint8/uint16 counts and scale 1...15")
    }
    let scanCount = try Self.product(document.shape[0], document.shape[1], label: "scan count")
    let pixels = try Self.product(document.shape[2], document.shape[3], label: "detector pixels")
    let blocks = (scanCount - 1) / document.blockFrames + 1
    let streamCount = try Self.product(blocks, pixels, label: "ANS stream count")
    guard streamCount <= Int(UInt32.max) else {
      throw MetalANSFileError.invalid("QGANS has too many indexed streams for Metal")
    }
    let expectedCounts: [String: Int] = [
      "offsets": streamCount + 1,
      "model_ids": streamCount,
    ]
    let expectedDtypes: [String: (String, Int)] = [
      "payload": ("u1", 1), "offsets": ("<u8", 8), "model_ids": ("<u4", 4),
      "context_offsets": ("<u4", 4), "symbols": ("<u2", 2),
      "cumulative": ("<u2", 2), "frequencies": ("<u2", 2), "literal": ("u1", 1),
    ]
    guard Set(document.sections.keys) == Set(Self.sectionNames) else {
      throw MetalANSFileError.invalid("QGANS must declare exactly eight typed sections")
    }
    var cursor = Self.dataStart
    var parsed: [MetalANSFileSection] = []
    for name in Self.sectionNames {
      guard let section = document.sections[name],
        let expected = expectedDtypes[name],
        section.dtype == expected.0,
        section.count <= UInt64(Int.max),
        section.sha256.count == 64,
        section.sha256.allSatisfy({ $0.isHexDigit })
      else { throw MetalANSFileError.invalid("QGANS section \(name) is malformed") }
      let count = Int(section.count)
      let size = count.multipliedReportingOverflow(by: expected.1)
      guard !size.overflow else {
        throw MetalANSFileError.invalid("QGANS section \(name) byte count overflows")
      }
      let aligned = (cursor + 7) & ~UInt64(7)
      let end = aligned.addingReportingOverflow(UInt64(size.partialValue))
      guard !end.overflow, section.offset == aligned, end.partialValue <= fileBytes else {
        throw MetalANSFileError.invalid("QGANS section \(name) is outside the file")
      }
      if let required = expectedCounts[name], count != required {
        throw MetalANSFileError.invalid("QGANS section \(name) disagrees with geometry")
      }
      parsed.append(
        MetalANSFileSection(
          name: name, offset: aligned, byteCount: size.partialValue, elementCount: count,
          elementStride: expected.1, dtype: expected.0, sha256: section.sha256.lowercased()))
      cursor = end.partialValue
    }
    guard cursor == fileBytes else {
      throw MetalANSFileError.invalid("QGANS contains trailing or undeclared bytes")
    }
    guard let literal = parsed.first(where: { $0.name == "literal" }), literal.elementCount > 0,
      let contexts = parsed.first(where: { $0.name == "context_offsets" }),
      contexts.elementCount == literal.elementCount + 1,
      let symbols = parsed.first(where: { $0.name == "symbols" }),
      let cumulative = parsed.first(where: { $0.name == "cumulative" }),
      let frequencies = parsed.first(where: { $0.name == "frequencies" }),
      symbols.elementCount == cumulative.elementCount,
      symbols.elementCount == frequencies.elementCount
    else {
      throw MetalANSFileError.invalid("QGANS model table counts are inconsistent")
    }
    shape = document.shape
    detectorPixelCount = pixels
    logicalDtype = dtype
    blockFrames = document.blockFrames
    scale = document.scale
    logicalSHA256 = document.logicalSHA256.lowercased()
    sections = parsed
  }

  private struct Manifest: Decodable {
    let schema: String
    let codec: String
    let order: String
    let shape: [Int]
    let dtype: String
    let blockFrames: Int
    let scale: Int
    let logicalSHA256: String
    let sections: [String: Section]

    enum CodingKeys: String, CodingKey {
      case schema, codec, order, shape, dtype
      case blockFrames = "block_frames"
      case scale
      case logicalSHA256 = "logical_sha256"
      case sections
    }
  }

  private struct Section: Decodable {
    let offset: UInt64
    let count: UInt64
    let dtype: String
    let sha256: String
  }

  private static func product(_ lhs: Int, _ rhs: Int, label: String) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else { throw MetalANSFileError.invalid("QGANS \(label) overflows Int") }
    return result.partialValue
  }

  private static func readUInt64(_ data: Data, offset: Int) throws -> UInt64 {
    guard offset >= 0, offset <= data.count - 8 else {
      throw MetalANSFileError.invalid("QGANS header is truncated")
    }
    return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self) }
      .littleEndian
  }

  static func read(descriptor: Int32, offset: UInt64, count: Int) throws -> Data {
    guard count >= 0, offset <= UInt64(Int64.max) else {
      throw MetalANSFileError.io("QGANS read range overflows")
    }
    var result = Data(count: count)
    var position = 0
    while position < count {
      let readCount = result.withUnsafeMutableBytes { raw in
        pread(
          descriptor, raw.baseAddress!.advanced(by: position), count - position,
          off_t(offset + UInt64(position)))
      }
      guard readCount > 0 else {
        throw MetalANSFileError.io("QGANS read failed or reached EOF")
      }
      position += readCount
    }
    return result
  }

  static func read(
    into destination: UnsafeMutableRawPointer, descriptor: Int32, offset: UInt64, count: Int
  ) throws {
    guard count >= 0, offset <= UInt64(Int64.max), UInt64(count) <= UInt64(Int64.max) - offset else {
      throw MetalANSFileError.io("QGANS read range overflows")
    }
    var position = 0
    while position < count {
      let readCount = pread(
        descriptor, destination.advanced(by: position), count - position,
        off_t(offset + UInt64(position)))
      guard readCount > 0 else {
        throw MetalANSFileError.io("QGANS read failed or reached EOF")
      }
      position += readCount
    }
  }
}

/// File-backed QGANS input with bounded reads and optional whole-file authentication.
final class MetalANSFileReader {
  let descriptor: Int32
  let index: MetalANSFileIndex

  init(
    sourceURL: URL, expectedSHA256: String?, verifyChecksums: Bool
  ) throws {
    index = try MetalANSFileIndex(sourceURL: sourceURL)
    descriptor = sourceURL.path.withCString { Darwin.open($0, O_RDONLY | O_CLOEXEC) }
    guard descriptor >= 0 else {
      throw MetalANSFileError.io("could not reopen \(sourceURL.lastPathComponent)")
    }
    do {
      if let expectedSHA256 {
        guard expectedSHA256.count == 64, expectedSHA256.allSatisfy({ $0.isHexDigit }) else {
          throw MetalANSFileError.invalid("expected_sha256 must be a hexadecimal SHA-256 digest")
        }
        var digest = SHA256()
        let chunkBytes = 32 * 1024 * 1024
        var offset: UInt64 = 0
        while offset < index.fileBytes {
          let count = Int(min(UInt64(chunkBytes), index.fileBytes - offset))
          let data = try MetalANSFileIndex.read(descriptor: descriptor, offset: offset, count: count)
          digest.update(data: data)
          offset += UInt64(count)
        }
        let actual = digest.finalize().map { String(format: "%02x", $0) }.joined()
        guard actual == expectedSHA256.lowercased() else {
          throw MetalANSFileError.invalid("QGANS whole-file SHA-256 does not match the expected identity")
        }
      }
    } catch {
      Darwin.close(descriptor)
      throw error
    }
    self.verifyChecksums = verifyChecksums
  }

  deinit {
    Darwin.close(descriptor)
  }

  let verifyChecksums: Bool

  func upload(
    section: MetalANSFileSection, device: MTLDevice, queue: MTLCommandQueue,
    stagingBytes: Int
  ) throws -> MTLBuffer {
    guard let resident = device.makeBuffer(
      length: max(4, section.byteCount), options: .storageModePrivate),
      let staging = device.makeBuffer(
        length: max(4, min(stagingBytes, max(4, section.byteCount))), options: .storageModeShared)
    else {
      throw MetalANSFileError.allocation("Metal could not allocate QGANS section \(section.name)")
    }
    var digest = SHA256()
    var offset = 0
    while offset < section.byteCount {
      let count = min(staging.length, section.byteCount - offset)
      try MetalANSFileIndex.read(
        into: staging.contents(), descriptor: descriptor,
        offset: section.offset + UInt64(offset), count: count)
      if verifyChecksums {
        digest.update(
          bufferPointer: UnsafeRawBufferPointer(start: staging.contents(), count: count))
      }
      guard let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
      else { throw MetalANSFileError.io("Metal could not stage QGANS section \(section.name)") }
      blit.copy(from: staging, sourceOffset: 0, to: resident, destinationOffset: offset, size: count)
      blit.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed else {
        throw MetalANSFileError.io(
          "QGANS upload failed: \(command.error?.localizedDescription ?? "unknown error")")
      }
      offset += count
    }
    if verifyChecksums {
      let actual = digest.finalize().map { String(format: "%02x", $0) }.joined()
      guard actual == section.sha256 else {
        throw MetalANSFileError.invalid("QGANS section \(section.name) SHA-256 mismatch")
      }
    }
    return resident
  }

}

enum MetalANSFileError: LocalizedError {
  case invalid(String)
  case io(String)
  case allocation(String)

  var errorDescription: String? {
    switch self {
    case .invalid(let message), .io(let message), .allocation(let message): message
    }
  }
}
