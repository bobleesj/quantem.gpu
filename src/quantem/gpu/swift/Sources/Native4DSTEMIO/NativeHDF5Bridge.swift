import CNativeHDF5
import Foundation

struct NativeHDF5Stack {
  let frameCount: Int
  let detectorRows: Int
  let detectorColumns: Int
  let sourceBytes: Int
  let chunks: [NativeHDF5Chunk]

  var sourceDtype: String { sourceBytes == 1 ? "uint8" : "uint16" }
}

struct NativeHDF5Chunk {
  let offset: UInt64
  let size: UInt64
}

struct NativeHDF5Master {
  let externalFiles: [String]
  let expectedFrames: Int?
  let scanShape: (rows: Int, columns: Int)?
  let badPixelIndices: [Int]
  let reciprocalSampling: (row: Double, column: Double)?
  let acquisitionDate: String?
  let metadata: [String: String]
}

enum NativeHDF5Bridge {
  static func inspectStack(at url: URL, includeChunks: Bool) throws -> NativeHDF5Stack {
    var rawStack = qh5_stack_info()
    var rawChunks: UnsafeMutablePointer<qh5_chunk_info>?
    var rawChunkCount = 0
    var errorMessage: UnsafeMutablePointer<CChar>?
    let status = url.path.withCString {
      qh5_inspect_stack(
        $0,
        includeChunks ? 1 : 0,
        &rawStack,
        &rawChunks,
        &rawChunkCount,
        &errorMessage
      )
    }
    defer {
      qh5_free_chunks(rawChunks)
      qh5_free_error(errorMessage)
    }
    guard status == 0 else { throw hdf5Error(errorMessage) }
    let chunks = rawChunks.map {
      Array(UnsafeBufferPointer(start: $0, count: rawChunkCount)).map {
        NativeHDF5Chunk(offset: $0.offset, size: $0.size)
      }
    } ?? []
    return NativeHDF5Stack(
      frameCount: try exactInt(rawStack.frame_count, label: "frame count"),
      detectorRows: try exactInt(rawStack.detector_rows, label: "detector rows"),
      detectorColumns: try exactInt(rawStack.detector_columns, label: "detector columns"),
      sourceBytes: Int(rawStack.source_bytes),
      chunks: chunks
    )
  }

  static func inspectMaster(
    at url: URL,
    detectorRows: Int,
    detectorColumns: Int
  ) throws -> NativeHDF5Master {
    var raw = qh5_master_info()
    var errorMessage: UnsafeMutablePointer<CChar>?
    let status = url.path.withCString {
      qh5_inspect_master(
        $0,
        UInt64(detectorRows),
        UInt64(detectorColumns),
        &raw,
        &errorMessage
      )
    }
    defer {
      qh5_free_master_info(&raw)
      qh5_free_error(errorMessage)
    }
    guard status == 0 else { throw hdf5Error(errorMessage) }

    let externalFiles = strings(raw.external_files, count: raw.external_file_count)
    let badPixelIndices = raw.bad_pixel_indices.map {
      UnsafeBufferPointer(start: $0, count: raw.bad_pixel_count).map(Int.init)
    } ?? []
    var metadata: [String: String] = [:]
    if let items = raw.metadata {
      for item in UnsafeBufferPointer(start: items, count: raw.metadata_count) {
        guard let key = item.key, let value = item.value else { continue }
        metadata[String(cString: key)] = String(cString: value)
      }
    }
    let expectedFrames = raw.has_expected_frames != 0
      ? try exactInt(raw.expected_frames, label: "expected frame count")
      : nil
    let scanShape = raw.has_scan_shape != 0
      ? (
        rows: try exactInt(raw.scan_rows, label: "scan rows"),
        columns: try exactInt(raw.scan_columns, label: "scan columns")
      )
      : nil
    let reciprocalSampling = raw.has_reciprocal_sampling != 0
      ? (row: raw.reciprocal_row_mrad, column: raw.reciprocal_column_mrad)
      : nil
    return NativeHDF5Master(
      externalFiles: externalFiles,
      expectedFrames: expectedFrames,
      scanShape: scanShape,
      badPixelIndices: badPixelIndices,
      reciprocalSampling: reciprocalSampling,
      acquisitionDate: raw.acquisition_date.map { String(cString: $0) },
      metadata: metadata
    )
  }

  private static func strings(
    _ values: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
    count: Int
  ) -> [String] {
    guard let values else { return [] }
    return UnsafeBufferPointer(start: values, count: count).compactMap {
      $0.map { String(cString: $0) }
    }
  }

  private static func exactInt(_ value: UInt64, label: String) throws -> Int {
    guard let result = Int(exactly: value) else {
      throw Native4DSTEMIOError.invalidData("HDF5 \(label) exceeds this Mac's addressable range")
    }
    return result
  }

  private static func hdf5Error(
    _ message: UnsafeMutablePointer<CChar>?
  ) -> Native4DSTEMIOError {
    Native4DSTEMIOError.hdf5(
      message.map { String(cString: $0) } ?? "Native HDF5 inspection failed"
    )
  }
}
