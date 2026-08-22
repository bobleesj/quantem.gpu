import CryptoKit
import Darwin
import Foundation

struct NativeFileIdentity {
  let path: String
  let device: UInt64
  let inode: UInt64
  let bytes: UInt64
  let modificationNanoseconds: UInt64
}

func nativeCanonicalURL(_ input: URL) -> URL {
  input.standardizedFileURL
}

func nativeFileIdentity(for input: URL) throws -> NativeFileIdentity {
  let url = nativeCanonicalURL(input)
  var status = stat()
  let result = url.path.withCString { Darwin.lstat($0, &status) }
  guard result == 0, status.st_size >= 0, status.st_mtimespec.tv_sec >= 0,
    status.st_mtimespec.tv_nsec >= 0
  else {
    throw Native4DSTEMIOError.invalidData("Could not inspect \(url.path)")
  }
  let seconds = UInt64(status.st_mtimespec.tv_sec)
  let nanoseconds = UInt64(status.st_mtimespec.tv_nsec)
  return NativeFileIdentity(
    path: url.path,
    device: UInt64(status.st_dev),
    inode: UInt64(status.st_ino),
    bytes: UInt64(status.st_size),
    modificationNanoseconds: seconds * 1_000_000_000 + nanoseconds
  )
}

func nativeDatasetSignature(for files: [URL]) throws -> String {
  var digest = SHA256()
  for file in files {
    let identity = try nativeFileIdentity(for: file)
    digest.update(data: Data(identity.path.utf8))
    var device = identity.device.littleEndian
    var inode = identity.inode.littleEndian
    var bytes = identity.bytes.littleEndian
    var modificationNanoseconds = identity.modificationNanoseconds.littleEndian
    withUnsafeBytes(of: &device) { digest.update(bufferPointer: $0) }
    withUnsafeBytes(of: &inode) { digest.update(bufferPointer: $0) }
    withUnsafeBytes(of: &bytes) { digest.update(bufferPointer: $0) }
    withUnsafeBytes(of: &modificationNanoseconds) {
      digest.update(bufferPointer: $0)
    }
  }
  return digest.finalize().prefix(10).map { String(format: "%02x", $0) }.joined()
}

struct NativeSourceHashes {
  let master: String?
  let members: [String]
  let aggregate: String
}

private struct NativeContentSnapshot: Codable, Equatable {
  let path: String
  let device: UInt64
  let inode: UInt64
  let bytes: UInt64
  let modificationNanoseconds: UInt64
}

private func nativeContentSnapshot(for url: URL) throws -> NativeContentSnapshot {
  let identity = try nativeFileIdentity(for: url)
  return NativeContentSnapshot(
    path: identity.path,
    device: identity.device,
    inode: identity.inode,
    bytes: identity.bytes,
    modificationNanoseconds: identity.modificationNanoseconds
  )
}

private func nativeSHA256(of url: URL) throws -> String {
  let descriptor = Darwin.open(url.path, O_RDONLY)
  guard descriptor >= 0 else {
    throw Native4DSTEMIOError.invalidData(
      "Could not open \(url.path) for exact source hashing"
    )
  }
  defer { Darwin.close(descriptor) }
  if ProcessInfo.processInfo.environment[
    "QUANTEM_GPU_BENCHMARK_UNCACHED_SOURCE_READS"
  ] == "1" {
    guard Darwin.fcntl(descriptor, F_NOCACHE, 1) == 0 else {
      throw Native4DSTEMIOError.invalidData(
        "Could not enable uncached benchmark reads for \(url.path)"
      )
    }
  }
  let bufferBytes = 8 * 1024 * 1024
  let buffer = UnsafeMutableRawPointer.allocate(
    byteCount: bufferBytes,
    alignment: Int(getpagesize())
  )
  defer { buffer.deallocate() }
  var digest = SHA256()
  while true {
    let count = Darwin.read(descriptor, buffer, bufferBytes)
    if count < 0 && errno == EINTR { continue }
    guard count >= 0 else {
      throw Native4DSTEMIOError.invalidData(
        "Could not read \(url.path) for exact source hashing"
      )
    }
    if count == 0 { break }
    digest.update(
      bufferPointer: UnsafeRawBufferPointer(start: buffer, count: count)
    )
  }
  return digest.finalize().map { String(format: "%02x", $0) }.joined()
}

private func nativeSHA256(of files: [URL]) throws -> [String] {
  guard !files.isEmpty else { return [] }
  let workerCount = min(8, files.count)
  let resultLock = NSLock()
  nonisolated(unsafe) var results = [Result<String, Error>?](
    repeating: nil,
    count: files.count
  )

  DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
    for index in stride(from: worker, to: files.count, by: workerCount) {
      let result = Result { try nativeSHA256(of: files[index]) }
      resultLock.lock()
      results[index] = result
      resultLock.unlock()
    }
  }
  return try results.enumerated().map { index, result in
    guard let result else {
      throw Native4DSTEMIOError.invalidData(
        "Could not verify the exact identity of \(files[index].path)"
      )
    }
    return try result.get()
  }
}

private struct NativeSourceHashCache: Codable {
  let schema: String
  let snapshots: [NativeContentSnapshot]
  let masterHash: String?
  let memberHashes: [String]
  let aggregateHash: String
}

private let nativeSourceHashCacheSchema = "quantem.gpu.source-hashes/v1"
private let compatibleNativeSourceHashCacheSchemas = [
  nativeSourceHashCacheSchema,
  "live4dstem.source-hashes/v1",
]

func nativeSourceHashes(
  master: URL?,
  dataFiles: [URL],
  cacheFile: URL? = nil
) throws -> NativeSourceHashes {
  let hasMaster = master.map { FileManager.default.fileExists(atPath: $0.path) } ?? false
  let files = (hasMaster ? [master!] : []) + dataFiles
  let before = try files.map(nativeContentSnapshot)
  if let cacheFile,
    let data = try? Data(contentsOf: cacheFile),
    let cached = try? JSONDecoder().decode(NativeSourceHashCache.self, from: data),
    compatibleNativeSourceHashCacheSchemas.contains(cached.schema),
    cached.snapshots == before,
    cached.memberHashes.count == dataFiles.count,
    (cached.masterHash != nil) == hasMaster
  {
    return NativeSourceHashes(
      master: cached.masterHash,
      members: cached.memberHashes,
      aggregate: cached.aggregateHash
    )
  }
  let hashes = try nativeSHA256(of: files)
  guard before == (try files.map(nativeContentSnapshot)) else {
    throw Native4DSTEMIOError.invalidData(
      "A source file changed while its exact identity was being verified; retry"
    )
  }
  let masterHash = hasMaster ? hashes[0] : nil
  let memberHashes = hasMaster ? Array(hashes.dropFirst()) : hashes
  var aggregate = SHA256()
  aggregate.update(data: Data("live4dstem.dataset/v0.1\0".utf8))
  if let masterHash { aggregate.update(data: Data(masterHash.utf8)) }
  for hash in memberHashes {
    aggregate.update(data: Data([0]))
    aggregate.update(data: Data(hash.utf8))
  }
  let aggregateText = aggregate.finalize().map { String(format: "%02x", $0) }.joined()
  let result = NativeSourceHashes(
    master: masterHash,
    members: memberHashes,
    aggregate: aggregateText
  )
  if let cacheFile {
    let cached = NativeSourceHashCache(
      schema: nativeSourceHashCacheSchema,
      snapshots: before,
      masterHash: masterHash,
      memberHashes: memberHashes,
      aggregateHash: aggregateText
    )
    let data = try JSONEncoder().encode(cached)
    try FileManager.default.createDirectory(
      at: cacheFile.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    try data.write(to: cacheFile, options: .atomic)
  }
  return result
}
