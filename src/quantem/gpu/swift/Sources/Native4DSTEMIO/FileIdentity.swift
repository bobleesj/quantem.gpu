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

private struct NativeContentSnapshot: Equatable {
  let device: UInt64
  let inode: UInt64
  let bytes: UInt64
  let modificationNanoseconds: UInt64
}

private func nativeContentSnapshot(for url: URL) throws -> NativeContentSnapshot {
  let identity = try nativeFileIdentity(for: url)
  return NativeContentSnapshot(
    device: identity.device,
    inode: identity.inode,
    bytes: identity.bytes,
    modificationNanoseconds: identity.modificationNanoseconds
  )
}

private func nativeSHA256(of url: URL) throws -> String {
  let handle = try FileHandle(forReadingFrom: url)
  defer { try? handle.close() }
  var digest = SHA256()
  while let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty {
    digest.update(data: data)
  }
  return digest.finalize().map { String(format: "%02x", $0) }.joined()
}

func nativeSourceHashes(master: URL?, dataFiles: [URL]) throws -> NativeSourceHashes {
  let hasMaster = master.map { FileManager.default.fileExists(atPath: $0.path) } ?? false
  let files = (hasMaster ? [master!] : []) + dataFiles
  let before = try files.map(nativeContentSnapshot)
  let hashes = try files.map(nativeSHA256)
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
  return NativeSourceHashes(
    master: masterHash,
    members: memberHashes,
    aggregate: aggregateText
  )
}
