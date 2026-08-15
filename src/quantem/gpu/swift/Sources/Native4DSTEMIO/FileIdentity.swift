import CryptoKit
import Darwin
import Foundation

struct NativeFileIdentity {
  let path: String
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
    bytes: UInt64(status.st_size),
    modificationNanoseconds: seconds * 1_000_000_000 + nanoseconds
  )
}

func nativeDatasetSignature(for files: [URL]) throws -> String {
  var digest = SHA256()
  for file in files {
    let identity = try nativeFileIdentity(for: file)
    digest.update(data: Data(identity.path.utf8))
    var bytes = identity.bytes.littleEndian
    var modificationNanoseconds = identity.modificationNanoseconds.littleEndian
    withUnsafeBytes(of: &bytes) { digest.update(bufferPointer: $0) }
    withUnsafeBytes(of: &modificationNanoseconds) {
      digest.update(bufferPointer: $0)
    }
  }
  return digest.finalize().prefix(10).map { String(format: "%02x", $0) }.joined()
}
