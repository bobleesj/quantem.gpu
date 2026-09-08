import Foundation
import CryptoKit
import CommonCrypto

// A bounded full-original-file probe. Read and digest times are separate;
// this is not a packed-resident or native-presentation benchmark.
let path = CommandLine.arguments[1]
let expected = "5f5bbae2295aea62a1d6bd74c6611ae7c913d473c3ae83435fc85eda87abd325"
for implementation in ["cryptokit-a", "commoncrypto", "cryptokit-b"] {
  for repetition in 0..<3 {
    let stream = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
    var crypto = SHA256()
    var common = CC_SHA256_CTX()
    CC_SHA256_Init(&common)
    var reading = 0.0, hashing = 0.0, size = 0
    let started = CFAbsoluteTimeGetCurrent()
    while true {
      let count = try autoreleasepool {
        let readStarted = CFAbsoluteTimeGetCurrent()
        let block = try stream.read(upToCount: 32 * 1024 * 1024) ?? Data()
        reading += CFAbsoluteTimeGetCurrent() - readStarted
        let hashStarted = CFAbsoluteTimeGetCurrent()
        block.withUnsafeBytes { bytes in
          if implementation == "commoncrypto" {
            CC_SHA256_Update(&common, bytes.baseAddress, CC_LONG(bytes.count))
          } else {
            crypto.update(bufferPointer: bytes)
          }
        }
        hashing += CFAbsoluteTimeGetCurrent() - hashStarted
        return block.count
      }
      if count == 0 { break }
      size += count
    }
    try stream.close()
    let digest: String
    if implementation == "commoncrypto" {
      var bytes = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
      CC_SHA256_Final(&bytes, &common)
      digest = bytes.map { String(format: "%02x", $0) }.joined()
    } else {
      digest = crypto.finalize().map { String(format: "%02x", $0) }.joined()
    }
    precondition(digest == expected && size == 4_362_076_160)
    let record: [String: Any] = ["implementation": implementation,
      "repetition": repetition, "bytes": size, "read_s": reading,
      "hash_s": hashing, "total_s": CFAbsoluteTimeGetCurrent() - started,
      "sha256": digest]
    let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
    fflush(stdout)
  }
}
