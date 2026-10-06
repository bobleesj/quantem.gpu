// Production-library regression. CPU-constructed LZ4 bytes and bit-plane oracle
// are independent of both Metal decoder variants; no external acquisitions.
import Foundation
import Metal
import Metal4DSTEMKernels

func require(_ okay: Bool, _ reason: String) throws {
  if !okay { throw NSError(domain: reason, code: 1) }
}

do {
  guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    throw NSError(domain: "A Metal device is required", code: 1)
  }
  let library = try Metal4DSTEMKernels.makeHDF5Library(device: device)
  let unshuffle = try device.makeComputePipelineState(
    function:
      library.makeFunction(name: "h5unshuffle_u16_scalar_qh5idx")!)
  for decodeName in [
    "h5lz4dc_full_u16_scalar_qh5idx", "h5lz4dc_full_u16_aligned_fill_qh5idx",
    "h5lz4dc_full_u16_aligned_copy_qh5idx", "h5lz4dc_full_u16_aligned_fill_copy_qh5idx",
  ] {
    let decode = try device.makeComputePipelineState(
      function: library.makeFunction(name: decodeName)!)
    let original = (0..<8192).map { UInt8(truncatingIfNeeded: $0 * 13 + $0 / 97) }
    let valid: [UInt8] = [0xf0] + Array(repeating: 0xff, count: 32) + [17] + original
    let overflowMatch: [UInt8] = [0x1f, 7, 1, 0] + Array(repeating: 0xff, count: 32) + [13]
    var cases: [(String, [UInt8], Bool)] = [
      ("valid full literal block", valid, true),
      ("valid masked full literal block", valid, true),
      ("empty stream", [], false),
      ("literal beyond compressed bytes", Array(valid.dropLast()), false),
      ("unterminated literal extension", [0xf0, 0xff], false),
      ("missing offset byte", [0x10, 7, 1], false),
      ("zero match distance", [0x10, 7, 0, 0, 0], false),
      ("match outside prior history", [0x10, 7, 2, 0, 0], false),
      (
        "literal beyond decoded block",
        [0xf0] + Array(repeating: 0xff, count: 32) + [18] + original + [1], false
      ),
      ("match beyond decoded block", overflowMatch, false),
      ("unterminated match extension", [0x1f, 7, 1, 0, 0xff], false),
      ("incomplete decoded block", [0x10, 7], false),
      ("bytes after complete block", valid + [0], false),
    ]
    var expectedBlocks: [String: [UInt8]] = [:]

    do {
      func extensionBytes(_ value: Int) -> [UInt8] {
        var remaining = value
        var bytes: [UInt8] = []
        while remaining >= 255 {
          bytes.append(255)
          remaining -= 255
        }
        bytes.append(UInt8(remaining))
        return bytes
      }
      func literalSequence(_ bytes: [UInt8]) -> [UInt8] {
        [UInt8(min(15, bytes.count) << 4)]
          + (bytes.count >= 15 ? extensionBytes(bytes.count - 15) : []) + bytes
      }
      let patterns: [(String, [UInt8])] = [
        ("offset1-zero", [0]), ("offset1-nonzero", [0xd3]),
        ("offset2-zero", [0, 0]), ("offset2-nonzero", [0x3d, 0xa7]),
      ]
      for (label, pattern) in patterns {
        for alignment in 0..<16 {
          for requestedLength in [4, 31, 63, 64, 65, 257, 0] {
            let literalCount = 16 + alignment
            let matchCount = requestedLength == 0 ? 8192 - literalCount - 8 : requestedLength
            var prefix = (0..<literalCount).map { UInt8(truncatingIfNeeded: $0 * 17 + 5) }
            prefix.replaceSubrange((prefix.count - pattern.count)..<prefix.count, with: pattern)
            let matched = (0..<matchCount).map { pattern[$0 % pattern.count] }
            let tail = (0..<(8192 - literalCount - matchCount)).map {
              UInt8(truncatingIfNeeded: $0 * 29 + 3)
            }
            var encoded = [UInt8((min(15, literalCount) << 4) | min(15, matchCount - 4))]
            if literalCount >= 15 { encoded += extensionBytes(literalCount - 15) }
            encoded += prefix
            encoded += [UInt8(pattern.count), 0]
            if matchCount - 4 >= 15 { encoded += extensionBytes(matchCount - 19) }
            encoded += literalSequence(tail)
            let name = "\(label) alignment\(alignment) match\(matchCount)"
            expectedBlocks[name] = prefix + matched + tail
            cases.append((name, encoded, true))
          }
        }
      }

      for distance in [16, 17, 31, 32, 48, 512, 513, 1024] {
        for alignment in 0..<16 {
          for requestedLength in [16, 31, 63, 64, 65, 257, 0] {
            let literalCount = distance + alignment
            let matchCount = requestedLength == 0 ? 8192 - literalCount - 8 : requestedLength
            let prefix = (0..<literalCount).map { UInt8(truncatingIfNeeded: $0 * 17 + $0 / 97 + 5) }
            let matched = (0..<matchCount).map { prefix[prefix.count - distance + $0 % distance] }
            let tail = (0..<(8192 - literalCount - matchCount)).map {
              UInt8(truncatingIfNeeded: $0 * 29 + 3)
            }
            var encoded = [UInt8((min(15, literalCount) << 4) | min(15, matchCount - 4))]
            if literalCount >= 15 { encoded += extensionBytes(literalCount - 15) }
            encoded += prefix
            encoded += [UInt8(truncatingIfNeeded: distance), UInt8(distance >> 8)]
            if matchCount - 4 >= 15 { encoded += extensionBytes(matchCount - 19) }
            encoded += literalSequence(tail)
            let name = "offset\(distance)-copy alignment\(alignment) match\(matchCount)"
            expectedBlocks[name] = prefix + matched + tail
            cases.append((name, encoded, true))
          }
        }
      }
    }
    try require(cases.count == 1357, "Frozen alignment/copy corpus changed")
    // Additional guards do not replace any of the original 461 cases.
    cases.append(("zero distance with wide-length match", [0x1f, 7, 0, 0, 45], false))
    let saturated = Array(repeating: UInt8(255), count: 8192)
    expectedBlocks["saturated uint16"] = saturated
    cases.append(
      ("saturated uint16", [0xf0] + Array(repeating: 0xff, count: 32) + [17] + saturated, true))
    for (name, bytes, accepted) in cases {
      try autoreleasepool {
        let storage = bytes.isEmpty ? [UInt8(0)] : bytes
        let compressed = storage.withUnsafeBytes {
          device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        let words: [UInt32] = [0, UInt32(bytes.count)]
        let metadata = words.withUnsafeBytes {
          device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
        let scratch = device.makeBuffer(length: 8192 + 512, options: .storageModeShared)!
        memset(scratch.contents(), 0x6b, scratch.length)
        let output = device.makeBuffer(length: 8192 + 512, options: .storageModeShared)!
        let mask = device.makeBuffer(length: 4096, options: .storageModeShared)!
        let audit = device.makeBuffer(length: 8, options: .storageModeShared)!
        let errors = device.makeBuffer(length: 4, options: .storageModeShared)!
        memset(output.contents(), 0xa5, output.length)
        memset(mask.contents(), 0, mask.length)
        if name.contains("masked") {
          let maskBytes = mask.contents().assumingMemoryBound(to: UInt8.self)
          for pixel in stride(from: 0, to: mask.length, by: 7) { maskBytes[pixel] = 1 }
        }
        memset(audit.contents(), 0, audit.length)
        memset(errors.contents(), 0, errors.length)
        var zero64: UInt64 = 0
        var zero: UInt32 = 0
        var one: UInt32 = 1
        var pixels: UInt32 = 4096
        let command = queue.makeCommandBuffer()!
        let decoder = command.makeComputeCommandEncoder()!
        decoder.setComputePipelineState(decode)
        decoder.setBuffer(compressed, offset: 0, index: 0)
        decoder.setBuffer(metadata, offset: 0, index: 1)
        decoder.setBytes(&zero64, length: 8, index: 2)
        decoder.setBytes(&one, length: 4, index: 3)
        decoder.setBytes(&pixels, length: 4, index: 4)
        decoder.setBuffer(scratch, offset: 256, index: 5)
        decoder.setBytes(&zero, length: 4, index: 6)
        decoder.setBuffer(errors, offset: 0, index: 10)
        decoder.setBytes(&one, length: 4, index: 11)
        // Deliberately oversized launch verifies the per-thread block bound.
        decoder.dispatchThreads(
          MTLSize(width: 128, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        decoder.endEncoding()
        let image = command.makeComputeCommandEncoder()!
        image.setComputePipelineState(unshuffle)
        image.setBuffer(scratch, offset: 256, index: 0)
        image.setBytes(&one, length: 4, index: 3)
        image.setBytes(&pixels, length: 4, index: 4)
        image.setBuffer(output, offset: 256, index: 5)
        image.setBuffer(mask, offset: 0, index: 7)
        image.setBuffer(audit, offset: 0, index: 8)
        image.setBytes(&zero, length: 4, index: 9)
        image.setBuffer(errors, offset: 0, index: 10)
        image.setBytes(&one, length: 4, index: 11)

        image.dispatchThreadgroups(
          MTLSize(width: 2, height: 1, depth: 2),
          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        image.endEncoding()

        command.commit()
        command.waitUntilCompleted()
        if let error = command.error { throw error }
        for (buffer, fill) in [(scratch, UInt8(0x6b)), (output, UInt8(0xa5))] {
          let bytes = buffer.contents().assumingMemoryBound(to: UInt8.self)
          try require(
            (0..<256).allSatisfy { bytes[$0] == fill && bytes[8448 + $0] == fill },
            "buffer canary changed for \(name)")
        }
        let error = errors.contents().load(as: UInt32.self)
        try require((error == 0) == accepted, "incorrect validity result for \(name)")
        if accepted {
          let expectedBlock = expectedBlocks[name] ?? original
          let decoded = scratch.contents().advanced(by: 256).assumingMemoryBound(to: UInt8.self)
          try require(
            (0..<8192).allSatisfy { decoded[$0] == expectedBlock[$0] },
            "decoded bytes differ for \(name)")
          let result = output.contents().advanced(by: 256).bindMemory(
            to: UInt16.self, capacity: 4096)
          for pixel in 0..<4096 {
            var expected: UInt16 = 0
            for bit in 0..<16 {
              let byte = expectedBlock[bit * 512 + pixel / 8]
              expected |= UInt16((byte >> (pixel % 8)) & 1) << bit
            }
            if mask.contents().assumingMemoryBound(to: UInt8.self)[pixel] != 0 { expected = 0 }
            try require(result[pixel] == expected, "literal block decode differs at \(pixel)")
          }
          let auditWords = audit.contents().assumingMemoryBound(to: UInt32.self)
          var expectedMaximum: UInt32 = 0
          var expectedAbove: UInt32 = 0
          for pixel in 0..<4096 {
            expectedMaximum = max(expectedMaximum, UInt32(result[pixel]))
            expectedAbove += result[pixel] > 255 ? 1 : 0
          }
          try require(
            auditWords[0] == expectedMaximum && auditWords[1] == expectedAbove,
            "count audit differs from exact masked counts")

        } else {
          let untouched = UnsafeBufferPointer(
            start: output.contents().assumingMemoryBound(to: UInt8.self), count: output.length)
          try require(
            untouched.allSatisfy { $0 == 0xa5 }, "unshuffle consumed failed scratch in \(name)")

        }
        print("PASS \(name)")
      }
    }
    try require(cases.count == 1359, "Additional guard corpus changed")
    print("PASS \(decodeName) all \(cases.count) checked-stream cases")
  }
  print("PRODUCTION_DECODER_5436_CASES_PASS")
} catch {
  fputs("ERROR: \(error)\n", stderr)
  exit(1)
}
