import Foundation
import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMKernels
import XCTest

final class MetalTANSDecoderLiteralTests: XCTestCase {
  func testPairLookupPreservesLiteralEscapeZeroChainAndSignedCounts() throws {
    for bits in [4, 6] {
      try verifyEveryTopology(literal: true, pairLookupBits: bits)
      try verifyEveryTopology(literal: false, pairLookupBits: bits)
      try verifyEveryTopology(literal: false, pairLookupBits: bits, ordinaryPairs: true)
      try verifyEveryTopology(literal: false, zeroRuns: true, pairLookupBits: bits)
      try verifyEveryTopology(literal: false, mixedModels: true, pairLookupBits: bits)
    }
  }
  func testPacketMajorGridPreservesLiteralEscapeAndZeroChainCounts() throws {
    try verifyEveryTopology(literal: true, packetMajor: true)
    try verifyEveryTopology(literal: false, packetMajor: true)
    try verifyEveryTopology(literal: false, mixedModels: true, packetMajor: true)
  }
  func testBitExtractionPreservesLiteralEscapeAndZeroChainCounts() throws {
    try verifyEveryTopology(literal: true, bitExtract: true)
    try verifyEveryTopology(literal: false, bitExtract: true)
    try verifyEveryTopology(literal: false, mixedModels: true, bitExtract: true)
  }
  func testDeferredReductionPreservesLiteralEscapeAndZeroChainCounts() throws {
    try verifyEveryTopology(literal: true, deferred: true)
    try verifyEveryTopology(literal: false, deferred: true)
    try verifyEveryTopology(literal: false, mixedModels: true, deferred: true)
  }
  func testPrefetchedTransitionPreservesLiteralEscapeAndZeroChainCounts() throws {
    try verifyEveryTopology(literal: true, prefetch: true)
    try verifyEveryTopology(literal: false, prefetch: true)
    try verifyEveryTopology(literal: false, mixedModels: true, prefetch: true)
  }
  func testStagedReductionPreservesFullLiteralAndMixedCounts() throws {
    try verifyEveryTopology(literal: true, staged: true)
    try verifyEveryTopology(literal: false, mixedModels: true, staged: true)
  }
  func testConcurrentAtomicDispatchesPreserveLiteralAndMixedCounts() throws {
    try verifyEveryTopology(literal: true, concurrent: true)
    try verifyEveryTopology(literal: false, mixedModels: true, concurrent: true)
  }
  /// Independent small high-count oracle; never an app raw-resident fallback.
  func testEveryExperimentalTopologyPreservesFullUInt16LiteralCounts() throws {
    try verifyEveryTopology(literal: true)
  }

  func testPairedReductionPreservesEscapesAndSignedSixBitCounts() throws {
    try verifyEveryTopology(literal: false)
  }

  func testDeterministicZeroRunsPreserveScanPositionsAndSignedCounts() throws {
    try verifyEveryTopology(literal: false, zeroRuns: true)
  }

  func testMixedModelTailsPreserveLiteralEscapeAndZeroChainCounts() throws {
    try verifyEveryTopology(literal: false, mixedModels: true)
  }

  func testSignedPairBorrowCorrectionAtExactReductionBounds() {
    for a in [-2016, -1024, -1, 0, 1, 1024, 2016] {
      for b in [-2016, -1024, -1, 0, 1, 1024, 2016] {
        let packed = UInt32(truncatingIfNeeded: Int64(a) + 65536 * Int64(b))
        let low = Int(Int16(bitPattern: UInt16(truncatingIfNeeded: packed)))
        let corrected = (packed >> 16) + UInt32(low < 0 ? 1 : 0)
        let high = Int(Int16(bitPattern: UInt16(truncatingIfNeeded: corrected)))
        XCTAssertEqual(low, a)
        XCTAssertEqual(high, b)
      }
    }
  }

  private func verifyEveryTopology(
    literal: Bool, zeroRuns: Bool = false, mixedModels: Bool = false, concurrent: Bool = false,
    staged: Bool = false, prefetch: Bool = false, deferred: Bool = false,
    bitExtract: Bool = false, packetMajor: Bool = false, pairLookupBits: Int = 0,
    ordinaryPairs: Bool = false
  ) throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    let baselineFunction = try XCTUnwrap(library.makeFunction(name: "tans_detector_batch"))
    let arguments = baselineFunction.makeArgumentEncoder(bufferIndex: 0)
    XCTAssertEqual(arguments.encodedLength, 40)
    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
      try values.withUnsafeBytes { bytes in
        try XCTUnwrap(
          device.makeBuffer(
            bytes: bytes.baseAddress!, length: bytes.count,
            options: .storageModeShared))
      }
    }
    func isLiteral(_ col: Int) -> Bool { literal || (mixedModels && col % 3 == 0) }
    func usesZeroRun(_ col: Int) -> Bool { zeroRuns || (mixedModels && col % 3 == 2) }
    func value(_ scan: Int, _ col: Int) -> UInt32 {
      if ordinaryPairs { return scan % 2 == 0 ? 3 : 4 }
      if usesZeroRun(col) && (scan % 512 / 2) % 92 != (col * 7) % 92 { return 0 }
      if !isLiteral(col) { return UInt32((scan * 13 + col * 7) & 63) }
      if (scan + col) % 7 == 0 { return 65535 }
      return UInt32((scan * 131 + col * 8191) & 65535)
    }
    let columns = 32
    let scans = 16384
    var payload: [UInt32] = []
    for packet in 0..<32 {
      for col in 0..<columns {
        if isLiteral(col) {
          for pair in 0..<256 {
            let scan = packet * 512 + pair * 2
            payload.append(value(scan, col) | (value(scan + 1, col) << 16))
          }
        } else {
          // State zero, zero transition bits, every symbol is a 12-bit escape.
          // Independently encode the bitstream after the initial 10 state bits.
          var words = [UInt32](repeating: 0, count: mixedModels ? 256 : 97)
          var nextBit = 10
          if usesZeroRun(col) { words[0] = UInt32((col * 7) % 92) }
          for pair in 0..<256 {
            if ordinaryPairs {
              let bit = 10 + pair * 2
              words[bit / 32] |= UInt32((col + pair) & 3) << (bit % 32)
              continue
            }
            if usesZeroRun(col) && pair % 92 != (col * 7) % 92 { continue }
            let scan = packet * 512 + pair * 2
            let symbol = value(scan, col) | (value(scan + 1, col) << 6)
            // Zero-run oracle: one zero transition bit then an exact escape.
            let bit = usesZeroRun(col) ? nextBit + 1 : 10 + pair * 12
            let word = bit / 32
            let shift = bit % 32
            words[word] |= symbol << shift
            if shift > 20 { words[word + 1] |= symbol >> (32 - shift) }
            if usesZeroRun(col) { nextBit += 13 }
          }
          payload += words
        }
      }
    }
    // Dense format stores 32-stream checkpoints plus length-minus-one bytes.
    let wordsPerStream = literal || mixedModels ? 256 : 97
    let lengthByte = UInt32(wordsPerStream - 1)
    let offsets =
      (0..<32).map { UInt32($0 * columns * wordsPerStream) }
      + [UInt32](repeating: lengthByte * 0x0101_0101, count: 32 * 8)
    let encoded = try buffer(payload)
    let seeks = try buffer(offsets)
    let empty = try buffer([UInt32(0)])
    let zeroTable: [UInt32] = (0..<1024).map { state -> UInt32 in
      if state == 0 { return (UInt32(91) << 16) | UInt32(8191) }
      return UInt32(state - 1) << 16
    }
    let decoding = try buffer(
      ordinaryPairs
        ? [UInt32](repeating: (2 << 12) | (4 << 6) | 3, count: 1024)
        : mixedModels
          ? [UInt32](repeating: 4095, count: 1024) + zeroTable
          : zeroRuns
            ? zeroTable
            : [UInt32](repeating: literal ? 0 : 4095, count: 1024))
    let models = try buffer(
      mixedModels
        ? (0..<36864).map { UInt8($0 % 3 == 0 ? 255 : ($0 % 3 == 1 ? 0 : 1)) }
        : [UInt8](repeating: literal ? 255 : 0, count: 36864))
    let map = try buffer([Int32](repeating: -1, count: 36864))
    let ranks = try buffer((0..<36864).map(UInt32.init))
    let modelOffsets = try buffer([UInt32(0)])
    let descriptors = try buffer([UInt32(literal ? 255 : 0), 0])
    let mixedDescriptors = try buffer([UInt32(256), 0])
    let counts = try buffer([UInt32(1)])
    let table = try XCTUnwrap(device.makeBuffer(length: 40, options: .storageModeShared))
    var query: [UInt32] = [32, 0, 0, 32, 1]
    let modes =
      pairLookupBits > 0
      ? (mixedModels ? [102] : [101, 102])
      : packetMajor
        ? (mixedModels ? [99] : [99, 100])
        : bitExtract
          ? (mixedModels ? [97] : [97, 98])
          : deferred
            ? (mixedModels ? [93, 94, 95] : [93, 94, 95, 96])
            : prefetch
              ? (mixedModels ? [91] : [91, 92])
              : staged
                ? [90]
                : concurrent
                  ? [78]
                  : mixedModels
                    ? [0, 1, 2, 4, 8, 77, 78, 79, 80, 82, 83, 84, 86, 88]
                    : [
                      0, 1, 2, 4, 8, 32, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                      79,
                      80,
                      81,
                      82, 83, 84, 85, 86, 87, 88,
                    ]
    for mode in modes {
      let preparedZero = [80, 81].contains(mode)
      let mixedTopology = mode >= 77 && ![81, 85, 87, 92, 96, 98, 100, 101].contains(mode)
      let packets = mode == 71 ? 2 : (mode == 72 ? 4 : 1)
      let function: MTLFunction
      if mode == 0 {
        function = baselineFunction
      } else if mode >= 32 {
        var pair = [64, 66, 67, 68, 69, 70].contains(mode)
        var word32 = [65, 66, 67].contains(mode)
        var threads = UInt32([67, 70].contains(mode) ? 512 : 128)
        let constants = MTLFunctionConstantValues()
        constants.setConstantValue(&pair, type: .bool, index: 2)
        constants.setConstantValue(&word32, type: .bool, index: 3)
        constants.setConstantValue(&threads, type: .uint, index: 4)
        var packetCount = UInt32(packets)
        constants.setConstantValue(&packetCount, type: .uint, index: 5)
        var pairs: UInt32 = mode >= 69 ? 4 : 1
        constants.setConstantValue(&pairs, type: .uint, index: 6)
        var refill16 = mode == 73
        constants.setConstantValue(&refill16, type: .bool, index: 7)
        var directTable = mode == 74
        constants.setConstantValue(&directTable, type: .bool, index: 8)
        var zeroRunDecoder = mode == 75 || preparedZero
        constants.setConstantValue(&zeroRunDecoder, type: .bool, index: 9)
        var zeroArithmetic = mode == 76
        constants.setConstantValue(&zeroArithmetic, type: .bool, index: 10)
        var mixedTails = mixedTopology
        constants.setConstantValue(&mixedTails, type: .bool, index: 11)
        var mixedOnly = mode >= 78 && ![81, 85, 87, 92, 96, 98, 100, 101].contains(mode)
        constants.setConstantValue(&mixedOnly, type: .bool, index: 12)
        var signedPair = mode == 79
        constants.setConstantValue(&signedPair, type: .bool, index: 13)
        var prepared = preparedZero
        constants.setConstantValue(&prepared, type: .bool, index: 14)
        var unroll: UInt32 = mode == 82 ? 2 : (mode == 83 || mode == 85 ? 4 : (mode == 84 ? 8 : 1))
        constants.setConstantValue(&unroll, type: .uint, index: 15)
        var stagedReduction = staged
        constants.setConstantValue(&stagedReduction, type: .bool, index: 16)
        var prefetchCode = prefetch
        constants.setConstantValue(&prefetchCode, type: .bool, index: 17)
        var deferredPairs: UInt32 =
          mode == 93 ? 2 : ([94, 96].contains(mode) ? 4 : (mode == 95 ? 8 : 1))
        constants.setConstantValue(&deferredPairs, type: .uint, index: 18)
        var extract = bitExtract
        constants.setConstantValue(&extract, type: .bool, index: 19)
        var packetFirst = packetMajor
        constants.setConstantValue(&packetFirst, type: .bool, index: 20)
        var lookupBits = UInt32(pairLookupBits)
        constants.setConstantValue(&lookupBits, type: .uint, index: 21)
        function = try library.makeFunction(
          name: packets > 1
            ? "tans_detector_shared_packet_ilp_batch"
            : ([68, 69, 70].contains(mode)
              ? "tans_detector_cuda_funnel_batch" : "tans_detector_shared_model_batch"),
          constantValues: constants)
      } else {
        var streams = UInt32(mode == 8 ? 1 : mode)
        var coalesced = mode == 8
        let constants = MTLFunctionConstantValues()
        constants.setConstantValue(&streams, type: .uint, index: 0)
        constants.setConstantValue(&coalesced, type: .bool, index: 1)
        function = try library.makeFunction(
          name: "tans_detector_interleaved_batch", constantValues: constants)
      }
      let pipeline: MTLComputePipelineState
      if [86, 87, 88].contains(mode) {
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.maxTotalThreadsPerThreadgroup = mode == 88 ? 256 : 128
        pipeline = try device.makeComputePipelineState(
          descriptor: descriptor, options: [], reflection: nil)
        XCTAssertEqual(pipeline.maxTotalThreadsPerThreadgroup, mode == 88 ? 256 : 128)
      } else {
        pipeline = try device.makeComputePipelineState(function: function)
      }
      if mode >= 77 {
        print(
          "MIXED_PIPELINE mode=\(mode) static_threadgroup_bytes=\(pipeline.staticThreadgroupMemoryLength) max_threads=\(pipeline.maxTotalThreadsPerThreadgroup)"
        )
      }
      if mode == 78 { XCTAssertEqual(pipeline.staticThreadgroupMemoryLength, 0) }
      for scenario in 0..<(mode >= 79 ? 7 : 2) {
        let removeHalf = scenario == 1
        func coefficient(_ col: Int) -> Int32 {
          switch scenario {
          case 0: return 1
          case 1: return col < 16 ? -1 : 0
          case 2: return col % 2 == 0 ? 1 : -1
          case 3: return col % 2 == 0 ? -1 : 1
          case 4: return -1
          case 5: return col % 3 == 0 ? 0 : (col % 3 == 1 ? -1 : 1)
          default: return col < 27 ? -1 : 1
          }
        }
        // Group uniform negative signs and mark the other lanes inactive for paired reduction.
        let selected = try buffer(
          (0..<32).map {
            UInt32(
              (removeHalf && $0 >= 16 && mode >= 64 && mode != 65)
                || (scenario == 5 && $0 % 3 == 0) ? UInt32.max : UInt32($0))
          })
        let seed: [UInt32] = (0..<scans).map { scan in
          scenario != 0 ? (0..<32).reduce(UInt32(0)) { $0 + value(scan, $1) } : 0
        }
        let expected: [UInt32] = (0..<scans).map { scan in
          let delta = (0..<32).reduce(Int64(0)) {
            $0 + Int64(value(scan, $1)) * Int64(coefficient($1))
          }
          return UInt32(Int64(seed[scan]) + delta)
        }
        let signs = try buffer((0..<32).map(coefficient))
        let output = try buffer(seed)
        arguments.setArgumentBuffer(table, offset: 0)
        for (index, resource) in [encoded, seeks, empty, empty, output].enumerated() {
          arguments.setBuffer(resource, offset: 0, index: index)
        }
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(
          command.makeComputeCommandEncoder(
            dispatchType: concurrent ? .concurrent : .serial))
        var tableForDecoding = decoding
        var pairLookup: MTLBuffer?
        if pairLookupBits > 0 {
          let entries = (decoding.length / 4) << pairLookupBits
          let prepared = try buffer([UInt32](repeating: 0xDEAD_BEEF, count: entries))
          let prepare = try XCTUnwrap(library.makeFunction(name: "tans_prepare_pair_lookup"))
          encoder.setComputePipelineState(try device.makeComputePipelineState(function: prepare))
          encoder.setBuffer(decoding, offset: 0, index: 0)
          encoder.setBuffer(prepared, offset: 0, index: 1)
          var info = [UInt32(entries), UInt32(pairLookupBits)]
          encoder.setBytes(&info, length: 8, index: 2)
          encoder.dispatchThreads(
            MTLSize(width: entries, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          encoder.memoryBarrier(scope: .buffers)
          pairLookup = prepared
        }
        if preparedZero {
          let folded = try buffer([UInt32](repeating: 0xDEAD_BEEF, count: decoding.length / 4 + 16))
          let prepare = try XCTUnwrap(library.makeFunction(name: "tans_prepare_zero_runs"))
          encoder.setComputePipelineState(try device.makeComputePipelineState(function: prepare))
          encoder.setBuffer(decoding, offset: 0, index: 0)
          encoder.setBuffer(folded, offset: 0, index: 1)
          var entries = UInt32(decoding.length / 4)
          encoder.setBytes(&entries, length: 4, index: 2)
          encoder.dispatchThreads(
            MTLSize(width: Int(entries) + 32, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          encoder.memoryBarrier(scope: .buffers)
          tableForDecoding = folded
        }
        encoder.setComputePipelineState(pipeline)
        encoder.useResources([encoded, seeks, empty], usage: .read)
        encoder.useResource(output, usage: [.read, .write])
        for (index, resource) in [table, tableForDecoding, models, map, ranks, selected, signs]
          .enumerated()
        {
          encoder.setBuffer(resource, offset: 0, index: index)
        }
        encoder.setBytes(&query, length: 20, index: 7)
        encoder.setBuffer(modelOffsets, offset: 0, index: 8)
        if let pairLookup { encoder.setBuffer(pairLookup, offset: 0, index: 14) }
        if mode >= 32 {
          for (index, resource) in [
            mixedTopology ? mixedDescriptors : descriptors, selected, signs, modelOffsets, counts,
          ].enumerated() {
            encoder.setBuffer(resource, offset: 0, index: index + 9)
          }
        }
        // Two independent disjoint column contributions target the same atomic
        // output. The independent scalar expected array is unchanged.
        let splitSigns = try (0..<(concurrent ? 2 : 1)).map { part in
          try buffer(
            (0..<32).map { col in
              concurrent && col / 16 != part ? Int32(0) : coefficient(col)
            })
        }
        for contribution in splitSigns {
          encoder.setBuffer(contribution, offset: 0, index: 6)
          if mode >= 32 { encoder.setBuffer(contribution, offset: 0, index: 11) }
          encoder.dispatchThreadgroups(
            MTLSize(
              width: packetMajor ? 8 : 1,
              height: packetMajor
                ? 1 : (mode >= 32 ? ([67, 70].contains(mode) ? 2 : 8 / packets) : 32),
              depth: 1),
            threadsPerThreadgroup: MTLSize(
              width: mode >= 32 ? ([67, 70].contains(mode) ? 512 : 128) : 32, height: 1, depth: 1))
        }
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(mode): \(String(describing: command.error))")
        if preparedZero {
          let original = decoding.contents().assumingMemoryBound(to: UInt32.self)
          let folded = tableForDecoding.contents().assumingMemoryBound(to: UInt32.self)
          for index in 0..<(decoding.length / 4) {
            var expected = original[index]
            if expected & 65535 == 0 {
              var state = index % 1024
              var count = UInt32(0)
              while count < 63 {
                let transition = original[(index / 1024) * 1024 + state]
                if transition & 65535 != 0 { break }
                state = Int(transition >> 16)
                count += 1
              }
              expected = UInt32(state) << 16 | count << 26
            }
            XCTAssertEqual(
              folded[index], expected, "Prepared exact transition mode\(mode) state\(index)")
          }
          for index in (decoding.length / 4)..<(decoding.length / 4 + 16) {
            XCTAssertEqual(folded[index], 0xDEAD_BEEF, "Preparation write canary")
          }
        }
        let actual = Array(
          UnsafeBufferPointer(
            start: output.contents().assumingMemoryBound(to: UInt32.self),
            count: scans))
        XCTAssertEqual(actual, expected, "mode \(mode), signed scenario \(scenario)")
        if literal && scenario <= 1 { XCTAssertGreaterThan(actual.max()!, 65535) }
      }
    }
  }
}
