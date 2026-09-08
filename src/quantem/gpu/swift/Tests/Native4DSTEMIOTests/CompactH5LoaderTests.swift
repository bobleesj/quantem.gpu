import CryptoKit
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

final class CompactH5LoaderTests: XCTestCase {
  func testPackedEmptyForcedRebaseClearsPreviousDetectorImage() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    for direct in [false, true] {
      let fixture = try direct ? makeDirectCompactFixture() : makeCompactFixture(portable: true)
      defer { try? FileManager.default.removeItem(at: fixture.url) }
      let source = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
      defer { source.releaseResidentStorage() }
      let full = [UInt8](repeating: 1, count: 6)
      let empty = [UInt8](repeating: 0, count: 6)
      let excludedOnly: [UInt8] = [0, 0, 0, 1, 0, 0]
      for mask in [empty, excludedOnly] {
        try source.updateVirtualDetector(mask: full, forceRebase: true)
        XCTAssertTrue(try source.virtualDetectorValues().contains { $0 > 0 })
        let result = try source.updateVirtualDetector(mask: mask, forceRebase: true)
        XCTAssertEqual(result.mode, "rebase")
        XCTAssertEqual(result.fftDispatchCount, 0)
        XCTAssertEqual(try source.virtualDetectorValues(), [UInt32](repeating: 0, count: 128))
        // An unchanged empty request must retain the newly published zero image.
        try source.updateVirtualDetector(mask: mask)
        XCTAssertEqual(try source.virtualDetectorValues(), [UInt32](repeating: 0, count: 128))
      }
    }
  }

  func testPackedDPOrderDuplicatesAndMaskRecoveryPreserveExactCounts() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    for direct in [false, true] {
      let fixture =
        try direct
        ? makeDirectCompactFixture() : makeCompactFixture(portable: true, fullRange: true)
      defer { try? FileManager.default.removeItem(at: fixture.url) }
      let source = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
      defer { source.releaseResidentStorage() }
      if !direct { XCTAssertEqual(fixture.values[4].max(), UInt32(UInt16.max)) }
      // Explicit row/column requests cross row boundaries and revisit the same DP.
      // This verifies repeated single-DP access, not a batched selection API.
      let positions = [(7, 15), (0, 0), (4, 13), (0, 15), (1, 0), (4, 13)]
      let masks: [[UInt8]] = [
        [0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0],
      ]
      for mask in masks {
        try source.updateVirtualDetector(mask: mask)
        let expected = (0..<128).map { scan in
          mask.indices.reduce(UInt32(0)) { sum, pixel in
            sum + (mask[pixel] == 0 ? 0 : fixture.values[pixel][scan])
          }
        }
        XCTAssertEqual(try source.virtualDetectorValues(), expected)
        for (row, column) in positions {
          var result = try source.extractDiffraction(scanRow: row, scanColumn: column)
          XCTAssertEqual(result, fixture.values.map { $0[row * 16 + column] })
          result[0] = .max
          XCTAssertEqual(
            try source.extractDiffraction(scanRow: row, scanColumn: column),
            fixture.values.map { $0[row * 16 + column] })
        }
        for invalid in [[UInt8](repeating: 1, count: 5), [2, 0, 0, 0, 0, 0]] {
          XCTAssertThrowsError(try source.updateVirtualDetector(mask: invalid))
          XCTAssertEqual(try source.virtualDetectorValues(), expected)
        }
        for (row, column) in [(-1, 0), (8, 0), (0, -1), (0, 16)] {
          XCTAssertThrowsError(try source.extractDiffraction(scanRow: row, scanColumn: column))
          XCTAssertEqual(try source.virtualDetectorValues(), expected)
        }
      }
      let mean = try source.meanDiffractionPattern()
      let sums = fixture.values.map { $0.reduce(UInt64(0)) { $0 + UInt64($1) } }
      XCTAssertEqual(mean.detectorSum, sums)
      XCTAssertEqual(mean.mean, sums.map { Float($0) / 128 })
      source.releaseResidentStorage()
      XCTAssertThrowsError(try source.extractDiffraction(scanRow: 0, scanColumn: 0))
      XCTAssertThrowsError(try source.updateVirtualDetector(mask: masks[0]))
      XCTAssertThrowsError(try source.meanDiffractionPattern())
    }
  }

  func testInspectMatchesLoadMetadataWithoutMetalAllocation() throws {
    let fixture = try makeCompactFixture(portable: true, preparedDPC: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let allocated = device.currentAllocatedSize
    let metadata = try MetalCompactH5Loader.inspect(sourceURL: fixture.url)
    XCTAssertEqual(device.currentAllocatedSize, allocated)
    let source = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    defer { source.releaseResidentStorage() }
    XCTAssertEqual(metadata, source.metadata)
  }

  func testInspectIsNotPayloadAuthentication() throws {
    let fixture = try makeCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    var bytes = try Data(contentsOf: fixture.url)
    bytes[fixture.headerOffset] = 255
    try bytes.write(to: fixture.url)
    XCTAssertNoThrow(try MetalCompactH5Loader.inspect(sourceURL: fixture.url))
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    XCTAssertThrowsError(try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device))
    bytes[0] = 0
    try bytes.write(to: fixture.url)
    XCTAssertThrowsError(try MetalCompactH5Loader.inspect(sourceURL: fixture.url))
  }

  func testTrustedDirectPreparedProductsMatchVerifiedProducts() throws {
    let fixture = try makeDirectCompactFixture(preparedDPC: true, preparedDetectorProducts: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let verified = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    var expected: [[UInt32]] = []
    let names: [MetalCompactH5PreparedDetectorProductName] = [.bf, .abf, .adf]
    for name in names {
      try verified.activatePreparedDetectorProduct(name)
      expected.append(try verified.virtualDetectorValues())
    }
    verified.releaseResidentStorage()
    let trusted = try MetalCompactH5Loader.load(
      sourceURL: fixture.url, device: device, authenticationPolicy: .parallelMapped,
      verifyChecksums: false
    )
    XCTAssertFalse(trusted.loadMetrics.checksumsVerified)
    XCTAssertEqual(trusted.loadMetrics.decodedIntegrityMilliseconds, 0)
    XCTAssertEqual(trusted.loadMetrics.preparedDetectorProductAuthenticationMilliseconds, 0)
    for (offset, name) in names.enumerated() {
      try trusted.activatePreparedDetectorProduct(name)
      XCTAssertEqual(try trusted.virtualDetectorValues(), expected[offset])
    }
    for scan in 0..<128 {
      XCTAssertEqual(
        try trusted.extractDiffraction(scanRow: scan / 16, scanColumn: scan % 16),
        fixture.values.map { $0[scan] }
      )
    }
    trusted.releaseResidentStorage()
  }

  func testTrustedLoadPreservesEveryScanAndReportsSkippedChecks() throws {
    let fixture = try makeMultishardCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    for policy in [MetalCompactH5AuthenticationPolicy.boundedSequential, .boundedConcurrent] {
      let source = try MetalCompactH5Loader.load(
        sourceURL: fixture.url, device: device, authenticationPolicy: policy,
        verifyChecksums: false
      )
      XCTAssertFalse(source.loadMetrics.checksumsVerified)
      XCTAssertEqual(source.loadMetrics.decodedShardSHA256Checks, 0)
      XCTAssertEqual(source.loadMetrics.decodedIntegrityMilliseconds, 0)
      XCTAssertGreaterThan(source.loadMetrics.gpuPreparationMilliseconds, 0)
      XCTAssertEqual(source.loadMetrics.decodedPayloadCopyBytes, 0)
      for scan in 0..<640 {
        XCTAssertEqual(
          try source.extractDiffraction(scanRow: scan / 16, scanColumn: scan % 16),
          fixture.values.map { $0[scan] }
        )
      }
      source.releaseResidentStorage()
    }
  }

  func testTrustedLoadSkipsDigestButStillRejectsInvalidWidths() throws {
    let fixture = try makeCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    var bytes = try Data(contentsOf: fixture.url)
    // Change only the expected decoded digest, not the scientific payload.
    bytes[4096 + 76 + 64] ^= 1
    try bytes.write(to: fixture.url)
    XCTAssertThrowsError(try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device))
    let trusted = try MetalCompactH5Loader.load(
      sourceURL: fixture.url, device: device, verifyChecksums: false
    )
    XCTAssertEqual(
      try trusted.extractDiffraction(scanRow: 3, scanColumn: 7),
      fixture.values.map { $0[55] }
    )
    trusted.releaseResidentStorage()
    bytes[fixture.headerOffset] = 255
    try bytes.write(to: fixture.url)
    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device, verifyChecksums: false)
    )
  }

  func testTrustedPreparedAndNativeCacheParity() throws {
    let fixture = try makeCompactFixture(portable: true, preparedDPC: true)
    let cache = fixture.url.appendingPathExtension("qgmc")
    defer {
      try? FileManager.default.removeItem(at: fixture.url)
      try? FileManager.default.removeItem(at: cache)
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let verified = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    XCTAssertTrue(verified.loadMetrics.checksumsVerified)
    let expected = try verified.preparedDPCMomentValues()
    try verified.saveNativeCache(to: cache)
    verified.releaseResidentStorage()
    for cacheURL in [nil, cache] as [URL?] {
      let source = try MetalCompactH5Loader.load(
        sourceURL: fixture.url, device: device, authenticationPolicy: .parallelMapped,
        nativeCacheURL: cacheURL, verifyChecksums: false
      )
      XCTAssertEqual(try source.preparedDPCMomentValues(), expected)
      XCTAssertEqual(source.loadMetrics.preparedDPCAuthenticationMilliseconds, 0)
      XCTAssertEqual(source.loadMetrics.nativeCacheDescriptorSHA256Checks, 0)
      XCTAssertEqual(source.loadMetrics.mappedAuthenticationBytes, 0)
      XCTAssertEqual(source.loadMetrics.decodedShardSHA256Checks, 0)
      source.releaseResidentStorage()
    }
  }

  func testBoundedConcurrentPreservesEveryScanAcrossPartialWindow() throws {
    let fixture = try makeMultishardCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    let original = try Data(contentsOf: fixture.url)
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let sequential = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    let sequentialBudget = sequential.loadMetrics.plannedAdditionalBytes
    XCTAssertGreaterThan(sequential.loadMetrics.decodedPayloadCopyBytes, 0)
    let sequentialStaging = sequential.loadMetrics.maximumTransientBytes
    try sequential.updateVirtualDetector(mask: [1, 0, 1, 1, 0, 1])
    let expectedMask = try sequential.virtualDetectorValues()
    sequential.releaseResidentStorage()
    let parallel = try MetalCompactH5Loader.load(
      sourceURL: fixture.url, device: device, authenticationPolicy: .boundedConcurrent
    )
    XCTAssertEqual(parallel.metadata.shardCount, 5)
    XCTAssertEqual(parallel.loadMetrics.maximumInFlightShards, 3)
    XCTAssertEqual(parallel.loadMetrics.nativeCacheStatus, "notRequested")
    XCTAssertEqual(parallel.loadMetrics.mappedAuthenticationBytes, 0)
    XCTAssertEqual(parallel.loadMetrics.decodedShardSHA256Checks, 5)
    XCTAssertEqual(parallel.loadMetrics.sourceReadPolicy, "systemDefault")
    XCTAssertEqual(
      parallel.loadMetrics.plannedAdditionalBytes, sequentialBudget + 2 * sequentialStaging)
    XCTAssertEqual(parallel.loadMetrics.maximumTransientBytes, 3 * sequentialStaging)
    for scan in 0..<640 {
      XCTAssertEqual(
        try parallel.extractDiffraction(scanRow: scan / 16, scanColumn: scan % 16),
        fixture.values.map { $0[scan] }
      )
    }
    try parallel.updateVirtualDetector(mask: [1, 0, 1, 1, 0, 1])
    XCTAssertEqual(try parallel.virtualDetectorValues(), expectedMask)
    let required = parallel.loadMetrics.plannedAdditionalBytes
    parallel.releaseResidentStorage()
    let allocated = device.currentAllocatedSize
    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(
        sourceURL: fixture.url, device: device, authenticationPolicy: .boundedConcurrent,
        maximumAdditionalBytes: required - 1
      ))
    XCTAssertEqual(device.currentAllocatedSize, allocated)
    let admitted = try MetalCompactH5Loader.load(
      sourceURL: fixture.url, device: device, authenticationPolicy: .boundedConcurrent,
      maximumAdditionalBytes: required
    )
    admitted.releaseResidentStorage()
    XCTAssertEqual(try Data(contentsOf: fixture.url), original)
  }

  func testConcurrentCorruptionAndCancellationNeverPublish() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    try autoreleasepool {
      let warmup = try makeMultishardCompactFixture()
      defer { try? FileManager.default.removeItem(at: warmup.url) }
      let source = try MetalCompactH5Loader.load(
        sourceURL: warmup.url, device: device, authenticationPolicy: .boundedConcurrent
      )
      source.releaseResidentStorage()
    }
    let baselineAllocation = device.currentAllocatedSize
    print("COMPACT_CANCELLATION warmed_allocation=\(baselineAllocation)")
    for corruptWidth in [false, true] {
      let fixture = try makeMultishardCompactFixture()
      defer { try? FileManager.default.removeItem(at: fixture.url) }
      var data = try Data(contentsOf: fixture.url)
      // Corrupt the last shard, after a complete earlier window succeeded.
      let field = 4096 + 76 + 4 * 96 + (corruptWidth ? 32 : 0)
      let offset = data.withUnsafeBytes {
        Int($0.loadUnaligned(fromByteOffset: field, as: UInt64.self).littleEndian)
      }
      data[offset + (corruptWidth ? 0 : 2)] ^= corruptWidth ? 0xff : 1
      try data.write(to: fixture.url)
      try autoreleasepool {
        XCTAssertThrowsError(
          try MetalCompactH5Loader.load(
            sourceURL: fixture.url, device: device, authenticationPolicy: .boundedConcurrent
          )
        ) { error in
          XCTAssertTrue(error.localizedDescription.contains(corruptWidth ? "bits" : "SHA-256"))
        }
      }
      // Compare every failure with the same successful-load baseline. Driver
      // allocation can shrink between calls; a temporary low-water mark is
      // not a new memory budget for a later, different cancellation phase.
      assertAllocationRetires(device, to: baselineAllocation)
      print(
        "COMPACT_CANCELLATION corrupt_width=\(corruptWidth) allocated=\(device.currentAllocatedSize)"
      )
    }
    let fixture = try makeMultishardCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    let callingThread = Thread.current
    for cancelAt in Array(repeating: [1, 4, 5], count: 10).flatMap({ $0 }) {
      var calls = 0
      try autoreleasepool {
        XCTAssertThrowsError(
          try MetalCompactH5Loader.load(
            sourceURL: fixture.url, device: device, authenticationPolicy: .boundedConcurrent,
            shouldCancel: {
              XCTAssertEqual(Thread.current, callingThread)
              calls += 1
              return calls == cancelAt
            }
          )
        ) { error in
          guard case Metal4DSTEMStreamingIOError.cancelled = error else {
            return XCTFail("Expected cancellation, got \(error)")
          }
        }
      }
      assertAllocationRetires(device, to: baselineAllocation)
      print("COMPACT_CANCELLATION cancel_at=\(cancelAt) allocated=\(device.currentAllocatedSize)")
    }
  }

  private func assertAllocationRetires(
    _ device: MTLDevice, to baseline: Int, file: StaticString = #filePath, line: UInt = #line
  ) {
    // Completed commands can still be retiring driver-owned storage. Keep the
    // exact no-growth bound, but do not confuse deferred retirement with a leak.
    // This is a teardown assertion, not a load-latency measurement.
    let deadline = ContinuousClock.now.advanced(by: .seconds(2))
    while device.currentAllocatedSize > baseline && ContinuousClock.now < deadline {
      Thread.sleep(forTimeInterval: 0.01)
    }
    XCTAssertLessThanOrEqual(device.currentAllocatedSize, baseline, file: file, line: line)
  }

  func testSourceAvoidCachingPreservesParityAndRejectsIncompatibleModes() throws {
    let fixture = try makeCompactFixture(portable: true, preparedDPC: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    let original = try Data(contentsOf: fixture.url)
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let resident = try MetalCompactH5Loader.load(
      sourceURL: fixture.url, device: device, authenticationPolicy: .boundedConcurrent,
      sourceReadPolicy: .avoidCaching
    )
    XCTAssertEqual(resident.loadMetrics.sourceReadPolicy, "avoidCaching")
    XCTAssertEqual(resident.loadMetrics.maximumInFlightShards, 1)
    XCTAssertNotNil(try resident.preparedDPCMomentValues())
    XCTAssertEqual(
      try resident.extractDiffraction(scanRow: 3, scanColumn: 7),
      fixture.values.map { $0[3 * 16 + 7] }
    )
    resident.releaseResidentStorage()
    let allocated = device.currentAllocatedSize
    for policy in [MetalCompactH5AuthenticationPolicy.boundedConcurrent, .parallelMapped] {
      XCTAssertThrowsError(
        try MetalCompactH5Loader.load(
          sourceURL: fixture.url, device: device, authenticationPolicy: policy,
          nativeCacheURL: policy == .boundedConcurrent ? fixture.url : nil,
          sourceReadPolicy: .avoidCaching
        ))
    }
    XCTAssertEqual(device.currentAllocatedSize, allocated)
    XCTAssertEqual(try Data(contentsOf: fixture.url), original)
  }

  func testLoadBudgetRejectsBeforeAnyMetalAllocation() throws {
    let fixture = try makeCompactFixture(portable: true, preparedDPC: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let allocated = device.currentAllocatedSize
    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(
        sourceURL: fixture.url, device: device, maximumAdditionalBytes: 0
      )
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("No Metal storage was allocated"))
    }
    XCTAssertEqual(device.currentAllocatedSize, allocated)
  }

  func testNativeCacheRestoresEveryUInt16ValueAndMoments() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture(portable: true, preparedDPC: true)
    let cache = fixture.url.appendingPathExtension("qgmc")
    defer {
      try? FileManager.default.removeItem(at: fixture.url)
      try? FileManager.default.removeItem(at: cache)
    }
    let originalBytes = try Data(contentsOf: fixture.url)
    let source = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    let originalMoments = try XCTUnwrap(source.preparedDPCMomentValues())
    try source.saveNativeCache(to: cache)
    source.releaseResidentStorage()
    var boundedPlan: UInt64?
    for policy in [MetalCompactH5AuthenticationPolicy.boundedSequential, .parallelMapped] {
      let restored = try MetalCompactH5Loader.load(
        sourceURL: fixture.url, device: device, authenticationPolicy: policy,
        nativeCacheURL: cache
      )
      XCTAssertEqual(restored.metadata, source.metadata)
      XCTAssertEqual(restored.loadMetrics.nativeCacheStatus, "hit")
      XCTAssertEqual(restored.loadMetrics.nativeCacheDescriptorSHA256Checks, 1)
      XCTAssertEqual(restored.loadMetrics.decodedShardSHA256Checks, 1)
      XCTAssertEqual(restored.loadMetrics.gpuDecodeMilliseconds, 0)
      XCTAssertGreaterThan(restored.loadMetrics.nativeCacheBytes, 65_536)
      XCTAssertGreaterThan(
        restored.loadMetrics.plannedAdditionalBytes,
        restored.loadMetrics.totalResidentBytes + restored.loadMetrics.mappedAuthenticationBytes
      )
      let required = restored.loadMetrics.plannedAdditionalBytes
      if policy == .boundedSequential {
        boundedPlan = required
      } else {
        XCTAssertEqual(required, try XCTUnwrap(boundedPlan) + restored.loadMetrics.nativeCacheBytes)
      }
      for scan in 0..<128 {
        XCTAssertEqual(
          try restored.extractDiffraction(scanRow: scan / 16, scanColumn: scan % 16),
          fixture.values.map { $0[scan] }
        )
      }
      XCTAssertEqual(try restored.preparedDPCMomentValues(), originalMoments)
      let mask: [UInt8] = [1, 0, 1, 0, 1, 0]
      _ = try restored.updateVirtualDetector(mask: mask)
      XCTAssertEqual(
        try restored.virtualDetectorValues(),
        (0..<128).map {
          fixture.values[0][$0] + fixture.values[2][$0] + fixture.values[4][$0]
        })
      XCTAssertNoThrow(
        try Metal4DSTEMResidentCapabilities.compact(restored).residentReceipt.validate())
      restored.releaseResidentStorage()
      let allocationsBeforeRejection = device.currentAllocatedSize
      XCTAssertThrowsError(
        try MetalCompactH5Loader.load(
          sourceURL: fixture.url, device: device, authenticationPolicy: policy,
          nativeCacheURL: cache, maximumAdditionalBytes: required - 1
        )
      ) { error in
        XCTAssertTrue(error.localizedDescription.contains("No Metal storage was allocated"))
      }
      XCTAssertEqual(device.currentAllocatedSize, allocationsBeforeRejection)
      let admitted = try MetalCompactH5Loader.load(
        sourceURL: fixture.url, device: device, authenticationPolicy: policy,
        nativeCacheURL: cache, maximumAdditionalBytes: required
      )
      XCTAssertEqual(admitted.loadMetrics.plannedAdditionalBytes, required)
      admitted.releaseResidentStorage()
    }
    XCTAssertEqual(try Data(contentsOf: fixture.url), originalBytes)
  }

  func testNativeCacheMissingAndStaleFallBackToExactSource() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture(portable: true)
    let cache = fixture.url.appendingPathExtension("qgmc")
    defer {
      try? FileManager.default.removeItem(at: fixture.url)
      try? FileManager.default.removeItem(at: cache)
    }
    let source = try MetalCompactH5Loader.load(
      sourceURL: fixture.url, device: device, nativeCacheURL: cache
    )
    XCTAssertEqual(source.loadMetrics.nativeCacheStatus, "miss")
    try source.saveNativeCache(to: cache)
    source.releaseResidentStorage()
    try FileManager.default.setAttributes(
      [.modificationDate: Date(timeIntervalSince1970: 1)], ofItemAtPath: fixture.url.path
    )
    let restored = try MetalCompactH5Loader.load(
      sourceURL: fixture.url, device: device, nativeCacheURL: cache
    )
    XCTAssertEqual(restored.loadMetrics.nativeCacheStatus, "stale")
    XCTAssertEqual(restored.loadMetrics.nativeCacheDescriptorSHA256Checks, 0)
    XCTAssertEqual(
      try restored.extractDiffraction(scanRow: 4, scanColumn: 13), fixture.values.map { $0[77] })
    restored.releaseResidentStorage()
  }

  func testNativeCachePayloadAndLookupCorruptionFailClosed() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture(portable: true)
    let cacheURL = fixture.url.appendingPathExtension("qgmc")
    defer {
      try? FileManager.default.removeItem(at: fixture.url)
      try? FileManager.default.removeItem(at: cacheURL)
    }
    let source = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    try source.saveNativeCache(to: cacheURL)
    source.releaseResidentStorage()
    let file = try FileHandle(forReadingFrom: cacheURL)
    let cache = try CompactNativeCache.read(from: file)
    try file.close()
    let original = try Data(contentsOf: cacheURL)
    for offset in [cache.shards[0].payloadOffset, cache.shards[0].descriptorsOffset] {
      var changed = original
      changed[Int(offset)] ^= 1
      try changed.write(to: cacheURL)
      for policy in [MetalCompactH5AuthenticationPolicy.boundedSequential, .parallelMapped] {
        XCTAssertThrowsError(
          try MetalCompactH5Loader.load(
            sourceURL: fixture.url, device: device, authenticationPolicy: policy,
            nativeCacheURL: cacheURL
          )
        ) { error in
          XCTAssertTrue(
            error.localizedDescription.contains("SHA-256")
              || error.localizedDescription.contains("parallel authentication"))
        }
      }
    }
  }

  func testNativeCachePublicationNeverOverwritesAndCancellationRemovesPartial() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture(portable: true)
    let directory = fixture.url.deletingLastPathComponent()
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    let cache = directory.appendingPathComponent("source.qgmc")
    defer {
      try? FileManager.default.removeItem(at: fixture.url)
      try? FileManager.default.removeItem(at: directory)
    }
    let source = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    XCTAssertThrowsError(try source.saveNativeCache(to: cache, shouldCancel: { true }))
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [])
    try source.saveNativeCache(to: cache)
    let bytes = try Data(contentsOf: cache)
    XCTAssertThrowsError(try source.saveNativeCache(to: cache))
    XCTAssertEqual(try Data(contentsOf: cache), bytes)
    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(
        sourceURL: fixture.url, device: device, nativeCacheURL: cache,
        shouldCancel: { true }
      ))
    source.releaseResidentStorage()
    XCTAssertThrowsError(
      try source.saveNativeCache(to: directory.appendingPathComponent("released.qgmc")))
  }

  func testSyntheticCompactSourceLoadsAndInteractsExactly() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    let source = try MetalCompactH5Loader.load(
      sourceURL: fixture.url,
      device: device
    )
    XCTAssertEqual(source.metadata.scanRows, 8)
    XCTAssertEqual(source.metadata.scanColumns, 16)
    XCTAssertEqual(source.metadata.detectorRows, 2)
    XCTAssertEqual(source.metadata.detectorColumns, 3)
    XCTAssertEqual(source.metadata.excludedDetectorPixels, [3])
    XCTAssertEqual(source.metadata.rawAccessMode, "mask_applied_only_legacy")
    XCTAssertNil(source.metadata.maskedDetectorPixelsSHA256)
    XCTAssertNil(source.metadata.maskedDetectorRawValues)
    XCTAssertEqual(source.metadata.detectorCalibration?.detectorCenterRow, 0.75)
    XCTAssertEqual(source.metadata.detectorCalibration?.detectorCenterColumn, 1.25)
    XCTAssertEqual(source.metadata.detectorCalibration?.brightFieldRadius, 1.5)
    XCTAssertEqual(source.metadata.detectorCalibration?.dpcRotationDegrees, 176.25)
    XCTAssertEqual(source.metadata.detectorCalibration?.dpcComponentOrderExchanged, false)
    XCTAssertEqual(source.loadMetrics.decodedShardSHA256Checks, 1)
    XCTAssertGreaterThan(fixture.values[4].max()!, 255)
    XCTAssertThrowsError(try Metal4DSTEMResidentCapabilities.compact(source))

    let selectedScan = 77
    XCTAssertEqual(
      try source.extractDiffraction(scanRow: 4, scanColumn: 13),
      fixture.values.map { $0[selectedScan] }
    )

    let mask: [UInt8] = [1, 0, 1, 0, 1, 0]
    let rebase = try source.updateVirtualDetector(mask: mask)
    XCTAssertEqual(rebase.mode, "rebase")
    XCTAssertEqual(rebase.fftDispatchCount, 0)
    let expected = (0..<128).map { scan in
      fixture.values[0][scan] + fixture.values[2][scan] + fixture.values[4][scan]
    }
    XCTAssertEqual(try source.virtualDetectorValues(), expected)

    let translatedMask: [UInt8] = [0, 1, 1, 0, 1, 0]
    let delta = try source.updateVirtualDetector(mask: translatedMask)
    XCTAssertEqual(delta.mode, "delta")
    XCTAssertEqual(delta.changedDetectorPixels, 2)
    let translatedExpected = (0..<128).map { scan in
      fixture.values[1][scan] + fixture.values[2][scan] + fixture.values[4][scan]
    }
    XCTAssertEqual(try source.virtualDetectorValues(), translatedExpected)
    let fresh = try source.updateVirtualDetector(
      mask: translatedMask,
      forceRebase: true
    )
    XCTAssertEqual(fresh.mode, "rebase")
    XCTAssertEqual(try source.virtualDetectorValues(), translatedExpected)
  }

  func testResidentTiltSeriesUpdatesEveryExactImageInOneSubmission() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture(portable: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    let sources = try (0..<2).map { _ in
      try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    }
    defer {
      for source in sources { source.releaseResidentStorage() }
    }

    let brightFieldMask: [UInt8] = [1, 0, 1, 0, 1, 0]
    let brightField = try MetalCompactH5ResidentSource.updateVirtualDetectors(
      sources, mask: brightFieldMask
    )
    let expectedBrightField = (0..<128).map { scan in
      fixture.values[0][scan] + fixture.values[2][scan] + fixture.values[4][scan]
    }
    XCTAssertEqual(brightField.submissionCount, 1)
    XCTAssertEqual(brightField.sources.count, 2)
    XCTAssertTrue(brightField.sources.allSatisfy { $0.mode == "rebase" })
    for source in sources {
      XCTAssertEqual(try source.virtualDetectorValues(), expectedBrightField)
    }

    let translatedMask: [UInt8] = [0, 1, 1, 0, 1, 0]
    let translated = try MetalCompactH5ResidentSource.updateVirtualDetectors(
      sources, mask: translatedMask
    )
    let expectedTranslated = (0..<128).map { scan in
      fixture.values[1][scan] + fixture.values[2][scan] + fixture.values[4][scan]
    }
    XCTAssertEqual(translated.submissionCount, 1)
    XCTAssertTrue(
      translated.sources.allSatisfy {
        $0.mode == "delta" && $0.changedDetectorPixels == 2
      }
    )
    for source in sources {
      XCTAssertEqual(try source.virtualDetectorValues(), expectedTranslated)
    }

    let cleared = try MetalCompactH5ResidentSource.updateVirtualDetectors(
      sources, mask: Array(repeating: 0, count: 6), forceRebase: true
    )
    XCTAssertEqual(cleared.submissionCount, 1)
    for source in sources {
      XCTAssertEqual(try source.virtualDetectorValues(), Array(repeating: 0, count: 128))
    }
  }

  func testResidentTiltSeriesRejectsDuplicateOwnership() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture(portable: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    let source = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    defer { source.releaseResidentStorage() }

    XCTAssertThrowsError(
      try MetalCompactH5ResidentSource.updateVirtualDetectors(
        [source, source], mask: [1, 0, 1, 0, 1, 0]
      )
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("same source twice"))
    }
  }

  func testPortableV1SourcePublishesExactUInt16Receipt() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture(portable: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    let source = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    let capabilities = try Metal4DSTEMResidentCapabilities.compact(source)

    XCTAssertEqual(capabilities.representation, .packed)
    XCTAssertEqual(capabilities.residentReceipt.sourceShape, [8, 16, 2, 3])
    XCTAssertEqual(capabilities.residentReceipt.workingShape, [8, 16, 2, 3])
    XCTAssertEqual(capabilities.residentReceipt.sourceDtype, "uint16")
    XCTAssertEqual(capabilities.residentReceipt.workingDtype, "uint16")
    XCTAssertEqual(capabilities.residentReceipt.sourceLogicalTensorBytes, 1_536)
    XCTAssertEqual(capabilities.residentReceipt.workingLogicalTensorBytes, 1_536)
    XCTAssertNoThrow(try capabilities.residentReceipt.validate())
  }

  func testPreparedV1MomentsAndMaskedOriginalsLoadExactly() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture(portable: true, preparedDPC: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    let source = try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    let contract = try XCTUnwrap(source.metadata.preparedDPCMoments)
    XCTAssertEqual(contract.workingDtype, "uint16")
    XCTAssertEqual(contract.workingLogicalSHA256, source.metadata.workingLogicalSHA256)
    XCTAssertEqual(source.metadata.maskedDetectorRawValues, [0])
    XCTAssertNotNil(source.metadata.maskedDetectorPixelsSHA256)

    let selectedPixels = [0, 1, 2, 4, 5]
    let expectedTotal = (0..<128).map { scan in
      selectedPixels.reduce(UInt64(0)) { $0 + UInt64(fixture.values[$1][scan]) }
    }
    let expectedRow = (0..<128).map { scan in
      selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(fixture.values[$1][scan]) * UInt64($1 / 3)
      }
    }
    let expectedColumn = (0..<128).map { scan in
      selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(fixture.values[$1][scan]) * UInt64($1 % 3)
      }
    }
    let moments = try XCTUnwrap(source.preparedDPCMomentValues())
    XCTAssertEqual(moments.total, expectedTotal)
    XCTAssertEqual(moments.detectorRowMoment, expectedRow)
    XCTAssertEqual(moments.detectorColumnMoment, expectedColumn)
    XCTAssertNotNil(try source.preparedDPCValues())
  }

  func testPreparedV1MomentIdentityMismatchFailsClosed() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture(
      portable: true,
      preparedDPC: true,
      preparedDPCOverrides: ["working_logical_sha256": String(repeating: "0", count: 64)]
    )
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("working_logical_sha256"))
    }
  }

  func testChangedDecodedPayloadFailsItsAuthenticatedHash() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    var changed = try Data(contentsOf: fixture.url)
    changed[8_204] ^= 1
    try changed.write(to: fixture.url, options: .atomic)

    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("decoded SHA-256"))
    }
  }

  func testSyntheticDirectV3SourceLoadsAndInteractsExactly() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    let source = try MetalCompactH5Loader.load(
      sourceURL: fixture.url,
      device: device
    )
    XCTAssertEqual(source.metadata.schema, "quantem.gpu.packed-detector-h5/v3")
    XCTAssertEqual(source.metadata.payloadCodec, "direct-bitpacked-u32")
    XCTAssertEqual(source.metadata.rawAccessMode, "mask_applied_only_legacy")
    XCTAssertNil(source.metadata.maskedDetectorPixelsSHA256)
    XCTAssertNil(source.metadata.maskedDetectorRawValues)
    XCTAssertEqual(source.metadata.scanTile, 32)
    XCTAssertEqual(source.metadata.payloadChunkBytes, 0)
    XCTAssertEqual(source.metadata.excludedDetectorPixels, [3])
    XCTAssertNil(source.metadata.detectorCalibration)
    let logicalHash = try source.hashLogicalWorkingU8()
    XCTAssertEqual(logicalHash.sha256, source.metadata.workingLogicalSHA256)
    XCTAssertEqual(logicalHash.logicalBytes, 128 * 6)
    XCTAssertEqual(logicalHash.stagingBytes, 128 * 6)
    XCTAssertGreaterThan(
      source.loadMetrics.totalResidentBytes,
      source.loadMetrics.residentBytes
    )

    let expectedDetectorSum = fixture.values.map {
      $0.reduce(UInt64(0)) { $0 + UInt64($1) }
    }
    let mean = try source.meanDiffractionPattern()
    XCTAssertEqual(mean.detectorSum, expectedDetectorSum)
    XCTAssertEqual(
      mean.mean,
      expectedDetectorSum.map { Float($0) / 128 }
    )
    XCTAssertEqual(mean.dispatchCount, 1)
    XCTAssertEqual(mean.readbackBytes, 6 * 8)
    let cachedMean = try source.meanDiffractionPattern()
    XCTAssertEqual(cachedMean.detectorSum, expectedDetectorSum)
    XCTAssertEqual(cachedMean.dispatchCount, 0)
    XCTAssertEqual(cachedMean.wallMilliseconds, 0)
    XCTAssertEqual(cachedMean.gpuMilliseconds, 0)
    XCTAssertThrowsError(try Metal4DSTEMResidentCapabilities.compact(source))

    let selectedScan = 77
    XCTAssertEqual(
      try source.extractDiffraction(scanRow: 4, scanColumn: 13),
      fixture.values.map { $0[selectedScan] }
    )

    let mask: [UInt8] = [1, 0, 1, 1, 1, 0]
    let rebase = try source.updateVirtualDetector(mask: mask)
    XCTAssertEqual(rebase.mode, "rebase")
    let expected = (0..<128).map { scan in
      fixture.values[0][scan] + fixture.values[2][scan] + fixture.values[4][scan]
    }
    XCTAssertEqual(try source.virtualDetectorValues(), expected)

    let translatedMask: [UInt8] = [0, 1, 1, 0, 1, 0]
    let delta = try source.updateVirtualDetector(mask: translatedMask)
    XCTAssertEqual(delta.mode, "delta")
    XCTAssertEqual(delta.changedDetectorPixels, 2)
    let translatedExpected = (0..<128).map { scan in
      fixture.values[1][scan] + fixture.values[2][scan] + fixture.values[4][scan]
    }
    XCTAssertEqual(try source.virtualDetectorValues(), translatedExpected)
  }

  func testChangedDirectV3HeaderFailsCanonicalCoverage() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    var changed = try Data(contentsOf: fixture.url)
    changed[fixture.headerOffset + 4] ^= 1
    try changed.write(to: fixture.url, options: .atomic)

    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("direct-header validation"))
    }
  }

  func testParallelMappedAuthenticationLoadsExactDirectV3Payload() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    let source = try MetalCompactH5Loader.load(
      sourceURL: fixture.url,
      device: device,
      authenticationPolicy: .parallelMapped
    )
    XCTAssertEqual(
      source.loadMetrics.mappedAuthenticationBytes,
      try FileManager.default.attributesOfItem(atPath: fixture.url.path)[.size]
        as? UInt64
    )
    XCTAssertEqual(
      try source.extractDiffraction(scanRow: 4, scanColumn: 13),
      fixture.values.map { $0[77] }
    )
  }

  func testParallelMappedAuthenticationRejectsChangedDirectV3Payload() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture()
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    var changed = try Data(contentsOf: fixture.url)
    changed[8_192] ^= 1
    try changed.write(to: fixture.url, options: .atomic)

    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(
        sourceURL: fixture.url,
        device: device,
        authenticationPolicy: .parallelMapped
      )
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("parallel authentication failed"))
    }
  }

  func testDirectV3BindsRawExclusionConstantsToOrderedPixels() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture(rawExclusions: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    let source = try MetalCompactH5Loader.load(
      sourceURL: fixture.url,
      device: device
    )
    XCTAssertEqual(source.metadata.rawAccessMode, "exact_exclusion_constants")
    XCTAssertEqual(source.metadata.maskedDetectorRawValues, [UInt16.max])
    var pixel = UInt32(3).littleEndian
    let expected = Swift.withUnsafeBytes(of: &pixel) {
      SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
    }
    XCTAssertEqual(source.metadata.maskedDetectorPixelsSHA256, expected)
  }

  func testCancelledDirectV3GenerationNeverPublishes() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    for cancelAt in [1, 3] {
      let fixture = try makeDirectCompactFixture()
      defer { try? FileManager.default.removeItem(at: fixture.url) }
      var calls = 0

      XCTAssertThrowsError(
        try MetalCompactH5Loader.load(
          sourceURL: fixture.url,
          device: device,
          shouldCancel: {
            calls += 1
            return calls == cancelAt
          }
        )
      ) { error in
        guard case Metal4DSTEMStreamingIOError.cancelled = error else {
          XCTFail("Expected cancellation before resident publication, got \(error)")
          return
        }
      }
    }
  }

  func testPartialRawExclusionFieldsRemainNonportable() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture(partialRawExclusions: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    let source = try MetalCompactH5Loader.load(
      sourceURL: fixture.url,
      device: device
    )
    XCTAssertEqual(source.metadata.rawAccessMode, "mask_applied_only_legacy")
    XCTAssertNil(source.metadata.maskedDetectorPixelsSHA256)
    XCTAssertNil(source.metadata.maskedDetectorRawValues)
  }

  func testPreparedDPCMomentsPrimeExactCenteredDisplayMaps() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture(
      rawExclusions: true,
      preparedDPC: true
    )
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    let source = try MetalCompactH5Loader.load(
      sourceURL: fixture.url,
      device: device
    )
    let contract = try XCTUnwrap(source.metadata.preparedDPCMoments)
    XCTAssertEqual(contract.scanCount, 128)
    XCTAssertEqual(contract.fileBytes, 128 * 8 * 4)
    XCTAssertEqual(contract.selectedDetectorPixels, 5)
    let maps = try XCTUnwrap(source.preparedDPCValues())
    XCTAssertNotNil(
      try source.preparedDPCDisplayBuffer(component: .row)
    )
    let residentDPC = try Metal4DSTEMDPCProcessor(device: device).process(
      centeredRowBuffer: try XCTUnwrap(
        source.preparedDPCDisplayBuffer(component: .row)
      ),
      centeredColumnBuffer: try XCTUnwrap(
        source.preparedDPCDisplayBuffer(component: .column)
      ),
      configuration: Metal4DSTEMDPCConfiguration(
        scanRows: 8,
        scanColumns: 16,
        rotationDegrees: 0,
        transposeComponents: false
      )
    )
    XCTAssertEqual(residentDPC.phaseBuffer.length, 128 * 4)
    XCTAssertEqual(residentDPC.gradientFFTBuffer.length, 128 * 8)
    XCTAssertEqual(residentDPC.phaseFFTBuffer.length, 128 * 8)
    XCTAssertEqual(residentDPC.metrics.uploadBytes, 0)
    XCTAssertEqual(residentDPC.metrics.readbackBytes, 0)
    let selectedPixels = [0, 1, 2, 4, 5]
    var expectedRow = [Float](repeating: 0, count: 128)
    var expectedColumn = [Float](repeating: 0, count: 128)
    var expectedTotal = [UInt64](repeating: 0, count: 128)
    var expectedRowMoment = [UInt64](repeating: 0, count: 128)
    var expectedColumnMoment = [UInt64](repeating: 0, count: 128)
    for scan in 0..<128 {
      let total = selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(fixture.values[$1][scan])
      }
      let rowMoment = selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(fixture.values[$1][scan]) * UInt64($1 / 3)
      }
      let columnMoment = selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(fixture.values[$1][scan]) * UInt64($1 % 3)
      }
      expectedTotal[scan] = total
      expectedRowMoment[scan] = rowMoment
      expectedColumnMoment[scan] = columnMoment
      if total != 0 {
        expectedRow[scan] = Float(Double(rowMoment) / Double(total))
        expectedColumn[scan] = Float(Double(columnMoment) / Double(total))
      }
    }
    let rowMean = Float(expectedRow.reduce(Double(0)) { $0 + Double($1) } / 128)
    let columnMean = Float(
      expectedColumn.reduce(Double(0)) { $0 + Double($1) } / 128
    )
    for scan in 0..<128 {
      expectedRow[scan] -= rowMean
      expectedColumn[scan] -= columnMean
    }
    XCTAssertEqual(maps.row, expectedRow)
    XCTAssertEqual(maps.column, expectedColumn)
    let moments = try XCTUnwrap(source.preparedDPCMomentValues())
    XCTAssertEqual(moments.total, expectedTotal)
    XCTAssertEqual(moments.detectorRowMoment, expectedRowMoment)
    XCTAssertEqual(moments.detectorColumnMoment, expectedColumnMoment)
    XCTAssertTrue(
      try Metal4DSTEMResidentCapabilities.compact(source)
        .fullInteractiveResident
    )
    XCTAssertEqual(source.loadMetrics.preparedDPCBytes, 128 * 8 * 4)
    XCTAssertGreaterThan(source.loadMetrics.preparedDPCReadMilliseconds, 0)
    XCTAssertGreaterThan(source.loadMetrics.preparedDPCAuthenticationMilliseconds, 0)
    XCTAssertGreaterThan(source.loadMetrics.preparedDPCPrimeMilliseconds, 0)
  }

  func testPreparedDPCManifestMismatchesFailClosed() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let cases: [([String: Any], String)] = [
      (["working_uint8_sha256": String(repeating: "0", count: 64)], "working_uint8_sha256"),
      (["detector_mask_sha256": String(repeating: "0", count: 64)], "detector_mask_sha256"),
      (["file_bytes": 4], "byte range"),
      (["file_offset": 8_192], "overlaps shard 0 payload"),
      (["layout": ["total_lo"]], "word layout"),
    ]
    for (overrides, message) in cases {
      let fixture = try makeDirectCompactFixture(
        preparedDPC: true,
        preparedDPCOverrides: overrides
      )
      defer { try? FileManager.default.removeItem(at: fixture.url) }
      XCTAssertThrowsError(
        try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
      ) { error in
        XCTAssertTrue(error.localizedDescription.contains(message))
      }
    }
  }

  func testChangedPreparedDPCBytesFailAuthenticatedLoad() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture(preparedDPC: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    var changed = try Data(contentsOf: fixture.url)
    changed[changed.count - 1] ^= 1
    try changed.write(to: fixture.url, options: .atomic)

    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("prepared DPC SHA-256"))
    }
  }

  func testPreparedDetectorProductsActivateByAuthenticatedCopy() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture(preparedDetectorProducts: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    let source = try MetalCompactH5Loader.load(
      sourceURL: fixture.url,
      device: device
    )
    let prepared = try XCTUnwrap(source.metadata.preparedDetectorProducts)
    XCTAssertEqual(prepared.products.map(\.name), ["bf", "abf", "adf"])
    for name in [
      MetalCompactH5PreparedDetectorProductName.bf,
      .abf,
      .adf,
    ] {
      let product = try XCTUnwrap(
        prepared.products.first { $0.name == name.rawValue }
      )
      let file = try Data(contentsOf: fixture.url)
      let maskStart = Int(product.maskFileOffset)
      let maskEnd = Int(product.maskFileOffset + product.maskFileBytes)
      let mask = [UInt8](file[maskStart..<maskEnd])
      let expected = (0..<128).map { scan in
        mask.indices.reduce(UInt32(0)) {
          $0 + (mask[$1] == 0 ? 0 : fixture.values[$1][scan])
        }
      }
      let metrics = try source.activatePreparedDetectorProduct(name)
      XCTAssertEqual(metrics.mode, "prepared")
      XCTAssertEqual(try source.virtualDetectorValues(), expected)
    }
    XCTAssertEqual(source.loadMetrics.preparedDetectorProductBytes, 3 * (6 + 512))
    XCTAssertGreaterThan(source.loadMetrics.preparedDetectorProductReadMilliseconds, 0)
    XCTAssertGreaterThan(
      source.loadMetrics.preparedDetectorProductAuthenticationMilliseconds,
      0
    )
  }

  func testPreparedDetectorProductManifestMismatchesFailClosed() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let cases: [([String: Any], [String: [String: Any]], String)] = [
      (
        ["working_uint8_sha256": String(repeating: "0", count: 64)],
        [:],
        "working_uint8_sha256"
      ),
      (
        ["detector_calibration_sha256": String(repeating: "0", count: 64)],
        [:],
        "detector_calibration_sha256"
      ),
      (["product_order": ["adf", "abf", "bf"]], [:], "shape or order"),
      ([:], ["bf": ["mask_file_offset": 8_192]], "overlap"),
      ([:], ["abf": ["outer_radius_px": 9.0]], "geometry"),
    ]
    for (rootOverrides, productOverrides, message) in cases {
      let fixture = try makeDirectCompactFixture(
        preparedDetectorProducts: true,
        preparedDetectorOverrides: rootOverrides,
        preparedDetectorProductOverrides: productOverrides
      )
      defer { try? FileManager.default.removeItem(at: fixture.url) }
      XCTAssertThrowsError(
        try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
      ) { error in
        XCTAssertTrue(error.localizedDescription.contains(message))
      }
    }
  }

  func testChangedPreparedDetectorValuesFailAuthenticatedLoad() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture(preparedDetectorProducts: true)
    defer { try? FileManager.default.removeItem(at: fixture.url) }
    var changed = try Data(contentsOf: fixture.url)
    changed[changed.count - 1] ^= 1
    try changed.write(to: fixture.url, options: .atomic)

    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("prepared ADF values SHA-256"))
    }
  }

  func testPreparedDetectorSelectedCountMismatchFailsLoad() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let fixture = try makeDirectCompactFixture(
      preparedDetectorProducts: true,
      preparedDetectorProductOverrides: [
        "bf": ["selected_detector_pixels": 0]
      ]
    )
    defer { try? FileManager.default.removeItem(at: fixture.url) }

    XCTAssertThrowsError(
      try MetalCompactH5Loader.load(sourceURL: fixture.url, device: device)
    ) { error in
      XCTAssertTrue(error.localizedDescription.contains("BF mask selects"))
    }
  }
}

private struct CompactFixture {
  let url: URL
  let values: [[UInt32]]
  let headerOffset: Int
}

private func makeCompactFixture(
  portable: Bool = false,
  fullRange: Bool = false,
  preparedDPC: Bool = false,
  preparedDPCOverrides: [String: Any] = [:],
  scanOrigin: Int = 0
) throws -> CompactFixture {
  let widths: [UInt8] = [2, 3, 4, 16, fullRange ? 16 : 9, 2]
  var values: [[UInt32]] = []
  values.reserveCapacity(widths.count)
  for (pixel, width) in widths.enumerated() {
    let valueLimit = 1 << Int(width)
    var pixelValues: [UInt32] = []
    pixelValues.reserveCapacity(128)
    for scan in 0..<128 {
      let value =
        pixel == 3
        ? UInt32(0)
        : UInt32(((scan + scanOrigin) * (pixel + 3) + pixel) % valueLimit)
      pixelValues.append(value)
    }
    values.append(pixelValues)
  }
  values[3] = [UInt32](repeating: 0, count: 128)
  if fullRange {
    let boundaries: [UInt32] = [0, 255, 256, UInt32(UInt16.max)]
    values[4] = (0..<128).map { boundaries[$0 % boundaries.count] }
  }
  var decodedWords = [UInt32](
    repeating: 0,
    count: widths.reduce(0) { $0 + Int($1) * 4 }
  )
  var wordOffset = 0
  for pixel in widths.indices {
    let width = Int(widths[pixel])
    for scan in 0..<128 {
      let bit = scan * width
      let word = wordOffset + bit / 32
      let shift = bit % 32
      decodedWords[word] |= values[pixel][scan] << UInt32(shift)
      if shift + width > 32 {
        decodedWords[word + 1] |= values[pixel][scan] >> UInt32(32 - shift)
      }
    }
    wordOffset += width * 4
  }
  var decoded = Data()
  for word in decodedWords { decoded.appendLE(word) }
  var payload = Data()
  var lengths: [UInt8] = []
  for offset in stride(from: 0, to: decoded.count, by: 128) {
    let block = decoded[offset..<min(offset + 128, decoded.count)]
    var encoded = Data([0xf0, UInt8(block.count - 15)])
    encoded.append(block)
    lengths.append(UInt8(encoded.count - 1))
    payload.append(encoded)
  }
  let decodedSHA = SHA256.hash(data: decoded).map {
    String(format: "%02x", $0)
  }.joined()
  let sourceIdentityBytes = Data(0..<32)
  let sourceIdentity = sourceIdentityBytes.map {
    String(format: "%02x", $0)
  }.joined()
  let binaryOffset: UInt32 = 4_096
  let binaryBytes = UInt32(172)
  let payloadOffset: UInt64 = 8_192
  let lengthsOffset = payloadOffset + UInt64(payload.count)
  let widthsOffset = lengthsOffset + UInt64(lengths.count)
  let preparedOffset = (widthsOffset + UInt64(widths.count) + 3) & ~UInt64(3)
  var preparedData = Data()
  var manifest: [String: Any] = [
    "schema": "quantem.gpu.packed-detector-h5/v1",
    "status": "complete",
    "source_shape": [8, 16, 2, 3],
    "source_dtype": "uint16",
    "working_dtype": "uint16",
    "source_identity_sha256": sourceIdentity,
    "source_raw_logical_sha256": String(repeating: "f", count: 64),
    "scan_bin": 1,
    "detector_bin": 1,
    "crop": NSNull(),
    "shard_count": 1,
    "scans_per_shard": 128,
    "payload_chunk_bytes": 128,
    "payload_chunk_codec": "independent raw LZ4 blocks",
    "payload_chunk_length_codec": "uint8 encoded_bytes_minus_one",
    "descriptor_codec": "uint8 five-bit widths",
    "masked_detector_pixels": [[1, 0]],
    "detector_calibration": [
      "schema": "quantem.gpu.detector-calibration/v1",
      "source_identity_sha256": sourceIdentity,
      "detector_center_px": [0.75, 1.25],
      "bright_field_radius_px": 1.5,
      "dpc_rotation_degrees": 176.25,
      "dpc_component_order_exchanged": false,
      "method": "test-fixture",
    ],
  ]
  if portable {
    var pixel = UInt32(3).littleEndian
    manifest["detector_mask_sha256"] = Swift.withUnsafeBytes(of: &pixel) {
      SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
    }
    manifest["masked_detector_payload_policy"] = "retained_exactly_in_payload"
  }
  if preparedDPC {
    var workingData = Data()
    for scan in 0..<128 {
      for pixel in values.indices {
        workingData.appendLE(UInt16(values[pixel][scan]))
      }
    }
    let workingSHA = SHA256.hash(data: workingData).map {
      String(format: "%02x", $0)
    }.joined()
    manifest["working_logical_sha256"] = workingSHA
    manifest["masked_detector_raw_values"] = [0]
    var pixel = UInt32(3).littleEndian
    manifest["masked_detector_pixels_sha256"] = Swift.withUnsafeBytes(of: &pixel) {
      SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
    }
    let selectedPixels = [0, 1, 2, 4, 5]
    for scan in 0..<128 {
      let total = selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(values[$1][scan])
      }
      let rowMoment = selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(values[$1][scan]) * UInt64($1 / 3)
      }
      let columnMoment = selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(values[$1][scan]) * UInt64($1 % 3)
      }
      for value in [total, rowMoment, columnMoment] {
        preparedData.appendLE(UInt32(value & 0xffff_ffff))
        preparedData.appendLE(UInt32(value >> 32))
      }
      preparedData.appendLE(UInt32(0))
      preparedData.appendLE(UInt32(0))
    }
    var contract: [String: Any] = [
      "schema": "quantem.gpu.prepared-dpc-moments/v2",
      "source_identity_sha256": sourceIdentity,
      "working_logical_sha256": workingSHA,
      "working_dtype": "uint16",
      "detector_mask_sha256": manifest["detector_mask_sha256"]!,
      "detector_selection": "all-nonexcluded-v1",
      "scan_count": 128,
      "selected_detector_pixels": 5,
      "detector_columns": 3,
      "maximum_value": Int(UInt16.max),
      "dtype": "little-endian-u32",
      "word_order": "little-endian-u32-pairs",
      "words_per_scan": 8,
      "layout": [
        "total_lo", "total_hi", "row_lo", "row_hi",
        "column_lo", "column_hi", "padding_0", "padding_1",
      ],
      "file_offset": preparedOffset,
      "file_bytes": preparedData.count,
      "sha256": SHA256.hash(data: preparedData).map {
        String(format: "%02x", $0)
      }.joined(),
      "total_bound": String(5 * Int(UInt16.max)),
      "row_moment_bound": String(2 * Int(UInt16.max)),
      "column_moment_bound": String(6 * Int(UInt16.max)),
      "narrow_integer": true,
      "narrow_products": true,
    ]
    contract.merge(preparedDPCOverrides) { _, replacement in replacement }
    manifest["prepared_dpc_moments"] = contract
  }
  let header = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])

  var binary = Data([0x51, 0x47, 0x49, 0x58, 0x00, 0x00, 0x00, 0x01])
  for value: UInt32 in [1, 128, 8, 16, 2, 3, 128] { binary.appendLE(value) }
  binary.appendLE(UInt32(1))
  binary.appendLE(UInt32(3))
  binary.append(sourceIdentityBytes)
  for value: UInt64 in [
    payloadOffset,
    UInt64(payload.count),
    lengthsOffset,
    UInt64(lengths.count),
    widthsOffset,
    UInt64(widths.count),
    UInt64(decoded.count),
  ] { binary.appendLE(value) }
  binary.appendLE(UInt32(widths.count))
  binary.appendLE(UInt32(lengths.count))
  binary.append(Data(hex: decodedSHA))
  XCTAssertEqual(binary.count, Int(binaryBytes))

  var prelude = Data([0x51, 0x47, 0x50, 0x55, 0x48, 0x35, 0x00, 0x01])
  prelude.appendLE(UInt32(header.count))
  prelude.appendLE(crc32ForFixture(header))
  prelude.appendLE(binaryOffset)
  prelude.appendLE(binaryBytes)
  var file = Data(count: Int(preparedOffset) + preparedData.count)
  file.replaceSubrange(0..<prelude.count, with: prelude)
  file.replaceSubrange(24..<(24 + header.count), with: header)
  file.replaceSubrange(
    Int(binaryOffset)..<(Int(binaryOffset) + binary.count),
    with: binary
  )
  file.replaceSubrange(
    Int(payloadOffset)..<(Int(payloadOffset) + payload.count),
    with: payload
  )
  file.replaceSubrange(
    Int(lengthsOffset)..<(Int(lengthsOffset) + lengths.count),
    with: lengths
  )
  file.replaceSubrange(
    Int(widthsOffset)..<(Int(widthsOffset) + widths.count),
    with: widths
  )
  if !preparedData.isEmpty {
    file.replaceSubrange(
      Int(preparedOffset)..<(Int(preparedOffset) + preparedData.count),
      with: preparedData
    )
  }
  let url = FileManager.default.temporaryDirectory.appendingPathComponent(
    "quantem-compact-\(UUID().uuidString).h5"
  )
  try file.write(to: url, options: .atomic)
  return CompactFixture(url: url, values: values, headerOffset: Int(widthsOffset))
}

private func makeDirectCompactFixture(
  rawExclusions: Bool = false,
  partialRawExclusions: Bool = false,
  preparedDPC: Bool = false,
  preparedDPCOverrides: [String: Any] = [:],
  preparedDetectorProducts: Bool = false,
  preparedDetectorOverrides: [String: Any] = [:],
  preparedDetectorProductOverrides: [String: [String: Any]] = [:]
) throws -> CompactFixture {
  let tileWidths: [[UInt8]] = [
    [2, 3, 4, 5],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [0, 0, 0, 0],
    [8, 7, 8, 6],
    [2, 2, 2, 2],
  ]
  var values = [[UInt32]](
    repeating: [UInt32](repeating: 0, count: 128),
    count: tileWidths.count
  )
  var payloadWords: [UInt32] = []
  var headers: [UInt32] = []
  for pixel in tileWidths.indices {
    let base = UInt32(payloadWords.count)
    var packedWidths: UInt32 = 0
    for tile in 0..<4 {
      let width = Int(tileWidths[pixel][tile])
      packedWidths |= UInt32(width) << UInt32(tile * 4)
      let tileBase = payloadWords.count
      payloadWords.append(contentsOf: repeatElement(0, count: width))
      for localScan in 0..<32 {
        let scan = tile * 32 + localScan
        let value =
          pixel == 3 || width == 0
          ? UInt32(0)
          : UInt32((scan * (pixel + 3) + pixel) % (1 << width))
        values[pixel][scan] = value
        if width != 0 {
          let bit = localScan * width
          let word = tileBase + bit / 32
          let shift = bit % 32
          payloadWords[word] |= value << UInt32(shift)
          if shift + width > 32 {
            payloadWords[word + 1] |= value >> UInt32(32 - shift)
          }
        }
      }
    }
    headers.append(base)
    headers.append(packedWidths)
  }
  var payload = Data()
  for word in payloadWords { payload.appendLE(word) }
  var headerData = Data()
  for word in headers { headerData.appendLE(word) }
  let payloadSHA = SHA256.hash(data: payload).map {
    String(format: "%02x", $0)
  }.joined()
  var logicalValues = Data()
  for scan in 0..<128 {
    for pixel in values.indices {
      logicalValues.append(UInt8(values[pixel][scan]))
    }
  }
  let logicalSHA = SHA256.hash(data: logicalValues).map {
    String(format: "%02x", $0)
  }.joined()
  let sourceIdentityBytes = Data((32..<64).map(UInt8.init))
  let sourceIdentity = sourceIdentityBytes.map {
    String(format: "%02x", $0)
  }.joined()
  var manifest: [String: Any] = [
    "schema": "quantem.gpu.packed-detector-h5/v3",
    "status": "complete",
    "payload_codec": "direct-bitpacked-u32",
    "source_identity_sha256": sourceIdentity,
    "source_raw_logical_sha256": String(repeating: "d", count: 64),
    "source_shape": [8, 16, 2, 3],
    "source_dtype": "uint16",
    "working_dtype": "uint8",
    "working_value_definition":
      "all admitted source counts exactly; authenticated dead pixels set to zero",
    "prepared_uint8_sha256": logicalSHA,
    "detector_mask_sha256": String(repeating: "a", count: 64),
    "masked_detector_pixels": [3],
    "scan_bin": 1,
    "detector_bin": 1,
    "crop": NSNull(),
    "scan_tile": 32,
    "shard_count": 1,
  ]
  if rawExclusions || partialRawExclusions {
    manifest["masked_detector_raw_values"] = [Int(UInt16.max)]
  }
  if rawExclusions {
    var pixel = UInt32(3).littleEndian
    let pixelSHA = Swift.withUnsafeBytes(of: &pixel) {
      SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
    }
    manifest["masked_detector_pixels_sha256"] = pixelSHA
  }
  let binaryOffset: UInt32 = 4_096
  let binaryBytes = UInt32(180)
  let payloadOffset: UInt64 = 8_192
  let headersOffset = payloadOffset + UInt64(payload.count)
  var cursor = headersOffset + UInt64(headerData.count)
  let preparedOffset = cursor
  var preparedData = Data()
  var extraRanges: [(UInt64, Data)] = []
  if preparedDPC {
    let selectedPixels = [0, 1, 2, 4, 5]
    for scan in 0..<128 {
      let total = selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(values[$1][scan])
      }
      let rowMoment = selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(values[$1][scan]) * UInt64($1 / 3)
      }
      let columnMoment = selectedPixels.reduce(UInt64(0)) {
        $0 + UInt64(values[$1][scan]) * UInt64($1 % 3)
      }
      for value in [total, rowMoment, columnMoment] {
        preparedData.appendLE(UInt32(value & 0xffff_ffff))
        preparedData.appendLE(UInt32(value >> 32))
      }
      preparedData.appendLE(UInt32(0))
      preparedData.appendLE(UInt32(0))
    }
    var contract: [String: Any] = [
      "schema": "quantem.gpu.prepared-dpc-moments/v1",
      "source_identity_sha256": sourceIdentity,
      "working_uint8_sha256": logicalSHA,
      "detector_mask_sha256": String(repeating: "a", count: 64),
      "detector_selection": "all-nonexcluded-v1",
      "scan_count": 128,
      "selected_detector_pixels": 5,
      "detector_columns": 3,
      "dtype": "little-endian-u32",
      "word_order": "little-endian-u32-pairs",
      "words_per_scan": 8,
      "layout": [
        "total_lo", "total_hi", "row_lo", "row_hi",
        "column_lo", "column_hi", "padding_0", "padding_1",
      ],
      "file_offset": preparedOffset,
      "file_bytes": preparedData.count,
      "sha256": SHA256.hash(data: preparedData).map {
        String(format: "%02x", $0)
      }.joined(),
      "total_bound": String(5 * 255),
      "row_moment_bound": String(2 * 255),
      "column_moment_bound": String(6 * 255),
      "narrow_integer": true,
      "narrow_products": true,
    ]
    contract.merge(preparedDPCOverrides) { _, replacement in replacement }
    manifest["prepared_dpc_moments"] = contract
    extraRanges.append((preparedOffset, preparedData))
    cursor += UInt64(preparedData.count)
  }
  if preparedDetectorProducts {
    let calibration: [String: Any] = [
      "schema": "quantem.gpu.detector-calibration/v1",
      "source_identity_sha256": sourceIdentity,
      "detector_center_px": [0, 0],
      "bright_field_radius_px": 1,
      "method": "synthetic-test",
    ]
    manifest["detector_calibration"] = calibration
    let canonicalCalibration =
      "{\"bright_field_radius_px\":\"f64be:3ff0000000000000\","
      + "\"detector_center_px\":[\"f64be:0000000000000000\","
      + "\"f64be:0000000000000000\"],"
      + "\"method\":\"synthetic-test\","
      + "\"schema\":\"quantem.gpu.detector-calibration/v1\","
      + "\"source_identity_sha256\":\"\(sourceIdentity)\"}"
    let calibrationSHA = SHA256.hash(data: Data(canonicalCalibration.utf8)).map {
      String(format: "%02x", $0)
    }.joined()
    let geometries: [(String, Double, Double)] = [
      ("bf", 0, 1),
      ("abf", 0.5, 1),
      ("adf", 1, 2),
    ]
    var records: [[String: Any]] = []
    for (name, inner, outer) in geometries {
      var mask = [UInt8](repeating: 0, count: 6)
      for pixel in 0..<6 where pixel != 3 {
        let row = Double(pixel / 3)
        let column = Double(pixel % 3)
        let distance = (row * row + column * column).squareRoot()
        mask[pixel] = distance >= inner && distance <= outer ? 1 : 0
      }
      let maskData = Data(mask)
      var productValues = Data()
      for scan in 0..<128 {
        let sum = mask.indices.reduce(UInt32(0)) {
          $0 + (mask[$1] == 0 ? 0 : values[$1][scan])
        }
        productValues.appendLE(sum)
      }
      let maskOffset = cursor
      extraRanges.append((maskOffset, maskData))
      cursor += UInt64(maskData.count)
      cursor = (cursor + 3) & ~UInt64(3)
      let valuesOffset = cursor
      extraRanges.append((valuesOffset, productValues))
      cursor += UInt64(productValues.count)
      var record: [String: Any] = [
        "name": name,
        "center_px": [0, 0],
        "inner_radius_px": inner,
        "outer_radius_px": outer,
        "selected_detector_pixels": mask.reduce(0) { $0 + Int($1) },
        "mask_file_offset": maskOffset,
        "mask_file_bytes": maskData.count,
        "mask_sha256": SHA256.hash(data: maskData).map {
          String(format: "%02x", $0)
        }.joined(),
        "values_file_offset": valuesOffset,
        "values_file_bytes": productValues.count,
        "values_sha256": SHA256.hash(data: productValues).map {
          String(format: "%02x", $0)
        }.joined(),
      ]
      record.merge(preparedDetectorProductOverrides[name] ?? [:]) {
        _, replacement in replacement
      }
      records.append(record)
    }
    var contract: [String: Any] = [
      "schema": "quantem.gpu.prepared-detector-products/v1",
      "source_identity_sha256": sourceIdentity,
      "working_uint8_sha256": logicalSHA,
      "detector_mask_sha256": String(repeating: "a", count: 64),
      "detector_calibration_sha256": calibrationSHA,
      "detector_calibration_digest_encoding":
        "canonical-json-numbers-as-f64be-hex/v1",
      "scan_shape": [8, 16],
      "detector_shape": [2, 3],
      "product_dtype": "little-endian-u32",
      "mask_dtype": "uint8-binary-row-major",
      "mask_rule": "quantem.gpu.detector-mask-inclusive/v1",
      "product_order": ["bf", "abf", "adf"],
      "products": records,
    ]
    contract.merge(preparedDetectorOverrides) { _, replacement in replacement }
    manifest["prepared_detector_products"] = contract
  }
  let header = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])

  var binary = Data([0x51, 0x47, 0x49, 0x58, 0x00, 0x00, 0x00, 0x03])
  for value: UInt32 in [1, 0, 8, 16, 2, 3, 128, 32, 1] {
    binary.appendLE(value)
  }
  binary.appendLE(UInt32(1))
  binary.appendLE(UInt32(3))
  binary.append(sourceIdentityBytes)
  for value: UInt64 in [
    payloadOffset,
    UInt64(payload.count),
    0,
    0,
    headersOffset,
    UInt64(headerData.count),
    UInt64(payload.count),
  ] { binary.appendLE(value) }
  binary.appendLE(UInt32(headers.count))
  binary.appendLE(UInt32(0))
  binary.append(Data(hex: payloadSHA))
  XCTAssertEqual(binary.count, Int(binaryBytes))

  var prelude = Data([0x51, 0x47, 0x50, 0x55, 0x48, 0x35, 0x00, 0x01])
  prelude.appendLE(UInt32(header.count))
  prelude.appendLE(crc32ForFixture(header))
  prelude.appendLE(binaryOffset)
  prelude.appendLE(binaryBytes)
  XCTAssertLessThanOrEqual(24 + header.count, Int(binaryOffset))
  var file = Data(count: Int(cursor))
  file.replaceSubrange(0..<prelude.count, with: prelude)
  file.replaceSubrange(24..<(24 + header.count), with: header)
  file.replaceSubrange(
    Int(binaryOffset)..<(Int(binaryOffset) + binary.count),
    with: binary
  )
  file.replaceSubrange(
    Int(payloadOffset)..<(Int(payloadOffset) + payload.count),
    with: payload
  )
  file.replaceSubrange(
    Int(headersOffset)..<(Int(headersOffset) + headerData.count),
    with: headerData
  )
  for (offset, data) in extraRanges {
    file.replaceSubrange(Int(offset)..<(Int(offset) + data.count), with: data)
  }
  let url = FileManager.default.temporaryDirectory.appendingPathComponent(
    "quantem-compact-direct-v3-\(UUID().uuidString).h5"
  )
  try file.write(to: url, options: .atomic)
  return CompactFixture(
    url: url,
    values: values,
    headerOffset: Int(headersOffset)
  )
}

/// Five distinct shards exercise ordering and a final two-shard partial window.
private func makeMultishardCompactFixture() throws -> CompactFixture {
  var fixtures: [CompactFixture] = []
  defer {
    for fixture in fixtures { try? FileManager.default.removeItem(at: fixture.url) }
  }
  for shard in 0..<5 {
    fixtures.append(try makeCompactFixture(portable: true, scanOrigin: shard * 7))
  }
  let sources = try fixtures.map { try Data(contentsOf: $0.url) }
  let first = sources[0]
  let headerBytes = first.withUnsafeBytes {
    Int($0.loadUnaligned(fromByteOffset: 8, as: UInt32.self).littleEndian)
  }
  var manifest = try XCTUnwrap(
    JSONSerialization.jsonObject(with: first[24..<(24 + headerBytes)]) as? [String: Any]
  )
  manifest["source_shape"] = [40, 16, 2, 3]
  manifest["shard_count"] = 5
  let header = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
  var binary = Data(first[4096..<(4096 + 76)])
  var count = Data()
  count.appendLE(UInt32(5))
  binary.replaceSubrange(8..<12, with: count)
  var rows = Data()
  rows.appendLE(UInt32(40))
  binary.replaceSubrange(16..<20, with: rows)
  var file = Data(count: 8192)
  for source in sources {
    var record = Data(source[(4096 + 76)..<(4096 + 76 + 96)])
    let relocation = UInt64(file.count - 8192)
    for field in [0, 16, 32] {
      let old = record.withUnsafeBytes {
        $0.loadUnaligned(fromByteOffset: field, as: UInt64.self).littleEndian
      }
      var value = Data()
      value.appendLE(old + relocation)
      record.replaceSubrange(field..<(field + 8), with: value)
    }
    binary.append(record)
    file.append(source[8192...])
  }
  var prelude = Data(first[0..<8])
  prelude.appendLE(UInt32(header.count))
  prelude.appendLE(crc32ForFixture(header))
  prelude.appendLE(UInt32(4096))
  prelude.appendLE(UInt32(binary.count))
  file.replaceSubrange(0..<prelude.count, with: prelude)
  file.replaceSubrange(24..<(24 + header.count), with: header)
  file.replaceSubrange(4096..<(4096 + binary.count), with: binary)
  let url = FileManager.default.temporaryDirectory
    .appendingPathComponent("quantem-compact-multishard-\(UUID().uuidString).h5")
  try file.write(to: url)
  return CompactFixture(
    url: url,
    values: (0..<6).map { pixel in fixtures.flatMap { $0.values[pixel] } },
    headerOffset: fixtures[0].headerOffset
  )
}

private func crc32ForFixture(_ data: Data) -> UInt32 {
  var crc = UInt32.max
  for byte in data {
    crc ^= UInt32(byte)
    for _ in 0..<8 {
      crc = (crc >> 1) ^ (0xedb8_8320 & (UInt32(0) &- (crc & 1)))
    }
  }
  return ~crc
}

extension Data {
  fileprivate init(hex: String) {
    self.init()
    var index = hex.startIndex
    while index < hex.endIndex {
      let next = hex.index(index, offsetBy: 2)
      append(UInt8(hex[index..<next], radix: 16)!)
      index = next
    }
  }

  fileprivate mutating func appendLE(_ value: UInt16) {
    var little = value.littleEndian
    Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
  }

  fileprivate mutating func appendLE(_ value: UInt32) {
    var little = value.littleEndian
    Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
  }

  fileprivate mutating func appendLE(_ value: UInt64) {
    var little = value.littleEndian
    Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
  }
}
