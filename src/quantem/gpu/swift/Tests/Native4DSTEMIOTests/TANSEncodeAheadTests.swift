import CryptoKit
import Foundation
import Metal
import XCTest

@_spi(EntropySeriesPrototype) @testable import Metal4DSTEMStreamingIO

/// Pipelined ("encode ahead") detector queries: the next query is planned,
/// encoded and committed while the previous one still runs on the GPU. Every
/// published count must equal the synchronous query's, including after a
/// failed query rolls back its seeds and those of the query seeded from it.
final class TANSEncodeAheadTests: XCTestCase {
  private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func increment() {
      lock.lock()
      count += 1
      lock.unlock()
    }
    var value: Int {
      lock.lock()
      defer { lock.unlock() }
      return count
    }
  }

  func testCompletionNotifiesOnceAfterSealAndEveryCommand() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let completion = TANSDetectorCompletion()
    let early = Counter()
    completion.notify { early.increment() }
    var commands: [MTLCommandBuffer] = []
    for _ in 0..<3 {
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      completion.track(command)
      command.commit()
      commands.append(command)
    }
    for command in commands { command.waitUntilCompleted() }
    // Every command completed, but more may still be tracked: nothing fires.
    XCTAssertEqual(early.value, 0)
    XCTAssertFalse(completion.isComplete)
    let fired = expectation(description: "sealed completion")
    completion.notify { fired.fulfill() }
    completion.seal()
    wait(for: [fired], timeout: 10)
    XCTAssertEqual(early.value, 1, "each handler runs exactly once")
    XCTAssertTrue(completion.isComplete)
    let late = Counter()
    completion.notify { late.increment() }
    XCTAssertEqual(late.value, 1, "a handler added after completion runs at once")
    completion.seal()
    XCTAssertEqual(early.value, 1)

    let empty = TANSDetectorCompletion()
    let immediate = Counter()
    empty.notify { immediate.increment() }
    empty.seal()
    XCTAssertEqual(immediate.value, 1, "a query with no deferred command completes at seal")
  }

  /// An all-66 pointer walk (the app's lockstep order, a single-acquisition
  /// hover frame that splits the seeds into two groups, atlas bases and a
  /// jump) run twice: sequentially, and with two queries in flight. Every
  /// frame's 66 images must be bit-identical. Midway, one pipelined query is
  /// forced to fail after its commands completed: it and the query seeded from
  /// it must throw, the previous frame's images must stay intact, and the walk
  /// resumed from the restored seeds must still match the sequential frames.
  func testPipelinedAll66WalkMatchesSequentialAndRollsBackWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_SPI_FIXTURE"] else {
      throw XCTSkip("Requires the complete sealed 66-acquisition archive and a 128 GB-class device")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let source = try ExperimentalMetalEntropySeries(
      directory: URL(fileURLWithPath: path), acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.release() }
    try source.prepareDetectorIndex(maximumIndexBytes: 2 << 30)
    source.configureInteractiveGrouping(mixedModelTails: true, chooseCheaperBase: true)
    let valid = source.validDetectorMask
    func mask(_ row: Double, _ col: Double, _ inner: Double = 40, _ outer: Double = 80) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - col
        let radius2 = r * r + c * c
        return valid[q] != 0 && radius2 >= inner * inner && radius2 <= outer * outer ? 1 : 0
      }
    }
    func hashes(_ images: [MTLBuffer]) -> [String] {
      images.map { buffer in
        SHA256.hash(
          data: Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none)
        )
        .map { String(format: "%02x", $0) }.joined()
      }
    }
    let budget: UInt64 = 1 << 30
    // Exact atlas fields on the 2 px lattice the walk crosses, as the app
    // builds them while idle; frames may start from them instead of a seed.
    try source.beginDetectorAtlas(maximumBytes: 1 << 30)
    var lattice: [(Double, Double)] = [(97.5, 91.5)]
    for row in [95.5, 97.5] {
      for column in stride(from: 95.5, through: 117.5, by: 2) { lattice.append((row, column)) }
    }
    for (row, column) in lattice { try source.appendDetectorAtlasField(mask: mask(row, column)) }

    let inspected = 7
    let lockstep = [inspected] + (0..<66).filter { $0 != inspected }
    var frames: [(mask: [UInt8], acquisitions: [Int])] = []
    var column = 95.5
    for step in 0..<16 {
      // 100 px/s at 120 Hz input (0.83 px per frame), then 2.5 px steps.
      column += step < 10 ? 0.83 : 2.5
      frames.append((mask(95.5 + 0.2 * Double(step), column), lockstep))
    }
    // Hover frame: acquisition 33 alone moves; the next all-66 frame has two
    // seed groups (33 and the rest).
    frames.insert((mask(96.0, 99.0), [33]), at: 6)
    // A jump back onto an atlas centre, and a final small step.
    frames.append((mask(97.5, 91.5), lockstep))
    frames.append((mask(97.7, 92.3), lockstep))
    let start = mask(95.5, 95.5)

    // Sequential: submit and finish each frame before the next.
    _ = try source.detectorImages(mask: start, maximumAdditionalBytes: budget, rebase: true)
    var sequential: [[String]] = []
    for frame in frames {
      let submission = try source.submitDetectorImages(
        mask: frame.mask, maximumAdditionalBytes: budget, selectedAcquisitions: frame.acquisitions)
      XCTAssertEqual(submission.acquisitions, frame.acquisitions)
      let result = try source.finishDetectorImages(submission)
      XCTAssertEqual(result.images.count, frame.acquisitions.count)
      XCTAssertEqual(result.commandTiming["queries_in_flight_at_submit"], 0)
      sequential.append(hashes(result.images))
    }
    XCTAssertEqual(source.detectorQueriesInFlight, 0)
    // Independent anchors: unseeded recomputes of the first and last frames.
    for index in [0, frames.count - 1] {
      let reference = try source.detectorImages(
        mask: frames[index].mask, maximumAdditionalBytes: budget, rebase: true,
        selectedAcquisitions: frames[index].acquisitions)
      XCTAssertEqual(hashes(reference), sequential[index], "sequential frame \(index) is exact")
    }

    // Pipelined: frame n+1 is submitted before frame n is finished.
    _ = try source.detectorImages(mask: start, maximumAdditionalBytes: budget, rebase: true)
    let failAt = 10
    var pipelined = [[String]?](repeating: nil, count: frames.count)
    var inFlight: [(index: Int, submission: ExperimentalMetalEntropySeries.DetectorSubmission)] = []
    var next = 0
    var injected = false
    var checkedThirdSubmit = false
    var overlapped = 0
    var atlasFrames = 0
    var lastFinished: (index: Int, images: [MTLBuffer])?
    while next < frames.count || !inFlight.isEmpty {
      while inFlight.count < ExperimentalMetalEntropySeries.maximumDetectorQueriesInFlight,
        next < frames.count
      {
        inFlight.append(
          (
            next,
            try source.submitDetectorImages(
              mask: frames[next].mask, maximumAdditionalBytes: budget,
              selectedAcquisitions: frames[next].acquisitions)
          ))
        next += 1
      }
      if inFlight.count == 2 {
        XCTAssertEqual(source.detectorQueriesInFlight, 2)
        if !checkedThirdSubmit {
          checkedThirdSubmit = true
          XCTAssertThrowsError(
            try source.submitDetectorImages(
              mask: start, maximumAdditionalBytes: budget, selectedAcquisitions: [0]),
            "a third query in flight would rewrite a slot a rollback may restore")
          XCTAssertEqual(source.detectorQueriesInFlight, 2)
        }
        if !injected, inFlight[0].index == failAt {
          injected = true
          source.experimentalFailedDetectorQuerySequences = [inFlight[0].submission.sequence]
          // Finish the dependent query first: it must see the earlier failure.
          XCTAssertThrowsError(try source.finishDetectorImages(inFlight[1].submission)) { error in
            XCTAssertTrue(
              String(describing: error).contains("seeded from failed query"), "\(error)")
          }
          XCTAssertThrowsError(try source.finishDetectorImages(inFlight[0].submission)) { error in
            XCTAssertTrue(String(describing: error).contains("injected failure"), "\(error)")
          }
          source.experimentalFailedDetectorQuerySequences = []
          XCTAssertEqual(source.detectorQueriesInFlight, 0)
          if let last = lastFinished {
            XCTAssertEqual(
              hashes(last.images), pipelined[last.index], "rollback keeps frame images")
          }
          print("ENCODE_AHEAD_ROLLBACK failed_frame=\(failAt) resumed=true")
          next = inFlight[0].index
          inFlight.removeAll()
          continue
        }
      }
      let (index, submission) = inFlight.removeFirst()
      let completed = expectation(description: "GPU completion of frame \(index)")
      submission.whenGPUCompleted { completed.fulfill() }
      wait(for: [completed], timeout: 120)
      let result = try source.finishDetectorImages(submission)
      pipelined[index] = hashes(result.images)
      if result.commandTiming["queries_in_flight_at_submit"] == 1 { overlapped += 1 }
      if result.atlasField != nil { atlasFrames += 1 }
      // The previous frame's images are still intact: only the third later
      // query of an acquisition rewrites its ring slot.
      if let last = lastFinished {
        XCTAssertEqual(hashes(last.images), pipelined[last.index], "frame \(last.index) ring slot")
      }
      lastFinished = (index, result.images)
    }
    XCTAssertTrue(injected)
    XCTAssertEqual(source.detectorQueriesInFlight, 0)
    XCTAssertGreaterThan(overlapped, frames.count / 2, "most frames were submitted ahead")
    for index in frames.indices {
      XCTAssertEqual(pipelined[index], sequential[index], "pipelined frame \(index)")
    }
    print(
      "ENCODE_AHEAD_TEST frames=\(frames.count) images=\(frames.reduce(0) { $0 + $1.acquisitions.count }) bit_identical=true overlapped=\(overlapped) atlas_frames=\(atlasFrames) atlas_fields=\(source.detectorAtlasFieldCount)"
    )
  }

  /// A query's completion includes the earlier queries it is finished after,
  /// so a caller that awaits it never blocks in the finish step even when an
  /// independent later query completes first on the GPU.
  func testCompletionWaitsForEarlierQueries() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    // Two queues, so the later command can complete while the earlier one is
    // not even committed.
    let earlierQueue = try XCTUnwrap(device.makeCommandQueue())
    let laterQueue = try XCTUnwrap(device.makeCommandQueue())
    let earlier = TANSDetectorCompletion()
    let earlierCommand = try XCTUnwrap(earlierQueue.makeCommandBuffer())
    earlier.track(earlierCommand)
    earlier.seal()
    let later = TANSDetectorCompletion()
    later.track(earlier)
    let laterCommand = try XCTUnwrap(laterQueue.makeCommandBuffer())
    later.track(laterCommand)
    later.seal()
    let fired = Counter()
    later.notify { fired.increment() }
    laterCommand.commit()
    laterCommand.waitUntilCompleted()
    XCTAssertFalse(later.isComplete, "the earlier query has not completed")
    XCTAssertEqual(fired.value, 0)
    let done = expectation(description: "completion after the earlier query")
    later.notify { done.fulfill() }
    earlierCommand.commit()
    wait(for: [done], timeout: 10)
    XCTAssertTrue(earlier.isComplete)
    XCTAssertTrue(later.isComplete)
    XCTAssertEqual(fired.value, 1)
    // Tracking an already completed query adds nothing to wait for.
    let third = TANSDetectorCompletion()
    third.track(later)
    third.seal()
    XCTAssertTrue(third.isComplete)
  }

  private final class SeriesBox: @unchecked Sendable {
    let source: ExperimentalMetalEntropySeries
    var error: Error?
    var submitted = false
    init(_ source: ExperimentalMetalEntropySeries) { self.source = source }
  }

  /// Calls around unfinished pipelined queries, and failures during submit.
  /// A synchronous query or index build is refused while a submission is
  /// unfinished (it would rewrite images its caller has not read). A
  /// cancelled submit, and a query whose second seed group fails after its
  /// first group was committed, leave nothing in flight and every seed and
  /// returned image as it was; the retried query is exact. Release with a
  /// query in flight waits for it, and finishing it afterwards reports the
  /// release.
  func testRefusalsSubmitFailuresAndReleaseWithQueriesInFlightWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_SPI_FIXTURE"] else {
      throw XCTSkip("Requires the complete sealed 66-acquisition archive and a 128 GB-class device")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let source = try ExperimentalMetalEntropySeries(
      directory: URL(fileURLWithPath: path), acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.release() }
    try source.prepareDetectorIndex(maximumIndexBytes: 2 << 30)
    let valid = source.validDetectorMask
    func mask(_ row: Double, _ col: Double, _ inner: Double = 40, _ outer: Double = 80) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - col
        let radius2 = r * r + c * c
        return valid[q] != 0 && radius2 >= inner * inner && radius2 <= outer * outer ? 1 : 0
      }
    }
    func hashes(_ images: [MTLBuffer]) -> [String] {
      images.map { buffer in
        SHA256.hash(
          data: Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none)
        )
        .map { String(format: "%02x", $0) }.joined()
      }
    }
    let budget: UInt64 = 1 << 30
    let all = Array(0..<66)
    let others = all.filter { $0 != 33 }
    let m1 = mask(95.5, 96.3)
    let m2 = mask(95.7, 97.1)
    let m3 = mask(96.0, 99.0)
    let m4 = mask(95.9, 97.9)
    // Exact references: unseeded recomputes.
    func reference(_ mask: [UInt8], _ acquisitions: [Int]) throws -> [String] {
      hashes(
        try source.detectorImages(
          mask: mask, maximumAdditionalBytes: budget, rebase: true,
          selectedAcquisitions: acquisitions))
    }
    let r1 = try reference(m1, all)
    let r2 = try reference(m2, all)
    let r3 = try reference(m3, [33])
    let r4 = try reference(m4, [33] + others)
    _ = try source.detectorImages(
      mask: mask(95.5, 95.5), maximumAdditionalBytes: budget, rebase: true)

    // Refused while a submission is unfinished; nothing changes.
    let first = try source.submitDetectorImages(
      mask: m1, maximumAdditionalBytes: budget, selectedAcquisitions: all)
    XCTAssertThrowsError(try source.detectorImages(mask: m2, maximumAdditionalBytes: budget)) {
      XCTAssertTrue(String(describing: $0).contains("Finish the 1 submitted"), "\($0)")
    }
    XCTAssertThrowsError(try source.prepareDetectorIndex(maximumIndexBytes: 2 << 30)) {
      XCTAssertTrue(String(describing: $0).contains("Finish the 1 submitted"), "\($0)")
    }
    XCTAssertEqual(source.detectorQueriesInFlight, 1)
    XCTAssertEqual(hashes(try source.finishDetectorImages(first).images), r1)
    let second = try source.detectorImages(mask: m2, maximumAdditionalBytes: budget)
    XCTAssertEqual(hashes(second), r2, "the synchronous query runs once the submission finished")

    // A submit cancelled before its commit: nothing in flight, seeds unchanged.
    let box = SeriesBox(source)
    let cancelled = expectation(description: "cancelled submit")
    Task.detached { [box, m3, budget] in
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        _ = try box.source.submitDetectorImages(
          mask: m3, maximumAdditionalBytes: budget, selectedAcquisitions: [33])
        box.submitted = true
      } catch {
        box.error = error
      }
      cancelled.fulfill()
    }
    wait(for: [cancelled], timeout: 120)
    XCTAssertFalse(box.submitted)
    XCTAssertTrue(box.error is CancellationError, "\(String(describing: box.error))")
    XCTAssertEqual(source.detectorQueriesInFlight, 0)
    XCTAssertEqual(source.experimentalDetectorSeedMask(acquisition: 33), m2)
    XCTAssertEqual(hashes(second), r2)

    // Split the seeds: 33 moves alone, so the next query has two seed groups,
    // 33 first (one command) and the other 65 (many 64-record commands).
    let hover = try source.submitDetectorImages(
      mask: m3, maximumAdditionalBytes: budget, selectedAcquisitions: [33])
    let hoverImages = try source.finishDetectorImages(hover).images
    XCTAssertEqual(hashes(hoverImages), r3)
    source.experimentalDetectorRecordsPerCommand = 64
    source.experimentalFailAfterSubmittedCommands = 1
    XCTAssertThrowsError(
      try source.submitDetectorImages(
        mask: m4, maximumAdditionalBytes: budget, selectedAcquisitions: [33] + others)
    ) {
      XCTAssertTrue(
        String(describing: $0).contains("Injected detector submission failure"), "\($0)")
    }
    source.experimentalFailAfterSubmittedCommands = nil
    XCTAssertEqual(source.detectorQueriesInFlight, 0)
    XCTAssertEqual(
      source.experimentalDetectorSeedMask(acquisition: 33), m3,
      "the committed first group's seed is restored")
    XCTAssertEqual(source.experimentalDetectorSeedMask(acquisition: 0), m2)
    XCTAssertEqual(hashes(hoverImages), r3, "returned images are untouched")
    XCTAssertEqual(hashes(second), r2)
    let retried = try source.finishDetectorImages(
      try source.submitDetectorImages(
        mask: m4, maximumAdditionalBytes: budget, selectedAcquisitions: [33] + others))
    XCTAssertEqual(retried.commandTiming["seed_groups"], 2)
    XCTAssertEqual(hashes(retried.images), r4, "the retry from the restored seeds is exact")
    XCTAssertEqual(source.experimentalDetectorSeedMask(acquisition: 33), m4)

    // Release with a query in flight waits for it; finishing it reports that.
    let last = try source.submitDetectorImages(
      mask: m1, maximumAdditionalBytes: budget, selectedAcquisitions: [7])
    XCTAssertEqual(source.detectorQueriesInFlight, 1)
    source.release()
    XCTAssertEqual(source.detectorQueriesInFlight, 0)
    XCTAssertThrowsError(try source.finishDetectorImages(last)) {
      XCTAssertTrue(String(describing: $0).contains("released"), "\($0)")
    }
    print(
      "ENCODE_AHEAD_LIFECYCLE refused_sync=true cancelled_submit=true partial_group_failure_rolled_back=true retry_exact=true release_in_flight=true"
    )
  }
}
