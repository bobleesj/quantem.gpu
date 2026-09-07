import Darwin
import Foundation
import XCTest

@testable import Native4DSTEMIO

final class FileIdentityTests: XCTestCase {
  func testRestoredModificationTimeDoesNotReuseChangedSourceHashes() throws {
    let files = try fixture()
    let first = try nativeSourceHashes(
      master: nil, dataFiles: [files.source], cacheFile: files.cache)
    let before = try nativeFileIdentity(for: files.source)
    let firstSignature = try nativeDatasetSignature(for: [files.source])
    var status = stat()
    XCTAssertEqual(files.source.path.withCString { Darwin.lstat($0, &status) }, 0)

    let handle = try FileHandle(forWritingTo: files.source)
    defer { try? handle.close() }
    try handle.write(contentsOf: Data("omega".utf8))
    try handle.synchronize()
    var times = [status.st_atimespec, status.st_mtimespec]
    XCTAssertEqual(Darwin.futimens(handle.fileDescriptor, &times), 0)

    let changed = try nativeFileIdentity(for: files.source)
    XCTAssertEqual(changed.inode, before.inode)
    XCTAssertEqual(changed.bytes, before.bytes)
    XCTAssertEqual(changed.modificationNanoseconds, before.modificationNanoseconds)
    XCTAssertNotEqual(changed.changeNanoseconds, before.changeNanoseconds)
    XCTAssertNotEqual(try nativeDatasetSignature(for: [files.source]), firstSignature)

    let refreshed = try nativeSourceHashes(
      master: nil, dataFiles: [files.source], cacheFile: files.cache)
    let independentlyHashed = try nativeSourceHashes(master: nil, dataFiles: [files.source])
    XCTAssertNotEqual(refreshed.aggregate, first.aggregate)
    XCTAssertEqual(refreshed.aggregate, independentlyHashed.aggregate)
    XCTAssertEqual(refreshed.members, independentlyHashed.members)
    try assertUnchangedOpenUsesCache(
      source: files.source, cache: files.cache, expected: refreshed.aggregate)
  }

  func testStatusChangeRefreshesSnapshotWithoutChangingContentIdentity() throws {
    let files = try fixture()
    let first = try nativeSourceHashes(
      master: nil, dataFiles: [files.source], cacheFile: files.cache)
    let before = try nativeFileIdentity(for: files.source)
    let cacheBefore = try nativeFileIdentity(for: files.cache)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o400], ofItemAtPath: files.source.path)

    let changed = try nativeFileIdentity(for: files.source)
    XCTAssertEqual(changed.modificationNanoseconds, before.modificationNanoseconds)
    XCTAssertNotEqual(changed.changeNanoseconds, before.changeNanoseconds)
    let refreshed = try nativeSourceHashes(
      master: nil, dataFiles: [files.source], cacheFile: files.cache)
    XCTAssertEqual(refreshed.aggregate, first.aggregate)
    XCTAssertEqual(refreshed.members, first.members)
    XCTAssertNotEqual(try nativeFileIdentity(for: files.cache).inode, cacheBefore.inode)
    try assertUnchangedOpenUsesCache(
      source: files.source, cache: files.cache, expected: refreshed.aggregate)
  }

  func testLegacySnapshotWithoutChangeTimeIsRefreshedOnce() throws {
    let files = try fixture()
    let expected = try nativeSourceHashes(
      master: nil, dataFiles: [files.source], cacheFile: files.cache)
    var object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(contentsOf: files.cache)) as? [String: Any]
    )
    var snapshots = try XCTUnwrap(object["snapshots"] as? [[String: Any]])
    for index in snapshots.indices { snapshots[index].removeValue(forKey: "changeNanoseconds") }
    object["snapshots"] = snapshots
    object["memberHashes"] = [String(repeating: "0", count: 64)]
    object["aggregateHash"] = String(repeating: "0", count: 64)
    try JSONSerialization.data(withJSONObject: object).write(to: files.cache, options: .atomic)

    let refreshed = try nativeSourceHashes(
      master: nil, dataFiles: [files.source], cacheFile: files.cache)
    XCTAssertEqual(refreshed.aggregate, expected.aggregate)
    XCTAssertEqual(refreshed.members, expected.members)
    let current = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(contentsOf: files.cache)) as? [String: Any]
    )
    let currentSnapshots = try XCTUnwrap(current["snapshots"] as? [[String: Any]])
    XCTAssertNotNil(currentSnapshots.first?["changeNanoseconds"])
    try assertUnchangedOpenUsesCache(
      source: files.source, cache: files.cache, expected: refreshed.aggregate)
  }

  private func fixture() throws -> (source: URL, cache: URL) {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("NativeFileIdentityTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("counts.h5")
    try Data("alpha".utf8).write(to: source)
    return (source, root.appendingPathComponent("source-hashes.json"))
  }

  private func assertUnchangedOpenUsesCache(source: URL, cache: URL, expected: String) throws {
    let before = try nativeFileIdentity(for: cache)
    let data = try Data(contentsOf: cache)
    let hashes = try nativeSourceHashes(master: nil, dataFiles: [source], cacheFile: cache)
    let after = try nativeFileIdentity(for: cache)
    XCTAssertEqual(hashes.aggregate, expected)
    // A cache miss always rewrites this file atomically. Stable identity and
    // bytes therefore verify the unchanged reopen takes the cache-hit branch.
    XCTAssertEqual(after.inode, before.inode)
    XCTAssertEqual(after.modificationNanoseconds, before.modificationNanoseconds)
    XCTAssertEqual(after.changeNanoseconds, before.changeNanoseconds)
    XCTAssertEqual(try Data(contentsOf: cache), data)
  }
}
