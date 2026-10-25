import Foundation
import XCTest

@testable import MetalSSBKernels

/// A shipped kernel's source must match the digest its version was validated with, and
/// `docs/performance/kernels.md` must list that version and digest. Changing a kernel therefore
/// means a new version, a registry row and fresh measurements.
final class SSBKernelRegistryTests: XCTestCase {
  private let entries = [
    SSBKernelRegistry.redraw, SSBKernelRegistry.perTerm,
    SSBKernelRegistry.fifthAgreement, SSBKernelRegistry.jointAgreement,
    SSBKernelRegistry.lowOrderDerivatives, SSBKernelRegistry.fifthGradient, SSBKernelRegistry.direct,
  ]

  func testShippedSourceMatchesRegisteredVersion() throws {
    for entry in entries {
      let shipped = try SSBKernelRegistry.sourceSHA256(entry)
      XCTAssertEqual(shipped, entry.registeredSourceSHA256,
        "\(entry.name) source changed: register a new version (\(shipped)) in SSBKernelRegistry and docs/performance/kernels.md")
    }
  }

  func testRegistryDocumentListsEveryVersion() throws {
    let document = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
      .appendingPathComponent("docs/performance/kernels.md")
    let text = try String(contentsOf: document, encoding: .utf8)
    for entry in entries {
      XCTAssertTrue(text.contains("`\(entry.name)` v\(entry.version)"), "kernels.md lacks \(entry.name) v\(entry.version)")
      XCTAssertTrue(text.contains(String(entry.registeredSourceSHA256.prefix(12))),
        "kernels.md lacks the \(entry.name) v\(entry.version) source digest")
    }
  }

  func testIdentityFingerprintCoversSourceAndVariant() {
    let a = MetalKernelIdentity(name: "k", version: "0.2", sourceSHA256: "abc", variant: "factor=3 degree=6 depth=on")
    let b = MetalKernelIdentity(name: "k", version: "0.2", sourceSHA256: "abc", variant: "factor=3 degree=2 depth=on")
    let c = MetalKernelIdentity(name: "k", version: "0.2", sourceSHA256: "abd", variant: "factor=3 degree=6 depth=on")
    XCTAssertNotEqual(a.fingerprint, b.fingerprint)
    XCTAssertNotEqual(a.fingerprint, c.fingerprint)
    XCTAssertEqual(a.fingerprint.count, 64)
  }
}
