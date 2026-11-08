import Foundation
import XCTest

@testable import MetalDirectPtychoKernels

/// A shipped kernel's source must match the digest its version was validated with, and
/// `docs/performance/kernels.md` must list that version and digest. Changing a kernel therefore
/// means a new version, a registry row and fresh measurements.
final class SSBKernelRegistryTests: XCTestCase {
  private let entries = [
    DirectPtychoKernelRegistry.redraw, DirectPtychoKernelRegistry.perTerm,
    DirectPtychoKernelRegistry.fifthAgreement, DirectPtychoKernelRegistry.jointAgreement,
    DirectPtychoKernelRegistry.lowOrderDerivatives, DirectPtychoKernelRegistry.fifthGradient,
    DirectPtychoKernelRegistry.direct,
  ]

  func testShippedSourceMatchesRegisteredVersion() throws {
    for entry in entries {
      let shipped = try DirectPtychoKernelRegistry.sourceSHA256(entry)
      XCTAssertEqual(
        shipped, entry.registeredSourceSHA256,
        "\(entry.name) source changed: register a new version (\(shipped)) in DirectPtychoKernelRegistry and docs/performance/kernels.md"
      )
    }
  }

  /// A packaged app copies this target's SwiftPM bundle into `Contents/Resources`, where the loader looks it up by
  /// name: that name must be the bundle SwiftPM builds, and the packaged copy must give the registered digests.
  func testPackagedAppLayoutLoadsTheBuiltBundle() throws {
    let built = Bundle.module.bundleURL
    XCTAssertEqual(built.lastPathComponent, MetalDirectPtychoShaderSource.packagedBundleName)
    let files = FileManager.default
    let resources = files.temporaryDirectory.appendingPathComponent(
      "direct-ptycho-packaged-\(UUID().uuidString)", isDirectory: true)
    try files.createDirectory(at: resources, withIntermediateDirectories: true)
    defer { try? files.removeItem(at: resources) }
    let packaged = resources.appendingPathComponent(
      MetalDirectPtychoShaderSource.packagedBundleName)
    try files.copyItem(at: built, to: packaged)
    for entry in entries {
      let source = try MetalDirectPtychoShaderSource.load(entry.files, packagedResources: resources)
      XCTAssertEqual(
        MetalKernelIdentity.sha256(Data(source.utf8)), entry.registeredSourceSHA256, entry.name)
    }
    // The packaged copy is read before the package's own bundle.
    let marker = "// packaged copy\n"
    try Data(marker.utf8).write(to: packaged.appendingPathComponent("Resources/ssb_redraw.metal"))
    XCTAssertEqual(
      try MetalDirectPtychoShaderSource.load(["ssb_redraw"], packagedResources: resources), marker)
  }

  func testRegistryDocumentListsEveryVersion() throws {
    let document = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
      .appendingPathComponent("docs/performance/kernels.md")
    let text = try String(contentsOf: document, encoding: .utf8)
    for entry in entries {
      XCTAssertTrue(
        text.contains("`\(entry.name)` v\(entry.version)"),
        "kernels.md lacks \(entry.name) v\(entry.version)")
      XCTAssertTrue(
        text.contains(String(entry.registeredSourceSHA256.prefix(12))),
        "kernels.md lacks the \(entry.name) v\(entry.version) source digest")
    }
  }

  func testIdentityFingerprintCoversSourceAndVariant() {
    let a = MetalKernelIdentity(
      name: "k", version: "0.2", sourceSHA256: "abc", variant: "factor=3 degree=6 depth=on")
    let b = MetalKernelIdentity(
      name: "k", version: "0.2", sourceSHA256: "abc", variant: "factor=3 degree=2 depth=on")
    let c = MetalKernelIdentity(
      name: "k", version: "0.2", sourceSHA256: "abd", variant: "factor=3 degree=6 depth=on")
    XCTAssertNotEqual(a.fingerprint, b.fingerprint)
    XCTAssertNotEqual(a.fingerprint, c.fingerprint)
    XCTAssertEqual(a.fingerprint.count, 64)
  }
}
