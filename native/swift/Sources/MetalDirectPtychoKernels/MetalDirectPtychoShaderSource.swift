import Foundation

/// Load the same resources from a Swift package or a bundled native app.
enum MetalDirectPtychoShaderSource {
  /// This target's SwiftPM resource bundle (`<package>_<target>`), which a packaged app copies into
  /// `Contents/Resources`. Before the module was renamed it was `MetalKernels_MetalSSBKernels.bundle`.
  static let packagedBundleName = "MetalKernels_MetalDirectPtychoKernels.bundle"

  /// `packagedResources` is a packaged app's resource folder; the package's own bundle is the fallback.
  static func load(
    _ names: [String], packagedResources: URL? = Bundle.main.resourceURL
  ) throws -> String {
    try names.map { name in
      let packaged = packagedResources?.appendingPathComponent(
        "\(packagedBundleName)/Resources/\(name).metal")
      let url =
        packaged.flatMap {
          FileManager.default.fileExists(atPath: $0.path) ? $0 : nil
        } ?? Bundle.module.url(forResource: name, withExtension: "metal", subdirectory: "Resources")
        ?? Bundle.module.url(forResource: name, withExtension: "metal")
      guard let url else { throw MetalDirectPtychoError.missingResource("\(name).metal") }
      return try String(contentsOf: url, encoding: .utf8)
    }.joined(separator: "\n")
  }
}
