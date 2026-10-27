import Foundation

/// Load the same resources from a Swift package or a bundled native app.
enum MetalSSBShaderSource {
  static func load(_ names: [String]) throws -> String {
    try names.map { name in
      let packaged = Bundle.main.resourceURL?.appendingPathComponent(
        "MetalKernels_MetalSSBKernels.bundle/Resources/\(name).metal")
      let url =
        packaged.flatMap {
          FileManager.default.fileExists(atPath: $0.path) ? $0 : nil
        } ?? Bundle.module.url(forResource: name, withExtension: "metal", subdirectory: "Resources")
        ?? Bundle.module.url(forResource: name, withExtension: "metal")
      guard let url else { throw MetalSSBError.missingResource("\(name).metal") }
      return try String(contentsOf: url, encoding: .utf8)
    }.joined(separator: "\n")
  }
}
