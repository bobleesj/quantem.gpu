import Foundation

/// The native client and Python Metal backend compile the same codec sources.
public enum MetalCountResources {
  public static func source(_ name: String) throws -> String {
    let resources: Bundle
    if Bundle.main.bundleURL.pathExtension == "app" {
      guard let url = Bundle.main.resourceURL?
        .appendingPathComponent("MetalKernels_MetalCountResources.bundle"),
        let packaged = Bundle(url: url)
      else { throw CocoaError(.fileNoSuchFile) }
      resources = packaged
    } else {
      resources = .module
    }
    guard
      let url = resources.url(forResource: name, withExtension: "msl", subdirectory: "kernels")
    else {
      throw CocoaError(.fileNoSuchFile)
    }
    return try String(contentsOf: url, encoding: .utf8)
  }
}
