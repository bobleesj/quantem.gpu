import Foundation

/// The native client and Python Metal backend compile the same codec sources.
public enum MetalCountResources {
  public static func source(_ name: String) throws -> String {
    guard
      let url = Bundle.module.url(forResource: name, withExtension: "msl", subdirectory: "kernels")
    else {
      throw CocoaError(.fileNoSuchFile)
    }
    return try String(contentsOf: url, encoding: .utf8)
  }
}
