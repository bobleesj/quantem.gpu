import Foundation

/// A contiguous, little-endian 4D uint8/uint16 array in scan-row/column order.
/// Metadata readers implement this contract; the shared Metal loader owns encoding.
public protocol NativeCountArray {
  var url: URL { get }
  var dataset: Native4DSTEMDataset { get }
  var shape: [Int] { get }
  var dataOffset: Int { get }
  func assertUnchanged() throws
}

extension NativeDM4Source: NativeCountArray {}
