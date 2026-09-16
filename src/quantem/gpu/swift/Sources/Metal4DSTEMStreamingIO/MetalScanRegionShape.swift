/// Scan-position membership for mean diffraction. Circles use square bounds
/// and include pixel centers on the circumference, without fractional weights.
public enum MetalScanRegionShape: UInt32, Codable, Sendable {
  case rectangle = 0
  case circle = 1

  /// Number of selected scan positions, not the area of the bounding box.
  /// Circle bounds must be square. The row-wise chord count is O(diameter).
  public func sampleCount(rowCount: Int, columnCount: Int) -> Int {
    precondition(rowCount > 0 && columnCount > 0)
    guard self == .circle else { return rowCount * columnCount }
    precondition(rowCount == columnCount)
    let diameter = rowCount
    return (0..<diameter).reduce(0) { total, row in
      let offset = 2 * row + 1 - diameter
      let chord = Int(Double(diameter * diameter - offset * offset).squareRoot())
      let first = max(0, (diameter - chord) / 2)
      let last = min(diameter - 1, (diameter - 1 + chord) / 2)
      return total + max(0, last - first + 1)
    }
  }
}
