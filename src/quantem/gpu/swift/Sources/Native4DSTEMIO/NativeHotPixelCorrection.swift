import Foundation

/// Correct source-marked detector pixels without changing the source file.
public enum NativeHotPixelCorrection {
  /// Replace each marked pixel with the integer median of its valid 3x3 neighbors.
  ///
  /// Other marked pixels are excluded from the neighborhood. At detector edges,
  /// only in-bounds neighbors contribute. An even neighborhood uses the floored
  /// mean of its two central values, matching the CUDA and Metal load policy.
  public static func median3x3(
    values: [UInt32],
    detectorRows: Int,
    detectorColumns: Int,
    markedPixels: [Int]
  ) -> [UInt32] {
    guard detectorRows > 0, detectorColumns > 0,
      detectorRows.multipliedReportingOverflow(by: detectorColumns).overflow == false,
      detectorRows * detectorColumns == values.count,
      !markedPixels.isEmpty
    else { return values }

    let marked = Set(markedPixels.filter { values.indices.contains($0) })
    guard !marked.isEmpty else { return values }
    var corrected = values
    for pixel in marked {
      let row = pixel / detectorColumns
      let column = pixel % detectorColumns
      var neighbors: [UInt32] = []
      neighbors.reserveCapacity(8)
      for rowOffset in -1...1 {
        for columnOffset in -1...1 where rowOffset != 0 || columnOffset != 0 {
          let neighborRow = row + rowOffset
          let neighborColumn = column + columnOffset
          guard 0..<detectorRows ~= neighborRow,
            0..<detectorColumns ~= neighborColumn
          else { continue }
          let neighbor = neighborRow * detectorColumns + neighborColumn
          if !marked.contains(neighbor) { neighbors.append(values[neighbor]) }
        }
      }
      neighbors.sort()
      guard !neighbors.isEmpty else {
        corrected[pixel] = 0
        continue
      }
      if neighbors.count.isMultiple(of: 2) {
        let upper = neighbors.count / 2
        corrected[pixel] = UInt32(
          (UInt64(neighbors[upper - 1]) + UInt64(neighbors[upper])) / 2)
      } else {
        corrected[pixel] = neighbors[neighbors.count / 2]
      }
    }
    return corrected
  }
}
