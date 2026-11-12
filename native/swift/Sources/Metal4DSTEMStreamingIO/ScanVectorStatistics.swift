import Foundation

/// Center a measured vector field over positions where both components are finite.
/// Invalid positions remain NaN in both outputs; they are never interpreted as zero.
public enum ScanVectorStatistics {
  @discardableResult
  public static func center(row: UnsafePointer<Float>, column: UnsafePointer<Float>,
    count: Int, intoRow: UnsafeMutablePointer<Float>, intoColumn: UnsafeMutablePointer<Float>) -> Int {
    var rowTotal = 0.0, columnTotal = 0.0
    var valid = 0
    for i in 0..<count where row[i].isFinite && column[i].isFinite {
      rowTotal += Double(row[i]); columnTotal += Double(column[i]); valid += 1
    }
    let r = Float(rowTotal / Double(max(1, valid))), c = Float(columnTotal / Double(max(1, valid)))
    for i in 0..<count {
      let measured = row[i].isFinite && column[i].isFinite
      intoRow[i] = measured ? row[i] - r : .nan
      intoColumn[i] = measured ? column[i] - c : .nan
    }
    return valid
  }
}
