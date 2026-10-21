import Foundation

/// The scan-detector rotation from the centre-of-mass curl and the 180-degree branch test, ported from Python
/// `quantem.gpu.dpc.workflow` (`center_of_mass`, `find_optimal_rotation`) and `quantem.gpu.ssb.branches`.
///
/// The curl fixes the rotation modulo 180 degrees. SSB fits both branches independently; on the right branch
/// the phase at the atomic columns, located on the incoherent dark-field image, is positive. When that test
/// is not decisive the branch with the larger mean-centred phase skewness is kept.
public enum SSBRotationBranches {
  // Python's `branches` constants.
  public static let darkFieldInnerRadius = 1.1
  static let columnSmoothingPixels = 1.0
  static let columnHighPassAngstroms = 2.1
  static let columnSeparationAngstroms = 2.0
  static let columnTopFraction = 0.2
  public static let columnZMinimum = 3.0
  public static let columnEffectMinimum = 0.25

  // MARK: Rotation from the centre-of-mass curl

  /// Count-weighted detector centre of each scan position from exact integer moments, divided in double
  /// precision and stored as float32, 0 where a pattern holds no counts (Python `DetectorSession.center_of_mass`).
  public static func centerOfMass(total: [UInt64], rowMoment: [UInt64], columnMoment: [UInt64])
    -> (row: [Float], column: [Float])
  {
    let row = total.indices.map { total[$0] == 0 ? 0 : Float(Double(rowMoment[$0]) / Double(total[$0])) }
    let column = total.indices.map {
      total[$0] == 0 ? 0 : Float(Double(columnMoment[$0]) / Double(total[$0]))
    }
    return (row, column)
  }

  /// The rotation in degrees, in [0, 180), that minimizes the mean squared curl of the centre-of-mass field
  /// (`find_optimal_rotation` with the axes as given). The mean subtraction of `dpc.center_of_mass` leaves
  /// every central difference unchanged, so the field is used as it is.
  public static func curlRotationDegrees(row: [Float], column: [Float], rows: Int, columns: Int) -> Double {
    precondition(rows >= 3 && columns >= 3 && row.count == rows * columns && column.count == row.count)
    var curlPower = 0.0
    var divergencePower = 0.0
    var crossPower = 0.0
    for r in 1..<(rows - 1) {
      for c in 1..<(columns - 1) {
        let i = r * columns + c
        let curl = Double(
          Float(0.5) * (column[i + columns] - column[i - columns]) - Float(0.5) * (row[i + 1] - row[i - 1]))
        let divergence = Double(
          Float(0.5) * (row[i + columns] - row[i - columns]) + Float(0.5) * (column[i + 1] - column[i - 1]))
        curlPower += curl * curl
        divergencePower += divergence * divergence
        crossPower += curl * divergence
      }
    }
    let count = Double((rows - 2) * (columns - 2))
    curlPower /= count
    divergencePower /= count
    crossPower /= count
    let halfDifference = 0.5 * (curlPower - divergencePower)
    let amplitude = hypot(halfDifference, crossPower)
    if amplitude == 0 { return 0 }
    var angle = (0.5 * (atan2(crossPower, halfDifference) + .pi))
      .truncatingRemainder(dividingBy: .pi)
    if angle < 0 { angle += .pi }
    let degrees = (angle * (180 / Double.pi)).truncatingRemainder(dividingBy: 180)
    return degrees == 180 ? 0 : degrees
  }

  /// The other 180-degree branch in [0, 360): an angle at or above 180 has 180 subtracted, exactly
  /// (`geometry.frames.opposite_rotation_deg`).
  public static func opposite(_ degrees: Double) -> Double {
    var angle = degrees.truncatingRemainder(dividingBy: 360)
    if angle < 0 { angle += 360 }
    if angle >= 180 { return angle - 180 }
    let other = angle + 180
    return other >= 360 ? 0 : other
  }

  /// `optics.physics.wrap_degrees`: an angle in [0, period).
  public static func wrap(_ degrees: Double, period: Double = 360) -> Double {
    var wrapped = degrees.truncatingRemainder(dividingBy: period)
    if wrapped < 0 { wrapped += period }
    return wrapped == period ? 0 : wrapped
  }

  // MARK: SciPy ndimage, mode "reflect"

  static func reflect(_ index: Int, _ count: Int) -> Int {
    if count == 1 { return 0 }
    var i = index
    let period = 2 * count
    i %= period
    if i < 0 { i += period }
    return i < count ? i : period - 1 - i
  }

  /// `scipy.ndimage.gaussian_filter` (truncate 4.0) of a row-major image, axis 0 then axis 1.
  static func gaussianFilter(_ image: [Double], rows: Int, columns: Int, sigma: (Double, Double)) -> [Double] {
    func kernel(_ sd: Double) -> [Double] {
      let radius = Int(4.0 * sd + 0.5)
      let sigma2 = sd * sd
      var phi = (-radius...radius).map { exp(-0.5 / sigma2 * Double($0 * $0)) }
      let total = phi.reduce(0, +)
      phi = phi.map { $0 / total }
      return phi
    }
    var output = image
    for axis in 0..<2 {
      let weights = kernel(axis == 0 ? sigma.0 : sigma.1)
      let radius = weights.count / 2
      let input = output
      let length = axis == 0 ? rows : columns
      for r in 0..<rows {
        for c in 0..<columns {
          let position = axis == 0 ? r : c
          // The symmetric kernel: centre term, then the mirrored pairs from the outermost in (SciPy's
          // NI_Correlate1D).
          var sum = weights[radius] * input[r * columns + c]
          for j in stride(from: radius, through: 1, by: -1) {
            let plus = reflect(position + j, length)
            let minus = reflect(position - j, length)
            let a = axis == 0 ? input[plus * columns + c] : input[r * columns + plus]
            let b = axis == 0 ? input[minus * columns + c] : input[r * columns + minus]
            sum += (b + a) * weights[radius - j]
          }
          output[r * columns + c] = sum
        }
      }
    }
    return output
  }

  /// `scipy.ndimage.maximum_filter` / `minimum_filter` with a rectangular `size`, origin 0.
  static func rankFilter(
    _ image: [Double], rows: Int, columns: Int, size: (Int, Int), maximum: Bool
  ) -> [Double] {
    var output = image
    for axis in 0..<2 {
      let width = axis == 0 ? size.0 : size.1
      let input = output
      let length = axis == 0 ? rows : columns
      let before = width / 2
      for r in 0..<rows {
        for c in 0..<columns {
          let position = axis == 0 ? r : c
          var value = maximum ? -Double.infinity : Double.infinity
          for j in (position - before)..<(position - before + width) {
            let k = reflect(j, length)
            let v = axis == 0 ? input[k * columns + c] : input[r * columns + k]
            value = maximum ? max(value, v) : min(value, v)
          }
          output[r * columns + c] = value
        }
      }
    }
    return output
  }

  /// NumPy's default (linear) quantile.
  static func quantile(_ values: [Double], _ q: Double) -> Double {
    let sorted = values.sorted()
    let index = q * Double(sorted.count - 1)
    let lower = Int(index.rounded(.down))
    let upper = min(lower + 1, sorted.count - 1)
    let t = index - Double(lower)
    let a = sorted[lower]
    let b = sorted[upper]
    let difference = b - a
    return t >= 0.5 ? b - difference * (1 - t) : a + difference * t
  }

  /// The high-pass sigma and the peak window in scan pixels for one scan step (`column_filter_px`).
  static func columnFilter(scanSampling: (Double, Double)) -> (sigma: (Double, Double), window: (Int, Int)) {
    func window(_ step: Double) -> Int {
      // Python's round: half to even.
      2 * max(1, Int((columnSeparationAngstroms / step).rounded(.toNearestOrEven))) + 1
    }
    return (
      (columnHighPassAngstroms / scanSampling.0, columnHighPassAngstroms / scanSampling.1),
      (window(scanSampling.0), window(scanSampling.1))
    )
  }

  /// Atomic columns on a dark-field image, as (row, column) scan pixels (`locate_atomic_columns`): a
  /// 1-pixel gaussian, a 2.1 A high pass, local maxima 2 A apart, the top 20% by prominence.
  public static func locateAtomicColumns(
    darkField: [Float], rows: Int, columns: Int, scanSampling: (Double, Double)
  ) -> [(row: Int, column: Int)] {
    let image = darkField.map(Double.init)
    let (sigma, window) = columnFilter(scanSampling: scanSampling)
    let smoothed = gaussianFilter(
      image, rows: rows, columns: columns, sigma: (columnSmoothingPixels, columnSmoothingPixels))
    let background = gaussianFilter(smoothed, rows: rows, columns: columns, sigma: sigma)
    let filtered = zip(smoothed, background).map { $0 - $1 }
    let minimum = rankFilter(filtered, rows: rows, columns: columns, size: window, maximum: false)
    let maximum = rankFilter(filtered, rows: rows, columns: columns, size: window, maximum: true)
    var peaks: [(Int, Int)] = []
    var heights: [Double] = []
    for i in filtered.indices {
      let prominence = filtered[i] - minimum[i]
      if filtered[i] == maximum[i] && prominence > 0 {
        peaks.append((i / columns, i % columns))
        heights.append(prominence)
      }
    }
    if peaks.isEmpty { return [] }
    let threshold = quantile(heights, 1 - columnTopFraction)
    return zip(peaks, heights).filter { $0.1 >= threshold }.map { (row: $0.0.0, column: $0.0.1) }
  }

  public struct ColumnTest: Codable, Sendable {
    public let z: Double?
    public let effect: Double?
    public let positiveFraction: Double?
    public let count: Int
  }

  /// The high-passed, mean-centred phase at the columns: z = mean / (std / sqrt(n)) (`column_phase_test`).
  public static func columnPhaseTest(
    phase: [Float], rows: Int, columns: Int, at positions: [(row: Int, column: Int)],
    scanSampling: (Double, Double)
  ) -> ColumnTest {
    let values = phase.map(Double.init)
    let mean = values.reduce(0, +) / Double(values.count)
    let (sigma, _) = columnFilter(scanSampling: scanSampling)
    let centred = values.map { $0 - mean }
    let background = gaussianFilter(centred, rows: rows, columns: columns, sigma: sigma)
    let sampled = positions.map { centred[$0.row * columns + $0.column] - background[$0.row * columns + $0.column] }
    guard sampled.count >= 3 else { return ColumnTest(z: nil, effect: nil, positiveFraction: nil, count: sampled.count) }
    let sampleMean = sampled.reduce(0, +) / Double(sampled.count)
    let spread = sqrt(sampled.map { ($0 - sampleMean) * ($0 - sampleMean) }.reduce(0, +) / Double(sampled.count))
    let positive = Double(sampled.filter { $0 > 0 }.count) / Double(sampled.count)
    guard spread > 0 else { return ColumnTest(z: nil, effect: nil, positiveFraction: positive, count: sampled.count) }
    let z = sampleMean / (spread / sqrt(Double(sampled.count)))
    return ColumnTest(z: z, effect: z / sqrt(Double(sampled.count)), positiveFraction: positive, count: sampled.count)
  }

  /// `mean(p^3) / mean(p^2)^1.5` with `p = phase - mean(phase)` (`results.phase_skewness`).
  public static func phaseSkewness(_ phase: [Float]) -> Double {
    let values = phase.map(Double.init)
    let mean = values.reduce(0, +) / Double(values.count)
    var second = 0.0
    var third = 0.0
    for value in values {
      let p = value - mean
      second += p * p
      third += p * p * p
    }
    second /= Double(values.count)
    third /= Double(values.count)
    return second <= 0 ? 0 : third / pow(second, 1.5)
  }

  public struct Branch: Codable, Sendable {
    public let scanDetectorRotationDegrees: Double
    public let phaseSkewness: Double
    public let column: ColumnTest?
  }

  public enum Decision: String, Codable, Sendable {
    case columns
    case phaseTail = "phase_tail"
  }

  /// Score both branches' native-sampling phases and choose one (`compare_branches`); `keepFirst` keeps
  /// the first (a given rotation) and still reports what the evidence chose.
  public static func compare(
    phases: [[Float]], rotations: [Double], rows: Int, columns: Int, darkField: [Float]?,
    scanSampling: (Double, Double), keepFirst: Bool = false
  ) -> (branches: [Branch], chosen: Int, kept: Int, decidedBy: Decision, columns: Int) {
    precondition(phases.count == 2 && rotations.count == 2)
    let positions =
      darkField.map {
        $0.count == rows * columns
          ? locateAtomicColumns(darkField: $0, rows: rows, columns: columns, scanSampling: scanSampling)
          : []
      }
    let branches = zip(phases, rotations).map { phase, rotation in
      Branch(
        scanDetectorRotationDegrees: rotation, phaseSkewness: phaseSkewness(phase),
        column: positions.map {
          columnPhaseTest(phase: phase, rows: rows, columns: columns, at: $0, scanSampling: scanSampling)
        })
    }
    let decisive = branches.contains { branch in
      guard let z = branch.column?.z, let effect = branch.column?.effect else { return false }
      return abs(z) >= columnZMinimum && abs(effect) >= columnEffectMinimum
    }
    let chosen: Int
    let decision: Decision
    if decisive {
      decision = .columns
      let scores = branches.map { $0.column?.z ?? -.infinity }
      chosen = scores[1] > scores[0] ? 1 : 0
    } else {
      decision = .phaseTail
      chosen = branches[0].phaseSkewness >= branches[1].phaseSkewness ? 0 : 1
    }
    return (branches, chosen, keepFirst ? 0 : chosen, decision, positions?.count ?? 0)
  }
}
