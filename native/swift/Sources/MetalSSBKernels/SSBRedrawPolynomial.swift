import Foundation

/// chi(row, col) as a bivariate polynomial in inverse-Angstrom frequency, `sum c[i][j] row^i col^j`, the form the
/// interactive redraw kernel (`ssb_redraw.metal`) evaluates. Built in double precision from the same float32
/// inputs the per-term kernels read; returned row-major 7 x 7 in float32.
enum SSBRedrawPolynomial {
  /// C10/C12 only (`model_probe`): factor * alpha^2 * (C12 cos 2(phi - phi12) + C10), alpha = |p| * wavelength.
  static func primary(_ params: [Float]) -> [Float] {
    let wl = Double(params[0]), c10 = Double(params[4]), c12 = Double(params[5])
    let c2 = Double(params[6]), s2 = Double(params[7]), factor = Double(params[8])
    var c = [Double](repeating: 0, count: 49)
    let scale = factor * wl * wl
    c[2 * 7 + 0] = scale * (c10 + c12 * c2)
    c[0 * 7 + 2] = scale * (c10 - c12 * c2)
    c[1 * 7 + 1] = scale * 2 * c12 * s2
    return c.map { Float($0) }
  }

  /// The 14-term Cartesian chi of `ssb_chi_full` (packed coefficients include 1/(n+1)):
  /// (2 pi / wl) * sum radial^p * (Re z^m a + Im z^m b), z = (row + i col) * wl.
  static func full(coefficients: [Float], wavelength: Float) -> [Float] {
    fullDouble(coefficients: coefficients.map(Double.init), wavelength: Double(wavelength)).map {
      Float($0)
    }
  }

  /// `full` in double precision (row-major 7 x 7).
  static func fullDouble(coefficients: [Double], wavelength: Double) -> [Double] {
    precondition(coefficients.count == 28)
    let azimuth = [0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6]
    let power = [1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0]
    typealias Poly = [Double]  // 7 x 7, index i * 7 + j for x^i y^j
    func multiply(_ a: Poly, _ b: Poly) -> Poly {
      var out = Poly(repeating: 0, count: 49)
      for i in 0..<7 {
        for j in 0..<7 where a[i * 7 + j] != 0 {
          for u in 0..<(7 - i) {
            for v in 0..<(7 - j) where b[u * 7 + v] != 0 && i + u < 7 && j + v < 7 {
              out[(i + u) * 7 + j + v] += a[i * 7 + j] * b[u * 7 + v]
            }
          }
        }
      }
      return out
    }
    var one = Poly(repeating: 0, count: 49)
    one[0] = 1
    var r2 = Poly(repeating: 0, count: 49)
    r2[2 * 7] = 1
    r2[2] = 1
    // z^m = re + i im with z = x + i y.
    var re = [one], im = [Poly(repeating: 0, count: 49)]
    var x = Poly(repeating: 0, count: 49), y = Poly(repeating: 0, count: 49)
    x[7] = 1
    y[1] = 1
    for m in 1...6 {
      let r = zip(multiply(re[m - 1], x), multiply(im[m - 1], y)).map { $0 - $1 }
      let i = zip(multiply(im[m - 1], x), multiply(re[m - 1], y)).map { $0 + $1 }
      re.append(r)
      im.append(i)
    }
    var radial = [one]
    for p in 1...3 { radial.append(multiply(radial[p - 1], r2)) }
    var chi = Poly(repeating: 0, count: 49)
    for term in 0..<14 {
      let a = coefficients[2 * term], b = coefficients[2 * term + 1]
      let angular = zip(re[azimuth[term]], im[azimuth[term]]).map { $0 * a + $1 * b }
      for (index, value) in multiply(radial[power[term]], angular).enumerated() {
        chi[index] += value
      }
    }
    let wl = wavelength
    return chi.indices.map { index in
      let degree = index / 7 + index % 7
      return chi[index] * (2 * Double.pi / wl) * pow(wl, Double(degree))
    }
  }
}
