import Foundation

/// Dimensionless Cartesian coordinates shared with the CUDA/MPS fifth-order fitter.
struct DirectPtychoFifthOrderCoordinates {
  let wavelengthAngstroms: Double
  let semiangleRadians: Double
  static let orders: [(Int, Int)] = [
    (1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2), (3, 4),
    (4, 1), (4, 3), (4, 5), (5, 0), (5, 2), (5, 4), (5, 6),
  ]

  func initial(_ point: DirectPtychoModelPoint) -> [Double] {
    [
      point.c10Angstroms / 100, point.c12Angstroms / 100 * cos(2 * point.phi12Radians),
      point.c12Angstroms / 100 * sin(2 * point.phi12Radians),
      point.sampleTiltMrad[0] / 5, point.sampleTiltMrad[1] / 5,
      point.depthSpreadAngstroms / 100,
    ] + Array(repeating: 0, count: 22)
  }

  /// These are C_nm/(n+1) Cartesian components in Angstrom, rounded once at the GPU boundary.
  func pack(_ z: [Double]) -> [Float] {
    var packed = Array(repeating: Float(0), count: 28)
    packed[0] = Float(z[0] * 50)
    packed[2] = Float(z[1] * 50)
    packed[3] = Float(z[2] * 50)
    var j = 6
    for (i, (n, m)) in Self.orders.enumerated() where n >= 2 {
      let scale = wavelengthAngstroms / (2 * Double.pi * pow(semiangleRadians, Double(n + 1)))
      packed[2 * i] = Float(z[j] * scale)
      j += 1
      if m != 0 {
        packed[2 * i + 1] = Float(z[j] * scale)
        j += 1
      }
    }
    return packed
  }

  func point(_ z: [Double]) -> DirectPtychoModelPoint {
    let packed = pack(z)
    let terms = Self.orders.enumerated().compactMap { i, order -> DirectPtychoModelHigherOrder? in
      let (n, m) = order
      guard n >= 2 else { return nil }
      // Match public_params: the Float Cartesian pack is converted to public polar coefficients.
      let magnitude = m == 0 ? packed[2 * i] : hypot(packed[2 * i], packed[2 * i + 1])
      let nanometers = magnitude * Float(n + 1) / 10
      // Match NumPy's float32 conversion of its Double pi, including rounding.
      let degrees =
        m == 0
        ? Float(0) : atan2(packed[2 * i + 1], packed[2 * i]) * (180 / Float(Double.pi)) / Float(m)
      return DirectPtychoModelHigherOrder(
        order: n, symmetry: m,
        magnitudeAngstroms: Double(nanometers) * 10, angleRadians: Double(degrees) * Double.pi / 180
      )
    }
    return DirectPtychoModelPoint(
      c10Angstroms: z[0] * 100,
      c12Angstroms: hypot(z[1], z[2]) * 100, phi12Radians: atan2(z[2], z[1]) / 2,
      sampleTiltMrad: [z[3] * 5, z[4] * 5], depthSpreadAngstroms: z[5] * 100,
      higherOrder: terms)
  }
}
