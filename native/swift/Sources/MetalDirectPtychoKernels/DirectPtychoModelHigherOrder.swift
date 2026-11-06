import Foundation

/// A polar higher-order coefficient retained in Double through fitting and serialization.
/// Magnitudes are Angstrom and azimuths radians; order and symmetry use the Krivanek convention.
public struct DirectPtychoModelHigherOrder: Codable, Equatable, Sendable {
  public let order: Int
  public let symmetry: Int
  public var magnitudeAngstroms: Double
  public var angleRadians: Double

  public init(
    order: Int, symmetry: Int, magnitudeAngstroms: Double = 0, angleRadians: Double = 0
  ) {
    self.order = order
    self.symmetry = symmetry
    self.magnitudeAngstroms = magnitudeAngstroms
    self.angleRadians = angleRadians
  }

  public init(_ term: MetalDirectPtychoHigherOrder) {
    self.init(
      order: term.order, symmetry: term.symmetry,
      magnitudeAngstroms: Double(term.magnitudeAngstroms), angleRadians: Double(term.angleRadians))
  }

  public var metal: MetalDirectPtychoHigherOrder {
    .init(
      order: order, symmetry: symmetry, magnitudeAngstroms: Float(magnitudeAngstroms),
      angleRadians: Float(angleRadians))
  }

  public static let supported = MetalDirectPtychoHigherOrder.supported.map(Self.init)

  static func validate(_ terms: [Self]) throws {
    var seen: Set<String> = []
    for term in terms {
      let name = "C\(term.order)\(term.symmetry)"
      guard supported.contains(where: { $0.order == term.order && $0.symmetry == term.symmetry }),
        term.magnitudeAngstroms.isFinite, term.angleRadians.isFinite,
        Float(term.magnitudeAngstroms).isFinite, Float(term.angleRadians).isFinite,
        seen.insert(name).inserted
      else {
        throw MetalDirectPtychoError.invalidGeometry(
          "Supply each supported order 2 through 5 once, with finite coefficients and azimuths.")
      }
    }
  }
}

extension DirectPtychoModelPoint {
  /// Cartesian coefficient pairs including 1/(n+1). Public reconstruction matches Python's
  /// pack_coefficients: magnitudes/azimuths round to Float before Double trig and final Float packing.
  /// An optimizer can pass its already encoded Cartesian rows directly to fifthOrderAgreements.
  var reconstructionCoefficients: [Float] {
    var terms = [
      (1, 0, c10Angstroms, 0.0), (1, 2, c12Angstroms, phi12Radians),
    ]
    for supported in DirectPtychoModelHigherOrder.supported {
      let term =
        higherOrder.first {
          $0.order == supported.order && $0.symmetry == supported.symmetry
        } ?? supported
      terms.append((term.order, term.symmetry, term.magnitudeAngstroms, term.angleRadians))
    }
    return terms.flatMap { order, symmetry, magnitude, angle in
      let scaled = Double(Float(magnitude)) / Double(order + 1)
      let phase = Double(symmetry) * Double(Float(angle))
      return [Float(scaled * cos(phase)), Float(scaled * sin(phase))]
    }
  }
}
