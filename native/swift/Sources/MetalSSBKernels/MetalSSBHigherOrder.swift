import Foundation

/// One polar aberration C_nm. Magnitudes are Angstrom, as C10 and C12 are in the engine;
/// azimuths are radians. Orders 2...5 follow ShowPtycho's polar convention.
public struct MetalSSBHigherOrder: Codable, Equatable, Sendable {
  public let order: Int
  public let symmetry: Int
  public var magnitudeAngstroms: Float
  public var angleRadians: Float

  public init(order: Int, symmetry: Int, magnitudeAngstroms: Float = 0, angleRadians: Float = 0) {
    self.order = order
    self.symmetry = symmetry
    self.magnitudeAngstroms = magnitudeAngstroms
    self.angleRadians = angleRadians
  }

  public var name: String { "C\(order)\(symmetry)" }
  public static let supported: [Self] = [
    (2, 1), (2, 3), (3, 0), (3, 2), (3, 4),
    (4, 1), (4, 3), (4, 5), (5, 0), (5, 2), (5, 4), (5, 6),
  ].map {
    Self(order: $0.0, symmetry: $0.1)
  }
}
