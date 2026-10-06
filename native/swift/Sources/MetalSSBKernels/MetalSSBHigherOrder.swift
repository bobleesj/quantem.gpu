import Foundation

/// One polar aberration C_nm. Magnitudes use the same units as C10/C12;
/// azimuths are radians. Orders 2...5 follow ShowPtycho's polar convention.
public struct MetalSSBHigherOrder: Codable, Equatable, Sendable {
  public let order: Int
  public let symmetry: Int
  public var magnitudeNanometers: Float
  public var angleRadians: Float

  public init(order: Int, symmetry: Int, magnitudeNanometers: Float = 0, angleRadians: Float = 0) {
    self.order = order
    self.symmetry = symmetry
    self.magnitudeNanometers = magnitudeNanometers
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
