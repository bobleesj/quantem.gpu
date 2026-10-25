import CryptoKit
import Foundation

/// Which GPU kernel produced a result: a registered name and version, the SHA-256 of the exact bundled Metal source
/// it was compiled from, the pipeline variant (function constants), and one fingerprint over both. The source digest
/// is computed from the shipped `.metal` text when the model is built, so it cannot drift from what ran.
/// `docs/performance/kernels.md` records each version's status and measurements.
public struct MetalKernelIdentity: Codable, Equatable, Hashable, Sendable {
  public let name: String
  public let version: String
  /// SHA-256 (hex) of the concatenated source files of this kernel.
  public let sourceSHA256: String
  /// The specialization, e.g. `factor=3 degree=6 depth=on`.
  public let variant: String
  /// SHA-256 (hex) of `sourceSHA256 + "\n" + variant`.
  public let fingerprint: String

  public init(name: String, version: String, sourceSHA256: String, variant: String) {
    self.name = name
    self.version = version
    self.sourceSHA256 = sourceSHA256
    self.variant = variant
    fingerprint = Self.sha256(Data("\(sourceSHA256)\n\(variant)".utf8))
  }

  /// "ssb-redraw v0.2 · 3f2a9c… (factor=3 degree=6 depth=on)" for status lines.
  public var label: String { "\(name) v\(version) · \(fingerprint.prefix(12)) (\(variant))" }

  static func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
}

/// The registered native SSB reconstruction and fit kernels; see `docs/performance/kernels.md`.
enum SSBKernelRegistry {
  struct Entry {
    let name: String
    let version: String
    let files: [String]
    /// The source digest this version was validated with. `SSBKernelRegistryTests` fails when the shipped source
    /// changes without a new version here and in the registry document.
    let registeredSourceSHA256: String
  }

  /// SSB (`phase_of_mean`) at 1x-4x: the only production path there.
  static let redraw = Entry(
    name: "ssb-redraw", version: "0.2", files: ["ssb_redraw"],
    registeredSourceSHA256: "b20d7914811d0fb2941d59907ba6e15607829f0988b08498f262d9942a3a5e34")
  /// The earlier per-term / separable coherent kernels: explicit 8x only, and the 1x-4x parity reference in tests.
  /// v0.2 is v0.1 without the retired `complex_wave` amplitude kernel (`model_wave_amplitude`); the per-term
  /// kernels are unchanged.
  static let perTerm = Entry(
    name: "ssb-per-term", version: "0.2", files: ["ssb_model", "ssb_fifth_order", "ssb_separable"],
    registeredSourceSHA256: "05177e0c2cde4620094e8f2a7aeaa816daadeaaaed1384a20bdd5103dc68114c")
  /// The fifth-order agreement objective of the fit (`fifth_agreement_rows` with the `fifth_probe`, `fifth_power`
  /// and `fifth_score` passes) on all BF pixels or every n-th: the only production objective kernel there, also the
  /// value pass of `fifthGradient`. v0.3 added the BF stride to v0.2 (same per-pair terms). v0.1 was
  /// `fifth_agreement` in `ssb_fifth_order` (ssb-per-term), now the parity reference of the tests.
  static let fifthAgreement = Entry(
    name: "ssb-fifth-agreement", version: "0.3", files: ["ssb_fifth_order", "ssb_fifth_agreement"],
    registeredSourceSHA256: "2d385e96d9c44dfaba4b5e67c1180bbc3830009d529b1aebac68e969c299c152")
  /// The low-order agreement objective with angle-sum phases (`joint_agreement_angle_sums`), used by the primary
  /// fit's search trials: the only production `.angleSums` kernel. v0.1 was `joint_objective_angle_sums` in
  /// `ssb_joint`, now the parity reference of the tests. v0.3 is v0.2 with the retired `mean_phase` kernels removed
  /// from `ssb_joint`; the agreement kernels are unchanged.
  static let jointAgreement = Entry(
    name: "ssb-joint-agreement", version: "0.3", files: ["ssb_joint", "ssb_joint_agreement"],
    registeredSourceSHA256: "6283577c146ab5a92b5643b5b696d10c3985585a590ff7ed67569d46d7035aa1")

  /// The primary fit's exact low-order value, gradient and Hessian (trust-region Newton), `SSBLowOrderDerivatives`:
  /// the value pass, the derivative pass and the fused pass of `ssb_low_order_derivatives` (with the aperture and
  /// quotient helpers of `ssb_joint_agreement`). v0.1 is `lowOrderDerivativesReference`. v0.3 is v0.2 with the
  /// retired `mean_phase` kernels removed from `ssb_joint`; the derivative kernels are unchanged.
  static let lowOrderDerivatives = Entry(
    name: "ssb-low-order-derivatives", version: "0.3",
    files: ["ssb_joint", "ssb_joint_agreement", "ssb_low_order_derivatives"],
    registeredSourceSHA256: "0158114dc4138747c7faed27019542d6ffec91c4f2b101109bfa6f015887f215")
  /// v0.1 (`ssb_low_order`): retired, the parity reference of v0.2 / v0.3 in the tests.
  static let lowOrderDerivativesReference = Entry(
    name: "ssb-low-order-derivatives", version: "0.1", files: ["ssb_low_order"],
    registeredSourceSHA256: "40da0bd64def2fcb21831f57777020331306e755495047031a40c41ef727b48b")

  /// The exact fifth-order agreement gradient on all BF pixels or every n-th (`fifth_gradient_scale`,
  /// `fifth_gradient_rows`) after the `fifthAgreement` value pass (fitting, not reconstruction). v0.1
  /// (`fifth_gradient` with its strided copy of the per-term objective) is retired.
  static let fifthGradient = Entry(
    name: "ssb-fifth-gradient", version: "0.2",
    files: ["ssb_fifth_order", "ssb_fifth_agreement", "ssb_fifth_order_gradient"],
    registeredSourceSHA256: "9a751b5a12baee7e35949fa73fddc266e1b59c3d0614cefe09e064bbff04f8d8")

  /// The regularized direct reconstructions (`phase`, `complex`): the k-only pixel terms, the normal equations of every
  /// scan frequency with its supported aliases (`direct_normal_equations`, float32 per BF pixel, float-float across
  /// batches) and the wave of the solved spectrum; the systems are solved in float64 on the CPU.
  static let direct = Entry(
    name: "ssb-direct", version: "0.1", files: ["ssb_joint", "ssb_fifth_order", "ssb_direct"],
    registeredSourceSHA256: "8d1237658f22e859004f998cbfcba22ffffa0c0de9f907d21c6bde66bf5672f3")

  /// SHA-256 of an entry's shipped source files, concatenated in order.
  static func sourceSHA256(_ entry: Entry) throws -> String {
    MetalKernelIdentity.sha256(Data(try MetalSSBShaderSource.load(entry.files).utf8))
  }
}
