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

  /// `phase_of_mean` and `complex_wave` at 1x-4x: the only production path there.
  static let redraw = Entry(
    name: "ssb-redraw", version: "0.2", files: ["ssb_redraw"],
    registeredSourceSHA256: "b20d7914811d0fb2941d59907ba6e15607829f0988b08498f262d9942a3a5e34")
  /// The earlier per-term / separable coherent kernels: explicit 8x only, and the 1x-4x parity reference in tests.
  static let perTerm = Entry(
    name: "ssb-per-term", version: "0.1", files: ["ssb_model", "ssb_fifth_order", "ssb_separable"],
    registeredSourceSHA256: "1c390cc7b833c1cd83ec77cdd63670edc78a33a42426e2ad1512950677983fa5")
  /// The fifth-order agreement objective of the fit (`fifth_agreement_rows` with the `fifth_probe`, `fifth_power`
  /// and `fifth_score` passes): the only production objective kernel there. v0.1 was `fifth_agreement` in
  /// `ssb_fifth_order` (ssb-per-term), now the parity reference of the tests.
  static let fifthAgreement = Entry(
    name: "ssb-fifth-agreement", version: "0.2", files: ["ssb_fifth_order", "ssb_fifth_agreement"],
    registeredSourceSHA256: "f2f700147e47edaac7f91e41e32e955ca016df4de4236b4b6b314e0f0a23bfad")
  /// `mean_phase` (C10/C12 only).
  static let meanPhase = Entry(
    name: "ssb-mean-phase", version: "0.1", files: ["ssb_joint", "ssb_joint_fft", "ssb_fft"],
    registeredSourceSHA256: "4aa7c1da63913a45e8075c41a36497be32fbe527e5aaceaddb70eb340b9f7e53")

  /// The primary fit's exact low-order gradient and Hessian (trust-region Newton), `SSBLowOrderDerivatives`.
  static let lowOrderDerivatives = Entry(
    name: "ssb-low-order-derivatives", version: "0.1", files: ["ssb_low_order"],
    registeredSourceSHA256: "40da0bd64def2fcb21831f57777020331306e755495047031a40c41ef727b48b")

  /// The exact fifth-order agreement gradient and its BF / band-subsample objective (fitting, not reconstruction).
  static let fifthGradient = Entry(
    name: "ssb-fifth-gradient", version: "0.1", files: ["ssb_fifth_order_gradient"],
    registeredSourceSHA256: "dc8d80c38d16c05d6d43fba517620f772475995581d1dda42ca65d4c792d52e5")

  /// SHA-256 of an entry's shipped source files, concatenated in order.
  static func sourceSHA256(_ entry: Entry) throws -> String {
    MetalKernelIdentity.sha256(Data(try MetalSSBShaderSource.load(entry.files).utf8))
  }
}
