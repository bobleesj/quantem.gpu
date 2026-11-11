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

  /// "direct-redraw v0.3 · 3f2a9c… (factor=3 degree=6 depth=on)" for status lines.
  public var label: String { "\(name) v\(version) · \(fingerprint.prefix(12)) (\(variant))" }

  static func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
}

/// The registered native direct ptychography reconstruction and fit kernels; see `docs/performance/kernels.md`.
enum DirectPtychoKernelRegistry {
  struct Entry {
    let name: String
    let version: String
    let files: [String]
    /// The source digest this version was validated with. `DirectKernelRegistryTests` fails when the shipped source
    /// changes without a new version here and in the registry document.
    let registeredSourceSHA256: String
  }

  /// Direct ptychography (`phase_of_mean`) at 1x-4x: the only production path there. v0.3 evaluates chi(k +- q) - chi(k) as
  /// polynomials expanded about each BF pixel and forms gamma from the half sum and half difference of the two
  /// phase differences, which resolves nearly cancelling sidebands at the lowest scan frequencies; v0.2 (differences
  /// of separately rounded chi values) is retired.
  static let redraw = Entry(
    name: "direct-redraw", version: "0.3", files: ["direct_redraw"],
    registeredSourceSHA256: "fc370d789c0c92428e79c20bda7c53d678b515de7a1b1cdc28b109d5e03fff35")
  /// The earlier per-term / separable coherent kernels: explicit 8x only, and the 1x-4x parity reference in tests.
  /// v0.2 is v0.1 without the retired `complex_wave` amplitude kernel (`model_wave_amplitude`); v0.3 is v0.2 without
  /// the deleted objective kernel `fifth_agreement` in `direct_fifth_order`. The reconstruction kernels are unchanged.
  static let perTerm = Entry(
    name: "direct-per-term", version: "0.3",
    files: ["direct_model", "direct_fifth_order", "direct_separable"],
    registeredSourceSHA256: "4d9f450efa180a20437d648b444fac4a4c6263ee2d78d8201dc6b42a92e9b93f")
  /// The fifth-order agreement objective of the fit (`fifth_agreement_rows` with the `fifth_probe`, `fifth_power`
  /// and `fifth_score` passes) on all BF pixels or every n-th: the only production objective kernel there, also the
  /// value pass of `fifthGradient`. v0.3 added the BF stride to v0.2 (same per-pair terms); v0.4 is v0.3 with v0.1,
  /// `fifth_agreement` in `direct_fifth_order`, deleted (2026-10-25), the kernels unchanged.
  /// v0.5 skips per-pixel arithmetic at zero-weight band frequencies while retaining
  /// every output slot and barrier. Objective and gradient results remain exact.
  static let fifthAgreement = Entry(
    name: "direct-fifth-agreement", version: "0.5",
    files: ["direct_fifth_order", "direct_fifth_agreement"],
    registeredSourceSHA256: "8ebd884e75879a1daebb735a09396941d4f2bfdf2547b2d2993b2eaff5cec64f")
  /// The low-order agreement objective with angle-sum phases (`joint_agreement_angle_sums`), used by the primary
  /// fit's search trials, the only low-order agreement kernel. v0.3 is v0.2 with the retired `mean_phase` kernels
  /// removed from `direct_joint`; v0.4 is v0.3 with v0.1 (`joint_objective_angle_sums`) and the per-term agreement
  /// (`joint_objective_batch`, `joint_trial_probe`) deleted from `direct_joint` (2026-10-25). The agreement kernels are
  /// unchanged.
  static let jointAgreement = Entry(
    name: "direct-joint-agreement", version: "0.4",
    files: ["direct_joint", "direct_joint_agreement"],
    registeredSourceSHA256: "3bb7e0fa64beb0787c1796ccd7e632f47d5ee6949cbaca6debbffd952dbbce1b")

  /// The primary fit's exact low-order value, gradient and Hessian (trust-region Newton), `DirectPtychoLowOrderDerivatives`:
  /// the value pass, the derivative pass and the fused pass of `direct_low_order_derivatives` (with the aperture and
  /// quotient helpers of `direct_joint_agreement`). v0.3 is v0.2 with the retired `mean_phase` kernels removed from
  /// `direct_joint`; v0.4 is v0.3 with the deleted agreement kernels removed from `direct_joint` (2026-10-25). The
  /// derivative kernels are unchanged. v0.1 (`direct_low_order`) was deleted on 2026-10-25.
  static let lowOrderDerivatives = Entry(
    name: "direct-low-order-derivatives", version: "0.4",
    files: ["direct_joint", "direct_joint_agreement", "direct_low_order_derivatives"],
    registeredSourceSHA256: "68bdced4bc2dfeb8263fcc9cd0b3463aed019493c641906b931518396981e561")

  /// The exact fifth-order agreement gradient on all BF pixels or every n-th (`fifth_gradient_scale`,
  /// `fifth_gradient_rows`) after the `fifthAgreement` value pass (fitting, not reconstruction). v0.1
  /// (`fifth_gradient` with its strided copy of the per-term objective) is retired; v0.3 is v0.2 with
  /// `fifth_agreement` deleted from `direct_fifth_order` (2026-10-25), the kernels unchanged.
  /// v0.4 uses the zero-weight-aware v0.5 value pass; its gradient kernels are unchanged.
  static let fifthGradient = Entry(
    name: "direct-fifth-gradient", version: "0.4",
    files: ["direct_fifth_order", "direct_fifth_agreement", "direct_fifth_order_gradient"],
    registeredSourceSHA256: "484134db7776e01314f4084d5f6540c8d74db129f5dd97394182bd2fa61e1cb0")

  /// The regularized direct reconstructions (`phase`, `complex`): the k-only pixel terms, the normal equations of every
  /// scan frequency with its supported aliases (float32 per BF pixel, float-float across batches;
  /// `direct_normal_tiled`, one thread per frequency and alias, for complex systems of two or more aliases,
  /// `direct_normal_equations` otherwise), the drag windows' Chebyshev coefficients and their evaluation
  /// (`direct_window_*`), and the wave of the solved spectrum; the systems are solved in float64 on the CPU. v0.4
  /// takes the k-only aperture term from CUDA's square-pixel probe geometry, as the transfer does. Retired: v0.1
  /// (`direct_normal_equations` alone, no drag windows), v0.2 and v0.3 (v0.2 with the deleted agreement kernels
  /// removed from the shared `direct_joint` and `direct_fifth_order`; both take A(k) from the general aperture formula).
  static let direct = Entry(
    name: "direct-regularized", version: "0.4",
    files: ["direct_joint", "direct_fifth_order", "direct_regularized"],
    registeredSourceSHA256: "199b8f9f28415fe84bc930de50f8a63d9a78c26747c304b90e2e33ddad2b1720")

  /// SHA-256 of an entry's shipped source files, concatenated in order.
  static func sourceSHA256(_ entry: Entry) throws -> String {
    MetalKernelIdentity.sha256(Data(try MetalDirectPtychoShaderSource.load(entry.files).utf8))
  }
}
