import Foundation
import Metal

/// Half-open scan bounds in source pixel coordinates, ordered (row, column).
/// An explicit crop keeps the original scan spacing and every detector pixel.
public struct SSBScanRegion: Codable, Equatable, Sendable {
  public let row: Int
  public let column: Int
  public let size: Int
  public init(row: Int, column: Int, size: Int) {
    self.row = row
    self.column = column
    self.size = size
  }
  public func validate(sourceRows: Int, sourceColumns: Int) throws {
    guard [128, 256, 512].contains(size), row >= 0, column >= 0,
      row <= sourceRows - size, column <= sourceColumns - size
    else {
      throw MetalSSBError.invalidGeometry(
        "Choose a 128, 256 or 512 pixel square inside the scan. Scan positions are never binned to fit."
      )
    }
  }
  /// Copy exact UInt32 detector columns on the caller's command buffer.
  /// The source contains full scan planes; the destination contains ROI planes.
  public func encodeColumns(
    source: MTLBuffer, destination: MTLBuffer, count: Int,
    sourceRows: Int, sourceColumns: Int, commands: MTLCommandBuffer
  ) throws {
    try validate(sourceRows: sourceRows, sourceColumns: sourceColumns)
    guard (1...32).contains(count), source.length >= count * sourceRows * sourceColumns * 4,
      destination.length >= count * size * size * 4,
      source.device.registryID == destination.device.registryID,
      source.device.registryID == commands.commandQueue.device.registryID,
      let blit = commands.makeBlitCommandEncoder()
    else {
      throw MetalSSBError.invalidGeometry(
        "ROI extraction needs full source columns and a matching GPU destination.")
    }
    for k in 0..<count {
      for r in 0..<size {
        blit.copy(
          from: source, sourceOffset: ((k * sourceRows + row + r) * sourceColumns + column) * 4,
          to: destination, destinationOffset: (k * size + r) * size * 4, size: size * 4)
      }
    }
    blit.endEncoding()
  }
}

extension MetalSSBCalibration {
  /// The uncapped output sampling nominal circular-aperture SSB support needs: output Nyquist frequency equal to
  /// 2 * semiangle / wavelength on the more coarsely sampled scan axis. Matches Python
  /// `optics.physics.ssb_upsampling_requirement`.
  public func ssbUpsamplingRequirement() throws -> Double {
    let values = [voltageKV, semiangleMrad, scanSamplingRowAngstroms, scanSamplingColumnAngstroms]
    guard values.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw MetalSSBError.invalidGeometry(
        "Automatic output sampling needs positive energy, convergence angle and scan spacing.")
    }
    let wavelength = Self.wavelengthAngstroms(voltageKV: voltageKV)
    return max(scanSamplingRowAngstroms, scanSamplingColumnAngstroms) * 4 * semiangleMrad * 1e-3
      / wavelength
  }

  /// Automatic output sampling: the smallest of 1, 2, 3 and 4x covering nominal circular-aperture SSB support,
  /// or 4x when the requirement exceeds the automatic limit. Matches Python `optics.physics.ssb_upsampling_factor`.
  /// This is a grid selection, not a measured resolution claim or detector-angle cutoff; a capped 4x grid does not
  /// imply full nominal bandwidth (`ssbUpsamplingRequirement` gives the requirement). 8x stays available as an
  /// explicit `SSBOutputSampling.factor(8)`.
  public func ssbUpsamplingFactor() throws -> Int {
    let required = try ssbUpsamplingRequirement()
    return [1, 2, 3, 4].first(where: { Double($0) >= required - 1e-12 }) ?? 4
  }
}

/// SSB output sampling: Auto (from the acquisition calibration, 1x-4x) or an explicit factor. Upsampling
/// unfolds the measured Fourier aliases on a finer output grid; it is not interpolation, and fitting and source
/// data are unchanged.
public enum SSBOutputSampling: Codable, Equatable, Hashable, Sendable {
  case auto
  case factor(Int)

  /// The viewer choices: Auto and explicit 1x to 4x.
  public static let choices: [SSBOutputSampling] = [
    .auto, .factor(1), .factor(2), .factor(3), .factor(4),
  ]

  /// The factors a reconstruction accepts explicitly (8x as in Python, outside the viewer choices).
  public static let supportedFactors = [1, 2, 3, 4, 8]

  /// The output factor for this calibration.
  public func resolve(_ calibration: MetalSSBCalibration) throws -> Int {
    switch self {
    case .auto:
      return try calibration.ssbUpsamplingFactor()
    case .factor(let value):
      guard Self.supportedFactors.contains(value) else {
        throw MetalSSBError.invalidGeometry("Use an SSB sampling factor of 1, 2, 3, 4 or 8.")
      }
      return value
    }
  }

  /// "Auto" or "3×".
  public var label: String {
    switch self {
    case .auto: return "Auto"
    case .factor(let value): return "\(value)×"
    }
  }
}
