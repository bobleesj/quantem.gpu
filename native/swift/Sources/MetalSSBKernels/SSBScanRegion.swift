import Foundation
import Metal

/// Half-open scan bounds in source pixel coordinates, ordered (row, column).
/// An explicit crop keeps the original scan spacing and every detector pixel.
public struct SSBScanRegion: Codable, Equatable, Sendable {
  public let row: Int
  public let column: Int
  public let size: Int
  public init(row: Int, column: Int, size: Int) {
    self.row = row; self.column = column; self.size = size
  }
  public func validate(sourceRows: Int, sourceColumns: Int) throws {
    guard [128, 256, 512].contains(size), row >= 0, column >= 0,
      row <= sourceRows - size, column <= sourceColumns - size else {
      throw MetalSSBError.invalidGeometry("Choose a 128, 256 or 512 pixel square inside the scan. Scan positions are never binned to fit.")
    }
  }
  /// Copy exact UInt32 detector columns on the caller's command buffer.
  /// The source contains full scan planes; the destination contains ROI planes.
  public func encodeColumns(source: MTLBuffer, destination: MTLBuffer, count: Int,
                            sourceRows: Int, sourceColumns: Int, commands: MTLCommandBuffer) throws {
    try validate(sourceRows: sourceRows, sourceColumns: sourceColumns)
    guard (1...32).contains(count), source.length >= count * sourceRows * sourceColumns * 4,
      destination.length >= count * size * size * 4,
      source.device.registryID == destination.device.registryID,
      source.device.registryID == commands.commandQueue.device.registryID,
      let blit = commands.makeBlitCommandEncoder() else {
      throw MetalSSBError.invalidGeometry("ROI extraction needs full source columns and a matching GPU destination.")
    }
    for k in 0..<count {
      for r in 0..<size {
        blit.copy(from: source, sourceOffset: ((k * sourceRows + row + r) * sourceColumns + column) * 4,
          to: destination, destinationOffset: (k * size + r) * size * 4, size: size * 4)
      }
    }
    blit.endEncoding()
  }
}

extension MetalSSBCalibration {
  /// Smallest supported factor covering nominal circular-aperture SSB support.
  /// Matches Python `optics.physics.ssb_upsampling_factor`. This is a grid
  /// selection, not a measured resolution claim or detector-angle cutoff.
  public func ssbUpsamplingFactor() throws -> Int {
    let values = [voltageKV, semiangleMrad, scanSamplingRowAngstroms, scanSamplingColumnAngstroms]
    guard values.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw MetalSSBError.invalidGeometry("Automatic output sampling needs positive energy, convergence angle and scan spacing.")
    }
    let h = 6.62607015e-34, mass = 9.1093837015e-31, charge = 1.602176634e-19, c = 299792458.0
    let energy = voltageKV * 1000 * charge
    let wavelength = h / sqrt(2 * mass * energy * (1 + energy / (2 * mass * c * c))) * 1e10
    let required = max(scanSamplingRowAngstroms, scanSamplingColumnAngstroms) * 4 * semiangleMrad * 1e-3 / wavelength
    guard let factor = [1, 2, 3, 4, 8].first(where: { Double($0) >= required - 1e-12 }) else {
      throw MetalSSBError.invalidGeometry("Nominal SSB support needs more than 8× output. Review scan calibration; no limited grid was substituted.")
    }
    return factor
  }
}
