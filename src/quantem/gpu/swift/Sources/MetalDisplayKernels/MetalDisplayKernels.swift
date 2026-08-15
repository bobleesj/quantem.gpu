import Foundation
import Metal
import simd

/// Display normalization applied before lookup-table colormapping.
public enum MetalDisplayScale: UInt32, CaseIterable, Sendable {
  case linear = 0
  case logarithmic = 1

  public var title: String {
    switch self {
    case .linear: "Linear"
    case .logarithmic: "Log"
    }
  }
}

/// Colormaps bundled with the native Metal display package.
public enum MetalColormap: String, CaseIterable, Sendable {
  case gray
  case viridis
  case plasma
  case inferno
  case magma
  case magenta
  case hot
  case hsv
  case turbo
  case rdBu = "RdBu"
  case cividis
  case seismic
  case rdBuReversed = "RdBu_r"
  case twilight
  case twilightShifted = "twilight_shifted"

  public var title: String {
    switch self {
    case .rdBu: "RdBu"
    case .rdBuReversed: "RdBu reversed"
    case .twilightShifted: "Twilight shifted"
    default: rawValue.capitalized
    }
  }
}

/// Buffer layout shared by Swift and the Metal display kernels.
@frozen
public struct MetalDisplayParameters: Sendable {
  public var rows: UInt32
  public var cols: UInt32
  public var low: UInt32
  public var high: UInt32
  @usableFromInline var scaleMode: UInt32
  public var lutCount: UInt32
  @usableFromInline var padding0: UInt32 = 0
  @usableFromInline var padding1: UInt32 = 0

  /// Display normalization encoded in the Metal parameter buffer.
  public var scale: MetalDisplayScale {
    get { MetalDisplayScale(rawValue: scaleMode)! }
    set { scaleMode = newValue.rawValue }
  }

  /// Create display parameters for one row-major `uint32` image.
  public init(
    rows: Int,
    cols: Int,
    low: UInt32,
    high: UInt32,
    scale: MetalDisplayScale,
    lutCount: Int = 256
  ) {
    precondition(rows > 0 && cols > 0, "Image rows and columns must be positive.")
    precondition(lutCount > 0, "LUT count must be positive.")
    self.rows = UInt32(rows)
    self.cols = UInt32(cols)
    self.low = low
    self.high = max(low, high)
    self.scaleMode = scale.rawValue
    self.lutCount = UInt32(lutCount)
  }
}

/// Buffer layout for row-major `float32` derived images such as CoM and DPC.
@frozen
public struct MetalFloatDisplayParameters: Sendable {
  public var rows: UInt32
  public var cols: UInt32
  public var low: Float
  public var high: Float
  @usableFromInline var scaleMode: UInt32
  public var lutCount: UInt32
  @usableFromInline var padding0: UInt32 = 0
  @usableFromInline var padding1: UInt32 = 0

  public var scale: MetalDisplayScale {
    get { MetalDisplayScale(rawValue: scaleMode)! }
    set { scaleMode = newValue.rawValue }
  }

  public init(
    rows: Int,
    cols: Int,
    low: Float,
    high: Float,
    scale: MetalDisplayScale,
    lutCount: Int = 256
  ) {
    precondition(rows > 0 && cols > 0, "Image rows and columns must be positive.")
    precondition(lutCount > 0, "LUT count must be positive.")
    self.rows = UInt32(rows)
    self.cols = UInt32(cols)
    self.low = low
    self.high = max(low, high)
    self.scaleMode = scale.rawValue
    self.lutCount = UInt32(lutCount)
  }
}

/// Errors raised while loading shared display resources.
public enum MetalDisplayKernelsError: LocalizedError {
  case missingResource(String)
  case invalidColormap(String)
  case invalidColormapResource(String)
  case libraryCompilation(String)
  case allocation(String)

  public var errorDescription: String? {
    switch self {
    case .missingResource(let name): "MetalDisplayKernels is missing \(name)."
    case .invalidColormap(let name): "MetalDisplayKernels could not decode colormap \(name)."
    case .invalidColormapResource(let message):
      "MetalDisplayKernels could not load colormaps.json: \(message)"
    case .libraryCompilation(let message):
      "Metal display shader compilation failed: \(message)"
    case .allocation(let message): "Metal display allocation failed: \(message)"
    }
  }
}

/// Shared Metal display resources and stable shader function names.
public enum MetalDisplayKernels {
  public static let vertexFunction = "metal_display_vertex"
  public static let fragmentFunction = "metal_display_fragment"
  public static let rangeFunction = "metal_range_u32"
  public static let histogramFunction = "metal_histogram_u32"
  public static let floatFragmentFunction = "metal_display_fragment_f32"
  public static let floatHistogramFunction = "metal_histogram_f32"

  /// Compile the bundled display shader source for a Metal device.
  public static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
    guard let url = resourceURL(name: "display", extension: "metal") else {
      throw MetalDisplayKernelsError.missingResource("display.metal")
    }
    do {
      return try device.makeLibrary(
        source: String(contentsOf: url, encoding: .utf8),
        options: nil
      )
    } catch {
      throw MetalDisplayKernelsError.libraryCompilation(error.localizedDescription)
    }
  }

  /// Return one 256-entry RGBA lookup table.
  public static func lut(_ colormap: MetalColormap) throws -> [SIMD4<Float>] {
    let pointsByName = try controlPoints.get()
    guard let points = pointsByName[colormap.rawValue], points.count >= 2 else {
      throw MetalDisplayKernelsError.invalidColormap(colormap.rawValue)
    }
    return (0..<256).map { index in
      let position = Float(index) / 255 * Float(points.count - 1)
      let lower = min(Int(floor(position)), points.count - 1)
      let upper = min(lower + 1, points.count - 1)
      let fraction = position - Float(lower)
      let rgb = points[lower] + fraction * (points[upper] - points[lower])
      return SIMD4<Float>(
        round(rgb.x) / 255,
        round(rgb.y) / 255,
        round(rgb.z) / 255,
        1
      )
    }
  }

  /// Allocate a shared Metal buffer containing one RGBA lookup table.
  public static func makeLUTBuffer(
    device: MTLDevice,
    colormap: MetalColormap
  ) throws -> MTLBuffer {
    let values = try lut(colormap)
    guard
      let buffer = values.withUnsafeBytes({ bytes in
        bytes.baseAddress.flatMap {
          device.makeBuffer(bytes: $0, length: bytes.count, options: .storageModeShared)
        }
      })
    else {
      throw MetalDisplayKernelsError.allocation("256-entry \(colormap.rawValue) LUT")
    }
    return buffer
  }

  private static func resourceURL(name: String, extension suffix: String) -> URL? {
    let packagedURL = Bundle.main.resourceURL?
      .appendingPathComponent(
        "MetalKernels_MetalDisplayKernels.bundle",
        isDirectory: true
      )
      .appendingPathComponent("Resources", isDirectory: true)
      .appendingPathComponent("\(name).\(suffix)")
    return packagedURL.flatMap {
      FileManager.default.fileExists(atPath: $0.path) ? $0 : nil
    } ?? Bundle.module.url(
      forResource: name,
      withExtension: suffix,
      subdirectory: "Resources"
    ) ?? Bundle.module.url(forResource: name, withExtension: suffix)
  }

  private static let controlPoints:
    Result<
      [String: [SIMD3<Float>]], MetalDisplayKernelsError
    > = {
      guard let url = resourceURL(name: "colormaps", extension: "json") else {
        return .failure(.missingResource("colormaps.json"))
      }
      do {
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode([String: [[Float]]].self, from: data)
        var result: [String: [SIMD3<Float>]] = [:]
        for (name, rows) in decoded {
          guard rows.count >= 2, rows.allSatisfy({ $0.count == 3 }) else {
            return .failure(.invalidColormap(name))
          }
          result[name] = rows.map { SIMD3($0[0], $0[1], $0[2]) }
        }
        return .success(result)
      } catch {
        return .failure(.invalidColormapResource(error.localizedDescription))
      }
    }()
}
