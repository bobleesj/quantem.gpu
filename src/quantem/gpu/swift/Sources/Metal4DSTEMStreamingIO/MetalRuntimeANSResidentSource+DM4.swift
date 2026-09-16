import Metal
import Native4DSTEMIO

extension MetalRuntimeANSResidentSource {
  /// Encode complete camera counts through the shared bounded Metal array path.
  public static func load(
    camera: NativeDM4Source, device: MTLDevice,
    maximumAdditionalBytes: UInt64? = nil,
    shouldCancel: () -> Bool = { false },
    progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> MetalRuntimeANSResidentSource {
    try load(
      array: camera, device: device, maximumAdditionalBytes: maximumAdditionalBytes,
      shouldCancel: shouldCancel, progress: progress)
  }
}
