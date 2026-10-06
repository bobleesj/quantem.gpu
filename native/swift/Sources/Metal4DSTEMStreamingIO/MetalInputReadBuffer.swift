import Metal

/// Retains one staging allocation while a serial input reader owns its bytes.
/// The caller must wait for that reader before GPU use, reuse, or load teardown.
struct MetalInputReadBuffer: @unchecked Sendable {
  let buffer: MTLBuffer
}
