import Foundation

@main
enum CheckInteractionMode {
  static func main() {
    let normal = PairedRuntimeConfiguration(mode: .normal)
    let fast = PairedRuntimeConfiguration(mode: .fast)
    precondition(normal.value("QGPU_PAIRED_RUNTIME_POLAR_INDEX") == nil)
    precondition(fast.value("QGPU_PAIRED_RUNTIME_POLAR_INDEX") == "1")
    precondition(fast.value("QGPU_PAIRED_RUNTIME_POLAR_LAYOUT") == "radial1fine4")
    precondition(fast.value("QGPU_PAIRED_RUNTIME_COMPACT_EVENT_MAX_NONZERO") == "192")
    precondition(fast.value("QGPU_PAIRED_RUNTIME_BLOCK_STRIDE") == nil)
    // Explicit production modes ignore benchmark overrides even after a refresh.
    setenv("QGPU_PAIRED_RUNTIME_POLAR_LAYOUT", "invalid-test-layout", 1)
    setenv("QGPU_PAIRED_RUNTIME_BLOCK_STRIDE", "8", 1)
    pairedRuntimeEnvironmentDidChange()
    precondition(fast.value("QGPU_PAIRED_RUNTIME_POLAR_LAYOUT") == "radial1fine4")
    precondition(fast.value("QGPU_PAIRED_RUNTIME_BLOCK_STRIDE") == nil)
    precondition(normal.value("QGPU_PAIRED_RUNTIME_BLOCK_STRIDE") == nil)
    precondition(PairedRuntimeConfiguration(mode: nil)
      .value("QGPU_PAIRED_RUNTIME_POLAR_LAYOUT") == "invalid-test-layout")
    print("PASS Normal/Fast immutable configuration and experimental override isolation")
  }
}
