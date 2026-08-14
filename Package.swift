// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "MetalKernels",
  platforms: [.macOS(.v14), .iOS(.v17)],
  products: [
    .library(name: "MetalDisplayKernels", targets: ["MetalDisplayKernels"]),
    .library(name: "Metal4DSTEMKernels", targets: ["Metal4DSTEMKernels"]),
    .executable(
      name: "metal-display-benchmark",
      targets: ["MetalDisplayBenchmark"]
    ),
  ],
  targets: [
    .target(
      name: "MetalDisplayKernels",
      path: "src/quantem/gpu/swift/Sources/MetalDisplayKernels",
      resources: [.copy("Resources")]
    ),
    .target(
      name: "Metal4DSTEMKernels",
      path: "src/quantem/gpu/swift/Sources/Metal4DSTEMKernels",
      resources: [.copy("Resources")]
    ),
    .executableTarget(
      name: "MetalDisplayBenchmark",
      dependencies: ["MetalDisplayKernels"],
      path: "src/quantem/gpu/swift/Benchmarks/MetalDisplayBenchmark"
    ),
    .testTarget(
      name: "MetalDisplayKernelsTests",
      dependencies: ["MetalDisplayKernels"],
      path: "src/quantem/gpu/swift/Tests/MetalDisplayKernelsTests"
    ),
    .testTarget(
      name: "Metal4DSTEMKernelsTests",
      dependencies: ["Metal4DSTEMKernels"],
      path: "src/quantem/gpu/swift/Tests/Metal4DSTEMKernelsTests"
    ),
  ]
)
