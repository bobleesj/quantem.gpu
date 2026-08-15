// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "MetalKernels",
  platforms: [.macOS(.v14), .iOS(.v17)],
  products: [
    .library(name: "MetalDisplayKernels", targets: ["MetalDisplayKernels"]),
    .library(name: "Metal4DSTEMKernels", targets: ["Metal4DSTEMKernels"]),
    .library(name: "Native4DSTEMIO", targets: ["Native4DSTEMIO"]),
    .executable(
      name: "metal-display-benchmark",
      targets: ["MetalDisplayBenchmark"]
    ),
    .executable(
      name: "native-4dstem-io-benchmark",
      targets: ["Native4DSTEMIOBenchmark"]
    ),
  ],
  targets: [
    .binaryTarget(
      name: "CHDF5",
      path: "src/quantem/gpu/swift/Vendor/CHDF5.xcframework"
    ),
    .target(
      name: "CNativeHDF5",
      dependencies: ["CHDF5"],
      path: "src/quantem/gpu/swift/Sources/CNativeHDF5",
      linkerSettings: [.linkedLibrary("z")]
    ),
    .target(
      name: "Native4DSTEMIO",
      dependencies: ["CNativeHDF5"],
      path: "src/quantem/gpu/swift/Sources/Native4DSTEMIO",
      resources: [.copy("Resources")]
    ),
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
    .executableTarget(
      name: "Native4DSTEMIOBenchmark",
      dependencies: ["Native4DSTEMIO"],
      path: "src/quantem/gpu/swift/Benchmarks/Native4DSTEMIOBenchmark"
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
    .testTarget(
      name: "Native4DSTEMIOTests",
      dependencies: ["Native4DSTEMIO"],
      path: "src/quantem/gpu/swift/Tests/Native4DSTEMIOTests",
      resources: [.copy("Fixtures")]
    ),
  ]
)
