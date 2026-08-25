// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "MetalKernels",
  platforms: [.macOS(.v14), .iOS(.v17)],
  products: [
    .library(name: "MetalDisplayKernels", targets: ["MetalDisplayKernels"]),
    .library(name: "Metal4DSTEMKernels", targets: ["Metal4DSTEMKernels"]),
    .library(name: "MetalImageFFT", targets: ["MetalImageFFT"]),
    .library(name: "MetalImageRuntime", targets: ["MetalImageRuntime"]),
    .library(name: "MetalSSBKernels", targets: ["MetalSSBKernels"]),
    .library(name: "Native4DSTEMIO", targets: ["Native4DSTEMIO"]),
    .library(
      name: "Metal4DSTEMStreamingIO",
      targets: ["Metal4DSTEMStreamingIO"]
    ),
    .executable(
      name: "metal-display-benchmark",
      targets: ["MetalDisplayBenchmark"]
    ),
    .executable(
      name: "native-4dstem-io-benchmark",
      targets: ["Native4DSTEMIOBenchmark"]
    ),
    .executable(
      name: "metal-image-fft-benchmark",
      targets: ["MetalImageFFTBenchmark"]
    ),
    .executable(
      name: "metal-image-runtime-benchmark",
      targets: ["MetalImageRuntimeBenchmark"]
    ),
    .executable(
      name: "metal-ssb-benchmark",
      targets: ["MetalSSBBenchmark"]
    ),
    .executable(
      name: "metal-4dstem-binning-benchmark",
      targets: ["Metal4DSTEMBinningBenchmark"]
    ),
    .executable(
      name: "metal-4dstem-indexed-load-benchmark",
      targets: ["Metal4DSTEMStreamingIOBenchmark"]
    ),
    .executable(
      name: "metal-4dstem-dpc-benchmark",
      targets: ["Metal4DSTEMDPCBenchmark"]
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
    .target(
      name: "Metal4DSTEMStreamingIO",
      dependencies: [
        "CMetal4DSTEMInteractions",
        "Metal4DSTEMKernels",
        "Native4DSTEMIO",
      ],
      path: "src/quantem/gpu/swift/Sources/Metal4DSTEMStreamingIO"
    ),
    .target(
      name: "CMetal4DSTEMInteractions",
      path: "src/quantem/gpu/swift/Sources/CMetal4DSTEMInteractions"
    ),
    .target(
      name: "MetalImageFFT",
      path: "src/quantem/gpu/swift/Sources/MetalImageFFT",
      resources: [.copy("Resources")],
      linkerSettings: [
        .linkedFramework("MetalPerformanceShaders"),
        .linkedFramework("MetalPerformanceShadersGraph"),
      ]
    ),
    .target(
      name: "MetalImageRuntime",
      dependencies: ["MetalDisplayKernels"],
      path: "src/quantem/gpu/swift/Sources/MetalImageRuntime"
    ),
    .target(
      name: "MetalSSBKernels",
      path: "src/quantem/gpu/swift/Sources/MetalSSBKernels",
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
    .executableTarget(
      name: "MetalImageFFTBenchmark",
      dependencies: ["MetalImageFFT"],
      path: "src/quantem/gpu/swift/Benchmarks/MetalImageFFTBenchmark",
      exclude: ["compare_torch_fft.py"]
    ),
    .executableTarget(
      name: "MetalImageRuntimeBenchmark",
      dependencies: ["MetalImageRuntime", "MetalImageFFT"],
      path: "src/quantem/gpu/swift/Benchmarks/MetalImageRuntimeBenchmark"
    ),
    .executableTarget(
      name: "MetalSSBBenchmark",
      dependencies: ["MetalSSBKernels"],
      path: "src/quantem/gpu/swift/Benchmarks/MetalSSBBenchmark"
    ),
    .executableTarget(
      name: "Metal4DSTEMBinningBenchmark",
      dependencies: ["Metal4DSTEMKernels"],
      path: "src/quantem/gpu/swift/Benchmarks/Metal4DSTEMBinningBenchmark"
    ),
    .executableTarget(
      name: "Metal4DSTEMStreamingIOBenchmark",
      dependencies: ["Metal4DSTEMStreamingIO", "Native4DSTEMIO"],
      path: "src/quantem/gpu/swift/Benchmarks/Metal4DSTEMStreamingIOBenchmark"
    ),
    .executableTarget(
      name: "Metal4DSTEMDPCBenchmark",
      dependencies: ["Metal4DSTEMKernels"],
      path: "src/quantem/gpu/swift/Benchmarks/Metal4DSTEMDPCBenchmark"
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
      name: "MetalImageFFTTests",
      dependencies: ["MetalImageFFT"],
      path: "src/quantem/gpu/swift/Tests/MetalImageFFTTests"
    ),
    .testTarget(
      name: "MetalImageRuntimeTests",
      dependencies: ["MetalImageRuntime"],
      path: "src/quantem/gpu/swift/Tests/MetalImageRuntimeTests"
    ),
    .testTarget(
      name: "MetalSSBKernelsTests",
      dependencies: ["MetalSSBKernels"],
      path: "src/quantem/gpu/swift/Tests/MetalSSBKernelsTests"
    ),
    .testTarget(
      name: "Native4DSTEMIOTests",
      dependencies: ["Metal4DSTEMStreamingIO", "Native4DSTEMIO"],
      path: "src/quantem/gpu/swift/Tests/Native4DSTEMIOTests",
      resources: [.copy("Fixtures")]
    ),
  ]
)
