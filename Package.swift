// swift-tools-version: 6.0
import PackageDescription

// Metal sources and data that Python also reads live once, beside the Python that
// reads them. Paths are relative to a target directory under native/swift/Sources.
let pythonPackage = "../../../../src/quantem/gpu"

let package = Package(
  name: "MetalKernels",
  platforms: [.macOS(.v14), .iOS(.v17)],
  products: [
    .library(name: "MetalDisplayKernels", targets: ["MetalDisplayKernels"]),
    .library(name: "Metal4DSTEMKernels", targets: ["Metal4DSTEMKernels"]),
    .library(name: "MetalImageFFT", targets: ["MetalImageFFT"]),
    .library(name: "MetalScientificNumerics", targets: ["MetalScientificNumerics"]),
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
      name: "metal-compact-h5-benchmark",
      targets: ["MetalCompactH5Benchmark"]
    ),
    .executable(
      name: "metal-original-hdf5-benchmark",
      targets: ["MetalOriginalHDF5Benchmark"]
    ),
    .executable(
      name: "metal-runtime-ans-benchmark",
      targets: ["MetalRuntimeANSBenchmark"]
    ),
    .executable(
      name: "metal-qem-detector-update-benchmark",
      targets: ["QEMDetectorUpdateBenchmark"]
    ),
    .executable(
      name: "metal-paired-runtime-tans-benchmark",
      targets: ["MetalPairedRuntimeTANSBenchmark"]
    ),
    .executable(
      name: "metal-paired-runtime-tans-series-benchmark",
      targets: ["MetalPairedRuntimeTANSSeriesBenchmark"]
    ),
    .executable(
      name: "metal-4dstem-dpc-benchmark",
      targets: ["Metal4DSTEMDPCBenchmark"]
    ),
  ],
  targets: [
    .binaryTarget(
      name: "CHDF5",
      path: "native/swift/Vendor/CHDF5.xcframework"
    ),
    .target(
      name: "CNativeHDF5",
      dependencies: ["CHDF5"],
      path: "native/swift/Sources/CNativeHDF5",
      linkerSettings: [.linkedLibrary("z")]
    ),
    .target(
      name: "Native4DSTEMIO",
      dependencies: ["CNativeHDF5"],
      path: "native/swift/Sources/Native4DSTEMIO",
      resources: [.copy("Resources")]
    ),
    .target(
      name: "MetalDisplayKernels",
      path: "native/swift/Sources/MetalDisplayKernels",
      resources: [
        .copy("\(pythonPackage)/display/colormaps.json"),
        .copy("\(pythonPackage)/display/metal/display.metal"),
      ]
    ),
    .target(
      name: "Metal4DSTEMKernels",
      path: "native/swift/Sources/Metal4DSTEMKernels",
      resources: [
        .copy("Resources"),
        .copy("\(pythonPackage)/io/hdf5/mps/kernels/qh5idx.metal"),
        .copy("\(pythonPackage)/resident/mps/kernels/runtime_spatial.msl"),
      ]
    ),
    .target(
      name: "Metal4DSTEMStreamingIO",
      dependencies: [
        "CMetal4DSTEMInteractions",
        "Metal4DSTEMKernels",
        "Native4DSTEMIO",
        "MetalCountResources",
      ],
      path: "native/swift/Sources/Metal4DSTEMStreamingIO"
    ),
    .target(
      name: "MetalCountResources",
      path: "native/swift/Sources/MetalCountResources",
      resources: [
        .copy("Resources"),
        .copy("\(pythonPackage)/io/hdf5/mps/kernels/save_uint16.msl"),
        .copy("\(pythonPackage)/resident/mps/kernels/count_tables.msl"),
        .copy("\(pythonPackage)/resident/mps/kernels/hot_pixels.msl"),
        .copy("\(pythonPackage)/resident/mps/kernels/precision.msl"),
        .copy("\(pythonPackage)/resident/mps/kernels/streamed_counts.msl"),
      ]
    ),
    .target(
      name: "MetalScientificNumerics",
      dependencies: ["Metal4DSTEMStreamingIO", "MetalCountResources"],
      path: "native/swift/Sources/MetalScientificNumerics",
      resources: [.copy("Resources")],
      linkerSettings: [.linkedFramework("MetalPerformanceShadersGraph")]
    ),
    .target(
      name: "CMetal4DSTEMInteractions",
      path: "native/swift/Sources/CMetal4DSTEMInteractions"
    ),
    .target(
      name: "MetalImageFFT",
      path: "native/swift/Sources/MetalImageFFT",
      resources: [.copy("Resources")],
      linkerSettings: [
        .linkedFramework("MetalPerformanceShaders"),
        .linkedFramework("MetalPerformanceShadersGraph"),
      ]
    ),
    .target(
      name: "MetalImageRuntime",
      dependencies: ["MetalDisplayKernels"],
      path: "native/swift/Sources/MetalImageRuntime"
    ),
    .target(
      name: "MetalSSBKernels",
      path: "native/swift/Sources/MetalSSBKernels",
      resources: [.copy("Resources")]
    ),
    .executableTarget(
      name: "MetalDisplayBenchmark",
      dependencies: ["MetalDisplayKernels"],
      path: "native/swift/Benchmarks/MetalDisplayBenchmark"
    ),
    .executableTarget(
      name: "Native4DSTEMIOBenchmark",
      dependencies: ["Native4DSTEMIO"],
      path: "native/swift/Benchmarks/Native4DSTEMIOBenchmark"
    ),
    .executableTarget(
      name: "MetalImageFFTBenchmark",
      dependencies: ["MetalImageFFT"],
      path: "native/swift/Benchmarks/MetalImageFFTBenchmark",
      exclude: ["compare_torch_fft.py"]
    ),
    .executableTarget(
      name: "MetalImageRuntimeBenchmark",
      dependencies: ["MetalImageRuntime", "MetalImageFFT"],
      path: "native/swift/Benchmarks/MetalImageRuntimeBenchmark"
    ),
    .executableTarget(
      name: "MetalSSBBenchmark",
      dependencies: ["MetalSSBKernels"],
      path: "native/swift/Benchmarks/MetalSSBBenchmark"
    ),
    .executableTarget(
      name: "Metal4DSTEMBinningBenchmark",
      dependencies: ["Metal4DSTEMKernels"],
      path: "native/swift/Benchmarks/Metal4DSTEMBinningBenchmark"
    ),
    .executableTarget(
      name: "Metal4DSTEMStreamingIOBenchmark",
      dependencies: ["Metal4DSTEMStreamingIO", "Native4DSTEMIO"],
      path: "native/swift/Benchmarks/Metal4DSTEMStreamingIOBenchmark"
    ),
    .executableTarget(
      name: "MetalCompactH5Benchmark",
      dependencies: ["Metal4DSTEMStreamingIO"],
      path: "native/swift/Benchmarks/MetalCompactH5Benchmark"
    ),
    .executableTarget(
      name: "MetalOriginalHDF5Benchmark",
      dependencies: ["Native4DSTEMIO", "Metal4DSTEMStreamingIO"],
      path: "native/swift/Benchmarks/MetalOriginalHDF5Benchmark"
    ),
    .executableTarget(
      name: "QEMDetectorUpdateBenchmark",
      dependencies: ["Native4DSTEMIO", "Metal4DSTEMStreamingIO"],
      path: "native/swift/Benchmarks/QEMDetectorUpdateBenchmark"
    ),
    .executableTarget(
      name: "MetalRuntimeANSBenchmark",
      dependencies: ["Native4DSTEMIO", "Metal4DSTEMStreamingIO"],
      path: "native/swift/Benchmarks/MetalRuntimeANSBenchmark"
    ),
    .executableTarget(
      name: "MetalPairedRuntimeTANSBenchmark",
      dependencies: ["Native4DSTEMIO", "Metal4DSTEMStreamingIO"],
      path: "native/swift/Benchmarks/MetalPairedRuntimeTANSBenchmark"
    ),
    .executableTarget(
      name: "MetalPairedRuntimeTANSSeriesBenchmark",
      dependencies: ["Native4DSTEMIO", "Metal4DSTEMStreamingIO"],
      path: "native/swift/Benchmarks/MetalPairedRuntimeTANSSeriesBenchmark"
    ),
    .executableTarget(
      name: "Metal4DSTEMDPCBenchmark",
      dependencies: ["Metal4DSTEMKernels"],
      path: "native/swift/Benchmarks/Metal4DSTEMDPCBenchmark"
    ),
    .testTarget(
      name: "MetalDisplayKernelsTests",
      dependencies: ["MetalDisplayKernels"],
      path: "native/swift/Tests/MetalDisplayKernelsTests"
    ),
    .testTarget(
      name: "Metal4DSTEMKernelsTests",
      dependencies: ["Metal4DSTEMKernels"],
      path: "native/swift/Tests/Metal4DSTEMKernelsTests"
    ),
    .testTarget(
      name: "MetalImageFFTTests",
      dependencies: ["MetalImageFFT"],
      path: "native/swift/Tests/MetalImageFFTTests"
    ),
    .testTarget(
      name: "MetalImageRuntimeTests",
      dependencies: ["MetalImageRuntime"],
      path: "native/swift/Tests/MetalImageRuntimeTests"
    ),
    .testTarget(
      name: "MetalSSBKernelsTests",
      dependencies: ["MetalSSBKernels"],
      path: "native/swift/Tests/MetalSSBKernelsTests"
    ),
    .testTarget(
      name: "MetalScientificNumericsTests",
      dependencies: ["MetalScientificNumerics", "Metal4DSTEMStreamingIO"],
      path: "native/swift/Tests/MetalScientificNumericsTests",
      resources: [.copy("Fixtures")]
    ),
    .testTarget(
      name: "Native4DSTEMIOTests",
      dependencies: ["Metal4DSTEMStreamingIO", "Native4DSTEMIO", "MetalScientificNumerics"],
      path: "native/swift/Tests/Native4DSTEMIOTests",
      resources: [.copy("Fixtures")]
    ),
  ]
)
