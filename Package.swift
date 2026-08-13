// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "QuantEMGPU",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "QuantEMMetalDisplay", targets: ["QuantEMMetalDisplay"]),
        .executable(
            name: "quantem-metal-display-benchmark",
            targets: ["QuantEMMetalDisplayBenchmark"]
        ),
    ],
    targets: [
        .target(
            name: "QuantEMMetalDisplay",
            path: "src/quantem/gpu/display/swift/Sources/QuantEMMetalDisplay",
            resources: [.copy("Resources")]
        ),
        .executableTarget(
            name: "QuantEMMetalDisplayBenchmark",
            dependencies: ["QuantEMMetalDisplay"],
            path: "src/quantem/gpu/display/swift/Benchmarks/QuantEMMetalDisplayBenchmark"
        ),
        .testTarget(
            name: "QuantEMMetalDisplayTests",
            dependencies: ["QuantEMMetalDisplay"],
            path: "src/quantem/gpu/display/swift/Tests/QuantEMMetalDisplayTests"
        ),
    ]
)
