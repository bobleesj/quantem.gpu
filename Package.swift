// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "QuantEMGPU",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "QuantEMMetalDisplay", targets: ["QuantEMMetalDisplay"]),
        .library(name: "QuantEM4DSTEMMetal", targets: ["QuantEM4DSTEMMetal"]),
        .executable(
            name: "quantem-metal-display-benchmark",
            targets: ["QuantEMMetalDisplayBenchmark"]
        ),
    ],
    targets: [
        .target(
            name: "QuantEMMetalDisplay",
            path: "src/quantem/gpu/swift/Sources/QuantEMMetalDisplay",
            resources: [.copy("Resources")]
        ),
        .target(
            name: "QuantEM4DSTEMMetal",
            path: "src/quantem/gpu/swift/Sources/QuantEM4DSTEMMetal",
            resources: [.copy("Resources")]
        ),
        .executableTarget(
            name: "QuantEMMetalDisplayBenchmark",
            dependencies: ["QuantEMMetalDisplay"],
            path: "src/quantem/gpu/swift/Benchmarks/QuantEMMetalDisplayBenchmark"
        ),
        .testTarget(
            name: "QuantEMMetalDisplayTests",
            dependencies: ["QuantEMMetalDisplay"],
            path: "src/quantem/gpu/swift/Tests/QuantEMMetalDisplayTests"
        ),
        .testTarget(
            name: "QuantEM4DSTEMMetalTests",
            dependencies: ["QuantEM4DSTEMMetal"],
            path: "src/quantem/gpu/swift/Tests/QuantEM4DSTEMMetalTests"
        ),
    ]
)
