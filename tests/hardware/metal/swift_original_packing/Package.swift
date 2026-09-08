// swift-tools-version: 6.0
import Foundation
import PackageDescription

let package = Package(
  name: "OriginalPackingParity",
  platforms: [.macOS(.v15)],
  dependencies: [
    .package(
      name: "quantem.gpu",
      path:
        ProcessInfo.processInfo.environment["QGPU_SOURCE_ROOT"] ?? "../../../..")
  ],
  targets: [
    .executableTarget(
      name: "EMPADSourceParity",
      dependencies: [
        .product(name: "Native4DSTEMIO", package: "quantem.gpu"),
        .product(name: "Metal4DSTEMStreamingIO", package: "quantem.gpu"),
      ]
    ),
    .executableTarget(
      name: "OriginalZeroTailParity",
      dependencies: [.product(name: "Metal4DSTEMKernels", package: "quantem.gpu")],
      swiftSettings: [.swiftLanguageMode(.v5)]
    ),
    .executableTarget(
      name: "DisplayRangeParity",
      dependencies: [.product(name: "MetalDisplayKernels", package: "quantem.gpu")],
      swiftSettings: [.swiftLanguageMode(.v5)]
    ),
    .executableTarget(
      name: "PackingPlanParity",
      dependencies: [
        .product(name: "Native4DSTEMIO", package: "quantem.gpu"),
        .product(name: "Metal4DSTEMStreamingIO", package: "quantem.gpu"),
      ],
      swiftSettings: [.swiftLanguageMode(.v5)]
    ),
    .executableTarget(
      name: "OriginalDecoderParity",
      dependencies: [.product(name: "Metal4DSTEMKernels", package: "quantem.gpu")],
      swiftSettings: [.swiftLanguageMode(.v5)]
    ),
    .executableTarget(
      name: "OriginalPackingParity",
      dependencies: [
        .product(name: "Native4DSTEMIO", package: "quantem.gpu"),
        .product(name: "Metal4DSTEMStreamingIO", package: "quantem.gpu"),
      ],
      swiftSettings: [.swiftLanguageMode(.v5)]
    ),
  ]
)
