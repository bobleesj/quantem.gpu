// swift-tools-version: 6.0
import Foundation
import PackageDescription

let package = Package(
  name: "OriginalWindowMomentsParity",
  platforms: [.macOS(.v15)],
  dependencies: [
    .package(name: "quantem.gpu",
             path: ProcessInfo.processInfo.environment["QGPU_SOURCE_ROOT"] ?? "../../../..")
  ],
  targets: [
    .executableTarget(
      name: "OriginalWindowMomentsParity",
      dependencies: [.product(name: "Native4DSTEMIO", package: "quantem.gpu"),
                     .product(name: "Metal4DSTEMStreamingIO", package: "quantem.gpu")],
      swiftSettings: [.swiftLanguageMode(.v5)]
    )
  ]
)
