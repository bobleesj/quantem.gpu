import CryptoKit
import Foundation
import Metal
import MetalSSBKernels

private func require(_ condition: @autoclosure () -> Bool, _ message: String) {
  if !condition() {
    FileHandle.standardError.write(Data("FAIL: \(message)\n".utf8))
    exit(1)
  }
}

private struct Candidate: Decodable {
  let c10: Float
  let c12: Float
  let phi12: Float
  let rotation: Float
  let loss: Double
  let cudaLoss: Double
  let cudaObjectRelativeL2: Double
  var aberrations: MetalSSBAberrations {
    MetalSSBAberrations(c10Nanometers: c10, c12Nanometers: c12, phi12Radians: phi12)
  }
}

private struct Reference: Decodable {
  let size: Int
  let brightfieldCount: Int
  let cases: [Candidate]
  let cudaRevision: String
  let filesSHA256: [String: String]
}

private func complexValues(_ buffer: MTLBuffer, count: Int) -> [SIMD2<Float>] {
  Array(UnsafeBufferPointer(
    start: buffer.contents().assumingMemoryBound(to: SIMD2<Float>.self), count: count))
}

private func relativeError(_ values: [SIMD2<Float>], _ expected: [SIMD2<Float>]) -> Double {
  var numerator = 0.0
  var denominator = 0.0
  for (value, reference) in zip(values, expected) {
    let delta = value - reference
    numerator += Double(delta.x) * Double(delta.x) + Double(delta.y) * Double(delta.y)
    denominator += Double(reference.x) * Double(reference.x) + Double(reference.y) * Double(reference.y)
  }
  return sqrt(numerator / max(denominator, 1e-30))
}

private func timing(_ label: String, _ seconds: [Double]) {
  let sorted = seconds.sorted().map { $0 * 1000 }
  let mean = sorted.reduce(0, +) / Double(sorted.count)
  print(String(format: "%@ mean=%.3f p50=%.3f p95=%.3f ms", label, mean,
    sorted[sorted.count / 2], sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]))
}

@main enum SSBScanSizeCheck {
  static func main() throws {
    guard CommandLine.arguments.count == 2, let device = MTLCreateSystemDefaultDevice() else {
      fatalError("Usage: ssb-scan-size-check <exported-reference-directory>; requires Apple Metal")
    }
    print("Metal device: \(device.name); deterministic native-count controls, not a real-data benchmark")
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    for size in [128, 256, 512] {
      let folder = root.appendingPathComponent(String(size))
      func data(_ name: String) throws -> Data { try Data(contentsOf: folder.appendingPathComponent(name)) }
      let reference = try JSONDecoder().decode(Reference.self, from: data("reference.json"))
      require(reference.size == size, "Reference size mismatch")
      for (name, expectedHash) in reference.filesSHA256 {
        let digest = SHA256.hash(data: try data(name)).map { String(format: "%02x", $0) }.joined()
        require(digest == expectedHash, "Reference checksum failed: \(name)")
      }
      let geometry = try JSONDecoder().decode(MetalSSBGeometry.self, from: data("geometry.json"))
      let counts = try data("counts.uint8")
      require(counts.count == reference.brightfieldCount * size * size, "Count shape mismatch")
      let source = counts.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
      }
      print("\(size)x\(size), full disk \(reference.brightfieldCount) BF, CUDA \(reference.cudaRevision)")
      // Calibration follows the native scan without silently changing sampling.
      let calibration = MetalSSBCalibration(beamEnergyKeV: 300, semiangleMrad: 30,
        scanStepRowAngstroms: 2, scanStepColumnAngstroms: 3,
        detectorStepRowMrad: 10, detectorStepColumnMrad: 10, centerRow: 4, centerColumn: 4)
      let calibrated = try calibration.geometry(detectorRows: 9, detectorColumns: 9,
        detectorSum: Array(repeating: 100, count: 81), scanRows: size, scanColumns: size)
      require(calibrated.geometry.qxByRow.count == size, "Calibration native size")
      require(abs(calibrated.geometry.qyByColumn[1] - 1 / Float(size * 3)) < 1e-8,
        "Calibration preserves column sampling")
      for budget: Int? in [nil, 0, 32 * size * (size / 2 + 1) * 8] {
        let mode = budget == nil ? "cached" : budget == 0 ? "streamed" : "hybrid"
        let engine = try MetalSSBEngine(device: device, geometry: geometry, cacheBudgetBytes: budget)
        let preparation = Date()
        try engine.prepare(brightfield: source)
        print(String(format: "%@ prepare=%.3f ms", mode, -preparation.timeIntervalSinceNow * 1000))
        for (index, candidate) in reference.cases.enumerated() {
          let expectedData = try data("object-\(index).complex64")
          let expected = expectedData.withUnsafeBytes {
            Array($0.bindMemory(to: SIMD2<Float>.self))
          }
          let result = try engine.reconstruct(aberrations: candidate.aberrations,
            rotationDegrees: candidate.rotation)
          require(result.provenance.scanRows == size && result.provenance.scanColumns == size,
            "Native result provenance")
          require(result.provenance.logicalBrightfieldCount == reference.brightfieldCount,
            "All BF evidence retained")
          let error = relativeError(complexValues(result.object, count: size * size), expected)
          require(error < 1e-4, "\(size) \(mode) candidate \(index): object relative L2 \(error)")
          let phases = try engine.phase(of: result)
          let actualPhase = phases.contents().assumingMemoryBound(to: Float.self)
          var maxPhaseError: Float = 0
          for pixel in expected.indices {
            let angle = atan2(expected[pixel].y, expected[pixel].x)
            let delta = actualPhase[pixel] - angle
            maxPhaseError = max(maxPhaseError, abs(atan2(sin(delta), cos(delta))))
          }
          require(maxPhaseError < 2e-4, "\(size) \(mode) phase error \(maxPhaseError)")
          let loss = try engine.phaseVariance(aberrations: candidate.aberrations,
            rotationDegrees: candidate.rotation)
          let lossError = abs(Double(loss.loss) - candidate.loss)
          print("\(mode) candidate=\(index) objectL2=\(error) phaseMax=\(maxPhaseError) lossError=\(lossError)")
          require(lossError <= 1e-6 + 1e-5 * abs(candidate.loss),
            "\(size) \(mode) loss \(loss.loss), expected \(candidate.loss); do not rewrite the reference")
          // Alternate loss and object to exercise cache-layout transitions.
          let restored = try engine.reconstruct(aberrations: candidate.aberrations,
            rotationDegrees: candidate.rotation)
          require(relativeError(complexValues(restored.object, count: size * size), expected) < 1e-4,
            "Reconstruct after loss")
          let saved = try MetalSSBSavedRun(result: result, sourceIdentity: "scan-\(size)",
            backendRevision: "test", geometry: geometry, aberrations: candidate.aberrations,
            rotationDegrees: candidate.rotation)
          let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
          defer { try? FileManager.default.removeItem(at: temporary) }
          try saved.save(to: temporary)
          let loaded = try MetalSSBSavedRun.load(from: temporary, matchingSourceIdentity: "scan-\(size)")
          let reopened = try loaded.reconstruction(device: device)
          require(relativeError(complexValues(reopened.object, count: size * size),
            complexValues(result.object, count: size * size)) == 0, "Saved result exact roundtrip")
        }
        if budget == nil {
          let candidate = reference.cases[1]
          if size < 512 {
            let referenceObject = complexValues(try engine.reconstruct(
              aberrations: candidate.aberrations).object, count: size * size)
            // The native source width is not narrowed for smaller scans.
            for countType in [MetalSSBCountType.uint16, .uint32] {
              let wide = device.makeBuffer(length: counts.count * countType.byteWidth,
                options: .storageModeShared)!
              if countType == .uint16 {
                let pointer = wide.contents().assumingMemoryBound(to: UInt16.self)
                for (index, value) in counts.enumerated() { pointer[index] = UInt16(value) }
              } else {
                let pointer = wide.contents().assumingMemoryBound(to: UInt32.self)
                for (index, value) in counts.enumerated() { pointer[index] = UInt32(value) }
              }
              try engine.prepare(brightfield: wide, countType: countType)
              let wideResult = try engine.reconstruct(aberrations: candidate.aberrations)
              require(relativeError(complexValues(wideResult.object, count: size * size), referenceObject) == 0,
                "Equal native counts must produce identical \(countType) output at \(size)")
            }
            try engine.prepare(brightfield: source)
            let streamed = try MetalSSBEngine(device: device, geometry: geometry, cacheBudgetBytes: 0)
            try streamed.prepare(brightfield: source)
            for prototype in MetalSSBHigherOrder.supported {
              var term = prototype
              term.magnitudeNanometers = 0.5
              term.angleRadians = 0.17
              var higher = candidate.aberrations
              higher.higherOrder = [term]
              let cached = try engine.reconstruct(aberrations: higher)
              let direct = try streamed.reconstruct(aberrations: higher)
              require(relativeError(complexValues(cached.object, count: size * size),
                complexValues(direct.object, count: size * size)) < 1e-4,
                "\(size) higher-order \(term.name) cached/streamed parity")
            }
            let fit = try engine.optimize(start: candidate.aberrations,
              rotationDegrees: 0, globalTrials: 200)
            require(fit.loss.isFinite, "Native-size 200-trial optimizer")
            print("\(size) 200-trial fit wall=\(fit.elapsedSeconds) seconds; refine=\(fit.refinementEvaluations)")
          }
          var redraw: [Double] = [], lossTimes: [Double] = []
          // Time separately: switching row/column layouts is not part of a
          // sustained drag, but is still covered by the parity checks above.
          for step in 0..<23 {
            let value = try engine.reconstruct(aberrations: candidate.aberrations)
            if step >= 3 { redraw.append(value.wallSeconds) }
          }
          for step in 0..<23 {
            let value = try engine.phaseVariance(aberrations: candidate.aberrations)
            if step >= 3 { lossTimes.append(value.wallSeconds) }
          }
          timing("\(size) cached object", redraw)
          timing("\(size) cached objective", lossTimes)
        }
      }
    }
    print("PASS: native 128/256/512 CUDA + independent equation parity, saved results, cache modes")
  }
}
