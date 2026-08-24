@preconcurrency import Metal
import XCTest

@testable import MetalImageFFT

final class MetalImageFFTTests: XCTestCase {
  private final class ConcurrentOutputs: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [Result<[Float], Error>?] = [nil, nil]

    func set(_ result: Result<[Float], Error>, at index: Int) {
      lock.withLock { storage[index] = result }
    }

    func value(at index: Int) throws -> [Float] {
      try lock.withLock {
        guard let result = storage[index] else {
          throw NSError(
            domain: "MetalImageFFTTests",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Concurrent FFT result was not published."]
          )
        }
        return try result.get()
      }
    }
  }

  func testFloatArbitraryShapeMatchesDirectDFTAndFFTShift() throws {
    let device = try metalDevice()
    let fft = try MetalImageFFT(device: device)
    let rows = 3
    let columns = 5
    let values = (0..<(rows * columns)).map { index in
      Float((index * 7) % 11) - 4
    }
    let source = try makeBuffer(device: device, values: values)

    let result = try fft.logMagnitude(
      source: source,
      rows: rows,
      columns: columns,
      scalarType: .float32
    )
    let actual = Self.floatValues(result.buffer, count: values.count)
    var expectedMaximum: Float = 0
    for outputRow in 0..<rows {
      for outputColumn in 0..<columns {
        let frequencyRow = (outputRow + (rows + 1) / 2) % rows
        let frequencyColumn = (outputColumn + (columns + 1) / 2) % columns
        var real = 0.0
        var imaginary = 0.0
        for inputRow in 0..<rows {
          for inputColumn in 0..<columns {
            let angle =
              -2 * Double.pi
              * (Double(frequencyRow * inputRow) / Double(rows)
                + Double(frequencyColumn * inputColumn) / Double(columns))
            let value = Double(values[inputRow * columns + inputColumn])
            real += value * cos(angle)
            imaginary += value * sin(angle)
          }
        }
        let expected = Float(log1p(hypot(real, imaginary)))
        expectedMaximum = max(expectedMaximum, expected)
        XCTAssertEqual(
          actual[outputRow * columns + outputColumn],
          expected,
          accuracy: 3e-4
        )
      }
    }
    XCTAssertEqual(result.minimum, 0)
    XCTAssertEqual(result.maximum, expectedMaximum, accuracy: 3e-4)
    XCTAssertEqual(result.rows, rows)
    XCTAssertEqual(result.columns, columns)
  }

  func testPowerOfTwoImpulseIsFlatForEveryIntegerScalarType() throws {
    let device = try metalDevice()
    let fft = try MetalImageFFT(device: device)
    let rows = 4
    let columns = 8
    let expected = log1p(Float(7))

    let uint8Result = try impulseResult(
      fft: fft,
      device: device,
      values: [UInt8](repeating: 0, count: rows * columns),
      impulse: 7,
      rows: rows,
      columns: columns,
      scalarType: .uint8
    )
    let uint16Result = try impulseResult(
      fft: fft,
      device: device,
      values: [UInt16](repeating: 0, count: rows * columns),
      impulse: 7,
      rows: rows,
      columns: columns,
      scalarType: .uint16
    )
    let uint32Result = try impulseResult(
      fft: fft,
      device: device,
      values: [UInt32](repeating: 0, count: rows * columns),
      impulse: 7,
      rows: rows,
      columns: columns,
      scalarType: .uint32
    )

    for result in [uint8Result, uint16Result, uint32Result] {
      for value in Self.floatValues(result.buffer, count: rows * columns) {
        XCTAssertEqual(value, expected, accuracy: 2e-5)
      }
      XCTAssertEqual(result.maximum, expected, accuracy: 2e-5)
    }
  }

  func testNonfiniteFloatInputsBecomeZeroWithoutChangingSource() throws {
    let device = try metalDevice()
    let fft = try MetalImageFFT(device: device)
    let values: [Float] = [1, .nan, 2, .infinity]
    let source = try makeBuffer(device: device, values: values)

    _ = try fft.logMagnitude(
      source: source,
      rows: 2,
      columns: 2,
      scalarType: .float32
    )

    let unchanged = Self.floatValues(source, count: values.count)
    XCTAssertEqual(unchanged[0], 1)
    XCTAssertTrue(unchanged[1].isNaN)
    XCTAssertEqual(unchanged[2], 2)
    XCTAssertTrue(unchanged[3].isInfinite)
  }

  func testInPlaceOutputReusesDestinationAndWarmTransformStaysInteractive() throws {
    let device = try metalDevice()
    let fft = try MetalImageFFT(device: device)
    let rows = 256
    let columns = 256
    let values = (0..<(rows * columns)).map { index -> Float in
      let row = index / columns
      let column = index - row * columns
      return sin(Float(row) * 0.11) + cos(Float(column) * 0.07)
    }
    let source = try makeBuffer(device: device, values: values)
    let output = try XCTUnwrap(
      device.makeBuffer(
        length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared
      )
    )

    let first = try fft.logMagnitude(
      source: source,
      rows: rows,
      columns: columns,
      scalarType: .float32,
      output: output
    )
    XCTAssertTrue(first.buffer === output)

    var warm: [Double] = []
    for _ in 0..<8 {
      let started = CFAbsoluteTimeGetCurrent()
      let result = try fft.logMagnitude(
        source: source,
        rows: rows,
        columns: columns,
        scalarType: .float32,
        output: output
      )
      warm.append((CFAbsoluteTimeGetCurrent() - started) * 1_000)
      XCTAssertTrue(result.buffer === output)
      XCTAssertEqual(result.maximum, first.maximum, accuracy: 1e-4)
    }
    let median = warm.sorted()[warm.count / 2]
    XCTAssertLessThan(
      median,
      16.0,
      "Warm 256×256 FFT should stay inside a 60 Hz frame, but p50 was \(median) ms"
    )
  }

  func testConcurrentTransformsDoNotOverwriteSharedShapeWorkspace() throws {
    let device = try metalDevice()
    let fft = try MetalImageFFT(device: device)
    let rows = 256
    let columns = 256
    let count = rows * columns
    let firstValues = (0..<count).map { index -> Float in
      let row = index / columns
      let column = index - row * columns
      return sin(Float(row) * 0.031) + cos(Float(column) * 0.071)
    }
    let secondValues = (0..<count).map { index -> Float in
      let row = index / columns
      let column = index - row * columns
      return 5 * sin(Float(row) * 0.113) - 3 * cos(Float(column) * 0.047)
    }
    let sources = [
      try makeBuffer(device: device, values: firstValues),
      try makeBuffer(device: device, values: secondValues),
    ]
    let expected = try sources.map { source in
      let reference = try MetalImageFFT(device: device)
      let result = try reference.logMagnitude(
        source: source,
        rows: rows,
        columns: columns,
        scalarType: .float32
      )
      return Self.floatValues(result.buffer, count: count)
    }

    for _ in 0..<20 {
      let gate = DispatchSemaphore(value: 0)
      let group = DispatchGroup()
      let outputs = ConcurrentOutputs()
      for index in sources.indices {
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
          gate.wait()
          outputs.set(
            Result {
              let result = try fft.logMagnitude(
                source: sources[index],
                rows: rows,
                columns: columns,
                scalarType: .float32
              )
              return Self.floatValues(result.buffer, count: count)
            },
            at: index
          )
          group.leave()
        }
      }
      gate.signal()
      gate.signal()
      XCTAssertEqual(group.wait(timeout: .now() + 10), .success)

      for index in sources.indices {
        let actual = try outputs.value(at: index)
        XCTAssertEqual(actual.count, expected[index].count)
        for sample in stride(from: 0, to: count, by: 257) {
          XCTAssertEqual(
            actual[sample],
            expected[index][sample],
            accuracy: 2e-4,
            "Concurrent FFT \(index) was overwritten at sample \(sample)."
          )
        }
      }
    }
  }

  func testWarm512UInt32BrightFieldFFTStaysInside120Hz() throws {
    let device = try metalDevice()
    let fft = try MetalImageFFT(device: device)
    let rows = 512
    let columns = 512
    let values = (0..<(rows * columns)).map { UInt32(($0 * 13) % 1024) }
    let source = try makeBuffer(device: device, values: values)
    let output = try XCTUnwrap(
      device.makeBuffer(
        length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared
      )
    )
    try fft.prewarm(rows: rows, columns: columns)
    _ = try fft.logMagnitude(
      source: source,
      rows: rows,
      columns: columns,
      scalarType: .uint32,
      output: output
    )
    var warm: [Double] = []
    for _ in 0..<8 {
      let started = CFAbsoluteTimeGetCurrent()
      _ = try fft.logMagnitude(
        source: source,
        rows: rows,
        columns: columns,
        scalarType: .uint32,
        output: output
      )
      warm.append((CFAbsoluteTimeGetCurrent() - started) * 1_000)
    }
    let median = warm.sorted()[warm.count / 2]
    XCTAssertLessThan(
      median,
      8.33,
      "Warm 512×512 uint32 BF/ADF FFT must stay inside 120 Hz, but p50 was \(median) ms"
    )
  }

  func testWarm2048StaysAheadOfTypicalTorchMPSDisplayFFT() throws {
    try XCTSkipUnless(
      ProcessInfo.processInfo.environment["QUANTEM_GPU_ENFORCE_METAL_BENCHMARKS"] == "1",
      "Set QUANTEM_GPU_ENFORCE_METAL_BENCHMARKS=1 for device-qualified performance gates"
    )
    let device = try metalDevice()
    let fft = try MetalImageFFT(device: device)
    let rows = 2048
    let columns = 2048
    let count = rows * columns
    let values = (0..<count).map { index -> Float in
      let row = index / columns
      return sin(Float(row) * 0.017)
    }
    let source = try makeBuffer(device: device, values: values)
    let output = try XCTUnwrap(
      device.makeBuffer(
        length: count * MemoryLayout<Float>.stride,
        options: .storageModeShared
      )
    )
    try fft.prewarm(rows: rows, columns: columns)
    _ = try fft.logMagnitude(
      source: source,
      rows: rows,
      columns: columns,
      scalarType: .float32,
      output: output
    )
    var warm: [Double] = []
    for _ in 0..<5 {
      let started = CFAbsoluteTimeGetCurrent()
      _ = try fft.logMagnitude(
        source: source,
        rows: rows,
        columns: columns,
        scalarType: .float32,
        output: output
      )
      warm.append((CFAbsoluteTimeGetCurrent() - started) * 1_000)
    }
    let median = warm.sorted()[warm.count / 2]
    XCTAssertLessThan(
      median,
      2.2,
      "Warm 2048×2048 Metal FFT should beat torch MPS display FFT (~2.3 ms on M5 Max), but p50 was \(median) ms"
    )
  }

  func testRejectsUndersizedInputWithCorrectiveSizes() throws {
    let device = try metalDevice()
    let fft = try MetalImageFFT(device: device)
    let source = try XCTUnwrap(device.makeBuffer(length: 12, options: .storageModeShared))

    XCTAssertThrowsError(
      try fft.logMagnitude(
        source: source,
        rows: 2,
        columns: 2,
        scalarType: .float32
      )
    ) { error in
      guard case MetalImageFFTError.inputBufferTooSmall(let required, let actual) = error
      else { return XCTFail("Unexpected error: \(error)") }
      XCTAssertEqual(required, 16)
      XCTAssertEqual(actual, 12)
    }
  }

  private func impulseResult<T: FixedWidthInteger>(
    fft: MetalImageFFT,
    device: MTLDevice,
    values: [T],
    impulse: T,
    rows: Int,
    columns: Int,
    scalarType: MetalImageFFTScalarType
  ) throws -> MetalImageFFTResult {
    var values = values
    values[columns + 3] = impulse
    return try fft.logMagnitude(
      source: makeBuffer(device: device, values: values),
      rows: rows,
      columns: columns,
      scalarType: scalarType
    )
  }

  private func metalDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Metal is unavailable on this runner")
    }
    return device
  }

  private func makeBuffer<T>(device: MTLDevice, values: [T]) throws -> MTLBuffer {
    let length = values.count * MemoryLayout<T>.stride
    return try values.withUnsafeBytes { bytes in
      try XCTUnwrap(
        device.makeBuffer(bytes: bytes.baseAddress!, length: length, options: .storageModeShared)
      )
    }
  }

  private static func floatValues(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
  }
}
