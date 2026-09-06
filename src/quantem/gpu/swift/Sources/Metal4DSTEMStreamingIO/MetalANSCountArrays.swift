import Foundation
import Metal4DSTEMKernels

/// Advanced in-memory boundary for the shared exact count-ANS contract.
///
/// Streams are ordered by `(scan block, detector row, detector column)`.
/// Each literal stream contains little-endian uint16 values, even when the
/// declared logical dtype is uint8. This type does not define a file format.
public struct MetalANSCountArrays: Sendable {
  public let shape: [Int]
  public let blockFrames: Int
  public let scale: Int
  public let logicalDtype: Metal4DSTEMIntegerDType
  public let payload: [UInt8]
  public let offsets: [UInt64]
  public let modelIDs: [UInt32]
  public let contextOffsets: [UInt32]
  public let symbols: [UInt16]
  public let cumulative: [UInt16]
  public let frequencies: [UInt16]
  public let literal: [UInt8]

  public init(
    shape: [Int], blockFrames: Int, scale: Int,
    logicalDtype: Metal4DSTEMIntegerDType,
    payload: [UInt8], offsets: [UInt64], modelIDs: [UInt32],
    contextOffsets: [UInt32], symbols: [UInt16], cumulative: [UInt16],
    frequencies: [UInt16], literal: [UInt8]
  ) throws {
    self.shape = shape
    self.blockFrames = blockFrames
    self.scale = scale
    self.logicalDtype = logicalDtype
    self.payload = payload
    self.offsets = offsets
    self.modelIDs = modelIDs
    self.contextOffsets = contextOffsets
    self.symbols = symbols
    self.cumulative = cumulative
    self.frequencies = frequencies
    self.literal = literal
    try validateTables()
  }

  public var scanCount: Int { shape[0] * shape[1] }
  public var detectorPixelCount: Int { shape[2] * shape[3] }
  public var blockCount: Int { (scanCount - 1) / blockFrames + 1 }

  private func validateTables() throws {
    func require(_ condition: Bool, _ reason: String) throws {
      guard condition else {
        throw Metal4DSTEMStreamingIOError.invalidRequest("Exact ANS counts: " + reason)
      }
    }
    try require(
      shape.count == 4 && shape.allSatisfy { $0 > 0 }, "require four positive dimensions.")
    try require(
      (1...Int(UInt32.max)).contains(blockFrames) && (1...15).contains(scale),
      "blockFrames must fit uint32 and scale must be 1 through 15.")
    try require(
      logicalDtype == .uint8 || logicalDtype == .uint16, "require native uint8 or uint16.")
    var values = 1
    for dimension in shape {
      let product = values.multipliedReportingOverflow(by: dimension)
      try require(!product.overflow, "logical shape exceeds the supported integer range.")
      values = product.partialValue
    }
    try require(detectorPixelCount <= Int(UInt32.max), "detector count exceeds the dispatch range.")
    let streamProduct = blockCount.multipliedReportingOverflow(by: detectorPixelCount)
    try require(
      !streamProduct.overflow && streamProduct.partialValue <= Int(UInt32.max),
      "stream count exceeds the current native validation dispatch range.")
    let streamCount = streamProduct.partialValue
    try require(
      offsets.count == streamCount + 1 && modelIDs.count == streamCount,
      "stream offsets and selectors disagree with the declared geometry.")
    try require(
      offsets.first == 0 && offsets.last == UInt64(payload.count),
      "stream offsets must cover the complete payload exactly once.")
    for index in 1..<offsets.count {
      try require(offsets[index] >= offsets[index - 1], "stream offsets are not ordered.")
    }
    try require(
      !literal.isEmpty && literal.count <= Int(UInt32.max)
        && contextOffsets.count == literal.count + 1,
      "each model needs one literal flag and context interval.")
    try require(
      symbols.count <= Int(UInt32.max) && symbols.count == cumulative.count
        && symbols.count == frequencies.count,
      "symbol, cumulative, and frequency table lengths disagree.")
    try require(
      contextOffsets.first == 0 && contextOffsets.last == UInt32(symbols.count),
      "context offsets must cover the model tables exactly once.")
    for model in literal.indices {
      try require(literal[model] <= 1, "literal flags must be zero or one.")
      let first = Int(contextOffsets[model])
      let stop = Int(contextOffsets[model + 1])
      try require(first <= stop && stop <= symbols.count, "context interval is out of bounds.")
      if literal[model] == 1 {
        try require(first == stop, "literal models cannot contain entropy entries.")
      } else {
        try require(first < stop, "entropy models cannot be empty.")
        var total = 0
        for symbol in first..<stop {
          try require(
            frequencies[symbol] > 0 && Int(cumulative[symbol]) == total,
            "positive frequencies and cumulative starts must partition every model slot.")
          if symbol > first {
            try require(
              symbols[symbol] > symbols[symbol - 1], "model symbols must be unique and sorted.")
          }
          total += Int(frequencies[symbol])
          try require(total <= 1 << scale, "model frequencies exceed the declared scale.")
        }
        try require(total == 1 << scale, "model frequencies do not cover the declared scale.")
      }
    }
    for stream in modelIDs.indices {
      let model = Int(modelIDs[stream])
      try require(model < literal.count, "stream model selector is out of bounds.")
      let frames = min(blockFrames, scanCount - (stream / detectorPixelCount) * blockFrames)
      let bytes = offsets[stream + 1] - offsets[stream]
      try require(
        literal[model] == 1 ? bytes == UInt64(frames) * 2 : bytes >= 4,
        "stream length cannot represent its declared scan block.")
    }
  }
}
