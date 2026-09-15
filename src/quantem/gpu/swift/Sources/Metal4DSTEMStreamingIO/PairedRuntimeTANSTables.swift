import CryptoKit
import Foundation

/// Deterministic tables for the exact paired runtime tANS representation.
///
/// This builder is intentionally independent of the Metal encoder and HDF5
/// loader. Its hashes are part of the `metal-runtime-paired-tans-v1` ABI.
struct PairedRuntimeTANSTables: Sendable {
  static let modelCount = 32
  static let stateCount = 1_024
  static let symbolCount = 1_089
  static let escapeSymbol = 1_088

  let frequencies: [UInt32]
  let encoding: [UInt16]
  let packedDecoding: [UInt32]
  let frequencySHA256: String
  let encodingSHA256: String
  let packedDecodingSHA256: String

  static func build() throws -> Self {
    let frequencies = buildFrequencies()
    let tables = try buildTransitions(frequencies: frequencies)
    return Self(
      frequencies: frequencies,
      encoding: tables.encoding,
      packedDecoding: tables.decoding,
      frequencySHA256: sha256LittleEndian(frequencies),
      encodingSHA256: sha256LittleEndian(tables.encoding),
      packedDecodingSHA256: sha256LittleEndian(tables.decoding))
  }

  /// Exhaustively verify that every supported encoder transition is inverted
  /// by the packed decoder table.
  func validateTransitionInversion() throws -> Int {
    guard frequencies.count == Self.modelCount * Self.symbolCount,
      encoding.count == Self.modelCount * Self.stateCount,
      packedDecoding.count == Self.modelCount * Self.stateCount
    else {
      throw PairedRuntimeTANSTableError.invalidExtent
    }

    var checked = 0
    for model in 0..<Self.modelCount {
      let frequencyBase = model * Self.symbolCount
      let tableBase = model * Self.stateCount
      var start = 0
      for symbol in 0..<Self.symbolCount {
        let frequency = Int(frequencies[frequencyBase + symbol])
        defer { start += frequency }
        guard frequency > 0 else { continue }
        let expectedPair =
          symbol == Self.escapeSymbol
          ? UInt32(4_095)
          : UInt32(symbol / 33) | (UInt32(symbol % 33) << 6)
        let leadingBit = Int.bitWidth - frequency.leadingZeroBitCount - 1
        for oldState in 0..<Self.stateCount {
          let y = Self.stateCount + oldState
          var bits = 10 - leadingBit
          if y < frequency << bits { bits -= 1 }
          let rank = (y >> bits) - frequency
          let low = bits == 0 ? 0 : y & ((1 << bits) - 1)
          let newState = Int(encoding[tableBase + start + rank])
          let code = packedDecoding[tableBase + newState]
          guard code & 4_095 == expectedPair,
            Int((code >> 12) & 15) == bits,
            Int(code >> 16) + low == oldState
          else {
            throw PairedRuntimeTANSTableError.nonInvertibleTransition(
              model: model, symbol: symbol, state: oldState)
          }
          checked += 1
        }
      }
      guard start == Self.stateCount else {
        throw PairedRuntimeTANSTableError.invalidFrequencyTotal(model: model, total: start)
      }
    }
    return checked
  }

  private static func buildFrequencies() -> [UInt32] {
    let choices = [4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768]
    let lowerMean = 0.002
    let upperMean = 32.0
    let logLower = log(lowerMean)
    let logUpper = log(upperMean)
    var logFactorial = [Double](repeating: 0, count: 33)
    for value in 1..<33 {
      logFactorial[value] = logFactorial[value - 1] + log(Double(value))
    }

    var result = [UInt32](repeating: 0, count: modelCount * symbolCount)
    for model in 0..<modelCount {
      let interpolation = Double(model) / Double(modelCount - 1)
      let mean = exp(logLower + interpolation * (logUpper - logLower))
      let logMean = log(mean)
      var probability = [Double](repeating: 0, count: 33)
      for value in 0..<33 {
        probability[value] = exp(Double(value) * logMean - logFactorial[value] - mean)
      }
      var joint = [Double](repeating: 0, count: symbolCount)
      for a in 0..<32 {
        for b in 0..<32 {
          joint[a * 33 + b] = probability[a] * probability[b]
        }
      }
      let jointRanks = descendingRanks(joint)

      var bestCost = Double.infinity
      var bestFrequency = [UInt32](repeating: 0, count: symbolCount)
      for choice in choices {
        var supported = jointRanks.map { $0 < choice }
        supported[escapeSymbol] = true
        var selected = [Double](repeating: 0, count: symbolCount)
        var selectedTotal = 0.0
        var supportedCount = 0
        for symbol in 0..<symbolCount where supported[symbol] {
          selected[symbol] = joint[symbol]
          selectedTotal += selected[symbol]
          supportedCount += 1
        }
        selected[escapeSymbol] = max(0, 1 - selectedTotal)

        let available = Double(stateCount - supportedCount)
        var allocation = [Double](repeating: 0, count: symbolCount)
        var frequency = [UInt32](repeating: 0, count: symbolCount)
        var allocated = 0
        for symbol in 0..<symbolCount where supported[symbol] {
          allocation[symbol] = selected[symbol] * available
          frequency[symbol] = UInt32(floor(allocation[symbol])) + 1
          allocated += Int(frequency[symbol])
        }
        let remainder = stateCount - allocated
        var fractions = [Double](repeating: -.infinity, count: symbolCount)
        for symbol in 0..<symbolCount where supported[symbol] {
          fractions[symbol] = allocation[symbol] - floor(allocation[symbol])
        }
        let fractionRanks = descendingRanks(fractions)
        for symbol in 0..<symbolCount where fractionRanks[symbol] < remainder {
          frequency[symbol] += 1
        }

        var cost = 13 * selected[escapeSymbol]
        for symbol in 0..<symbolCount {
          cost += selected[symbol] * log2(Double(stateCount) / Double(max(frequency[symbol], 1)))
        }
        if cost < bestCost {
          bestCost = cost
          bestFrequency = frequency
        }
      }
      let total = bestFrequency.reduce(UInt64(0)) { $0 + UInt64($1) }
      precondition(total == UInt64(stateCount), "Every paired tANS model must contain 1024 states")
      result.replaceSubrange(
        (model * symbolCount)..<((model + 1) * symbolCount), with: bestFrequency)
    }
    return result
  }

  private static func buildTransitions(
    frequencies: [UInt32]
  ) throws -> (encoding: [UInt16], decoding: [UInt32]) {
    guard frequencies.count == modelCount * symbolCount else {
      throw PairedRuntimeTANSTableError.invalidExtent
    }
    var encoding = [UInt16](repeating: 0, count: modelCount * stateCount)
    var decoding = [UInt32](repeating: UInt32.max, count: modelCount * stateCount)
    for model in 0..<modelCount {
      let frequencyBase = model * symbolCount
      let tableBase = model * stateCount
      var start = 0
      for symbol in 0..<symbolCount {
        let frequency = Int(frequencies[frequencyBase + symbol])
        defer { start += frequency }
        guard frequency > 0 else { continue }
        var rank = 0
        for state in 0..<stateCount {
          let sequence = (state * 43) & (stateCount - 1)
          guard start <= sequence, sequence < start + frequency else { continue }
          let n = frequency + rank
          let leadingBit = Int.bitWidth - n.leadingZeroBitCount - 1
          let bits = 10 - leadingBit
          let base = (n << bits) - stateCount
          let pair =
            symbol == escapeSymbol
            ? UInt32(4_095)
            : UInt32(symbol / 33) | (UInt32(symbol % 33) << 6)
          decoding[tableBase + state] = pair | (UInt32(bits) << 12) | (UInt32(base) << 16)
          encoding[tableBase + start + rank] = UInt16(state)
          rank += 1
        }
        guard rank == frequency else {
          throw PairedRuntimeTANSTableError.unassignedEncodingRank(
            model: model, symbol: symbol)
        }
      }
      guard start == stateCount else {
        throw PairedRuntimeTANSTableError.invalidFrequencyTotal(model: model, total: start)
      }
      guard !decoding[tableBase..<(tableBase + stateCount)].contains(UInt32.max) else {
        throw PairedRuntimeTANSTableError.unassignedDecodingState(model: model)
      }
    }
    return (encoding, decoding)
  }

  private static func descendingRanks(_ values: [Double]) -> [Int] {
    let order = values.indices.sorted {
      if values[$0] == values[$1] { return $0 < $1 }
      return values[$0] > values[$1]
    }
    var ranks = [Int](repeating: 0, count: values.count)
    for (rank, index) in order.enumerated() { ranks[index] = rank }
    return ranks
  }

  private static func sha256LittleEndian<T: FixedWidthInteger>(_ values: [T]) -> String {
    var digest = SHA256()
    for value in values {
      var littleEndian = value.littleEndian
      withUnsafeBytes(of: &littleEndian) { digest.update(bufferPointer: $0) }
    }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

enum PairedRuntimeTANSTableError: Error, Equatable {
  case invalidExtent
  case invalidFrequencyTotal(model: Int, total: Int)
  case unassignedEncodingRank(model: Int, symbol: Int)
  case unassignedDecodingState(model: Int)
  case nonInvertibleTransition(model: Int, symbol: Int, state: Int)
}
