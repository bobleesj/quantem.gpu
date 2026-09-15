import Foundation

/// Small-lookahead table for collapsing ordinary paired-tANS transitions.
///
/// The returned `[UInt32]` is interleaved per model:
///
/// ```text
/// ordinary decoding[1024] UInt32
/// macro[state: 0..<1024][lookahead: 0..<(1 << lookaheadBits)] UInt64 as low, high UInt32
/// ```
///
/// A macro `UInt64` uses bits 0..<36 for three 12-bit decoded pairs, bits
/// 36..<46 for the terminal state, bits 46..<49 for the consumed lookahead
/// bits, and bits 49..<51 for the decoded-pair count. Unused pair fields are
/// zero. The table stops before escapes and before consuming more than the
/// configured width. Words are serialized little-endian by resident writers.
enum PairedRuntimeTANSMacroTable {
  static let modelCount = 32
  static let stateCount = 1_024
  static let maximumPairs = 3
  static let escapePair = UInt32(4_095)

  /// Build the ordinary-plus-macro decoding table for a two- or four-bit lookahead.
  static func build(decoding: [UInt32], lookaheadBits: Int = 4) throws -> [UInt32] {
    guard decoding.count == modelCount * stateCount else {
      throw PairedRuntimeTANSMacroTableError.invalidDecodingExtent(decoding.count)
    }
    guard lookaheadBits == 2 || lookaheadBits == 4 else {
      throw PairedRuntimeTANSMacroTableError.invalidLookaheadWidth(lookaheadBits)
    }
    let lookaheadCount = 1 << lookaheadBits
    let wordsPerModel = stateCount + stateCount * lookaheadCount * 2
    let byteCount = modelCount * wordsPerModel * MemoryLayout<UInt32>.stride
    var result = [UInt32]()
    result.reserveCapacity(modelCount * wordsPerModel)
    for model in 0..<modelCount {
      let ordinary = model * stateCount
      result.append(contentsOf: decoding[ordinary..<(ordinary + stateCount)])
      for state in 0..<stateCount {
        for lookahead in 0..<lookaheadCount {
          let macro = try makeMacro(
            decoding: decoding, model: model, state: state, lookahead: lookahead,
            lookaheadBits: lookaheadBits)
          result.append(UInt32(truncatingIfNeeded: macro))
          result.append(UInt32(truncatingIfNeeded: macro >> 32))
        }
      }
    }
    precondition(result.count * MemoryLayout<UInt32>.stride == byteCount)
    _ = try validate(interleaved: result, decoding: decoding, lookaheadBits: lookaheadBits)
    return result
  }

  /// Exhaustively compare every packed entry with scalar transition steps.
  @discardableResult
  static func validate(
    interleaved: [UInt32], decoding: [UInt32], lookaheadBits: Int = 4
  ) throws -> Int {
    guard decoding.count == modelCount * stateCount else {
      throw PairedRuntimeTANSMacroTableError.invalidDecodingExtent(decoding.count)
    }
    guard lookaheadBits == 2 || lookaheadBits == 4 else {
      throw PairedRuntimeTANSMacroTableError.invalidLookaheadWidth(lookaheadBits)
    }
    let lookaheadCount = 1 << lookaheadBits
    let wordsPerModel = stateCount + stateCount * lookaheadCount * 2
    guard interleaved.count == modelCount * wordsPerModel else {
      throw PairedRuntimeTANSMacroTableError.invalidInterleavedExtent(interleaved.count)
    }
    var checked = 0
    for model in 0..<modelCount {
      let sourceBase = model * stateCount
      let modelBase = model * wordsPerModel
      guard interleaved[modelBase..<(modelBase + stateCount)].elementsEqual(
        decoding[sourceBase..<(sourceBase + stateCount)])
      else { throw PairedRuntimeTANSMacroTableError.ordinaryTableMismatch(model: model) }
      let macroBase = modelBase + stateCount
      for state in 0..<stateCount {
        for lookahead in 0..<lookaheadCount {
          let word = macroBase + (state * lookaheadCount + lookahead) * 2
          let actual = UInt64(interleaved[word]) | (UInt64(interleaved[word + 1]) << 32)
          let expected = try makeMacro(
            decoding: decoding, model: model, state: state, lookahead: lookahead,
            lookaheadBits: lookaheadBits)
          guard actual == expected else {
            throw PairedRuntimeTANSMacroTableError.macroMismatch(
              model: model, state: state, lookahead: lookahead)
          }
          checked += 1
        }
      }
    }
    return checked
  }

  private static func makeMacro(
    decoding: [UInt32], model: Int, state initialState: Int,
    lookahead: Int, lookaheadBits: Int
  ) throws -> UInt64 {
    var state = initialState
    var consumed = 0
    var count = 0
    var packed = UInt64(0)
    while count < maximumPairs {
      let code = decoding[model * stateCount + state]
      let pair = code & 4_095
      let bits = Int((code >> 12) & 15)
      let base = Int(code >> 16)
      guard bits <= 10, base + (1 << bits) <= stateCount else {
        throw PairedRuntimeTANSMacroTableError.invalidTransition(model: model, state: state)
      }
      if pair == escapePair || consumed + bits > lookaheadBits { break }
      let low = bits == 0
        ? 0 : (lookahead >> (lookaheadBits - consumed - bits)) & ((1 << bits) - 1)
      packed |= UInt64(pair) << (12 * count)
      state = base + low
      consumed += bits
      count += 1
    }
    packed |= UInt64(state) << 36
    packed |= UInt64(consumed) << 46
    packed |= UInt64(count) << 49
    return packed
  }
}

enum PairedRuntimeTANSMacroTableError: Error, Equatable {
  case invalidDecodingExtent(Int)
  case invalidInterleavedExtent(Int)
  case invalidLookaheadWidth(Int)
  case invalidTransition(model: Int, state: Int)
  case ordinaryTableMismatch(model: Int)
  case macroMismatch(model: Int, state: Int, lookahead: Int)
}
