import Foundation

/// An explicit, path-scoped supplier declaration, not inferred calibration.
/// Measurements remain unchanged; clients must not subtract a second background.
/// Example: `source.backgroundSubtractionEvidence?.statement`.
public struct NativeBackgroundSubtractionEvidence: Sendable {
  public let documentURL: URL
  /// Original supplier document name, retained when evidence is restored from QEM.
  public let documentName: String
  public let statement: String
  private let identity: NativeFileIdentity

  static func restored(statement: String, documentName: String?, container: URL) throws -> Self {
    let name = documentName ?? container.lastPathComponent
    guard !name.isEmpty, name.utf8.count <= 255,
      !name.contains("/"), !name.contains("\\")
    else {
      throw EMPADError("Saved supplier document name is invalid; re-export the original acquisition.")
    }
    return Self(
      documentURL: container, documentName: name, statement: statement,
      identity: try nativeFileIdentity(for: container)
    )
  }

  func validateUnchanged() throws {
    guard identity == (try nativeFileIdentity(for: documentURL)) else {
      throw EMPADError(
        "The supplier README changed during loading. Reopen the folder to refresh its correction status."
      )
    }
  }

  static func discover(raw: URL, metadata: URL?) -> Self? {
    let targets = [raw, metadata].compactMap { $0?.resolvingSymlinksInPath().standardizedFileURL }
    var directory = raw.deletingLastPathComponent().resolvingSymlinksInPath()
    var matches: [Self] = []
    for _ in 0..<6 {
      // Do not inspect a volume root or walk into unrelated home-level notes.
      if directory.pathComponents.count <= 3 { break }
      for name in ["readme.txt", "README.txt", "README.md", "readme.md", "README"] {
        let document = directory.appendingPathComponent(name)
        guard let before = try? nativeFileIdentity(for: document), before.bytes <= 65536,
          let text = try? String(contentsOf: document, encoding: .utf8),
          let after = try? nativeFileIdentity(for: document), before == after
        else { continue }
        let result = declarations(text, directory: directory, targets: targets)
        if result.conflict { return nil }
        if let statement = result.statement {
          matches.append(Self(
            documentURL: document, documentName: document.lastPathComponent,
            statement: statement, identity: before))
        }
      }
      directory.deleteLastPathComponent()
    }
    return matches.first
  }

  private static func declarations(_ text: String, directory: URL, targets: [URL])
    -> (statement: String?, conflict: Bool)
  {
    var applies = false
    var statement: String?
    for rawLine in text.components(separatedBy: .newlines) {
      let line = rawLine.trimmingCharacters(in: .whitespaces)
      if line.isEmpty { continue }
      // Every unindented line ends the preceding scope. Only an existing
      // contained relative path starts a new scope; declarations are indented.
      if rawLine.first?.isWhitespace == false {
        applies = false
        let path = line.replacingOccurrences(of: "\\", with: "/")
        let components = path.split(separator: "/")
        if !path.hasPrefix("/"), !components.contains(".."), path != "." {
          let candidate = directory.appendingPathComponent(path).resolvingSymlinksInPath()
            .standardizedFileURL
          var isDirectory: ObjCBool = false
          if candidate.path.hasPrefix(directory.path + "/"),
            FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDirectory)
          {
            applies = targets.contains {
              $0 == candidate || (isDirectory.boolValue && $0.path.hasPrefix(candidate.path + "/"))
            }
            continue
          }
        }
        continue
      }
      guard applies else { continue }
      for clause in line.components(separatedBy: CharacterSet(charactersIn: ".;!?")) {
        let normalized = clause.lowercased().replacingOccurrences(of: "-", with: " ")
          .split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard
          normalized.contains("background subtract") || normalized.contains("background correct")
        else { continue }
        let words = Set(normalized.split(whereSeparator: { !$0.isLetter }).map(String.init))
        if !words.isDisjoint(with: [
          "not", "no", "never", "uncorrected", "unsure", "unknown",
          "whether", "if", "may", "might", "possibly", "maybe", "unless",
        ]) {
          return (nil, true)
        }
        if normalized == "already background subtracted"
          || normalized == "already background corrected"
        {
          statement = clause.trimmingCharacters(in: .whitespaces)
        }
      }
    }
    return (statement, false)
  }
}
