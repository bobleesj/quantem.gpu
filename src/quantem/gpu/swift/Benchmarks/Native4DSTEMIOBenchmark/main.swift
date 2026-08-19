import Foundation
import Native4DSTEMIO

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
  fputs("usage: native-4dstem-io-benchmark INPUT CACHE [--catalog-only]\n", stderr)
  exit(EXIT_FAILURE)
}

do {
  let input = URL(fileURLWithPath: arguments[1])
  let cache = URL(fileURLWithPath: arguments[2], isDirectory: true)
  let mode: Native4DSTEMCatalogMode = arguments.contains("--catalog-only")
    ? .catalogOnly
    : .indexed
  let started = ContinuousClock.now
  let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: cache).prepare(
    input: input,
    mode: mode
  )
  let duration = ContinuousClock.now - started
  let seconds = Double(duration.components.seconds)
    + Double(duration.components.attoseconds) / 1e18
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.sortedKeys]
  let output = try encoder.encode(catalog)
  FileHandle.standardOutput.write(output)
  FileHandle.standardOutput.write(Data("\n".utf8))
  fputs(
    String(format: "NATIVE_HDF5 mode=%@ datasets=%d wall=%.6f\n", mode == .catalogOnly ? "catalog" : "indexed", catalog.datasets.count, seconds),
    stderr
  )
} catch {
  fputs("NATIVE_HDF5 ERROR \(error.localizedDescription)\n", stderr)
  exit(EXIT_FAILURE)
}
