import Foundation
import Darwin
import Native4DSTEMIO

let args = Array(CommandLine.arguments.dropFirst())
guard args.count == 2 else {
  fatalError("usage: CatalogCalibrationParity input cache-directory")
}
do {
  let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: URL(fileURLWithPath: args[1]))
    .prepare(input: URL(fileURLWithPath: args[0]), mode: .catalogOnly)
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.sortedKeys]
  print(String(decoding: try encoder.encode(catalog), as: UTF8.self))
} catch Native4DSTEMIOError.noDatasets {
  fputs("EMPTY_CATALOG\n", stderr)
  exit(2)
} catch {
  fputs("CATALOG_ERROR: \(error.localizedDescription)\n", stderr)
  exit(3)
}
