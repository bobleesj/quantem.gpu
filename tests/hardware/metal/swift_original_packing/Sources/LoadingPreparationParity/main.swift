import Foundation
import Metal
import Metal4DSTEMKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// A user interrupts background preparation, then opens the same acquisition.
// Completed sidecars may be reused; partial evidence must never be published.
final class CancellationProbe: @unchecked Sendable {
  let lock = NSLock()
  var calls = 0
  func cancelled() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    calls += 1
    return calls > 4
  }
}
let args = CommandLine.arguments
guard args.count == 3, let device = MTLCreateSystemDefaultDevice() else {
  fatalError("usage: LoadingPreparationParity acquisition disposable-index-directory")
}
let source = URL(fileURLWithPath: args[1])
let cache = URL(fileURLWithPath: args[2])
let probe = CancellationProbe()
do {
  _ = try Native4DSTEMCatalogBuilder(cacheDirectory: cache, shouldCancel: { probe.cancelled() })
    .prepare(input: source)
  fatalError("Background preparation did not cancel")
} catch is CancellationError { }
let builder = Native4DSTEMCatalogBuilder(cacheDirectory: cache)
let recovered = try builder.prepare(input: source).datasets[0]
let reopened = try builder.prepare(input: source).datasets[0]
precondition(recovered.sourceIdentitySHA256 != nil)
precondition(recovered.sourceIdentitySHA256 == reopened.sourceIdentitySHA256)
_ = try Native4DSTEMIndexedSource.open(dataset: reopened)
let before = device.currentAllocatedSize
try MetalCompactH5Loader.prepareLoadingResources(device: device)
let library = try Metal4DSTEMKernels.makeHDF5Library(device: device)
try MetalCompactH5Loader.prepareLoadingResources(device: device)
let reused = try Metal4DSTEMKernels.makeHDF5Library(device: device)
precondition(ObjectIdentifier(library as AnyObject) == ObjectIdentifier(reused as AnyObject))
precondition(device.currentAllocatedSize - before < 64 * 1024 * 1024,
             "Warmup unexpectedly allocated acquisition-sized storage")
print("LOADING_PREPARATION_CANCEL_RECOVER_REUSE_PASS")
