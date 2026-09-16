import Foundation
import Metal
import Native4DSTEMIO

/// Bounded input-window read accounting shared with the loader's reader queue.
final class EMPADReadStatus: @unchecked Sendable {
  private let lock = NSLock()
  private var elapsed = 0.0
  private var error: Error?
  private let completion = DispatchSemaphore(value: 0)

  func finish(seconds: Double, failure: Error?) {
    lock.lock()
    elapsed = seconds
    error = failure
    lock.unlock()
    completion.signal()
  }

  func wait() { completion.wait() }

  var seconds: Double { lock.lock(); defer { lock.unlock() }; return elapsed }
  var failure: Error? { lock.lock(); defer { lock.unlock() }; return error }
}

/// Read one EMPAD window on the reader queue so the GPU can pack the previous
/// one. `readFrames` opens its own handle and only writes its destination, so
/// it is safe to run against a buffer no other reader owns.
func startEMPADRead(
  source: NativeEMPADSource, frames: Range<Int>, into buffer: MTLBuffer,
  status: EMPADReadStatus, queue: DispatchQueue
) {
  let bytes = frames.count * 16384 * 4
  let retained = MetalInputReadBuffer(buffer: buffer)
  let indices = Array(frames)
  queue.async {
    // Retain the allocation, not just its raw destination pointer.
    defer { withExtendedLifetime(retained) {} }
    let destination = UnsafeMutableRawBufferPointer(start: retained.buffer.contents(), count: bytes)
    let started = CFAbsoluteTimeGetCurrent()
    do {
      try source.readFrames(indices, into: destination)
      status.finish(seconds: CFAbsoluteTimeGetCurrent() - started, failure: nil)
    } catch {
      status.finish(seconds: CFAbsoluteTimeGetCurrent() - started, failure: error)
    }
  }
}
