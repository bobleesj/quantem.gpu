import Metal

/// Read-only view of a completely resident native half spectrum. Partial caches retain the gathered path.
struct DirectPtychoSpectrumStorage {
  let buffers: [MTLBuffer]
  let chunkCapacity: Int
  let planeStride: Int
}

/// Bounded argument buffer; retains the exact prepared spectra, with no additional scientific arrays.
final class DirectPtychoIndexedBand {
  let source: DirectPtychoSpectrumStorage
  private let arguments: MTLBuffer
  private let indices: MTLBuffer

  init(device: MTLDevice, source: DirectPtychoSpectrumStorage, indices values: [UInt32]) throws {
    self.source = source
    indices = try DirectCoherentReconstruction.buffer(device, values)
    let chunks = MTLArgumentDescriptor()
    chunks.index = 0; chunks.dataType = .pointer; chunks.arrayLength = 12; chunks.access = .readOnly
    let mapping = MTLArgumentDescriptor()
    mapping.index = 12; mapping.dataType = .pointer; mapping.access = .readOnly
    let capacity = MTLArgumentDescriptor()
    capacity.index = 13; capacity.dataType = .uint
    let stride = MTLArgumentDescriptor()
    stride.index = 14; stride.dataType = .uint
    guard let encoder = device.makeArgumentEncoder(arguments: [chunks, mapping, capacity, stride]) else {
      throw MetalDirectPtychoError.commandQueue
    }
    arguments = try DirectCoherentReconstruction.allocate(device, encoder.encodedLength, shared: true)
    encoder.setArgumentBuffer(arguments, offset: 0)
    for i in 0..<12 { encoder.setBuffer(i < source.buffers.count ? source.buffers[i] : nil, offset: 0, index: i) }
    encoder.setBuffer(indices, offset: 0, index: 12)
    encoder.constantData(at: 13).storeBytes(of: UInt32(source.chunkCapacity), as: UInt32.self)
    encoder.constantData(at: 14).storeBytes(of: UInt32(source.planeStride), as: UInt32.self)
  }

  func bind(_ encoder: MTLComputeCommandEncoder) {
    encoder.setBuffer(arguments, offset: 0, index: 30)
    encoder.useResource(indices, usage: .read)
    for buffer in source.buffers { encoder.useResource(buffer, usage: .read) }
  }
}
