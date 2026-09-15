import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMStreamingIO

/// Independent original-count oracle for every output bit, not just image sums.
@main struct ConversionFixture {
  static func main() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let pixels = 64, packets = 8, stride = 20
    var streams = [[UInt16]]()
    for packet in 0..<packets {
      for rank in 0..<pixels {
        var values = [UInt16](repeating: 0, count: 512)
        if rank == 1 { values = Array(repeating: [UInt16(32767),32768,65535][packet%3], count: 512) }
        if rank == 2 { values = (0..<512).map { UInt16(($0*7919+packet*97)&65535) } }
        if rank == 3 || rank == 4 {
          for (position,value) in zip([0,31,32,255,511], rank == 3 ? [1,7,128,17,4] : [1,7,263,8,128]) {
            values[position] = UInt16(value)
          }
        }
        if rank >= 5 {
          values = (0..<512).map { scan in
            let mixed = (scan*1103515245 + rank*12345 + packet*7907) & 0x7fffffff
            if rank%4 == 0 { return mixed%53 == 0 ? 65535 : UInt16(mixed%3) }
            return UInt16(mixed % (1 + rank%32))
          }
        }
        streams.append(values)
      }
    }
    let codec = try MetalPairedRuntimeTANSSyntheticCodec(device: device)
    let encoded = try codec.roundTrip(streams: streams, logicalDtype: .uint16)
    precondition(encoded.decodedStreams == streams)
    var payload = [UInt8](), offsets: [UInt32] = [0], modes = encoded.modes
    for index in streams.indices {
      let rank = index%pixels, values = streams[index]
      func word(_ value: UInt16) { payload.append(UInt8(truncatingIfNeeded:value)); payload.append(UInt8(value>>8)) }
      switch rank {
      case 0: modes[index] = 253
      case 1: modes[index] = 255; word(values[0])
      case 2: modes[index] = 254; values.forEach { word($0) }
      case 3:
        modes[index] = 252
        for scan in values.indices where values[scan] != 0 { word(UInt16(scan<<7) | (values[scan]-1)) }
      case 4:
        modes[index] = 251
        var next = 0
        for scan in values.indices where values[scan] != 0 {
          let gap = scan-next, value = Int(values[scan])
          payload.append(UInt8((min(gap,31)<<3) | (value <= 7 ? value : 0)))
          if gap >= 31 {
            payload.append(UInt8(min(gap-31,255)))
            if gap-31 >= 255 { payload.append(UInt8(gap-31-255)) }
          }
          if value > 7 { payload.append(UInt8(value-8)) }
          next = scan+1
        }
      default: payload.append(contentsOf: encoded.payload[Int(encoded.offsets[index])..<Int(encoded.offsets[index+1])])
      }
      offsets.append(UInt32(payload.count))
    }
    let payloadBytes = payload.count
    payload += Array(repeating: 0, count: 8)
    let tables = try PairedRuntimeTANSTables.build()
    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    func buffer<T>(_ values: [T]) -> MTLBuffer {
      values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    }
    for (compact, staged) in [(false,false),(true,false),(false,true),(true,true)] {
      var offsetBuffer = buffer(offsets)
      if compact {
        let bases = (0...streams.count/32).map { offsets[$0*32] }
        let starts = offsets.indices.map { UInt16(offsets[$0]-bases[$0/32]) }
        var bytes = bases.withUnsafeBytes { Array($0) }
        bytes += starts.withUnsafeBytes { Array($0) }
        offsetBuffer = buffer(bytes)
      }
      let constants = MTLFunctionConstantValues()
      var enabled = compact
      constants.setConstantValue(&enabled, type: .bool, index: 20)
      var stageEnabled = staged
      constants.setConstantValue(&stageEnabled,type:.bool,index:80)
      func pipeline(_ name: String) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: library.makeFunction(name:name,constantValues:constants))
      }
      let measure = try pipeline("paired_runtime_packed_measure")
      let headersPipeline = try pipeline("paired_runtime_packed_headers")
      let write = try pipeline("paired_runtime_packed_write")
      let input = buffer(payload), table = buffer(tables.packedDecoding), modeBuffer = buffer(modes)
      let ranks = buffer((0..<pixels).reversed().map(UInt32.init))
      let headers = buffer(Array(repeating: UInt32(0),count:pixels*stride))
      let sums = buffer(Array(repeating: UInt32(0),count:pixels))
      let lengths = buffer(Array(repeating: UInt32(0),count:pixels))
      let widths = buffer(Array(repeating: UInt32(0),count:pixels))
      let failure = buffer([UInt32(0)])
      let staging = buffer(Array(repeating:UInt16(0xbeef),count:pixels*4096))
      func run(_ kernel: MTLComputePipelineState,_ buffers:[MTLBuffer],_ count:Int) {
        let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(kernel)
        for (index,value) in buffers.enumerated() { encoder.setBuffer(value,offset:0,index:index) }
        var parameters = [UInt32(pixels),UInt32(packets),UInt32(payloadBytes),UInt32(packets),UInt32(stride),0]
        encoder.setBytes(&parameters,length:parameters.count*4,index:buffers.count)
        if staged && buffers.count == 8 { encoder.setBuffer(staging,offset:0,index:9) }
        encoder.dispatchThreads(MTLSize(width:count,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:64,height:1,depth:1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        precondition(command.status == .completed && command.error == nil)
      }
      run(measure,[input,offsetBuffer,modeBuffer,table,headers,sums,ranks,failure],streams.count)
      precondition(failure.contents().load(as:UInt32.self) == 0)
      run(headersPipeline,[headers,lengths,widths],pixels)
      let header = headers.contents().assumingMemoryBound(to:UInt32.self)
      let size = lengths.contents().assumingMemoryBound(to:UInt32.self)
      var words = UInt32(0)
      for pixel in 0..<pixels { header[pixel*stride] = words; words += size[pixel] }
      let output = buffer(Array(repeating:UInt32(0xdeadbeef),count:Int(words)))
      run(write,[input,offsetBuffer,modeBuffer,table,headers,output,ranks,failure],streams.count)
      precondition(failure.contents().load(as:UInt32.self) == 0)
      let bits = output.contents().assumingMemoryBound(to:UInt32.self)
      let returned = buffer(Array(repeating:UInt16(0xffff),count:pixels*4096))
      let packedLibrary = try Metal4DSTEMKernels.makeCompactH5Library(device:device)
      let unpack = try device.makeComputePipelineState(function:packedLibrary.makeFunction(name:"compact_h5_conversion_window")!)
      let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
      encoder.setComputePipelineState(unpack)
      encoder.setBuffer(output,offset:0,index:0); encoder.setBuffer(headers,offset:0,index:1)
      encoder.setBuffer(returned,offset:0,index:2)
      var unpackParameters: [UInt32] = [4096,UInt32(pixels),128,32,20,2,2,1]
      encoder.setBytes(&unpackParameters,length:32,index:3)
      encoder.dispatchThreads(MTLSize(width:pixels*4096,height:1,depth:1),
        threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1))
      encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
      precondition(command.status == .completed && command.error == nil)
      let returnedValues = returned.contents().assumingMemoryBound(to:UInt16.self)
      var checked = 0
      for pixel in 0..<pixels {
        var at = Int(header[pixel*stride])
        var expectedSum = UInt32(0)
        for tile in 0..<128 {
          var width = Int((header[pixel*stride+4+tile/8] >> UInt32((tile%8)*4))&15)
          if width == 15 { width = 16 }
          for lane in 0..<32 {
            var value = UInt32(0)
            for plane in 0..<width { value |= ((bits[at+plane] >> UInt32(lane))&1) << UInt32(plane) }
            let scan = tile*32+lane, rank = pixels-1-pixel
            let expected = UInt32(streams[(scan/512)*pixels+rank][scan%512])
            precondition(value == expected,"Wrong count at pixel \(pixel), scan \(scan)")
            precondition(UInt32(returnedValues[scan*pixels+pixel]) == expected,"Reverse window changed a count")
            expectedSum += expected; checked += 1
          }
          at += width
        }
        precondition(sums.contents().assumingMemoryBound(to:UInt32.self)[pixel] == expectedSum)
      }
      // Unknown/interleaved modes fail closed instead of becoming zeros.
      modeBuffer.contents().assumingMemoryBound(to:UInt8.self)[0] = 96
      run(measure,[input,offsetBuffer,modeBuffer,table,headers,sums,ranks,failure],streams.count)
      precondition(failure.contents().load(as:UInt32.self) != 0)
      print("{\"phase\":\"synthetic-parity\",\"pass\":true,\"compact_offsets\":\(compact),\"staged\":\(staged),\"exact_counts\":\(checked),\"reverse_window_exact\":true,\"modes\":\(Set(modes).sorted()),\"invalid_mode_rejected\":true}")
    }
  }
}
