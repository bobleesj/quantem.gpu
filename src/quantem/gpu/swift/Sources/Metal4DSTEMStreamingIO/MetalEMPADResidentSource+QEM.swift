import Foundation
import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

extension MetalEMPADResidentSource {
  /// Save original float32 bit patterns plus any active mean-dark plane.
  /// Example: `try resident.saveQEM(to: destination)`; no original is overwritten.
  public func saveQEM(
    to destination: URL, userConfirmedBackgroundCorrected: Bool = false,
    calibrationOverrides: NativeQEMCalibration.Overrides? = nil,
    sourceDocuments: [NativeMetadataDocument] = [],
    shouldCancel: () -> Bool = { false }
  ) throws {
    guard !isReleased, !chunks.isEmpty else {
      throw qemError("Reload the acquisition before saving.")
    }
    let overrides =
      try calibrationOverrides ?? NativeQEMCalibration.read(metadata: source.microscopeMetadata)
    try NativeQEMCalibration.validate(overrides)
    let writer = try NativeQEMWriter(destination: destination)
    guard let queue = device.makeCommandQueue() else {
      throw qemError("Metal queue unavailable; retry saving.")
    }
    var table = [[String: Int]]()
    for chunk in chunks {
      if shouldCancel() { throw CancellationError() }
      var entry = ["first": chunk.firstFrame, "scans": chunk.frameCount]
      for (name, buffer) in [
        ("payload", chunk.payload), ("offset", chunk.offsets), ("model", chunk.models),
      ] {
        guard let staging = device.makeBuffer(length: buffer.length, options: .storageModeShared),
          let command = queue.makeCommandBuffer(), let copy = command.makeBlitCommandEncoder()
        else { throw qemError("Free memory for a bounded ANS save window.") }
        copy.copy(
          from: buffer, sourceOffset: 0, to: staging, destinationOffset: 0, size: buffer.length)
        copy.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
          throw qemError("ANS transfer failed; retry saving.")
        }
        entry[name + "_offset"] = try writer.append(
          Data(bytesNoCopy: staging.contents(), count: staging.length, deallocator: .none),
          shouldCancel: shouldCancel)
        entry[name + "_bytes"] = staging.length
      }
      table.append(entry)
    }
    var retainedMetadata = source.microscopeMetadata
    retainedMetadata.removeValue(forKey: NativeMetadataDocument.metadataKey)
    retainedMetadata.removeValue(forKey: NativeQEMMetadataUnits.metadataKey)
    var description: [String: Any] = [
      "format_identifier": source.formatIdentifier,
      "format_name": source.formatName, "microscope_metadata": retainedMetadata,
    ]
    description["user_confirmed_background_corrected"] =
      userConfirmedBackgroundCorrected
      || source.microscopeMetadata["qem_user_confirmed_background_corrected"] == "true"
    if let calibration = source.scanCalibration {
      description["scan_calibration"] = try JSONSerialization.jsonObject(
        with: JSONEncoder().encode(calibration))
    }
    if let step = source.diffractionSamplingInverseNanometers {
      description["diffraction_sampling_inv_nm"] = step
    }
    if let date = source.acquisitionDate { description["acquisition_date"] = date }
    if let evidence = source.backgroundSubtractionEvidence {
      description["supplier_background_statement"] = evidence.statement
    }
    if let background {
      description["background"] = [
        "schema": MetalEMPADBackground.schema,
        "identity": background.identitySHA256,
        "values_float32_le": Data(
          bytesNoCopy: background.values.contents(), count: source.detectorPixelCount * 4, deallocator: .none
        ).base64EncodedString(),
      ]
    }
    let microscope = NativeMicroscopeMetadata(metadata: source.microscopeMetadata)
    let dataset = Native4DSTEMDataset(
      id: sourceIdentitySHA256,
      label: source.rawURL.lastPathComponent, masterPath: source.rawURL.path,
      dataFiles: [source.rawURL.path], indexFiles: [], scanRows: source.scanRows,
      scanCols: source.scanColumns, detectorRows: source.detectorShape.row, detectorCols: source.detectorShape.column,
      sourceDtype: "float32", sourceBytes: source.sourceBytes, badPixelIndices: [],
      kPixelSizeRow: source.diffractionSamplingInverseNanometers ?? microscope.angularRowMrad,
      kPixelSizeCol: source.diffractionSamplingInverseNanometers ?? microscope.angularColumnMrad,
      kPixelUnit: source.diffractionSamplingInverseNanometers == nil ? "mrad" : "1/nm",
      acquisitionDate: source.acquisitionDate,
      metadata: source.microscopeMetadata.merging(
        ["sourceFormat": source.formatName], uniquingKeysWith: { _, new in new }),
      sourceScanCalibration: source.scanCalibration)
    var scientific = try NativeQEMCalibration.applying(
      overrides,
      to: NativeQEMMetadata.acquisition(dataset))
    scientific = try NativeMetadataDocument.adding(sourceDocuments, to: scientific)
    if background != nil {
      scientific["processing"] = [
        ["operation": "lossless_storage", "changes_measurements": false],
        [
          "operation": "mean_dark_subtraction", "changes_measurements": true,
          "scope": "display_and_products", "original_measurements_preserved": true,
        ],
      ]
    }
    try writer.finish(
      header: [
        "version": 1, "profile": MetalFloatANS.codec,
        "codec": MetalFloatANS.codec,
        "shape": [source.scanRows, source.scanColumns, source.detectorShape.row, source.detectorShape.column],
        "dtype": "float32", "logical_sha256": logicalSHA256, "chunks": table,
        "empad": description, "scientific_metadata": scientific,
      ], shouldCancel: shouldCancel)
  }

  static func restoreQEM(
    _ source: NativeEMPADSource, device: MTLDevice,
    memoryBudgetBytes: UInt64, shouldCancel: () -> Bool
  ) throws -> MetalEMPADResidentSource {
    let file = try NativeQEMFile(url: source.rawURL)
    guard file.codec == MetalFloatANS.codec, file.header["version"] as? Int == 1,
      let entries = file.header["chunks"] as? [[String: Int]],
      let logicalHash = file.header["logical_sha256"] as? String, logicalHash.count == 64,
      UInt64(device.currentAllocatedSize) + UInt64(file.bodyBytes) + 65536 <= memoryBudgetBytes
    else { throw qemError("Unsupported or oversized EMPAD QEM; update the reader or free memory.") }
    if shouldCancel() { throw CancellationError() }
    let mapped = try file.verifiedMapping()
    let ans = try MetalFloatANS(device: device, pixels: source.detectorPixelCount)
    var chunks = [Chunk]()
    var cursor = 0
    var first = 0
    try mapped.withUnsafeBytes { bytes in
      for entry in entries {
        if shouldCancel() { throw CancellationError() }
        guard entry["first"] == first, let count = entry["scans"], count > 0, count <= 512,
          first <= source.frameCount - count
        else { throw qemError("Invalid float ANS chunk coverage; recopy the file.") }
        var buffers = [MTLBuffer]()
        for name in ["payload", "offset", "model"] {
          guard entry[name + "_offset"] == cursor, let length = entry[name + "_bytes"],
            length > 0, length <= file.bodyBytes - cursor,
            name != "offset" || length == (ans.lanes + 1) * 4,
            name != "model" || length == ans.lanes,
            let buffer = device.makeBuffer(
              bytes: bytes.baseAddress!.advanced(by: file.bodyStart + cursor),
              length: length, options: .storageModeShared)
          else {
            throw qemError(
              "Invalid float ANS spans or insufficient memory; recopy the file or free memory.")
          }
          buffers.append(buffer)
          cursor += length
        }
        let offsets = buffers[1].contents().assumingMemoryBound(to: UInt32.self)
        let models = buffers[2].contents().assumingMemoryBound(to: UInt8.self)
        guard offsets[0] == 0, Int(offsets[ans.lanes]) <= buffers[0].length else {
          throw qemError("Invalid float ANS payload bounds.")
        }
        for stream in 0..<ans.lanes {
          let begin = Int(offsets[stream])
          let end = Int(offsets[stream + 1])
          let model = Int(models[stream])
          let length = end - begin
          guard end >= begin, end <= buffers[0].length,
            (model < 64 && length >= 4 && length <= count * 2)
              || (model == 252 && length % 2 == 0 && length <= count * 2)
              || (model == 253 && length == 0)
              || (model == 254 && length == count * 2)
              || (model == 255 && length == 2)
          else { throw qemError("Invalid float ANS stream bounds or model.") }
        }
        chunks.append(
          Chunk(
            firstFrame: first, frameCount: count,
            payload: buffers[0], offsets: buffers[1], models: buffers[2]))
        first += count
      }
    }
    guard first == source.frameCount, cursor == file.bodyBytes else {
      throw qemError("Incomplete EMPAD QEM; recopy the file.")
    }
    guard let queue = device.makeCommandQueue() else { throw qemError("Metal queue unavailable.") }
    for chunk in chunks {
      if shouldCancel() { throw CancellationError() }
      guard let command = queue.makeCommandBuffer() else {
        throw qemError("Metal command unavailable.")
      }
      let workspace = try ans.workspace(
        frames: chunk.frameCount, budget: memoryBudgetBytes, command: command)
      try ans.encode(chunk, into: workspace, command: command)
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed,
        workspace.errors.contents().load(as: UInt32.self) == 0
      else { throw qemError("Corrupt float ANS stream; recopy or re-export the acquisition.") }
    }
    let library = try Metal4DSTEMKernels.makeEMPADLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      let constants = MTLFunctionConstantValues()
      var decoded = true
      constants.setConstantValue(&decoded, type: .bool, index: 0)
      constants.setConstantValue(&decoded, type: .bool, index: 1)
      var pixels = UInt32(source.detectorPixelCount)
      var columns = UInt32(source.detectorShape.column)
      constants.setConstantValue(&pixels, type: .uint, index: 20)
      constants.setConstantValue(&columns, type: .uint, index: 21)
      let function = try library.makeFunction(name: name, constantValues: constants)
      return try device.makeComputePipelineState(function: function)
    }
    var background: MetalEMPADBackground?
    if let description = file.header["empad"] as? [String: Any],
      let saved = description["background"] as? [String: String]
    {
      guard saved["schema"] == MetalEMPADBackground.schema,
        let text = saved["values_float32_le"], let values = Data(base64Encoded: text),
        values.count == source.detectorPixelCount * 4,
        let identity = saved["identity"], identity.count == 64
      else { throw qemError("Invalid saved dark calibration; recopy the file.") }
      let buffer = try values.withUnsafeBytes { bytes -> MTLBuffer in
        guard bytes.bindMemory(to: Float.self).allSatisfy(\.isFinite),
          let buffer = device.makeBuffer(
            bytes: bytes.baseAddress!, length: values.count, options: .storageModeShared)
        else { throw qemError("Invalid saved dark values or insufficient memory.") }
        return buffer
      }
      background = MetalEMPADBackground(source: source, values: buffer, identity: identity)
    }
    try source.validateUnchanged()
    return try MetalEMPADResidentSource(
      source: source, device: device, ans: ans,
      diffraction: pipeline("empad_diffraction"), detector: pipeline("empad_virtual_image"),
      chunks: chunks,
      logicalSHA256: logicalHash, centerOfMass: pipeline("empad_center_of_mass_simd"),
      mean: pipeline("empad_mean_diffraction"),
      serialDetector: false, detectorThreads: 128, serialCenterOfMass: false,
      incremental: pipeline("empad_virtual_image_changes"), memoryBudgetBytes: memoryBudgetBytes,
      reusedSourceHash: false, background: background)
  }

  private static func qemError(_ message: String) -> Native4DSTEMIOError { .invalidData(message) }
  private func qemError(_ message: String) -> Native4DSTEMIOError { Self.qemError(message) }
}
