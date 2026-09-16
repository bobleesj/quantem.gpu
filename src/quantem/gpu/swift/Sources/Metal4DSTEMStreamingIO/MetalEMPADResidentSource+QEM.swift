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
      guard
        let payload = device.makeBuffer(length: chunk.payload.length, options: .storageModeShared),
        let descriptors = device.makeBuffer(
          length: chunk.descriptors.length, options: .storageModeShared),
        let command = queue.makeCommandBuffer(), let copy = command.makeBlitCommandEncoder()
      else {
        throw qemError(
          "Not enough memory to save a bounded compressed window; free memory and retry.")
      }
      copy.copy(
        from: chunk.payload, sourceOffset: 0, to: payload, destinationOffset: 0,
        size: payload.length)
      copy.copy(
        from: chunk.descriptors, sourceOffset: 0, to: descriptors, destinationOffset: 0,
        size: descriptors.length)
      copy.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed else {
        throw qemError("Compressed transfer failed; retry saving.")
      }
      let start = try writer.append(
        Data(bytesNoCopy: payload.contents(), count: payload.length, deallocator: .none),
        shouldCancel: shouldCancel)
      let descriptorStart = try writer.append(
        Data(bytesNoCopy: descriptors.contents(), count: descriptors.length, deallocator: .none),
        shouldCancel: shouldCancel)
      table.append([
        "first": chunk.firstFrame, "scans": chunk.frameCount,
        "payload_offset": start, "payload_bytes": payload.length,
        "descriptor_offset": descriptorStart, "descriptor_bytes": descriptors.length,
      ])
    }
    var description: [String: Any] = [
      "format_identifier": source.formatIdentifier,
      "format_name": source.formatName, "microscope_metadata": source.microscopeMetadata,
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
          bytesNoCopy: background.values.contents(), count: 65536, deallocator: .none
        ).base64EncodedString(),
      ]
    }
    let microscope = NativeMicroscopeMetadata(metadata: source.microscopeMetadata)
    let dataset = Native4DSTEMDataset(
      id: sourceIdentitySHA256,
      label: source.rawURL.lastPathComponent, masterPath: source.rawURL.path,
      dataFiles: [source.rawURL.path], indexFiles: [], scanRows: source.scanRows,
      scanCols: source.scanColumns, detectorRows: 128, detectorCols: 128,
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
        "version": 1, "profile": "empad-xor-row-packed-v1",
        "codec": "empad-xor-row-packed-v1",
        "shape": [source.scanRows, source.scanColumns, 128, 128],
        "dtype": "float32", "logical_sha256": logicalSHA256, "chunks": table,
        "empad": description, "scientific_metadata": scientific,
      ], shouldCancel: shouldCancel)
  }

  static func restoreQEM(
    _ source: NativeEMPADSource, device: MTLDevice,
    memoryBudgetBytes: UInt64, shouldCancel: () -> Bool
  ) throws -> MetalEMPADResidentSource {
    let file = try NativeQEMFile(url: source.rawURL)
    guard file.codec == "empad-xor-row-packed-v1", file.header["version"] as? Int == 1,
      let entries = file.header["chunks"] as? [[String: Int]],
      let logicalHash = file.header["logical_sha256"] as? String, logicalHash.count == 64,
      UInt64(device.currentAllocatedSize) + UInt64(file.bodyBytes) + 65536 <= memoryBudgetBytes
    else { throw qemError("Unsupported or oversized EMPAD QEM; update the reader or free memory.") }
    if shouldCancel() { throw CancellationError() }
    let mapped = try file.verifiedMapping()
    var chunks = [Chunk]()
    var cursor = 0
    var first = 0
    try mapped.withUnsafeBytes { bytes in
      for entry in entries {
        if shouldCancel() { throw CancellationError() }
        guard entry["first"] == first, let count = entry["scans"], count > 0, count <= 512,
          first <= source.frameCount - count,
          entry["payload_offset"] == cursor, let length = entry["payload_bytes"], length >= 4,
          length % 4 == 0, length <= file.bodyBytes - cursor,
          entry["descriptor_offset"] == cursor + length,
          entry["descriptor_bytes"] == count * 128 * 16,
          count * 128 * 16 <= file.bodyBytes - cursor - length
        else { throw qemError("Invalid EMPAD QEM chunk bounds; recopy the file.") }
        let descriptorPointer = bytes.baseAddress!.advanced(by: file.bodyStart + cursor + length)
        var words = 0
        for index in 0..<count * 128 {
          let descriptor = descriptorPointer.loadUnaligned(
            fromByteOffset: index * 16, as: SIMD4<UInt32>.self)
          guard descriptor.y <= 32, descriptor.z <= 32 - descriptor.y,
            Int(descriptor.w) == words
          else { throw qemError("Invalid EMPAD QEM row descriptor; recopy the file.") }
          words += Int(descriptor.y) * 4
        }
        guard max(4, words * 4) == length,
          let payload = device.makeBuffer(
            bytes: bytes.baseAddress!.advanced(by: file.bodyStart + cursor),
            length: length, options: .storageModeShared),
          let descriptors = device.makeBuffer(
            bytes: descriptorPointer, length: count * 128 * 16, options: .storageModeShared)
        else {
          throw qemError(
            "Invalid packed length or insufficient Metal memory; recopy the file or free memory.")
        }
        chunks.append(
          Chunk(firstFrame: first, frameCount: count, payload: payload, descriptors: descriptors))
        cursor += length + count * 128 * 16
        first += count
      }
    }
    guard first == source.frameCount, cursor == file.bodyBytes else {
      throw qemError("Incomplete EMPAD QEM; recopy the file.")
    }
    let library = try Metal4DSTEMKernels.makeEMPADLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw qemError("Missing EMPAD kernel; rebuild the app.")
      }
      return try device.makeComputePipelineState(function: function)
    }
    var background: MetalEMPADBackground?
    if let description = file.header["empad"] as? [String: Any],
      let saved = description["background"] as? [String: String]
    {
      guard saved["schema"] == MetalEMPADBackground.schema,
        let text = saved["values_float32_le"], let values = Data(base64Encoded: text),
        values.count == 65536,
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
      source: source, device: device,
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
