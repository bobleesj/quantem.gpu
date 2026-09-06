import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMStreamingIO

private enum OracleDtype: String {
  case u32
  case u64
}

private enum RadialPredicate {
  case lowerInclusiveUpperExclusive
  case lowerExclusiveUpperInclusive
}

private struct DetectorCase {
  let name: String
  let centerRow: Double
  let centerColumn: Double
  let innerRadius: Double
  let outerRadius: Double
  let predicate: RadialPredicate
  let oracleURL: URL
  let oracleDtype: OracleDtype
}

private struct SelectedOracle {
  let row: Int
  let column: Int
  let url: URL
}

private struct OracleSuite {
  let sourceIdentitySHA256: String
  let cases: [DetectorCase]
  let selected: SelectedOracle?
}

private struct ResidentDPCValidationResult {
  let receipt: [String: Any]
  let pass: Bool
  let peakDeviceAllocatedBytes: UInt64
}

private struct ResidentABAResult {
  let receipt: [String: Any]
  let peakDeviceAllocatedBytes: UInt64
}

private struct Options {
  let sourceURL: URL
  let oracleManifestURL: URL?
  let outputURL: URL?
  let interactionRepeats: Int
  let switchSourceURL: URL?
  let residentSwitchSourceURL: URL?
  let logicalHash: Bool
  let sourceAuditURL: URL?
  let dpcOracleManifestURL: URL?
  let authenticationPolicy: MetalCompactH5AuthenticationPolicy
  let loadRepeats: Int
  let trajectoryRuns: Int
  let trajectoryStatesPerRun: Int
}

@main
enum MetalCompactH5Benchmark {
  static func main() throws {
    let options = try parseOptions()
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw benchmarkError("No physical Metal device is available.")
    }
    let suite = try options.oracleManifestURL.map(loadOracleSuite)
    let source = try MetalCompactH5Loader.load(
      sourceURL: options.sourceURL,
      device: device,
      authenticationPolicy: options.authenticationPolicy
    )
    if let suite, suite.sourceIdentitySHA256 != source.metadata.sourceIdentitySHA256 {
      throw benchmarkError(
        "Oracle source identity \(suite.sourceIdentitySHA256) does not match compact "
          + "source \(source.metadata.sourceIdentitySHA256)."
      )
    }
    let sourceAuditReceipt = try options.sourceAuditURL.map {
      try validateSourceAudit($0, source: source)
    }
    let dpcValidation = try options.dpcOracleManifestURL.map {
      try validateResidentDPC(source: source, device: device, manifestURL: $0)
    }
    let dpcPass = dpcValidation?.pass ?? true
    let residentABA = try options.residentSwitchSourceURL.map {
      try runResidentABA(primary: source, secondaryURL: $0, device: device)
    }
    var logicalHashReceipt: Any = NSNull()
    var logicalHashPass = true
    if options.logicalHash {
      let metrics = try source.hashLogicalWorkingU8()
      let expected = source.metadata.workingLogicalSHA256
      logicalHashPass = expected == nil || metrics.sha256 == expected
      logicalHashReceipt = [
        "status": logicalHashPass ? "pass" : "fail",
        "sha256_u8_scan_major": metrics.sha256,
        "expected_sha256": expected.map { $0 as Any } ?? NSNull(),
        "logical_bytes": metrics.logicalBytes,
        "staging_bytes": metrics.stagingBytes,
        "wall_ms": metrics.wallMilliseconds,
        "gpu_ms": metrics.gpuMilliseconds,
      ]
    }

    let selectedStart = ContinuousClock.now
    let selectedRow = suite?.selected?.row ?? source.metadata.scanRows / 2
    let selectedColumn = suite?.selected?.column ?? source.metadata.scanColumns / 2
    let selectedValues = try source.extractDiffraction(
      scanRow: selectedRow,
      scanColumn: selectedColumn
    )
    let selectedMilliseconds = milliseconds(from: selectedStart)
    var selectedReceipt: [String: Any] = [
      "scan_row": selectedRow,
      "scan_column": selectedColumn,
      "wall_ms": selectedMilliseconds,
      "output_sha256_u32_le": sha256(values: selectedValues),
      "comparison": "not_requested",
    ]
    if let selected = suite?.selected {
      let expected = try [UInt8](Data(contentsOf: selected.url))
      guard expected.count == selectedValues.count else {
        throw benchmarkError(
          "Selected oracle has \(expected.count) pixels, expected \(selectedValues.count)."
        )
      }
      let mismatchCount = zip(selectedValues, expected).reduce(0) {
        $0 + ($1.0 == UInt32($1.1) ? 0 : 1)
      }
      selectedReceipt["comparison"] = mismatchCount == 0 ? "pass" : "fail"
      selectedReceipt["mismatch_count"] = mismatchCount
      selectedReceipt["oracle_path"] = selected.url.path
      selectedReceipt["oracle_sha256"] = sha256(data: Data(expected))
    }

    var caseReceipts: [[String: Any]] = []
    var interactionWall: [Double] = []
    var interactionGPU: [Double] = []
    var firstDetectorReadyMilliseconds: Double?
    if let suite {
      for detectorCase in suite.cases {
        let mask = radialMask(
          rows: source.metadata.detectorRows,
          columns: source.metadata.detectorColumns,
          detectorCase: detectorCase
        )
        let metrics = try source.updateVirtualDetector(mask: mask)
        if firstDetectorReadyMilliseconds == nil {
          firstDetectorReadyMilliseconds = metrics.wallMilliseconds
        }
        interactionWall.append(metrics.wallMilliseconds)
        interactionGPU.append(metrics.gpuMilliseconds)
        let actual = try source.virtualDetectorValues()
        let comparison = try compare(
          actual: actual,
          oracleURL: detectorCase.oracleURL,
          dtype: detectorCase.oracleDtype
        )
        caseReceipts.append([
          "name": detectorCase.name,
          "center_row": detectorCase.centerRow,
          "center_column": detectorCase.centerColumn,
          "inner_radius": detectorCase.innerRadius,
          "outer_radius": detectorCase.outerRadius,
          "requested_detector_pixels": mask.reduce(0) { $0 + Int($1) },
          "effective_detector_pixels": mask.enumerated().reduce(0) {
            $0
              + ($1.element != 0
                && !source.metadata.excludedDetectorPixels.contains($1.offset) ? 1 : 0)
          },
          "mode": metrics.mode,
          "changed_detector_pixels": metrics.changedDetectorPixels,
          "wall_ms": metrics.wallMilliseconds,
          "gpu_ms": metrics.gpuMilliseconds,
          "fft_dispatch_count": metrics.fftDispatchCount,
          "output_sha256_u32_le": sha256(values: actual),
          "oracle_path": detectorCase.oracleURL.path,
          "oracle_dtype": detectorCase.oracleDtype.rawValue,
          "comparison": comparison,
        ])
      }
      if suite.cases.count > 1 {
        for repeatIndex in 0..<options.interactionRepeats {
          for detectorCase in suite.cases {
            let metrics = try source.updateVirtualDetector(
              mask: radialMask(
                rows: source.metadata.detectorRows,
                columns: source.metadata.detectorColumns,
                detectorCase: detectorCase
              )
            )
            interactionWall.append(metrics.wallMilliseconds)
            interactionGPU.append(metrics.gpuMilliseconds)
            guard metrics.fftDispatchCount == 0 else {
              throw benchmarkError(
                "FFT-off interaction unexpectedly dispatched an FFT at repeat "
                  + "\(repeatIndex)."
              )
            }
          }
        }
      }
    }

    let allDetectorPass = caseReceipts.allSatisfy {
      ($0["comparison"] as? [String: Any])?["mismatch_count"] as? Int == 0
    }
    let selectedPass = (selectedReceipt["comparison"] as? String) != "fail"
    var trajectoryReceipt: Any = NSNull()
    var trajectoryPass = true
    if options.trajectoryStatesPerRun > 0 {
      let result = try runTrajectory(
        source: source,
        device: device,
        runs: options.trajectoryRuns,
        statesPerRun: options.trajectoryStatesPerRun
      )
      trajectoryReceipt = result.receipt
      trajectoryPass = result.pass
    }
    let load = source.loadMetrics
    var loadSeries: [[String: Any]] = [
      loadSample(
        sampleIndex: 0,
        load: load,
        firstResidentProductMilliseconds: selectedMilliseconds
      )
    ]
    if options.loadRepeats > 1 {
      source.releaseResidentStorage()
      for sampleIndex in 1..<options.loadRepeats {
        let repeated = try MetalCompactH5Loader.load(
          sourceURL: options.sourceURL,
          device: device,
          authenticationPolicy: options.authenticationPolicy
        )
        let productStart = ContinuousClock.now
        _ = try repeated.extractDiffraction(
          scanRow: selectedRow,
          scanColumn: selectedColumn
        )
        let productMilliseconds = milliseconds(from: productStart)
        loadSeries.append(
          loadSample(
            sampleIndex: sampleIndex,
            load: repeated.loadMetrics,
            firstResidentProductMilliseconds: productMilliseconds
          )
        )
        repeated.releaseResidentStorage()
      }
    }
    let loadReadySamples = loadSeries.compactMap {
      $0["resident_ready_ms"] as? Double
    }
    let firstProductSamples = loadSeries.compactMap {
      $0["first_resident_product_ms"] as? Double
    }
    var switchReceipt: Any = NSNull()
    if let switchSourceURL = options.switchSourceURL {
      let allocatedBeforeRelease = device.currentAllocatedSize
      source.releaseResidentStorage()
      let allocatedAfterRelease = device.currentAllocatedSize
      let switchStart = ContinuousClock.now
      let next = try MetalCompactH5Loader.load(
        sourceURL: switchSourceURL,
        device: device,
        authenticationPolicy: options.authenticationPolicy
      )
      let nextPattern = try next.extractDiffraction(
        scanRow: next.metadata.scanRows / 2,
        scanColumn: next.metadata.scanColumns / 2
      )
      switchReceipt = [
        "from_source_identity_sha256": source.metadata.sourceIdentitySHA256,
        "to_path": switchSourceURL.path,
        "to_source_identity_sha256": next.metadata.sourceIdentitySHA256,
        "old_source_released": source.isReleased,
        "device_allocated_bytes_before_release": allocatedBeforeRelease,
        "device_allocated_bytes_after_release": allocatedAfterRelease,
        "next_resident_bytes": next.loadMetrics.residentBytes,
        "next_resident_ready_ms": next.loadMetrics.totalMilliseconds,
        "next_selected_diffraction_ms": milliseconds(from: switchStart)
          - next.loadMetrics.totalMilliseconds,
        "next_selected_diffraction_sha256_u32_le": sha256(values: nextPattern),
        "switch_ready_ms": milliseconds(from: switchStart),
      ]
      next.releaseResidentStorage()
    }
    let receipt: [String: Any] = [
      "schema": "quantem.gpu.metal-compact-h5-benchmark/v2",
      "status": allDetectorPass && selectedPass && logicalHashPass && trajectoryPass
        && dpcPass
        ? "pass" : "fail",
      "source": [
        "path": options.sourceURL.path,
        "bytes": source.metadata.sourceBytes,
        "schema": source.metadata.schema,
        "payload_codec": source.metadata.payloadCodec,
        "source_dtype": source.metadata.sourceDtype.map { $0 as Any } ?? NSNull(),
        "embedded_scientific_semantics": source.metadata.embeddedScientificSemantics,
        "identity_sha256": source.metadata.sourceIdentitySHA256,
        "raw_logical_sha256": source.metadata.sourceRawLogicalSHA256
          .map { $0 as Any } ?? NSNull(),
        "shape": [
          source.metadata.scanRows,
          source.metadata.scanColumns,
          source.metadata.detectorRows,
          source.metadata.detectorColumns,
        ],
        "excluded_detector_pixels": source.metadata.excludedDetectorPixels,
        "detector_mask_sha256": source.metadata.detectorMaskSHA256
          .map { $0 as Any } ?? NSNull(),
        "masked_detector_pixels_sha256": source.metadata.maskedDetectorPixelsSHA256
          .map { $0 as Any } ?? NSNull(),
        "masked_detector_raw_values": source.metadata.maskedDetectorRawValues
          .map { $0.map(Int.init) as Any } ?? NSNull(),
        "raw_access_mode": source.metadata.rawAccessMode,
        "working_logical_sha256": source.metadata.workingLogicalSHA256
          .map { $0 as Any } ?? NSNull(),
        "scan_tile": source.metadata.scanTile,
      ],
      "hardware": [
        "backend": "Metal",
        "device_name": device.name,
        "registry_id": String(device.registryID),
        "unified_memory": device.hasUnifiedMemory,
        "max_buffer_length": device.maxBufferLength,
        "recommended_working_set_bytes": device.recommendedMaxWorkingSetSize,
        "operating_system": ProcessInfo.processInfo.operatingSystemVersionString,
        "ui_free": true,
        "presentation_measurement": false,
      ],
      "load": [
        "cache_state": "uncontrolled",
        "cold_claim": false,
        "authentication_policy": authenticationPolicyName(options.authenticationPolicy),
        "metadata_ms": load.metadataMilliseconds,
        "source_read_ms": load.sourceReadMilliseconds,
        "descriptor_preparation_ms": load.descriptorPreparationMilliseconds,
        "gpu_decode_ms": load.gpuDecodeMilliseconds,
        "decoded_integrity_sha256_ms": load.decodedIntegrityMilliseconds,
        "private_upload_ms": load.privateUploadMilliseconds,
        "resident_ready_ms": load.totalMilliseconds,
        "resident_bytes": load.residentBytes,
        "maximum_transient_bytes": load.maximumTransientBytes,
        "device_allocated_bytes_before": load.deviceAllocatedBytesBefore,
        "device_allocated_bytes_after": load.deviceAllocatedBytesAfter,
        "decoded_shard_sha256_checks": load.decodedShardSHA256Checks,
        "mapped_authentication_bytes": load.mappedAuthenticationBytes,
        "logical_dense_allocation_bytes": 0,
      ],
      "first_resident_compute": selectedReceipt,
      "prepared_reopen_series": [
        "cache_state": "uncontrolled",
        "cold_claim": false,
        "sample_count": loadSeries.count,
        "samples": loadSeries,
        "resident_ready_ms": distribution(loadReadySamples),
        "first_resident_product_ms": distribution(firstProductSamples),
      ],
      "logical_working_hash": logicalHashReceipt,
      "first_detector_ready_ms":
        firstDetectorReadyMilliseconds
        .map { $0 as Any } ?? NSNull(),
      "detector_cases": caseReceipts,
      "interaction": [
        "measurement_scope": "compute_only_no_present",
        "sample_count": interactionWall.count,
        "wall_ms": distribution(interactionWall),
        "gpu_ms": distribution(interactionGPU),
        "fft_enabled": false,
        "fft_dispatch_count": 0,
      ],
      "trajectory": trajectoryReceipt,
      "release_then_switch": switchReceipt,
      "oracle_manifest": options.oracleManifestURL
        .map { $0.path as Any } ?? NSNull(),
      "external_source_audit": sourceAuditReceipt ?? NSNull(),
      "resident_dpc_validation": dpcValidation?.receipt ?? NSNull(),
      "already_resident_aba": residentABA?.receipt ?? NSNull(),
      "memory": [
        "peak_process_rss_bytes": processPeakResidentBytes().map { $0 as Any }
          ?? NSNull(),
        "peak_device_allocated_bytes": max(
          max(
            loadSeries.compactMap { $0["device_allocated_bytes_after"] as? UInt64 }
              .max() ?? 0,
            dpcValidation?.peakDeviceAllocatedBytes ?? 0
          ),
          residentABA?.peakDeviceAllocatedBytes ?? 0
        ),
      ],
    ]
    if let invalid = firstInvalidJSONValue(receipt, path: "receipt") {
      throw benchmarkError("Receipt contains a non-JSON value at \(invalid).")
    }
    let data = try JSONSerialization.data(
      withJSONObject: receipt,
      options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )
    if let outputURL = options.outputURL {
      try data.write(to: outputURL, options: .atomic)
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
    if !allDetectorPass || !selectedPass || !logicalHashPass || !trajectoryPass
      || !dpcPass
    {
      Foundation.exit(2)
    }
  }
}

private func loadOracleSuite(_ url: URL) throws -> OracleSuite {
  let data = try Data(contentsOf: url)
  guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
    let schema = root["schema"] as? String
  else { throw benchmarkError("Oracle manifest is not a JSON object with a schema.") }
  let directory = url.deletingLastPathComponent()
  if schema == "quantem.gpu.packed-detector-oracles/v1" {
    guard let identity = root["source_identity_sha256"] as? String,
      let rawCases = root["cases"] as? [[String: Any]]
    else { throw benchmarkError("Packed detector oracle manifest is incomplete.") }
    let cases = try rawCases.map { item -> DetectorCase in
      guard let name = item["name"] as? String,
        let geometry = item["geometry"] as? [String: Any],
        let image = item["image"] as? [String: Any],
        let relative = image["path"] as? String
      else { throw benchmarkError("A packed detector oracle case is incomplete.") }
      return DetectorCase(
        name: name,
        centerRow: number(geometry["center_row"]),
        centerColumn: number(geometry["center_column"]),
        innerRadius: number(geometry["inner_radius"]),
        outerRadius: number(geometry["outer_radius"]),
        predicate: .lowerInclusiveUpperExclusive,
        oracleURL: directory.appendingPathComponent(relative),
        oracleDtype: .u32
      )
    }
    return OracleSuite(
      sourceIdentitySHA256: identity,
      cases: cases,
      selected: nil
    )
  }
  guard schema.hasSuffix("exact-companion/v1"),
    let source = root["source"] as? [String: Any],
    let identity = source["source_identity_sha256"] as? String,
    let contract = root["product_contract"] as? [String: Any],
    let center = contract["detector_center"] as? [NSNumber],
    let bands = contract["canonical_bands_pixels"] as? [String: [String: Any]],
    let arrays = contract["arrays"] as? [String: [String: Any]],
    let selectedCoordinate = contract["selected_scan_coordinate"] as? [NSNumber]
  else { throw benchmarkError("Unsupported oracle manifest schema \(schema).") }
  let cases = try ["bf", "abf", "adf"].map { name -> DetectorCase in
    guard let band = bands[name], let array = arrays[name],
      let relative = array["path"] as? String
    else { throw benchmarkError("Exact product oracle \(name) is missing.") }
    return DetectorCase(
      name: name,
      centerRow: center[0].doubleValue,
      centerColumn: center[1].doubleValue,
      innerRadius: number(band["inner_radius_exclusive"]),
      outerRadius: number(band["outer_radius_inclusive"]),
      predicate: .lowerExclusiveUpperInclusive,
      oracleURL: directory.appendingPathComponent(relative),
      oracleDtype: .u64
    )
  }
  guard let selectedArray = arrays["selected_center_masked"],
    let selectedPath = selectedArray["path"] as? String
  else { throw benchmarkError("Exact selected-diffraction oracle is missing.") }
  return OracleSuite(
    sourceIdentitySHA256: identity,
    cases: cases,
    selected: SelectedOracle(
      row: selectedCoordinate[0].intValue,
      column: selectedCoordinate[1].intValue,
      url: directory.appendingPathComponent(selectedPath)
    )
  )
}

private func validateSourceAudit(
  _ url: URL,
  source: MetalCompactH5ResidentSource
) throws -> [String: Any] {
  let data = try Data(contentsOf: url)
  guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
    root["schema"] as? String
      == "quantem.gpu.android-vulkan-real-fixture-audit/v1",
    root["status"] as? String == "ADMITTED",
    root["source_identity_sha256"] as? String
      == source.metadata.sourceIdentitySHA256,
    let shape = root["source_shape"] as? [NSNumber],
    shape.map(\.intValue) == [
      source.metadata.scanRows,
      source.metadata.scanColumns,
      source.metadata.detectorRows,
      source.metadata.detectorColumns,
    ],
    root["source_dtype"] as? String == "uint16",
    (root["scan_bin"] as? NSNumber)?.intValue == 1,
    (root["detector_bin"] as? NSNumber)?.intValue == 1,
    root["crop"] is NSNull,
    let range = root["range_audit"] as? [String: Any],
    range["complete"] as? Bool == true,
    range["uint8_working_representation_admitted"] as? Bool == true,
    (range["values_above_255"] as? NSNumber)?.uint64Value == 0,
    range["prepared_uint8_sha256"] as? String
      == source.metadata.workingLogicalSHA256,
    let rawLogicalSHA256 = range["logical_source_sha256"] as? String
  else {
    throw benchmarkError(
      "External source audit does not bind the compact source, exact geometry, "
        + "uint16 raw semantics, and lossless uint8 working identity."
    )
  }
  return [
    "status": "pass",
    "path": url.path,
    "sha256": sha256(data: data),
    "source_dtype": "uint16",
    "source_logical_sha256": rawLogicalSHA256,
    "working_logical_sha256": source.metadata.workingLogicalSHA256!,
    "scan_bin": 1,
    "detector_bin": 1,
    "crop": NSNull(),
  ]
}

private func radialMask(
  rows: Int,
  columns: Int,
  detectorCase: DetectorCase
) -> [UInt8] {
  let innerSquared = detectorCase.innerRadius * detectorCase.innerRadius
  let outerSquared = detectorCase.outerRadius * detectorCase.outerRadius
  return (0..<(rows * columns)).map { pixel in
    let row = Double(pixel / columns)
    let column = Double(pixel % columns)
    let distanceSquared =
      (row - detectorCase.centerRow) * (row - detectorCase.centerRow)
      + (column - detectorCase.centerColumn) * (column - detectorCase.centerColumn)
    switch detectorCase.predicate {
    case .lowerInclusiveUpperExclusive:
      return distanceSquared >= innerSquared && distanceSquared < outerSquared ? 1 : 0
    case .lowerExclusiveUpperInclusive:
      if detectorCase.innerRadius == 0 {
        return distanceSquared <= outerSquared ? 1 : 0
      }
      return distanceSquared > innerSquared && distanceSquared <= outerSquared ? 1 : 0
    }
  }
}

private func compare(
  actual: [UInt32],
  oracleURL: URL,
  dtype: OracleDtype
) throws -> [String: Any] {
  let data = try Data(contentsOf: oracleURL)
  let stride = dtype == .u32 ? 4 : 8
  guard data.count == actual.count * stride else {
    throw benchmarkError(
      "Oracle \(oracleURL.path) has \(data.count) bytes, expected \(actual.count * stride)."
    )
  }
  var mismatchCount = 0
  var maximumAbsoluteDifference: UInt64 = 0
  for index in actual.indices {
    let expected: UInt64 = data.withUnsafeBytes { raw in
      if dtype == .u32 {
        return UInt64(
          UInt32(
            littleEndian: raw.loadUnaligned(
              fromByteOffset: index * 4,
              as: UInt32.self
            )
          )
        )
      }
      return UInt64(
        littleEndian: raw.loadUnaligned(
          fromByteOffset: index * 8,
          as: UInt64.self
        )
      )
    }
    let observed = UInt64(actual[index])
    if observed != expected {
      mismatchCount += 1
      maximumAbsoluteDifference = max(
        maximumAbsoluteDifference,
        observed > expected ? observed - expected : expected - observed
      )
    }
  }
  return [
    "status": mismatchCount == 0 ? "pass" : "fail",
    "mismatch_count": mismatchCount,
    "maximum_absolute_difference": maximumAbsoluteDifference,
    "compared_values": actual.count,
    "oracle_sha256": sha256(data: data),
  ]
}

private func validateResidentDPC(
  source: MetalCompactH5ResidentSource,
  device: MTLDevice,
  manifestURL: URL
) throws -> ResidentDPCValidationResult {
  let manifestData = try Data(contentsOf: manifestURL)
  guard
    let root = try JSONSerialization.jsonObject(with: manifestData)
      as? [String: Any],
    root["schema"] as? String == "quantem-gpu-android-qh5-index-manifest-v2",
    let cache = root["exact_product_cache"] as? [String: Any],
    cache["status"] as? String == "PASS",
    cache["source_identity_sha256"] as? String
      == source.metadata.sourceIdentitySHA256,
    let contract = cache["contract"] as? [String: Any],
    let rotation = contract["dpc_rotation_degrees"] as? NSNumber,
    contract["dpc_component_order_exchanged"] as? Bool == false,
    let artifacts = cache["artifacts"] as? [String: [String: Any]]
  else {
    throw benchmarkError(
      "DPC oracle does not bind the exact source, rotation, and component order."
    )
  }

  func artifactURL(_ name: String) throws -> URL {
    guard let relative = artifacts[name]?["path"] as? String else {
      throw benchmarkError("DPC oracle is missing artifact \(name).")
    }
    return manifestURL.deletingLastPathComponent().appendingPathComponent(relative)
  }

  guard let moments = try source.preparedDPCMomentValues(),
    let maps = try source.preparedDPCValues(),
    let rowBuffer = try source.preparedDPCDisplayBuffer(component: .row),
    let columnBuffer = try source.preparedDPCDisplayBuffer(component: .column)
  else {
    throw benchmarkError("The exact prepared DPC source is incomplete.")
  }
  let total = try compareUInt64(
    actual: moments.total,
    oracleURL: artifactURL("total_intensity")
  )
  let rowMoment = try compareUInt64(
    actual: moments.detectorRowMoment,
    oracleURL: artifactURL("detector_row_moment")
  )
  let columnMoment = try compareUInt64(
    actual: moments.detectorColumnMoment,
    oracleURL: artifactURL("detector_column_moment")
  )
  let centeredRow = try compareFloat(
    actual: maps.row,
    oracleURL: artifactURL("com_row"),
    absoluteTolerance: 0
  )
  let centeredColumn = try compareFloat(
    actual: maps.column,
    oracleURL: artifactURL("com_column"),
    absoluteTolerance: 0
  )

  let mean = try source.meanDiffractionPattern()
  let diffractionSum = try compareUInt64(
    actual: mean.detectorSum,
    oracleURL: artifactURL("diffraction_sum")
  )
  let meanDiffraction = try compareFloat(
    actual: mean.mean,
    oracleURL: artifactURL("mean_diffraction"),
    absoluteTolerance: 0
  )

  let processor = try Metal4DSTEMDPCProcessor(device: device)
  let result = try processor.process(
    centeredRowBuffer: rowBuffer,
    centeredColumnBuffer: columnBuffer,
    configuration: Metal4DSTEMDPCConfiguration(
      scanRows: source.metadata.scanRows,
      scanColumns: source.metadata.scanColumns,
      rotationDegrees: rotation.doubleValue,
      transposeComponents: false
    )
  )
  let phasePointer = result.phaseBuffer.contents().bindMemory(
    to: Float.self,
    capacity: source.metadata.scanCount
  )
  let phase = Array(
    UnsafeBufferPointer(start: phasePointer, count: source.metadata.scanCount)
  )
  let idpc = try compareFloat(
    actual: phase,
    oracleURL: artifactURL("idpc"),
    absoluteTolerance: 2e-5
  )
  let comparisons = [
    total,
    rowMoment,
    columnMoment,
    centeredRow,
    centeredColumn,
    diffractionSum,
    meanDiffraction,
    idpc,
  ]
  let pass =
    comparisons.allSatisfy { $0["status"] as? String == "pass" }
    && result.metrics.uploadBytes == 0
    && result.metrics.readbackBytes == 0
  return ResidentDPCValidationResult(
    receipt: [
      "status": pass ? "pass" : "fail",
      "oracle_manifest_path": manifestURL.path,
      "oracle_manifest_sha256": sha256(data: manifestData),
      "rotation_degrees": rotation.doubleValue,
      "transpose_components": false,
      "exact_moments": [
        "total": total,
        "detector_row": rowMoment,
        "detector_column": columnMoment,
      ],
      "centered_dpc": [
        "row": centeredRow,
        "column": centeredColumn,
      ],
      "mean_diffraction": [
        "detector_sum": diffractionSum,
        "mean": meanDiffraction,
        "wall_ms": mean.wallMilliseconds,
        "gpu_ms": mean.gpuMilliseconds,
        "dispatch_count": mean.dispatchCount,
        "audit_readback_bytes": mean.readbackBytes,
      ],
      "idpc": [
        "comparison": idpc,
        "wall_ms": result.metrics.wallMilliseconds,
        "gpu_ms": result.metrics.gpuMilliseconds,
        "fft_dispatch_count": result.metrics.fftDispatchCount,
        "total_dispatch_count": result.metrics.totalDispatchCount,
        "upload_bytes": result.metrics.uploadBytes,
        "interaction_path_readback_bytes": result.metrics.readbackBytes,
        "parity_audit_readback_bytes": UInt64(source.metadata.scanCount * 4),
        "synchronization_count": result.metrics.synchronizationCount,
        "device_allocated_bytes_before": result.metrics.deviceAllocatedBytesBefore,
        "device_allocated_bytes_after": result.metrics.deviceAllocatedBytesAfter,
        "gradient_fft_storage_mode": result.gradientFFTBuffer.storageMode.rawValue,
        "phase_fft_storage_mode": result.phaseFFTBuffer.storageMode.rawValue,
      ],
    ],
    pass: pass,
    peakDeviceAllocatedBytes: max(
      result.metrics.deviceAllocatedBytesBefore,
      result.metrics.deviceAllocatedBytesAfter
    )
  )
}

private func runResidentABA(
  primary: MetalCompactH5ResidentSource,
  secondaryURL: URL,
  device: MTLDevice
) throws -> ResidentABAResult {
  let secondary = try MetalCompactH5Loader.load(
    sourceURL: secondaryURL,
    device: device,
    authenticationPolicy: .boundedSequential
  )
  defer { secondary.releaseResidentStorage() }
  guard
    primary.metadata.sourceIdentitySHA256
      != secondary.metadata.sourceIdentitySHA256
  else {
    throw benchmarkError("Resident A-B-A requires two distinct source identities.")
  }
  let peakDevice = UInt64(device.currentAllocatedSize)
  let recorder = Metal4DSTEMPublicationRecorder(signpostsEnabled: false)
  let sequence = [primary, secondary, primary]
  var samples: [[String: Any]] = []
  var wallSamples: [Double] = []
  for (offset, source) in sequence.enumerated() {
    let generation = UInt64(offset + 1)
    let started = ContinuousClock.now
    guard
      try recorder.begin(
        generation: generation,
        sourceIdentitySHA256: source.metadata.sourceIdentitySHA256,
        representation: .packed
      )
    else {
      throw benchmarkError("A new resident A-B-A generation was rejected.")
    }
    _ = try source.extractDiffraction(
      scanRow: source.metadata.scanRows / 2,
      scanColumn: source.metadata.scanColumns / 2
    )
    let detector = try source.activatePreparedDetectorProduct(.adf)
    _ = try recorder.record(
      generation: generation,
      milestone: .residentReady,
      counters: Metal4DSTEMPublicationCounters(
        sourceBytes: source.metadata.sourceBytes,
        residentBytes: source.loadMetrics.totalResidentBytes,
        processRSSBytes: processPeakResidentBytes(),
        peakProcessRSSBytes: processPeakResidentBytes(),
        deviceAllocatedBytes: UInt64(device.currentAllocatedSize),
        peakDeviceAllocatedBytes: peakDevice,
        storageReadBytes: 0,
        uploadBytes: 0,
        readbackBytes: 0,
        synchronizationCount: 2
      ),
      detail: "already-resident exact DP and prepared ADF ready; no presentation claim"
    )
    let wall = milliseconds(from: started)
    wallSamples.append(wall)
    samples.append([
      "generation": generation,
      "source_identity_sha256": source.metadata.sourceIdentitySHA256,
      "wall_ms": wall,
      "storage_read_bytes": 0,
      "upload_bytes": 0,
      "readback_bytes": 0,
      "detector_mode": detector.mode,
      "actual_present": false,
    ])
  }
  guard
    try !recorder.record(
      generation: 1,
      milestone: .firstResidentPresent,
      detail: "intentional stale-generation rejection probe"
    )
  else {
    throw benchmarkError("The stale A generation unexpectedly published.")
  }
  let eventsData = try JSONEncoder().encode(recorder.events())
  let events = try JSONSerialization.jsonObject(with: eventsData)
  return ResidentABAResult(
    receipt: [
      "boundary": "exact-switch-to-resident-ready",
      "samples": samples,
      "distribution_ms": distribution(wallSamples),
      "source_storage_reads_during_switch": 0,
      "complete_sources_retained": 2,
      "actual_present": false,
      "events": events,
      "secondary_load": [
        "boundary": "prepared-reopen-to-resident-ready",
        "resident_ready_ms": secondary.loadMetrics.totalMilliseconds,
        "resident_bytes": secondary.loadMetrics.totalResidentBytes,
      ],
    ],
    peakDeviceAllocatedBytes: peakDevice
  )
}

private func compareUInt64(
  actual: [UInt64],
  oracleURL: URL
) throws -> [String: Any] {
  let data = try Data(contentsOf: oracleURL)
  guard data.count == actual.count * MemoryLayout<UInt64>.stride else {
    throw benchmarkError("Exact UInt64 oracle size differs at \(oracleURL.path).")
  }
  var mismatchCount = 0
  var maximumAbsoluteDifference: UInt64 = 0
  for index in actual.indices {
    let expected = data.withUnsafeBytes { raw in
      UInt64(
        littleEndian: raw.loadUnaligned(
          fromByteOffset: index * MemoryLayout<UInt64>.stride,
          as: UInt64.self
        )
      )
    }
    if actual[index] != expected {
      mismatchCount += 1
      maximumAbsoluteDifference = max(
        maximumAbsoluteDifference,
        actual[index] > expected
          ? actual[index] - expected : expected - actual[index]
      )
    }
  }
  return [
    "status": mismatchCount == 0 ? "pass" : "fail",
    "mismatch_count": mismatchCount,
    "maximum_absolute_difference": maximumAbsoluteDifference,
    "compared_values": actual.count,
    "actual_sha256": sha256Generic(values: actual),
    "oracle_path": oracleURL.path,
    "oracle_sha256": sha256(data: data),
  ]
}

private func compareFloat(
  actual: [Float],
  oracleURL: URL,
  absoluteTolerance: Double
) throws -> [String: Any] {
  let data = try Data(contentsOf: oracleURL)
  guard data.count == actual.count * MemoryLayout<Float>.stride else {
    throw benchmarkError("Float32 oracle size differs at \(oracleURL.path).")
  }
  var violationCount = 0
  var maximumAbsoluteError = 0.0
  var squaredError = 0.0
  for index in actual.indices {
    let expected = data.withUnsafeBytes { raw in
      Float(
        bitPattern: UInt32(
          littleEndian: raw.loadUnaligned(
            fromByteOffset: index * MemoryLayout<Float>.stride,
            as: UInt32.self
          )
        )
      )
    }
    let error = abs(Double(actual[index]) - Double(expected))
    if !actual[index].isFinite || error > absoluteTolerance {
      violationCount += 1
    }
    maximumAbsoluteError = max(maximumAbsoluteError, error)
    squaredError += error * error
  }
  return [
    "status": violationCount == 0 ? "pass" : "fail",
    "atol": absoluteTolerance,
    "rtol": 0.0,
    "violation_count": violationCount,
    "maximum_absolute_error": maximumAbsoluteError,
    "rms_error": sqrt(squaredError / Double(actual.count)),
    "compared_values": actual.count,
    "actual_sha256": sha256Generic(values: actual),
    "oracle_path": oracleURL.path,
    "oracle_sha256": sha256(data: data),
  ]
}

private func sha256Generic<T>(values: [T]) -> String {
  values.withUnsafeBytes { sha256(data: Data($0)) }
}

private func distribution(_ values: [Double]) -> [String: Any] {
  guard !values.isEmpty else {
    return ["n": 0, "p50": NSNull(), "p95": NSNull(), "max": NSNull()]
  }
  let sorted = values.sorted()
  func percentile(_ value: Double) -> Double {
    sorted[max(0, min(sorted.count - 1, Int(ceil(value * Double(sorted.count))) - 1))]
  }
  return [
    "n": sorted.count,
    "p50": percentile(0.50),
    "p95": percentile(0.95),
    "max": sorted.last!,
  ]
}

private func runTrajectory(
  source: MetalCompactH5ResidentSource,
  device: MTLDevice,
  runs: Int,
  statesPerRun: Int
) throws -> (receipt: [String: Any], pass: Bool) {
  let families = [
    "detector_center",
    "bf_radius",
    "abf_annulus",
    "adf_annulus",
    "custom_aperture",
  ]
  let allocatedBefore = device.currentAllocatedSize
  var traces: [[String: Any]] = []
  traces.reserveCapacity(families.count * runs * statesPerRun)
  var familyReceipts: [[String: Any]] = []
  var allAuditsPass = true
  var sequence = 0
  let auditStates = Set([0, (statesPerRun - 1) / 2, statesPerRun - 1])
  for family in families {
    var wall: [Double] = []
    var gpu: [Double] = []
    var overhead: [Double] = []
    var changedWall: [Double] = []
    var changedGPU: [Double] = []
    var zeroChangeStates = 0
    var auditCount = 0
    var auditFailures = 0
    var outputFingerprints = Set<String>()
    var parameterFingerprints = Set<String>()
    for run in 0..<runs {
      for state in 0..<statesPerRun {
        let inputNanoseconds = DispatchTime.now().uptimeNanoseconds
        let generated = trajectoryMask(
          family: family,
          state: state,
          run: run,
          runCount: runs,
          statesPerRun: statesPerRun,
          rows: source.metadata.detectorRows,
          columns: source.metadata.detectorColumns
        )
        let parameterData = try JSONSerialization.data(
          withJSONObject: generated.parameters,
          options: [.sortedKeys]
        )
        let parameterSHA = sha256(data: parameterData)
        let maskSHA = sha256(data: Data(generated.mask))
        let handlerNanoseconds = DispatchTime.now().uptimeNanoseconds
        let metrics = try source.updateVirtualDetector(
          mask: generated.mask,
          forceRebase: state == 0
        )
        let publicationNanoseconds = DispatchTime.now().uptimeNanoseconds
        let output = try source.virtualDetectorValues()
        let outputSHA = sha256(values: output)
        let fingerprintNanoseconds = DispatchTime.now().uptimeNanoseconds
        var auditStatus = "not_sampled"
        if auditStates.contains(state) {
          let incremental = output
          _ = try source.updateVirtualDetector(
            mask: generated.mask,
            forceRebase: true
          )
          let fresh = try source.virtualDetectorValues()
          let mismatchCount = zip(incremental, fresh).reduce(0) {
            $0 + ($1.0 == $1.1 ? 0 : 1)
          }
          auditCount += 1
          if mismatchCount != 0 {
            auditFailures += 1
            allAuditsPass = false
          }
          auditStatus = mismatchCount == 0 ? "pass" : "fail"
        }
        wall.append(metrics.wallMilliseconds)
        gpu.append(metrics.gpuMilliseconds)
        overhead.append(max(0, metrics.wallMilliseconds - metrics.gpuMilliseconds))
        if metrics.changedDetectorPixels == 0 {
          zeroChangeStates += 1
        } else {
          changedWall.append(metrics.wallMilliseconds)
          changedGPU.append(metrics.gpuMilliseconds)
        }
        outputFingerprints.insert(outputSHA)
        parameterFingerprints.insert(parameterSHA)
        traces.append([
          "run_id": run,
          "gesture_family": family,
          "request_sequence": sequence,
          "state_index": state,
          "dataset_generation": 0,
          "dataset_identity_sha256": source.metadata.sourceIdentitySHA256,
          "parameter_sha256": parameterSHA,
          "mask_sha256": maskSHA,
          "input_ns": inputNanoseconds,
          "handler_ready_ns": handlerNanoseconds,
          "publication_ns": publicationNanoseconds,
          "fingerprint_complete_ns": fingerprintNanoseconds,
          "render_submit_ns": NSNull(),
          "compositor_frame_token": NSNull(),
          "actual_present_ns": NSNull(),
          "terminal_status": "completed",
          "mode": metrics.mode,
          "changed_detector_pixels": metrics.changedDetectorPixels,
          "wall_ms": metrics.wallMilliseconds,
          "gpu_ms": metrics.gpuMilliseconds,
          "framework_overhead_ms": max(
            0,
            metrics.wallMilliseconds - metrics.gpuMilliseconds
          ),
          "output_sha256_u32_le": outputSHA,
          "fresh_rebase_audit": auditStatus,
          "parameters": generated.parameters,
        ])
        sequence += 1
      }
    }
    let wallDistribution = distribution(wall)
    let changedWallDistribution = distribution(changedWall)
    familyReceipts.append([
      "gesture_family": family,
      "run_count": runs,
      "requested_states": runs * statesPerRun,
      "completed_states": runs * statesPerRun,
      "superseded_states": 0,
      "cancelled_states": 0,
      "error_states": 0,
      "zero_change_states": zeroChangeStates,
      "unique_parameter_fingerprints": parameterFingerprints.count,
      "unique_output_fingerprints": outputFingerprints.count,
      "wall_ms": wallDistribution,
      "gpu_ms": distribution(gpu),
      "framework_overhead_ms": distribution(overhead),
      "changed_state_count": changedWall.count,
      "changed_state_wall_ms": changedWallDistribution,
      "changed_state_gpu_ms": distribution(changedGPU),
      "states_over_8_33_ms": wall.count(where: { $0 > 8.33 }),
      "changed_states_over_8_33_ms": changedWall.count(where: { $0 > 8.33 }),
      "fresh_rebase_audits": auditCount,
      "fresh_rebase_failures": auditFailures,
      "compute_publication_p95_lte_8_33":
        (wallDistribution["p95"] as? Double).map { $0 <= 8.33 } ?? false,
      "changed_state_p95_lte_8_33":
        (changedWallDistribution["p95"] as? Double).map { $0 <= 8.33 } ?? false,
    ])
  }
  let allocatedAfter = device.currentAllocatedSize
  return (
    [
      "schema": "quantem.gpu.swift-metal-compact-v3-trajectory/v1",
      "status": allAuditsPass ? "pass" : "fail",
      "protocol": [
        "runs_per_family": runs,
        "states_per_run": statesPerRun,
        "requested_states_per_family": runs * statesPerRun,
        "quantile_method": "nearest_rank",
        "synchronous_requests": true,
        "prefetch_active": false,
        "compute_target_ms": 8.33,
        "actual_presentation_gate": "pending",
      ],
      "memory": [
        "device_allocated_bytes_before_trajectory": allocatedBefore,
        "device_allocated_bytes_after_trajectory": allocatedAfter,
        "net_device_allocation_bytes": Int(allocatedAfter) - Int(allocatedBefore),
        "persistent_detector_entry_buffer_bytes": source.metadata.detectorPixelCount
          * MemoryLayout<UInt32>.stride * 2,
      ],
      "families": familyReceipts,
      "traces": traces,
    ],
    allAuditsPass
  )
}

private func trajectoryMask(
  family: String,
  state: Int,
  run: Int,
  runCount: Int,
  statesPerRun: Int,
  rows: Int,
  columns: Int
) -> (mask: [UInt8], parameters: [String: Double]) {
  let phase =
    2 * Double.pi
    * (Double(state) + Double(run) / Double(runCount))
    / Double(statesPerRun - 1)
  var centerRow = 95.5
  var centerColumn = 95.5
  var parameters: [String: Double]
  var mask = [UInt8](repeating: 0, count: rows * columns)
  if family == "custom_aperture" {
    centerRow += 3 * sin(phase)
    centerColumn += 3 * cos(phase)
    let angle = 0.25 * sin(phase)
    let cosine = cos(angle)
    let sine = sin(angle)
    for pixel in mask.indices {
      let relativeRow = Double(pixel / columns) - centerRow
      let relativeColumn = Double(pixel % columns) - centerColumn
      let rotatedRow = cosine * relativeRow + sine * relativeColumn
      let rotatedColumn = -sine * relativeRow + cosine * relativeColumn
      mask[pixel] =
        (rotatedRow / 13) * (rotatedRow / 13)
          + (rotatedColumn / 27) * (rotatedColumn / 27) <= 1 ? 1 : 0
    }
    parameters = [
      "center_row": centerRow,
      "center_column": centerColumn,
      "semi_axis_row": 13,
      "semi_axis_column": 27,
      "rotation_radians": angle,
    ]
    return (mask, parameters)
  }
  let inner: Double
  let outer: Double
  switch family {
  case "detector_center":
    centerRow += 2.5 * sin(phase)
    centerColumn += 2.5 * cos(phase)
    inner = 0
    outer = 48
  case "bf_radius":
    inner = 0
    outer = 46 + 2 * sin(phase)
  case "abf_annulus":
    inner = 24 + 1.5 * sin(phase)
    outer = 48 + 1.5 * cos(phase)
  default:
    inner = 48 + sin(phase)
    outer = 94 + cos(phase)
  }
  let innerSquared = inner * inner
  let outerSquared = outer * outer
  for pixel in mask.indices {
    let row = Double(pixel / columns)
    let column = Double(pixel % columns)
    let distanceSquared =
      (row - centerRow) * (row - centerRow)
      + (column - centerColumn) * (column - centerColumn)
    mask[pixel] =
      distanceSquared > innerSquared && distanceSquared <= outerSquared
      ? 1 : 0
  }
  parameters = [
    "center_row": centerRow,
    "center_column": centerColumn,
    "inner_radius_exclusive": inner,
    "outer_radius_inclusive": outer,
  ]
  return (mask, parameters)
}

private func sha256(values: [UInt32]) -> String {
  values.withUnsafeBytes { sha256(data: Data($0)) }
}

private func sha256(data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func number(_ value: Any?) -> Double {
  (value as? NSNumber)?.doubleValue ?? .nan
}

private func milliseconds(from start: ContinuousClock.Instant) -> Double {
  let duration = start.duration(to: .now)
  return Double(duration.components.seconds) * 1_000
    + Double(duration.components.attoseconds) / 1.0e15
}

private func parseOptions() throws -> Options {
  var arguments = Array(CommandLine.arguments.dropFirst())
  guard !arguments.isEmpty else {
    throw benchmarkError(
      "Usage: metal-compact-h5-benchmark SOURCE [--oracle-manifest PATH] "
        + "[--source-audit PATH] [--output PATH] [--interaction-repeats N] "
        + "[--load-repeats N] [--logical-hash] "
        + "[--dpc-oracle-manifest PATH] "
        + "[--resident-switch-source PATH] "
        + "[--parallel-mapped-authentication | --bounded-concurrent-authentication] [--trajectory-runs N] "
        + "[--trajectory-states-per-run N]"
    )
  }
  let source = URL(fileURLWithPath: arguments.removeFirst())
  var oracle: URL?
  var output: URL?
  var repeats = 3
  var switchSource: URL?
  var residentSwitchSource: URL?
  var logicalHash = false
  var sourceAudit: URL?
  var dpcOracleManifest: URL?
  var authenticationPolicy: MetalCompactH5AuthenticationPolicy = .boundedSequential
  var loadRepeats = 1
  var trajectoryRuns = 3
  var trajectoryStatesPerRun = 0
  while !arguments.isEmpty {
    let flag = arguments.removeFirst()
    if flag == "--logical-hash" {
      logicalHash = true
      continue
    }
    if flag == "--parallel-mapped-authentication" || flag == "--bounded-concurrent-authentication" {
      guard authenticationPolicy == .boundedSequential else {
        throw benchmarkError("Choose only one authentication policy.")
      }
      authenticationPolicy =
        flag == "--parallel-mapped-authentication"
        ? .parallelMapped : .boundedConcurrent
      continue
    }
    guard !arguments.isEmpty else { throw benchmarkError("Missing value for \(flag).") }
    let value = arguments.removeFirst()
    switch flag {
    case "--oracle-manifest": oracle = URL(fileURLWithPath: value)
    case "--output": output = URL(fileURLWithPath: value)
    case "--interaction-repeats":
      guard let parsed = Int(value), parsed >= 0 else {
        throw benchmarkError("Interaction repeats must be a nonnegative integer.")
      }
      repeats = parsed
    case "--load-repeats":
      guard let parsed = Int(value), parsed > 0 else {
        throw benchmarkError("Load repeats must be a positive integer.")
      }
      loadRepeats = parsed
    case "--trajectory-runs":
      guard let parsed = Int(value), parsed >= 3 else {
        throw benchmarkError("Trajectory runs must be at least three.")
      }
      trajectoryRuns = parsed
    case "--trajectory-states-per-run":
      guard let parsed = Int(value), parsed > 1 else {
        throw benchmarkError("Trajectory states per run must be greater than one.")
      }
      trajectoryStatesPerRun = parsed
    case "--switch-source": switchSource = URL(fileURLWithPath: value)
    case "--resident-switch-source":
      residentSwitchSource = URL(fileURLWithPath: value)
    case "--source-audit": sourceAudit = URL(fileURLWithPath: value)
    case "--dpc-oracle-manifest":
      dpcOracleManifest = URL(fileURLWithPath: value)
    default: throw benchmarkError("Unknown option \(flag).")
    }
  }
  return Options(
    sourceURL: source,
    oracleManifestURL: oracle,
    outputURL: output,
    interactionRepeats: repeats,
    switchSourceURL: switchSource,
    residentSwitchSourceURL: residentSwitchSource,
    logicalHash: logicalHash,
    sourceAuditURL: sourceAudit,
    dpcOracleManifestURL: dpcOracleManifest,
    authenticationPolicy: authenticationPolicy,
    loadRepeats: loadRepeats,
    trajectoryRuns: trajectoryRuns,
    trajectoryStatesPerRun: trajectoryStatesPerRun
  )
}

private func loadSample(
  sampleIndex: Int,
  load: MetalCompactH5LoadMetrics,
  firstResidentProductMilliseconds: Double
) -> [String: Any] {
  [
    "sample_index": sampleIndex,
    "resident_ready_ms": load.totalMilliseconds,
    "source_read_policy": load.sourceReadPolicy,
    "source_page_state": "source_pages_unspecified",
    "native_cache_status": load.nativeCacheStatus,
    "maximum_in_flight_shards": load.maximumInFlightShards,
    "shard_pipeline_wall_ms": load.shardPipelineMilliseconds,
    "descriptor_preparation_work_ms": load.descriptorPreparationMilliseconds,
    "gpu_decode_work_ms": load.gpuDecodeMilliseconds,
    "source_read_ms": load.sourceReadMilliseconds,
    "authentication_ms": load.decodedIntegrityMilliseconds,
    "private_upload_ms": load.privateUploadMilliseconds,
    "first_resident_product_ms": firstResidentProductMilliseconds,
    "ready_plus_first_product_ms": load.totalMilliseconds
      + firstResidentProductMilliseconds,
    "resident_bytes": load.residentBytes,
    "maximum_transient_bytes": load.maximumTransientBytes,
    "planned_additional_bytes": load.plannedAdditionalBytes,
    "mapped_authentication_bytes": load.mappedAuthenticationBytes,
    "device_allocated_bytes_before": load.deviceAllocatedBytesBefore,
    "device_allocated_bytes_after": load.deviceAllocatedBytesAfter,
  ]
}

private func processPeakResidentBytes() -> UInt64? {
  var usage = rusage()
  guard getrusage(RUSAGE_SELF, &usage) == 0, usage.ru_maxrss >= 0 else {
    return nil
  }
  return UInt64(usage.ru_maxrss)
}

private func authenticationPolicyName(
  _ policy: MetalCompactH5AuthenticationPolicy
) -> String {
  switch policy {
  case .boundedSequential: "bounded_sequential"
  case .boundedConcurrent: "bounded_concurrent_three_shards"
  case .parallelMapped: "parallel_mapped_full_file"
  }
}

private func benchmarkError(_ message: String) -> NSError {
  NSError(
    domain: "quantem.gpu.metal-compact-h5-benchmark",
    code: 1,
    userInfo: [NSLocalizedDescriptionKey: message]
  )
}

private func firstInvalidJSONValue(_ value: Any, path: String) -> String? {
  if value is NSNull || value is String || value is Bool || value is Int
    || value is UInt || value is UInt32 || value is UInt64 || value is Double
  {
    return nil
  }
  if let dictionary = value as? [String: Any] {
    for key in dictionary.keys.sorted() {
      if let invalid = firstInvalidJSONValue(
        dictionary[key]!,
        path: "\(path).\(key)"
      ) {
        return invalid
      }
    }
    return nil
  }
  if let array = value as? [Any] {
    for (index, element) in array.enumerated() {
      if let invalid = firstInvalidJSONValue(
        element,
        path: "\(path)[\(index)]"
      ) {
        return invalid
      }
    }
    return nil
  }
  return "\(path) (\(String(reflecting: type(of: value))))"
}
