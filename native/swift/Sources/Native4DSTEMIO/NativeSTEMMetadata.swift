import Foundation

/// The legacy paired ARINA microscope header. No virtual image is decoded.
/// Pixel Size is in nm, High Tension in V, and Camera Length in m.
struct NativeSTEMMetadata {
  let raw: [String: String]

  init?(url: URL) {
    guard let header = try? NativeHDF5Bridge.inspectMaster(
      at: url, detectorRows: 0, detectorColumns: 0)
    else { return nil }
    raw = header.metadata.filter { $0.key.hasPrefix("STEM Metadata@") }
    guard !raw.isEmpty else { return nil }
  }

  private func number(_ name: String) -> Double? {
    guard let text = raw["STEM Metadata@" + name], let value = Double(text),
      value.isNormal, value > 0
    else { return nil }
    return value
  }

  var shape: (rows: Int, columns: Int)? {
    guard let row = number("NumPixelsY"), let column = number("NumPixelsX"),
      let rows = Int(exactly: row), let columns = Int(exactly: column)
    else { return nil }
    return (rows, columns)
  }

  var isCalibrated: Bool { number("IsPixelSizeCalibrated") == 1 }
  var nominalPixelNanometer: Double? {
    guard let pixel = number("Pixel Size"), (pixel * 10).isFinite else { return nil }
    return pixel
  }
  var sampling: (row: Double, column: Double)? {
    guard isCalibrated, let pixel = nominalPixelNanometer else { return nil }
    return (pixel, pixel)
  }

  func normalized(source: URL, master: NativeHDF5Master) -> [String: String] {
    var metadata = raw
    metadata["sourceFormat"] = "ARINA HDF5 + STEM metadata"
    metadata["microscope_metadata_source"] = source.lastPathComponent
    metadata["scan_sampling_calibrated"] = isCalibrated ? "true" : "false"
    var warnings: [String] = []
    if let pixel = nominalPixelNanometer {
      metadata["scan_sampling_nominal_A"] = String(pixel * 10)
      if !isCalibrated {
        warnings.append(
          "The recorded scan step is nominal (IsPixelSizeCalibrated is not 1). Confirm scan calibration before quantitative analysis.")
      }
    } else {
      warnings.append("The recorded scan step is invalid; supply a verified scan calibration.")
    }
    for (name, path, unit) in [
      ("High Tension", "electron_source/accelerating_voltage", "V"),
      ("Camera Length", "imaging_system/camera_length", "m"),
    ] {
      if let value = number(name) {
        metadata["electron_microscope/" + path] = String(value)
        metadata["electron_microscope/" + path + "@units"] = unit
      }
    }
    let detectorVoltage = NativeMicroscopeMetadata(metadata: master.metadata).voltageKV
    if let voltage = number("High Tension"), let detectorVoltage,
      abs(voltage / 1000 - detectorVoltage) > 1e-6 * detectorVoltage
    {
      warnings.append(
        "Detector photon_energy disagrees with microscope High Tension; the microscope voltage is used. Both recorded values are retained.")
    }
    if !warnings.isEmpty {
      metadata["microscope_metadata_warning"] = warnings.joined(separator: " ")
    }
    return metadata
  }
}
