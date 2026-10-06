import Foundation

@main struct ARINAMicroscopeMetadataCheck {
  static func main() {
    let root = "entry/instrument/detector/"
    var fields = [root + "description": "Dectris ARINA Si",
      root + "detectorSpecific/photon_energy": "200000",
      root + "frame_time": "0.0000496"]
    let initial = NativeMicroscopeMetadata(metadata: fields)
    precondition(initial.beamEnergyKeV == 200)
    precondition(abs(initial.dwellTimeMicroseconds! - 49.6) < 1e-12)
    fields[root + "count_time"] = "49.5"
    fields[root + "count_time@units"] = "us"
    precondition(NativeMicroscopeMetadata(metadata: fields).dwellTimeMicroseconds == 49.5)
    fields["electron_microscope/electron_source/accelerating_voltage"] = "300 kV"
    precondition(NativeMicroscopeMetadata(metadata: fields).beamEnergyKeV == 300)
    fields.removeValue(forKey: "electron_microscope/electron_source/accelerating_voltage")
    fields[root + "description"] = "X-ray detector"
    precondition(NativeMicroscopeMetadata(metadata: fields).beamEnergyKeV == nil)
    print("ARINA microscope metadata: passed")
  }
}
