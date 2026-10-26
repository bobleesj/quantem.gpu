import Foundation
import XCTest

@testable import MetalSSBKernels

final class SSBFifthOrderSearchTests: XCTestCase {
  func testStandardDefaultAndLegacySavedOptions() throws {
    let current = SSBFitOptions()
    XCTAssertEqual(current.trials, 200)
    XCTAssertEqual(current.maximumOrder, 5)
    XCTAssertEqual(current.maxHigherOrderEvaluations, 5000)
    XCTAssertFalse(current.compareStandard)
    XCTAssertEqual(SSBFitOptions(tilt: false).maximumOrder, 1)
    // The retired phase-variance fit still decodes from a saved run, and is refused as a fit.
    var retired = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(SSBFitOptions(tilt: false)))
        as? [String: Any])
    retired["objective"] = "phase_variance"
    let saved = try JSONDecoder().decode(
      SSBFitOptions.self, from: JSONSerialization.data(withJSONObject: retired))
    XCTAssertEqual(saved.objective, .phaseVariance)
    XCTAssertEqual(saved.method, "phase variance")
    XCTAssertThrowsError(try saved.validate()) { error in
      XCTAssertTrue(error.localizedDescription.contains("retired"))
    }
    let encoded = try JSONEncoder().encode(current)
    XCTAssertEqual(try JSONDecoder().decode(SSBFitOptions.self, from: encoded), current)
    var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    legacy.removeValue(forKey: "maximumOrder")
    legacy.removeValue(forKey: "maxHigherOrderEvaluations")
    legacy.removeValue(forKey: "higherOrderSearch")
    let restored = try JSONDecoder().decode(
      SSBFitOptions.self, from: JSONSerialization.data(withJSONObject: legacy))
    XCTAssertEqual(restored.maximumOrder, 1)
    XCTAssertThrowsError(try SSBFitOptions(tilt: false, maximumOrder: 5).validate())
    XCTAssertThrowsError(try SSBFitOptions(maximumOrder: 3).validate())
    // Options saved before the exact-gradient search name the retired finite-difference Newton: they decode, a
    // primary-only fit validates, and a fifth-order fit with it is refused.
    XCTAssertEqual(current.higherOrderSearch, .exactGradientCoarseToFine)
    XCTAssertEqual(restored.higherOrderSearch, .finiteDifferenceNewton)
    XCTAssertNoThrow(try restored.validate())
    var newton = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    newton["higherOrderSearch"] = "finiteDifferenceNewton"
    let savedNewton = try JSONDecoder().decode(
      SSBFitOptions.self, from: JSONSerialization.data(withJSONObject: newton))
    XCTAssertEqual(savedNewton.maximumOrder, 5)
    XCTAssertThrowsError(try savedNewton.validate()) { error in
      XCTAssertTrue(error.localizedDescription.contains("retired"))
    }
  }

  func testCartesianToPublicPolarMatchesFloat32Reference() {
    let coordinates = SSBFifthOrderCoordinates(
      wavelengthAngstroms: 0.019687489,
      semiangleRadians: 0.03)
    // Anonymous synthetic coordinates frozen from the Python Float32 pack/public conversion.
    let z: [Double] = [
      -1.2148537333180551, -0.02590920140154509, 1.5701581542310723,
      -0.5076410025895716, 0.44410351593545533, 0.2,
      -0.09726977141422134, -1.131280169655178, -0.13046010805877908,
      1.2691678349495035, -0.8160884741983179, -0.5330417578288759,
      0.10574935462407158, -1.3756949965907002, 1.4630812229731325,
      0.6325716777037829, -0.2264357105774506, 0.14724315261372992,
      -0.6920755999218762, 0.8242429216439771, -1.2022212308886475,
      1.4191766206550835, -1.0354852583601648, -0.32229486246216194,
      -0.09525845318648304, -1.4382900565469332, 0.3837394139109495,
      0.06826902058444961,
    ]
    let expectedBits: [UInt32] = [
      3_262_314_627, 0, 3_215_315_363, 1_117_586_444, 3_241_450_613, 3_271_772_441, 3_245_489_468,
      1_125_337_495, 3_309_653_662, 0, 3_305_168_857, 1_137_478_049, 3_316_010_324, 1_169_218_982,
      1_201_622_899, 3_336_838_006, 1_184_126_082, 3_350_088_656, 1_204_786_422, 3_356_975_891,
      1_253_713_690,
      0, 3_397_899_002, 3_383_302_624, 3_368_545_165, 3_401_361_644, 1_237_931_766, 1_217_349_357,
    ]
    let expected: [(order: Int, symmetry: Int, nanometers: Double, degrees: Double)] = [
      (2, 1, 39.53097915649414, -94.91432189941406),
      (2, 3, 44.4190673828125, 31.95630645751953),
      (3, 0, -1262.765380859375, 0.0),
      (3, 2, 840.8707275390625, 84.38943481445312),
      (3, 4, 3107.47216796875, 33.309200286865234),
      (4, 1, 43317.625, -19.695362091064453),
      (4, 3, 45618.49609375, -25.996355056762695),
      (4, 5, 93977.578125, -11.113092422485352),
      (5, 0, 3659912.0, 0.0),
      (5, 2, 2796771.5, -81.3556137084961),
      (5, 4, 3717329.5, -23.447296142578125),
      (5, 6, 1005163.6875, 1.6812739372253418),
    ]
    XCTAssertEqual(coordinates.pack(z).map(\.bitPattern), expectedBits)
    let point = coordinates.point(z)
    XCTAssertEqual(point.higherOrder.count, expected.count)
    for (actual, reference) in zip(point.higherOrder, expected) {
      XCTAssertEqual(actual.order, reference.order)
      XCTAssertEqual(actual.symmetry, reference.symmetry)
      XCTAssertEqual(actual.magnitudeAngstroms, reference.nanometers * 10)
      // Float.pi differs by one ULP from Float(Double.pi), changing these public azimuths.
      XCTAssertEqual(actual.angleRadians, reference.degrees * Double.pi / 180)
    }
  }
}
