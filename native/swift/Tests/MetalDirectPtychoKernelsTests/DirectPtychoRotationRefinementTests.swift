import Foundation
import Metal
import XCTest

@testable import MetalDirectPtychoKernels

final class DirectPtychoRotationRefinementTests: XCTestCase {
  func testMetalPriorPreservesTheInitialTrialsAndFifthOrderWorkflow() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try DirectFifthOrderGradientTests.model(device)
    let fixedOptions = DirectPtychoFitOptions(compareStandard: false, maximumOrder: 1)
    let fixed = try DirectPtychoModelFit.findAberrations(
      model: model, rotationDegrees: 17.25,
      darkField: nil, scanSampling: (0.5, 0.5), options: fixedOptions, checkRotation: false)
    let bounded = try DirectPtychoModelFit.findAberrations(
      model: model, rotationDegrees: 17.25,
      darkField: nil, scanSampling: (0.5, 0.5), options: fixedOptions, checkRotation: false,
      rotationSearchHalfWidthDegrees: 2)
    let search = try XCTUnwrap(bounded.rotationRefinement)
    XCTAssertEqual(search.initialAgreement, fixed.fit.sampleAgreement)
    XCTAssertGreaterThanOrEqual(search.finalAgreement, search.initialAgreement)
    XCTAssertTrue(search.evaluatedDegrees.allSatisfy { (15.25...19.25).contains($0) })
    XCTAssertEqual(bounded.fit.trialCount, 200)
    XCTAssertEqual(search.initialTrialCount, 200)
    let initialTimings = try XCTUnwrap(search.initialTimings)
    XCTAssertEqual(Set(initialTimings.keys), Set(fixed.fit.timings.keys))
    XCTAssertGreaterThan(try XCTUnwrap(initialTimings["coarse_search_s"]), 0)
    for (key, secondsOrCount) in initialTimings {
      XCTAssertGreaterThanOrEqual(try XCTUnwrap(bounded.fit.timings[key]), secondsOrCount)
    }
    XCTAssertGreaterThan(try XCTUnwrap(bounded.fit.timings["rotation_refinement_s"]), 0)
    XCTAssertFalse(bounded.comparedBranches)
    XCTAssertEqual(model.scanDetectorRotationDegrees, bounded.fit.scanDetectorRotationDegrees)
    let refined = try DirectPtychoModelFit.refineToFifthOrder(bounded, model: model)
    XCTAssertEqual(refined.fit.options.maximumOrder, 5)
    XCTAssertEqual(refined.fit.scanDetectorRotationDegrees, bounded.fit.scanDetectorRotationDegrees)
    let restored = try JSONDecoder().decode(
      DirectPtychoBranchedFit.self,
      from: JSONEncoder().encode(refined))
    XCTAssertEqual(restored.rotationRefinement?.evaluatedDegrees, search.evaluatedDegrees)
    XCTAssertEqual(restored.rotationRefinement?.initialTrialCount, 200)
    XCTAssertEqual(restored.rotationRefinement?.initialTimings, initialTimings)
    XCTAssertThrowsError(
      try DirectPtychoModelFit.findAberrations(
        model: model, rotationDegrees: 0,
        darkField: nil, scanSampling: (0.5, 0.5), rotationSearchHalfWidthDegrees: 10))
  }

  func testBoundedEntryRejectsInvalidOriginalOptionsBeforeRetargetingOrEvaluation() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try DirectFifthOrderGradientTests.model(device)
    let originalRotation = model.scanDetectorRotationDegrees
    let invalidOptions = [
      DirectPtychoFitOptions(maximumOrder: 2),
      DirectPtychoFitOptions(tilt: false, maximumOrder: 5),
      DirectPtychoFitOptions(tiltLimitMrad: 10, maximumOrder: 5),
    ]
    for options in invalidOptions {
      XCTAssertThrowsError(
        try DirectPtychoModelFit.findAberrations(
          model: model, rotationDegrees: originalRotation + 7,
          darkField: nil, scanSampling: (0.5, 0.5), options: options, checkRotation: false,
          progress: { _, _, _ in XCTFail("Invalid options must not start fitting") },
          rotationSearchHalfWidthDegrees: 2,
          onInitialFit: { _ in XCTFail("Invalid options must not publish a primary fit") },
          onRefinedFit: { _ in XCTFail("Invalid options must not publish a fifth-order fit") }))
      XCTAssertEqual(model.scanDetectorRotationDegrees, originalRotation)
    }
  }

  func testBoundedEntryCancelsBeforePublishingAnyFit() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try DirectFifthOrderGradientTests.model(device)
    XCTAssertThrowsError(
      try DirectPtychoModelFit.findAberrations(
        model: model, rotationDegrees: 17.25,
        darkField: nil, scanSampling: (0.5, 0.5), checkRotation: false,
        progress: { _, _, _ in XCTFail("Cancelled entry must not evaluate the fit") },
        rotationSearchHalfWidthDegrees: 2,
        onInitialFit: { _ in XCTFail("Cancelled entry must not publish a primary fit") },
        onRefinedFit: { _ in XCTFail("Cancelled entry must not publish a fifth-order fit") },
        isCancelled: { true })
    ) { XCTAssertTrue($0 is CancellationError) }
  }

  func testCancellationDuringAngularWalkDoesNotPublishOrRefine() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try DirectFifthOrderGradientTests.model(device)
    var cancelled = false
    var angularUpdates = 0
    XCTAssertThrowsError(
      try DirectPtychoModelFit.findAberrations(
        model: model, rotationDegrees: 17.25,
        darkField: nil, scanSampling: (0.5, 0.5), checkRotation: false,
        progress: { stage, _, _ in
          if stage == "Refining rotation" {
            angularUpdates += 1
            cancelled = true
          }
        }, rotationSearchHalfWidthDegrees: 2,
        onInitialFit: { _ in XCTFail("Cancelled angular search must not publish a primary fit") },
        onRefinedFit: { _ in XCTFail("Cancelled angular search must not enter fifth order") },
        isCancelled: { cancelled })
    ) { XCTAssertTrue($0 is CancellationError) }
    XCTAssertEqual(angularUpdates, 1)
  }

  func testNoTiltPrimaryRemainsNoTiltThroughoutBoundedRotation() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try DirectFifthOrderGradientTests.model(device)
    let options = DirectPtychoFitOptions(tilt: false, maximumOrder: 1)
    let bounded = try DirectPtychoModelFit.findAberrations(
      model: model, rotationDegrees: 17.25,
      darkField: nil, scanSampling: (0.5, 0.5), options: options, checkRotation: false,
      rotationSearchHalfWidthDegrees: 2,
      onRefinedFit: { _ in XCTFail("Primary-only fitting must not enter fifth order") })
    let search = try XCTUnwrap(bounded.rotationRefinement)
    XCTAssertFalse(bounded.fit.options.tilt)
    XCTAssertEqual(bounded.fit.options.maximumOrder, 1)
    XCTAssertEqual(bounded.fit.point.sampleTiltMrad, [0, 0])
    XCTAssertEqual(bounded.fit.point.depthSpreadAngstroms, 0)
    XCTAssertEqual(bounded.fit.trialCount, 200)
    XCTAssertNil(bounded.refinedFit)
    XCTAssertNil(bounded.fit.higherOrderRefinement)
    XCTAssertGreaterThanOrEqual(search.finalAgreement, search.initialAgreement)
    XCTAssertTrue(search.evaluatedDegrees.allSatisfy { (15.25...19.25).contains($0) })
    XCTAssertEqual(model.scanDetectorRotationDegrees, bounded.fit.scanDetectorRotationDegrees)
  }

  func testRequestedStandardComparisonUsesTheSelectedRotation() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try DirectFifthOrderGradientTests.model(device)
    let options = DirectPtychoFitOptions(compareStandard: true, maximumOrder: 1)
    let bounded = try DirectPtychoModelFit.findAberrations(
      model: model, rotationDegrees: 17.25,
      darkField: nil, scanSampling: (0.5, 0.5), options: options, checkRotation: false,
      rotationSearchHalfWidthDegrees: 2)
    let agreement = try XCTUnwrap(bounded.fit.standardAgreement)
    XCTAssertGreaterThan(agreement, 0)
    XCTAssertTrue(bounded.fit.options.compareStandard)
    XCTAssertGreaterThan(try XCTUnwrap(bounded.fit.timings["standard_comparison_s"]), 0)
    XCTAssertEqual(
      try XCTUnwrap(bounded.fit.tiltFitGain),
      try XCTUnwrap(bounded.fit.sampleAgreement) / agreement, accuracy: 1e-12)
    let fixedAtSelectedRotation = try DirectPtychoModelFit.findAberrations(
      model: model, rotationDegrees: bounded.fit.scanDetectorRotationDegrees,
      darkField: nil, scanSampling: (0.5, 0.5), options: options, checkRotation: false)
    XCTAssertEqual(
      bounded.fit.rawStandardAgreement, fixedAtSelectedRotation.fit.rawStandardAgreement)
    XCTAssertEqual(bounded.fit.standardAgreement, fixedAtSelectedRotation.fit.standardAgreement)
  }

  func testOmittedAndNilPriorKeepTheSameFixedRotationFit() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal required") }
    let model = try DirectFifthOrderGradientTests.model(device)
    let options = DirectPtychoFitOptions(compareStandard: false, maximumOrder: 1)
    let omitted = try DirectPtychoModelFit.findAberrations(
      model: model, rotationDegrees: 17.25,
      darkField: nil, scanSampling: (0.5, 0.5), options: options, checkRotation: false)
    let explicitNil = try DirectPtychoModelFit.findAberrations(
      model: model, rotationDegrees: 17.25,
      darkField: nil, scanSampling: (0.5, 0.5), options: options, checkRotation: false,
      rotationSearchHalfWidthDegrees: nil)
    XCTAssertNil(omitted.rotationRefinement)
    XCTAssertNil(explicitNil.rotationRefinement)
    XCTAssertEqual(explicitNil.fit.point, omitted.fit.point)
    XCTAssertEqual(explicitNil.fit.rawSampleAgreement, omitted.fit.rawSampleAgreement)
    XCTAssertEqual(explicitNil.fit.bandPower, omitted.fit.bandPower)
    XCTAssertEqual(explicitNil.fit.options, omitted.fit.options)
    XCTAssertEqual(explicitNil.fit.trialCount, omitted.fit.trialCount)
    XCTAssertEqual(
      explicitNil.fit.scanDetectorRotationDegrees, omitted.fit.scanDetectorRotationDegrees)
  }

  func testInteriorPeakAndNoOutOfWindowEvaluation() throws {
    for target in [-9.3, -2.37, 0, 0.4, 3.17, 9.3] {
      var evaluated: [Double] = []
      let result = try DirectPtychoBoundedRotationSearch.search(halfWidth: 10) { angle in
        evaluated.append(angle)
        return 1 - pow(angle - target, 2)
      }
      XCTAssertEqual(result, target, accuracy: 0.05)
      XCTAssertTrue(evaluated.allSatisfy { abs($0) <= 10 })
      XCTAssertEqual(Set(evaluated).count, evaluated.count)
    }
  }
  func testBoundaryAndNarrowPrior() throws {
    for halfWidth in [0.1, 2.3, 10.0] {
      let result = try DirectPtychoBoundedRotationSearch.search(halfWidth: halfWidth) { $0 }
      XCTAssertEqual(result, halfWidth)
    }
    let evidence = DirectPtychoRotationRefinement(
      centerDegrees: 90, halfWidthDegrees: 10,
      fittedDegrees: 100, evaluatedDegrees: [90, 100], initialAgreement: 0.1, finalAgreement: 0.2,
      seconds: 1)
    XCTAssertTrue(evidence.hitBoundary)
  }
  func testFlatObjectiveRetainsThePrior() throws {
    XCTAssertEqual(try DirectPtychoBoundedRotationSearch.search(halfWidth: 10) { _ in 1 }, 0)
  }
  func testInvalidPriorNonfiniteScoreAndCancellationFailClosed() {
    for width in [0, -1, .nan, .infinity, 90] {
      XCTAssertThrowsError(
        try DirectPtychoBoundedRotationSearch.search(halfWidth: width) { _ in 1 })
    }
    XCTAssertThrowsError(try DirectPtychoBoundedRotationSearch.search(halfWidth: 10) { _ in .nan })
    XCTAssertThrowsError(
      try DirectPtychoBoundedRotationSearch.search(
        halfWidth: 10,
        evaluate: { _ in
          XCTFail("Cancelled search must not evaluate")
          return 1
        }, isCancelled: { true }))
  }
}
