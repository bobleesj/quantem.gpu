// The former name of MetalDirectPtychoKernels, kept for one pin cycle so that code importing `MetalSSBKernels`
// still compiles. It re-exports MetalDirectPtychoKernels and maps each renamed public type to its new name; it has
// no code or resources of its own. The SSB-method types (`SSBPhaseArtifact`, `MetalSSBSavedRun`, ...) were not
// renamed and come through the re-export.
@_exported import MetalDirectPtychoKernels

@available(*, deprecated, renamed: "MetalDirectPtychoEngine")
public typealias MetalSSBEngine = MetalDirectPtychoEngine

@available(*, deprecated, renamed: "MetalDirectPtychoJointModel")
public typealias MetalSSBJointModel = MetalDirectPtychoJointModel

@available(*, deprecated, renamed: "MetalDirectPtychoJointPhase")
public typealias MetalSSBJointPhase = MetalDirectPtychoJointPhase

@available(*, deprecated, renamed: "MetalDirectPtychoCalibration")
public typealias MetalSSBCalibration = MetalDirectPtychoCalibration

@available(*, deprecated, renamed: "MetalDirectPtychoGeometry")
public typealias MetalSSBGeometry = MetalDirectPtychoGeometry

@available(*, deprecated, renamed: "MetalDirectPtychoAberrations")
public typealias MetalSSBAberrations = MetalDirectPtychoAberrations

@available(*, deprecated, renamed: "MetalDirectPtychoHigherOrder")
public typealias MetalSSBHigherOrder = MetalDirectPtychoHigherOrder

@available(*, deprecated, renamed: "MetalDirectPtychoBrightfieldEstimate")
public typealias MetalSSBBrightfieldEstimate = MetalDirectPtychoBrightfieldEstimate

@available(*, deprecated, renamed: "MetalDirectPtychoCountType")
public typealias MetalSSBCountType = MetalDirectPtychoCountType

@available(*, deprecated, renamed: "MetalDirectPtychoError")
public typealias MetalSSBError = MetalDirectPtychoError

@available(*, deprecated, renamed: "MetalDirectPtychoProvenance")
public typealias MetalSSBProvenance = MetalDirectPtychoProvenance

@available(*, deprecated, renamed: "MetalDirectPtychoResult")
public typealias MetalSSBResult = MetalDirectPtychoResult

@available(*, deprecated, renamed: "MetalDirectPtychoJointSavedRun")
public typealias MetalSSBJointSavedRun = MetalDirectPtychoJointSavedRun

@available(*, deprecated, renamed: "DirectPtychoFit")
public typealias SSBFit = DirectPtychoFit

@available(*, deprecated, renamed: "DirectPtychoFitOptions")
public typealias SSBFitOptions = DirectPtychoFitOptions

@available(*, deprecated, renamed: "DirectPtychoModelFit")
public typealias SSBModelFit = DirectPtychoModelFit

@available(*, deprecated, renamed: "DirectPtychoModelPoint")
public typealias SSBModelPoint = DirectPtychoModelPoint

@available(*, deprecated, renamed: "DirectPtychoModelHigherOrder")
public typealias SSBModelHigherOrder = DirectPtychoModelHigherOrder

@available(*, deprecated, renamed: "DirectPtychoBranchedFit")
public typealias SSBBranchedFit = DirectPtychoBranchedFit

@available(*, deprecated, renamed: "DirectPtychoAgreementSearch")
public typealias SSBAgreementSearch = DirectPtychoAgreementSearch

@available(*, deprecated, renamed: "DirectPtychoFifthOrderGradientSearch")
public typealias SSBFifthOrderGradientSearch = DirectPtychoFifthOrderGradientSearch

@available(*, deprecated, renamed: "DirectPtychoHigherOrderConsistency")
public typealias SSBHigherOrderConsistency = DirectPtychoHigherOrderConsistency

@available(*, deprecated, renamed: "DirectPtychoJointParameters")
public typealias SSBJointParameters = DirectPtychoJointParameters

@available(*, deprecated, renamed: "DirectPtychoJointTrial")
public typealias SSBJointTrial = DirectPtychoJointTrial

@available(*, deprecated, renamed: "DirectPtychoJointOptimizationResult")
public typealias SSBJointOptimizationResult = DirectPtychoJointOptimizationResult

@available(*, deprecated, renamed: "DirectPtychoJointRefinement")
public typealias SSBJointRefinement = DirectPtychoJointRefinement

@available(*, deprecated, renamed: "DirectPtychoObjective")
public typealias SSBObjective = DirectPtychoObjective

@available(*, deprecated, renamed: "DirectPtychoRotationBranches")
public typealias SSBRotationBranches = DirectPtychoRotationBranches

@available(*, deprecated, renamed: "DirectPtychoScanRegion")
public typealias SSBScanRegion = DirectPtychoScanRegion

@available(*, deprecated, renamed: "DirectPtychoOutputSampling")
public typealias SSBOutputSampling = DirectPtychoOutputSampling

@available(*, deprecated, renamed: "DirectPtychoUpsamplingChoice")
public typealias SSBUpsamplingChoice = DirectPtychoUpsamplingChoice

@available(*, deprecated, renamed: "DirectPtychoReconstructionMethod")
public typealias SSBReconstructionMethod = DirectPtychoReconstructionMethod

@available(*, deprecated, renamed: "DirectPtychoRegularization")
public typealias SSBDirectRegularization = DirectPtychoRegularization

@available(*, deprecated, renamed: "DirectPtychoSolveError")
public typealias SSBDirectError = DirectPtychoSolveError

@available(*, deprecated, renamed: "DirectPtychoDetectorWeighting")
public typealias SSBDetectorWeighting = DirectPtychoDetectorWeighting

@available(*, deprecated, renamed: "DirectPtychoDragControl")
public typealias SSBDragControl = DirectPtychoDragControl

@available(*, deprecated, renamed: "DirectPtychoDragPreview")
public typealias SSBDragPreview = DirectPtychoDragPreview

@available(*, deprecated, renamed: "DirectPtychoDragWindowState")
public typealias SSBDragWindowState = DirectPtychoDragWindowState
