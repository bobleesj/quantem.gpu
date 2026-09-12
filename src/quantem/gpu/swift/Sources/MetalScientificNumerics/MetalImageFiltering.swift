import Foundation
import Metal
import MetalPerformanceShadersGraph

struct ImageConvolutionPlan {
  let graph: MPSGraph
  let input, weights, output: MPSGraphTensor
}

extension MetalImageOperations {
  /// Reflect-padded Gaussian filtering with a normalized two-dimensional kernel.
  public func gaussian(_ image: GPUImage, sigma: Double) throws -> GPUImage {
    guard sigma.isFinite else { throw Self.invalid("Use a finite Gaussian sigma.") }
    if sigma <= 0 { return image }
    guard 2 * sigma < Double(min(image.rows, image.columns)) else {
      throw Self.invalid("Gaussian reflection padding must be smaller than the image.")
    }
    let width = 2 * Int(2 * sigma) + 1
    let weights = try buffer(width * 4)
    try run("gaussian_pdf", [weights], words: [UInt32(width)], floats: [Float(sigma)], count: width)
    let total = try sum(GPUImage(buffer: weights, rows: 1, columns: width))
    try run("normalize_pdf", [weights, total], words: [UInt32(width)], count: width)
    return try convolve(image, weights: weights, width: width, gaussian: true)
  }

  public func gradientMagnitude(_ image: GPUImage, sigma: Double) throws -> GPUImage {
    guard image.rows > 1, image.columns > 1 else {
      throw Self.invalid("Gradient reflection padding requires at least two rows and columns.")
    }
    let weights = try buffer(9 * 4)
    try run("sobel_weights", [weights], words: [0], count: 9)
    let row = try convolve(image, weights: weights, width: 3, gaussian: false)
    try run("sobel_weights", [weights], words: [1], count: 9)
    let column = try convolve(image, weights: weights, width: 3, gaussian: false)
    let a = try gaussian(row, sigma: sigma)
    let b = try gaussian(column, sigma: sigma)
    let result = try allocate(image.rows, image.columns)
    try run(
      "magnitude", [a.buffer, b.buffer, result.buffer],
      words: [UInt32(image.rows * image.columns)], count: image.rows * image.columns)
    return result
  }

  private func convolve(_ image: GPUImage, weights: MTLBuffer, width: Int, gaussian: Bool)
    throws -> GPUImage
  {
    let key = "\(image.rows),\(image.columns),\(width),\(gaussian)"
    let imageShape = [1, 1, image.rows, image.columns].map(NSNumber.init)
    let weightsShape = [NSNumber(value: gaussian ? width : width * width)]
    if convolutionPlans[key] == nil {
      let graph = MPSGraph()
      graph.options = .none
      let input = graph.placeholder(shape: imageShape, dataType: .float32, name: nil)
      let rawWeights = graph.placeholder(shape: weightsShape, dataType: .float32, name: nil)
      let kernel: MPSGraphTensor
      if gaussian {
        let column = graph.reshape(rawWeights, shape: [NSNumber(value: width), 1], name: nil)
        let row = graph.reshape(rawWeights, shape: [1, NSNumber(value: width)], name: nil)
        kernel = graph.reshape(
          graph.multiplication(column, row, name: nil),
          shape: [1, 1, NSNumber(value: width), NSNumber(value: width)], name: nil)
      } else {
        kernel = graph.reshape(
          rawWeights,
          shape: [1, 1, NSNumber(value: width), NSNumber(value: width)], name: nil)
      }
      let pad = NSNumber(value: width / 2)
      let padded = graph.padTensor(
        input, with: .reflect,
        leftPadding: [0, 0, pad, pad], rightPadding: [0, 0, pad, pad],
        constantValue: 0, name: nil)
      let descriptor = MPSGraphConvolution2DOpDescriptor(
        strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1,
        groups: 1, paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0,
        paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
      let output = graph.convolution2D(padded, weights: kernel, descriptor: descriptor, name: nil)
      convolutionPlans[key] = ImageConvolutionPlan(
        graph: graph, input: input,
        weights: rawWeights, output: output)
    }
    let plan = convolutionPlans[key]!
    let result = try allocate(image.rows, image.columns)
    plan.graph.run(
      with: queue,
      feeds: [
        plan.input: MPSGraphTensorData(image.buffer, shape: imageShape, dataType: .float32),
        plan.weights: MPSGraphTensorData(weights, shape: weightsShape, dataType: .float32),
      ], targetOperations: nil,
      resultsDictionary: [
        plan.output: MPSGraphTensorData(result.buffer, shape: imageShape, dataType: .float32)
      ])
    return result
  }
}
