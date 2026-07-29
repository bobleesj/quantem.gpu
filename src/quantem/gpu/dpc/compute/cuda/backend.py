"""CUDA center-of-mass dispatch."""

from quantem.gpu.detector.compute.cuda.kernels import cuda_center_of_mass


def center_of_mass(data, mask=None):
    """Run the exact CUDA center-of-mass kernel."""

    return cuda_center_of_mass(data, mask)
