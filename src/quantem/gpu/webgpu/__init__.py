"""Browser source manifest for consumer builds.

``sources.json`` lists the TypeScript and JSON resources of the WebGPU kernels
(detector, DPC, HDF5/bslz4 IO, SSB, device and QEM tables) as paths relative to
``quantem.gpu``. quantem.widget's build copies them from this manifest; this is
a build-time resource directory, not a Python GPU execution backend.
"""
