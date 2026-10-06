"""MPS bitshuffle+LZ4 decoding.

Importing ``decode`` compiles Metal kernels, so the package itself imports no
Metal module and stays importable on Linux and Windows.
"""
