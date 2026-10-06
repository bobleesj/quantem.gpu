"""Browser source graph for consumer builds.

``sources.json`` lists every TypeScript and JSON resource that ``index.ts``
reaches, as paths relative to ``quantem.gpu``. Consumer build scripts read that
manifest directly; this is a build-time resource directory, not a Python GPU
execution backend.
"""
