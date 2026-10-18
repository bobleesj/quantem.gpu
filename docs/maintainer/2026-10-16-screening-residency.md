# Reusing acquisition storage for screening and SSB

Screening and SSB can borrow the same `io.load` result. The caller owns its
lifetime and closes it after both workflows finish.

```python
from quantem.gpu import io, screening, SSB

with io.load(path) as data:
    preview = screening.prepare(data)
    with SSB(data, source_path=str(path), **calibration) as ssb:
        result = ssb.find_aberrations(tilt=True)
```

Screening excludes the stored flagged detector pixels from exact count sums,
CoM and the mean diffraction pattern. SSB retains the loader's median
replacement. Sharing storage does not change these separate policies.
Borrowed screening bypasses the file-product cache and never closes the source.
SSB records the source dtype, logical size and hot-pixel correction. Sessions
with a source path use their file signature; they do not calculate unused
resident checksums. Sessions without a source path retain resident identity.

The fit remains 200 adaptive trials plus Newton. No detector pixels, scan
positions or floating-point precision are removed. Parameter diagnostics use
batches of four on MPS, preserving each candidate's arithmetic; search batches
remain unchanged.

Validation uses exact preview counts, phase and loss on physical CUDA and MPS,
including flagged pixels inside the bright-field disk. Full acquisition
workflow timings and scientific comparisons are retained in the private
`joint-workflow-benchmark-20261016` evidence bundle. These changes do not qualify
native application UI behavior or establish a theoretical speed limit.
