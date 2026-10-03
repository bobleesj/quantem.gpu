# From acquisition to images

Load once, select the measurements you need, and calculate images from Python.
The same public calls work on supported MPS and CUDA paths. Follow the
[installation guide](install.md) first.

These examples use a Gold acquisition with a 512 × 512 scan and a 192 × 192
detector. Replace `gold_master.h5` with your file and keep its companion files
beside it. The acquisition is not bundled with the package.

## Load and inspect

```python
from functools import partial

from quantem.gpu import io, detector
from quantem.core.visualization import show_2d

show_2d = partial(show_2d, cmap="inferno")
data = io.load("gold_master.h5")
data.shape, data.dtype, data.sampling, data.units
```

The axes are `(scan_rows, scan_cols, detector_rows, detector_cols)`.
`data.metadata` holds the full record; unknown sampling or units remain `None`.
Keep `data` open across the following cells. The acquisition stays ANS encoded
on the selected GPU.

## Select patterns and detector regions

```python
show_2d(data[10, 12], norm="power_sqrt")
show_2d([data[10, 12], data[8, 10], data[0, 0]], norm="power_sqrt")
```

The second call shows three selected scan positions without a loop.
Selections are ordinary GPU Torch tensors. Indexing accepts integers, slices,
and ellipsis; index arrays and boolean masks are not supported. You can also select part of a pattern:

```python
pattern = data[10, 12]
crop = data[10, 12, 64:128, 64:128]
show_2d([pattern, crop], title=["Pattern", "Detector crop"], norm="power_sqrt")
```

![Gold pattern and selected detector crop](_static/gold-pattern-crop.png)

The illustration outlines the selected crop on the full pattern. The crop
selects detector pixels, not scan positions. A scan patch instead uses
`data[8:12, 10:16]`. See [I/O](api/io.md) for full indexing and 5D acquisition
selection. Each image above uses independent display contrast.

## Make BF and ADF images

```python
bf = detector.bf(data)
adf = detector.adf(data)
show_2d([bf, adf], title=["BF", "ADF"], norm="power_sqrt")
```

The first detector call fits the bright-field disk automatically. Subsequent
calls reuse that geometry for the same encoded acquisition. BF sums its central
disk; ADF sums the ring from one to two fitted radii within the detector.
These reduced images are NumPy arrays; the full acquisition stays on the GPU.

![Gold BF and ADF images](_static/gold-bf-adf.png)

Each panel covers the full 512 × 512 scan with independent square-root contrast.
Physical sampling is absent from this source, so the figures do not invent a
physical scale bar.

For the illustrated Gold file, loading corrects its four flagged detector pixels
by default. Set `hot_pixel_correction="none"` in `io.load` to retain the original
measurements. The metadata records the correction.

![Default Gold detector selections on the mean diffraction pattern beside their images](_static/gold-detector-preview.png)

The selection figure illustrates the automatic defaults, using the
[README's static preview recipe](https://github.com/bobleesj/quantem.gpu#preview-the-detector-selection).
Override the selection only when needed:

```python
bf = detector.bf(data, radius=45)
adf = detector.adf(data, inner=60, outer=85, unit="px")
```

Those manual radii differ from the automatic selection illustrated above.
For centers, physical-angle units, dark field, and fit inspection, see
[Detector and DPC API](api/images_dpc.md).

## Calculate DPC

```python
from quantem.gpu import dpc

result = dpc.run(data)
show_2d(result.phase, title="Integrated DPC phase")
```

The result contains CoM images and `rotation_deg`. DPC searches scan-detector
rotation unless you provide `rotation_angle_deg`. Its rotation estimate and
SSB aberration search solve different parts of the calibration.

(reconstruct-phase)=
## Reconstruct phase

Provide your microscope calibration explicitly. The numerical values below
illustrate the units; replace them with values verified for your acquisition.

```python
from quantem.gpu import SSB

ssb = SSB(data, voltage_kV=300, semiangle_mrad=30, scan_sampling_A=0.264)
aberrations = ssb.find_aberrations()
result = ssb.reconstruct(aberrations)
show_2d(result.phase, scalebar={"sampling": result.scan_sampling_A, "units": "Å"})
```

Inspect the result and the search before interpreting the phase:

```python
aberrations.report()
ssb.show_trials(best=5)
aberrations.aberrations["C10"]  # defocus, nm
```

`find_aberrations()` searches; `reconstruct(aberrations)` applies the selected
parameters. C10 and C12 are in nm; phi12 is in radians. For a joint specimen-tilt
and model depth-spread search, use `find_aberrations(tilt=True)`. Model depth
spread is not a thickness measurement.

On CUDA, `ssb.reconstruct(aberrations, upsample=2)` provides finer output
sampling with the same fitted parameters. MPS currently supports native output
sampling only. More output pixels do not guarantee more resolved detail.
Check `ssb.scan_shape`: the CUDA dense-array path may pad or crop a scan to a
supported grid. The low-level compact-source path rejects unsupported grids. An acquisition
loaded with `io.load` can instead enter the dense path after bright-field
selection, so check the working grid rather than inferring it from storage. See the
[SSB API](api/ssb.md) for those limits, saved results, and trial replay.

## Save and finish

```python
io.save("gold.qem", data)
ssb.close()
data.close()
```

Saving writes a new acquisition file and refuses to overwrite an existing path.
Close the SSB session before its borrowed acquisition. Keep both open while a
viewer or calculation still needs them. Continue with
[Save and share your data](api/qem-python.md) for reopening, metadata, and sharing.
