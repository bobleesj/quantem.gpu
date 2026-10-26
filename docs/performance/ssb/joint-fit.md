# Full-field and regional Direct Ptycho benchmark

Measure the installed Direct Ptycho fitter on your own H5 or QEM acquisition:

```bash
quantem-gpu benchmark-direct-ptycho scan.qem --out /tmp/ssb-benchmark.json
```

`quantem-gpu benchmark-ssb` is the same command under its earlier name.

CUDA or MPS is selected automatically. The command uses one seed, 200 trials
and Newton refinement per fit. An estimated full-field rotation checks both
180-degree branches, with 200 trials per branch. Regional fits keep the selected
rotation and jointly fit aberrations, sample tilt and depth. Every regional view
is a crop of a full-scan reconstruction with that region's fitted parameters.

The default sequence fits the full field, 256 × 256 and 128 × 128 regions,
then moves and resizes the smaller region. It retains all scan positions and
bright-field detector pixels. Output sampling is Auto. The 127 × 128 resize
also exercises the non-power-of-two FFT path without removing measured data.

Saved acquisition calibration is reused. Supply missing physical calibration
explicitly when an H5 file does not contain it:

```bash
quantem-gpu benchmark-direct-ptycho scan_master.h5 --out /tmp/ssb-benchmark.json \
  --voltage-kv 300 --semiangle-mrad 30 --scan-sampling-a 0.5
```

Missing detector sampling is fitted automatically. To use a measured detector
calibration, add `--detector-sampling-mrad 0.46`. Row and column sampling can be
given as two numbers. Source files are never changed.

## Public example

Without a local acquisition, use the same public gold scan as the release gate
(`huggingface_hub` is needed for the download):

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from quantem.gpu.benchmark import run_direct_ptycho

root = snapshot_download(
    "bobleesj/quantem-data", repo_type="dataset",
    revision="5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20",
    allow_patterns="4dstem/gold_512/*",
)
report = run_direct_ptycho(
    Path(root) / "4dstem/gold_512/gold_004_master.h5",
    calibration={"semiangle_mrad": 30, "scan_sampling_A": 0.5,
                 "detector_sampling_mrad": 0.46},
)
```

Those calibration values belong to this public acquisition. Use your own
recorded values for other files. Different specimens and region positions can
require different numbers of Newton steps even with the same trial budget.

## Read the timings

Six measurements give one first observation and five warm observations. These
are benchmark repetitions, not five searches during one widget interaction.
The report contains the first time and warm median/p90, actual trial counts,
Newton diagnostics, GPU identity, installed versions, source-code fingerprint,
full input dimensions, output sampling and repeated-result parity.

`full_field_fit` includes both rotation branches when estimated. The separate
`full_field_reconstruction` row measures Auto output generation. Each regional
row includes region preparation, its joint fit, full-field reconstruction and
the final crop. The `fits` records break these components down. Loading and
session construction are separate. GPU work is synchronized before stopping
each timer. These are compute timings; browser transfer and paint are excluded.

The first measurement is not a cold-disk test. Use a quiet GPU slot, respect
the host's compute lock and preserve other users' jobs. Compare identical
calibration and dimensions, and keep
first-use costs separate from warmed interaction times. Do not treat a faster
single observation as a demonstrated speedup.

The JSON omits source paths, acquisition names, host names and images. Keep raw
reports outside this public repository. Public documentation should contain
only the reviewed setup and timing summary. Source fingerprints and phase
hashes identify the tested implementation and outputs without publishing data.

For a before/after comparison, retain the known-good Auto phase outside the
repository and add `--reference-phase /path/to/reference.npy`. The command
compares the full array without alignment or resampling at `atol=rtol=1e-6`.
Use a reference from the same backend, acquisition and calibration. A mismatch
fails the run; the reference is never overwritten. Repeated self-consistency
alone does not establish parity with a previous implementation.

## Redraws across output sampling

For a calibrated session that is already open in a notebook, measure live
defocus changes separately from fitting:

```python
from quantem.gpu.benchmark import run_direct_ptycho_preview

report = run_direct_ptycho_preview(ssb, upsampling=(1, 2, 3, 4, 8), runs=11)
```

This keeps all active aberrations, tilt, depth and BF pixels. Each observation
changes C10 by 0.1 nm; the original session and saved calibration remain
unchanged. Results separate the first observation from warm samples and record
the full settings and output hashes. CUDA reconstruction and download are
separate; MPS preview already includes delivery to a host array. The
`phase_delivery` distribution is the comparable combined boundary.

These are reconstruction timings, **not notebook FPS**. Measure delivery to
the notebook and new scientific canvas frames separately. Changing the output
factor changes the evaluated frequency range; comparing 1x with 8x is a
workload comparison, not a quality-preserving speedup. Hashes do not replace
comparison with frozen arrays from the previous implementation.

## Release qualification

Run the command in a new environment installed from the candidate wheel:

```bash
quantem-gpu benchmark-direct-ptycho scan.qem --require-installed \
  --backend mps --out /tmp/ssb-release-mps.json
```

Run independently with `--backend cuda` on CUDA. Missing hardware, memory
allocation failures and skipped runs do not qualify a release. The command
rejects CPU numerical search, incomplete trial budgets, nonfinite phases and
changed repeated results. It checks both native and Auto full-field phase.

The GPU release workflow tests the exact wheel on self-hosted `quantem-cuda`
and `quantem-mps` runners before publishing. The widget release adds a fresh
Jupyter kernel and real browser drawing, moving, resizing, saving and restoring
regions. Both runners must be registered and available; an Ubuntu build alone
does not qualify either GPU backend.

Configure repository variables `SSB_CUDA_REFERENCE_PHASE` and
`SSB_MPS_REFERENCE_PHASE` with runner-local paths to approved Auto phase arrays
for the pinned public gold acquisition and calibration above. Freeze these
from the known-good GPU implementation before testing a candidate. Keep their
source and calibration provenance in the private evidence archive. A missing
reference or a mismatch fails publication; do not regenerate it to make a
candidate pass. These hardware gates do not replace the other repository tests
or publication/privacy checks.
