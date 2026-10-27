# Stored SSB fits carry a schema, and only the current one is read - 2026-10-11

> **Historical:** describes the code as of 2026-10-11; the `mean_phase` estimator is retired, and later schema versions replaced the one described. See the [current fit methods](../api/ssb.md).

## Question

A screening folder keeps each acquisition's fitted SSB values in `live/screen/<dataset>/config.json` under
`computed.ssb`. Values stored before the package restructure reconstruct a different image with the current engine
(two real scans: inverted contrast on one, a lattice peak six times weaker on the other, while a fresh fit reproduces
the earlier pictures). Which stored fields changed meaning, and how does a reader tell a current fit from an earlier one?

## Setup

One synthetic scan, simulated in the probe script itself: a periodic weak phase object with positive phase at the
atom columns, 300 kV, 30 mrad, defocus -10 nm, 64 x 64 positions at 0.5 A, 128 x 128 detector at 0.6152 mrad per
pixel. The same array went through the engine before the restructure and through this tree, each with its own fresh
CuPy kernel cache, `phase_estimator="mean_phase"`, 200 trials, seed 42, no 180-degree check. Scripts: scratch
`probes/engine_compare.py` (one tree), `probes/engine_diff.py`, `probes/engine_cross.py`.

## Numbers

Fixed values C10 -10 nm, C12 2 nm, phi12 0.4 rad:

| session | earlier engine loss | this tree loss | phases |
|---|---|---|---|
| `det_sampling=0.6152`, rotation 0, CoM as measured | 0.000520867 | 0.000520867 | identical, bit for bit |
| `det_sampling=0.6152`, rotation 0, CoM reversed | 0.000984409 | 0.000984409 | identical |
| `det_sampling=0.6152`, rotation 25, CoM as measured | 0.000892663 | 0.000892663 | identical |
| `det_sampling=0.6152`, rotation 25, CoM reversed | 0.000982096 | 0.000982096 | identical |
| automatic detector sampling, rotation 0 | 0.000331408 | 0.000521682 | correlation 0.17 |
| this tree at `det_sampling = 2 x 30 / 49` | | 0.000331408 | identical to the earlier automatic output |

Fit with automatic detector sampling (truth C10 -10 nm):

| engine | bright-field radius used | detector sampling | fitted C10 | C12 | phi12 | loss |
|---|---|---|---|---|---|---|
| earlier | 49 px (integer half maximum) | 2 x 30 / 49 = 1.224 mrad per pixel | -4.24 nm | 0.33 nm | -1.21 rad | 0.000268 |
| this tree | 48.69 px (disk edge) | 30 / 48.69 = 0.616 mrad per pixel | -9.90 nm | 0.01 nm | -0.14 rad | 0.000149 |

The earlier engine's fitted values, reconstructed by this tree with automatic sampling: loss 0.000974, correlation
0.13 with the picture the earlier engine made from the same values. This tree's own fit: correlation 0.86 with it.

## What changed, by stored field

| stored field | earlier engine | this tree | carries over |
|---|---|---|---|
| `rotation_angle_deg`, `com_reversed` | angle below 180 plus the flag, physical angle = angle + 180 x flag | the same | the convention does; a `com_reversed` chosen together with a defocus sign by an earlier fit does not |
| `aberrations` C10, C12 (nm), phi12 (rad) | chi sign, units and angle reference as now; fitted with detector sampling 2 x semiangle / half-maximum radius | fitted with semiangle / disk-edge radius | no: C10 came out at 0.4 to 0.5 of the current value |
| `aberrations` in a block without `aberration_unit` | Angstrom stored under an nm name (before 2026-09-24) | nm | no |
| `rotation_angle_deg` before 2026-09-27 | physical angle up to 360, no flag | below 180 plus flag | the angle does, read as physical |
| `loss` | phase variance with zero aperture weight beyond half the disk; a session that had not fitted reported the batched evaluator | exact reconstruction loss | no, another quantity |
| `bf_radius` | bright-field selection radius in pixels | the same | yes |
| `voltage_kV`, `semiangle_mrad`, `scan_sampling_A` | acquisition settings | the same | yes |

So the engine's conventions did not change; the automatic detector calibration did
([record](2026-10-05-ssb-detector-sampling.md)), and every stored fit was made without an explicit `det_sampling`.

## Decision

`persistence.SCHEMA` is 6 and `persistence.FIT_SCHEMA = "quantem.ssb-fit/6"` is derived from it. `write_fit_metadata`
writes `"schema": FIT_SCHEMA` into `computed.ssb`; `series.stored_fit(config)` returns a block only when it carries
that string, and `saved_fit` and `acquisition_settings` read through it. A saved fit record counts only at schema 6.
Any other stored fit is absent: nothing in it is read, a probe reference is fitted again, and `write_fit_metadata`
replaces such a block whole. The key and the value form follow the `.qem` scientific-metadata schema string
(`"schema": "quantem.scientific-metadata/2"`), whose readers also refuse another version instead of reinterpreting it.

Schema 5 could not serve unchanged: fit records of schema 5 exist that were written with the earlier sampling
(`"det_sampling": 1.0909` = 2 x 30 / 55 in their signature), so 5 does not prove the current convention.

## Rejected

- Converting earlier values (C10 x 2, or dividing unmarked Angstrom by 10): the ratio is not a constant, it depends on
  the disk radius and on which twin of the 180-degree ambiguity the earlier fit took.
- Accepting a block that declares `aberration_unit: "nm"`: 76 of 687 surveyed blocks declare it and every one of them
  was fitted with the earlier sampling.
- A separate marker number beside `SCHEMA`: two numbers for one contract can disagree.
- Reading the acquisition settings (voltage, semiangle, scan step) of an earlier block while ignoring its fit: one
  block would then be half trusted, and the settings are also in `dataset.yaml` or the call.
