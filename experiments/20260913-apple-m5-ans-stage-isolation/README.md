# Exact signed stage isolation

Seven full uint16 512x512x192x192 sources remain resident. Transition is
ADF center (5,8) to (12,20), radii48/94. The unchanged planner selects371 fields
and1067 residual detector pixels per source. Both stages use the existing
pipelines and signed coefficients, not checksum approximations. Partial
outputs start at zero and do not mutate the current detector image/history.

Nine repetitions, first excluded as warmup; stage order reverses on alternate
repetitions. All-seven return includes planning, temporary allocation and image
readback, but excludes input loading and parity comparisons. Temporary output
is1MiB/source plus small mask/status buffers; no dense resident duplicate.

| Work across seven sources | Median ms | Min-max ms |
|---|---:|---:|
| Index only |24.865|20.252-26.475|
| Residual only |47.579|37.385-48.709|
| Combined |66.328|59.192-72.294|

Standalone medians sum to72.444ms; combined is about6.1ms lower. This is
descriptive, not a production speedup: timing includes host work and separate
calls repeat preparation. These numbers do not isolate pure contention or
prove cache bandwidth/occupancy limits. The combined result is measured in
this diagnostic, not a claimed improvement over older74ms ordinary runs.

All63 source transitions pass elementwise checks that index+residual equals
combined and the ordinary target-image minus previous-image, moduloUInt32.
The ordinary reference uses the existing indexed path; this run does not add
a new independent raw-count oracle. Resident bytes remain unchanged per log.

Residual decode and summation is the larger target, approximately two-thirds
of separately measured work. Index processing still exceeds8.33ms alone for
seven sources, so both stages ultimately need improvement for120updates/s.
No displayed FPS measurement, installed app modification, push or release.
