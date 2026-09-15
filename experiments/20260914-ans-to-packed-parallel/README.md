# Bounded concurrent resident conversion

The 1–2 s seven-source target was **not met** by concurrent submission.
All results are native Metal on Apple M5 24 GB with the same seven distinct
512×512×192×192 uint16 residents and a 17,162,698,752-byte allocation ceiling.

| Order | Workers | Seven-source wall time | Sampled Metal high-water |
|---|---:|---:|---:|
| A1 | 1 | 6.607 s | 13.029 GB |
| B | 2 | 5.959 s | 13.144 GB |
| C | 4 | 6.665 s | 14.743 GB |
| A2 | 1 | 6.307 s | 13.029 GB |

Two workers beat both sequential controls by 5.5–9.8%. Four workers did not.
All runs ended at 12.046 GB packed residency and nominal thermal state.
Each run passed 21 complete detector-map and 77 complete DP comparisons,
plus cancellation and memory rejection checks. Counts/precision are unchanged.

The timer starts after originals and reference products are resident and ends
after all seven replacements are ready and old sources are released. Verification
is outside this timer except the source-validity DP check before each release.
No source files are read during conversion. These are backend timers, not
native UI presentation/FPS measurements.

Each source uses its own command queue. More workers hide some CPU/queue gaps
but compete for the same GPU and driver. Individual command intervals become
longer under overlap; their sum is not exclusive GPU work time. Four-worker
residency registration also showed 165–257 ms host stalls. This does not measure
hardware occupancy or establish the theoretical minimum.

Seven simultaneous whole-source conversions were not attempted: their old plus
new resident buffers alone total about 19.1 GB, above this experiment's ceiling.
The implemented scheduler is a benchmark harness, not a production transactional
mode-switch API. Two workers are a useful candidate, not a new viewer default.

Follow-up experiments: `20260914-ans-to-packed-staging` removes the second decode
with a reusable bounded window; `20260914-packed-to-ans` implements the return
path. Their source changes happened after these four control runs. Raw logs live
under `local-evidence://sep14-ans-conversion-followup/`. The initial manifest
validator rejected a missing dirty-diff hash, which was supplied while A1 was
running; subsequent registration validation passed.
