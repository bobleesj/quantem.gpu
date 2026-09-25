# Finite-recovery dispatch experiment

The original decoder launches the selected-column recovery shader once per ANS chunk. A prototype proved all restored source values finite and bounded on the 100×100×256×256 fixture, then skipped that shader. It did not alter arithmetic.

Native app, same process type and 720-step detector trajectory per arm: large-ADF-center distinct presentations were 57.1/s (control), 60.3/s (prototype), and 58.7/s (control). Reduction p50 was 15.99, 15.66, and 15.76 ms. The apparent gain is too small relative to run variation to justify an additional full-source validation pass at open. The prototype was removed; no source change from this experiment was retained.

The later parallel-column experiment uses the unchanged nonfinite recovery path and has its own parity and performance record.
