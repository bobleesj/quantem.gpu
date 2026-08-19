# Adding a backend or kernel

A new backend is complete only when its source, package resources, parity,
physical evidence, and consumer integration land together.

## 1. Choose the owning domain

Place the implementation under IO, detector, DPC, display, SSB, or another
existing scientific domain. If no domain owns the operation, define the
backend-neutral contract before writing accelerator code.

Do not create `cuda_*`, `mps_*`, `metal_*`, or `webgpu_*` public workflows.
Public callers select a backend through the domain API and receive the same
result/provenance model.

## 2. Freeze the reference

Record the current public call, fixture hash, parameters, output shape/dtype,
output hashes, floating metrics, and accepted tolerances. Include odd and
rectangular geometry and incomplete edge bins where applicable.

The backend under development cannot generate its own acceptance golden.

## 3. Implement without hidden policy

The backend may change memory layout, batching, fusion, scheduling, or kernel
topology. It may not silently change crop, bin, mask, precision, objective, or
scientific output.

Resource estimators belong in `quantem.gpu`; a consuming application owns the
user-visible choice and reason for an automatic resource policy.

## 4. Package the source

Update the appropriate package-data or SwiftPM resources and add a source
presence/compile test. Browser sources remain beside their scientific domain.
Native clients consume Swift products rather than copied `.metal` files.

## 5. Add parity and hardware evidence

Update `tests/parity/backend_matrix.json`, add focused numerical tests, and run
the qualified physical-device gate. Record source revision, hardware/runtime,
memory, stage timings, and run-level results using the
[benchmark methodology](../performance/methodology.md).

## 6. Integrate consumers

Test `quantem.widget`, Live4DSTEM, or `quantem.live` through a local package
override. Publish the shared package first, then pin consumers to the exact
verified revision. A copied kernel or consumer-side compatibility fork is not
an integration.

## 7. Migrate paths safely

Folder cleanup uses import-only compatibility shims for one reviewed migration
cycle. Move one domain per commit and delete shims only after all consumers
have migrated and repeated the parity gates.
