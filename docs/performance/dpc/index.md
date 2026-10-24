# CoM, DPC and iDPC

Measure detector moments, centering, scan-detector rotation and iDPC integration
separately from acquisition loading and image presentation. A fast moment
reducer does not establish end-to-end iDPC latency or phase parity.

## Existing evidence

- [Virtual-image validation record](../virtual-detectors/validation.md): includes retained CoM/DPC/iDPC browser measurements and backend gates.
- [Cross-backend parity](../parity/index.md): recorded agreements and unresolved iDPC discrepancies; preserve the original tolerances and detector-bin labels.
- [Platform measurement ledger](../results.md): historical timing and provenance.

These shared records are linked rather than duplicated into a new timing claim.
There is no new DPC benchmark run in this documentation reorganization.

## Implementation and reproduction

Use the [CoM/DPC/iDPC kernel guide](../../kernels/com-dpc-idpc.md) for the source
map and coordinate conventions, the [public API](../../api/images_dpc.md) for
calls, and the [testing guide](../../developer/testing.md) for available
hardware checks. Compare both vector components, rotation conventions and
integrated phase; do not substitute a single image similarity score.
