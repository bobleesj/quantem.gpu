# Native Normal/Fast integration

The native viewer can opt into the frozen extra-index representation through
`MetalPairedRuntimeTANSResidentSource.load(source:device:maximumAdditionalBytes:interaction:shouldCancel:progress:)`.
Pass `.normal` for the existing low-memory representation or `.fast` for the
extra exact region indexes and compact events. Configuration is immutable for
each resident; do not change process environment to switch a live application.
The existing source-support check still applies. This experiment qualifies
seven full uint16 512 x 512 x 192 x 192 original acquisitions on Apple M5 24 GB,
not every detector format, acquisition, or memory size.

The viewer loads replacements sequentially and keeps each old resident usable
until its replacement is ready. Additional workspace and indexes count against
the production admission budget. When buffered Normal encoding cannot fit its
temporary window, use the existing exact size-then-write encoder; do not raise
the budget or evict another selected acquisition. Cancellation retains completed
replacements, so the UI reports actual mixed-mode counts.

Readiness follows the requested representation, not merely a new image from an
old resident. A custom detector staged during replacement must apply exactly the
same marked-pixel exclusions as an ordinary detector drag. A later forced reload
starts a new load intent rather than inheriting a completed conversion target.

## Required evidence

- Native menu confirmation, seven visible BF/ABF/ADF images, distinct source
  identities, selected-DP round trips, and full custom-map hashes.
- Independent original-HDF5 sums for all seven full 262144-pixel output maps.
- Foreground drawable presentation timestamps for detector-center and radius
  trajectories. Kernel throughput or accepted-input counts are not display FPS.
- Cancel/retry, Fast-to-Normal, custom-detector Normal-to-Fast, forced reload,
  and playback with a window-local scan drag.
- Actual resident size, sampled Metal allocation peaks, exact executable/source
  provenance, retained failures, and an explicit distinction between warm-source
  preparation and cold I/O.

The manifest and registry retain results. The test implementation is in the
viewer: `check_fast_interaction.py`, `check_interaction_conversion.py`,
`check_fast_reload.py`, and `profile_playback_drag.py`. This is development
integration evidence, not notarization or release acceptance.
