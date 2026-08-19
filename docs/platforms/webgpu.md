# WebGPU

WebGPU is the browser runtime used by `quantem.widget` and portable exported
HTML. It is not a Python backend name and must not be described as Metal even
when the browser maps WebGPU onto Metal internally.

Reusable TypeScript/WGSL sources live beside their scientific domains in
`quantem.gpu`. The widget bundles those canonical files instead of maintaining
a second implementation.

Browser evidence distinguishes:

- source presence and TypeScript build;
- software-adapter smoke tests;
- real hardware adapter parity;
- full local-file load versus product-first paths; and
- warm interaction versus first-source loading.

Only real-adapter runs count as hardware performance. The adapter name,
browser/version, source bytes, upload/decode/compute/readback/present stages,
and corrected-output checksum are required.

The current large-source boundary and exact product-first results are recorded
under [Current verified results](../backends.md). `quantem.widget` remains the
owner of browser UI and export behavior.
