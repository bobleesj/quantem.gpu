# Residual decoder: zero-bit branch removal

FC16 removes the count==0 early return in the eager64-bit bit reader. The
existing mask yields zero for zero bits. Available bits stay below64 in this
specialization; reader32 keeps its original zero-count handling. No validation,
index layout, selected count, precision, or resident memory is removed.
The pipeline is opt-in and used only by stage isolation, not public defaults.

Same full seven-source uint16 signed ADF center-8 to center-20 transition as
the stage-isolation experiment. There are25 total repetitions grouped into
five alternating arms, five repeats each; first repeat per arm excluded.

| Arm | Residual ms | Combined ms |
|---|---:|---:|
| Off1 |46.205|62.003|
| On1 |43.951|59.202|
| Off2 |41.722|58.920|
| On2 |38.683|58.661|
| Off3 |43.196|67.574|

Timing is inconclusive. The first candidate fails to beat the next control;
the second looks promising but control variation is large. Do not claim a
qualified improvement or install this variant. Residual sample minima near37ms
occur with both settings. The unchanged index also varies across runs.

All175 source transitions passed exact elementwise index+residual=combined=
ordinary indexed target-minus-previous. The independent signed high-count
synthetic fixture passes512 outputs forFC16. Existing malformed-stream checks
ran on previous pipelines, notFC16: that qualification remains pending.

No extra resident storage. Diagnostic outputs allocate1MiB/source plus small
buffers, and all-seven return includes preparation/allocation/readback.
No on-screen FPS or theoretical bandwidth/occupancy floor measured. No push,
release, or installed app change.
