# Developer guide

Use this section when implementing, integrating, or verifying QuantEM.GPU.
Scientists using notebooks can start with the [Python workflow](../python-workflow.md).

## Find the task

| Task | Start here |
|---|---|
| Understand an operation's equations and array contract | [Scientific operations](../kernels/index.md) |
| Implement an operation on a device | [MPS, CUDA, and other backends](../platforms/index.md) |
| Build a native application | [Native integration](integration.md) |
| Implement a file reader or writer | [File formats and codecs](file-formats.md) |
| Run a remote CUDA service | [Remote compute](../remote/index.md) |
| Verify correctness or inspect measured performance | [Testing and benchmarks](../performance/index.md) |
| Change the code or documentation | [Contribution workflow](kernel-lifecycle.md) |
| Reproduce a dated experiment | [Historical records](../maintainer/index.md) |

Each operation has one scientific owner: its public coordinates, units, values,
and errors remain consistent while backends specialize allocation and kernels.
Applications own scheduling, presentation, and explicit resource policies.

## Before contributing

Read the [source layout](../concepts/kernel-architecture.md) and the
[writing conventions](writing.md). Keep Python workflows short; link advanced
options to their API owner. Native products and codec layouts belong here,
not in a scientist's first example.

Benchmark claims need their exact revision, device, input, timing boundary,
precision, memory, and parity evidence. Use the
[benchmark methodology](../performance/methodology.md) and
[test guide](testing.md) before changing a measured path. Historical numbers
retain their original qualification; a documentation reorganization does not
establish new hardware performance.
