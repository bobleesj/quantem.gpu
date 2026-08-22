# Developer guide

Development starts from one rule: add an implementation to an existing
scientific contract rather than creating a backend-specific workflow.

Read these pages in order:

1. [Choose the scientific operation](../kernels/index.md)
2. [Kernel architecture](../concepts/kernel-architecture.md)
3. [Kernel development lifecycle](kernel-lifecycle.md)
4. [Adding a backend or kernel](adding-backend.md)
5. [Scientific writing, notation, and units](writing.md)
6. [Testing and evidence](testing.md)
7. [Benchmark methodology](../performance/methodology.md)
8. [Cross-backend parity](../performance/parity.md)

The public Python modules and Swift products are documented under the
[API reference](../api/index.md). Backend internals may change behind those
contracts; consumer code should not import them.

```{tableofcontents}
```
