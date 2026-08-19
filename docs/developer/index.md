# Developer guide

Development starts from one rule: add an implementation to an existing
scientific contract rather than creating a backend-specific workflow.

Read these pages in order:

1. [Repository architecture](../maintainer/backend-layout-and-parity.md)
2. [Adding a backend or kernel](adding-backend.md)
3. [Scientific writing, notation, and units](writing.md)
4. [Testing and evidence](testing.md)
5. [Benchmark methodology](../performance/methodology.md)
6. [Cross-backend parity](../performance/parity.md)

The public Python modules and Swift products are documented under the
[API reference](../api/index.md). Backend internals may change behind those
contracts; consumer code should not import them.

```{tableofcontents}
```
