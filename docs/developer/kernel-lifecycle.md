# Kernel development lifecycle

Use this checklist for a CUDA, MPS, Metal, or WebGPU optimization. The goal is
a faster implementation of the existing scientific contract, not a
backend-specific product.

## 1. Start from the operation

Read the matching page under [Scientific kernels](../kernels/index.md). Freeze
its input shape/dtype, `(row, column) ≡ (r, c)` convention, parameters,
provenance, outputs, and accepted metric before changing code.

## 2. Locate every implementation

Use [Kernel architecture](../concepts/kernel-architecture.md) to find the
public adapter, CPU reference, accelerator sources, native products, packaged
resources, and tests. Search consumers only to understand the public boundary;
do not move application presentation or state into this package.

## 3. Profile the end-to-end path

Measure the actual workload and physical device. Record source open/index,
read/page-in, decode, conversion/binning, allocation, kernel stages,
synchronization, readback, first usable product, finalization, peak memory, and
cache state. A kernel microbenchmark is supporting evidence, not an end-to-end
load time.

## 4. Change one hypothesis

Typical hypotheses are fewer full-array traversals, a fused reduction, more
coalesced access, persistent pipelines, overlapping queues, fewer host
synchronizations, or smaller reusable scratch. Keep experimental paths guarded
until parity and physical evidence support them.

## 5. Prove scientific parity

Run the CPU or frozen cross-backend reference first, then compare the candidate
with the same source, parameters, crop/bin plan, and output contract. Integer
operations are byte-exact when arithmetic is identical. Floating operations
use the frozen operation-specific metric and tolerance.

## 6. Prove performance honestly

Report run-level values plus p50/p95/max where repeated. Keep cold source, warm
source/page cache, prepared index, resident interaction, and saved-result
reopen separate. Include both critical-path wall time and device intervals;
overlapping GPU intervals are not summed and called wall time.

## 7. Land a reviewable change

Update focused tests, the backend parity matrix, benchmark provenance, and the
operation/platform docs. Remove rejected experiments or record them in the
optimization ledger so the same regression is not repeated.
