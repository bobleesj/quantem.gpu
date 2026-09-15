# Current-source large-ADF baseline reproduction (harness retry)

This retry uses the same current-source binary and exact seven-source workload
as `20260913-apple-m5-ans-current-source-reproduction`. The first attempt
loaded all sources and passed the memory gates, but its inherited validator
stripped the `macro: false` field even though the current benchmark protocol
requires it. That harness failure occurred during the warmup response, before
any measured arm. The original failure and clean release remain preserved.

The retry retains `macro: false` and validates it explicitly, then invokes the
trusted-table and scan512 parity checks. All scientific inputs, environment
settings, arm order, timing count, memory limits, and binary are otherwise
unchanged.

Run with the same fresh build and source folder documented in the original
experiment. Outputs are retained in `results/`.
