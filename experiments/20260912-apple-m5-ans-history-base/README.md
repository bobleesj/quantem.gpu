# Compute a fresh detector result from the nearer saved mask

The previous experiment retained one exact detector image. This experiment
uses that same image as a base for a **different** target mask when doing so
requires less residual work. It adds no third output buffer or image atlas.

For target mask T and saved base H, compute `image(T) = image(H) + sum(T-H)`.
Only entering/leaving detector pixels contribute, with signed coefficients.
Raw mode chooses by changed-pixel count; indexed mode uses the existing
planner's work estimate. Strict ties keep the current base. If zero-base
selection is enabled, its clearing cost is included. Scientific counts,
uint16 precision and full acquisition dimensions do not change.

The candidate writes into the saved-history buffer. It skips copying the
current output only when using that history as its starting point. Success
rotates the buffers; failure discards the candidate history and preserves the
current image. Exact returns are separately marked `history_hit`; fresh
history-based computation is marked `history_base` and must not count as a
cache hit.

## Gates

- Seven distinct native 512×512×192×192 uint16 acquisitions.
- First freeze every mask with raw full-output parity.
- Compare off/on/off for the fresh trajectory A→B→C, where A is `adf-base`,
  B is `adf-center-8`, and C is `adf-center-1`. C is different from both saved
  masks. Test raw and indexed modes separately.
- Also test the complete BF/ABF/ADF center/radius trajectory to expose cases
  where extra planning costs more than it saves.
- Same history allocation in controls and candidates: 11,885,154,080 resident
  bytes, including the existing regional index. The history feature itself
  costs 7 MiB versus history-disabled storage; nearest-base adds no GPU bytes.
- Frozen full-map hashes and repeated-mask array equality outside timing.
- No UI, GPU fault-injection or release qualification is implied.

## Results and limitation

All 2,660 full-map hashes passed against 140 independent frozen maps. For the
fresh B→C reversal, all-seven warm medians were:

| Mode | Off before (ms) | On (ms) | Off after (ms) |
| --- | ---: | ---: | ---: |
| Raw ANS | 187.689 | 25.698 | 188.647 |
| Indexed | 42.236 | 20.994 | 45.009 |

The benchmark resets to a zero mask between cycles. Do not interpret the
large improvement of the first A in each cycle as continuous-drag performance;
the B→C comparison above follows the same A→B sequence in every arm. C is a
fresh residual computation, not an exact-history hit.

The full 20-mask trajectory exposed a regression: indexed ADF center+20 was
74.663 / 86.446 / 71.282 ms. For the first source, the chosen history plan
reduced residual pixels from 1067 to 986 but increased indexed fields from
371 to 589. The uncalibrated scalar cost allowed this bad tradeoff. Do not
promote this policy unchanged. The next experiment requires no increase in
either component before choosing history. Large ADF radius transition improved
82.507 / 49.194 / 88.084 ms, so the direction is useful but needs conservative
selection. No default or installed app was changed.
