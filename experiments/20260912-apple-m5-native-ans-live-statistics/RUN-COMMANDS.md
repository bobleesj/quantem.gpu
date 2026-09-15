# Live detector statistics experiment

The candidate build was the sibling `Live4DSTEM` release build using the
local compact-ANS resident path. The seven acquisitions are full
`512×512×192×192` `uint16` inputs; no saved ANS file, crop, bin, or clipping
was used.

```bash
cd ~/repos/Live4DSTEM
swift build -c release --disable-sandbox
python3 Tests/NativeUI/drive_folder.py \
  --exe .build/release/Live4DSTEM \
  --folder ~/data/maped-seven-tilts \
  --out /tmp/drive-seven-compact-ans-single-120-20260912-b \
  --count 7 --resident-mode compact-ans --min-presented-fps 110
```

The diagnostic replay used `--diagnostics` without the presentation gate so
the native log could retain Metal GPU-stage timings. The seven-way comparison
was separately run with `DIAG=1` through `profile_seven_tilts.py`.

The optimization only defers image-statistics reduction while a detector is
being dragged. The release event refreshes statistics and therefore commits
the new percentile window. During motion the previous validated range and
histogram are copied to each newly published surface, preventing allocator
contents from affecting display contrast.
