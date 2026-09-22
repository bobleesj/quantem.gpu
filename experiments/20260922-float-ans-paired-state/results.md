# Interleaved low/high rANS state updates

The two independent entropy table loads now begin before either lane's state update. The output word retains the original low/high lane order. On the full-resolution float32 ANS resident, large-edit GPU decode p50 was 6,970,918 / 6,152,828 / 6,980,086 / 6,178,669 / 6,990,411 ticks in an ordered entropy-specialized/interleaved/entropy-specialized/interleaved/entropy-specialized sweep. This is about 11.7% faster for that decoder stage; the reducer remained about 1.47 million ticks.

With the original 1600-pixel threshold, native large-ADF-center presentation was 95.6/96.7/95.5 updates/s, so the stage gain alone did not meaningfully improve the app. The native trace explained why: only 72 of 815 updates changed at least 1600 detector pixels. Most ordinary moves changed roughly 700 pixels and still took the serial path.

The full 180-mask output matched byte-for-byte on the real 256×256 file and on a synthetic finite/NaN/+Inf/−Inf file; the latter included finite-to-nonfinite transitions as the aperture moved. A 128×128 compatibility fixture also matched. The following threshold experiment made the interleaved path useful for ordinary movement. Source remains full-resolution, lossless ANS and GPU-resident; no decoded 4D volume is retained.
