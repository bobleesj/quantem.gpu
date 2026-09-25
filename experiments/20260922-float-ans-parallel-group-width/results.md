# Wider parallel decoder groups: refuted

On the same full-resolution float32 ANS resident, the 180-mask hardware harness exercised 89 large edits and produced bit-identical complete virtual images for 32-, 64-, and 128-thread groups. The test used a temporary dispatch-width override in `MetalFloatANS.encodeSelected`; it was removed after this result.

Large-edit GPU decode median ticks, ordered A/B/A/C/A: 32 threads 7,349,539; 64 threads 7,334,787; 32 threads 7,361,537; 128 threads 7,812,290; 32 threads 7,342,916. Width 64 was neutral, while 128 was about 6% slower. No native-app gate was warranted and no production source change was retained.
