# Original EMPAD materials and load-time diagnosis

## Verified finding

The extra second in the existing 4.36 GB MoS2-MoSe2 native runs is predominantly
SHA-256 calculation, not Metal unpacking. These are retained measured runs,
inspected again on 2026-09-08, not fresh cold-storage measurements.

| Source identity state | Read stage | SHA-256 stage | Full native load to presentation |
| --- | --- | --- | --- |
| Recomputed checksum, three opens | 0.331-0.362 s | 1.307-1.326 s | 2.187-2.270 s |
| Reused checksum, three opens | 0.321-0.407 s | 0.000002-0.000005 s | 0.896-0.989 s |

Every reported load reads all 65,536 frames and rebuilds the resident. The
checksum cache holds identity metadata, not a 2D image or a reusable 4D resident.
The analyzed source has a 256x256 scan and 128x128 physical float32 detector.
The two extra rows in each 130x128 file record are instrument footer rows.
No physical detector pixels are binned, clipped, or cropped.

The read, hash, and analyze entries are instrumented wall stages and include
overlap. Do not add them as independent GPU timings. OS page-cache state was not
controlled; these numbers do not establish cold I/O performance.

Evidence: `empad-final-native-miss/console.log`,
`empad-final-native-hit/console.log`, and their native presentation records,
retained in the local validation archive. Exact app hash and production patch
identity are in `manifest.json`.

## Additional materials

Three distinct original [SnSe acquisitions](https://zenodo.org/records/10079791)
were selected: acquisitions 7, 8, and 10, each 4,362,076,160 bytes. The
[authors' reader](https://github.com/Chuqiao2333/2D_ferroelectric_SnSe/blob/main/helper_function.py)
confirms C-order float32 130x128 records with 128x128 physical detector data.
Credit: Chuqiao Shi and Yimo Han, CC BY 4.0.

The published filenames explicitly specify 256x256 scans. Local XML sidecars
describe that known geometry; they are not original instrument XML or a claim
of physical calibration. RAW filenames and bytes are unchanged.

Also identified: original [cryo-cell and organelle acquisitions](https://zenodo.org/records/10825339).
These were not downloaded or tested. The newer record version contains analysis
files rather than the original RAW acquisitions.

## Pending validation

SnSe transfers are in progress. No SnSe load, parity, or FPS result is claimed.
The immediate validation job resumes partial downloads, verifies the published
MD5 and computes SHA-256, then runs these checks serially:

1. All physical float32 samples bitwise equal to the original file, with detector
   integrations and DPC products checked against float64 references.
2. Native original load, actual displayed buffer parity, BF/ABF/ADF, contrast,
   colormaps, appearance, FFT, failed-file recovery, forced reload, ARINA folder
   replacement, and EMPAD return.
3. Separate checksum-disabled and checksum-enabled native runs. Only individual
   `hash_cached=1` records are checksum hits; an enabled-cache first open can miss.
4. If all three fixtures pass, a three-file native folder-switch and unique-scan
   stress run, with a measured presentation-rate gate of 118 FPS. A failed speed
   gate must remain a failed gate, not be reported as successful interactivity.

Network downloads finish before timing. No cold-I/O claim is made. Tests defer
if another viewer is already running, and reject a changed executable or harness.
Machine-readable job status, checksums, native logs, and screenshots are written
to the local `2026-09-08 SnSe Materials` validation directory. This run does not
modify the installed app, authorize a release, or establish 120 FPS everywhere.
