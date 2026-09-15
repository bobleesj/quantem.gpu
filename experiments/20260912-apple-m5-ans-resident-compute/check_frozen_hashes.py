"""Compare every resident-loop result against independently checked prior hashes."""
import json
import sys

reference = {}
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        if row.get("event") == "ans_opt_independent_parity":
            reference[row["mask"], row["source"]] = row["sha256_u32_le"]
assert len(reference) == 140, "Expected twenty independent maps for all seven sources"
checked = 0
with open(sys.argv[2], encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        assert row["event"] != "ans_resident_loop_error", row
        if row["event"] != "ans_resident_loop_result":
            continue
        assert row["fullmap_parity"], row.get("fullmap_mismatches")
        for sample in row["samples"]:
            assert sample["sha256_u32_le"] == reference[sample["mask"], sample["source"]]
            checked += 1
print(json.dumps({"independent_reference_maps": len(reference),
                  "full_map_hashes_checked": checked, "exact": True}))
