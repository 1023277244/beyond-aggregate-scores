"""Repeat the pooling analysis on the same shape-eligible cohort at all radii."""
import sys

sys.dont_write_bytecode = True
import numpy as np
from run_region_audit import (OUT, ROOT, STRATA, SWEEP, checkpoints, collect_maps,
                              collect_pairs, controls_for, cv, now, read, save,
                              sha, summarize, write_csv)


def main():
    old_lock = read(OUT / "NEIGHBORHOOD_FREEZE.json")
    assert old_lock["protocol_sha"] == sha(OUT / "NEIGHBORHOOD_PROTOCOL.md")
    assert old_lock["code_sha"] == sha(OUT / "check_neighborhood.py")
    assert old_lock["primary_receipt_sha"] == sha(OUT / "RECEIPT.json")
    main_freeze = read(OUT / "ANALYSIS_FREEZE.json")
    assert all(sha(ROOT / name) == digest for name, digest in main_freeze["files"].items())
    frozen = read(SWEEP / "ALL_MODELS_FROZEN.json")
    assert all(sha(ROOT / name) == digest for name, digest in frozen["models"].items())
    lock = OUT / "NEIGHBORHOOD_RECOVERY_FREEZE.json"
    assert not lock.exists()
    radii = (0, 1, 2)
    save(lock, dict(frozen_at_utc=now(), radii=radii, original_lock_sha=sha(OUT / "NEIGHBORHOOD_FREEZE.json"),
                    initial_failure="No feasible non-overlapping control at two cells. No results saved.",
                    code_sha=sha(__file__), protocol_sha=sha(OUT / "NEIGHBORHOOD_RECOVERY.md")))
    paths = checkpoints()
    manifest, maps, scores, masks, density, test_error, cal_error = collect_maps(frozen["encoder_sha"], paths)
    expanded = {
        radius: {i: cv.dilate(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)).astype(bool)
                 for i, mask in masks.items()}
        for radius in radii
    }
    excluded, excluded_indices, results, coverage = [], [], [], []
    for index, mask in expanded[2].items():
        try:
            controls_for(mask, density)
        except AssertionError as error:
            if str(error) != "No feasible non-overlapping control":
                raise
            row = manifest[index]
            excluded.append(dict(group_id=row["group_id"], defect_type=row["anomaly_class"],
                                 reason="No disjoint in-bounds translation at radius 2"))
            excluded_indices.append(index)
    for index in excluded_indices:
        for radius in radii:
            expanded[radius].pop(index)
    for radius in radii:
        pairs, inspection, matches = collect_pairs(manifest, maps, scores, expanded[radius], density, tuple(paths))
        assert sum(map(len, pairs.values())) == len(masks) - len(excluded)
        rows, _ = summarize(pairs, tuple(paths), 25)
        results.extend(dict(row, radius_cells=radius) for row in rows)
        coverage.append(dict(radius_cells=radius, samples_by_defect={kind: len(pairs[kind]) for kind in STRATA},
                             controls=len(matches), masks_max_cells=max(r["region_cells"] for r in inspection)))
        print(f"Finished radius {radius} on {len(masks)-len(excluded)} eligible defective parts.", flush=True)
    names = ("neighborhood_effects.csv", "neighborhood_exclusions.csv")
    write_csv(OUT / names[0], results)
    write_csv(OUT / names[1], excluded, fields=["group_id", "defect_type", "reason"])
    save(OUT / "NEIGHBORHOOD_RECEIPT.json", dict(completed_at_utc=now(),
        output_files={n: sha(OUT / n) for n in names}, max_abs_test_error=test_error,
        max_abs_calibration_error=cal_error, coverage=coverage))


if __name__ == "__main__":
    main()
