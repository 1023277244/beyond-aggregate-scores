"""Recompute the paper's main point estimates from released scores and records."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

DATA = Path(__file__).resolve().parent / "data"
SEEDS = (42, 0, 1)
ARMS = ("M", "A50", "A67", "A83", "S0")
CASES = ("Scratch", "MissingPart", "CharacterOmission", "Deformation")


def rows(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def checked(actual, expected):
    if not math.isclose(actual, float(expected), abs_tol=1e-12, rel_tol=0):
        raise ValueError(f"Reproduction mismatch: {actual} != {expected}")
    return actual


def effect(wins, labels, threshold, budget):
    normal = [score for group, score in wins.items() if labels[group] == "OK"]
    diagnostic_cut = sorted(normal, reverse=True)[budget]
    false_alarms = sum(score > threshold for score in normal)
    diagnostic_false_alarms = sum(score > diagnostic_cut for score in normal)
    for kind in ("Overall", *CASES):
        defects = [score for group, score in wins.items()
                   if labels[group] == kind or (kind == "Overall" and labels[group] != "OK")]
        pair_sum = sum(1 if score > reference else .5 if score == reference else 0
                       for score in defects for reference in normal)
        yield kind, dict(auroc=pair_sum/(len(defects)*len(normal)),
            tp=sum(x > threshold for x in defects), fp=false_alarms,
            lowfp_tp=sum(x > diagnostic_cut for x in defects), lowfp_fp=diagnostic_false_alarms)


def ceramic_sweep():
    labels = {row["group_id"]: row["defect_type"] for row in rows("ceramic_test_units.csv")}
    assert len(labels) == 224 and list(labels.values()).count("OK") == 25
    scores, calibration = defaultdict(dict), defaultdict(list)
    for row in rows("ceramic_raw_scores.csv"):
        scores[(row["arm"], int(row["seed"]))][row["group_id"]] = float(row["score"])
    for row in rows("ceramic_calibration_scores.csv"):
        calibration[(row["arm"], int(row["seed"]))].append(float(row["score"]))
    expected = {(r["arm"], int(r["seed"]), r["stratum"]): r for r in rows("ceramic_sweep_seed_metrics.csv")}
    summary = defaultdict(list)
    assert len(scores) == 15
    for arm, seed in scores:
        assert len(calibration[arm, seed]) == 6 and set(scores[arm, seed]) == set(labels)
        for kind, metrics in effect(scores[arm, seed], labels, max(calibration[arm, seed]), 1):
            original = expected[arm, seed, kind]
            for name, value in metrics.items():
                checked(value, original[name])
            summary[arm, kind].append(metrics["auroc"])
    sweep = {arm: {kind: round(100*mean(summary[arm, kind]), 4)
                  for kind in ("Overall", *CASES)} for arm in ARMS}
    assert all(sweep[a]["Scratch"] > sweep[b]["Scratch"] and
               sweep[a]["MissingPart"] < sweep[b]["MissingPart"]
               for a, b in zip(ARMS, ARMS[1:]))
    return sweep


def fixed_product_criteria():
    comparison = defaultdict(dict)
    for row in rows("all_product_seed_metrics.csv"):
        if row["panel"] not in ("Fixed panel I", "Fixed panel II") or row["stratum"] == "Overall":
            continue
        comparison[row["product"], row["stratum"]][row["arm"], int(row["seed"])] = float(row["auroc"])
    qualified = defaultdict(set)
    for (product, _), metrics in comparison.items():
        differences = [metrics["M", seed] - metrics["S0", seed] for seed in SEEDS]
        for direction in (1, -1):
            if mean(differences) * direction >= .05 and sum(d * direction > 0 for d in differences) >= 2:
                qualified[product].add(direction)
    products = sorted(product for product, signs in qualified.items() if len(signs) == 2)
    assert len(set(product for product, _ in comparison)) == 9
    assert products == ["bag_buckle", "ceramic_fuse", "ceramic_wave_filter", "pulse_transformer"]
    return products


def regional():
    data = rows("ceramic_per_part_region_scores.csv")
    expected = {(r["defect_type"], r["comparison"]): r for r in rows("ceramic_region_effects.csv") if r["arm"] == "M"}
    comparisons = ("full", "annotation_excluded", "control_excluded")
    output = {}
    for kind in CASES:
        effects = {}
        for comparison in comparisons:
            values = defaultdict(list)
            for row in data:
                if row["defect_type"] == kind:
                    values[row["arm"]].append(float(row[f"{comparison}_win_rate"]))
            change = 100*(mean(values["M"])-mean(values["S0"]))
            effects[comparison] = checked(change, expected[kind, comparison]["mean_pp"])
        difference = effects["control_excluded"] - effects["annotation_excluded"]
        checked(difference, expected[kind, "annotation_specific_attenuation"]["mean_pp"])
        output[kind] = {key: round(value, 4) for key, value in effects.items()}
    return output


if __name__ == "__main__":
    print(json.dumps(dict(ceramic_sweep_percent=ceramic_sweep(),
        products_meeting_opposing_effect_criterion=fixed_product_criteria(),
        regional_effects_pp=regional()), indent=2))
