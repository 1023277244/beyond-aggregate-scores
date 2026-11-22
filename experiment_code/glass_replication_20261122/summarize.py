"""Summarize frozen GLASS results with paired conditional bootstrap intervals."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "formal"
SEEDS = [42, 0, 1]
PRODUCTS = {"ceramic_wave_filter": "Ceramic filter", "nylon_ball_head": "Nylon ball head"}
TYPES = {
    "CharacterOmission": "Missing marking", "MissingPart": "Missing part",
    "OilStain": "Oil stain", "PorosityDS": "Porosity (DS)",
}
DRAWS = 2000


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


freeze = json.loads((OUT / "ALL_TEST_SCORES_FROZEN.json").read_text(encoding="utf-8"))
metrics_path, score_path = OUT / "METRICS.csv", OUT / "RAW_TEST_SCORES.csv"
assert sha(metrics_path) == freeze["metrics_sha256"]
assert sha(score_path) == freeze["scores_sha256"]
metrics = {(r["product"], int(r["seed"]), r["arm"], r["stratum"]): r for r in read(metrics_path)}
scores = {(r["product"], int(r["seed"]), r["arm"], r["group_id"]): float(r["score"]) for r in read(score_path)}
parts = {}
for r in read(score_path):
    assert parts.setdefault(r["product"], {}).setdefault(r["group_id"], r["defect_type"]) == r["defect_type"]

lines = [
    "# GLASS paired strength replication",
    "",
    "The experiment uses the complete GLASS scoring and optimization path with global gradient-based and local image anomaly synthesis. The Real-IAD Variety adaptation uses frozen WRN50-2, a 256-pixel input, 24 epochs, batch size two, and the last scheduled checkpoint. The local paste weights are 2/3 (strong) and 1/3 (weak).",
    "",
    "All twelve fitted endpoints and six-normal calibration thresholds were sealed before test evaluation. Each line below reports the weaker-minus-stronger change. Conditional paired bootstrap intervals resample the same class-stratified test parts across methods and seeds. Seeds are repeated fits on a common test set.",
]
summaries, outcomes = [], {}
for product, name in PRODUCTS.items():
    rng = np.random.default_rng(1120)
    normal = sorted(part for part, kind in parts[product].items() if kind == "OK")
    categories = sorted(set(parts[product].values()) - {"OK"})
    normal_multiplicity = rng.multinomial(len(normal), [1 / len(normal)] * len(normal), size=DRAWS)
    overall_bootstrap = np.zeros(DRAWS)
    total_defects = 0
    product_summary = []
    for category in categories:
        positives = sorted(part for part, kind in parts[product].items() if kind == category)
        seed_changes = []
        pair_changes = []
        for seed in SEEDS:
            comparisons = {}
            for arm in ["strong", "weak"]:
                x = np.array([scores[product, seed, arm, group] for group in positives])[:, None]
                y = np.array([scores[product, seed, arm, group] for group in normal])
                comparisons[arm] = (x > y).astype(float) + .5 * (x == y)
                assert np.isclose(comparisons[arm].mean(), float(metrics[product, seed, arm, category]["auroc"]))
            contrast = comparisons["weak"] - comparisons["strong"]
            pair_changes.append(contrast)
            seed_changes.append(contrast.mean())
        positive_multiplicity = rng.multinomial(len(positives), [1 / len(positives)] * len(positives), size=DRAWS)
        draws = ((positive_multiplicity @ np.mean(pair_changes, axis=0)) * normal_multiplicity).sum(axis=1)
        draws /= len(positives) * len(normal)
        overall_bootstrap += draws * len(positives)
        total_defects += len(positives)
        product_summary.append((category, len(positives), np.asarray(seed_changes), draws))
    overall_seed = np.array([
        float(metrics[product, seed, "weak", "Overall"]["auroc"])
        - float(metrics[product, seed, "strong", "Overall"]["auroc"])
        for seed in SEEDS
    ])
    product_summary.append(("Overall", total_defects, overall_seed, overall_bootstrap / total_defects))
    gain, loss = [], []
    lines.extend(["", f"## {name}", "", "| Defect | N | Strong AUROC | Weak AUROC | Δ (pp) | Paired 95% interval (pp) | Seed changes (pp) |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for kind, n, delta, draws in product_summary:
        strong = np.mean([float(metrics[product, seed, "strong", kind]["auroc"]) for seed in SEEDS])
        weak = np.mean([float(metrics[product, seed, "weak", kind]["auroc"]) for seed in SEEDS])
        low, high = np.percentile(draws, [2.5, 97.5])
        assert np.isclose(weak - strong, delta.mean())
        lines.append(f"| {TYPES.get(kind, kind)} | {n} | {strong:.3f} | {weak:.3f} | {delta.mean()*100:+.1f} | "
                     f"[{low*100:+.1f}, {high*100:+.1f}] | " + ", ".join(f"{x*100:+.1f}" for x in delta) + " |")
        summaries.append(dict(product=product, defect_type=kind, n=n, strong_auroc=strong, weak_auroc=weak,
                              delta_pp=delta.mean()*100, lower95_pp=low*100, upper95_pp=high*100,
                              seed42_pp=delta[0]*100, seed0_pp=delta[1]*100, seed1_pp=delta[2]*100))
        if kind != "Overall" and delta.mean() >= .05 and (delta > 0).sum() >= 2:
            gain.append(kind)
        if kind != "Overall" and delta.mean() <= -.05 and (delta < 0).sum() >= 2:
            loss.append(kind)
    outcomes[product] = dict(opposing_effects=bool(gain and loss), gains=gain, losses=loss)
    lines.extend(["", f"Opposing-effect criterion: **{'met' if gain and loss else 'not met'}**. The rule uses mean effects of at least +5 and at most −5 percentage points on different types with the same direction in at least two seeds.", "",
                  "| Seed | Arm | Normal-calibrated detections | Normal-calibrated false alarms | Matched-budget detections | Diagnostic false alarms |",
                  "|---|---|---:|---:|---:|---:|"])
    for seed in SEEDS:
        for arm in ["strong", "weak"]:
            row = metrics[product, seed, arm, "Overall"]
            lines.append(f"| {seed} | {arm} | {row['tp']}/{row['n']} | {row['fp']}/{row['normal_n']} | "
                         f"{row['diagnostic_tp']}/{row['n']} | {row['diagnostic_fp']}/{row['normal_n']} |")

lines.extend(["", "The matched-budget columns describe a test-normal ROC diagnostic. The fixed decisions use thresholds set before test access. Full defect-level alarm counts are in METRICS.csv."])
(OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
with (OUT / "PAIRED_EFFECTS.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
(OUT / "DECISIONS.json").write_text(json.dumps(outcomes, indent=2) + "\n", encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print("\n".join(lines))
