"""Summarize all sealed scores and reproduce the archived endpoint metrics."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OLD = ROOT / "strength_extension_20261121_v1/A"
PRODUCT = "ceramic_wave_filter"
DATA = OLD / PRODUCT
sys.path[:0] = [str(OLD), str(ROOT / "bottle_cap_pilot/public_benchmark")]

from common import read, save, sha, now
import analyze_e2_cross_product as analysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "Overall": ("Overall", "#363D4A", "o", "--"),
    "Scratch": ("Scratch", "#005E7A", "o", "-"),
    "MissingPart": ("Missing part", "#B44826", "s", "-"),
    "CharacterOmission": ("Missing marking", "#33786D", "^", "-"),
    "Deformation": ("Deformation", "#765080", "D", "-"),
}


def verify_archived_scores(raw, cal):
    old_test = pd.read_csv(OLD / "raw_test_scores.csv")
    old_cal = pd.read_csv(OLD / "normal_calibration_scores.csv")
    keys = ["seed", "arm", "group_id"]
    checks = {}
    for kind, new, old in (("test", raw, old_test), ("calibration", cal, old_cal)):
        current = new[new.arm.isin(("S0", "M"))].set_index(keys).sort_index()
        reference = old[old["product"] == PRODUCT].set_index(keys).sort_index()
        assert len(current) == 6 * (224 if kind == "test" else 6)
        assert reference.index.equals(current.index)
        fields = ["score"] if kind == "test" else ["score", "threshold"]
        delta = (current[fields] - reference[fields]).abs().to_numpy()
        checks[kind] = {"archived_scores_checked": len(current), "max_abs_delta": float(delta.max())}
        np.testing.assert_allclose(current[fields], reference[fields], atol=1e-5, rtol=0)
    return checks


def plot_results(means, arms):
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    coefficients = list(arms.values())
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.1, 3.25), layout="constrained")
    for axis in (left, right):
        axis.set_xticks(coefficients, ["1/3", "1/2", "2/3", "5/6", "1"])
        axis.set_xlabel("Synthetic blend coefficient $\\alpha$")
        axis.grid(axis="y", color="#e4e7ea", linewidth=.65)
        axis.set_axisbelow(True)
        axis.spines[["left", "bottom"]].set_color("#9097a1")

    for name, (label, color, marker, line) in LABELS.items():
        curve = means[means.stratum == name].set_index("arm").loc[list(arms)]
        left.plot(coefficients, 100 * curve.auroc, color=color, marker=marker, ls=line,
                  lw=1.65, markersize=4.5, label=label)
        if name in ("Scratch", "MissingPart"):
            right.plot(coefficients, curve.lowfp_tp, color=color, marker=marker, lw=1.65,
                       markersize=4.5, label=label)
    left.set_title("(a) Defect-specific ranking", loc="left", fontweight="bold")
    left.set_ylabel("AUROC (%)")
    left.legend(loc="best", frameon=False, fontsize=8, ncol=2, columnspacing=.9)
    right.set_title("(b) Detections at a matched FP budget", loc="left", fontweight="bold")
    right.set_ylabel("Mean detections (out of 50)")
    right.set_ylim(0, 50)
    right.legend(loc="best", frameon=False, fontsize=8)
    right.text(.025, .955, "Post-hoc ROC diagnostic\n≤1 FP among 25 test normals",
               transform=right.transAxes, va="top", fontsize=7.2, color="#545c65")
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"ceramic_strength_curve.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def report(seed_metrics, means, arms, checks):
    lines = [
        "# Ceramic filter strength sweep",
        "",
        "The same C1 ceramic-filter units were evaluated at five fixed blend coefficients. "
        "The points are arithmetic means of three matched seeds. All four real defect types "
        "are shown. The product was selected after earlier analysis.",
        "",
        "| α | Overall AUROC | Scratch | Missing part | Missing marking | Deformation |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    names = list(LABELS)
    fractions = dict(zip(arms, ("1/3", "1/2", "2/3", "5/6", "1")))
    for arm, frac in fractions.items():
        one = means[means.arm == arm].set_index("stratum")
        lines.append("| " + " | ".join([frac, *[f"{100*one.loc[name, 'auroc']:.2f}%" for name in names]]) + " |")
    lines.extend([
        "",
        "## Matched false-positive budget",
        "",
        "Maximum of one false positive among 25 test normals. This is a post-hoc ROC diagnostic. "
        "Counts below are ordered by seed 42, 0 and 1. Each defect type in this table has 50 test units.",
        "",
        "| α | Scratch | Missing part |",
        "|---:|:---|:---|",
    ])
    for arm, frac in fractions.items():
        row = [frac]
        for name in ("Scratch", "MissingPart"):
            vals = seed_metrics[(seed_metrics.arm == arm) & (seed_metrics.stratum == name)]
            row.append(" / ".join(str(x) for x in vals.set_index("seed").loc[[42, 0, 1], "lowfp_tp"]))
        lines.append("| " + " | ".join(row) + " |")
    lines.extend([
        "",
        "## Normal-calibrated alarms",
        "",
        "Thresholds use six normal calibration units for each head. Cells give true detections out of 199, "
        "then false alarms out of 25 normals.",
        "",
        "| α | Seed 42 TP, FP | Seed 0 TP, FP | Seed 1 TP, FP |",
        "|---:|:---|:---|:---|",
    ])
    for arm, frac in fractions.items():
        rows = seed_metrics[(seed_metrics.arm == arm) & (seed_metrics.stratum == "Overall")].set_index("seed")
        cells = [f"{rows.loc[s, 'tp']}, {rows.loc[s, 'fp']}" for s in (42, 0, 1)]
        lines.append("| " + " | ".join([frac, *cells]) + " |")
    lines.extend([
        "",
        "The endpoint compatibility check compared "
        f"{checks['test']['archived_scores_checked']} test scores and "
        f"{checks['calibration']['archived_scores_checked']} normal calibration scores. "
        f"Maximum absolute test-score difference was {checks['test']['max_abs_delta']:.3g}. "
        "Training/validation versus test had zero shared group IDs or exact native/resized pixel hashes.",
        "",
        "The per-seed CSV contains every class and both operating points. Paired uncertainty is obtained "
        "by resampling the same official part groups across coefficients and seeds, conditional on "
        "this product and these fits.",
        "",
    ])
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    frozen = read(OUT / "FREEZE.json")
    scored = read(OUT / "ALL_TEST_SCORES_FROZEN.json")
    unlock = read(OUT / "EVALUATION_UNLOCK.json")
    assert scored["unlock_sha"] == sha(OUT / "EVALUATION_UNLOCK.json")
    assert scored["scores_sha"] == sha(OUT / "raw_test_scores.csv")
    assert unlock["calibration_sha"] == sha(OUT / "calibration.csv")
    assert all(sha(ROOT / name) == digest for name, digest in frozen["evaluation_files"].items())
    raw, cal = (pd.read_csv(OUT / name) for name in ("raw_test_scores.csv", "calibration.csv"))
    compatibility = verify_archived_scores(raw, cal)
    arms = frozen["alphas"]
    analysis.NAMES = tuple(arms)
    analysis.CONTRASTS = tuple(("S0", arm) for arm in arms if arm != "S0")
    manifest = pd.DataFrame(read(DATA / "reserved_manifest.json"))
    raw = raw.merge(manifest[["group_id", "split", "label", "anomaly_class"]],
                    on="group_id", validate="many_to_one")
    cal = cal.assign(split="calibration", label=0, anomaly_class="OK")
    metrics, contrasts, paired_parts = analysis.summarize(pd.concat([raw, cal], ignore_index=True))
    assert len(metrics) == 75
    metrics.insert(2, "alpha", metrics.arm.map(arms))
    means = metrics.groupby(["arm", "stratum"], as_index=False).mean(numeric_only=True).drop(columns="seed")
    metrics.to_csv(OUT / "all_seed_metrics.csv", index=False, encoding="utf-8-sig")
    means.to_csv(OUT / "three_seed_means.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(OUT / "paired_uncertainty.csv", index=False, encoding="utf-8-sig")
    paired_parts.to_csv(OUT / "paired_parts.csv", index=False, encoding="utf-8-sig")
    plot_results(means, arms)
    report(metrics, means, arms, compatibility)
    names = ("RESULTS.md", "all_seed_metrics.csv", "three_seed_means.csv",
             "paired_uncertainty.csv", "paired_parts.csv", "ceramic_strength_curve.pdf",
             "ceramic_strength_curve.png")
    save(OUT / "VERIFICATION.json", dict(completed_at_utc=now(), endpoint_compatibility=compatibility,
        result_hashes={name: sha(OUT / name) for name in names}, score_rows=len(raw),
        fitted_configurations=len(raw.groupby(["seed", "arm"])), reporting_script=sha(__file__)))
    print("Complete five-level ceramic-filter sweep", flush=True)


if __name__ == "__main__":
    main()
