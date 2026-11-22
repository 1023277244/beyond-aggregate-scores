"""Draw the fixed endpoint comparison from the verified regional results."""
import csv
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from run_region_audit import OUT, read, sha
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    receipt = read(OUT / "RECEIPT.json")
    assert sha(OUT / "paired_region_effects.csv") == receipt["output_files"]["paired_region_effects.csv"]
    with (OUT / "paired_region_effects.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {(r["defect_type"], r["comparison"]): r for r in csv.DictReader(stream) if r["arm"] == "M"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 2.65), sharey=True, layout="constrained")
    for ax, kind, title, color in zip(axes, ("Scratch", "MissingPart"),
            ("(a) Scratch", "(b) Missing part"), ("#005E7A", "#B44826")):
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.axvline(0, color="#86919C", lw=.9, ls="--", zorder=0)
        ax.grid(axis="y", color="#e7e9eb", linewidth=.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#9AA3AA")
        ax.set_xlim(-45, 44)
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_ylim(-.55, 2.55)
        ax.set_xlabel("Pairwise ranking change (pp), $\\alpha=1/3$ vs. $1$")
        ax.set_yticks([2, 1, 0], ["Full readout", "Exclude matched controls", "Exclude annotation"])
        ax.tick_params(axis="y", length=0)
        for level, condition, marker in zip((2, 1, 0),
                ("full", "control_excluded", "annotation_excluded"), ("o", "s", "D")):
            row = rows[kind, condition]
            point, low, high = (float(row[k]) for k in ("mean_pp", "lower95_pp", "upper95_pp"))
            ax.errorbar(point, level, xerr=[[point-low], [high-point]], fmt=marker,
                        markersize=5.2, capsize=2.8, elinewidth=1.35, color=color)
            ax.annotate(f"{point:+.1f}", (point, level), xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom", color=color, fontsize=8.3)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"ceramic_region_effects.{suffix}", facecolor="white", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
