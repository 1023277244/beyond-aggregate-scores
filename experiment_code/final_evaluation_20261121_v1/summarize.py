"""Compute all prespecified endpoints once scores for every category are frozen."""
import sys
sys.dont_write_bytecode=True

import numpy as np
import pandas as pd

from common import ROOT, OUT, HISTORY, PRODUCTS, SEEDS, ARMS, now, sha, save, read, write_csv, verify_freeze
sys.path.insert(0,str(HISTORY))
import analyze_e2_cross_product as analysis


def main():
    freeze=verify_freeze()
    scored=read(OUT/"ALL_TEST_SCORES_FROZEN.json")
    assert scored["freeze_sha"]==freeze and scored["all_products_completed"]
    assert sha(OUT/"raw_test_scores.csv")==scored["scores_sha"]
    analysis.NAMES=ARMS
    analysis.CONTRASTS=(("S0","M"),)
    raw=pd.read_csv(OUT/"raw_test_scores.csv")
    calibration=pd.read_csv(OUT/"normal_calibration_scores.csv")
    metrics, contrasts, events, decisions=[],[],[],{}
    for product in PRODUCTS:
        manifest=pd.DataFrame(read(OUT/product/"reserved_manifest.json"))
        one=raw[raw["product"]==product].merge(
            manifest[["group_id","label","anomaly_class","split"]],on="group_id",validate="many_to_one")
        cal=calibration[calibration["product"]==product].copy()
        cal["split"]="calibration"
        cal["label"]=0
        cal["anomaly_class"]="OK"
        scores=pd.concat([one,cal],ignore_index=True)
        assert not scores.duplicated(["group_id","seed","arm"]).any()
        scores.to_csv(OUT/product/"all_scores_with_labels.csv",index=False)
        frame,diff,paired=analysis.summarize(scores)
        frame["product"]=product
        frame["actual_fixed_fpr"]=frame.fp/frame.normal_n
        frame["actual_diagnostic_fpr"]=frame.lowfp_fp/frame.normal_n
        frame["allowed_diagnostic_fp"]=frame.normal_n//25
        frame["record_type"]="seed_metric"
        frame["status"]="complete"
        frame["diagnostic_uses_test_labels"]=True
        metrics.append(frame)
        diff["product"]=product
        diff["record_type"]="paired_contrast"
        diff["status"]="complete"
        contrasts.append(diff)
        paired["product"]=product
        events.append(paired)
        decisions[product]=analysis.decision(diff)
    seed_metrics=pd.concat(metrics,ignore_index=True)
    means=seed_metrics.groupby(["product","arm","stratum"]).mean(numeric_only=True).reset_index()
    means["seed"]="mean_of_3_fixed_seeds"
    means["record_type"]="three_seed_mean"
    means["status"]="complete"
    # Average counts are explicitly distinguishable from each seed's integer counts.
    contrast=pd.concat(contrasts,ignore_index=True)
    contrast["seed"]="paired_mean_of_3"
    contrast["arm"]=contrast["contrast"]
    contrast=contrast.rename(columns={"mean_delta":"paired_mean_delta"})
    combined=pd.concat([seed_metrics,means,contrast],ignore_index=True)
    combined.to_csv(ROOT/"FINAL_EVALUATION_RESULTS.csv",index=False,encoding="utf-8-sig")
    pd.concat(events,ignore_index=True).to_csv(OUT/"paired_event_changes.csv",index=False)
    seed_metrics.to_csv(OUT/"all_seed_metrics.csv",index=False)
    means.to_csv(OUT/"three_seed_means.csv",index=False)
    save(OUT/"confirmatory_decisions.json",dict(calculated_at_utc=now(),products=decisions,
        descriptive_rule=True,all_categories_reported=True,no_further_method_selection=True))
    print(means[means.stratum=="Overall"][["product","arm","auroc","fp","fixed_recall"]].to_string(index=False))
    print("All categories",decisions)
    (OUT/"statistics_complete.json").write_text(__import__("json").dumps(dict(
        completed_at_utc=now(),freeze_sha=freeze,results_sha=sha(ROOT/"FINAL_EVALUATION_RESULTS.csv"),
        result_rows=len(combined),all18_heads_reported=True),indent=2),encoding="utf-8")


if __name__=="__main__":
    main()

