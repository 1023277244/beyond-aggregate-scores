"""Paired event comparisons and stratified group bootstrap for the locked E1 study."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e1_strength_v1"
PRIOR = ROOT / "supervision_preservation_v1"
PROTOCOL = ROOT / "e1_protocol.json"
SEEDS = tuple(json.loads(PROTOCOL.read_text(encoding="utf-8"))["seeds"])
ARMS = ("S0_current_frozen","S1_blended_frozen")

NAMES = ("S0", "M", "S1")
KINDS = ("Damage", "Discoloration", "ImpactDamage", "Warpage")
STRATA = ("Overall",) + KINDS
CONTRASTS = (("S0","M"), ("M","S1"))
DRAWS = 2000


def bootstrap_weights(size, rng):
    indices = rng.integers(size, size=(DRAWS,size))
    counts = np.zeros((DRAWS,size),dtype=np.float64)
    np.add.at(counts,(np.arange(DRAWS)[:,None],indices),1)
    return indices,counts


def analyze():
    scores = pd.read_csv(OUT / "original_scores.csv")
    assert not scores.duplicated(["seed","arm","group_id"]).any()
    assert len(scores)==3*3*249
    rng = np.random.default_rng(1120)
    order = scores.query("seed==42 and arm=='S0' and split=='test'").sort_values("group_id")
    ok_ids = order.loc[order.label==0,"group_id"].tolist()
    bad_ids = [order.loc[order.anomaly_class==k,"group_id"].tolist() for k in KINDS]
    assert [len(ok_ids),*[len(ids) for ids in bad_ids]]==[25,50,50,49,50]
    ni,nw = bootstrap_weights(len(ok_ids),rng)
    pos_weights = [bootstrap_weights(len(ids),rng)[1] for ids in bad_ids]
    summary, paired, changes, distributions = [],[],[],{}
    for seed in SEEDS:
        predictions = {}
        for arm in NAMES:
            group = scores[(scores.seed==seed)&(scores.arm==arm)].set_index("group_id")
            calibration = group[group.split=="calibration"]
            assert len(calibration)==6 and (calibration.label==0).all()
            cut = calibration.score.max()
            v = group.score
            normal = v.loc[ok_ids].to_numpy()
            low_cut = np.sort(normal)[-2]
            boot_cuts = np.partition(normal[ni],-2,axis=1)[:,-2]
            all_ids = ok_ids + sum(bad_ids,[])
            events = group.loc[all_ids,["score","label","anomaly_class"]].copy()
            events["fixed"] = events.score>cut
            events["lowfp"] = events.score>low_cut
            predictions[arm] = events
            class_reps = []
            for ids,pw in zip(bad_ids,pos_weights):
                pos = v.loc[ids].to_numpy()
                win = (pos[:,None]>normal).astype(float)+.5*(pos[:,None]==normal)
                auc = np.sum((pw @ win)*nw,axis=1)/(len(ids)*len(normal))
                fixed = (pw @ (pos>cut))/len(ids)
                low = np.sum(pw*(pos[None,:]>boot_cuts[:,None]),axis=1)/len(ids)
                class_reps.append(np.column_stack((auc,fixed,low)))
            all_reps = np.average(class_reps,axis=0,weights=[len(g) for g in bad_ids])
            for k, bids, reps in zip(STRATA,[sum(bad_ids,[])]+bad_ids,[all_reps]+class_reps):
                cohort=events.loc[ok_ids+bids]
                pos=events.loc[bids]
                auc=roc_auc_score(cohort.label,cohort.score)
                fpr,tpr,_=roc_curve(cohort.label,cohort.score,drop_intermediate=False)
                assert np.isclose(tpr[fpr<=1/25+1e-12].max(),pos.lowfp.mean())
                summary.append(dict(seed=seed,arm=arm,stratum=k,n_defects=len(bids),auroc=auc,
                    calibration_threshold=cut,tp=int(pos.fixed.sum()),fn=int((~pos.fixed).sum()),
                    fp=int((normal>cut).sum()),fixed_recall=pos.fixed.mean(),
                    lowfp_cutoff=low_cut,lowfp_tp=int(pos.lowfp.sum()),lowfp_fp=int((normal>low_cut).sum()),
                    lowfp_recall=pos.lowfp.mean()))
                distributions[seed,arm,k]=reps
        for old,new in CONTRASTS:
            left,right=predictions[old],predictions[new]
            for gid in left.index:
                l,r=left.loc[gid],right.loc[gid]
                row=dict(seed=seed,contrast=new+"-"+old,group_id=gid,anomaly_class=l.anomaly_class,
                         label=int(l.label),before_score=l.score,after_score=r.score,score_change=r.score-l.score)
                for field in ("fixed","lowfp"):
                    b,a=int(l[field]),int(r[field])
                    row.update({field+"_before":b,field+"_after":a,field+"_alarm_change":a-b})
                changes.append(row)
    df=pd.DataFrame(summary)
    df.to_csv(OUT/"metrics.csv",index=False)
    vals=("auroc","fixed_recall","lowfp_recall")
    for old,new in CONTRASTS:
        for k in STRATA:
            a=df[(df.stratum==k)&(df.arm==old)].set_index("seed")
            b=df[(df.stratum==k)&(df.arm==new)].set_index("seed")
            delta=(b[list(vals)]-a[list(vals)]).loc[list(SEEDS)]
            reps=np.mean([distributions[s,new,k]-distributions[s,old,k] for s in SEEDS],axis=0)
            for j,name in enumerate(vals):
                lo,hi=np.quantile(reps[:,j],[.025,.975])
                paired.append(dict(contrast=new+"-"+old,stratum=k,metric=name,mean_delta=delta[name].mean(),
                    lower95=lo,upper95=hi,**{f"seed{s}":delta.loc[s,name] for s in SEEDS}))
    differences=pd.DataFrame(paired)
    differences.to_csv(OUT/"paired_uncertainty.csv",index=False)
    ev=pd.DataFrame(changes)
    ev.to_csv(OUT/"paired_events.csv",index=False)
    flips=[]
    for (seed,c,k), part in ev.groupby(["seed","contrast","anomaly_class"]):
        for field in ("fixed","lowfp"):
            plus=(part[field+"_alarm_change"]==1).sum()
            minus=(part[field+"_alarm_change"]==-1).sum()
            flips.append(dict(seed=seed,contrast=c,stratum=k,threshold=field,n=len(part),
                              newly_alarmed=plus,no_longer_alarmed=minus,unchanged=len(part)-plus-minus))
    pd.DataFrame(flips).to_csv(OUT/"alarm_changes.csv",index=False)
    # Historical operating points and per-type ranking checked against independently saved summaries.
    for seed in SEEDS:
        source=PRIOR/f"seed{seed}"
        for prior_file,overall in (("metrics.csv",True),("per_defect_metrics.csv",False)):
            hist=pd.read_csv(source/prior_file)
            for arm,prev in zip(("S0","S1"),ARMS[:2]):
                ref=hist[(hist.method==prev)&(hist.condition=="original")]
                for _,r in ref.iterrows():
                    k="Overall" if overall else r.defect_type
                    out=df[(df.seed==seed)&(df.arm==arm)&(df.stratum==k)].iloc[0]
                    assert np.allclose(out[["auroc","tp","fp","fn"]].astype(float),
                                       r[["auroc","tp","fp","fn"]].astype(float),atol=1e-10,rtol=0)
    return df,differences


def pixel_summary():
    rows=[]
    for seed in SEEDS:
        inp=pd.read_csv(OUT/f"seed{seed}_inputs.csv").query("synthetic==1")
        assert len(inp)==36
        for arm in NAMES:
            alpha=inp[arm+"_alpha"]
            rows.append(dict(seed=seed,arm=arm,mean_alpha=alpha.mean(),
                sampled_mean_alpha=np.average(alpha,weights=inp.draws),
                mean_masked_rgb_mae=inp[arm+"_mae"].mean(), mean_masked_rgb_rmse=inp[arm+"_rmse"].mean(),
                clipped_channels=inp[arm+"_clipped_channels"].sum(),
                max_channel_rounding_error=inp[arm+"_max_quant_error"].max()))
    pix=pd.DataFrame(rows)
    pix.to_csv(OUT/"pixel_strength_summary.csv",index=False)
    return pix


def render(df, diffs):
    fig,ax=plt.subplots(figsize=(8.6,4.5),layout="constrained")
    for idx,(old,new) in enumerate(CONTRASTS):
        q=diffs[(diffs.contrast==new+"-"+old)&(diffs.metric=="auroc")].set_index("stratum").loc[list(STRATA)]
        y=np.arange(len(STRATA))+(-.12 if idx==0 else .12)
        color=["#205e8e","#b05026"][idx]
        ax.hlines(y,q.lower95,q.upper95,color=color,lw=2)
        ax.scatter(q.mean_delta,y,color=color,s=38,label=new+" minus "+old,zorder=3)
    ax.set_yticks(range(5),STRATA);ax.invert_yaxis()
    ax.axvline(0,color="#333333",lw=.8)
    ax.set_xlabel("Paired change in AUROC (three fitted seeds averaged)")
    ax.set_title("Changing synthetic strength on the receiver module")
    ax.legend(frameon=False)
    fig.text(.5,-.015,"95% paired group-bootstrap intervals conditional on this product and the fitted models.",ha="center",fontsize=9)
    fig.savefig(OUT/"paired_auc.png",dpi=160,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    df,diffs=analyze()
    pix=pixel_summary()
    render(df,diffs)
    verification=dict(protocol=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),analysis_script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         scores_verified_rows=249*9, metrics_rows=len(df), contrasts_rows=len(diffs), true_events_in_paired_csv=3*2*199,
         group_bootstrap_draws=DRAWS, bootstrap_seed=1120, old_rank_and_confusion_metrics_match=True)
    (OUT/"analysis_verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    print(df.groupby(["stratum","arm"]).auroc.mean().unstack().to_string())
    print(df[df.stratum=="Overall"][["seed","arm","tp","fn","fp","lowfp_tp"]].to_string(index=False))
    print(diffs[diffs.metric=="auroc"].to_string(index=False))
    print(pix.to_string(index=False))
