"""Use locked E2 endpoints; replay the development analysis as a regression check."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from analyze_e1_strength import bootstrap_weights, DRAWS, SEEDS, NAMES, CONTRASTS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"e2_cross_product_v1"
PROTOCOL=ROOT/"e2_cross_product_protocol.json"
ENDPOINTS=("auroc","fixed_recall","lowfp_recall")


def summarize(scores):
    assert not scores.duplicated(["seed","arm","group_id"]).any()
    ref=scores[(scores.seed==42)&(scores.arm=="S0")].set_index("group_id").sort_index()
    test=ref[ref.split=="test"]
    normals=test.index[test.label==0].tolist()
    types=sorted(set(test.loc[test.label==1,"anomaly_class"]))
    ids={k:test.index[test.anomaly_class==k].tolist() for k in types}
    defects=sum(ids.values(),[])
    groups=normals+defects
    kfp=len(normals)//25
    rng=np.random.default_rng(1120)
    ni,nw=bootstrap_weights(len(normals),rng) if normals else (None,None)
    weights={k:bootstrap_weights(len(v),rng)[1] for k,v in ids.items()}
    metricrows,uncertainty,events=[],[],[]
    reps={}
    for seed in SEEDS:
        pred={}
        for arm in NAMES:
            one=scores[(scores.seed==seed)&(scores.arm==arm)].set_index("group_id").sort_index()
            assert ref[["split","label","anomaly_class"]].equals(one[["split","label","anomaly_class"]])
            c=one[one.split=="calibration"]
            assert len(c)==6 and (c.label==0).all() and np.isfinite(one.score).all()
            cut=c.score.max()
            normal=one.loc[normals,"score"].to_numpy()
            postcut=np.sort(normal)[-kfp-1] if normals else np.nan
            bootcuts=np.partition(normal[ni],-kfp-1,axis=1)[:,-kfp-1] if normals else None
            pred[arm]=one.loc[groups,"score"].to_numpy(),cut,postcut
            classreps=[]
            for kind,groupids in ids.items():
                pos=one.loc[groupids,"score"].to_numpy()
                w=weights[kind]
                au,lr=np.full(DRAWS,np.nan),np.full(DRAWS,np.nan)
                if normals:
                    pairs=(pos[:,None]>normal).astype(float)+.5*(pos[:,None]==normal)
                    au=((w@pairs)*nw).sum(axis=1)/(len(pos)*len(normals))
                    lr=(w*(pos[None,:]>bootcuts[:,None])).sum(axis=1)/len(pos)
                fixed=w@(pos>cut)/len(pos)
                classreps.append(np.column_stack([au,fixed,lr]))
            totalreps=np.average(classreps,axis=0,weights=[len(v) for v in ids.values()]) if defects else np.full((DRAWS,3),np.nan)
            for (kind,groupids),dist in zip([("Overall",defects),*ids.items()],[totalreps,*classreps]):
                vals=one.loc[groupids,"score"].to_numpy()
                sample=one.loc[normals+groupids]
                auc=roc_auc_score(sample.label,sample.score) if normals and groupids else np.nan
                tp=int((vals>cut).sum())
                fp=int((normal>cut).sum())
                ltp=int((vals>postcut).sum()) if normals else np.nan
                lfp=int((normal>postcut).sum()) if normals else np.nan
                if normals and groupids:
                    fpr,tpr,_=roc_curve(sample.label,sample.score,drop_intermediate=False)
                    assert lfp<=kfp and np.isclose(ltp/len(vals),tpr[fpr<=kfp/len(normals)+1e-12].max())
                metricrows.append(dict(seed=seed,arm=arm,stratum=kind,n=len(vals),normal_n=len(normal),
                    auroc=auc,cal_threshold=cut,tp=tp,fn=len(vals)-tp,fp=fp,tn=len(normal)-fp,
                    fixed_recall=tp/len(vals) if len(vals) else np.nan,lowfp_threshold=postcut,
                    lowfp_tp=ltp,lowfp_fp=lfp,lowfp_recall=ltp/len(vals) if len(vals) else np.nan))
                reps[seed,arm,kind]=dist
        for a,b in CONTRASTS:
            av,ac,ap=pred[a]
            bv,bc,bp=pred[b]
            for i,g in enumerate(groups):
                events.append(dict(seed=seed,contrast=b+"-"+a,group_id=g,type=ref.loc[g,"anomaly_class"],
                    before_score=av[i],after_score=bv[i],fixed_before=int(av[i]>ac),fixed_after=int(bv[i]>bc),
                    lowfp_before=int(av[i]>ap) if normals else np.nan,lowfp_after=int(bv[i]>bp) if normals else np.nan))
    frame=pd.DataFrame(metricrows)
    for a,b in CONTRASTS:
        for kind in ("Overall",*types):
            x=frame[(frame.arm==a)&(frame.stratum==kind)].set_index("seed")
            y=frame[(frame.arm==b)&(frame.stratum==kind)].set_index("seed")
            ds=(y[list(ENDPOINTS)]-x[list(ENDPOINTS)]).loc[list(SEEDS)]
            sampled=np.mean([reps[s,b,kind]-reps[s,a,kind] for s in SEEDS],axis=0)
            for j,field in enumerate(ENDPOINTS):
                lo,hi=np.quantile(sampled[:,j],[.025,.975]) if x.n.iloc[0]>=2 else (np.nan,np.nan)
                uncertainty.append(dict(contrast=b+"-"+a,stratum=kind,n=x.n.iloc[0],metric=field,
                    mean_delta=ds[field].mean(),lower95=lo,upper95=hi,**{f"seed{s}":ds.loc[s,field] for s in SEEDS}))
    return frame,pd.DataFrame(uncertainty),pd.DataFrame(events)


def decision(diffs):
    main=diffs[diffs.contrast=="M-S0"]
    au=main[main.metric=="auroc"].set_index("stratum")
    overall=au.loc["Overall","mean_delta"]
    au=au.drop(index="Overall").query("n>=2")
    low=main[(main.metric=="lowfp_recall")&(main.stratum!="Overall")&(main.n>=2)].set_index("stratum")
    sc=[f"seed{s}" for s in SEEDS]
    gains=list(au.index[(au.mean_delta>=.05)&((au[sc]>0).sum(axis=1)>=2)])
    losses=list(au.index[(au.mean_delta<=-.05)&((au[sc]<0).sum(axis=1)>=2)])
    lowloss=list(low.index[(low.mean_delta<=-.10)&((low[sc]<0).sum(axis=1)>=2)])
    return dict(overall_delta=overall,type_delta_range=au.mean_delta.max()-au.mean_delta.min(),
                gain_types=gains,loss_types=losses,tradeoff=bool(gains and losses) if len(au)>=2 else None,
                low_recall_loss=lowloss,masked_loss=bool(overall>=-.02 and lowloss) if np.isfinite(overall) and len(low) else None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check-e1",action="store_true")
    parser.add_argument("--sensor-supplement",action="store_true")
    args=parser.parse_args()
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()==(ROOT/"e2_cross_product_protocol.sha256").read_text().split()[0]
    if args.check_e1:
        old=ROOT/"e1_strength_v1"
        _,check,_=summarize(pd.read_csv(old/"original_scores.csv"))
        ref=pd.read_csv(old/"paired_uncertainty.csv")
        cols=["contrast","stratum","metric"]
        x=ref.set_index(cols).sort_index()
        y=check.set_index(cols).sort_index()
        assert x.index.equals(y.index)
        assert np.allclose(x.values,y[x.columns].values,atol=1e-10,rtol=0)
        print("All 30 E1 point contrasts, 60 interval endpoints and 90 seed differences reproduced.")
        return
    products=["silicon_cell_sensor"] if args.sensor_supplement else json.loads(PROTOCOL.read_text())["products"]
    base=OUT/"supplement" if args.sensor_supplement else OUT
    decisions={}
    for name in products:
        path=base/name
        status=json.loads((path/"status.json").read_text()) if (path/"status.json").exists() else {}
        if status.get("state")!="scored":
            decisions[name]=dict(state=status.get("state","awaiting_archive"))
            continue
        frame,delta,events=summarize(pd.read_csv(path/"scores.csv"))
        frame.to_csv(path/"metrics.csv",index=False)
        delta.to_csv(path/"paired_uncertainty.csv",index=False)
        events.to_csv(path/"paired_events.csv",index=False)
        decisions[name]={"state":"analyzed",**decision(delta)}
    (base/"product_decisions.json").write_text(json.dumps(decisions,indent=2),encoding="utf-8")
    print(json.dumps(decisions,indent=2))


if __name__=="__main__":
    main()
