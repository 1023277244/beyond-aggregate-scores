"""No test scoring entry point. Save all18 final heads and normal-only calibration."""
import sys
sys.dont_write_bytecode = True

import numpy as np
import torch

from common import ROOT, OUT, HISTORY, PRODUCTS, SEEDS, ARMS, now, sha, save, read, write_csv, verify_freeze
sys.path.insert(0,str(HISTORY))
from run_e3_cross_product import prepare_seed, array_hash
from run_e2_cross_product import features
from run_student_probe import Head, balance_bce
from run_public_pilot import encoder
from run_layer3_probe import fingerprint, cpu_state
from run_pilot import image_tensor


def configure():
    import cv2 as cv
    cv.setNumThreads(3)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()


def main():
    configure()
    freeze = verify_freeze()
    status = read(OUT/"readiness.json")
    assert status["state"]=="train_ready", "All three data packages must be available"
    model = encoder("ResNet18_PatchCore","cuda")[0].requires_grad_(False).eval()
    body_sha = fingerprint(model.state_dict())
    for product in PRODUCTS:
        folder=OUT/product
        normal=read(folder/"train_manifest.json")
        ready=read(folder/"train_ready.json")
        assert ready["freeze_sha"]==freeze and sha(folder/"train_only_inputs.npz")==ready["cache_sha256"]
        assert all(r["label"]==0 and r["author_split"]=="train" for r in normal)
        with np.load(folder/"train_only_inputs.npz") as data:
            images=data["images"]
        for seed in SEEDS:
            p=prepare_seed(images,normal,seed,ARMS)
            assert len(p["indices"])==12 and all(normal[i]["split"]=="normal_pool" for i in p["indices"])
            clean=features(model,p["queries"]["S0"][::8],"cuda").flatten(0,1)
            mean,std=clean.mean(0),clean.std(0).clamp_min(.1)
            torch.manual_seed(seed)
            initial=cpu_state(Head(mean,std))
            audit=dict(freeze=freeze,data=ready["cache_sha256"],groups=p["groups"],
                       initial=fingerprint(initial),schedule=array_hash(p["schedule"].numpy()),
                       target_masks=array_hash(p["targets"]))
            write_csv(folder/f"seed{seed}_synthetic_inputs.csv",p["audit"])
            for arm in ARMS:
                path=folder/f"seed{seed}_{arm}.pt"
                if path.exists():
                    assert torch.load(path,weights_only=True,map_location="cpu")["audit"]==audit
                    continue
                desc=features(model,p["queries"][arm],"cuda")
                head=Head(mean,std).cuda()
                head.load_state_dict(initial)
                optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
                targets=torch.as_tensor(p["targets"],device="cuda")
                logs=[]
                for step,batch in enumerate(p["schedule"].cuda(),1):
                    loss=balance_bce(head(desc[batch])[0],targets[batch])
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if step in (1,100,200,300,400,500,600):
                        logs.append(dict(step=step,training_bce=float(loss.detach())))
                state={k:v for k,v in cpu_state(head).items() if not k.startswith("adapter.")}
                temp=path.with_suffix(".part")
                torch.save(dict(head=state,audit=audit,arm=arm,seed=seed,steps=600,
                                encoder_sha=body_sha,saved_at_utc=now()),temp)
                temp.replace(path)
                write_csv(folder/f"seed{seed}_{arm}_training_loss.csv",logs)
                print(product,seed,arm,"600/600",flush=True)
    assert fingerprint(model.state_dict())==body_sha
    endpoints={p.relative_to(OUT).as_posix():sha(p) for p in OUT.glob("*/seed*_*.pt")}
    assert len(endpoints)==36
    save(OUT/"ALL_MODELS_FROZEN.json",dict(frozen_at_utc=now(),freeze_sha=freeze,
        endpoints=endpoints,encoder_sha=body_sha,test_pixels_read_for_training=0))
    calibration=[]
    with torch.inference_mode():
        for product in PRODUCTS:
            folder=OUT/product
            manifest=read(folder/"train_manifest.json")
            with np.load(folder/"train_only_inputs.npz") as z:images=z["images"]
            indices=[i for i,r in enumerate(manifest) if r["split"]=="calibration"]
            assert len(indices)==6
            desc=features(model,images[indices],"cuda")
            for seed in SEEDS:
                for arm in ARMS:
                    state=torch.load(folder/f"seed{seed}_{arm}.pt",weights_only=True,map_location="cpu")["head"]
                    head=Head(state["mean"],state["std"]).cuda()
                    del head.adapter
                    head.load_state_dict(state)
                    vals=head(desc)[1].cpu().numpy()
                    threshold=float(vals.max())
                    for i,value in zip(indices,vals):
                        calibration.append(dict(product=product,seed=seed,arm=arm,
                            group_id=manifest[i]["group_id"],score=float(value),threshold=threshold))
    write_csv(OUT/"normal_calibration_scores.csv",calibration)
    save(OUT/"EVALUATION_UNLOCK.json",dict(unlocked_at_utc=now(),freeze_sha=freeze,
        models_frozen_sha=sha(OUT/"ALL_MODELS_FROZEN.json"),calibration_sha=sha(OUT/"normal_calibration_scores.csv"),
        endpoints=36,test_used_for_training_or_selection=False))
    print("All36 endpoints and normal calibration frozen; final evaluation unlocked",flush=True)


if __name__=="__main__":
    main()

