"""Execute the four preselected products with fixed strength, model and budgets."""
import time
import zipfile
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from run_e2_cross_product import (
    ROOT, WEIGHTS, ANNOTATIONS, archive_path, manifest_for, read_images,
    file_sha, array_hash, features, score_and_time,
)
from run_student_probe import Head, balance_bce, read_json, variants
from run_public_pilot import encoder
from run_layer3_probe import fingerprint, cpu_state
from run_e1_strength import save_json
from run_pilot import write_csv

OUT = ROOT / "e3_cross_product_v1"
PROTOCOL = ROOT / "e3_cross_product_protocol.json"
RESERVATION = ROOT / "additional_product_reservation_v1.json"


def prepare_seed(images, manifest, seed, arms):
    pool = np.array([i for i,r in enumerate(manifest) if r["split"] == "normal_pool"])
    indices = np.random.default_rng(seed).permutation(pool)[:12].tolist()
    groups = [manifest[i]["group_id"] for i in indices]
    ranks = {g:r for r,g in enumerate(sorted(groups))}
    queries = {arm:[] for arm in arms}
    targets, audit = [], []
    schedule = torch.randint(0,96,(600,4),generator=torch.Generator().manual_seed(seed))
    draws = np.bincount(schedule.numpy().ravel(), minlength=96)
    for index, group in zip(indices, groups):
        source = images[index]
        source_float = source.astype(np.float32)
        for position, (operation, old, mask) in enumerate(variants(source,index)):
            query = len(targets)
            targets.append(cv.resize(mask,(32,32),interpolation=cv.INTER_AREA).astype(np.float32).reshape(-1))
            for arm in arms:
                new = old
                if mask.any():
                    alpha = {"S0":1.,"M":1/3,
                             "S1":(.1,.3,.6)[(ranks[group]+position-5)%3]}[arm]
                    if arm != "S0":
                        # Preserve each historical arm's floating point construction.
                        new_float = ((2/3)*source_float + (1/3)*old.astype(np.float32)) if arm=="M" else ((1-alpha)*source_float + alpha*old.astype(np.float32))
                        new = np.clip(np.rint(new_float),0,255).astype(np.uint8)
                    delta = new.astype(np.float64)-source
                    ideal_delta = alpha*(old.astype(np.float64)-source)
                    error = np.abs(delta-ideal_delta)
                    assert error.max() <= .501 and np.all(delta[mask==0]==0)
                    rounded = np.rint(source.astype(np.float64)+ideal_delta)
                    audit.append(dict(seed=seed,arm=arm,query=query,group=group,source_index=index,
                        operation=operation,alpha=alpha,draws=int(draws[query]),
                        source_sha=array_hash(source),original_sha=array_hash(old),query_sha=array_hash(new),
                        mask_sha=array_hash(mask.astype(bool)),mae=float(np.abs(delta[mask!=0]).mean()),
                        rounding=float(error.max()),clipped_channels=int(((rounded<0)|(rounded>255)).sum()),
                        changed_pixels=int(np.any(delta!=0,axis=2).sum())))
                else:
                    assert np.array_equal(new,old)
                queries[arm].append(new)
    assert len(targets)==96 and len(groups)==len(set(groups))==12
    return dict(indices=indices,groups=groups,queries={a:np.stack(v) for a,v in queries.items()},
                targets=np.stack(targets),schedule=schedule,audit=audit)


def train_seed(path, prepared, model, seed, lock, spec):
    clean = features(model, prepared["queries"]["S0"][::8],"cuda").flatten(0,1)
    mean, std = clean.mean(0), clean.std(0).clamp_min(.1)
    torch.manual_seed(seed)
    initial = cpu_state(Head(mean,std))
    initial_sha = fingerprint(initial)
    schedule_sha = array_hash(prepared["schedule"].numpy())
    labels = torch.as_tensor(prepared["targets"],device="cuda")
    schedule = prepared["schedule"].cuda()
    for arm in spec["arms"]:
        checkpoint = path/f"{seed}_{arm}.pt"
        if checkpoint.exists():
            existing = torch.load(checkpoint,map_location="cpu",weights_only=True)
            assert existing["lock"]==lock and existing["initial"]==initial_sha
            assert existing["schedule_sha"]==schedule_sha
            assert existing["queries_sha"]==array_hash(prepared["queries"][arm])
            continue
        x = features(model,prepared["queries"][arm],"cuda")
        head = Head(mean,std).cuda()
        head.load_state_dict(initial)
        optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        logs=[]
        for step,batch in enumerate(schedule,start=1):
            logits,_,_ = head(x[batch])
            loss = balance_bce(logits,labels[batch])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in (1,100,200,300,400,500,600):
                logs.append(dict(step=step,bce=float(loss.detach())))
        weights = {k:v for k,v in cpu_state(head).items() if not k.startswith("adapter.")}
        write_csv(path/f"{seed}_{arm}_inputs.csv",[r for r in prepared["audit"] if r["arm"]==arm])
        write_csv(path/f"{seed}_{arm}_loss.csv",logs)
        temporary = checkpoint.with_suffix(".part")
        torch.save(dict(head=weights,groups=prepared["groups"],initial=initial_sha,
             schedule_sha=schedule_sha,mask_sha=array_hash(prepared["targets"]),
             queries_sha=array_hash(prepared["queries"][arm]),steps=600,lock=lock),temporary)
        temporary.replace(checkpoint)
        print(f"{path.name} seed {seed} {arm} 600/600",flush=True)


def main():
    spec = read_json(PROTOCOL)
    protocol_hash = file_sha(PROTOCOL)
    assert protocol_hash==(ROOT/"e3_cross_product_protocol.sha256").read_text().split()[0]
    reservation = read_json(RESERVATION)
    assert [r["category"] for r in reservation["selected"]]==spec["products"]
    assert file_sha(ANNOTATIONS)=="5d16bd5f9ce8c9ca7f20009edfb011e8775fb7febadb9e8edf84c09604204985"
    OUT.mkdir(exist_ok=True)
    sources = [Path(__file__), PROTOCOL, RESERVATION, WEIGHTS, ANNOTATIONS]
    sources += [ROOT/n for n in ("run_e2_cross_product.py","run_student_probe.py","run_public_pilot.py",
                                 "run_layer3_probe.py")]
    sources += [ROOT.parent/n for n in ("run_pilot.py","structured_pruning.py")]
    frozen = {str(p.relative_to(ROOT.parent)):file_sha(p) for p in sources}
    if (OUT/"input_lock.json").exists():
        assert frozen==read_json(OUT/"input_lock.json")
    else:
        save_json(OUT/"input_lock.json",frozen)
    cv.setNumThreads(3)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model = encoder("ResNet18_PatchCore","cuda")[0].requires_grad_(False).eval()
    initial_encoder = fingerprint(model.state_dict())
    start = time.monotonic()
    for entry in reservation["selected"]:
        name = entry["category"]
        primary_path = OUT/name
        primary_path.mkdir(exist_ok=True)
        status_path = primary_path/"status.json"
        if status_path.exists() and read_json(status_path).get("state")=="complete":
            print(name,"already complete with identical source locks",flush=True)
            continue
        archive = archive_path(name,entry["bytes"])
        if archive is None:
            save_json(status_path,dict(state="archive_needed",bytes=entry["bytes"]))
            continue
        try:
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None:
                    raise ValueError("Archive CRC failure")
            manifest, meta = manifest_for(name)
            save_json(primary_path/"manifest.json",manifest)
            save_json(primary_path/"metadata.json",meta)
            if meta["eligible_train"]<18:
                save_json(status_path,dict(state="insufficient_normal_groups",metadata=meta))
                continue
            images, natives = read_images(archive,manifest,name)
        except (ValueError,KeyError,zipfile.BadZipFile) as exc:
            save_json(status_path,dict(state="data_audit_stopped",error=str(exc)))
            print(name,str(exc),flush=True)
            continue
        # All nine input arms are inspected before any head for this product is trained.
        preparations = {s:prepare_seed(images,manifest,s,spec["arms"]) for s in spec["seeds"]}
        audit = [r for p in preparations.values() for r in p["audit"]]
        write_csv(primary_path/"generation_audit.csv",audit)
        zeros = {f"{s}_{a}":sum(r["seed"]==s and r["arm"]==a and r["changed_pixels"]==0 for r in audit)
                 for s in spec["seeds"] for a in spec["arms"]}
        supplementary = any(zeros.values())
        path = OUT/"supplement"/name if supplementary else primary_path
        path.mkdir(parents=True,exist_ok=True)
        save_json(path/"manifest.json",manifest)
        save_json(path/"metadata.json",meta)
        lock = dict(protocol=protocol_hash,reservation=file_sha(RESERVATION),
             annotation=meta["annotation_sha256"],archive=file_sha(archive),sources=frozen,
             encoder=initial_encoder,analysis_tier="supplement" if supplementary else "primary")
        save_json(status_path,dict(state="training",tier=lock["analysis_tier"],zero_change_counts=zeros,
              primary_input_guard_stopped=supplementary,result_path=str(path.relative_to(OUT)),lock=lock))
        print(name,"tier",lock["analysis_tier"],"zero-change",zeros,flush=True)
        for seed,p in preparations.items():
            train_seed(path,p,model,seed,lock,spec)
        assert fingerprint(model.state_dict())==initial_encoder
        score_and_time(path,manifest,images,natives,model,"cuda",spec,lock)
        save_json(path/"score_state.json",dict(state="scored",lock=lock,
                     checkpoint_sha={p.name:file_sha(p) for p in path.glob("*.pt")},
                     scores_sha=file_sha(path/"scores.csv"),timing_sha=file_sha(path/"timing.csv")))
        save_json(status_path,dict(state="complete",tier=lock["analysis_tier"],zero_change_counts=zeros,
              primary_input_guard_stopped=supplementary,result_path=str(path.relative_to(OUT)),lock=lock,
              groups=len(manifest),heads=9))
        print(name,"finished",round(time.monotonic()-start,1),"seconds",flush=True)
        del preparations,images,natives
        torch.cuda.empty_cache()
    assert frozen=={str(p.relative_to(ROOT.parent)):file_sha(p) for p in sources}
    save_json(OUT/"run_state.json",dict(state="finished",protocol_sha=protocol_hash,
        elapsed_seconds=time.monotonic()-start,encoder_unchanged=fingerprint(model.state_dict())==initial_encoder))


if __name__=="__main__":
    main()
