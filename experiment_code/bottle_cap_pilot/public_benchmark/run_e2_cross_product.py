"""Prepare and execute the locked, normal-only cross-product study on local author archives."""
import argparse
import collections
import hashlib
import json
import time
import zipfile
from pathlib import Path, PurePosixPath

import cv2 as cv
import numpy as np
import torch

from run_e1_strength import array_hash, save_json
from run_student_probe import ROOT, Head, balance_bce, read_json, variants
from run_supervision_preservation import prepare_queries
from run_layer3_probe import fingerprint, cpu_state
from run_public_pilot import encoder
from run_pilot import image_tensor, write_csv

OUT = ROOT / "e2_cross_product_v1"
PROTO = ROOT / "e2_cross_product_protocol.json"
WEIGHTS = ROOT / "assets/resnet18-f37072fd.pth"
ANNOTATIONS = ROOT / "downloads/realiadvariety_jsons.zip"


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def archive_path(product, expected):
    locations = [ROOT/"downloads", Path.home()/"Downloads"]
    for base in locations:
        candidates = [base/(product+".zip"), *sorted(base.glob(product+" (*).zip"))]
        for path in candidates:
            if path.is_file() and path.stat().st_size == expected:
                return path
    return None


def manifest_for(product):
    with zipfile.ZipFile(ANNOTATIONS) as z:
        raw = z.read("Real-IAD_Variety_jsons/"+product+".json")
        author = json.loads(raw)
    selected=[]
    missing={}
    for split in ("train","test"):
        seen=set()
        chosen=set()
        for row in author[split]:
            path=PurePosixPath(row["image_path"])
            group=path.parent.as_posix()
            seen.add(group)
            if "_C1_" not in path.name:
                continue
            if group in chosen or row["category"]!=product:
                raise ValueError("duplicate C1 or category mismatch")
            chosen.add(group)
            label=int(row["anomaly_class"]!="OK")
            if split=="train" and label:
                raise ValueError("training set includes positive")
            selected.append(dict(group_id=group,sample_id=str(path),label=label,author_split=split,
                                 anomaly_class=row["anomaly_class"],split=split))
        missing[split]=sorted(seen-chosen)
    training=[r for r in selected if r["author_split"]=="train"]
    testing=[r for r in selected if r["author_split"]=="test"]
    if {r["group_id"] for r in training}&{r["group_id"] for r in testing}:
        raise ValueError("official train and test groups overlap")
    cal={r["group_id"] for r in sorted(training,key=lambda r:hashlib.sha256(("20261120:cal:"+r["group_id"]).encode()).hexdigest())[:6]}
    for r in training:
        r["split"]="calibration" if r["group_id"] in cal else "normal_pool"
    summary=dict(product=product,annotation_sha256=hashlib.sha256(raw).hexdigest(),missing_C1=missing,
        eligible_train=len(training),calibration_groups=sorted(cal),test_normals=sum(not r["label"] for r in testing),
        test_defects=dict(collections.Counter(r["anomaly_class"] for r in testing if r["label"])))
    return selected,summary


def read_images(path, manifest, product):
    resized=[]
    train_originals={}
    pixels={"train":set(),"test":set()}
    with zipfile.ZipFile(path) as z:
        names=set(z.namelist())
        for i,r in enumerate(manifest):
            rel=r["sample_id"]
            found=[v for v in (rel,product+"/"+rel) if v in names]
            if len(found)!=1:
                raise ValueError(f"Missing or ambiguous C1 entry {rel}")
            blob=z.read(found[0])
            image=cv.imdecode(np.frombuffer(blob,np.uint8),cv.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable C1 {rel}")
            pixels[r["author_split"]].add(array_hash(image))
            resized.append(cv.resize(image,(256,256),interpolation=cv.INTER_AREA))
            if r["split"]=="normal_pool":
                train_originals[i]=image
    if pixels["train"] & pixels["test"]:
        raise ValueError("Identical decoded C1 image across fit and test")
    return np.stack(resized),train_originals


def features(model, images,device):
    with torch.no_grad():
        return torch.cat([model(image_tensor(img[None],device)) for img in images])


def train_product(path, manifest, images, model, seed, device, lock):
    pool=np.array([i for i,r in enumerate(manifest) if r["split"]=="normal_pool"])
    indices=np.random.default_rng(seed).permutation(pool)[:12].tolist()
    cohort=dict(indices=indices,group_ids=[manifest[i]["group_id"] for i in indices])
    old,distributed,masks,rec=prepare_queries(manifest,images,cohort)
    mid=old.copy()
    for i,r in enumerate(rec):
        if r["synthetic"]:
            clean=images[r["source_index"]].astype(np.float32)
            mid[i]=np.clip(np.rint((2/3)*clean+(1/3)*old[i].astype(np.float32)),0,255).astype(np.uint8)
    sched=torch.randint(0,96,(600,4),generator=torch.Generator().manual_seed(seed))
    counts=np.bincount(sched.numpy().ravel(),minlength=96)
    schedule_sha=array_hash(sched.numpy())
    sched=sched.to(device)
    pseudo=torch.as_tensor(masks,device=device)
    norm=features(model,images[indices],device).flatten(0,1)
    mean,std=norm.mean(0),norm.std(0).clamp_min(.1)
    torch.manual_seed(seed)
    init=cpu_state(Head(mean,std))
    initial=fingerprint(init)
    for arm,queries in (("S0",old),("M",mid),("S1",distributed)):
        ckpt=path/f"{seed}_{arm}.pt"
        if ckpt.exists():
            pkg=torch.load(ckpt,map_location="cpu",weights_only=True)
            if pkg["lock"]!=lock or pkg["initial"]!=initial or pkg["schedule_sha"]!=schedule_sha:
                raise ValueError("Checkpoint lock mismatch")
            continue
        audit=[]
        for q,r in enumerate(rec):
            if not r["synthetic"]:
                assert np.array_equal(queries[q],old[q])
                continue
            source=images[r["source_index"]].astype(np.float32)
            delta=old[q].astype(np.float32)-source
            alpha={"S0":1., "M":1/3, "S1":r["alpha"]}[arm]
            diff=queries[q].astype(np.float32)-source
            # Recover the original binary geometric mask rather than relying on actual RGB differences.
            mask=next(mask for name,_,mask in variants(images[r["source_index"]],r["source_index"]) if name==r["variant"]).astype(bool)
            err=np.abs(diff-alpha*delta)
            assert np.all(diff[~mask]==0) and err.max()<=.501 and np.any(diff)
            audit.append(dict(query=q,operation=r["variant"],group=r["group_id"],alpha=alpha,draws=int(counts[q]),
                              mask_sha=array_hash(mask),mae=float(np.abs(diff[mask]).mean()),rounding=float(err.max())))
        desc=features(model,queries,device)
        h=Head(mean,std).to(device)
        h.load_state_dict(init)
        opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001)
        for batch in sched:
            out,_,_=h(desc[batch])
            loss=balance_bce(out,pseudo[batch])
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        state={k:v for k,v in cpu_state(h).items() if not k.startswith("adapter.")}
        write_csv(path/f"{seed}_{arm}_inputs.csv",audit)
        part=ckpt.with_suffix(".part")
        torch.save(dict(head=state,groups=cohort["group_ids"],initial=initial,schedule_sha=schedule_sha,
                        mask_sha=array_hash(masks),queries_sha=array_hash(queries),steps=600,lock=lock),part)
        part.replace(ckpt)
        print(f"{path.name} seed {seed} {arm}: training finished",flush=True)


@torch.inference_mode()
def score_and_time(path, manifest, images, natives, model, device, spec, lock):
    heads={}
    for seed in spec["seeds"]:
        for arm in spec["arms"]:
            pkg=torch.load(path/f"{seed}_{arm}.pt",weights_only=True,map_location="cpu")
            assert pkg["lock"]==lock
            state=pkg["head"]
            head=Head(state["mean"],state["std"])
            del head.adapter
            head.load_state_dict(state)
            heads[seed,arm]=head.to(device).eval()
    scores=[]
    for i,r in enumerate(manifest):
        desc=model(image_tensor(images[i:i+1],device))
        for (seed,arm),head in heads.items():
            scores.append(dict(seed=seed,arm=arm,group_id=r["group_id"],split=r["split"],label=r["label"],
                               anomaly_class=r["anomaly_class"],score=float(head(desc)[1])))
    write_csv(path/"scores.csv",scores)
    model_bytes=sum(x.numel()*x.element_size() for x in model.state_dict().values())
    timings=[]
    for (seed,arm),head in heads.items():
        pkg=torch.load(path/f"{seed}_{arm}.pt",weights_only=True,map_location="cpu")
        fit=set(pkg["groups"])
        keys=[i for i in natives if manifest[i]["group_id"] in fit]
        ready=[image_tensor(images[i:i+1],device) for i in keys]
        for mode in ("device_tensor","decoded_native_image"):
            elapsed=[]
            for j in range(80):
                i=j%len(keys)
                torch.cuda.synchronize()
                start=time.perf_counter()
                x=ready[i] if mode=="device_tensor" else image_tensor(cv.resize(natives[keys[i]],(256,256),interpolation=cv.INTER_AREA)[None],device)
                head(model(x))[1].item()
                torch.cuda.synchronize()
                if j>=20: elapsed.append((time.perf_counter()-start)*1000)
            timings.append(dict(seed=seed,arm=arm,scope=mode,p50_ms=np.median(elapsed),p95_ms=np.percentile(elapsed,95),
                           tensor_bytes=model_bytes+sum(t.numel()*t.element_size() for t in head.state_dict().values()),
                           device=torch.cuda.get_device_name(),runs=len(elapsed)))
    write_csv(path/"timing.csv",timings)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifests-only",action="store_true")
    args=parser.parse_args()
    digest=file_sha(PROTO)
    assert digest==(ROOT/"e2_cross_product_protocol.sha256").read_text().split()[0]
    assert file_sha(ANNOTATIONS)=="5d16bd5f9ce8c9ca7f20009edfb011e8775fb7febadb9e8edf84c09604204985"
    spec=read_json(PROTO)
    reservation=read_json(ROOT/"heldout_category_reservation_v1.json")
    assert [r["category"] for r in reservation["selected"]]==spec["products"]
    OUT.mkdir(exist_ok=True)
    cv.setNumThreads(3)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=None
    for entry in reservation["selected"]:
        name=entry["category"]
        path=OUT/name
        path.mkdir(exist_ok=True)
        manifest,summary=manifest_for(name)
        save_json(path/"manifest.json",manifest)
        save_json(path/"metadata.json",summary)
        if args.manifests_only: continue
        archive=archive_path(name,entry["bytes"])
        if summary["eligible_train"]<18:
            save_json(path/"status.json",dict(state="insufficient_training_groups",eligible=summary["eligible_train"]))
            continue
        if archive is None:
            save_json(path/"status.json",dict(state="archive_needed",expected_bytes=entry["bytes"]))
            print(name,"archive needed",flush=True)
            continue
        lock=dict(protocol=digest,annotation=summary["annotation_sha256"],weights=file_sha(WEIGHTS),
                  archive=file_sha(archive),code=file_sha(Path(__file__)))
        status=path/"status.json"
        if status.exists() and read_json(status).get("state")=="scored" and read_json(status).get("lock")==lock: continue
        images,natives=read_images(archive,manifest,name)
        if model is None: model=encoder("ResNet18_PatchCore","cuda")[0].requires_grad_(False).eval()
        save_json(status,dict(state="training",lock=lock))
        for seed in spec["seeds"]:
            train_product(path,manifest,images,model,seed,"cuda",lock)
        score_and_time(path,manifest,images,natives,model,"cuda",spec,lock)
        save_json(status,dict(state="scored",lock=lock,heads=9,groups=len(manifest)))
        print(name,"scored",flush=True)


if __name__=="__main__":
    main()
