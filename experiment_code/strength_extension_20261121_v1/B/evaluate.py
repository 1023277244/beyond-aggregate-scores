"""Unified test evaluation, enabled only by the complete18-model freeze."""
import sys
sys.dont_write_bytecode=True
import zipfile

import cv2 as cv
import numpy as np
import torch

from common import ROOT, OUT, HISTORY, PRODUCTS, SEEDS, ARMS, now, sha, save, read, write_csv, verify_freeze
from prepare_train import member_for, pixel_hash
sys.path.insert(0,str(HISTORY))
from run_public_pilot import encoder
from run_student_probe import Head
from run_pilot import image_tensor


def main():
    freeze=verify_freeze()
    unlock=read(OUT/"EVALUATION_UNLOCK.json")
    assert unlock["freeze_sha"]==freeze and unlock["endpoints"]==18
    assert sha(OUT/"ALL_MODELS_FROZEN.json")==unlock["models_frozen_sha"]
    assert sha(OUT/"normal_calibration_scores.csv")==unlock["calibration_sha"]
    endpoints=read(OUT/"ALL_MODELS_FROZEN.json")
    assert len(endpoints["endpoints"])==18
    assert all(sha(OUT/f)==h for f,h in endpoints["endpoints"].items())
    assert not (OUT/"ALL_TEST_SCORES_FROZEN.json").exists(), "Do not rescore a completed final evaluation"
    tests, audits, pixels = {}, [], []
    for product in PRODUCTS:
        folder=OUT/product
        ready=read(folder/"train_ready.json")
        archive=__import__("pathlib").Path(ready["archive_path"])
        assert sha(archive)==ready["archive_sha256"]
        training=read(folder/"normal_input_hashes.json")
        old_native={r["decoded_sha256"] for r in training}
        old_small={r["input_sha256"] for r in training}
        manifest=[r for r in read(folder/"reserved_manifest.json") if r["author_split"]=="test"]
        train_groups={r["group_id"] for r in read(folder/"train_manifest.json")}
        assert not train_groups & {r["group_id"] for r in manifest}
        images=[]
        with zipfile.ZipFile(archive) as z:
            names=set(z.namelist())
            for r in manifest:
                blob=z.read(member_for(names,product,r["sample_id"]))
                image=cv.imdecode(np.frombuffer(blob,np.uint8),cv.IMREAD_COLOR)
                assert image is not None
                resized=cv.resize(image,(256,256),interpolation=cv.INTER_AREA)
                a,b=pixel_hash(image),pixel_hash(resized)
                assert a not in old_native and b not in old_small, "Cross-split exact pixel duplicate"
                pixels.append(dict(product=product,group_id=r["group_id"],decoded_sha256=a,input_sha256=b))
                images.append(resized)
        tests[product]=(manifest,np.stack(images))
        audits.append(dict(product=product,test_images=len(images),normal_train=len(training),
            group_overlap=0,decoded_pixel_overlap=0,resized_pixel_overlap=0,
            checked_after_all_models_frozen=True))
    save(OUT/"test_input_audit.json",dict(created_at_utc=now(),products=audits))
    write_csv(OUT/"test_input_hashes.csv",pixels)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=encoder("ResNet18_PatchCore","cuda")[0].requires_grad_(False).eval()
    rows=[]
    # No label, type or score is used for selecting heads or deciding to continue.
    with torch.inference_mode():
        for product in PRODUCTS:
            manifest,images=tests[product]
            heads={}
            for seed in SEEDS:
                for arm in ARMS:
                    state=torch.load(OUT/product/f"seed{seed}_{arm}.pt",weights_only=True,map_location="cpu")["head"]
                    head=Head(state["mean"],state["std"]).cuda()
                    del head.adapter
                    head.load_state_dict(state)
                    heads[seed,arm]=head.eval()
            for record,image in zip(manifest,images):
                f=model(image_tensor(image[None],"cuda"))
                for (seed,arm),head in heads.items():
                    rows.append(dict(product=product,seed=seed,arm=arm,group_id=record["group_id"],
                        score=float(head(f)[1])))
            print("Completed reserved test scoring",product,len(images),flush=True)
    write_csv(OUT/"raw_test_scores.csv",rows)
    assert all(sha(OUT/f)==h for f,h in endpoints["endpoints"].items())
    save(OUT/"ALL_TEST_SCORES_FROZEN.json",dict(completed_at_utc=now(),freeze_sha=freeze,
        unlock_sha=sha(OUT/"EVALUATION_UNLOCK.json"),scores_sha=sha(OUT/"raw_test_scores.csv"),
        all_products_completed=True,model_files_unchanged=True))
    print("All test scores frozen. Statistics may now be computed.",flush=True)


if __name__=="__main__":
    main()

