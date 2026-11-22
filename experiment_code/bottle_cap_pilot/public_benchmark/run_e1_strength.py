"""Run the three prespecified fixed-strength heads using the original study inputs."""
import csv
import hashlib
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from run_public_pilot import encoder, sha
from run_student_probe import Head, ROOT, SOURCE, balance_bce, read_json, variants
from run_supervision_preservation import ARMS, OUT as PRIOR, prepare_queries
from run_layer3_probe import fingerprint, frozen_prefix, final_features, cpu_state
from run_pilot import image_tensor, write_csv

OUT = ROOT / "e1_strength_v1"
PROTOCOL = ROOT / "e1_protocol.json"
SEEDS = (42, 0, 1)


def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def training_inputs(images, manifest, cohort, prior, schedule, seed):
    strong, distributed, masks, records = prepare_queries(manifest, images, cohort)
    assert [array_hash(tensor) for tensor in (strong, distributed)] == [r["query_sha256"] for r in prior]
    assert all(array_hash(masks) == r["masks_sha256"] for r in prior)
    counts = np.bincount(schedule.ravel(), minlength=len(records))
    fixed = strong.copy()
    audit = []
    for index in cohort["indices"]:
        source = images[index]
        for operation, old, mask in variants(source, index):
            q = len(audit)
            r = records[q]
            assert (r["source_index"], r["variant"]) == (index, operation)
            assert np.array_equal(old, strong[q])
            row = dict(seed=seed, query=q, group=r["group_id"], operation=operation,
                       synthetic=r["synthetic"], draws=int(counts[q]), source_sha=array_hash(source),
                       old_sha=array_hash(old), mask_sha=array_hash(mask))
            if r["synthetic"]:
                area = mask.astype(bool)
                assert np.array_equal(old[~area], source[~area])
                source32 = source.astype(np.float32)
                ideal = (2/3)*source32 + (1/3)*old.astype(np.float32)
                fixed[q] = np.clip(np.rint(ideal), 0, 255).astype(np.uint8)
                base = source[area].astype(np.float64)
                delta = old[area].astype(np.float64) - base
                for arm, img, alpha in (("S0", old, 1.), ("M", fixed[q], 1/3),
                                        ("S1", distributed[q], r["alpha"])):
                    assert np.array_equal(img[~area], source[~area])
                    change = img[area].astype(np.float64) - base
                    rounded = np.rint(base + alpha*delta)
                    err = np.abs(change-alpha*delta)
                    row.update({arm+"_alpha": alpha, arm+"_mae": np.abs(change).mean(),
                                arm+"_rmse": np.sqrt(np.square(change).mean()),
                                arm+"_max_quant_error": err.max(),
                                arm+"_clipped_channels": int(((rounded<0)|(rounded>255)).sum()),
                                arm+"_changed_pixels": int(np.any(change!=0, axis=1).sum())})
                    assert err.max() < 0.501 and row[arm+"_changed_pixels"] > 0
            else:
                assert np.array_equal(fixed[q], distributed[q])
            audit.append(row)
    # write_csv uses the first row's keys; include every audit field even on normal queries.
    fields = dict.fromkeys(k for r in audit for k in r)
    write_csv(OUT / f"seed{seed}_inputs.csv", [{**fields, **r} for r in audit])
    return fixed, masks


def run_seed(seed, images, manifest, model, device):
    cohort = next(r for r in read_json(SOURCE / "cohorts.json") if r["seed"]==seed and r["budget"]==12)
    assert len(set(cohort["group_ids"])) == 12
    prior = read_json(PRIOR / f"seed{seed}/structural_audit.json")[:2]
    assert [r["method"] for r in prior] == list(ARMS[:2])
    schedule = torch.randint(0, 96, (600, 4), generator=torch.Generator().manual_seed(seed))
    assert all(array_hash(schedule.numpy()) == r["schedule_sha256"] for r in prior)
    queries, masks = training_inputs(images, manifest, cohort, prior, schedule.numpy(), seed)
    with torch.no_grad():
        desc = torch.cat([final_features(model.body, frozen_prefix(model, image_tensor(im[None],device))) for im in queries])
    clean = desc[torch.arange(0, 96, 8, device=device)].flatten(0,1)
    mean, std = clean.mean(0), clean.std(0).clamp_min(.1)
    torch.manual_seed(seed)
    head = Head(mean, std).to(device)
    identity = fingerprint(model.state_dict()) + ":" + fingerprint(head.state_dict())
    assert all(identity == r["initial_fingerprint"] for r in prior)
    optimizer = torch.optim.AdamW([dict(params=head.parameters(), lr=.001)], weight_decay=.0001)
    labels = torch.as_tensor(masks, device=device)
    logs = []
    for step, batch in enumerate(schedule.to(device), start=1):
        patch, _, _ = head(desc[batch])
        loss = balance_bce(patch, labels[batch])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (1,100,200,300,400,500,600):
            logs.append(dict(seed=seed, step=step, loss=float(loss.detach())))
    assert fingerprint(model.state_dict()) == identity.split(":")[0]
    write_csv(OUT / f"seed{seed}_loss.csv", logs)
    torch.save(dict(head=cpu_state(head),seed=seed,groups=cohort["group_ids"],
                    initial_fingerprint=identity,protocol_sha=sha(PROTOCOL),step=600,
                    schedule_sha256=array_hash(schedule.numpy()), masks_sha256=array_hash(masks),
                    queries_sha256=array_hash(queries)), OUT / f"seed{seed}_M.pt")
    print(f"seed {seed}: input and initialization hashes passed; 600/600", flush=True)


@torch.inference_mode()
def replay(images, manifest, model, device):
    funcs = {}
    reference = {}
    for seed in SEEDS:
        for arm, name in zip(("S0","M","S1"), (ARMS[0], None, ARMS[1])):
            if name:
                package = torch.load(PRIOR / f"seed{seed}/{name}_training.pt", map_location="cpu", weights_only=True)
                assert fingerprint(package["encoder"]) == fingerprint(model.state_dict())
            else:
                package = torch.load(OUT / f"seed{seed}_M.pt", weights_only=True, map_location="cpu")
            state = package["head"]
            h = Head(state["mean"],state["std"])
            h.load_state_dict(state)
            funcs[seed,arm] = h.to(device).eval()
        saved = read_csv(PRIOR / f"seed{seed}/event_scores.csv")
        for arm, name in zip(("S0","S1"), ARMS[:2]):
            reference[seed,arm] = {r["sample_id"]:float(r["score"]) for r in saved
                                   if r["condition"]=="original" and r["method"]==name}
    result = []
    errors = []
    for i, r in enumerate(manifest):
        feat = model(image_tensor(images[i:i+1],device))
        for (seed,arm),h in funcs.items():
            val = float(h(feat)[1])
            if arm != "M":
                errors.append(abs(val-reference[seed,arm][r["sample_id"]]))
            result.append(dict(seed=seed,arm=arm,sample_id=r["sample_id"],group_id=r["group_id"],
                               split=r["split"],anomaly_class=r["anomaly_class"],label=r["label"],score=val))
    assert len(errors)==249*6 and max(errors)<1e-6
    write_csv(OUT / "original_scores.csv", result)
    save_json(OUT / "replay.json",dict(archived_scores_verified=len(errors),maximum_error=max(errors),
              saved_M_scores=len(result)//3, device=torch.cuda.get_device_name(),torch_version=str(torch.__version__)))


def main():
    cv.setNumThreads(3)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    OUT.mkdir(exist_ok=True)
    lock_files = [PROTOCOL, Path(__file__), SOURCE / "cohorts.json", SOURCE / "base_inputs.npz",
                  ROOT / "data/manifest_C1.json", ROOT / "assets/resnet18-f37072fd.pth",
                  ROOT / "run_supervision_preservation.py", ROOT / "run_student_probe.py"]
    lock_files.extend(PRIOR / f"seed{s}/{filename}" for s in SEEDS for filename in ("structural_audit.json","event_scores.csv"))
    lock = {str(p.relative_to(ROOT)):sha(p) for p in lock_files}
    lockpath = OUT / "input_lock.json"
    if lockpath.exists():
        assert read_json(lockpath)==lock, "Input or training code changed since completed seeds"
    else:
        save_json(lockpath,lock)
    with np.load(SOURCE / "base_inputs.npz") as data:
        images = data["images"]
    manifest = read_json(ROOT / "data/manifest_C1.json")
    device = torch.device("cuda")
    model,_ = encoder("ResNet18_PatchCore",device)
    model.requires_grad_(False).eval()
    for seed in SEEDS:
        if not (OUT / f"seed{seed}_M.pt").exists():
            run_seed(seed,images,manifest,model,device)
    replay(images,manifest,model,device)


if __name__=="__main__":
    main()
