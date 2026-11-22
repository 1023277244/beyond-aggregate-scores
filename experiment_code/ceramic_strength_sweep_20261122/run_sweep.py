"""Freeze, train and evaluate a 5-level sweep using the locked extension-A recipe."""
import argparse
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
HISTORY = ROOT / "bottle_cap_pilot/public_benchmark"
OLD = ROOT / "strength_extension_20261121_v1/A"
PRODUCT = "ceramic_wave_filter"
DATA = OLD / PRODUCT
SEEDS = (42, 0, 1)
ALPHAS = {"M": 1/3, "A50": 1/2, "A67": 2/3, "A83": 5/6, "S0": 1.0}
NEW = ("A50", "A67", "A83")
DEVICE = "cuda"

sys.path[:0] = [str(OLD), str(HISTORY)]
from common import now, read, save, sha, write_csv
from train import configure
from prepare_train import member_for, pixel_hash
from run_e3_cross_product import prepare_seed, train_seed, array_hash, features
from run_public_pilot import encoder
from run_student_probe import Head
from run_layer3_probe import cpu_state, fingerprint
from run_pilot import image_tensor

import cv2 as cv
import numpy as np
import torch


def checkpoints():
    return {
        (seed, arm): (OUT / "heads" / f"{seed}_{arm}.pt" if arm in NEW
                      else DATA / f"seed{seed}_{arm}.pt")
        for seed in SEEDS for arm in ALPHAS
    }


def freeze():
    assert not (OUT / "FREEZE.json").exists()
    original = read(OLD.parent / "FREEZE.json")
    sources = [Path(__file__), OUT / "PROTOCOL.md", OLD.parent / "FREEZE.json",
               OLD / "ALL_MODELS_FROZEN.json", OLD / "ALL_TEST_SCORES_FROZEN.json",
               DATA / "train_ready.json", DATA / "train_manifest.json",
               DATA / "train_only_inputs.npz", DATA / "normal_input_hashes.json"]
    sources += [OLD / name for name in ("common.py", "train.py", "prepare_train.py", "evaluate.py")]
    sources += [HISTORY / name for name in (
        "run_e3_cross_product.py", "run_e2_cross_product.py", "run_student_probe.py",
        "run_public_pilot.py", "run_layer3_probe.py", "analyze_e1_strength.py",
        "analyze_e2_cross_product.py", "e1_protocol.json", "assets/resnet18-f37072fd.pth")]
    sources += [HISTORY.parent / name for name in ("run_pilot.py", "structured_pruning.py")]
    sources += [path for (_, arm), path in checkpoints().items() if arm not in NEW]
    sources += [DATA / f"seed{seed}_synthetic_inputs.csv" for seed in SEEDS]
    old_endpoints = read(OLD / "ALL_MODELS_FROZEN.json")["endpoints"]
    digests = {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}
    for name, digest in digests.items():
        if name in original["files"] and digest != original["files"][name]:
            raise ValueError(f"Archived source hash mismatch: {name}")
    assert all(sha(path) == old_endpoints[path.relative_to(OLD).as_posix()]
               for (_, arm), path in checkpoints().items() if arm not in NEW)
    evaluation_files = {path.relative_to(ROOT).as_posix(): sha(path) for path in (
        DATA / "reserved_manifest.json", OLD / "raw_test_scores.csv",
        OLD / "normal_calibration_scores.csv")}
    save(OUT / "FREEZE.json", dict(created_at_utc=now(), scope="same-product follow-up",
        product=PRODUCT, camera="C1", seeds=SEEDS, alphas=ALPHAS,
        fit_groups=12, calibration_groups=6, test_groups=224, new_fits=9,
        fit_steps=600, verified_files=digests, evaluation_files=evaluation_files))
    print("Frozen protocol for all five coefficients", flush=True)


def validate_inputs():
    frozen = read(OUT / "FREEZE.json")
    for name, digest in frozen["verified_files"].items():
        assert sha(ROOT / name) == digest, name
    return frozen


def add_intermediates(prepared, images):
    # The old generator supplies the exact strong and weak endpoint images.
    # Modify only the synthetic queries, preserving every augmentation and mask.
    strong = prepared["queries"]["S0"]
    source_rows = [r for r in prepared["audit"] if r["arm"] == "S0"]
    for arm in NEW:
        alpha = ALPHAS[arm]
        queries = strong.copy()
        for record in source_rows:
            index = record["query"]
            source = images[record["source_index"]].astype(np.float64)
            delta = strong[index].astype(np.float64) - source
            blended = np.rint(source + alpha * delta)
            assert blended.min() >= 0 and blended.max() <= 255
            observed = blended - source
            error = np.abs(observed - alpha * delta).max()
            assert error <= .50000001 and np.all(observed[delta == 0] == 0)
            query = blended.astype(np.uint8)
            queries[index] = query
            prepared["audit"].append(dict(record, arm=arm, alpha=alpha,
                query_sha=array_hash(query), rounding=float(error),
                changed_pixels=int(np.any(observed != 0, axis=2).sum()),
                mae=record["mae"] * float(np.abs(observed).sum() / np.abs(delta).sum())
                    if np.any(delta) else 0.0))
        prepared["queries"][arm] = queries


def load_heads(model_paths):
    heads = {}
    for key, path in model_paths.items():
        state = torch.load(path, map_location="cpu", weights_only=True)["head"]
        head = Head(state["mean"], state["std"]).to(DEVICE)
        del head.adapter
        head.load_state_dict(state)
        heads[key] = head.eval()
    return heads


def train():
    validate_inputs()
    assert not (OUT / "EVALUATION_UNLOCK.json").exists()
    configure()
    lock = sha(OUT / "FREEZE.json")
    paths = checkpoints()
    model = encoder("ResNet18_PatchCore", DEVICE)[0].requires_grad_(False).eval()
    encoder_sha = fingerprint(model.state_dict())
    ready = read(DATA / "train_ready.json")
    manifest = read(DATA / "train_manifest.json")
    assert all(r["label"] == 0 and r["author_split"] == "train" for r in manifest)
    assert len({r["group_id"] for r in manifest}) == len(manifest)
    assert sha(DATA / "train_only_inputs.npz") == ready["cache_sha256"]
    with np.load(DATA / "train_only_inputs.npz") as data:
        images = data["images"]
    (OUT / "heads").mkdir(exist_ok=True)
    for seed in SEEDS:
        prepared = prepare_seed(images, manifest, seed, ("S0", "M"))
        clean = features(model, prepared["queries"]["S0"][::8], DEVICE).flatten(0, 1)
        torch.manual_seed(seed)
        initial = fingerprint(cpu_state(Head(clean.mean(0), clean.std(0).clamp_min(.1))))
        expected = dict(freeze=ready["freeze_sha"], data=ready["cache_sha256"],
            groups=prepared["groups"], initial=initial,
            schedule=array_hash(prepared["schedule"].numpy()),
            target_masks=array_hash(prepared["targets"]))
        for arm in ("S0", "M"):
            state = torch.load(paths[seed, arm], map_location="cpu", weights_only=True)
            assert state["audit"] == expected and state["steps"] == 600
            assert state["encoder_sha"] == encoder_sha
        archived = np.genfromtxt(DATA / f"seed{seed}_synthetic_inputs.csv", dtype=None,
                                delimiter=",", names=True, encoding="utf-8")
        keys = ("arm", "query", "group", "source_sha", "original_sha", "query_sha", "mask_sha", "draws")
        assert len(archived) == len(prepared["audit"])
        assert all(all(row[k] == old[k] for k in keys)
                   for row, old in zip(prepared["audit"], archived))
        add_intermediates(prepared, images)
        train_seed(OUT / "heads", prepared, model, seed, lock, {"arms": NEW})
    assert fingerprint(model.state_dict()) == encoder_sha
    save(OUT / "ALL_MODELS_FROZEN.json", dict(frozen_at_utc=now(), protocol_sha=lock,
        encoder_sha=encoder_sha, models={p.relative_to(ROOT).as_posix(): sha(p) for p in paths.values()}))
    indices = [i for i, r in enumerate(manifest) if r["split"] == "calibration"]
    assert len(indices) == 6
    calibration = []
    with torch.inference_mode():
        desc = features(model, images[indices], DEVICE)
        for (seed, arm), head in load_heads(paths).items():
            scores = head(desc)[1].cpu().numpy()
            calibration.extend(dict(seed=seed, arm=arm, group_id=manifest[i]["group_id"],
                score=float(value), threshold=float(scores.max())) for i, value in zip(indices, scores))
    write_csv(OUT / "calibration.csv", calibration)
    save(OUT / "EVALUATION_UNLOCK.json", dict(created_at_utc=now(), protocol_sha=lock,
        frozen_models_sha=sha(OUT / "ALL_MODELS_FROZEN.json"),
        calibration_sha=sha(OUT / "calibration.csv"), endpoints=len(paths)))
    print("All 15 endpoints and 15 normal calibration thresholds frozen", flush=True)


def evaluate():
    frozen = validate_inputs()
    unlock = read(OUT / "EVALUATION_UNLOCK.json")
    assert unlock["protocol_sha"] == sha(OUT / "FREEZE.json") and unlock["endpoints"] == 15
    assert unlock["frozen_models_sha"] == sha(OUT / "ALL_MODELS_FROZEN.json")
    assert unlock["calibration_sha"] == sha(OUT / "calibration.csv")
    endpoints = read(OUT / "ALL_MODELS_FROZEN.json")
    assert len(endpoints["models"]) == 15
    for files in (endpoints["models"], frozen["evaluation_files"]):
        assert all(sha(ROOT / name) == digest for name, digest in files.items())
    assert not (OUT / "ALL_TEST_SCORES_FROZEN.json").exists()

    manifest = [r for r in read(DATA / "reserved_manifest.json") if r["author_split"] == "test"]
    test_ids = {r["group_id"] for r in manifest}
    assert len(test_ids) == len(manifest) == 224
    assert not test_ids & {r["group_id"] for r in read(DATA / "train_manifest.json")}
    assert all(r["camera"] == "C1" and r["split"] == "test" for r in manifest)
    normal_hashes = read(DATA / "normal_input_hashes.json")
    native = {r["decoded_sha256"] for r in normal_hashes}
    resized = {r["input_sha256"] for r in normal_hashes}
    ready = read(DATA / "train_ready.json")
    archive = Path(ready["archive_path"])
    assert sha(archive) == ready["archive_sha256"]
    configure()
    model = encoder("ResNet18_PatchCore", DEVICE)[0].requires_grad_(False).eval()
    assert fingerprint(model.state_dict()) == endpoints["encoder_sha"]
    rows, pixels = [], []
    with zipfile.ZipFile(archive) as data, torch.inference_mode():
        members = set(data.namelist())
        heads = load_heads(checkpoints())
        for record in manifest:
            blob = data.read(member_for(members, PRODUCT, record["sample_id"]))
            image = cv.imdecode(np.frombuffer(blob, np.uint8), cv.IMREAD_COLOR)
            assert image is not None
            small = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)
            hashes = pixel_hash(image), pixel_hash(small)
            assert hashes[0] not in native and hashes[1] not in resized
            group = record["group_id"]
            pixels.append(dict(group_id=group, decoded_sha256=hashes[0], input_sha256=hashes[1]))
            desc = model(image_tensor(small[None], DEVICE))
            rows.extend(dict(seed=seed, arm=arm, group_id=group, score=float(head(desc)[1]))
                        for (seed, arm), head in heads.items())
    assert len(rows) == 15 * 224
    assert all(sha(ROOT / name) == digest for name, digest in endpoints["models"].items())
    write_csv(OUT / "raw_test_scores.csv", rows)
    write_csv(OUT / "test_input_hashes.csv", pixels)
    save(OUT / "ALL_TEST_SCORES_FROZEN.json", dict(completed_at_utc=now(),
        unlock_sha=sha(OUT / "EVALUATION_UNLOCK.json"), scores_sha=sha(OUT / "raw_test_scores.csv"),
        group_overlap=0, native_pixel_overlap=0, resized_pixel_overlap=0,
        test_groups=len(test_ids), score_rows=len(rows)))
    print(f"Frozen all {len(rows)} test scores", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "train", "evaluate"))
    args = parser.parse_args()
    {"freeze": freeze, "train": train, "evaluate": evaluate}[args.stage]()
