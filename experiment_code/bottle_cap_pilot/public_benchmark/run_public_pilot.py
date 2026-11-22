"""Fixed normal-budget baselines on the first Real-IAD Variety category."""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torchvision.models import resnet18, wide_resnet50_2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from run_mouth_bank_sweep import select_order
from run_pilot import image_tensor, write_csv
from structured_pruning import PatchFeatures, bank_score

OUT = ROOT / "pilot_smd_C1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_image(path):
    image = cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(path)
    return image


def descriptor(image, kind):
    if kind == "HSV_hist":
        base = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    else:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        padded = np.pad(gray, 1, mode="reflect")
        base = np.zeros_like(gray)
        for bit, (dy, dx) in enumerate(((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))):
            base |= ((padded[1 + dy:257 + dy, 1 + dx:257 + dx] >= gray).astype(np.uint8) << bit)
    cells = []
    for y in range(0, 256, 64):
        for x in range(0, 256, 64):
            cell = base[y:y + 64, x:x + 64]
            histogram = cv.calcHist([cell], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel() if kind == "HSV_hist" else np.bincount(cell.ravel(), minlength=256).astype(np.float32)
            cells.append(np.sqrt(histogram / max(float(histogram.sum()), 1)))
    return np.concatenate(cells).astype(np.float32) / 4


def changed_inputs(images, masks, condition):
    if condition == "original":
        return images, masks
    if condition.startswith("brightness_"):
        factor = float(condition.split("_")[1])
        return np.clip(np.rint(images.astype(np.float32) * factor), 0, 255).astype(np.uint8), masks
    if condition.startswith("blur_sigma"):
        sigma = float(condition.removeprefix("blur_sigma"))
        return np.stack([cv.GaussianBlur(image, (0, 0), sigma) for image in images]), masks
    _, axis, value = condition.split("_")
    offset = int(value.replace("minus", "-").replace("plus", ""))
    matrix = np.asarray([[1., 0., offset if axis == "x" else 0], [0., 1., offset if axis == "y" else 0]], dtype=np.float32)
    kwargs = {"dsize": (256, 256), "flags": cv.INTER_NEAREST, "borderMode": cv.BORDER_REFLECT_101}
    return np.stack([cv.warpAffine(image, matrix, **kwargs) for image in images]), np.stack([cv.warpAffine(mask, matrix, **kwargs) for mask in masks])


def metric(labels, values, threshold):
    predicted = values > threshold
    fpr, tpr, _ = roc_curve(labels, values)
    return {"tp": int((labels & predicted).sum()), "fn": int((labels & ~predicted).sum()),
            "fp": int((~labels & predicted).sum()), "tn": int((~labels & ~predicted).sum()),
            "auroc": float(roc_auc_score(labels, values)), "ap": float(average_precision_score(labels, values)),
            "minimum_defect_margin": float(values[labels].min() - threshold),
            "normal_max": float(values[~labels].max()), "defect_min": float(values[labels].min()),
            "oracle_roc_tpr_at_fpr_le_005": float(tpr[fpr <= .05 + 1e-12].max())}


def encoder(name, device):
    if name == "WRN_PatchCore":
        body, path = wide_resnet50_2(weights=None), ROOT / "assets/wide_resnet50_2-95faca4d.pth"
    else:
        body, path = resnet18(weights=None), ROOT / "assets/resnet18-f37072fd.pth"
    state = torch.load(path, weights_only=True, map_location="cpu")
    source_dtypes = dict(__import__("collections").Counter(str(v.dtype) for v in state.values()))
    body.load_state_dict(state, strict=True)
    del state, body.layer4, body.fc
    model = PatchFeatures(body).to(device).eval()
    return model, source_dtypes


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "banks").mkdir(exist_ok=True)
    cv.setNumThreads(3)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = read_json(ROOT / "pilot_protocol.json")
    rows = read_json(ROOT / "data/manifest_C1.json")
    calibration = [i for i, row in enumerate(rows) if row["split"] == "calibration"]
    pool = np.asarray([i for i, row in enumerate(rows) if row["split"] == "normal_pool"])
    all_train = [i for i, row in enumerate(rows) if row["author_split"] == "train"]
    train_position = {index: pos for pos, index in enumerate(all_train)}
    test = np.asarray([row["split"] == "test" for row in rows])
    y = np.asarray([row["label"] for row in rows], dtype=bool)[test]
    if len(calibration) != 6 or len(pool) != 19 or y.sum() != 199 or (~y).sum() != 25:
        raise ValueError("Unexpected predeclared split")
    dependencies = [Path(__file__), ROOT / "pilot_protocol.json", ROOT / "data/manifest_C1.json", ROOT / "data/audit.json",
                    ROOT / "sources/archive_verification.json", ROOT / "assets/manifest.json", ROOT.parent / "run_pilot.py",
                    ROOT.parent / "structured_pruning.py", ROOT.parent / "run_mouth_bank_sweep.py"]
    dependencies.extend(ROOT / "assets" / name for name in ("resnet18-f37072fd.pth", "wide_resnet50_2-95faca4d.pth"))
    lock = {str(path.resolve()): sha(path) for path in dependencies}
    (OUT / "run_lock.json").write_text(json.dumps({"sha256": lock, "device": str(device), "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU", "scope": protocol["stage"]}, indent=2), encoding="utf-8")
    images, masks = [], []
    for row in rows:
        path = ROOT / row["image"]
        if sha(path) != row["sha256"]:
            raise ValueError("Source image changed")
        image = load_image(path)
        images.append(cv.resize(image, (256, 256), interpolation=cv.INTER_AREA))
        if row["mask"]:
            mask_path = ROOT / row["mask"]
            if sha(mask_path) != row["mask_sha256"]:
                raise ValueError("Source mask changed")
            raw = cv.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
            mask = (cv.resize((raw > 0).astype(np.float32), (256, 256), interpolation=cv.INTER_AREA) > 0).astype(np.uint8)
        else:
            mask = np.zeros((256, 256), dtype=np.uint8)
        masks.append(mask)
    images, masks = np.stack(images), np.stack(masks)
    np.savez_compressed(OUT / "base_inputs.npz", images=images, masks=masks)
    cohorts = []
    for seed in protocol["seeds"]:
        order = np.random.default_rng(seed).permutation(pool)
        for budget in protocol["normal_training_budgets"]:
            indices = order[:budget].tolist()
            if set(indices) & set(calibration):
                raise ValueError("Calibration leakage")
            cohorts.append({"seed": seed, "budget": budget, "indices": indices,
                            "group_ids": [rows[i]["group_id"] for i in indices]})
    (OUT / "cohorts.json").write_text(json.dumps(cohorts, indent=2), encoding="utf-8")
    all_metrics, all_scores, per_defect, fits, visibility, source_checks = [], [], [], [], [], []
    methods = ("HSV_hist", "LBP_hist", "ResNet18_PatchCore", "WRN_PatchCore")
    started = time.monotonic()

    def status(**items):
        (OUT / "status.json").write_text(json.dumps({**items, "elapsed_seconds": time.monotonic() - started}, indent=2), encoding="utf-8")

    for name in methods:
        status(state="fitting", method=name)
        if name.endswith("PatchCore"):
            model, dtypes = encoder(name, device)
            source_checks.append({"method": name, "source_state_tensor_dtypes": dtypes,
                                  "inference_parameter_dtype": str(next(model.parameters()).dtype),
                                  "parameters": sum(p.numel() for p in model.parameters())})
            training_features = torch.cat([model(image_tensor(images[i:i + 1], device)).cpu() for i in all_train])
        else:
            training_features = np.stack([descriptor(images[i], name) for i in all_train])
        configs = []
        for cohort in cohorts:
            positions = [train_position[i] for i in cohort["indices"]]
            config = {"seed": cohort["seed"], "budget": cohort["budget"], "group_ids": cohort["group_ids"]}
            if name.endswith("PatchCore"):
                normal = training_features[positions].flatten(0, 1).to(device)
                order = select_order(normal, cohort["seed"], 128)
                bank = normal[order]
                package = {"bank": bank.cpu(), "selected_patch_indices": order.cpu(), "training_group_ids": cohort["group_ids"]}
                torch.save(package, OUT / "banks" / f"{name}_n{cohort['budget']}_seed{cohort['seed']}.pt")
                cal_values = [float(bank_score(training_features[train_position[i]:train_position[i] + 1].to(device), bank).item()) for i in calibration]
                config["bank"] = bank
                del normal
            else:
                bank = training_features[positions]
                cal_values = [float(np.linalg.norm(bank - training_features[train_position[i]], axis=1).min()) for i in calibration]
                config["bank"] = bank
            config["threshold"] = max(cal_values)
            configs.append(config)
            fits.append({"method": name, "seed": cohort["seed"], "budget": cohort["budget"],
                         "training_group_ids": cohort["group_ids"], "calibration_group_ids": [rows[i]["group_id"] for i in calibration],
                         "calibration_scores": cal_values, "threshold": config["threshold"]})
        for condition in protocol["conditions"]:
            changed, transformed_masks = changed_inputs(images, masks, condition)
            retained = transformed_masks.reshape(len(rows), -1).sum(1)
            if name == methods[0]:
                visibility.extend({"sample_id": row["sample_id"], "label": row["label"], "split": row["split"],
                                   "condition": condition, "base_mask_pixels": int(masks[i].sum()), "retained_mask_pixels": int(retained[i])}
                                  for i, row in enumerate(rows))
            score_matrix = np.zeros((len(rows), len(configs)), dtype=np.float64)
            for i, image in enumerate(changed):
                query = model(image_tensor(image[None], device)) if name.endswith("PatchCore") else descriptor(image, name)
                for j, config in enumerate(configs):
                    score_matrix[i, j] = float(bank_score(query, config["bank"]).item()) if name.endswith("PatchCore") else float(np.linalg.norm(config["bank"] - query, axis=1).min())
                if i % 40 == 0:
                    status(state="scoring", method=name, condition=condition, processed=i + 1, total=len(rows))
            for j, config in enumerate(configs):
                values, threshold = score_matrix[:, j], config["threshold"]
                if condition == "original" and abs(float(values[calibration].max()) - threshold) > 1e-5:
                    raise ValueError("Original calibration scores do not replay")
                v = values[test]
                summary = metric(y, v, threshold)
                keep = ~y | (retained[test] > 0)
                secondary = metric(y[keep], v[keep], threshold) if y[keep].any() else None
                record = {"method": name, "condition": condition, "budget": config["budget"], "seed": config["seed"],
                          "threshold": threshold, **summary, "normal_count": 25, "defect_count": 199,
                          "anomalies_without_retained_mask": int((y & (retained[test] == 0)).sum()),
                          "mask_retained_auroc": secondary["auroc"] if secondary else None,
                          "mask_retained_recall": secondary["tp"] / int(y[keep].sum()) if secondary else None}
                all_metrics.append(record)
                for i, row in enumerate(rows):
                    all_scores.append({"sample_id": row["sample_id"], "group_id": row["group_id"], "split": row["split"],
                                       "label": row["label"], "anomaly_class": row["anomaly_class"], "method": name,
                                       "condition": condition, "budget": config["budget"], "seed": config["seed"],
                                       "score": float(values[i]), "threshold": threshold, "predicted": int(values[i] > threshold)})
                test_types = np.asarray([row["anomaly_class"] for row in rows])[test]
                for defect in sorted(set(test_types) - {"OK"}):
                    subset = (test_types == "OK") | (test_types == defect)
                    per_defect.append({"method": name, "condition": condition, "budget": config["budget"], "seed": config["seed"],
                                       "defect_type": defect, "defect_count": int((test_types == defect).sum()),
                                       **metric(y[subset], v[subset], threshold)})
            write_csv(OUT / "metrics.csv", all_metrics)
            write_csv(OUT / "event_scores.csv", all_scores)
            write_csv(OUT / "per_defect_metrics.csv", per_defect)
            print(json.dumps({"method": name, "condition": condition, "last_seed42_budget12": all_metrics[-1]}), flush=True)
        (OUT / "fits.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
        if name.endswith("PatchCore"):
            del model, training_features, bank, configs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(OUT / "mask_retention.csv", visibility)
    (OUT / "encoder_sources.json").write_text(json.dumps(source_checks, indent=2), encoding="utf-8")
    if len(all_metrics) != 324 or len(all_scores) != 324 * len(rows):
        raise ValueError("Incomplete predeclared grid")
    if any(sha(Path(path)) != expected for path, expected in lock.items()):
        raise ValueError("Locked input changed during scoring")
    verification = {"metrics_rows": len(all_metrics), "score_rows": len(all_scores), "methods": 4, "conditions": 9,
                    "seeds": [0, 1, 42], "budgets": [3, 6, 12], "calibration_events": 6, "test_groups": 224,
                    "hashes_unchanged": True, "private_banks_reused": False, "new_method_trained": False,
                    "scope": "One-category C1 development pilot, no full benchmark/generalization claim"}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    status(state="complete", **verification)
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
