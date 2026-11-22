"""Three matched head-only students, with no development-label selection.

This is a feasibility diagnostic, not a novel method or full encoder distillation.
"""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from run_public_pilot import encoder, changed_inputs, metric, sha
from run_pilot import image_tensor, write_csv
from run_mouth_bank_sweep import select_order
from structured_pruning import bank_score

SOURCE = ROOT / "pilot_smd_C1"
OUT = ROOT / "student_head_probe_v1"
ARMS = ("S0_synthetic_noKD", "S1_scoreKD", "S2_featureKD")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Head(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.shared = nn.Linear(384, 128)
        self.score = nn.Linear(128, 1)
        self.adapter = nn.Linear(128, 1536)

    def forward(self, features, with_adapter=False):
        hidden = F.relu(self.shared((features - self.mean) / self.std))
        logits = self.score(hidden).squeeze(-1)
        image_score = logits.topk(11, dim=1).values.mean(1)
        return logits, image_score, self.adapter(hidden) if with_adapter else None


def balance_bce(logits, mask):
    losses = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    pos, neg = mask > .5, mask <= .5
    if pos.any() and neg.any():
        return .5 * losses[pos].mean() + .5 * losses[neg].mean()
    return losses.mean()


def variants(image, source_index):
    rng = np.random.default_rng(420000 + source_index)
    outputs = [("original", image.copy(), np.zeros((256, 256), np.uint8))]
    for factor in (.9, 1.1):
        outputs.append((f"brightness{factor}", np.clip(np.rint(image.astype(np.float32) * factor), 0, 255).astype(np.uint8), np.zeros((256, 256), np.uint8)))
    for offset in (-6, 6):
        matrix = np.asarray([[1., 0., offset], [0., 1., 0.]], np.float32)
        outputs.append((f"shift_x{offset}", cv.warpAffine(image, matrix, (256, 256), borderMode=cv.BORDER_REFLECT_101), np.zeros((256, 256), np.uint8)))
    for variant in range(3):
        width, height = rng.integers(48, 81, size=2).tolist()
        x, y = rng.integers(0, 257 - width), rng.integers(0, 257 - height)
        altered, mask = image.copy(), np.zeros((256, 256), np.uint8)
        if variant < 2:
            sx, sy = rng.integers(0, 257 - width), rng.integers(0, 257 - height)
            altered[y:y + height, x:x + width] = image[sy:sy + height, sx:sx + width]
            name = f"cutpaste{variant}"
        else:
            altered[y:y + height, x:x + width] = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            name = "local_noise"
        mask[y:y + height, x:x + width] = 1
        outputs.append((name, altered, mask))
    return outputs


@torch.inference_mode()
def features(model, images, device):
    return torch.cat([model(image_tensor(image[None], device)) for image in images])


def main():
    OUT.mkdir(exist_ok=True)
    for name in ("heads", "deployment"):
        (OUT / name).mkdir(exist_ok=True)
    if (OUT / "status.json").exists():
        raise FileExistsError("Do not overwrite a completed or interrupted probe; use a new version")
    cv.setNumThreads(3)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This predeclared probe expects the available CUDA device")
    manifest = read_json(ROOT / "data/manifest_C1.json")
    protocol = read_json(ROOT / "student_probe_protocol.json")
    cohort = next(r for r in read_json(SOURCE / "cohorts.json") if r["seed"] == 42 and r["budget"] == 12)
    train = cohort["indices"]
    calibration = [i for i, r in enumerate(manifest) if r["split"] == "calibration"]
    test = [i for i, r in enumerate(manifest) if r["split"] == "test"]
    if len(train) != 12 or len(calibration) != 6 or set(train) & set(calibration + test):
        raise ValueError("Training/calibration isolation failed")
    locked = read_json(SOURCE / "run_lock.json")["sha256"]
    if any(sha(Path(path)) != expected for path, expected in locked.items()):
        raise ValueError("Frozen source changed")
    dependencies = [Path(__file__), ROOT / "student_probe_protocol.json", ROOT / "run_public_pilot.py",
                    SOURCE / "cohorts.json", ROOT / "data/manifest_C1.json", SOURCE / "base_inputs.npz"]
    lock = {str(path.resolve()): sha(path) for path in dependencies}
    (OUT / "run_lock.json").write_text(json.dumps(dict(sha256=lock, source_run_sha256=locked, scope=protocol["stage"]), indent=2), encoding="utf-8")
    start = time.monotonic()

    def status(**items):
        record = dict(**items, elapsed_seconds=time.monotonic() - start)
        (OUT / "status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)

    status(state="preparing_training_only_inputs")
    data = np.load(SOURCE / "base_inputs.npz")
    images, masks = data["images"], data["masks"]
    # No image or mask from calibration/test enters these query tensors.
    query_images, query_masks, records = [], [], []
    for local, index in enumerate(train):
        if manifest[index]["label"] != 0 or manifest[index]["split"] != "normal_pool":
            raise ValueError("Non-normal training query")
        for name, altered, mask in variants(images[index], index):
            records.append(dict(query_index=len(records), source_index=index, local_group=local,
                                group_id=manifest[index]["group_id"], variant=name, synthetic=int(mask.any())))
            query_images.append(altered)
            query_masks.append(cv.resize(mask, (32, 32), interpolation=cv.INTER_AREA).astype(np.float32).reshape(-1))
    query_images = np.stack(query_images)
    pseudo_masks = torch.as_tensor(np.stack(query_masks), device=device)
    rn, _ = encoder("ResNet18_PatchCore", device)
    wrn, _ = encoder("WRN_PatchCore", device)
    rn.requires_grad_(False)
    wrn.requires_grad_(False)
    train_rn, train_wrn = features(rn, query_images, device), features(wrn, query_images, device)
    # Inference-mode tensors must become ordinary tensors for autograd inputs.
    train_rn, train_wrn = train_rn.clone(), train_wrn.clone()
    clean_indices = torch.arange(0, 96, 8, device=device)
    clean_rn, clean_wrn = train_rn[clean_indices], train_wrn[clean_indices]
    rn_mean = clean_rn.flatten(0, 1).mean(0)
    rn_std = clean_rn.flatten(0, 1).std(0).clamp_min(.1)
    wrn_mean = clean_wrn.flatten(0, 1).mean(0)
    wrn_std = clean_wrn.flatten(0, 1).std(0).clamp_min(.1)
    teacher_scores = torch.zeros(96, device=device)
    fold_rows = []
    with torch.no_grad():
        for local, index in enumerate(train):
            others = [j for j in range(12) if j != local]
            normal = clean_wrn[others].flatten(0, 1)
            selected = select_order(normal, 42, 128)
            bank = normal[selected]
            teacher_scores[local * 8:(local + 1) * 8] = bank_score(train_wrn[local * 8:(local + 1) * 8], bank)
            fold_rows.append(dict(excluded_group=manifest[index]["group_id"],
                included_groups=[manifest[train[j]]["group_id"] for j in others],
                selected_patch_indices=selected.cpu().tolist(), bank_entries=len(bank)))
            status(state="cross_fitting_teacher", completed_groups=local + 1, total=12)
    clean_scores = teacher_scores[clean_indices]
    target_mean, target_std = clean_scores.mean(), clean_scores.std().clamp_min(.01)
    score_targets = ((teacher_scores - target_mean) / target_std - 2).clamp(-8, 8).detach()
    synthetic = np.asarray([bool(r["synthetic"]) for r in records])
    teacher_values = teacher_scores.cpu().numpy()
    target_audit = dict(queries=len(records), normal_queries=int((~synthetic).sum()), synthetic_queries=int(synthetic.sum()),
        cross_fitted_groups=12, no_training_group_self_matches=True, clean_teacher_mean=float(target_mean),
        clean_teacher_std=float(target_std), synthetic_vs_normal_teacher_auc=float(roc_auc_score(synthetic, teacher_values)),
        normalized_target_min=float(score_targets.min()), normalized_target_max=float(score_targets.max()),
        caveat="Synthetic labels only; this AUC does not measure physical anomaly detection")
    (OUT / "teacher_target_audit.json").write_text(json.dumps(target_audit, indent=2), encoding="utf-8")
    (OUT / "teacher_folds.json").write_text(json.dumps(fold_rows, indent=2), encoding="utf-8")
    for i, row in enumerate(records):
        row.update(teacher_score=float(teacher_scores[i]), normalized_score_target=float(score_targets[i]))
    write_csv(OUT / "training_queries.csv", records)
    del wrn
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    initial = Head(rn_mean, rn_std).to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    students, logs, trained_paths = {}, [], []
    for arm in ARMS:
        head = Head(rn_mean, rn_std).to(device)
        head.load_state_dict(initial_state)
        head.train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        generator = torch.Generator(device="cpu").manual_seed(42)
        for step in range(600):
            indices = torch.randint(0, 96, (4,), generator=generator).to(device)
            logits, scalar, restored = head(train_rn[indices], with_adapter=arm == "S2_featureKD")
            bce = balance_bce(logits, pseudo_masks[indices])
            kd = torch.zeros((), device=device)
            if arm == "S1_scoreKD":
                kd = F.smooth_l1_loss(scalar, score_targets[indices])
            elif arm == "S2_featureKD":
                kd = F.mse_loss(restored, (train_wrn[indices] - wrn_mean) / wrn_std)
            loss = bce + .1 * kd
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == 599:
                logs.append(dict(arm=arm, step=step + 1, total_loss=float(loss), bce=float(bce), kd=float(kd)))
                status(state="training_head", arm=arm, step=step + 1, total_steps=600, total_loss=float(loss))
        head.eval()
        students[arm] = head
        path = OUT / "heads" / f"{arm}.pt"
        torch.save(dict(state_dict=head.cpu().state_dict(), protocol_sha256=lock[str((ROOT / "student_probe_protocol.json").resolve())],
                        training_group_ids=cohort["group_ids"], steps=600, arm=arm), path)
        head.to(device)
        trained_paths.append(path)
        deployed_head = {k: v.cpu() for k, v in head.state_dict().items() if not k.startswith("adapter.")}
        torch.save(dict(encoder=rn.cpu().state_dict(), head=deployed_head, arm=arm,
                        inference="RN prefix + shared128 + score; top11 average; no adapter, teacher or bank"),
                   OUT / "deployment" / f"{arm}.pt")
        rn.to(device)
    write_csv(OUT / "training_log.csv", logs)
    # Evaluation begins only after all final heads exist; none are selected by evaluation.
    if len(trained_paths) != 3 or not all(path.exists() for path in trained_paths):
        raise ValueError("Incomplete training gate")
    final_head_hashes = {str(path.resolve()): sha(path) for path in trained_paths}
    (OUT / "training_complete_gate.json").write_text(json.dumps(dict(final_head_sha256=final_head_hashes,
        evaluated_test_before_all_final_heads=False, encoder_frozen=True), indent=2), encoding="utf-8")
    del train_rn, train_wrn, clean_rn, clean_wrn, initial, optimizer
    torch.cuda.empty_cache()
    all_scores, all_metrics, per_defect, fits = [], [], [], []
    thresholds = {}
    with torch.inference_mode():
        for arm, head in students.items():
            cal_values = [float(head(rn(image_tensor(images[i:i + 1], device)))[1].item()) for i in calibration]
            thresholds[arm] = max(cal_values)
            fits.append(dict(method=arm, seed=42, budget=12, threshold=thresholds[arm],
                             training_group_ids=cohort["group_ids"], calibration_scores=cal_values,
                             calibration_group_ids=[manifest[i]["group_id"] for i in calibration]))
        y = np.asarray([manifest[i]["label"] for i in test], dtype=bool)
        types = np.asarray([manifest[i]["anomaly_class"] for i in test])
        conditions = read_json(ROOT / "pilot_protocol.json")["conditions"]
        for condition in conditions:
            changed, _ = changed_inputs(images, masks, condition)
            matrix = np.zeros((len(images), 3))
            for index, image in enumerate(changed):
                extracted = rn(image_tensor(image[None], device))
                for j, arm in enumerate(ARMS):
                    matrix[index, j] = students[arm](extracted)[1].item()
            for j, arm in enumerate(ARMS):
                values, threshold = matrix[:, j], thresholds[arm]
                all_metrics.append(dict(method=arm, condition=condition, budget=12, seed=42,
                                        threshold=threshold, **metric(y, values[test], threshold)))
                for index, row in enumerate(manifest):
                    all_scores.append(dict(sample_id=row["sample_id"], group_id=row["group_id"], split=row["split"],
                        label=row["label"], anomaly_class=row["anomaly_class"], method=arm, condition=condition,
                        budget=12, seed=42, score=float(values[index]), threshold=threshold, predicted=int(values[index] > threshold)))
                for defect in sorted(set(types) - {"OK"}):
                    keep = ~y | (types == defect)
                    per_defect.append(dict(method=arm, condition=condition, defect_type=defect,
                        defect_count=int((types == defect).sum()), **metric(y[keep], values[test][keep], threshold)))
            status(state="evaluating_final_heads", condition=condition, original_if_available=all_metrics[:3])
    write_csv(OUT / "event_scores.csv", all_scores)
    write_csv(OUT / "metrics.csv", all_metrics)
    write_csv(OUT / "per_defect_metrics.csv", per_defect)
    (OUT / "fits.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    if any(sha(Path(path)) != expected for path, expected in lock.items()):
        raise ValueError("Probe dependencies changed")
    if any(sha(Path(path)) != expected for path, expected in final_head_hashes.items()):
        raise ValueError("Final trained head changed after test evaluation")
    verification = dict(metrics_rows=len(all_metrics), score_rows=len(all_scores), arms=list(ARMS),
        same_12_normal_training_groups=True, same_96_training_queries=True, calibration_groups_excluded_from_training=True,
        synthetic_supervision_shared=True, real_anomalies_used_for_training=False, frozen_encoder=True,
        fixed_600_steps=True, final_heads_saved_before_test_scoring=True, no_test_checkpoint_selection=True,
        new_relation_or_reliability_loss=False, scope=protocol["stage"])
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    status(state="complete", **verification)


if __name__ == "__main__":
    main()
