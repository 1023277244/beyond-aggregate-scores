"""Bounded F0/F1/F2 diagnostic: unfreeze the actual scoring stage, layer3."""
import copy
import hashlib
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from run_public_pilot import encoder, changed_inputs, metric, sha
from run_student_probe import Head, variants, balance_bce, features, ROOT, SOURCE, read_json
from run_pilot import image_tensor, write_csv
from run_mouth_bank_sweep import select_order
from structured_pruning import bank_score

OUT = ROOT / "student_layer3_probe_v1"
ARMS = ("F0_noKD", "F1_scoreKD", "F2_featureKD")
HISTORICAL = ROOT / "student_head_probe_v1"


def fingerprint(state):
    hasher = hashlib.sha256()
    for key, value in sorted(state.items()):
        hasher.update(key.encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def frozen_prefix(model, image):
    x = (image - model.mean) / model.std
    body = model.body
    first = body.layer1(body.maxpool(body.relu(body.bn1(body.conv1(x)))))
    return body.layer2(first)


def final_features(body, second):
    third = body.layer3(second)
    second = F.avg_pool2d(second, 3, 1, 1)
    third = F.avg_pool2d(third, 3, 1, 1)
    third = F.interpolate(third, size=second.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat((second, third), dim=1).permute(0, 2, 3, 1).flatten(1, 2)


def gradients(loss, parameters):
    if not loss.requires_grad:
        return [None] * len(parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


def gradient_stats(first, second):
    a = sum(float(g.detach().square().sum()) for g in first if g is not None) ** .5
    b = sum(float(g.detach().square().sum()) for g in second if g is not None) ** .5
    dot = sum(float((g.detach() * h.detach()).sum()) for g, h in zip(first, second) if g is not None and h is not None)
    return dict(bce_gradient_norm=a, weighted_kd_gradient_norm=b,
                kd_to_bce_gradient_ratio=b / max(a, 1e-12),
                gradient_cosine=dot / max(a * b, 1e-12))


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def train_seed(seed, manifest, images, protocol, device, status):
    cohort = next(r for r in read_json(SOURCE / "cohorts.json") if r["seed"] == seed and r["budget"] == 12)
    indices = cohort["indices"]
    if len(indices) != 12 or any(manifest[i]["split"] != "normal_pool" or manifest[i]["label"] != 0 for i in indices):
        raise ValueError("Non-fitting group entered training")
    seed_out = OUT / f"seed{seed}"
    seed_out.mkdir(exist_ok=False)
    queries, masks, records = [], [], []
    for local, index in enumerate(indices):
        for name, query, mask in variants(images[index], index):
            records.append(dict(query_index=len(records), source_index=index, local_group=local,
                group_id=manifest[index]["group_id"], variant=name, synthetic=int(mask.any())))
            queries.append(query)
            masks.append(cv.resize(mask, (32, 32), interpolation=cv.INTER_AREA).astype(np.float32).reshape(-1))
    queries = np.stack(queries)
    synthetic = np.asarray([bool(r["synthetic"]) for r in records])
    query_hash = hashlib.sha256(queries.tobytes() + np.stack(masks).tobytes()).hexdigest()
    pseudo = torch.as_tensor(np.stack(masks), device=device)
    rn, _ = encoder("ResNet18_PatchCore", device)
    wrn, _ = encoder("WRN_PatchCore", device)
    rn.requires_grad_(False)
    wrn.requires_grad_(False)
    if hasattr(rn.body, "layer4") or hasattr(rn.body, "fc"):
        raise ValueError("Unexpected unused encoder stages")
    original_encoder = cpu_state(rn)
    with torch.no_grad():
        prefix = torch.cat([frozen_prefix(rn, image_tensor(query[None], device)) for query in queries])
        initial_features = torch.cat([final_features(rn.body, prefix[i:i + 1]) for i in range(96)])
        forward_diff = float((initial_features[0:1] - rn(image_tensor(queries[0:1], device))).abs().max())
        if forward_diff > 1e-6:
            raise ValueError("Cached-prefix training does not reproduce the actual encoder")
    teacher_features = features(wrn, queries, device).clone()
    clean = torch.arange(0, 96, 8, device=device)
    rn_mean = initial_features[clean].flatten(0, 1).mean(0)
    rn_std = initial_features[clean].flatten(0, 1).std(0).clamp_min(.1)
    wrn_mean = teacher_features[clean].flatten(0, 1).mean(0)
    wrn_std = teacher_features[clean].flatten(0, 1).std(0).clamp_min(.1)
    target_scores = torch.zeros(96, device=device)
    folds = []
    with torch.no_grad():
        for local, index in enumerate(indices):
            others = [j for j in range(12) if j != local]
            normal = teacher_features[clean[others]].flatten(0, 1)
            order = select_order(normal, seed, 128)
            target_scores[local * 8:(local + 1) * 8] = bank_score(teacher_features[local * 8:(local + 1) * 8], normal[order])
            folds.append(dict(excluded_group=manifest[index]["group_id"],
                included_groups=[manifest[indices[j]]["group_id"] for j in others], selected_patch_indices=order.cpu().tolist()))
    teacher_mean, teacher_std = target_scores[clean].mean(), target_scores[clean].std().clamp_min(.01)
    unbounded = (target_scores - teacher_mean) / teacher_std - 2
    score_targets = unbounded.clamp(-8, 8).detach()
    previous_score_diff = None
    if seed == 42:
        import csv
        with (HISTORICAL / "training_queries.csv").open(encoding="utf-8-sig", newline="") as stream:
            previous = list(csv.DictReader(stream))
        if any((r["group_id"], r["variant"]) != (old["group_id"], old["variant"]) for r, old in zip(records, previous)):
            raise ValueError("seed42 training queries differ from historical control")
        previous_score_diff = float(np.max(np.abs(target_scores.cpu().numpy() - np.asarray([float(r["teacher_score"]) for r in previous]))))
        if previous_score_diff > 1e-5:
            raise ValueError("seed42 teacher targets no longer match historical control")
    for i, row in enumerate(records):
        row.update(teacher_score=float(target_scores[i]), normalized_score_target=float(score_targets[i]),
                   score_target_clipped=int(abs(float(unbounded[i])) > 8))
    write_csv(seed_out / "training_queries.csv", records)
    target_stats = []
    for name in sorted({r["variant"] for r in records}):
        positions = [i for i, r in enumerate(records) if r["variant"] == name]
        vals = target_scores[positions].cpu().numpy()
        target_stats.append(dict(variant=name, count=len(positions), teacher_min=float(vals.min()), teacher_max=float(vals.max()),
            teacher_mean=float(vals.mean()), target_clipped=sum(records[i]["score_target_clipped"] for i in positions)))
    write_csv(seed_out / "teacher_variant_ranges.csv", target_stats)
    (seed_out / "teacher_folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    (seed_out / "teacher_target_audit.json").write_text(json.dumps(dict(query_sha256=query_hash,
        queries=96, teacher_synthetic_vs_normal_auc=float(roc_auc_score(synthetic, target_scores.cpu().numpy())),
        clean_teacher_mean=float(teacher_mean), clean_teacher_std=float(teacher_std),
        historical_seed42_max_score_diff=previous_score_diff, prefix_forward_max_diff=forward_diff,
        synthetic_scope="CutPaste/noise training labels, not physical Discoloration evidence"), indent=2), encoding="utf-8")
    del wrn, initial_features
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    initial_head = cpu_state(Head(rn_mean, rn_std))
    initial_fingerprint = fingerprint(original_encoder) + ":" + fingerprint(initial_head)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    schedule = torch.randint(0, 96, (600, 4), generator=generator).to(device)
    schedule_hash = hashlib.sha256(schedule.cpu().numpy().tobytes()).hexdigest()
    students, paths, logs, structural, training_fits = {}, [], [], [], []
    for arm in ARMS:
        rn, _ = encoder("ResNet18_PatchCore", device)
        rn.load_state_dict(original_encoder, strict=True)
        rn.eval().requires_grad_(False)
        rn.body.layer3.requires_grad_(True)
        head = Head(rn_mean, rn_std).to(device)
        head.load_state_dict(initial_head, strict=True)
        if fingerprint(rn.state_dict()) + ":" + fingerprint(head.state_dict()) != initial_fingerprint:
            raise ValueError("Matched arms have different initial weights")
        named_trainable = [name for name, p in rn.named_parameters() if p.requires_grad]
        if not named_trainable or any(not name.startswith("body.layer3.") for name in named_trainable):
            raise ValueError("Wrong effective stage was unfrozen")
        parameters = list(rn.body.layer3.parameters()) + list(head.shared.parameters()) + list(head.score.parameters())
        optimizer = torch.optim.AdamW([dict(params=rn.body.layer3.parameters(), lr=.0001),
                                       dict(params=head.parameters(), lr=.001)], weight_decay=.0001)
        first_layer3_gradient = None
        for step in range(600):
            batch = schedule[step]
            actual = final_features(rn.body, prefix[batch])
            patch, scalar, restored = head(actual, with_adapter=arm == "F2_featureKD")
            bce = balance_bce(patch, pseudo[batch])
            kd = torch.zeros((), device=device)
            if arm == "F1_scoreKD":
                kd = F.smooth_l1_loss(scalar, score_targets[batch])
            elif arm == "F2_featureKD":
                kd = F.mse_loss(restored, (teacher_features[batch] - wrn_mean) / wrn_std)
            loss = bce + .1 * kd
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss: hard stop")
            diagnostic = step % 100 == 0 or step == 599
            extra = gradient_stats(gradients(bce, parameters), gradients(.1 * kd, parameters)) if diagnostic else {}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0 or diagnostic:
                layer3_grad = sum(float(p.grad.detach().square().sum()) for p in rn.body.layer3.parameters() if p.grad is not None) ** .5
                if not np.isfinite(layer3_grad) or layer3_grad <= 0:
                    raise ValueError("Unfrozen scoring stage has no finite gradient")
                if step == 0:
                    first_layer3_gradient = layer3_grad
            optimizer.step()
            if diagnostic:
                entry = dict(seed=seed, method=arm, step=step + 1, bce=float(bce.detach()), kd=float(kd.detach()),
                    weighted_kd=float((.1 * kd).detach()), total_loss=float(loss.detach()),
                    weighted_kd_to_bce_loss_ratio=float((.1 * kd / bce.clamp_min(1e-12)).detach()),
                    layer3_gradient_norm=layer3_grad, **extra)
                logs.append(entry)
                status(state="training_layer3", seed=seed, arm=arm, step=step + 1, total_steps=600,
                       bce=entry["bce"], weighted_kd=entry["weighted_kd"],
                       kd_gradient_ratio=extra["kd_to_bce_gradient_ratio"])
        rn.eval()
        head.eval()
        current = cpu_state(rn)
        allowed = {"body.layer3." + name for name, _ in rn.body.layer3.named_parameters()}
        changed = [key for key in current if not torch.equal(current[key], original_encoder[key])]
        if not changed or any(key not in allowed for key in changed):
            raise ValueError("Frozen prefix or BN running buffers changed, or no layer3 update")
        maximum_change = max(float((current[key] - original_encoder[key]).abs().max()) for key in changed)
        structural.append(dict(method=arm, seed=seed, initialization_fingerprint=initial_fingerprint,
            batch_schedule_sha256=schedule_hash, training_query_sha256=query_hash,
            trainable_encoder_names=named_trainable, changed_encoder_keys=changed,
            maximum_layer3_parameter_change=maximum_change, first_layer3_gradient_norm=first_layer3_gradient,
            frozen_prefix_and_all_bn_running_buffers_unchanged=True))
        with torch.no_grad():
            fitted = torch.cat([head(final_features(rn.body, prefix[i:i + 4]))[1] for i in range(0, 96, 4)]).cpu().numpy()
        write_csv(seed_out / f"{arm}_training_predictions.csv", [dict(**r, student_score=float(fitted[i])) for i, r in enumerate(records)])
        for label, subset in (("normal_variants", ~synthetic), ("synthetic_anomalies", synthetic)):
            targets = score_targets.cpu().numpy()[subset]
            error = fitted[subset] - targets
            training_fits.append(dict(method=arm, seed=seed, subset=label,
                teacher_target_min=float(targets.min()), teacher_target_max=float(targets.max()),
                student_min=float(fitted[subset].min()), student_max=float(fitted[subset].max()),
                score_target_mae=float(np.abs(error).mean()), score_target_rmse=float(np.sqrt(np.mean(error ** 2)))))
        path = seed_out / f"{arm}_training.pt"
        torch.save(dict(encoder=current, head=cpu_state(head), seed=seed, arm=arm, steps=600,
                        training_group_ids=cohort["group_ids"]), path)
        paths.append(path)
        students[arm] = (rn, head)
        del optimizer
    write_csv(seed_out / "training_log.csv", logs)
    write_csv(seed_out / "training_target_fit.csv", training_fits)
    (seed_out / "structural_audit.json").write_text(json.dumps(structural, indent=2), encoding="utf-8")
    trained_hashes = {str(path.resolve()): sha(path) for path in paths}
    (seed_out / "training_complete_gate.json").write_text(json.dumps(dict(final_checkpoints_sha256=trained_hashes,
        test_scoring_before_all_three_checkpoints=False), indent=2), encoding="utf-8")
    del teacher_features, prefix, pseudo
    torch.cuda.empty_cache()
    return students, cohort, seed_out, trained_hashes


@torch.no_grad()
def evaluate_seed(seed, students, cohort, seed_out, manifest, images, masks, device, status):
    calibration = [i for i, r in enumerate(manifest) if r["split"] == "calibration"]
    test = np.asarray([r["split"] == "test" for r in manifest])
    labels = np.asarray([r["label"] for r in manifest], dtype=bool)[test]
    types = np.asarray([r["anomaly_class"] for r in manifest])[test]
    fits, scores, summaries, defects, curves = [], [], [], [], []
    thresholds = {}
    for arm, (rn, head) in students.items():
        values = [float(head(rn(image_tensor(images[i:i + 1], device)))[1].item()) for i in calibration]
        threshold = max(values)
        thresholds[arm] = threshold
        fits.append(dict(method=arm, seed=seed, budget=12, threshold=threshold, calibration_scores=values,
            calibration_group_ids=[manifest[i]["group_id"] for i in calibration], training_group_ids=cohort["group_ids"]))
        torch.save(dict(encoder=cpu_state(rn), head={k: v for k, v in cpu_state(head).items() if not k.startswith("adapter.")},
            threshold=threshold, arm=arm, seed=seed, protocol_sha256=sha(ROOT / "student_layer3_protocol.json"),
            inference="ResNet18 prefix through layer3 + shared128 + score; top11 average; no teacher/adapter/normal retrieval bank"),
            seed_out / f"{arm}_deployment.pt")
    for condition in read_json(ROOT / "pilot_protocol.json")["conditions"]:
        altered, _ = changed_inputs(images, masks, condition)
        matrix = np.zeros((len(images), 3))
        for j, arm in enumerate(ARMS):
            rn, head = students[arm]
            for i, image in enumerate(altered):
                matrix[i, j] = head(rn(image_tensor(image[None], device)))[1].item()
            values, threshold = matrix[:, j], thresholds[arm]
            summaries.append(dict(method=arm, seed=seed, budget=12, condition=condition, threshold=threshold,
                                  **metric(labels, values[test], threshold)))
            for i, row in enumerate(manifest):
                scores.append(dict(sample_id=row["sample_id"], group_id=row["group_id"], split=row["split"],
                    label=row["label"], anomaly_class=row["anomaly_class"], method=arm, seed=seed, budget=12,
                    condition=condition, score=float(values[i]), threshold=threshold, predicted=int(values[i] > threshold)))
            fpr, tpr, cuts = roc_curve(labels, values[test], drop_intermediate=False)
            curves.extend(dict(method=arm, seed=seed, condition=condition, fpr=float(a), tpr=float(b), threshold=float(c),
                scope="test-label diagnostic only") for a, b, c in zip(fpr, tpr, cuts))
            for defect in ("Damage", "Discoloration", "ImpactDamage", "Warpage"):
                keep = ~labels | (types == defect)
                defects.append(dict(method=arm, seed=seed, condition=condition, defect_type=defect,
                                    defect_count=int((types == defect).sum()), **metric(labels[keep], values[test][keep], threshold)))
        status(state="evaluating_final_layer3_models", seed=seed, condition=condition)
    write_csv(seed_out / "metrics.csv", summaries)
    write_csv(seed_out / "event_scores.csv", scores)
    write_csv(seed_out / "per_defect_metrics.csv", defects)
    write_csv(seed_out / "full_roc.csv", curves)
    (seed_out / "fits.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    return summaries, defects


def decide_replication(summaries, defects):
    import csv
    def csv_rows(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    old_overall = next(float(r["auroc"]) for r in csv_rows(HISTORICAL / "metrics.csv") if r["method"] == "S0_synthetic_noKD" and r["condition"] == "original")
    old_discolor = next(float(r["auroc"]) for r in csv_rows(HISTORICAL / "per_defect_metrics.csv") if r["method"] == "S0_synthetic_noKD" and r["condition"] == "original" and r["defect_type"] == "Discoloration")
    overall = {r["method"]: r["auroc"] for r in summaries if r["condition"] == "original"}
    by_type = {(r["method"], r["defect_type"]): r["auroc"] for r in defects if r["condition"] == "original"}
    capacity = overall["F0_noKD"] >= old_overall + .02 or by_type[("F0_noKD", "Discoloration")] >= old_discolor + .05
    kd = any(overall[arm] >= overall["F0_noKD"] + .01 or
        (by_type[(arm, "Discoloration")] >= by_type[("F0_noKD", "Discoloration")] + .03 and
         overall[arm] >= overall["F0_noKD"] - .01 and by_type[(arm, "ImpactDamage")] >= by_type[("F0_noKD", "ImpactDamage")] - .03)
        for arm in ARMS[1:])
    return dict(capacity_signal=capacity, kd_signal=kd, run_all_three_arms_for_seeds_0_1=capacity or kd,
                historical_h0_overall=old_overall, historical_h0_discoloration=old_discolor,
                seed42_overall=overall, gate_scope="Prospective development replication allocation, not statistical significance")


def main():
    OUT.mkdir(exist_ok=False)
    cv.setNumThreads(3)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    protocol = read_json(ROOT / "student_layer3_protocol.json")
    source_lock = read_json(SOURCE / "run_lock.json")["sha256"]
    historical_lock = read_json(HISTORICAL / "run_lock.json")["sha256"]
    deps = [Path(__file__), ROOT / "student_layer3_protocol.json", ROOT / "run_student_probe.py",
            ROOT / "heldout_category_reservation_v1.json", SOURCE / "base_inputs.npz", SOURCE / "cohorts.json",
            HISTORICAL / "metrics.csv", HISTORICAL / "per_defect_metrics.csv", HISTORICAL / "training_queries.csv"]
    lock = {str(path.resolve()): sha(path) for path in deps}
    for group in (source_lock, historical_lock):
        if any(sha(Path(path)) != value for path, value in group.items()):
            raise ValueError("Earlier frozen study changed")
    (OUT / "run_lock.json").write_text(json.dumps(dict(sha256=lock, source_sha256=source_lock,
        historical_sha256=historical_lock, scope=protocol["stage"]), indent=2), encoding="utf-8")
    start = time.monotonic()
    def status(**items):
        record = dict(**items, elapsed_seconds=time.monotonic() - start)
        (OUT / "status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
    status(state="preparing_seed42")
    data = np.load(SOURCE / "base_inputs.npz")
    images, masks = data["images"], data["masks"]
    manifest = read_json(ROOT / "data/manifest_C1.json")
    completed = []
    for seed in (42, 0, 1):
        if seed != 42 and not read_json(OUT / "replication_decision.json")["run_all_three_arms_for_seeds_0_1"]:
            break
        students, cohort, seed_out, checkpoints = train_seed(seed, manifest, images, protocol, device, status)
        summaries, defects = evaluate_seed(seed, students, cohort, seed_out, manifest, images, masks, device, status)
        if any(sha(Path(path)) != value for path, value in checkpoints.items()):
            raise ValueError("Checkpoint changed after evaluation")
        completed.append(seed)
        if seed == 42:
            decision = decide_replication(summaries, defects)
            (OUT / "replication_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
            status(state="replication_gate_decided", **decision)
        del students
        torch.cuda.empty_cache()
    for group in (lock, source_lock, historical_lock):
        if any(sha(Path(path)) != value for path, value in group.items()):
            raise ValueError("Locked study source changed during the run")
    status(state="complete", completed_seeds=completed, training_runs=3 * len(completed),
           fixed_steps_per_run=600, no_new_relation_loss=True, reserved_categories_untouched=True)


if __name__ == "__main__":
    main()
