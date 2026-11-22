"""Match real masks to frozen pooling locations and measure paired score changes."""
import csv
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SWEEP = ROOT / "ceramic_strength_sweep_20261122"
sys.path.insert(0, str(SWEEP))

import numpy as np
from run_sweep import (DATA, DEVICE, PRODUCT, configure, checkpoints, cv, encoder,
                       features, fingerprint, image_tensor, load_heads, member_for,
                       now, read, save, sha, torch, validate_inputs, write_csv)

GRID, TOP, CONTROL_COUNT, DRAWS = 32, 11, 16, 2000
STRATA = ("Scratch", "MissingPart", "CharacterOmission", "Deformation")
CONDITIONS = ("full", "annotation_excluded", "control_excluded")


def verify_and_freeze():
    if (OUT / "ANALYSIS_FREEZE.json").exists():
        raise RuntimeError("This one-shot diagnostic is already frozen.")
    sweep_freeze = validate_inputs()
    frozen = read(SWEEP / "ALL_MODELS_FROZEN.json")
    scored = read(SWEEP / "ALL_TEST_SCORES_FROZEN.json")
    assert frozen["protocol_sha"] == sha(SWEEP / "FREEZE.json")
    assert scored["unlock_sha"] == sha(SWEEP / "EVALUATION_UNLOCK.json")
    assert scored["scores_sha"] == sha(SWEEP / "raw_test_scores.csv")
    unlock = read(SWEEP / "EVALUATION_UNLOCK.json")
    assert unlock["frozen_models_sha"] == sha(SWEEP / "ALL_MODELS_FROZEN.json")
    assert unlock["calibration_sha"] == sha(SWEEP / "calibration.csv")
    assert all(sha(ROOT / name) == digest for name, digest in frozen["models"].items())
    assert all(sha(ROOT / name) == digest for name, digest in sweep_freeze["evaluation_files"].items())
    files = [Path(__file__), OUT / "PROTOCOL.md", SWEEP / "FREEZE.json",
             SWEEP / "ALL_MODELS_FROZEN.json", SWEEP / "ALL_TEST_SCORES_FROZEN.json",
             SWEEP / "EVALUATION_UNLOCK.json"]
    save(OUT / "ANALYSIS_FREEZE.json", dict(frozen_at_utc=now(), analysis_of_frozen_models=True,
        fits=15, test_parts=224, top_cells=TOP, target_controls=CONTROL_COUNT, bootstrap_draws=DRAWS,
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    return frozen["encoder_sha"]


def read_scores(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {(int(r["seed"]), r["arm"], r["group_id"]): float(r["score"])
                for r in csv.DictReader(stream)}


def collect_maps(encoder_sha, model_paths):
    configure()
    model = encoder("ResNet18_PatchCore", DEVICE)[0].requires_grad_(False).eval()
    assert fingerprint(model.state_dict()) == encoder_sha
    heads = load_heads(model_paths)
    test_reference = read_scores(SWEEP / "raw_test_scores.csv")
    cal_reference = read_scores(SWEEP / "calibration.csv")
    manifest = [r for r in read(DATA / "reserved_manifest.json") if r["author_split"] == "test"]
    train_manifest = read(DATA / "train_manifest.json")
    assert len({r["group_id"] for r in manifest}) == len(manifest) == 224
    assert all(r["camera"] == "C1" for r in manifest)
    cal_indices = [i for i, row in enumerate(train_manifest) if row["split"] == "calibration"]
    assert len(cal_indices) == 6 and all(train_manifest[i]["label"] == 0 for i in cal_indices)
    with np.load(DATA / "train_only_inputs.npz") as cache:
        cal_images = cache["images"][cal_indices]
    ready = read(DATA / "train_ready.json")
    archive_path = Path(ready["archive_path"])
    assert sha(archive_path) == ready["archive_sha256"]

    maps, masks, score_rows = [], {}, []
    max_test_error, max_cal_error = 0., 0.
    calibration_top = np.zeros(GRID * GRID, dtype=float)
    with zipfile.ZipFile(archive_path) as archive, torch.inference_mode():
        members = set(archive.namelist())
        descriptors = features(model, cal_images, DEVICE)
        for key, head in heads.items():
            logits, predicted, _ = head(descriptors)
            expected = [cal_reference[(*key, train_manifest[i]["group_id"])] for i in cal_indices]
            max_cal_error = max(max_cal_error, float(np.max(np.abs(predicted.cpu().numpy() - expected))))
            # Top-k density on validation normals is fixed for every region and test part.
            top_positions = np.argsort(-logits.cpu().numpy(), axis=1, kind="stable")[:, :TOP]
            np.add.at(calibration_top, top_positions.ravel(), 1)
        calibration_top /= len(heads) * len(cal_indices)
        for index, row in enumerate(manifest):
            sample, group = row["sample_id"], row["group_id"]
            blob = archive.read(member_for(members, PRODUCT, sample))
            native = cv.imdecode(np.frombuffer(blob, np.uint8), cv.IMREAD_COLOR)
            assert native is not None
            image = cv.resize(native, (256, 256), interpolation=cv.INTER_AREA)
            feature = model(image_tensor(image[None], DEVICE))
            responses = [head(feature) for head in heads.values()]
            scores = np.array([float(r[1]) for r in responses])
            assert np.all(np.isfinite(scores))
            max_test_error = max(max_test_error,
                max(abs(score - test_reference[(*key, group)]) for key, score in zip(heads, scores)))
            raw = np.stack([r[0].reshape(-1).cpu().numpy() for r in responses])
            assert np.all(np.isfinite(raw))
            maps.append(raw)
            score_rows.append(scores)
            if row["label"] == 1:
                filename = sample.rsplit(".", 1)[0] + "_mask.png"
                annotation = archive.read(member_for(members, PRODUCT, filename))
                mask = cv.imdecode(np.frombuffer(annotation, np.uint8), cv.IMREAD_GRAYSCALE)
                assert mask is not None and mask.shape == native.shape[:2] and mask.any(), filename
                down = cv.resize((mask != 0).astype(np.float64), (GRID, GRID), interpolation=cv.INTER_AREA)
                masks[index] = down > 0
    assert max_test_error < 1e-5 and max_cal_error < 1e-5
    assert len(masks) == 199
    return (manifest, np.stack(maps, axis=1), np.stack(score_rows, axis=1),
            masks, calibration_top.reshape(GRID, GRID), max_test_error, max_cal_error)


def controls_for(mask, density):
    y, x = np.where(mask)
    offsets = [(dy, dx) for dy in range(-y.min(), GRID-y.max())
               for dx in range(-x.min(), GRID-x.max())]
    true_density = density[mask].mean()
    candidates = []
    for dy, dx in offsets:
        ys, xs = y + dy, x + dx
        if not mask[ys, xs].any():
            mismatch = abs(density[ys, xs].mean() - true_density)
            candidates.append((mismatch, dy * dy + dx * dx, dy, dx))
    candidates.sort()
    selected = candidates[:CONTROL_COUNT]
    assert selected, "No feasible non-overlapping control"
    return true_density, [(mismatch, dy, dx, (y+dy)*GRID + x+dx)
                          for mismatch, _, dy, dx in selected]


def drop_region(top_indices, top_values, original, cells):
    membership = np.zeros(GRID * GRID, dtype=bool)
    membership[cells] = True
    removed = membership[top_indices]
    untouched = ~removed[..., :TOP].any(axis=-1)
    keep = (~removed) & (np.cumsum(~removed, axis=-1) <= TOP)
    assert np.all(keep.sum(axis=-1) == TOP)
    pooled = np.where(keep, top_values, 0).sum(axis=-1, dtype=np.float64) / TOP
    return np.where(untouched, original, pooled), removed[:, 0, :TOP].sum(axis=-1)


def pair_wins(scores):
    defect, normals = scores[:, :1], scores[:, 1:]
    return (defect > normals).astype(float) + .5 * (defect == normals)


def collect_pairs(manifest, maps, scores, masks, density, keys):
    normal_ids = [i for i, r in enumerate(manifest) if r["label"] == 0]
    assert len(normal_ids) == 25
    orders = np.argsort(-maps, axis=-1, kind="stable").astype(np.int16)
    sorted_logits = np.take_along_axis(maps, orders, axis=-1)
    bins, inspection_rows, control_rows = {k: [] for k in STRATA}, [], []
    for i, mask in masks.items():
        row = manifest[i]
        indices = [i, *normal_ids]
        cells = np.flatnonzero(mask)
        depth = TOP + len(cells)
        assert depth <= GRID * GRID
        top_indices, top_values = orders[:, indices, :depth], sorted_logits[:, indices, :depth]
        original = scores[:, indices]
        alt, top_hits = drop_region(top_indices, top_values, original, cells)
        target_density, controls = controls_for(mask, density)
        control_wins, control_hits = [], []
        for mismatch, dy, dx, translated in controls:
            check, hits = drop_region(top_indices, top_values, original, translated)
            control_wins.append(pair_wins(check))
            control_hits.append(hits)
            control_rows.append(dict(group_id=row["group_id"], defect_type=row["anomaly_class"],
                offset_y=int(dy), offset_x=int(dx), density_mismatch=float(mismatch)))
        outcomes = np.stack([pair_wins(original), pair_wins(alt),
                             np.mean(control_wins, axis=0)])
        bins[row["anomaly_class"]].append(outcomes)
        for h, (seed, arm) in enumerate(keys):
            inspection_rows.append(dict(group_id=row["group_id"], defect_type=row["anomaly_class"],
                seed=seed, arm=arm, region_cells=len(cells), control_regions=len(controls),
                calibration_top11_density=float(target_density), annotation_top11_hits=int(top_hits[h]),
                control_top11_hits=float(np.mean(control_hits, axis=0)[h]),
                full_win_rate=float(outcomes[0, h].mean()),
                annotation_excluded_win_rate=float(outcomes[1, h].mean()),
                control_excluded_win_rate=float(outcomes[2, h].mean())))
    return bins, inspection_rows, control_rows


def bootstrap_weights(n, rng):
    indices = rng.integers(n, size=(DRAWS, n))
    counts = np.zeros((DRAWS, n), dtype=float)
    np.add.at(counts, (np.arange(DRAWS)[:, None], indices), 1)
    return counts


def summarize(bins, keys, normal_n):
    rng = np.random.default_rng(1120)
    normal_w = bootstrap_weights(normal_n, rng)
    rows, levels = [], []
    arms = ("M", "A50", "A67", "A83", "S0")
    by_arm = {arm: [i for i, key in enumerate(keys) if key[1] == arm] for arm in arms}
    for kind, outcomes in bins.items():
        # part x exclusion condition x frozen head x normal comparator
        v = np.stack(outcomes)
        part_w = bootstrap_weights(len(v), rng)
        for arm in arms:
            group = v[:, :, by_arm[arm], :]
            for index, condition in enumerate(CONDITIONS):
                levels.append(dict(defect_type=kind, arm=arm, condition=condition, win_rate=group[:, index].mean()))
            if arm == "S0":
                continue
            changes = group - v[:, :, by_arm["S0"], :]
            excess = changes[:, 2] - changes[:, 1]
            contrasts = np.concatenate([changes, excess[:, None, :, :]], axis=1)
            for index, label in enumerate((*CONDITIONS, "annotation_specific_attenuation")):
                pair = contrasts[:, index].mean(axis=1)
                reps = ((part_w @ pair) * normal_w).sum(axis=1) / (len(v) * normal_n)
                lower, upper = np.quantile(reps, [.025, .975])
                seed_means = contrasts[:, index].mean(axis=(0, 2))
                rows.append(dict(defect_type=kind, arm=arm, comparison=label, defect_n=len(v),
                    normal_n=normal_n, mean_pp=100*float(pair.mean()), lower95_pp=100*float(lower),
                    upper95_pp=100*float(upper), seed42_pp=100*float(seed_means[0]),
                    seed0_pp=100*float(seed_means[1]), seed1_pp=100*float(seed_means[2])))
    return rows, levels


def main():
    encoder_sha = verify_and_freeze()
    models = checkpoints()
    manifest, maps, scores, masks, density, test_error, cal_error = collect_maps(encoder_sha, models)
    print("Replayed 3,360 whole-image scores and loaded all 199 masks.", flush=True)
    bins, per_part, controls = collect_pairs(manifest, maps, scores, masks, density, tuple(models))
    assert tuple(len(bins[kind]) for kind in STRATA) == (50, 50, 50, 49)
    rows, levels = summarize(bins, tuple(models), 25)
    write_csv(OUT / "paired_region_effects.csv", rows)
    write_csv(OUT / "region_win_rates.csv", levels)
    write_csv(OUT / "per_part_region_scores.csv", per_part)
    write_csv(OUT / "matched_control_regions.csv", controls)
    outputs = ("paired_region_effects.csv", "region_win_rates.csv",
               "per_part_region_scores.csv", "matched_control_regions.csv")
    save(OUT / "RECEIPT.json", dict(completed_at_utc=now(), masks=len(masks),
        score_replays=int(scores.size), max_abs_test_error=test_error, max_abs_calibration_error=cal_error,
        controls=len(controls), output_files={p: sha(OUT / p) for p in outputs}))
    print("Finished the fixed region test and all control-region analyses.", flush=True)


if __name__ == "__main__":
    main()
