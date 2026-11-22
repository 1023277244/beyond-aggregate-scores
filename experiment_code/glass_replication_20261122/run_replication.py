"""Run the paired Real-IAD Variety GLASS experiment in this directory."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import sys
import tarfile
import time
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCH = ROOT / "bottle_cap_pilot/public_benchmark"
UPSTREAM = BENCH / "sources/GLASS_f788d567"
sys.path[:0] = [str(HERE / "vendor"), str(UPSTREAM)]

import cv2
import imgaug
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
from torchvision import transforms
from torchvision.models import wide_resnet50_2
from tqdm import tqdm

from glass import GLASS
from perlin import perlin_mask
from utils import distribution_judge, torch_format_2_numpy_img

PRODUCTS = {
    "ceramic_wave_filter": ("strength_extension_20261121_v1/A", "strength_extension_20261121_v1/A/SPLITS.csv"),
    "nylon_ball_head": ("final_evaluation_20261121_v1", "FINAL_DATA_SPLITS.csv"),
}
ARMS = {"strong": 2 / 3, "weak": 1 / 3}
SEEDS = [42, 0, 1]
WEIGHTS = BENCH / "assets/wide_resnet50_2-95faca4d.pth"
DTD = HERE / "assets/dtd-r1.0.1.tar.gz"
TEXTURES = HERE / "assets/textures"
SIZE = 256
BATCH = 2
DEVICE = torch.device("cuda")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SCREEN_NORMALIZE = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
PHOTO_RESIZE = transforms.Resize((SIZE, SIZE))


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage_textures():
    assert digest(DTD, "md5") == "fff73e5086ae6bdbea199a49dfb8a4c1", "DTD checksum"
    TEXTURES.mkdir(exist_ok=True)
    with tarfile.open(DTD, "r:gz") as archive:
        for entry in archive:
            parts = PurePosixPath(entry.name).parts
            if len(parts) != 4 or parts[:2] != ("dtd", "images") or not entry.isfile():
                continue
            category, name = parts[2:]
            assert category.replace("-", "").isalpha(), category
            if not name.endswith(".jpg") or name.startswith("."):
                continue
            assert Path(name).name == name and ":" not in name, name
            folder = TEXTURES / category
            folder.mkdir(exist_ok=True)
            (folder / name).write_bytes(archive.extractfile(entry).read())
    count = len(list(TEXTURES.glob("*/*.jpg")))
    assert count == 5640, count
    print(f"Staged {count} verified DTD textures.", flush=True)


def split_sources(product, seed):
    old_root, split_path = PRODUCTS[product]
    folder = ROOT / old_root / product
    saved = [
        row for row in csv_rows(ROOT / split_path)
        if row["product"] == product and int(row["seed"]) == seed
    ]
    manifest = load(folder / "train_manifest.json")
    ready = load(folder / "train_ready.json")
    normal_path = folder / "train_only_inputs.npz"
    assert digest(normal_path) == ready["cache_sha256"]
    with np.load(normal_path, allow_pickle=False) as cached:
        normals = cached["images"]
    assert len(normals) == len(manifest)
    assert all(int(row["label"]) == 0 and row["author_split"] == "train" for row in manifest)
    positions = {row["group_id"]: index for index, row in enumerate(manifest)}
    assert len(positions) == len(manifest)
    roles = {role: sorted(row["group_id"] for row in saved if row["role"] == role)
             for role in ["train", "calibration", "test"]}
    assert len(roles["train"]) == 12 and len(roles["calibration"]) == 6, roles
    assert len(set(roles["train"] + roles["calibration"] + roles["test"])) == sum(map(len, roles.values()))
    assert set(positions).isdisjoint(roles["test"])
    batches = {}
    for role in ["train", "calibration"]:
        batches[role] = torch.stack([
            SCREEN_NORMALIZE(np.ascontiguousarray(normals[positions[group], :, :, ::-1]))
            for group in roles[role]
        ])
    return folder, batches, roles


class LocalPairs(torch.utils.data.Dataset):
    """Render the same local texture and Perlin mask in either blend arm."""

    def __init__(self, images, textures, seed, alpha):
        self.images = images
        self.textures = textures
        self.seed = seed
        self.alpha = alpha
        self.epoch = 0
        self.audit = hashlib.sha256()
        self.transforms = [
            transforms.ColorJitter(contrast=(.8, 1.2)),
            transforms.ColorJitter(brightness=(.8, 1.2)),
            transforms.ColorJitter(saturation=(.8, 1.2), hue=(-.2, .2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        local_seed = self.seed * 1_000_000 + self.epoch * len(self) + idx
        rng_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                np.random.seed(local_seed)
                torch.random.default_generator.manual_seed(local_seed)
                imgaug.seed(local_seed)
                texture_path = self.textures[np.random.randint(len(self.textures))]
                ops = np.random.choice(len(self.transforms), 3, replace=False)
                with Image.open(texture_path) as texture_image:
                    texture = PHOTO_RESIZE(texture_image.convert("RGB"))
                    for index in ops:
                        texture = self.transforms[index](texture)
                    texture = SCREEN_NORMALIZE(texture)
                mask_s, mask = perlin_mask(image.shape, SIZE // 8, 0, 6, torch.tensor([1]), 1)
        finally:
            np.random.set_state(rng_state)
        self.audit.update(local_seed.to_bytes(8, "little"))
        self.audit.update(texture_path.name.encode())
        self.audit.update(texture.numpy().tobytes())
        self.audit.update(mask.tobytes())
        mask = torch.from_numpy(mask).bool()
        variant = torch.where(mask, image + self.alpha * (texture - image), image)
        return {"image": image, "aug": variant, "mask_s": torch.from_numpy(mask_s)}


def configure_runtime():
    assert torch.cuda.is_available(), "CUDA is required"
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def network(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    backbone = wide_resnet50_2(weights=None)
    backbone.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model = GLASS(DEVICE)
    model.load(backbone, ["layer2", "layer3"], DEVICE, (3, SIZE, SIZE), 1536, 1536)
    return model


def scores_for(model, images):
    scores = []
    for batch in images.split(BATCH):
        values, _ = model._predict(batch)
        scores.extend(float(value) for value in values)
    assert np.all(np.isfinite(scores))
    return scores


def train_one(product, seed, arm, epochs, output, textures):
    output.mkdir(parents=True, exist_ok=True)
    _, batches, roles = split_sources(product, seed)
    assert tuple(batches["train"].shape) == (12, 3, SIZE, SIZE)
    dataset = LocalPairs(batches["train"], textures, seed, ARMS[arm])
    model = network(seed)
    model.set_model_dir(str(output), "checkpoints")
    model.svd = int(distribution_judge(torch_format_2_numpy_img(batches["train"].mean(dim=0).numpy()), f"{product}_{seed}"))
    train_order = torch.Generator().manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    start = time.monotonic()
    with tqdm(range(epochs), disable=True) as progress:
        for epoch in progress:
            dataset.epoch = epoch
            with torch.no_grad():
                clean = batches["train"]
                embedded = []
                for x in clean.split(BATCH):
                    features = model._embed(x.to(DEVICE))[0]
                    projected = model.pre_projection(features)
                    embedded.append(projected.reshape(len(x), -1, projected.shape[-1]))
                model.c = torch.cat(embedded).mean(dim=0)

            sampler = torch.randperm(len(dataset), generator=train_order).tolist()
            loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, sampler=sampler, num_workers=0)
            desc, _, _ = model._train_discriminator(loader, epoch, progress, "")
            loss = float(desc.split("loss:")[1].split()[0])
            assert np.isfinite(loss), desc
            losses.append(loss)
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                print(f"{product} seed={seed} {arm} epoch={epoch+1}/{epochs} {desc} elapsed={time.monotonic()-start:.1f}s",
                      flush=True)
    checkpoint = output / "final.pth"
    torch.save({"discriminator": model.discriminator.state_dict(),
                "pre_projection": model.pre_projection.state_dict()}, checkpoint)
    calibration = scores_for(model, batches["calibration"])
    receipt = dict(product=product, seed=seed, arm=arm, alpha=ARMS[arm], epochs=epochs,
                   updates=epochs * (12 // BATCH), paired_inputs_sha256=dataset.audit.hexdigest(),
                   final_sha256=digest(checkpoint), train=roles["train"], calibration_groups=roles["calibration"],
                   calibration_scores=calibration, threshold=max(calibration), gas_projection=model.svd,
                   checkpoint_rule="last update", test_images_decoded=0, losses=losses,
                   seconds=time.monotonic()-start, peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20)
    model.logger.logger.close()
    save(output / "fit.json", receipt)
    return receipt


def fit_sweep(epochs, smoke):
    output = HERE / ("smoke" if smoke else "formal")
    output.mkdir(exist_ok=True)
    textures = sorted(TEXTURES.glob("*/*.jpg"))
    assert len(textures) == 5640
    products = ["ceramic_wave_filter"] if smoke else list(PRODUCTS)
    seeds = [42] if smoke else SEEDS
    config = dict(protocol="GLASS paired local-strength replication", source_commit="f788d567db552820d744ca2d93e989265aad8c45",
                  products=products, seeds=seeds, strengths=ARMS, epochs=epochs, batch_size=BATCH, input_size=SIZE,
                  global_gradient_steps=20, gas_projection="official FFT rule using mean normal training image",
                  backbone="ImageNet WRN50-2, frozen; layer2/layer3", foreground_mask=False,
                  precision="float32 with TF32", schedule="seeded order with paired texture, transforms and Perlin draws",
                  threshold="maximum of six independent normal calibration scores", checkpoint="last update",
                  candidate_split_hashes={p: digest(ROOT / PRODUCTS[p][1]) for p in products},
                  runner_sha256=digest(Path(__file__)), texture_archive_sha256=digest(DTD), weight_sha256=digest(WEIGHTS))
    protocol = output / "PROTOCOL.json"
    if protocol.exists():
        assert load(protocol)["config"] == config, "Use the frozen protocol when resuming"
    else:
        save(protocol, dict(frozen_at=stamp(), config=config))

    all_receipts = {}
    for product in products:
        for seed in seeds:
            receipts = []
            for arm in ARMS:
                key = f"{product}/seed{seed}_{arm}"
                destination = output / key
                if (destination / "fit.json").exists():
                    receipt = load(destination / "fit.json")
                    assert digest(destination / "final.pth") == receipt["final_sha256"]
                else:
                    receipt = train_one(product, seed, arm, epochs, destination, textures)
                receipts.append(receipt)
                all_receipts[key] = digest(destination / "fit.json")
            for field in ["paired_inputs_sha256", "train", "calibration_groups", "gas_projection", "updates"]:
                assert receipts[0][field] == receipts[1][field], (product, seed, field)
            print(f"Paired inputs verified for {product}, seed {seed}.", flush=True)
    save(output / "ALL_MODELS_FROZEN.json",
         dict(frozen_at=stamp(), protocol_sha256=digest(protocol), receipts=all_receipts))


def load_test(product):
    old_root, _ = PRODUCTS[product]
    path = ROOT / old_root / product
    ready = load(path / "train_ready.json")
    archive_path = Path(ready["archive_path"])
    assert digest(archive_path) == ready["archive_sha256"]
    rows = [r for r in load(path / "reserved_manifest.json") if r["author_split"] == "test"]
    assert len({row["group_id"] for row in rows}) == len(rows)
    normal_hashes = {r["input_sha256"] for r in load(path / "normal_input_hashes.json")}
    images = []
    with ZipFile(archive_path) as archive:
        for row in rows:
            member = row["sample_id"]
            names = [key for key in [member, product + "/" + member] if key in archive.NameToInfo]
            assert len(names) == 1
            encoded = archive.read(names[0])
            image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            assert hashlib.sha256(image.tobytes()).hexdigest() not in normal_hashes
            images.append(SCREEN_NORMALIZE(np.ascontiguousarray(image[:, :, ::-1])))
    return torch.stack(images), rows


def evaluate():
    output = HERE / "formal"
    freeze = load(output / "ALL_MODELS_FROZEN.json")
    assert digest(output / "PROTOCOL.json") == freeze["protocol_sha256"]
    assert len(freeze["receipts"]) == 12
    for key, expected in freeze["receipts"].items():
        folder = output / key
        assert digest(folder / "fit.json") == expected
        assert digest(folder / "final.pth") == load(folder / "fit.json")["final_sha256"]
    save(output / "EVALUATION_UNLOCK.json", dict(unlocked_at=stamp(), freeze_sha256=digest(output / "ALL_MODELS_FROZEN.json")))
    metrics, raw_scores = [], []
    for product in PRODUCTS:
        images, manifest = load_test(product)
        label = np.array([r["label"] for r in manifest])
        kind = np.array([r["anomaly_class"] for r in manifest])
        for seed in SEEDS:
            for arm in ARMS:
                key = f"{product}/seed{seed}_{arm}"
                folder = output / key
                fit = load(folder / "fit.json")
                assert set(fit["train"] + fit["calibration_groups"]).isdisjoint(r["group_id"] for r in manifest)
                model = network(seed)
                state = torch.load(folder / "final.pth", map_location=DEVICE, weights_only=True)
                model.pre_projection.load_state_dict(state["pre_projection"])
                model.discriminator.load_state_dict(state["discriminator"])
                score = np.asarray(scores_for(model, images))
                fp_budget = int((label == 0).sum()) // 25
                diagnostic = float(np.sort(score[label == 0])[::-1][fp_budget])
                fixed = float(fit["threshold"])
                fp = int((score[label == 0] > fixed).sum())
                diagnostic_fp = int((score[label == 0] > diagnostic).sum())
                assert diagnostic_fp <= fp_budget
                for part, value in zip(manifest, score):
                    raw_scores.append(dict(product=product, seed=seed, arm=arm, group_id=part["group_id"],
                                           label=part["label"], defect_type=part["anomaly_class"], score=float(value)))
                for stratum in ["Overall", *sorted(set(kind[label == 1]))]:
                    positive = label == 1 if stratum == "Overall" else kind == stratum
                    keep = positive | (label == 0)
                    metrics.append(dict(product=product, seed=seed, arm=arm, stratum=stratum,
                                        auroc=float(roc_auc_score(label[keep], score[keep])),
                                        n=int(positive.sum()), normal_n=int((label == 0).sum()),
                                        fixed_threshold=fixed, fp=fp, tp=int((score[positive] > fixed).sum()),
                                        diagnostic_threshold=diagnostic, diagnostic_fp=diagnostic_fp,
                                        diagnostic_tp=int((score[positive] > diagnostic).sum())))
                print(f"Scored {key}.", flush=True)
        write_csv(output / "METRICS.csv", metrics)
        write_csv(output / "RAW_TEST_SCORES.csv", raw_scores)
    save(output / "ALL_TEST_SCORES_FROZEN.json", dict(frozen_at=stamp(), metrics_sha256=digest(output / "METRICS.csv"),
                                                       scores_sha256=digest(output / "RAW_TEST_SCORES.csv")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "smoke", "fit", "evaluate", "all"])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.stage == "prepare":
        stage_textures()
        return
    configure_runtime()
    if args.stage == "smoke":
        fit_sweep(2, smoke=True)
        return
    if args.stage in ["fit", "all"]:
        fit_sweep(args.epochs, smoke=False)
    if args.stage in ["evaluate", "all"]:
        evaluate()


if __name__ == "__main__":
    main()
