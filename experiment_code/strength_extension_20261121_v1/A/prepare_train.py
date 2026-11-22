"""Decode only official training C1 images; keep all test pixels sealed."""
import collections
import zipfile

import cv2 as cv
import numpy as np

from common import ROOT, OUT, PRODUCTS, SEEDS, now, sha, save, read, verify_freeze, archive_for


def pixel_hash(image):
    import hashlib
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()


def member_for(names, product, relative):
    matches = [n for n in (relative,product+"/"+relative) if n in names]
    assert len(matches)==1, (product,relative)
    return matches[0]


def main():
    freeze = verify_freeze()
    expected = {r["product"]:r["expected_archive_bytes"] for r in read(OUT/"candidate_history_check.json")["selected"]}
    missing = [p for p in PRODUCTS if not any((folder/(p+".zip")).is_file() for folder in (
        __import__("pathlib").Path("/path/to/user/Downloads"),OUT/"downloads"))]
    if missing:
        save(OUT/"readiness.json",dict(state="awaiting_data",checked_at_utc=now(),missing=missing,freeze_sha=freeze))
        print("Missing",missing)
        return
    checks = {}
    for product in PRODUCTS:
        path = archive_for(product)
        assert path.stat().st_size == expected[product], "Archive incomplete or wrong revision"
        digest = sha(path)
        folder = OUT/product
        ready = folder/"train_ready.json"
        if ready.exists():
            old = read(ready)
            assert old["archive_sha256"]==digest and old["freeze_sha"]==freeze
            assert sha(folder/"train_only_inputs.npz")==old["cache_sha256"]
            checks[product]=old
            continue
        manifest = read(folder/"train_manifest.json")
        assert len(manifest)>=18 and all(r["author_split"]=="train" and r["label"]==0 for r in manifest)
        images, pixels = [], []
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            for r in manifest:
                blob = archive.read(member_for(names,product,r["sample_id"]))
                image = cv.imdecode(np.frombuffer(blob,np.uint8),cv.IMREAD_COLOR)
                assert image is not None
                resized = cv.resize(image,(256,256),interpolation=cv.INTER_AREA)
                import hashlib
                pixels.append(dict(group_id=r["group_id"],split=r["split"],
                    file_sha256=hashlib.sha256(blob).hexdigest(),
                    decoded_sha256=pixel_hash(image),input_sha256=pixel_hash(resized)))
                images.append(resized)
        assert len({r["decoded_sha256"] for r in pixels})==len(pixels), "Repeated official training pixels"
        assert len({r["input_sha256"] for r in pixels})==len(pixels), "Repeated resized training pixels"
        np.savez_compressed(folder/"train_only_inputs.npz",images=np.stack(images))
        save(folder/"normal_input_hashes.json",pixels)
        record = dict(state="train_ready",created_at_utc=now(),freeze_sha=freeze,
            archive_path=str(path),archive_sha256=digest,cache_sha256=sha(folder/"train_only_inputs.npz"),
            hash_inventory_sha256=sha(folder/"normal_input_hashes.json"),normal_images=len(images),
            decoded_test_images=0)
        save(ready,record)
        checks[product]=record
    save(OUT/"readiness.json",dict(state="train_ready",created_at_utc=now(),freeze_sha=freeze,products=checks))
    print("Three training caches ready. Test pixels remain unopened.")


if __name__=="__main__":
    main()

