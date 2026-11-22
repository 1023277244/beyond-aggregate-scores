"""Immutable phase boundaries for the fixed A/B extension."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
HISTORY=ROOT/"bottle_cap_pilot/public_benchmark"
PRODUCTS=["ceramic_wave_filter","pulse_transformer","bag_buckle","brooch_clasp_accessory","jam_jar_model","knob_cap"]
SEEDS=(42,0,1)
ARMS=("S0","M")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".part")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path,rows,fields=None):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".part")
    with temp.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def verify_freeze():
    frozen=read(OUT.parent/"FREEZE.json")
    for filename,value in frozen["files"].items():
        assert sha(ROOT/filename)==value,filename
    return sha(OUT.parent/"FREEZE.json")


def archive_for(product):
    for folder in (Path("/path/to/user/Downloads"),OUT.parent/"downloads",HISTORY/"downloads"):
        path=folder/(product+".zip")
        if path.is_file():
            return path
    raise FileNotFoundError(product+".zip")

