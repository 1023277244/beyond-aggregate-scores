"""Replay the original HSV end-face rule on the two archived aligned crops."""
import csv
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent


def anomaly_score(image):
    height, width = image.shape[:2]
    y, x = np.ogrid[:height, :width]
    disk = (x-width/2)**2 + (y-height/2)**2 < (.3*height)**2
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    yellow = cv.inRange(hsv, (15, 60, 60), (45, 255, 255))
    return -float((yellow[disk] > 0).mean())


if __name__ == "__main__":
    with (ROOT / "events.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            image = cv.imdecode(np.fromfile(ROOT / row["aligned_roi"], np.uint8), cv.IMREAD_COLOR)
            score = anomaly_score(image)
            predicted = int(score > float(row["threshold"]))
            assert np.isclose(score, float(row["score"]), atol=1e-12, rtol=0)
            assert predicted == int(row["predicted"])
            print(f"{row['sample_id']}: score={score:g}; {'ALARM' if predicted else 'PASS'}")
