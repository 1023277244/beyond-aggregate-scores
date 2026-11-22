"""Render two short event replays from a locally supplied source video."""
import argparse
import csv
from pathlib import Path

import cv2 as cv
import numpy as np
from score_events import anomaly_score

HERE = Path(__file__).resolve().parent
VIEW = (600, 400, 1120, 800)
FPS = 24


def render(source, target):
    cap = cv.VideoCapture(str(source))
    assert cap.isOpened()
    input_fps = cap.get(cv.CAP_PROP_FPS)
    assert input_fps > 0
    writer = cv.VideoWriter(str(target), cv.VideoWriter_fourcc(*"mp4v"), FPS, (720, 480))
    assert writer.isOpened()
    with (HERE / "events.csv").open(encoding="utf-8", newline="") as stream:
        events = list(csv.DictReader(stream))
    font = cv.FONT_HERSHEY_SIMPLEX
    try:
        for event in events:
            anchor = int(event["source_frame"])
            crop = cv.imdecode(np.fromfile(HERE / event["aligned_roi"], np.uint8), cv.IMREAD_COLOR)
            assert crop is not None
            score = anomaly_score(crop)
            alarm = int(score > float(event["threshold"]))
            assert alarm == int(event["predicted"])
            start = anchor - round(.75*input_fps)
            cap.set(cv.CAP_PROP_POS_FRAMES, start)
            frames = {}
            for i in range(start, anchor + round(.75*input_fps) + 1):
                ok, frame = cap.read()
                assert ok
                x0, y0, x1, y1 = VIEW
                assert frame.shape[0] >= y1 and frame.shape[1] >= x1
                frames[i] = cv.resize(frame[y0:y1, x0:x1], (468, 360), interpolation=cv.INTER_AREA)
            offsets = [round(i*input_fps/FPS) for i in range(-18, 19)]
            offsets = offsets[:18] + [0]*42 + offsets[19:]
            for offset in offsets:
                canvas = np.full((480, 720, 3), 250, np.uint8)
                canvas[58:418, 16:484] = frames[anchor+offset]
                cv.putText(canvas, "Conveyor end-face inspection", (16, 31), font, .82, (25, 31, 42), 2, cv.LINE_AA)
                cv.putText(canvas, "Recorded view", (16, 440), font, .54, (70, 75, 83), 1, cv.LINE_AA)
                cv.putText(canvas, event["sample_id"], (508, 73), font, .72, (25, 31, 42), 2, cv.LINE_AA)
                if offset == 0:
                    canvas[108:284, 509:685] = cv.resize(crop, (176, 176))
                    color = (45, 77, 180) if alarm else (100, 111, 22)
                    cv.putText(canvas, "ALARM" if alarm else "PASS", (509, 325), font, .95, color, 2, cv.LINE_AA)
                    cv.putText(canvas, "Archived aligned crop", (509, 358), font, .48, (60, 65, 75), 1, cv.LINE_AA)
                else:
                    cv.putText(canvas, "Event replay", (509, 199), font, .65, (65, 71, 81), 1, cv.LINE_AA)
                cv.putText(canvas, "Two selected development events | Decision at sampled frame", (16, 468),
                           font, .53, (45, 50, 60), 1, cv.LINE_AA)
                writer.write(canvas)
    finally:
        cap.release()
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.video, args.output)
