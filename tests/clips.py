import json

import cv2
import numpy as np

from vision.tracker import calib_box

from .synth import make_frame


def hand_path(i):
    return 320 + 150 * np.sin(i / 25), 330 - 40 * np.sin(i / 11)


def write_clip(folder, cfg, name="synthetic", frames=150, calib=(0, 30), still=(100, 130), size=(640, 480)):
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / f"{name}.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    box = calib_box(cfg, size)
    center = (box[0] + box[2] / 2, box[1] + box[3] / 2)
    truth = {}
    for i in range(frames):
        if calib[0] <= i < calib[1]:
            pos = center
        elif still[0] <= i < still[1]:
            pos = hand_path(still[0])
        else:
            pos = hand_path(i)
        truth[i] = pos
        writer.write(make_frame([pos], size=size, seed=i))
    writer.release()
    meta = {"name": name, "video": video.name, "fps": 30.0, "frames": frames, "size": list(size),
            "calib": list(calib), "roi": list(box), "still": list(still)}
    (folder / f"{name}.json").write_text(json.dumps(meta))
    labels = {"every": 10, "frames": {}}
    for i in range(calib[1], frames, 10):
        x, y = truth[i]
        poly = cv2.ellipse2Poly((int(x), int(y)), (42, 42), 0, 0, 360, 20).tolist()
        labels["frames"][str(i)] = {"A": [int(x), int(y)], "B": None, "polys": [poly]}
    (folder / f"{name}.labels.json").write_text(json.dumps(labels))
    return video
