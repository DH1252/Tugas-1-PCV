import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from game import render

HELP = [
    "left click: hand A palm   right click: hand B palm",
    "g: polygon mode (left add, right undo, Enter close)",
    "space/d: next   a: back   e: no hands   x: clear   s: save   q: quit",
]
COLORS = {"A": render.HAND_COLORS[0], "B": render.HAND_COLORS[1]}


def labels_path(video):
    return Path(video).with_suffix(".labels.json")


def load_labels(video, every):
    path = labels_path(video)
    if path.exists():
        return json.loads(path.read_text())
    return {"every": every, "frames": {}}


class Annotator:
    def __init__(self, video, every):
        self.video = video
        self.cap = cv2.VideoCapture(str(video))
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta_path = Path(video).with_suffix(".json")
        start = 0
        if meta_path.exists():
            calib = json.loads(meta_path.read_text()).get("calib")
            start = calib[1] if calib else 0
        self.indices = list(range(start, n, every))
        self.labels = load_labels(video, every)
        self.pos = 0
        self.poly_mode = False
        self.poly = []
        self.frame = None

    @property
    def index(self):
        return self.indices[self.pos]

    def entry(self, create=True):
        key = str(self.index)
        if key not in self.labels["frames"] and create:
            self.labels["frames"][key] = {"A": None, "B": None, "polys": []}
        return self.labels["frames"].get(key)

    def load(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.index)
        ok, frame = self.cap.read()
        self.frame = frame if ok else np.zeros((480, 640, 3), np.uint8)
        self.poly = []

    def on_mouse(self, event, x, y, flags, _):
        if self.poly_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.poly.append([x, y])
            elif event == cv2.EVENT_RBUTTONDOWN and self.poly:
                self.poly.pop()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.entry()["A"] = [x, y]
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.entry()["B"] = [x, y]

    def draw(self):
        view = self.frame.copy()
        e = self.entry(create=False)
        if e:
            for poly in e["polys"]:
                cv2.polylines(view, [np.array(poly, np.int32)], True, render.GREEN, 2)
            for name in ("A", "B"):
                if e[name]:
                    cv2.circle(view, tuple(e[name]), 8, COLORS[name], 2)
                    render.text(view, name, (e[name][0] + 10, e[name][1] - 10), 0.6, COLORS[name], 2)
        if self.poly:
            cv2.polylines(view, [np.array(self.poly, np.int32)], False, render.YELLOW, 1)
        status = "labeled" if e is not None else "unlabeled"
        mode = "POLYGON" if self.poly_mode else "palm"
        render.text(view, f"frame {self.index} ({self.pos + 1}/{len(self.indices)})  {status}  mode {mode}",
                    (10, 24), 0.55, render.WHITE, 1)
        for i, line in enumerate(HELP):
            render.text(view, line, (10, view.shape[0] - 60 + i * 20), 0.45, render.GREY, 1)
        return view

    def save(self):
        labels_path(self.video).write_text(json.dumps(self.labels, indent=1))

    def run(self):
        if not self.indices:
            print("no frames to label")
            return
        cv2.namedWindow("annotate")
        cv2.setMouseCallback("annotate", self.on_mouse)
        self.load()
        while True:
            cv2.imshow("annotate", self.draw())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("d")):
                self.pos = min(self.pos + 1, len(self.indices) - 1)
                self.load()
            elif key == ord("a"):
                self.pos = max(self.pos - 1, 0)
                self.load()
            elif key == ord("g"):
                self.poly_mode = not self.poly_mode
                self.poly = []
            elif key in (10, 13) and self.poly_mode and len(self.poly) >= 3:
                self.entry()["polys"].append(self.poly)
                self.poly = []
            elif key == ord("e"):
                self.labels["frames"][str(self.index)] = {"A": None, "B": None, "polys": []}
            elif key == ord("x"):
                self.labels["frames"].pop(str(self.index), None)
            elif key == ord("s"):
                self.save()
        self.save()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Label palm centres and hand polygons")
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()
    Annotator(args.video, args.every).run()


if __name__ == "__main__":
    main()
