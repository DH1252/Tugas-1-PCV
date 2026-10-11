import cv2
import numpy as np

from .synth import GLOVE, _noise, draw_hand

SIZE = (854, 480)
FPS = 30


def truth(i, fps=FPS, speed=1.0):
    t = i / fps * speed
    return (427 + 280 * np.sin(2 * np.pi * t / 1.6), 300 + 100 * np.sin(2 * np.pi * t / 0.8))


def blur(img, vx, vy, exposure):
    length = int(np.hypot(vx, vy) * exposure)
    if length < 2:
        return img
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    d = np.array([vx, vy]) / np.hypot(vx, vy) * c
    cv2.line(k, (int(round(c - d[0])), int(round(c - d[1]))), (int(round(c + d[0])), int(round(c + d[1]))), 1, 1)
    return cv2.filter2D(img, -1, k / k.sum())


def stress_frame(i, color=GLOVE, background=(105, 95, 85), fps=FPS, speed=1.0, exposure=1 / 60, light=0.35,
                 distractor=None):
    w, h = SIZE
    img = np.empty((h, w, 3), np.uint8)
    img[:] = background
    if distractor is not None:
        cv2.rectangle(img, distractor[0], distractor[1], color, -1)
    x, y = truth(i, fps, speed)
    x2, y2 = truth(i + 0.5, fps, speed)
    layer = img.copy()
    draw_hand(layer, (x, y), color=color)
    layer = blur(layer, (x2 - x) * 2 * fps, (y2 - y) * 2 * fps, exposure)
    gain = 1.0 + light * np.sin(2 * np.pi * i / fps / 3.0)
    out = np.clip(layer.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    out = cv2.add(out.astype(np.int16), _noise(out.shape, 4, i % 8), dtype=cv2.CV_16S)
    return np.clip(out, 0, 255).astype(np.uint8), (x, y)


def calibrate(tracker, cfg, color=GLOVE, n=None):
    bx, by, bw, bh = cfg["vision"]["calib_box"]
    center = ((bx + bw / 2) * SIZE[0], (by + bh / 2) * SIZE[1])
    for i in range(n or cfg["vision"]["calib_frames"]):
        img = np.empty((SIZE[1], SIZE[0], 3), np.uint8)
        img[:] = (105, 95, 85)
        draw_hand(img, center, color=color)
        img = cv2.add(img.astype(np.int16), _noise(img.shape, 4, i % 8), dtype=cv2.CV_16S)
        tracker.add_calibration_frame(np.clip(img, 0, 255).astype(np.uint8))
    tracker.finish_calibration(SIZE)


TOP_OFFSET = 100


def run(tracker, frames=300, radius=50, anchor="palm", **kw):
    lost, ids, prev = 0, set(), None
    for i in range(frames):
        frame, (x, y) = stress_frame(i, **kw)
        if anchor == "top":
            y -= TOP_OFFSET
        states = tracker.process(frame, i / FPS)
        near = [s for s in states if np.hypot(s.palm[0] - x, s.palm[1] - y) < radius]
        if not near:
            lost += 1
        else:
            ids.add(near[0].id)
    return lost / frames, len(ids)


def edge_truth(i, fps=FPS):
    t = i / fps
    return (427 + 440 * np.sin(2 * np.pi * t / 4.0), 250 + 220 * np.sin(2 * np.pi * t / 2.3))


def edge_frame(i, color=GLOVE, background=(105, 95, 85)):
    w, h = SIZE
    img = np.empty((h, w, 3), np.uint8)
    img[:] = background
    x, y = edge_truth(i)
    draw_hand(img, (x, y), color=color)
    img = cv2.add(img.astype(np.int16), _noise(img.shape, 4, i % 8), dtype=cv2.CV_16S)
    return np.clip(img, 0, 255).astype(np.uint8), (x, y)


def jumpiness(tracker, frames=240):
    steps, prev, lost = [], None, 0
    for i in range(frames):
        frame, truth = edge_frame(i)
        states = tracker.process(frame, i / FPS)
        visible = 40 < truth[0] < SIZE[0] - 40 and truth[1] < SIZE[1] - 20
        if not states:
            lost += visible
            prev = None
            continue
        point = np.array(states[0].palm)
        if prev is not None:
            steps.append(np.hypot(*((point - prev[0]) - (np.array(truth) - prev[1]))))
        prev = (point, np.array(truth))
    steps = np.array(steps)
    return {"p95": float(np.percentile(steps, 95)), "max": float(steps.max()), "lost": lost}
