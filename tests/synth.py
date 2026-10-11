import cv2
import numpy as np

SKIN = (120, 160, 210)
FACE = (112, 150, 205)
BACKGROUND = (105, 95, 85)
GLOVE = (60, 170, 60)
FINGER_ANGLES = (-10, 10, -30, 30, -75)


def draw_hand(img, center, scale=1.4, fingers=5, color=SKIN, arm=True):
    cx, cy = int(center[0]), int(center[1])
    r = int(30 * scale)
    if arm:
        cv2.line(img, (cx, cy), (cx, img.shape[0] + 200), color, int(1.4 * r))
    for a in FINGER_ANGLES[:fingers]:
        t = np.radians(a)
        end = (int(cx + np.sin(t) * 2.6 * r), int(cy - np.cos(t) * 2.6 * r))
        cv2.line(img, (cx, cy), end, color, max(3, int(0.38 * r)))
    cv2.circle(img, (cx, cy), r, color, -1)
    return r


def make_frame(hands=(), face=(320, 110), size=(640, 480), noise=4, seed=0, background=BACKGROUND):
    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = background
    if face is not None:
        cv2.ellipse(img, (int(face[0]), int(face[1])), (55, 70), 0, 0, 360, FACE, -1)
    for spec in hands:
        if isinstance(spec, dict):
            draw_hand(img, **spec)
        else:
            draw_hand(img, spec)
    if noise:
        img = cv2.add(img.astype(np.int16), _noise(img.shape, noise, seed % 8), dtype=cv2.CV_16S)
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


_NOISE = {}


def _noise(shape, sigma, k):
    key = (shape, sigma, k)
    if key not in _NOISE:
        _NOISE[key] = np.random.default_rng(k).normal(0, sigma, shape).astype(np.int16)
    return _NOISE[key]


def calibration_frames(cfg, n=None, size=(640, 480), face=(320, 110), color=SKIN):
    bx, by, bw, bh = cfg["vision"]["calib_box"]
    center = ((bx + bw / 2) * size[0], (by + bh / 2) * size[1])
    n = n or cfg["vision"]["calib_frames"]
    return [make_frame([{"center": center, "color": color}], face=face, size=size, seed=i) for i in range(n)], center
