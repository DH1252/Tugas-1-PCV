import cv2
import numpy as np

from .synth import _noise, draw_hand

BLUE = np.array((200, 95, 40), np.float32)
WHITE = np.array((235, 225, 220), np.float32)
SIZE = (854, 480)


def room(img):
    w = img.shape[1]
    img[:, : w // 2] = (150, 190, 210)
    img[:, w // 2:] = (70, 45, 30)
    cv2.rectangle(img, (620, 360), (820, 470), (190, 130, 70), -1)


def glossy_frame(center, seed=0, fade=0.55, background=(70, 80, 95), monitor=None, scene=False):
    w, h = SIZE
    img = np.empty((h, w, 3), np.uint8)
    img[:] = background
    if scene:
        room(img)
    if monitor is not None:
        cv2.rectangle(img, monitor[0], monitor[1], (190, 120, 60), -1)
    mask = np.zeros((h, w), np.uint8)
    r = draw_hand(mask, center, color=255)
    yy, xx = np.mgrid[0:h, 0:w]
    up = np.clip((center[1] - yy) / (2.6 * r), 0, 1)[..., None]
    shade = BLUE * (1 - fade * up) + WHITE * fade * up
    for k, (dx, dy) in enumerate(((-12, -70), (14, -95), (-30, -20))):
        spot = np.exp(-((xx - center[0] - dx) ** 2 + (yy - center[1] - dy) ** 2) / (2 * 7.0 ** 2))[..., None]
        shade = shade * (1 - 0.8 * spot) + 250 * 0.8 * spot
    img = np.where(mask[..., None] > 0, np.clip(shade, 0, 255), img).astype(np.uint8)
    img = cv2.add(img.astype(np.int16), _noise(img.shape, 4, seed % 8), dtype=cv2.CV_16S)
    hand = mask.copy()
    hand[int(center[1] + 1.3 * r):, :] = 0
    return np.clip(img, 0, 255).astype(np.uint8), hand


def calibrate(tracker, cfg, **kw):
    bx, by, bw, bh = cfg["vision"]["calib_box"]
    center = ((bx + bw / 2) * SIZE[0], (by + bh / 2) * SIZE[1])
    for i in range(cfg["vision"]["calib_frames"]):
        tracker.add_calibration_frame(glossy_frame(center, seed=i, **kw)[0])
    tracker.finish_calibration(SIZE)


def coverage(tracker, frames=15, **kw):
    recall, false = [], []
    for i in range(frames):
        center = (250 + 20 * i, 330)
        frame, hand = glossy_frame(center, seed=50 + i, **kw)
        tracker.process(frame, i / 30)
        m = cv2.resize(tracker.mask, SIZE, interpolation=cv2.INTER_NEAREST) > 0
        recall.append((m & (hand > 0)).sum() / (hand > 0).sum())
        glove = np.zeros(SIZE[::-1], np.uint8)
        draw_hand(glove, center, color=255)
        glove = cv2.dilate(glove, np.ones((15, 15), np.uint8)) > 0
        false.append((m & ~glove).sum() / max(m.sum(), 1))
    return float(np.mean(recall)), float(np.mean(false))
