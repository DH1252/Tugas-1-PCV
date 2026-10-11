import cv2
import numpy as np

from .synth import BACKGROUND, FACE, _noise, draw_hand

PALM = np.array((120, 160, 210), np.float32)
EDGE = np.array((70, 95, 165), np.float32)


def shaded_frame(center, size=(640, 480), face=None, seed=0, scale=1.4, fingers=5):
    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BACKGROUND
    if face is not None:
        cv2.ellipse(img, (int(face[0]), int(face[1])), (55, 70), 0, 0, 360, FACE, -1)
    mask = np.zeros((h, w), np.uint8)
    r = draw_hand(mask, center, scale=scale, fingers=fingers, color=255)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.clip(np.hypot(xx - center[0], yy - center[1]) / (2.6 * r), 0, 1)[..., None]
    light = (0.85 + 0.3 * (xx / w))[..., None]
    shade = ((1 - d) * PALM + d * EDGE) * light
    img = np.where(mask[..., None] > 0, np.clip(shade, 0, 255), img).astype(np.uint8)
    img = cv2.add(img.astype(np.int16), _noise(img.shape, 4, seed % 8), dtype=cv2.CV_16S)
    hand = mask.copy()
    hand[int(center[1] + 1.3 * r):, :] = 0
    return np.clip(img, 0, 255).astype(np.uint8), hand
