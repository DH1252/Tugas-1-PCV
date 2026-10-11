import cv2
import numpy as np


class WhiteBalance:
    def __init__(self, clip=(0.6, 1.6), smooth=0.1):
        self.clip = clip
        self.smooth = smooth
        self.gain = None

    def reset(self):
        self.gain = None

    def __call__(self, bgr):
        means = np.array(cv2.mean(bgr)[:3]) + 1e-6
        gain = np.clip(means.mean() / means, *self.clip)
        gain /= gain.max()
        if self.gain is None:
            self.gain = gain
        else:
            self.gain = (1 - self.smooth) * self.gain + self.smooth * gain
        return cv2.transform(bgr, np.diag(self.gain).astype(np.float32))


def hue_lut(shift):
    lut = np.arange(256, dtype=np.int32)
    lut[:180] = (lut[:180] + shift) % 180
    return lut.astype(np.uint8)


def centre_shift(bgr_frames, roi, s_min, v_min):
    x, y, w, h = roi
    hues = []
    for f in bgr_frames:
        hsv = cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        ok = (hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min)
        hues.append(hsv[..., 0][ok])
    hues = np.concatenate(hues) if hues else np.array([])
    if len(hues) < 20:
        return None
    a = hues.astype(np.float64) * np.pi / 90
    mean = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2
    return int(round(90 - mean)) % 180


def rotate_hue(hsv, lut):
    h, s, v = cv2.split(hsv)
    return cv2.merge([cv2.LUT(h, lut), s, v])


def shift_box(lo, hi, shift):
    return ((lo[0] + shift) % 180, lo[1], lo[2]), ((hi[0] + shift) % 180, hi[1], hi[2])


def reliability_gate(hsv, s_min, v_min, v_max):
    return cv2.inRange(hsv, (0, s_min, v_min), (255, 255, v_max))
