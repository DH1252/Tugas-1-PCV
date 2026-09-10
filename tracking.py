"""Color tracking: calibration, light correction, color masks, blobs and a Kalman tracker.

Per frame (Vision.__call__):
  1. prepare()        mirror, cut to 4:3, shrink to 320x240, blur
  2. LightCorrection  scale each channel so the walls match the calibration picture
  3. HSV + inRange    marker hue, saturation and brightness -> mask (0 or 255)
  4. exclusion        drop places that had the marker's color in the empty room
  5. clean()          opening removes specks, closing fills holes
  6. find_blobs()     connected components with area and palm center
  7. Tracker          Kalman filter, predicts through short dropouts

OpenCV HSV: H is 0..179 (degrees / 2) and wraps (red is near 0 and 179); S, V are 0..255.
OpenCV images are BGR.
"""
import math
from collections import namedtuple

import cv2
import numpy as np

W, H = 320, 240                                  # processing size
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
MIN_AREA = 40                                    # px; smaller blobs are noise
MAX_AREA = 0.6 * W * H                           # a close glove is big
USE_OPENCL = False                               # set by enable_opencl()

# Play area in processing px: the 16:9 crop minus a margin (see set_margin).
AREA_X, AREA_Y, AREA_W, AREA_H = 32.0, 48.0, 256.0, 144.0

# status: searching/tracking/coasting; pos: 0..1 on screen; vel: screen heights/s; t: capture time.
Saber = namedtuple("Saber", "status pos vel t")


def set_margin(margin):
    """margin: fraction of the 16:9 crop left out on each side (0.1 = 10%)."""
    global AREA_X, AREA_Y, AREA_W, AREA_H
    crop_h = W * 9 / 16                          # 180
    AREA_X = margin * W
    AREA_Y = (H - crop_h) / 2 + margin * crop_h
    AREA_W = W * (1 - 2 * margin)
    AREA_H = crop_h * (1 - 2 * margin)


def enable_opencl():
    """Run steps 3-5 on the GPU via cv2.UMat; returns the GPU name, or None."""
    global USE_OPENCL
    USE_OPENCL = cv2.ocl.haveOpenCL()
    if not USE_OPENCL:
        return None
    cv2.ocl.setUseOpenCL(True)
    return cv2.ocl.Device.getDefault().name()


def prepare(frame):
    """Step 1: returns (mirrored frame for display, small blurred frame for tracking)."""
    fh, fw = frame.shape[:2]
    if fw * 3 != fh * 4:                         # e.g. 1280x720: crop the sides to 4:3
        cw, ch = min(fw, fh * 4 // 3), min(fh, fw * 3 // 4)
        frame = frame[(fh - ch) // 2:(fh + ch) // 2, (fw - cw) // 2:(fw + cw) // 2]
    mirrored = cv2.flip(frame, 1)                # horizontal flip, like a mirror
    small = cv2.resize(mirrored, (W, H), interpolation=cv2.INTER_AREA)
    return mirrored, cv2.GaussianBlur(small, (5, 5), 0)


def to_screen(x, y):
    """Processing px -> screen position 0..1, clamped to the play area."""
    u = (x - AREA_X) / AREA_W
    v = (y - AREA_Y) / AREA_H
    return min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)


def hue_offset(h, center):
    """Signed hue difference across the wrap: hue_offset(2, 178) == 4."""
    return (np.asarray(h, np.float32) - center + 90) % 180 - 90


def clean(mask):
    """Step 5: opening (erode, dilate) removes specks; closing (dilate, erode) fills holes."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)


# ---------- step 2: light correction ----------

class LightCorrection:
    """Per-channel gains = median(background / current) over the top and side wall strips."""

    def __init__(self, background):
        walls = np.zeros((H, W), bool)
        walls[:36, :] = True                     # top strip
        walls[:, :32] = True                     # left strip
        walls[:, -32:] = True                    # right strip
        self.idx = np.flatnonzero(walls)[::8]    # every 8th wall pixel
        self.reference = background.reshape(-1, 3)[self.idx].astype(np.float32)
        self.gains = np.ones(3, np.float32)

    def __call__(self, small):
        current = small.reshape(-1, 3)[self.idx].astype(np.float32)
        # Black and blown-out pixels don't scale with the light.
        usable = ((current.min(axis=1) > 8) & (current.max(axis=1) < 250)
                  & (self.reference.min(axis=1) > 8) & (self.reference.max(axis=1) < 250))
        if usable.sum() > 50:
            gains = np.median(self.reference[usable] / current[usable], axis=0)
            gains = np.clip(gains, 0.25, 4.0)
            self.gains += 0.3 * (gains - self.gains)     # smoothing
        return np.clip(small * self.gains, 0, 255).astype(np.uint8)


# ---------- step 3: color model ----------

HUE_NAMES = [(8, "red"), (20, "orange"), (33, "yellow"), (80, "green"), (95, "cyan"),
             (130, "blue"), (150, "purple"), (170, "pink"), (180, "red")]


class ColorModel:
    """A marker's color as an HSV box: hue range plus minimum saturation and brightness."""

    def __init__(self, hsv_pixels):
        h = hsv_pixels[:, 0].astype(np.float32)
        s, v = hsv_pixels[:, 1], hsv_pixels[:, 2]
        # Circular mean: a plain mean of 2 and 178 (both red) gives 90 (cyan).
        angle = h * (np.pi / 90)
        self.hue = math.degrees(math.atan2(np.sin(angle).mean(), np.cos(angle).mean())) / 2 % 180
        # 2nd-98th percentile of the samples, plus 4 on each side.
        lo, hi = np.percentile(hue_offset(h, self.hue), [2, 98])
        self.h_lo = math.floor(self.hue + lo - 4)
        self.h_hi = math.ceil(self.hue + hi + 4)
        # Loose floors: motion blur washes the color out.
        self.s_min = int(np.clip(np.percentile(s, 5) - 40, 60, 140))
        self.v_min = int(np.clip(np.percentile(v, 5) - 50, 40, 120))
        self.name = next(name for limit, name in HUE_NAMES if self.hue < limit)
        rgb = cv2.cvtColor(np.uint8([[[round(self.hue) % 180, 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
        self.rgb = tuple(int(c) for c in rgb)    # display color

    def mask(self, hsv):
        """255 inside the box; a range crossing 0/179 (red) needs two inRange calls."""
        lo, hi, s, v = self.h_lo, self.h_hi, self.s_min, self.v_min
        if lo < 0:
            return cv2.bitwise_or(cv2.inRange(hsv, (0, s, v), (hi, 255, 255)),
                                  cv2.inRange(hsv, (lo + 180, s, v), (179, 255, 255)))
        if hi > 179:
            return cv2.bitwise_or(cv2.inRange(hsv, (lo, s, v), (179, 255, 255)),
                                  cv2.inRange(hsv, (0, s, v), (hi - 180, 255, 255)))
        return cv2.inRange(hsv, (lo, s, v), (hi, 255, 255))


