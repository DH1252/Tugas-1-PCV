from dataclasses import dataclass

import cv2
import numpy as np

from .segmentation import denoise, hsv_box


class CalibrationError(Exception):
    pass


def hs_hist(hsv, mask, bins):
    return cv2.calcHist([hsv], [0, 1], mask, list(bins), [0, 180, 0, 256])


def normalize(hist):
    hist = cv2.GaussianBlur(hist.astype(np.float32), (3, 3), 0)
    total = hist.sum()
    return hist / total if total > 0 else hist


def posterior(hand_p, bg_p, min_bin):
    p = hand_p / (hand_p + bg_p + 1e-12)
    p[hand_p < min_bin * hand_p.max()] = 0
    return (p * 255).astype(np.float32)


@dataclass
class ColorModel:
    hand_p: np.ndarray
    bg_p: np.ndarray
    lo: tuple
    hi: tuple
    crcb_lo: tuple
    crcb_hi: tuple
    face_box: tuple | None
    min_bin: float
    ratio: np.ndarray = None

    def __post_init__(self):
        self.refresh()

    def refresh(self):
        self.ratio = posterior(self.hand_p, self.bg_p, self.min_bin)

    def adapt(self, hist, alpha):
        if hist.sum() <= 0:
            return
        self.hand_p = (1 - alpha) * self.hand_p + alpha * normalize(hist)
        self.refresh()


def core_mask(hsv, mask, bins, mass):
    hist = hs_hist(hsv, mask, bins)
    flat = np.sort(hist.ravel())[::-1]
    cum = np.cumsum(flat)
    if cum[-1] <= 0:
        return mask
    cut = flat[min(np.searchsorted(cum, mass * cum[-1]), len(flat) - 1)]
    keep = (hist >= max(cut, 1)).astype(np.float32) * 255
    core = cv2.calcBackProject([hsv], [0, 1], keep, [0, 180, 0, 256], 1)
    return cv2.bitwise_and(core, mask)


def build_model(samples, roi, cfg, hue_shift=0):
    v = cfg["vision"]
    bins = (v["h_bins"], v["s_bins"])
    shape = samples[0]["hsv"].shape[:2]
    x, y, w, h = roi
    roi_mask = np.zeros(shape, np.uint8)
    roi_mask[y:y + h, x:x + w] = 255

    hand_hist = np.zeros(bins, np.float32)
    hsv_px, crcb_px = [], []
    for s in samples:
        m = roi_mask if s["gate"] is None else cv2.bitwise_and(roi_mask, s["gate"])
        m = core_mask(s["hsv"], m, bins, v["calib_core_mass"])
        hand_hist += hs_hist(s["hsv"], m, bins)
        hsv_px.append(s["hsv"][m > 0])
        crcb_px.append(s["ycrcb"][m > 0][:, 1:])
    hsv_px = np.concatenate(hsv_px)
    crcb_px = np.concatenate(crcb_px)
    if len(hsv_px) < 0.3 * w * h * len(samples):
        raise CalibrationError("not enough skin in the box")
    loose = hsv_box(hsv_px.reshape(-1, 1, 3), ((160 + hue_shift) % 180, 15, 40), ((50 + hue_shift) % 180, 230, 255))
    if np.count_nonzero(loose) < 0.5 * len(hsv_px):
        raise CalibrationError("the box does not look like skin")

    p_lo, p_hi = v["percentiles"]
    margin = np.array(v["box_margin"])
    lo = np.clip(np.percentile(hsv_px, p_lo, axis=0) - margin, 0, [179, 255, 255]).astype(int)
    hi = np.clip(np.percentile(hsv_px, p_hi, axis=0) + margin, 0, [179, 255, 255]).astype(int)
    cm = v["ycrcb_margin"]
    crcb_lo = np.clip(np.percentile(crcb_px, p_lo, axis=0) - cm, 0, 255).astype(int)
    crcb_hi = np.clip(np.percentile(crcb_px, p_hi, axis=0) + cm, 0, 255).astype(int)

    last = samples[-1]
    skin = hsv_box(last["hsv"], lo, hi)
    if last["gate"] is not None:
        skin = cv2.bitwise_and(skin, last["gate"])
    _, skin = denoise(skin, 5, 9)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(skin)

    gx, gy = int(0.1 * w), int(0.1 * h)
    exclude = np.zeros(shape, np.uint8)
    exclude[max(0, y - gy):y + h + gy, max(0, x - gx):x + w + gx] = 255
    face_i, best = None, v["face_min_area"] * shape[0] * shape[1]
    for i in range(1, n):
        comp = labels == i
        if exclude[comp].any():
            exclude[comp] = 255
            continue
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= best and centroids[i][1] < 0.75 * shape[0]:
            face_i, best = i, area
    face_box = None
    if face_i is not None:
        exclude[labels == face_i] = 255
        face_box = tuple(int(s) for s in stats[face_i, :4])
    exclude = cv2.dilate(exclude, np.ones((9, 9), np.uint8))

    bg_hist = np.zeros(bins, np.float32)
    keep = cv2.bitwise_not(exclude)
    for s in samples:
        m = keep if s["gate"] is None else cv2.bitwise_and(keep, s["gate"])
        bg_hist += hs_hist(s["hsv"], m, bins)

    return ColorModel(normalize(hand_hist), normalize(bg_hist), tuple(lo), tuple(hi),
                      tuple(crcb_lo), tuple(crcb_hi), face_box, v["min_hand_bin"])
