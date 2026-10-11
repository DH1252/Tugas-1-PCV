from dataclasses import dataclass, field

import cv2
import numpy as np

from .segmentation import back_project, denoise, ellipse, hsv_box, hysteresis


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
    tuned: dict = field(default_factory=dict)

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


def tune_gate(frames, roi, v):
    x, y, w, h = roi
    hsv = np.concatenate([cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2HSV).reshape(-1, 3) for f in frames])
    s_min = int(np.clip(0.5 * np.percentile(hsv[:, 1], 5), 10, v["s_min"]))
    v_min = int(np.clip(0.5 * np.percentile(hsv[:, 2], 5), 15, v["v_min"]))
    return {"s_min": s_min, "v_min": v_min}


def tune_kernels(roi):
    size = min(roi[2], roi[3])
    return {"open_kernel": int(np.clip(round(size * 0.1), 3, 7)) | 1,
            "close_kernel": int(np.clip(round(size * 0.35), 5, 21)) | 1}


def tune_thresholds(model, samples, hand_masks, bg_masks, v):
    hand, bg = [], []
    for s, hm, bm in zip(samples, hand_masks, bg_masks):
        prob = back_project(s["hsv"], model.ratio, v["backproject_blur"])
        hand.append(prob[hm > 0])
        bg.append(prob[bm > 0])
    hand, bg = np.concatenate(hand), np.concatenate(bg)
    if len(hand) == 0:
        return {}
    tpr = np.cumsum(np.bincount(hand, minlength=256)[::-1])[::-1] / len(hand)
    fpr = np.cumsum(np.bincount(bg, minlength=256)[::-1])[::-1] / max(len(bg), 1)
    ok = np.flatnonzero(fpr <= v["tune_fpr_high"])
    high = int(ok[0]) if len(ok) else int(np.argmax(tpr - fpr))
    keeps = np.flatnonzero(tpr >= 0.6)
    if len(keeps):
        high = min(high, int(keeps[-1]))
    high = int(np.clip(high, 40, 220))
    ok = np.flatnonzero(fpr <= v["tune_fpr_low"])
    low = int(ok[0]) if len(ok) else high
    keeps = np.flatnonzero(tpr >= 0.95)
    if len(keeps):
        low = min(low, int(keeps[-1]))
    low = int(np.clip(low, 10, high))
    return {"backproject_thresh": high, "backproject_low": low}


def grow_hand(model, samples, roi, avoid, tuned, v):
    shape = samples[0]["hsv"].shape[:2]
    x, y, w, h = roi
    window = np.zeros(shape, np.uint8)
    window[max(0, y - 3 * h):y + 4 * h, max(0, x - 3 * w):x + 4 * w] = 255
    window = cv2.bitwise_and(window, cv2.bitwise_not(avoid))
    regions = []
    for s in samples:
        prob = back_project(s["hsv"], model.ratio, v["backproject_blur"])
        if s["gate"] is not None:
            prob = cv2.bitwise_and(prob, s["gate"])
        m = hysteresis(prob, tuned["backproject_low"], tuned["backproject_thresh"])
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ellipse(tuned["close_kernel"]))
        n, labels = cv2.connectedComponents(m)
        ids = np.unique(labels[y:y + h, x:x + w])
        region = (np.isin(labels, ids[ids > 0]) * 255).astype(np.uint8)
        region = cv2.bitwise_and(region, window)
        region = cv2.bitwise_and(region, ((prob >= tuned["backproject_low"]) * 255).astype(np.uint8))
        regions.append(region)
    return regions


def build_model(samples, roi, cfg, tuned=None):
    v = cfg["vision"]
    bins = (v["h_bins"], v["s_bins"])
    shape = samples[0]["hsv"].shape[:2]
    x, y, w, h = roi
    roi_mask = np.zeros(shape, np.uint8)
    roi_mask[y:y + h, x:x + w] = 255

    hand_hist = np.zeros(bins, np.float32)
    hsv_px, crcb_px, cores = [], [], []
    for s in samples:
        m = roi_mask if s["gate"] is None else cv2.bitwise_and(roi_mask, s["gate"])
        m = core_mask(s["hsv"], m, bins, v["calib_core_mass"])
        cores.append(m)
        hand_hist += hs_hist(s["hsv"], m, bins)
        hsv_px.append(s["hsv"][m > 0])
        crcb_px.append(s["ycrcb"][m > 0][:, 1:])
    hsv_px = np.concatenate(hsv_px)
    crcb_px = np.concatenate(crcb_px)
    if len(hsv_px) < 0.3 * w * h * len(samples):
        raise CalibrationError("the box is too grey or dark to track by colour")

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
    face = np.zeros(shape, np.uint8)
    if face_i is not None:
        exclude[labels == face_i] = 255
        face[labels == face_i] = 255
        face_box = tuple(int(s) for s in stats[face_i, :4])
    exclude = cv2.dilate(exclude, np.ones((9, 9), np.uint8))

    bg_hist = np.zeros(bins, np.float32)
    keep = cv2.bitwise_not(exclude)
    bgs = []
    for s in samples:
        m = keep if s["gate"] is None else cv2.bitwise_and(keep, s["gate"])
        bgs.append(m)
        bg_hist += hs_hist(s["hsv"], m, bins)

    model = ColorModel(normalize(hand_hist), normalize(bg_hist), tuple(lo), tuple(hi),
                       tuple(crcb_lo), tuple(crcb_hi), face_box, v["min_hand_bin"])
    if tuned is None:
        return model

    tuned = dict(tuned)
    tuned.update(tune_thresholds(model, samples, cores, bgs, v))
    avoid = cv2.dilate(face, ellipse(15))
    for _ in range(v["grow_passes"]):
        regions = grow_hand(model, samples, roi, avoid, tuned, v)
        grown = np.zeros(bins, np.float32)
        for s, r in zip(samples, regions):
            grown += hs_hist(s["hsv"], r, bins)
        if grown.sum() <= 0:
            break
        model.hand_p = normalize(hand_hist + grown)
        model.refresh()
        tuned.update(tune_thresholds(model, samples, regions, bgs, v))
    model.tuned = {k: tuned[k] for k in ("backproject_thresh", "backproject_low") if k in tuned}
    return model
