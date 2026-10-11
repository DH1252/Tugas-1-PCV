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
    chroma_lo: tuple
    chroma_hi: tuple
    min_bin: float
    ratio: np.ndarray = None
    tuned: dict = field(default_factory=dict)
    distractors: np.ndarray = None
    core_lo: tuple = None
    core_hi: tuple = None

    def __post_init__(self):
        self.refresh()

    def refresh(self):
        self.ratio = posterior(self.hand_p, self.bg_p, self.min_bin)

    def adapt(self, hist, alpha):
        if hist.sum() <= 0:
            return
        self.hand_p = (1 - alpha) * self.hand_p + alpha * normalize(hist)
        self.refresh()


def sample_mask(shape, roi, shrink):
    x, y, w, h = roi
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (int(x + w / 2), int(y + h / 2)), (max(1, int(w * shrink / 2)), max(1, int(h * shrink / 2))),
                0, 0, 360, 255, -1)
    return m


def dominant_bins(hist, level=0.15):
    smooth = cv2.GaussianBlur(hist.astype(np.float32), (3, 3), 0)
    if smooth.max() <= 0:
        return np.ones_like(smooth, bool)
    peak = np.unravel_index(np.argmax(smooth), smooth.shape)
    n, labels = cv2.connectedComponents((smooth >= level * smooth.max()).astype(np.uint8), connectivity=8)
    return labels == labels[peak]


def core_mask(hsv, mask, bins, mass):
    hist = hs_hist(hsv, mask, bins)
    if hist.sum() <= 0:
        return mask
    keep = dominant_bins(hist)
    flat = np.sort(hist[keep])[::-1]
    cum = np.cumsum(flat)
    cut = flat[min(np.searchsorted(cum, mass * cum[-1]), len(flat) - 1)]
    keep = (keep & (hist >= max(cut, 1))).astype(np.float32) * 255
    core = cv2.calcBackProject([hsv], [0, 1], keep, [0, 180, 0, 256], 1)
    return cv2.bitwise_and(core, mask)


def tune_gate(frames, roi, v):
    x, y, w, h = roi
    bins = (v["h_bins"], v["s_bins"])
    hsv = []
    for f in frames:
        patch = cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        inner = sample_mask(patch.shape[:2], (0, 0, w, h), v["sample_shrink"])
        core = core_mask(patch, inner, bins, v["calib_core_mass"])
        hsv.append(patch[core > 0])
    hsv = np.concatenate(hsv)
    if len(hsv) == 0:
        return {}
    s_min = int(np.clip(0.5 * np.percentile(hsv[:, 1], 5), 10, v["s_min"]))
    v_min = int(np.clip(v["v_floor_ratio"] * np.percentile(hsv[:, 2], 5), 15, v["v_floor_max"]))
    return {"s_min": s_min, "v_min": v_min, "v_ref": float(np.median(hsv[:, 2]))}


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
    if len(bg) < v["tune_min_bg"]:
        fpr = np.zeros(256)
    else:
        fpr = np.cumsum(np.bincount(bg, minlength=256)[::-1])[::-1] / len(bg)
    ok = np.flatnonzero(fpr <= v["tune_fpr_high"])
    high = int(ok[0]) if len(ok) else int(np.argmax(tpr - fpr))
    if not fpr.any():
        high = v["backproject_thresh"]
    keeps = np.flatnonzero(tpr >= 0.6)
    if len(keeps):
        high = min(high, int(keeps[-1]))
    high = int(np.clip(high, v["tune_high_range"][0], v["tune_high_range"][1]))
    ok = np.flatnonzero(fpr <= v["tune_fpr_low"])
    low = int(ok[0]) if len(ok) else high
    if not fpr.any():
        low = v["backproject_low"]
    keeps = np.flatnonzero(tpr >= 0.95)
    if len(keeps):
        low = min(low, int(keeps[-1]))
    low = int(np.clip(low, min(v["tune_low_min"], high), high))
    return {"backproject_thresh": high, "backproject_low": low}


def grow_hand(model, samples, roi, tuned, v):
    shape = samples[0]["hsv"].shape[:2]
    x, y, w, h = roi
    window = np.zeros(shape, np.uint8)
    window[max(0, y - 3 * h):y + 4 * h, max(0, x - 3 * w):x + 4 * w] = 255
    regions = []
    for s in samples:
        prob = back_project(s["hsv"], model.ratio, v["backproject_blur"])
        if s["gate"] is not None:
            prob = cv2.bitwise_and(prob, s["gate"])
        high = min(tuned["backproject_thresh"], v["grow_thresh"][0])
        low = min(tuned["backproject_low"], v["grow_thresh"][1])
        m = hysteresis(prob, low, high)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ellipse(tuned["close_kernel"]))
        n, labels = cv2.connectedComponents(m)
        ids = np.unique(labels[y:y + h, x:x + w])
        region = (np.isin(labels, ids[ids > 0]) * 255).astype(np.uint8)
        region = cv2.bitwise_and(region, window)
        region = cv2.bitwise_and(region, ((prob >= low) * 255).astype(np.uint8))
        regions.append(region)
    return regions


def lookalikes(hand_p, samples, hand_zone, v):
    plain = hand_p / hand_p.max() if hand_p.max() > 0 else hand_p
    plain = np.where(plain >= v["min_hand_bin"], plain * 255, 0).astype(np.float32)
    away = cv2.bitwise_not(cv2.dilate(hand_zone, ellipse(25)))
    votes = np.zeros(samples[0]["hsv"].shape[:2], np.float32)
    for s in samples:
        prob = back_project(s["hsv"], plain, v["backproject_blur"])
        if s["gate"] is not None:
            prob = cv2.bitwise_and(prob, s["gate"])
        votes += cv2.bitwise_and((prob >= v["lookalike_thresh"]).astype(np.uint8) * 255, away) > 0
    found = ((votes >= 0.5 * len(samples)) * 255).astype(np.uint8)
    found = cv2.morphologyEx(found, cv2.MORPH_OPEN, ellipse(5))
    if not found.any() or found.mean() / 255 > v["lookalike_max_area"]:
        return None
    return cv2.dilate(found, ellipse(15))


def distractor_map(model, samples, hand_regions, params, v):
    found = np.zeros(samples[0]["hsv"].shape[:2], np.uint8)
    for s, region in zip(samples, hand_regions):
        prob = back_project(s["hsv"], model.ratio, v["backproject_blur"])
        if s["gate"] is not None:
            prob = cv2.bitwise_and(prob, s["gate"])
        m = hysteresis(prob, params["backproject_low"], params["backproject_thresh"])
        m = cv2.bitwise_and(m, cv2.bitwise_not(cv2.dilate(region, ellipse(25))))
        found = cv2.bitwise_or(found, m)
    found = cv2.morphologyEx(found, cv2.MORPH_OPEN, ellipse(5))
    if not found.any():
        return None
    return cv2.dilate(found, ellipse(15))


def hand_band(shape, roi):
    x, y, w, h = roi
    band = np.zeros(shape, np.uint8)
    band[max(0, y - h):, max(0, x - w // 4):x + w + w // 4] = 255
    return band


def trim(region, roi, limit):
    if cv2.countNonZero(region) <= limit:
        return region
    return cv2.bitwise_and(region, hand_band(region.shape, roi))


def hue_grow(samples, roi_mask, roi, h_lo, h_hi, v):
    lo = max(0, int(h_lo) - v["grow_hue_margin"])
    hi = min(179, int(h_hi) + v["grow_hue_margin"])
    limit = min(v["grow_max_area"] * roi_mask.size, v["grow_hand_ratio"] * cv2.countNonZero(roi_mask))
    regions = []
    for s in samples:
        band = cv2.inRange(s["hsv"], (lo, 0, 0), (hi, 255, 255))
        if s["gate"] is not None:
            band = cv2.bitwise_and(band, s["gate"])
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, ellipse(3))
        n, labels = cv2.connectedComponents(band)
        ids = np.unique(labels[roi_mask > 0])
        regions.append(trim((np.isin(labels, ids[ids > 0]) * 255).astype(np.uint8), roi, limit))
    return regions


def merge_maps(a, b):
    if a is None or b is None:
        return a if b is None else b
    return cv2.bitwise_or(a, b)


def build_model(samples, roi, cfg, tuned=None):
    v = cfg["vision"]
    bins = (v["h_bins"], v["s_bins"])
    shape = samples[0]["hsv"].shape[:2]
    x, y, w, h = roi
    roi_mask = sample_mask(shape, roi, v["sample_shrink"])

    hand_hist = np.zeros(bins, np.float32)
    hsv_px, chroma_px, cores = [], [], []
    for s in samples:
        m = roi_mask if s["gate"] is None else cv2.bitwise_and(roi_mask, s["gate"])
        m = core_mask(s["hsv"], m, bins, v["calib_core_mass"])
        cores.append(m)
        hand_hist += hs_hist(s["hsv"], m, bins)
        hsv_px.append(s["hsv"][m > 0])
        chroma_px.append(s["chroma"][m > 0])
    hsv_px = np.concatenate(hsv_px)
    chroma_px = np.concatenate(chroma_px)
    if len(hsv_px) < 0.5 * cv2.countNonZero(roi_mask) * len(samples):
        raise CalibrationError("the box is too grey or dark to track by colour")

    p_lo, p_hi = v["percentiles"]
    margin = np.array(v["box_margin"])
    lo = np.clip(np.percentile(hsv_px, p_lo, axis=0) - margin, 0, [179, 255, 255]).astype(int)
    hi = np.clip(np.percentile(hsv_px, p_hi, axis=0) + margin, 0, [179, 255, 255]).astype(int)
    cm = v["chroma_margin"]
    chroma_lo = np.clip(np.percentile(chroma_px, p_lo, axis=0) - cm, 0, 255).astype(int)
    chroma_hi = np.clip(np.percentile(chroma_px, p_hi, axis=0) + cm, 0, 255).astype(int)

    last = samples[-1]
    skin = hsv_box(last["hsv"], lo, hi)
    if last["gate"] is not None:
        skin = cv2.bitwise_and(skin, last["gate"])
    _, skin = denoise(skin, 5, 9)
    n, labels = cv2.connectedComponents(skin)

    glove = None
    if tuned is not None:
        glove = hue_grow(samples, roi_mask, roi, lo[0], hi[0], v)
        if glove is not None:
            for s, r in zip(samples, glove):
                hand_hist += hs_hist(s["hsv"], r, bins)

    gx, gy = int(0.1 * w), int(0.1 * h)
    exclude = np.zeros(shape, np.uint8)
    exclude[max(0, y - gy):y + h + gy, max(0, x - gx):x + w + gx] = 255
    reach = np.zeros(shape, np.uint8)
    reach[max(0, y - 3 * h):, max(0, x - 2 * w):x + 3 * w] = 255
    touching = np.unique(labels[exclude > 0])
    attached = (np.isin(labels, touching[touching > 0]) * 255).astype(np.uint8)
    limit = min(v["grow_max_area"] * shape[0] * shape[1], v["grow_hand_ratio"] * cv2.countNonZero(roi_mask))
    exclude = cv2.bitwise_or(exclude, trim(attached, roi, limit))
    if glove is not None:
        exclude = cv2.bitwise_or(exclude, glove[-1])
    hand_zone = cv2.dilate(exclude, np.ones((9, 9), np.uint8))
    early = lookalikes(normalize(hand_hist), samples, hand_zone, v)
    exclude = cv2.bitwise_or(hand_zone, reach)
    if early is not None:
        exclude = cv2.bitwise_or(exclude, early)

    bg_hist = np.zeros(bins, np.float32)
    keep = cv2.bitwise_not(exclude)
    bgs = []
    for s in samples:
        m = keep if s["gate"] is None else cv2.bitwise_and(keep, s["gate"])
        bgs.append(m)
        bg_hist += hs_hist(s["hsv"], m, bins)

    model = ColorModel(normalize(hand_hist), normalize(bg_hist + v["bg_prior"]), tuple(lo), tuple(hi),
                       tuple(chroma_lo), tuple(chroma_hi), v["min_hand_bin"])
    model.core_lo, model.core_hi = tuple(chroma_lo), tuple(chroma_hi)
    if tuned is None:
        model.distractors = merge_maps(early, distractor_map(model, samples, [hand_zone] * len(samples), v, v))
        return model

    tuned = dict(tuned)
    regions = [hand_zone] * len(samples)
    tuned.update(tune_thresholds(model, samples, cores, bgs, v))
    limit = v["grow_max_area"] * shape[0] * shape[1]
    for _ in range(v["grow_passes"]):
        grown_regions = grow_hand(model, samples, roi, tuned, v)
        if np.mean([np.count_nonzero(r) for r in grown_regions]) > limit:
            break
        regions = grown_regions
        grown = np.zeros(bins, np.float32)
        bg_hist = np.zeros(bins, np.float32)
        for i, (s, r) in enumerate(zip(samples, regions)):
            grown += hs_hist(s["hsv"], r, bins)
            bgs[i] = cv2.bitwise_and(bgs[i], cv2.bitwise_not(cv2.dilate(r, ellipse(15))))
            bg_hist += hs_hist(s["hsv"], bgs[i], bins)
        if grown.sum() <= 0:
            break
        model.hand_p = normalize(hand_hist + grown)
        model.bg_p = normalize(bg_hist + v["bg_prior"])
        model.refresh()
        tuned.update(tune_thresholds(model, samples, regions, bgs, v))
    model.tuned = {k: tuned[k] for k in ("backproject_thresh", "backproject_low") if k in tuned}
    grown_px = np.concatenate([s["chroma"][r > 0] for s, r in zip(samples, regions)])
    if len(grown_px) > len(chroma_px):
        model.chroma_lo = tuple(np.clip(np.percentile(grown_px, p_lo, axis=0) - cm, 0, 255).astype(int))
        model.chroma_hi = tuple(np.clip(np.percentile(grown_px, p_hi, axis=0) + cm, 0, 255).astype(int))
    model.distractors = merge_maps(early, distractor_map(model, samples, regions, tuned, v))
    return model
