from dataclasses import dataclass, field

import cv2
import numpy as np


def hsv_box(hsv, lo, hi):
    lo = tuple(int(v) for v in lo)
    hi = tuple(int(v) for v in hi)
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    a = cv2.inRange(hsv, (0, lo[1], lo[2]), hi)
    b = cv2.inRange(hsv, lo, (179, hi[1], hi[2]))
    return cv2.bitwise_or(a, b)


def back_project(hsv, hist, blur):
    prob = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    if blur > 1:
        prob = cv2.GaussianBlur(prob, (blur, blur), 0)
    return prob


def hysteresis(prob, low, high):
    weak = (prob >= low).astype(np.uint8)
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    if n <= 1:
        return weak * 255
    keep = np.zeros(n, bool)
    keep[np.unique(labels[prob >= high])] = True
    keep[0] = False
    return (keep[labels] * 255).astype(np.uint8)


def ellipse(k):
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return out


def denoise(mask, open_k, close_k):
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(open_k))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, ellipse(close_k))
    return opened, fill_holes(closed)


def baseline_morph(mask):
    k = np.ones((3, 3), np.uint8)
    return cv2.dilate(cv2.erode(mask, k, iterations=2), k, iterations=2)


class Motion:
    def __init__(self, thresh):
        self.thresh = thresh
        self.prev = None

    def reset(self):
        self.prev = None

    def __call__(self, gray):
        if self.prev is None or self.prev.shape != gray.shape:
            self.prev = gray
            return np.zeros_like(gray)
        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray
        moving = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)[1]
        return cv2.dilate(moving, ellipse(7))


@dataclass
class Candidate:
    contour: np.ndarray
    area: float
    solidity: float
    bbox: tuple
    center: tuple
    blob: np.ndarray = None
    hand_blob: np.ndarray = None
    radius: float = 0.0
    fingers: int = -1
    hand_contour: np.ndarray = None
    mean_prob: float = 1.0
    motion: float = 0.0
    proximity: float = 0.0
    score: float = 0.0
    conf: float = 1.0
    extra: dict = field(default_factory=dict)


def find_candidates(mask, min_blob, min_area, max_area, min_solidity, strict):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_blob:
            continue
        if strict and not (min_area <= area <= max_area):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if strict and solidity < min_solidity:
            continue
        m = cv2.moments(c)
        center = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else tuple(c[0, 0])
        out.append(Candidate(c, area, solidity, cv2.boundingRect(c), center))
    out.sort(key=lambda c: -c.area)
    return out


def blob_mask(cand, shape):
    if cand.blob is None:
        cand.blob = np.zeros(shape, np.uint8)
        cv2.drawContours(cand.blob, [cand.contour], -1, 255, cv2.FILLED)
    return cand.blob


def score_candidates(cands, prob, motion, predictions, gate, weights):
    if not cands:
        return
    largest = max(c.area for c in cands)
    for c in cands:
        blob = blob_mask(c, prob.shape)
        c.mean_prob = cv2.mean(prob, mask=blob)[0] / 255.0
        c.motion = cv2.mean(motion, mask=blob)[0] / 255.0 if motion is not None else 0.0
        if predictions:
            d = min(np.hypot(c.center[0] - p[0], c.center[1] - p[1]) for p in predictions)
            c.proximity = float(np.exp(-(d / gate) ** 2))
        faceness = (1.0 - c.motion) * np.clip((c.solidity - 0.85) / 0.1, 0.0, 1.0)
        c.score = (weights["size"] * c.area / largest + weights["motion"] * c.motion
                   + weights["proximity"] * c.proximity - weights["face"] * faceness)
