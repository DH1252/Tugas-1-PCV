import cv2
import numpy as np


def palm_center(dist, blob, bbox, top_bias):
    x, y, w, h = bbox
    d = dist[y:y + h, x:x + w] * (blob[y:y + h, x:x + w] > 0)
    if top_bias:
        d = d * (1 - top_bias * np.linspace(0, 1, h, dtype=np.float32))[:, None]
    _, _, _, (px, py) = cv2.minMaxLoc(d)
    return (x + px, y + py), float(dist[y + py, x + px])


def hand_region(mask, center, r, arm_cut, wrist_cut):
    region = np.zeros_like(mask)
    cv2.circle(region, (int(center[0]), int(center[1])), int(arm_cut * r), 255, -1)
    region[int(center[1] + wrist_cut * r) + 1:, :] = 0
    return cv2.bitwise_and(mask, region)


def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def count_fingers(contour, center, r, depth_ratio, max_angle, fist_ratio):
    if contour is None or len(contour) < 5 or r <= 0:
        return -1
    hull = cv2.convexHull(contour, returnPoints=False)
    pts = contour[:, 0, :].astype(np.float32)
    far = float(np.max(np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])))
    gaps = 0
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        defects = None
    if defects is not None:
        for s, e, f, depth in defects.reshape(-1, 4):
            if depth / 256.0 < depth_ratio * r:
                continue
            ps, pe, pf = pts[s], pts[e], pts[f]
            a, b = ps - pf, pe - pf
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
            if np.degrees(np.arccos(np.clip(cos, -1, 1))) > max_angle:
                continue
            if pf[1] > center[1] + r:
                continue
            gaps += 1
    if gaps:
        return min(gaps + 1, 5)
    return 0 if far < fist_ratio * r else 1


def measure(cand, dist, light, v):
    center, r = palm_center(dist, cand.blob, cand.bbox, v["palm_top_bias"])
    cand.center, cand.radius = center, r
    cand.hand_blob = hand_region(cand.blob, center, r, v["arm_cut"], v["wrist_cut"])
    near = cv2.bitwise_and(light, cv2.dilate(cand.blob, np.ones((5, 5), np.uint8)))
    region = hand_region(near, center, r, v["arm_cut"], v["wrist_cut"])
    cand.hand_contour = largest_contour(region)
    cand.fingers = count_fingers(cand.hand_contour, center, r, v["defect_depth"],
                                 v["defect_angle"], v["fist_ratio"])
