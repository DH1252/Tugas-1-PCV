import time

import cv2
import numpy as np

from . import hand, segmentation
from .calibration import CalibrationError, build_model, hs_hist
from .face import FaceTracker
from .filters import Track, assign
from .preprocess import WhiteBalance, hue_lut, reliability_gate, rotate_hue, shift_box
from .state import HandState

ENHANCEMENTS = tuple(f"E{i}" for i in range(1, 10))
NAMES = {
    "E1": "white balance",
    "E2": "hue gate",
    "E3": "calibration",
    "E4": "back-projection",
    "E5": "YCrCb",
    "E6": "open/close",
    "E7": "face + scoring",
    "E8": "palm geometry",
    "E9": "temporal",
}
RELOST_FRAMES = 90


def preset(kind):
    return dict.fromkeys(ENHANCEMENTS, kind != "baseline")


def cumulative(k):
    return {e: i < k for i, e in enumerate(ENHANCEMENTS)}


def calib_box(cfg, size):
    w, h = size
    bx, by, bw, bh = cfg["vision"]["calib_box"]
    return int(bx * w), int(by * h), int(bw * w), int(bh * h)


class HandTracker:
    def __init__(self, cfg, hands=1, toggles=None):
        self.cfg = cfg
        self.v = cfg["vision"]
        self.tc = cfg["tracking"]
        self.hands = hands
        self.toggles = preset("enhanced") if toggles is None else dict(toggles)
        self.lut = hue_lut(self.v["hue_shift"])
        self.wb = WhiteBalance(self.v["wb_clip"], self.v["wb_smooth"])
        self.motion = segmentation.Motion(self.v["motion_thresh"])
        self.calib_frames = []
        self.calib_roi = None
        self.model = None
        self.adapted = False
        self.debug_enabled = False
        self.debug = {}
        self.timings = {}
        self.mask = None
        self.hand_mask = None
        self.reset()

    @property
    def label(self):
        on = [k for k in ENHANCEMENTS if self.toggles[k]]
        if not on:
            return "baseline"
        if len(on) == len(ENHANCEMENTS):
            return "enhanced"
        return "+".join(on)

    @property
    def calibrated(self):
        return self.model is not None

    def reset(self):
        self.tracks = []
        self.next_id = 0
        self.prev = {}
        self.last_t = None
        self.frame_count = 0
        self.lost = 0
        self.wb.reset()
        self.motion.reset()
        self.face = None
        if self.model is not None and self.model.face_box is not None:
            self.face = FaceTracker(self.model.face_box, self.v["face_max_jump"], self.v["face_max_shift"])

    def set_toggles(self, toggles):
        rebuild = any(toggles[k] != self.toggles[k] for k in ("E1", "E2"))
        self.toggles = dict(toggles)
        if rebuild and self.calib_frames:
            try:
                self._build_model()
            except CalibrationError:
                self.model = None
        self.reset()

    def set_hands(self, n):
        self.hands = n
        self.reset()

    def calib_box(self, size):
        return calib_box(self.cfg, size)

    def start_calibration(self):
        self.calib_frames = []

    def add_calibration_frame(self, frame):
        self.calib_frames.append(self._small(frame))

    def finish_calibration(self, size, roi=None):
        s = self.v["scale"]
        self.calib_roi = tuple(int(round(c * s)) for c in (roi or self.calib_box(size)))
        self._build_model()
        self.reset()

    def matches(self, frame):
        return bool(self.calib_frames) and self.calib_frames[0].shape == self._small(frame).shape

    def save(self, path):
        np.savez_compressed(path, frames=np.stack(self.calib_frames), roi=np.array(self.calib_roi),
                            scale=self.v["scale"])

    def load(self, path):
        data = np.load(path)
        if float(data["scale"]) != self.v["scale"]:
            raise CalibrationError("saved calibration uses a different processing scale")
        self.calib_frames = list(data["frames"])
        self.calib_roi = tuple(int(c) for c in data["roi"])
        self._build_model()
        self.reset()

    def _build_model(self):
        wb = WhiteBalance(self.v["wb_clip"], self.v["wb_smooth"])
        samples = [self._prepare(f, wb) for f in self.calib_frames]
        shift = self.v["hue_shift"] if self.toggles["E2"] else 0
        self.model = None
        self.model = build_model(samples, self.calib_roi, self.cfg, shift)
        self.adapted = False

    def _small(self, frame):
        s = self.v["scale"]
        return cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def _prepare(self, small, wb):
        bgr = small
        if self.toggles["E1"]:
            k = int(self.v["blur"]) | 1
            if k > 1:
                bgr = cv2.GaussianBlur(bgr, (k, k), 0)
            bgr = wb(bgr)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gate = None
        if self.toggles["E2"]:
            hsv = rotate_hue(hsv, self.lut)
            gate = reliability_gate(hsv, self.v["s_min"], self.v["v_min"], self.v["v_max"])
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        return {"bgr": bgr, "hsv": hsv, "gate": gate, "ycrcb": ycrcb}

    def _fixed_box(self):
        lo, hi = self.v["baseline_lo"], self.v["baseline_hi"]
        return shift_box(lo, hi, self.v["hue_shift"]) if self.toggles["E2"] else (lo, hi)

    def _lap(self, name, clock):
        now = time.perf_counter()
        ms = (now - clock[0]) * 1000
        self.timings[name] = ms if name not in self.timings else 0.9 * self.timings[name] + 0.1 * ms
        clock[0] = now

    def _paddle(self, cand, inv):
        tc = self.tc
        if self.toggles["E8"] and cand.radius > 0:
            return float(np.clip(cand.radius * inv * tc["paddle_scale"], tc["paddle_min"], tc["paddle_max"]))
        return float(tc["baseline_paddle"])

    @staticmethod
    def _scaled(contour, inv):
        return None if contour is None else (contour * inv).astype(np.int32)

    def process(self, frame, t):
        T, v, tc = self.toggles, self.v, self.tc
        dt = 1 / 30 if self.last_t is None else min(max(t - self.last_t, 1e-3), 0.2)
        self.last_t = t
        self.frame_count += 1
        inv = 1.0 / v["scale"]
        start = time.perf_counter()
        clock = [start]

        p = self._prepare(self._small(frame), self.wb)
        hsv, gate = p["hsv"], p["gate"]
        self._lap("preprocess", clock)

        model = self.model if T["E3"] else None
        if model is not None and T["E4"]:
            prob = segmentation.back_project(hsv, model.ratio, v["backproject_blur"])
            mask = cv2.threshold(prob, v["backproject_thresh"], 255, cv2.THRESH_BINARY)[1]
        else:
            lo, hi = (model.lo, model.hi) if model is not None else self._fixed_box()
            mask = segmentation.hsv_box(hsv, lo, hi)
            prob = mask
        if gate is not None:
            mask = cv2.bitwise_and(mask, gate)
            prob = cv2.bitwise_and(prob, gate)
        if T["E5"]:
            if model is not None:
                lo, hi = model.crcb_lo, model.crcb_hi
            else:
                lo, hi = v["ycrcb_default_lo"][1:], v["ycrcb_default_hi"][1:]
            ycc = cv2.inRange(p["ycrcb"], (0, int(lo[0]), int(lo[1])), (255, int(hi[0]), int(hi[1])))
            mask = cv2.bitwise_and(mask, ycc)
        self._lap("segment", clock)

        if T["E6"]:
            light, mask = segmentation.denoise(mask, v["open_kernel"], v["close_kernel"])
        else:
            mask = segmentation.baseline_morph(mask)
            light = mask
        self._lap("denoise", clock)

        if T["E9"]:
            for tr in self.tracks:
                tr.predict(dt)
        preds = [tuple(tr.pred * v["scale"]) for tr in self.tracks] if T["E9"] else []
        motion = None
        if T["E7"]:
            if self.face is not None:
                self.face.update(prob)
                keep = cv2.bitwise_not(self.face.mask(mask.shape, v["face_margin"]))
                mask = cv2.bitwise_and(mask, keep)
                light = cv2.bitwise_and(light, keep)
            motion = self.motion(cv2.cvtColor(p["bgr"], cv2.COLOR_BGR2GRAY))
        n_px = mask.shape[0] * mask.shape[1]
        cands = segmentation.find_candidates(mask, v["min_blob"] * n_px, v["min_area"] * n_px,
                                             v["max_area"] * n_px, v["min_solidity"], T["E7"])
        if T["E7"]:
            segmentation.score_candidates(cands, prob, motion, preds, tc["gate"] * v["scale"], v["score_weights"])
            cands.sort(key=lambda c: -c.score)
        cands = cands[:self.hands + (2 if T["E9"] else 0)]
        for c in cands:
            segmentation.blob_mask(c, mask.shape)
            if not T["E7"]:
                c.mean_prob = cv2.mean(prob, mask=c.blob)[0] / 255.0
            c.conf = 0.5 * c.mean_prob + 0.3 * float(np.clip((c.solidity - 0.45) / 0.45, 0, 1))
        if T["E8"]:
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            for c in cands:
                hand.measure(c, dist, light, v)
        self._lap("contours", clock)

        states, used = self._track(cands, t, inv) if T["E9"] else self._raw(cands, dt, inv)
        self._lap("track", clock)

        self.mask = mask
        self.hand_mask = np.zeros_like(mask)
        for c in used:
            self.hand_mask = cv2.bitwise_or(self.hand_mask, c.hand_blob if c.hand_blob is not None else c.blob)
        self._adapt(used, hsv, gate)
        self._lap("total", [start])
        if self.debug_enabled:
            self._debug(p, gate, prob, mask, motion, cands, used)
        return states

    def _raw(self, cands, dt, inv):
        chosen = cands[:self.hands]
        if self.hands > 1:
            chosen = sorted(chosen, key=lambda c: c.center[0])
        states, prev = [], {}
        for i, c in enumerate(chosen):
            pos = (c.center[0] * inv, c.center[1] * inv)
            last = self.prev.get(i)
            vel = ((pos[0] - last[0]) / dt, (pos[1] - last[1]) / dt) if last else (0.0, 0.0)
            prev[i] = pos
            c.conf += 0.2
            states.append(HandState(i, pos, self._paddle(c, inv), vel, c.fingers, c.conf, True, pos,
                                    self._scaled(c.contour, inv)))
        self.prev = prev
        return states, chosen

    def _track(self, cands, t, inv):
        tc = self.tc
        centers = [(c.center[0] * inv, c.center[1] * inv) for c in cands]
        cost = [[float(np.hypot(tr.pred[0] - x, tr.pred[1] - y)) for x, y in centers] for tr in self.tracks]
        gates = [tc["gate"] * (1 + 0.5 * tr.missed) for tr in self.tracks]
        pairs = assign(cost, gates)
        matched_t = {i for i, _ in pairs}
        matched_c = {j for _, j in pairs}
        for i, j in pairs:
            tr, c = self.tracks[i], cands[j]
            c.conf += 0.2 * (1 - min(cost[i][j] / gates[i], 1))
            tr.update(centers[j], t, self._paddle(c, inv), c.fingers, c.conf, self._scaled(c.contour, inv))
        s = self.v["scale"]
        merged_blobs = [cands[j].blob for _, j in pairs]
        for i, tr in enumerate(self.tracks):
            if i in matched_t:
                continue
            x, y = int(tr.pred[0] * s), int(tr.pred[1] * s)
            inside = any(0 <= y < b.shape[0] and 0 <= x < b.shape[1] and b[y, x] for b in merged_blobs)
            if tr.confirmed and inside:
                tr.hold(t)
            else:
                tr.miss(t)
        used = [cands[j] for i, j in pairs if self.tracks[i].confirmed]
        self.tracks = [tr for tr in self.tracks
                       if (tr.confirmed and tr.missed <= tc["max_coast"] and tr.merged <= tc["max_merge"])
                       or (not tr.confirmed and tr.missed == 0)]
        for j, c in enumerate(cands):
            if j in matched_c or len(self.tracks) >= self.hands:
                continue
            tr = Track(self.next_id, centers[j], t, tc)
            self.next_id += 1
            tr.radius = self._paddle(c, inv)
            tr.conf = c.conf + 0.2
            tr.contour = self._scaled(c.contour, inv)
            if c.fingers >= 0:
                tr.votes.append(c.fingers)
            self.tracks.append(tr)
        states = [HandState(tr.id, tuple(tr.out), tr.radius, tuple(tr.kalman.velocity), tr.fingers,
                            tr.conf, tr.missed == 0 and tr.merged == 0, tr.raw, tr.contour)
                  for tr in self.tracks if tr.confirmed]
        return states, used

    def _adapt(self, used, hsv, gate):
        T, tc, v = self.toggles, self.tc, self.v
        if not (T["E3"] and T["E4"] and T["E9"]) or self.model is None:
            return
        self.lost = 0 if used else self.lost + 1
        if self.lost == RELOST_FRAMES and self.adapted:
            self._build_model()
            return
        if self.frame_count % tc["adapt_every"]:
            return
        best = max(used, key=lambda c: c.conf, default=None)
        if best is None or best.conf < tc["adapt_conf"]:
            return
        m = cv2.erode(best.blob, segmentation.ellipse(7))
        if gate is not None:
            m = cv2.bitwise_and(m, gate)
        self.model.adapt(hs_hist(hsv, m, (v["h_bins"], v["s_bins"])), tc["adapt_alpha"])
        self.adapted = True

    def _debug(self, p, gate, prob, mask, motion, cands, used):
        s = self.v["scale"]
        vis = p["bgr"].copy()
        cv2.drawContours(vis, [c.contour for c in cands], -1, (0, 200, 255), 1)
        for c in used:
            if c.hand_contour is not None:
                hull = cv2.convexHull(c.hand_contour)
                cv2.drawContours(vis, [hull], -1, (255, 200, 0), 1)
                cv2.drawContours(vis, [c.hand_contour], -1, (0, 255, 0), 1)
            if c.radius > 0:
                cv2.circle(vis, (int(c.center[0]), int(c.center[1])), int(c.radius), (0, 255, 0), 1)
            cv2.circle(vis, (int(c.center[0]), int(c.center[1])), 2, (0, 255, 0), -1)
        if self.face is not None:
            color = (0, 0, 255) if self.face.frozen else (255, 0, 255)
            cv2.ellipse(vis, self.face.ellipse, color, 1)
        for tr in self.tracks:
            cv2.circle(vis, (int(tr.pred[0] * s), int(tr.pred[1] * s)), 6, (255, 255, 255), 1)
            cv2.circle(vis, (int(tr.out[0] * s), int(tr.out[1] * s)), 3, (255, 255, 255), -1)
        blank = np.zeros_like(mask)
        self.debug = {
            "input": p["bgr"],
            "gate": gate if gate is not None else blank + 255,
            "probability": prob,
            "mask": mask,
            "motion": motion if motion is not None else blank,
            "result": vis,
        }
