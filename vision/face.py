import cv2
import numpy as np


class FaceTracker:
    def __init__(self, box, max_jump=1.35, max_shift=0.25):
        self.window = tuple(int(v) for v in box)
        self.max_jump = max_jump
        self.max_shift = max_shift
        self.ref_mass = None
        self.frozen = False

    @property
    def ellipse(self):
        x, y, w, h = self.window
        return (x + w / 2, y + h / 2), (w, h), 0.0

    def _mass(self, prob, window):
        x, y, w, h = window
        return float(prob[max(0, y):y + h, max(0, x):x + w].sum(dtype=np.float64))

    def update(self, prob):
        mass = self._mass(prob, self.window)
        if self.ref_mass is None:
            self.ref_mass = mass
        if self.ref_mass <= 0 or mass > self.max_jump * self.ref_mass:
            self.frozen = True
            return False
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        _, window = cv2.meanShift(prob, self.window, criteria)
        x, y, w, h = self.window
        shift = np.hypot(window[0] - x, window[1] - y) / np.sqrt(w * h)
        new_mass = self._mass(prob, window)
        ok = shift < self.max_shift and new_mass < self.max_jump * self.ref_mass
        self.frozen = not ok
        if ok:
            self.window = tuple(int(v) for v in window)
            self.ref_mass = 0.95 * self.ref_mass + 0.05 * new_mass
        return ok

    def mask(self, shape, margin):
        m = np.zeros(shape, np.uint8)
        (cx, cy), (w, h), angle = self.ellipse
        cv2.ellipse(m, ((cx, cy), (w * (1 + margin), h * (1 + margin)), angle), 255, -1)
        return m
