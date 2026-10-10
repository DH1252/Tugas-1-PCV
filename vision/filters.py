from collections import Counter, deque

import cv2
import numpy as np


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t
        if dt <= 0:
            return self.x.copy()
        self.t = t
        ad = self._alpha(self.d_cutoff, dt)
        self.dx = ad * (x - self.x) / dt + (1 - ad) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * np.linalg.norm(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


class Kalman:
    def __init__(self, pos, accel_noise, meas_noise):
        self.kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise ** 2
        self.kf.statePost = np.array([[pos[0]], [pos[1]], [0], [0]], np.float32)
        self.kf.errorCovPost = np.diag([meas_noise ** 2, meas_noise ** 2, 500.0 ** 2, 500.0 ** 2]).astype(np.float32)
        self.q = accel_noise ** 2

    def predict(self, dt):
        f = np.eye(4, dtype=np.float32)
        f[0, 2] = f[1, 3] = dt
        self.kf.transitionMatrix = f
        q = np.zeros((4, 4), np.float32)
        for i in (0, 1):
            q[i, i] = dt ** 4 / 4
            q[i, i + 2] = q[i + 2, i] = dt ** 3 / 2
            q[i + 2, i + 2] = dt ** 2
        self.kf.processNoiseCov = q * self.q
        return self.kf.predict()[:2, 0].astype(float)

    def damp(self, factor):
        self.kf.statePost[2:] *= factor

    def correct(self, pos):
        self.kf.correct(np.array([[pos[0]], [pos[1]]], np.float32))

    @property
    def position(self):
        return self.kf.statePost[:2, 0].astype(float)

    @property
    def velocity(self):
        return self.kf.statePost[2:, 0].astype(float)


class Track:
    def __init__(self, tid, pos, t, tc):
        self.id = tid
        self.tc = tc
        self.kalman = Kalman(pos, tc["accel_noise"], tc["meas_noise"])
        self.euro = OneEuro(**tc["one_euro"])
        self.pred = np.asarray(pos, float)
        self.out = self.euro(pos, t)
        self.raw = tuple(pos)
        self.hits = 1
        self.missed = 0
        self.merged = 0
        self.confirmed = tc["birth"] <= 1
        self.radius = None
        self.votes = deque(maxlen=tc["finger_vote"])
        self.conf = 0.0
        self.contour = None

    def predict(self, dt):
        self.pred = self.kalman.predict(dt)
        return self.pred

    def update(self, pos, t, radius, fingers, conf, contour):
        self.kalman.correct(pos)
        self.out = self.euro(pos, t)
        self.raw = tuple(pos)
        self.hits += 1
        self.missed = 0
        self.merged = 0
        if self.hits >= self.tc["birth"]:
            self.confirmed = True
        a = self.tc["radius_smooth"]
        self.radius = radius if self.radius is None else (1 - a) * self.radius + a * radius
        if fingers >= 0:
            self.votes.append(fingers)
        self.conf = conf
        self.contour = contour

    def miss(self, t):
        self.missed += 1
        self.out = self.euro(self.pred, t)
        self.conf *= 0.7

    def hold(self, t):
        self.merged += 1
        self.kalman.damp(0.5)
        self.out = self.euro(self.pred, t)
        self.conf *= 0.9

    @property
    def fingers(self):
        if not self.votes:
            return -1
        return Counter(self.votes).most_common(1)[0][0]


def assign(cost, gates):
    n_t = len(cost)
    n_c = len(cost[0]) if n_t else 0
    best = [(-1, 0.0), []]

    def rec(i, used, pairs, total):
        if i == n_t:
            key = (len(pairs), -total)
            if key > best[0]:
                best[0], best[1] = key, list(pairs)
            return
        rec(i + 1, used, pairs, total)
        for j in range(n_c):
            if j not in used and cost[i][j] <= gates[i]:
                used.add(j)
                pairs.append((i, j))
                rec(i + 1, used, pairs, total + cost[i][j])
                pairs.pop()
                used.discard(j)

    rec(0, set(), [], 0.0)
    return best[1]
