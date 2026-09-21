"""Orbs, physics and scoring (no drawing). Window px, y down; sizes in screen heights (H)."""
import math
import random
from collections import deque

import numpy as np
import pymunk

STEP = 1 / 60          # physics time step (s)
GRAVITY = 1.6          # H/s^2
ORB_R = 0.06           # orb radius (H)
HIT_TOL = 0.02         # extra hit radius (H)
V_MIN = 0.8            # minimum swing speed to cut (H/s)
LIVES = 3


def segment_hits_circle(p0, p1, r):
    """Does segment p0 -> p1 pass within r of the origin? Returns (hit, closest point)."""
    d = p1 - p0
    t = np.clip(-np.dot(p0, d) / max(np.dot(d, d), 1e-9), 0.0, 1.0)
    q = p0 + t * d                      # closest point to the origin
    return np.dot(q, q) <= r * r, q


class Orb:
    def __init__(self, kind, body, shape):
        self.kind = kind                # "A", "B" or "bomb"
        self.body, self.shape = body, shape
        self.history = deque(maxlen=30)  # (game time, x, y), last 0.5 s

    def pos_at(self, t):
        """Position at game time t (nearest recorded step)."""
        _, x, y = min(self.history, key=lambda s: abs(s[0] - t))
        return np.array([x, y])


