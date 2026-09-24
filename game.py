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


class Game:
    """One round: a saber cuts only its own color; bombs and missed orbs cost a life."""

    def __init__(self, w, h, sabers, seed=None):
        self.w, self.h = w, h
        self.sabers = list(sabers)
        self.rng = random.Random(seed)
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY * h)
        self.orbs = []
        self.score, self.lives = 0, LIVES
        self.time = 0.0                 # game time (s)
        self.next_launch = 1.0
        self.last = {}                  # saber -> (game time, position) of its previous sample
        self.events = []                # (kind, x, y) for popups
        self.practice = False           # no lives lost

    @property
    def over(self):
        return self.lives <= 0

    def advance(self, dt):
        """Run physics for dt seconds in fixed steps."""
        end = self.time + min(dt, 0.1)  # skip ahead after a long hiccup
        while self.time + STEP <= end:
            self.time += STEP
            self.space.step(STEP)
            for orb in self.orbs:
                orb.history.append((self.time, *orb.body.position))
            if self.time >= self.next_launch:
                self.launch()
        for orb in list(self.orbs):    # fell out the bottom
            if orb.body.position.y > self.h + orb.shape.radius and orb.body.velocity.y > 0:
                self.remove(orb)
                if orb.kind != "bomb":
                    self.lose_life()
                    self.events.append(("miss", orb.body.position.x, self.h - 60))

    def launch(self, kind=None):
        """Throw an orb up from below; launches speed up over time, bombs start after 20 s."""
        t = self.time
        self.next_launch = t + max(1.2 - 0.007 * t, 0.6)
        if kind is None:
            bomb = t > 20 and self.rng.random() < 0.15
            kind = "bomb" if bomb else self.rng.choice(self.sabers)
        r = ORB_R * self.h
        g = GRAVITY * self.h
        x = self.rng.uniform(0.15, 0.85) * self.w
        rise = (self.rng.uniform(0.45, 0.75) + ORB_R) * self.h   # height (px)
        flight = 2 * math.sqrt(2 * rise / g)                      # time up and down (s)
        body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, r))
        body.position = (x, self.h + r)
        # v = sqrt(2 g h) upward, drifting toward the middle.
        body.velocity = ((self.w / 2 - x) / flight * self.rng.uniform(0.2, 0.6), -math.sqrt(2 * g * rise))
        shape = pymunk.Circle(body, r)
        shape.filter = pymunk.ShapeFilter(group=1)    # same group: no orb-orb collisions
        self.space.add(body, shape)
        orb = Orb(kind, body, shape)
        orb.history.append((t, *body.position))
        self.orbs.append(orb)
        return orb

    def lose_life(self):
        if not self.practice:
            self.lives -= 1

    def remove(self, orb):
        self.space.remove(orb.body, orb.shape)
        self.orbs.remove(orb)

    def swing(self, key, saber, t):
        """Test the saber's path since its last sample against each orb at capture time t."""
        if saber.status == "searching":
            self.last.pop(key, None)
            return
        p1 = np.array([saber.pos[0] * self.w, saber.pos[1] * self.h])
        prev = self.last.get(key)
        self.last[key] = (t, p1)
        if prev is None or t - prev[0] > 0.2 or math.hypot(*saber.vel) < V_MIN:
            return
        t0, p0 = prev
        for orb in list(self.orbs):
            # Saber path relative to the orb at t0 and t.
            hit, _ = segment_hits_circle(p0 - orb.pos_at(t0), p1 - orb.pos_at(t),
                                         orb.shape.radius + HIT_TOL * self.h)
            if hit:
                self.hit(orb, key)

    def hit(self, orb, key):
        x, y = orb.body.position
        if orb.kind == "bomb":
            self.remove(orb)
            self.lose_life()
            self.events.append(("bomb", x, y))
        elif orb.kind == key:
            self.remove(orb)
            self.score += 10
            self.events.append(("slice", x, y))
        else:
            self.events.append(("wrong", x, y))   # wrong color: the orb stays
