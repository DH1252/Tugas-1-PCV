import numpy as np

from .entities import make_item


class Spawner:
    def __init__(self, cfg, rng, practice=False):
        self.g = cfg["game"]
        self.rng = rng
        self.practice = practice
        self.reset()

    def reset(self):
        self.elapsed = 0.0
        self.respawn = 0.0
        self.extra = self._extra_interval()

    def _extra_interval(self):
        lo, hi = self.g["extra_item_every"]
        return float(self.rng.uniform(lo, hi))

    def _ramp(self, pair):
        k = min(self.elapsed / self.g["ramp_seconds"], 1.0)
        return pair[0] + (pair[1] - pair[0]) * k

    def max_balls(self):
        if self.practice:
            return 1
        return min(1 + int(self.elapsed // self.g["add_ball_every"]), self.g["max_balls"])

    def gravity_mult(self):
        if self.practice:
            return self.g["practice_gravity"]
        return (1 + self.g["gravity_step"]) ** int(self.elapsed // self.g["gravity_step_every"])

    def on_drop(self):
        self.respawn = max(self.respawn, self.g["respawn_delay"])

    def _spawn(self, kind, width):
        r = self.g["items"][kind]["radius"]
        x = float(self.rng.uniform(r + 40, width - r - 40))
        vx = float(self.rng.uniform(-80, 80))
        return make_item(kind, (x, -r), (vx, 60.0), self.g["items"], hue=int(self.rng.integers(0, 180)))

    def update(self, dt, items, width):
        self.elapsed += dt
        self.respawn -= dt
        new = []
        keep = sum(1 for it in items if it.alive and it.keep_up)
        if keep < self.max_balls() and self.respawn <= 0:
            kind = "ball"
            if not self.practice:
                roll = self.rng.random()
                heavy = self._ramp(self.g["heavy_chance"])
                if roll < heavy:
                    kind = "heavy"
                elif roll < heavy + self.g["balloon_chance"]:
                    kind = "balloon"
            new.append(self._spawn(kind, width))
            self.respawn = self.g["spawn_gap"]
        if not self.practice:
            self.extra -= dt
            if self.extra <= 0:
                kind = "spike" if self.rng.random() < self._ramp(self.g["spike_chance"]) else "star"
                new.append(self._spawn(kind, width))
                self.extra = self._extra_interval()
        return new
