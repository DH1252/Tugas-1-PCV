from dataclasses import dataclass

import numpy as np

KEEP_UP = ("ball", "balloon", "heavy")


@dataclass
class Item:
    kind: str
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    gravity: float
    drag: float
    bounce: float
    cooldown: float = 0.0
    alive: bool = True
    hue: int = 0

    @property
    def keep_up(self):
        return self.kind in KEEP_UP


def make_item(kind, pos, vel, spec, hue=0):
    s = spec[kind]
    return Item(kind, np.array(pos, float), np.array(vel, float), float(s["radius"]),
                float(s["gravity"]), float(s["drag"]), float(s["bounce"]), hue=hue)


@dataclass
class Paddle:
    id: int
    prev: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    radius: float
