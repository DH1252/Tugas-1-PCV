from dataclasses import dataclass

import numpy as np


@dataclass
class HandState:
    id: int
    palm: tuple
    radius: float
    velocity: tuple = (0.0, 0.0)
    fingers: int = -1
    confidence: float = 1.0
    found: bool = True
    raw: tuple | None = None
    contour: np.ndarray | None = None

    @property
    def is_fist(self):
        return self.fingers == 0

    @property
    def is_open(self):
        return self.fingers >= 4
