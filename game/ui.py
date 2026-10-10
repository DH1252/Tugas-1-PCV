class Button:
    def __init__(self, label, rect, action):
        self.label = label
        self.rect = rect
        self.action = action
        self.progress = 0.0

    def contains(self, p):
        x, y, w, h = self.rect
        return x <= p[0] <= x + w and y <= p[1] <= y + h


class DwellMenu:
    def __init__(self, buttons, dwell):
        self.buttons = buttons
        self.dwell = dwell

    def reset(self):
        for b in self.buttons:
            b.progress = 0.0

    def update(self, points, dt):
        for b in self.buttons:
            if any(b.contains(p) for p in points):
                b.progress += dt / self.dwell
                if b.progress >= 1.0:
                    self.reset()
                    return b.action
            else:
                b.progress = max(0.0, b.progress - 2 * dt / self.dwell)
        return None


class GestureHold:
    def __init__(self, hold):
        self.hold = hold
        self.value = None
        self.time = 0.0
        self.fired = False

    def reset(self):
        self.value = None
        self.time = 0.0
        self.fired = False

    @property
    def progress(self):
        return 0.0 if self.value is None or self.fired else min(self.time / self.hold, 1.0)

    def update(self, value, dt):
        if value != self.value:
            self.value = value
            self.time = 0.0
            self.fired = False
            return None
        if value is None or self.fired:
            return None
        self.time += dt
        if self.time >= self.hold:
            self.fired = True
            return value
        return None
