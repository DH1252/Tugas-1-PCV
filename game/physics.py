import numpy as np


def integrate(item, dt, gravity, width, e_wall):
    item.vel[1] += gravity * item.gravity * dt
    if item.drag:
        item.vel *= max(0.0, 1.0 - item.drag * dt)
    item.pos += item.vel * dt
    r = item.radius
    if item.pos[0] < r:
        item.pos[0] = r
        item.vel[0] = abs(item.vel[0]) * e_wall
    elif item.pos[0] > width - r:
        item.pos[0] = width - r
        item.vel[0] = -abs(item.vel[0]) * e_wall
    if item.pos[1] < r and item.vel[1] < 0:
        item.pos[1] = r
        item.vel[1] = -item.vel[1] * e_wall


def touching(item, center, radius):
    return np.hypot(*(item.pos - center)) < radius + item.radius


def collide(item, center, hand_vel, radius, p):
    d = item.pos - center
    dist = float(np.hypot(*d))
    min_d = radius + item.radius
    if dist >= min_d:
        return False
    n = d / dist if dist > 1e-6 else np.array([0.0, -1.0])
    rel = item.vel - hand_vel
    vn = float(rel @ n)
    item.pos = center + n * min_d
    if vn >= 0:
        return False
    e = p["restitution_hand"] * item.bounce
    item.vel = hand_vel + rel - (1 + e) * vn * n
    if n[1] < 0:
        item.vel[1] = min(item.vel[1], -p["min_up_speed"] * item.bounce)
    speed = float(np.hypot(*item.vel))
    if speed > p["max_speed"]:
        item.vel *= p["max_speed"] / speed
    return True


def step(items, paddles, dt, p, size, gravity_mult=1.0):
    width, height = size
    events = []
    n = max(1, int(p["substeps"]))
    h = dt / n
    g = p["gravity"] * gravity_mult
    for i in range(n):
        f = (i + 1) / n
        centers = [pd.prev + (pd.pos - pd.prev) * f for pd in paddles]
        for item in items:
            if not item.alive:
                continue
            integrate(item, h, g, width, p["restitution_wall"])
            item.cooldown = max(0.0, item.cooldown - h)
            for pd, c in zip(paddles, centers):
                if item.kind in ("spike", "star"):
                    if touching(item, c, pd.radius):
                        item.alive = False
                        events.append((item.kind, item))
                        break
                elif collide(item, c, pd.vel, pd.radius, p) and item.cooldown <= 0:
                    events.append(("hit", item))
                    item.cooldown = p["hit_cooldown"]
            if item.alive and item.pos[1] - item.radius > height:
                item.alive = False
                events.append(("drop", item))
    return events
