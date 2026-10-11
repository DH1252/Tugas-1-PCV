import numpy as np

from game.entities import Paddle, make_item
from game.physics import World

DT = 1 / 60
SIZE = (640, 480)


def world(cfg, gravity=True):
    p = dict(cfg["physics"])
    if not gravity:
        p["gravity"] = 0
    return World(p, SIZE)


def ball(cfg, pos, vel=(0, 0), kind="ball"):
    return make_item(kind, pos, vel, cfg["game"]["items"])


def hand(x, y, prev=None, radius=40, hid=0):
    pos = np.array([x, y], float)
    return Paddle(hid, pos if prev is None else np.array(prev, float), pos, np.zeros(2), radius)


def run(w, paddles, frames, dt=DT):
    events = []
    for _ in range(frames):
        events += w.step(dt, paddles)
        for pd in paddles:
            pd.prev = pd.pos.copy()
    return events


def test_ball_bounces_off_still_hand(cfg):
    w = world(cfg)
    b = ball(cfg, (320, 200))
    w.add(b)
    events = run(w, [hand(320, 330)], 40)
    assert ("hit", b) in events
    assert b.vel[1] < 0 or b.pos[1] < 250


def test_gentle_touch_gives_min_up_speed(cfg):
    w = world(cfg, gravity=False)
    b = ball(cfg, (320, 240), (0, 60))
    w.add(b)
    run(w, [hand(320, 305)], 10)
    assert b.vel[1] <= -0.9 * cfg["physics"]["min_up_speed"]


def test_swat_launches_faster(cfg):
    speeds = []
    for move in (0, -25):
        w = world(cfg, gravity=False)
        b = ball(cfg, (320, 240), (0, 200))
        w.add(b)
        pd = hand(320, 320)
        fastest = 0.0
        for _ in range(8):
            pd.pos = pd.pos + np.array([0.0, move])
            w.step(DT, [pd])
            pd.prev = pd.pos.copy()
            fastest = max(fastest, -b.vel[1])
        speeds.append(fastest)
    assert speeds[1] > speeds[0]
    assert max(speeds) <= cfg["physics"]["max_speed"] + 1


def test_fast_hand_does_not_tunnel(cfg):
    w = world(cfg, gravity=False)
    b = ball(cfg, (320, 200))
    w.add(b)
    pd = hand(320, 120, prev=(320, 290), radius=30)
    w.step(DT, [pd])
    assert b.vel[1] < 0


def test_walls_and_drop(cfg):
    w = world(cfg)
    b = ball(cfg, (30, 100), (-600, 0))
    w.add(b)
    run(w, [], 10)
    assert b.vel[0] > 0 and b.pos[0] >= b.radius - 1

    w = world(cfg)
    b = ball(cfg, (320, 470), (0, 900))
    w.add(b)
    events = run(w, [], 10)
    assert ("drop", b) in events and b not in w.items


def test_spike_and_star_touch_without_bounce(cfg):
    w = world(cfg, gravity=False)
    spike = ball(cfg, (100, 100), (0, 50), "spike")
    star = ball(cfg, (300, 100), (0, 50), "star")
    w.add(spike)
    w.add(star)
    events = run(w, [hand(100, 150, hid=0), hand(300, 150, hid=1)], 20)
    assert ("spike", spike) in events and ("star", star) in events
    assert not w.items


def test_hand_bodies_follow_tracks(cfg):
    w = world(cfg)
    w.step(DT, [hand(100, 100, hid=3), hand(200, 100, hid=4)])
    assert set(w.hands) == {3, 4}
    w.step(DT, [hand(100, 100, hid=3, radius=60)])
    assert set(w.hands) == {3} and abs(w.hands[3][1].radius - 60) < 1e-6
