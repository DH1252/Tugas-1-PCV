import numpy as np

from game import physics
from game.entities import Paddle, make_item


def ball(cfg, pos, vel=(0, 0)):
    return make_item("ball", pos, vel, cfg["game"]["items"])


def no_gravity(cfg):
    p = dict(cfg["physics"])
    p["gravity"] = 0
    return p


def test_bounce_only_when_approaching(cfg):
    p = cfg["physics"]
    b = ball(cfg, (100, 75), (0, 200))
    assert physics.collide(b, np.array([100.0, 130.0]), np.zeros(2), 40, p)
    assert b.vel[1] < 0

    b = ball(cfg, (100, 75), (0, -200))
    assert not physics.collide(b, np.array([100.0, 130.0]), np.zeros(2), 40, p)
    assert b.vel[1] == -200


def test_restitution_and_min_up_speed(cfg):
    p = cfg["physics"]
    b = ball(cfg, (100, 75), (0, 1000))
    physics.collide(b, np.array([100.0, 130.0]), np.zeros(2), 40, p)
    assert np.isclose(b.vel[1], -1000 * p["restitution_hand"])

    b = ball(cfg, (100, 75), (0, 50))
    physics.collide(b, np.array([100.0, 130.0]), np.zeros(2), 40, p)
    assert np.isclose(b.vel[1], -p["min_up_speed"])


def test_hand_speed_adds_to_bounce_and_is_capped(cfg):
    p = cfg["physics"]
    still = ball(cfg, (100, 75), (0, 300))
    swat = ball(cfg, (100, 75), (0, 300))
    physics.collide(still, np.array([100.0, 130.0]), np.zeros(2), 40, p)
    physics.collide(swat, np.array([100.0, 130.0]), np.array([0.0, -900.0]), 40, p)
    assert swat.vel[1] < still.vel[1]
    assert np.hypot(*swat.vel) <= p["max_speed"] + 1e-6


def test_substeps_stop_tunnelling(cfg):
    p = no_gravity(cfg)
    b = ball(cfg, (320, 200))
    hand = Paddle(0, np.array([320.0, 290.0]), np.array([320.0, 110.0]), np.array([0.0, -5400.0]), 50)
    events = physics.step([b], [hand], 1 / 30, p, (640, 480))
    assert ("hit", b) in events
    assert b.vel[1] < 0

    p["substeps"] = 1
    b = ball(cfg, (320, 200))
    assert physics.step([b], [hand], 1 / 30, p, (640, 480)) == []


def test_fast_hand_from_below_pushes_ball_up(cfg):
    p = no_gravity(cfg)
    b = ball(cfg, (320, 200))
    hand = Paddle(0, np.array([320.0, 240.0]), np.array([320.0, 160.0]), np.array([0.0, -2400.0]), 30)
    physics.step([b], [hand], 1 / 30, p, (640, 480))
    assert b.vel[1] < 0


def test_walls_and_drop(cfg):
    p = cfg["physics"]
    b = ball(cfg, (10, 100), (-300, 0))
    physics.step([b], [], 1 / 30, p, (640, 480))
    assert b.vel[0] > 0 and b.pos[0] >= b.radius

    b = ball(cfg, (320, 490), (0, 900))
    events = physics.step([b], [], 1 / 30, p, (640, 480))
    assert events == [("drop", b)] and not b.alive


def test_spike_and_star_are_touched_not_bounced(cfg):
    p = cfg["physics"]
    items = cfg["game"]["items"]
    spike = make_item("spike", (100, 100), (0, 0), items)
    star = make_item("star", (300, 100), (0, 0), items)
    hands = [Paddle(0, np.array([100.0, 130.0]), np.array([100.0, 130.0]), np.zeros(2), 40),
             Paddle(1, np.array([300.0, 130.0]), np.array([300.0, 130.0]), np.zeros(2), 40)]
    events = physics.step([spike, star], hands, 1 / 30, p, (640, 480))
    assert ("spike", spike) in events and ("star", star) in events
