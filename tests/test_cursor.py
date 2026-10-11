import cv2
import numpy as np

from game.engine import Game
from vision.state import HandState
from vision.tracker import HandTracker

from . import stress
from .synth import GLOVE, draw_hand


def calibrated(cfg):
    tr = HandTracker(cfg)
    stress.calibrate(tr, cfg)
    return tr


def frame(points, size=stress.SIZE, fingers=5):
    img = np.empty((size[1], size[0], 3), np.uint8)
    img[:] = (105, 95, 85)
    for p in points:
        draw_hand(img, p, color=GLOVE, fingers=fingers)
    return img


def test_cursor_sits_on_top_of_blob(cfg_top):
    tr = calibrated(cfg_top)
    for i in range(5):
        states = tr.process(frame([(400, 300)]), i / 30)
    x, y = states[0].palm
    assert abs(x - 400) < 20
    assert 300 - 125 < y < 300 - 70


def test_top_cursor_is_steady_when_fingers_change(cfg_top):
    tr = calibrated(cfg_top)
    xs = []
    for i in range(40):
        states = tr.process(frame([(400, 300)], fingers=(5, 4, 3, 5)[i // 10]), i / 30)
        if i >= 4:
            xs.append(states[0].palm[0])
    assert max(xs) - min(xs) < 25


def test_single_frame_spike_is_ignored(cfg_top):
    tr = calibrated(cfg_top)
    for i in range(10):
        tr.process(frame([(300, 300)]), i / 30)
    before = tr.process(frame([(300, 300)]), 10 / 30)[0].palm
    spike = tr.process(frame([(650, 300)]), 11 / 30)
    assert spike and np.hypot(spike[0].palm[0] - before[0], spike[0].palm[1] - before[1]) < 20
    after = tr.process(frame([(300, 300)]), 12 / 30)[0].palm
    assert np.hypot(after[0] - before[0], after[1] - before[1]) < 20


def test_sustained_move_is_accepted(cfg_top):
    tr = calibrated(cfg_top)
    for i in range(10):
        tr.process(frame([(300, 300)]), i / 30)
    for i in range(10, 16):
        states = tr.process(frame([(420, 300)]), i / 30)
    assert abs(states[0].palm[0] - 420) < 25


def test_coasting_stays_on_screen(cfg_top):
    tr = calibrated(cfg_top)
    for i in range(10):
        states = tr.process(frame([(700 + 15 * i, 300)]), i / 30)
    for i in range(10, 14):
        states = tr.process(frame([]), i / 30)
        for s in states:
            assert 0 <= s.palm[0] < stress.SIZE[0] and 0 <= s.palm[1] < stress.SIZE[1]


def test_inner_region_maps_to_full_screen(cfg_top):
    game = Game(cfg_top, calibration_path=None, log_path=None)
    game.step(frame([]), 0.0)
    w, h = game.size
    mx, my = cfg_top["game"]["cursor_margin"]
    at = lambda x, y: game._to_screen(HandState(0, (x, y), 40)).palm
    assert at(mx * w, my * h) == (0.0, 0.0)
    assert np.allclose(at((1 - mx) * w, (1 - my) * h), (w - 1, h - 1), atol=1)
    assert np.allclose(at(w / 2, h / 2), (w / 2, h / 2))
    assert at(-50, 2 * h) == (0.0, h - 1)


def with_corner(points, seed=0):
    img = frame(points)
    cv2.rectangle(img, (720, 0), (854, 110), GLOVE, -1)
    return img


def test_static_corner_blob_is_not_tracked(cfg_top):
    tr = calibrated(cfg_top)
    for i in range(30):
        states = tr.process(with_corner([]), i / 30)
    assert states == []


def test_lost_hand_does_not_jump_to_far_blob(cfg_top):
    tr = calibrated(cfg_top)
    for i in range(10):
        tr.process(with_corner([(400, 330)]), i / 30)
    for i in range(10, 20):
        for s in tr.process(with_corner([]), i / 30):
            assert s.palm[0] < 650


def test_still_glove_takes_over_from_wrong_lock(cfg_top):
    from vision.filters import Track
    tr = calibrated(cfg_top)
    tr.process(with_corner([]), 0.0)
    wrong = Track(99, (785, 11), 0.0, cfg_top["tracking"])
    wrong.confirmed, wrong.area = True, 3000.0
    tr.tracks = [wrong]
    for i in range(1, 40):
        states = tr.process(with_corner([(400, 330)]), i / 30)
    assert states and abs(states[0].palm[0] - 400) < 30
