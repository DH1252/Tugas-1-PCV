import cv2
import numpy as np

from eval import evaluate
from game.engine import Game
from game.spawner import Spawner
from game.ui import DwellMenu, Button, GestureHold

from .clips import write_clip
from .synth import GLOVE, calibration_frames, make_frame


class Player:
    def __init__(self, game, pos):
        self.game = game
        self.hand = np.array(pos, float)
        self.t = 0.0
        self.i = 0

    def step(self, key=-1):
        out = self.game.step(make_frame([{"center": tuple(self.hand), "color": GLOVE}], seed=self.i), self.t, key)
        self.t += 1 / 30
        self.i += 1
        return out

    def chase(self):
        balls = [it for it in self.game.items if it.keep_up and it.pos[1] > 0]
        target = np.array([320.0, 380.0])
        if balls:
            b = max(balls, key=lambda it: it.pos[1])
            target = np.array([b.pos[0], np.clip(b.pos[1] + 90, 200, 430)])
        step = target - self.hand
        d = np.hypot(*step)
        self.hand += step if d < 36 else step / d * 36
        self.hand[0] = np.clip(self.hand[0], 60, 580)


def start(cfg, tmp_path, debug=False):
    game = Game(cfg, debug=debug, calibration_path=tmp_path / "calib.npz", log_path=tmp_path / "log.csv", seed=3)
    _, center = calibration_frames(cfg, n=1)
    player = Player(game, center)
    for _ in range(5):
        player.step()
    assert game.calib_phase == "wait"
    player.step(32)
    for _ in range(cfg["vision"]["calib_frames"] + 10):
        player.step()
    return game, player


def test_calibrate_menu_play(cfg, tmp_path):
    game, player = start(cfg, tmp_path, debug=True)
    assert game.state == "CALIBRATE" and game.calib_phase == "review"
    view, debug = player.step(13)
    assert game.state == "MENU" and (tmp_path / "calib.npz").exists()
    assert view.shape == (480, 640, 3) and debug is not None
    player.step(13)
    assert game.state == "PLAY"
    for _ in range(30 * 12):
        player.chase()
        player.step()
    assert game.hits > 5
    assert game.state in ("PLAY", "GAME_OVER")


def test_saved_calibration_skips_to_menu(cfg, tmp_path):
    start(cfg, tmp_path)[1].step(13)
    game = Game(cfg, calibration_path=tmp_path / "calib.npz", log_path=None)
    game.step(make_frame([]), 0.0)
    assert game.state == "MENU"


def test_auto_pause_when_hand_lost(cfg, tmp_path):
    game, player = start(cfg, tmp_path)
    player.step(13)
    player.step(13)
    for i in range(60):
        game.step(make_frame([], seed=i), player.t)
        player.t += 1 / 30
    assert game.state == "PAUSE"


def test_practice_logs_result(cfg, tmp_path):
    cfg["game"]["practice_seconds"] = 3
    game, player = start(cfg, tmp_path)
    player.step(13)
    player.step(ord("t"))
    assert game.state == "PLAY" and game.practice
    for _ in range(4 * 30):
        player.chase()
        player.step()
    assert game.state == "GAME_OVER"
    assert (tmp_path / "log.csv").read_text().startswith("time,tracker")


def test_keys_toggle_tracker(cfg, tmp_path):
    game, player = start(cfg, tmp_path)
    player.step(ord("b"))
    assert game.tracker.label == "baseline"
    player.step(ord("b"))
    assert game.tracker.label == "enhanced"
    player.step(ord("5"))
    assert not game.tracker.toggles["E5"]
    player.step(ord("h"))
    assert game.tracker.hands == 2


def test_spawner_ramp(cfg):
    s = Spawner(cfg, np.random.default_rng(0))
    assert s.max_balls() == 1
    s.elapsed = 200
    assert s.max_balls() == cfg["game"]["max_balls"]
    assert s.gravity_mult() > 1


def test_dwell_and_gesture_hold():
    menu = DwellMenu([Button("a", (0, 0, 10, 10), "go")], 1.0)
    assert [menu.update([(5, 5)], 0.25) for _ in range(4)][-1] == "go"
    hold = GestureHold(0.5)
    fired = [hold.update("fist", 0.1) for _ in range(10)]
    assert fired.count("fist") == 1


def test_evaluate_harness(cfg, tmp_path):
    write_clip(tmp_path / "clips", cfg, frames=90, still=(60, 80))
    evaluate.main(["--clips", str(tmp_path / "clips"), "--out", str(tmp_path / "out"),
                   "--practice-log", str(tmp_path / "none.csv")])
    text = (tmp_path / "out" / "ablation.md").read_text()
    assert "| baseline |" in text and "| +E9 |" in text
    rows = (tmp_path / "out" / "results.csv").read_text().splitlines()
    assert len(rows) == 1 + 1 + len(evaluate.ENHANCEMENTS)


def test_calibration_waits_for_space(cfg, tmp_path):
    game = Game(cfg, calibration_path=None, log_path=None)
    _, center = calibration_frames(cfg, n=1)
    player = Player(game, center)
    for _ in range(200):
        player.step()
    assert game.state == "CALIBRATE" and game.calib_phase == "wait"


def test_auto_countdown_option(cfg):
    cfg["game"]["calib_manual"] = False
    game = Game(cfg, calibration_path=None, log_path=None)
    _, center = calibration_frames(cfg, n=1)
    player = Player(game, center)
    for _ in range(int(cfg["game"]["calib_countdown"] * 30) + cfg["vision"]["calib_frames"] + 10):
        player.step()
    assert game.calib_phase == "review"


def test_drag_sample_box(cfg):
    game = Game(cfg, calibration_path=None, log_path=None)
    Player(game, (100, 100)).step()
    game.on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 300)
    game.on_mouse(cv2.EVENT_MOUSEMOVE, 160, 380)
    game.on_mouse(cv2.EVENT_LBUTTONUP, 160, 380)
    assert game.sample_box == (100, 300, 60, 80)
    game.step(make_frame([]), 1.0, ord("r"))
    assert game.sample_box == game.tracker.calib_box(game.size)


def test_ticks_between_camera_frames_extrapolate(cfg, tmp_path):
    game, player = start(cfg, tmp_path)
    player.step(13)
    player.step(13)
    assert game.state == "PLAY"
    for _ in range(5):
        player.hand += (6, 0)
        player.step()
    base = game.hands[0].palm[0]
    frame = make_frame([{"center": tuple(player.hand), "color": GLOVE}], seed=0)
    calls = []
    real = game.tracker.process
    game.tracker.process = lambda *a: calls.append(1) or real(*a)
    game.step(frame, player.t + 1 / 60, fresh=False)
    assert not calls
    assert game.hands and game.hands[0].palm[0] > base


def test_play_at_60_with_camera_at_30(cfg, tmp_path):
    game, player = start(cfg, tmp_path)
    player.step(13)
    player.step(13)
    frame = None
    for i in range(60 * 10):
        if i % 2 == 0:
            player.chase()
            frame = make_frame([{"center": tuple(player.hand), "color": GLOVE}], seed=i)
        game.step(frame, player.t, fresh=i % 2 == 0)
        player.t += 1 / 60
    assert game.hits > 3
    assert 50 < game.fps < 70 and 25 < game.cam_fps < 35
