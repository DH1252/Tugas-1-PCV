import csv
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from vision.calibration import CalibrationError
from vision.tracker import ENHANCEMENTS, NAMES, HandTracker, preset

from . import render
from .entities import Paddle
from .physics import World
from .spawner import Spawner
from .ui import Button, DwellMenu, GestureHold

CONFIRM_KEYS = (10, 13, 32)
QUIT_KEYS = (27, ord("q"))
HANDLERS = {"CALIBRATE": "_calibrate", "MENU": "_menu", "PLAY": "_play", "PAUSE": "_pause_state",
            "GAME_OVER": "_game_over"}


class Game:
    def __init__(self, cfg, hands=1, baseline=False, debug=False, calibration_path="calibration.npz",
                 log_path="practice_log.csv", seed=None, on_calibrated=None, load_calibration=True):
        self.cfg = cfg
        self.g = cfg["game"]
        self.p = cfg["physics"]
        self.tracker = HandTracker(cfg, hands=hands)
        self.enhanced_toggles = preset("enhanced")
        if baseline:
            self.tracker.set_toggles(preset("baseline"))
        self.debug = debug
        self.tracker.debug_enabled = debug
        self.rng = np.random.default_rng(seed)
        self.calib_path = Path(calibration_path) if calibration_path else None
        self.log_path = Path(log_path) if log_path else None
        self.on_calibrated = on_calibrated
        self.size = None
        self.last_t = None
        self.now = 0.0
        self.fps = 0.0
        self.cam_fps = 0.0
        self.track_t = None
        self.world = None
        self.quit = False
        self.hands = []
        self.paddles = []
        self.prev_pos = {}
        self.best = 0
        self.toast_text, self.toast_until = "", 0.0
        self.popups = []
        self.hold = GestureHold(self.g["gesture_hold"])
        self.practice = False
        self.items = []
        self.state = "CALIBRATE"
        self.sample_box = None
        self.drag = None
        if load_calibration and self.calib_path and self.calib_path.exists():
            try:
                self.tracker.load(self.calib_path)
                self.state = "MENU"
            except (CalibrationError, OSError, KeyError, ValueError):
                pass

    def _layout(self):
        w, h = self.size
        bw, bh = 300, 56
        x = w // 2 - bw // 2

        def stack(labels, top):
            return [Button(label, (x, int(top + i * (bh + 16)), bw, bh), action) for i, (label, action) in enumerate(labels)]

        dwell = self.g["dwell_time"]
        self.menu = DwellMenu(stack([("1  Play", "play"), ("2  Practice", "practice"), ("3  Recalibrate", "calibrate")],
                                    0.36 * h), dwell)
        self.pause_menu = DwellMenu(stack([("Resume", "resume"), ("Menu", "menu")], 0.42 * h), dwell)
        self.over_menu = DwellMenu(stack([("Play again", "again"), ("Menu", "menu")], 0.56 * h), dwell)
        half = 170
        self.review_menu = DwellMenu([Button("OK", (w // 2 - half - 10, int(0.8 * h), half, bh), "ok"),
                                      Button("Retry", (w // 2 + 10, int(0.8 * h), half, bh), "retry")], dwell)
        self.pause_button = DwellMenu([Button("II", (12, 64, 56, 56), "pause")], self.g["pause_dwell"])

    def _enter(self, state):
        self.state = state
        self.hold.reset()
        if state == "CALIBRATE":
            self.calib_phase = "wait"
            self.phase_t = 0.0
            self.review_menu.reset()
        elif state == "MENU":
            self.menu.reset()
        elif state == "PAUSE":
            self.pause_menu.reset()
        elif state == "GAME_OVER":
            self.over_menu.reset()
        elif state == "PLAY":
            self.no_hand = 0.0
            self.pause_button.reset()

    def _toast(self, msg, seconds=2.0):
        self.toast_text, self.toast_until = msg, self.now + seconds

    def step(self, frame, t, key=-1, fresh=True):
        if self.size is None:
            self.size = (frame.shape[1], frame.shape[0])
            if self.state == "MENU" and not self.tracker.matches(frame):
                self.state = "CALIBRATE"
            self.sample_box = self.tracker.calib_box(self.size)
            self._layout()
            self._enter(self.state)
        if self.last_t is not None and t > self.last_t:
            inst = 1.0 / (t - self.last_t)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        dt = 1 / 60 if self.last_t is None else min(max(t - self.last_t, 0.0), 0.05)
        self.last_t = self.now = t
        self._key(key)

        if self.state == "CALIBRATE" and self.calib_phase == "sample":
            if fresh:
                self.tracker.add_calibration_frame(frame)
            self.hands = []
        elif fresh:
            if self.track_t is not None and t > self.track_t:
                inst = 1.0 / (t - self.track_t)
                self.cam_fps = inst if self.cam_fps == 0 else 0.9 * self.cam_fps + 0.1 * inst
            self.measured = self.tracker.process(frame, t)
            self.track_t = t
            self.hands = self.measured
        else:
            self.hands = self._extrapolate(t)
        self.paddles = self._paddles()
        getattr(self, HANDLERS[self.state])(dt)
        self.prev_pos = {h.id: np.array(h.palm, float) for h in self.hands}

        view = render.dim(frame, 0.45 if self.state in ("MENU", "PAUSE", "GAME_OVER") else 0.6)
        overlay = view.copy()
        render.paddle_fill(overlay, self.hands)
        render.blend(view, overlay, 0.35)
        self._draw(view)
        debug_view = render.debug_grid(self.tracker.debug, self.tracker.timings, self._tune_info()) if self.debug else None
        return view, debug_view

    def _extrapolate(self, t):
        if self.track_t is None:
            return []
        ahead = min(max(t - self.track_t, 0.0), self.g["max_extrapolate"])
        out = []
        for h in getattr(self, "measured", []):
            vx, vy = h.velocity
            out.append(replace(h, palm=(h.palm[0] + vx * ahead, h.palm[1] + vy * ahead)))
        return out

    def _tune_info(self):
        t = self.tracker
        mode = "auto" if t.auto_tune and t.tuned else "manual"
        manual = f"  (slider: {', '.join(sorted(t.overrides))})" if t.overrides else ""
        return (f"{mode}: threshold {t.param('backproject_thresh')}/{t.param('backproject_low')}  "
                f"S>={t.param('s_min')} V>={t.param('v_min')}  open {t.param('open_kernel')} "
                f"close {t.param('close_kernel')}{manual}")

    def _paddles(self):
        out = []
        limit = self.p["max_hand_speed"]
        for h in self.hands:
            pos = np.array(h.palm, float)
            vel = np.array(h.velocity, float)
            speed = float(np.hypot(*vel))
            if speed > limit:
                vel *= limit / speed
            out.append(Paddle(h.id, self.prev_pos.get(h.id, pos), pos, vel, h.radius))
        return out

    def _points(self):
        return [h.palm for h in self.hands]

    def _gesture(self):
        best = max(self.hands, key=lambda h: h.confidence, default=None)
        if best is None or best.fingers < 0:
            return None
        if best.fingers == 0:
            return "fist"
        if best.fingers >= 4:
            return "open"
        return best.fingers

    def _key(self, key):
        if key is None or key < 0:
            return
        ch = chr(key) if key < 256 else ""
        if key in QUIT_KEYS:
            self.quit = True
        elif ch == "c":
            self._enter("CALIBRATE")
        elif ch == "d":
            self.debug = not self.debug
            self.tracker.debug_enabled = self.debug
            self.tracker.debug = {}
        elif ch == "b":
            if self.tracker.label == "baseline":
                self.tracker.set_toggles(self.enhanced_toggles)
            else:
                self.enhanced_toggles = dict(self.tracker.toggles)
                self.tracker.set_toggles(preset("baseline"))
            self._toast(f"tracker: {self.tracker.label}")
        elif ch == "h":
            self.tracker.set_hands(3 - self.tracker.hands)
            self._toast(f"{self.tracker.hands} hand mode")
        elif ch and ch in "123456789":
            name = f"E{ch}"
            toggles = dict(self.tracker.toggles)
            toggles[name] = not toggles[name]
            self.tracker.set_toggles(toggles)
            self._toast(f"{name} {NAMES[name]}: {'on' if toggles[name] else 'off'}")
        elif ch == "p":
            if self.state == "PLAY":
                self._pause()
            elif self.state == "PAUSE":
                self._enter("PLAY")
        elif ch == "r" and self.state == "CALIBRATE" and self.calib_phase == "review":
            self._enter("CALIBRATE")
        elif ch == "r" and self.state == "CALIBRATE" and self.calib_phase == "wait":
            self.sample_box = self.tracker.calib_box(self.size)
        elif ch == "t" and self.state == "MENU":
            self._start(practice=True)
        elif key in CONFIRM_KEYS:
            if self.state == "CALIBRATE" and self.calib_phase == "review":
                self._review("ok")
            elif self.state == "CALIBRATE" and self.calib_phase == "wait":
                self._start_sampling()
            elif self.state == "MENU":
                self._start(practice=False)
            elif self.state == "PAUSE":
                self._enter("PLAY")
            elif self.state == "GAME_OVER":
                self._start(self.practice)

    def on_mouse(self, event, x, y):
        if self.state != "CALIBRATE" or self.calib_phase != "wait":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y)
        elif event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP) and self.drag:
            x0, y0 = self.drag
            box = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            if box[2] >= 12 and box[3] >= 12:
                self.sample_box = box
            if event == cv2.EVENT_LBUTTONUP:
                self.drag = None

    def _start_sampling(self):
        self.calib_phase = "sample"
        self.tracker.start_calibration()

    def _calibrate(self, dt):
        self.phase_t += dt
        if self.calib_phase == "wait" and not self.g["calib_manual"] and self.phase_t >= self.g["calib_countdown"]:
            self._start_sampling()
        elif self.calib_phase == "sample" and len(self.tracker.calib_frames) >= self.cfg["vision"]["calib_frames"]:
            try:
                self.tracker.finish_calibration(self.size, self.sample_box)
                self.calib_phase = "review"
                self.review_menu.reset()
            except CalibrationError as e:
                self._toast(f"Calibration failed: {e}", 3.0)
                self._enter("CALIBRATE")
        elif self.calib_phase == "review":
            action = self.review_menu.update(self._points(), dt)
            if action:
                self._review(action)

    def _review(self, action):
        if action == "retry":
            self._enter("CALIBRATE")
            return
        if self.calib_path:
            try:
                self.tracker.save(self.calib_path)
            except OSError:
                pass
        if self.on_calibrated:
            self.on_calibrated()
        self._enter("MENU")

    def _menu(self, dt):
        action = self.menu.update(self._points(), dt)
        fired = self.hold.update(self._gesture(), dt)
        action = action or {1: "play", 2: "practice", 3: "calibrate", "fist": "play"}.get(fired)
        if action == "play":
            self._start(practice=False)
        elif action == "practice":
            self._start(practice=True)
        elif action == "calibrate":
            self._enter("CALIBRATE")

    def _start(self, practice):
        self.practice = practice
        self.items = []
        self.popups = []
        self.score = 0
        self.lives = self.g["lives"]
        self.streak = self.longest = self.hits = self.drops = 0
        self.play_time = 0.0
        self.spawner = Spawner(self.cfg, self.rng, practice)
        self.world = World(self.p, self.size)
        self.items = self.world.items
        self._enter("PLAY")

    def _pause(self, reason="Paused"):
        self.pause_reason = reason
        self._enter("PAUSE")

    def _play(self, dt):
        self.play_time += dt
        self.no_hand = 0.0 if self.hands else self.no_hand + dt
        if self.no_hand >= self.g["auto_pause"]:
            self._pause("Show your hands")
            return
        if self.pause_button.update(self._points(), dt):
            self._pause()
            return
        for item in self.spawner.update(dt, self.items, self.size[0]):
            self.world.add(item)
        self.world.set_gravity_mult(self.spawner.gravity_mult())
        for kind, item in self.world.step(dt, self.paddles):
            self._event(kind, item)
        self.popups = [p for p in self.popups if p[2] > self.now]
        if self.practice and self.play_time >= self.g["practice_seconds"]:
            self._log_practice()
            self._enter("GAME_OVER")
        elif self.lives <= 0:
            self.best = max(self.best, self.score)
            self._enter("GAME_OVER")

    def _event(self, kind, item):
        pos = (int(item.pos[0]), int(item.pos[1]))
        if kind == "hit":
            balls = sum(1 for it in self.items if it.alive and it.keep_up)
            self.hits += 1
            self.streak += 1
            self.longest = max(self.longest, self.streak)
            points = max(1, balls) * min(1 + self.streak // self.g["streak_every"], self.g["max_multiplier"])
            self.score += points
            self.popups.append((f"+{points}", pos, self.now + 0.6))
        elif kind == "drop" and item.keep_up:
            self.drops += 1
            self.streak = 0
            self.spawner.on_drop()
            if not self.practice:
                self.lives -= 1
                self.popups.append(("-1", (pos[0], self.size[1] - 30), self.now + 0.8))
        elif kind == "spike" and not self.practice:
            self.lives -= 1
            self.streak = 0
            self.popups.append(("Ouch!", pos, self.now + 0.8))
        elif kind == "star":
            if not self.practice and self.lives < self.g["lives"]:
                self.lives += 1
                self.popups.append(("+1 life", pos, self.now + 0.8))
            else:
                self.score += self.g["star_points"]
                self.popups.append((f"+{self.g['star_points']}", pos, self.now + 0.8))

    def _pause_state(self, dt):
        action = self.pause_menu.update(self._points(), dt)
        gesture = self._gesture()
        if self.hold.update("open" if gesture == "open" else None, dt) == "open":
            action = "resume"
        if action == "resume":
            self._enter("PLAY")
        elif action == "menu":
            self._enter("MENU")

    def _game_over(self, dt):
        action = self.over_menu.update(self._points(), dt)
        if self.hold.update("fist" if self._gesture() == "fist" else None, dt) == "fist":
            action = "again"
        if action == "again":
            self._start(self.practice)
        elif action == "menu":
            self._enter("MENU")

    def _log_practice(self):
        if not self.log_path:
            return
        minutes = max(self.play_time, 1e-6) / 60
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tracker": self.tracker.label,
            "hands": self.tracker.hands,
            "seconds": round(self.play_time, 1),
            "hits": self.hits,
            "drops": self.drops,
            "hits_per_min": round(self.hits / minutes, 2),
            "drops_per_min": round(self.drops / minutes, 2),
            "longest_streak": self.longest,
        }
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)

    def _draw(self, img):
        w, h = self.size
        getattr(self, "_draw_" + self.state.lower())(img)
        render.paddle_outline(img, self.hands, self.debug)
        render.confidence_bars(img, self.hands, (w - 112, 14))
        on = "".join(k[1] if self.tracker.toggles[k] else "." for k in ENHANCEMENTS)
        render.text(img, f"{self.tracker.label}  E[{on}]  {self.tracker.hands} hand  {self.fps:.0f} fps  cam {self.cam_fps:.0f}",
                    (10, h - 12), 0.45, render.GREY, 1)
        if self.now < self.toast_until:
            render.text(img, self.toast_text, (w // 2, h - 44), 0.7, render.YELLOW, 2, center=True)

    def _draw_calibrate(self, img):
        w, h = self.size
        x, y, bw, bh = self.sample_box
        if self.calib_phase == "review":
            if self.tracker.mask is not None:
                mask = cv2.resize(self.tracker.mask, self.size, interpolation=cv2.INTER_NEAREST) > 0
                img[mask] = (0.5 * img[mask] + np.array([0, 120, 0])).clip(0, 255).astype(np.uint8)
            render.text(img, "Your hand or glove should be green, the rest dark", (w // 2, 40), 0.65, center=True)
            render.text(img, "Hover OK / Retry, or press Enter / r", (w // 2, 70), 0.55, render.GREY, 1, center=True)
            for b in self.review_menu.buttons:
                render.draw_button(img, b)
            return
        color = render.YELLOW if self.calib_phase == "wait" else render.GREEN
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
        render.text(img, "Fill the box with your palm or glove", (w // 2, 40), 0.8, center=True)
        if self.calib_phase == "wait" and self.g["calib_manual"]:
            render.text(img, "Press SPACE to capture.  Drag the mouse to draw a new box, r to reset it.",
                        (w // 2, 72), 0.5, render.GREY, 1, center=True)
        elif self.calib_phase == "wait":
            left = max(0.0, self.g["calib_countdown"] - self.phase_t)
            render.text(img, str(int(np.ceil(left))), (x + bw // 2, y - 16), 1.2, render.YELLOW, 3, center=True)
        else:
            k = len(self.tracker.calib_frames) / self.cfg["vision"]["calib_frames"]
            cv2.rectangle(img, (x, y + bh + 8), (x + int(bw * k), y + bh + 16), render.GREEN, -1)
            render.text(img, "Hold still...", (x + bw // 2, y - 16), 0.7, render.GREEN, 2, center=True)

    def _gesture_hint(self, img):
        if self.hold.progress > 0 and self.hands:
            best = max(self.hands, key=lambda h: h.confidence)
            c = (int(best.palm[0]), int(best.palm[1] - best.radius - 20))
            render.draw_ring(img, c, 16, self.hold.progress)
            label = {"fist": "fist", "open": "open"}.get(self.hold.value, str(self.hold.value))
            render.text(img, label, (c[0], c[1] + 6), 0.5, render.WHITE, 1, center=True)

    def _draw_menu(self, img):
        w, h = self.size
        render.text(img, "KEEP IT UP", (w // 2, int(0.18 * h)), 1.6, render.YELLOW, 4, center=True)
        render.text(img, "Hover a button, or hold up 1 / 2 / 3 fingers. Fist = play.",
                    (w // 2, int(0.27 * h)), 0.5, render.GREY, 1, center=True)
        for b in self.menu.buttons:
            render.draw_button(img, b)
        if self.best:
            render.text(img, f"best {self.best}", (w // 2, int(0.9 * h)), 0.6, render.WHITE, 1, center=True)
        self._gesture_hint(img)

    def _draw_items(self, img):
        for it in self.items:
            render.draw_item(img, it, self.now)
        for msg, pos, _ in self.popups:
            render.text(img, msg, pos, 0.7, render.YELLOW, 2, center=True)

    def _draw_hud(self, img):
        render.text(img, f"{self.score}", (12, 40), 1.1, render.WHITE, 3)
        if self.practice:
            left = max(0, self.g["practice_seconds"] - self.play_time)
            render.text(img, f"practice {int(left)}s  hits {self.hits}  drops {self.drops}", (110, 34), 0.6)
        else:
            render.hearts(img, self.lives, self.g["lives"], (130, 30))
            balls = sum(1 for it in self.items if it.keep_up)
            render.text(img, f"balls {balls}  streak {self.streak}", (220, 36), 0.6)

    def _draw_play(self, img):
        self._draw_items(img)
        self._draw_hud(img)
        render.draw_button(img, self.pause_button.buttons[0])

    def _draw_pause(self, img):
        w, h = self.size
        self._draw_items(img)
        self._draw_hud(img)
        render.text(img, self.pause_reason, (w // 2, int(0.3 * h)), 1.2, render.YELLOW, 3, center=True)
        render.text(img, "Open palm 1 s or Enter to resume", (w // 2, int(0.37 * h)), 0.55, render.GREY, 1, center=True)
        for b in self.pause_menu.buttons:
            render.draw_button(img, b)
        self._gesture_hint(img)

    def _draw_game_over(self, img):
        w, h = self.size
        title = "PRACTICE DONE" if self.practice else "GAME OVER"
        render.text(img, title, (w // 2, int(0.22 * h)), 1.4, render.YELLOW, 4, center=True)
        if self.practice:
            minutes = max(self.play_time, 1e-6) / 60
            lines = [f"hits/min {self.hits / minutes:.1f}   drops/min {self.drops / minutes:.1f}",
                     f"longest streak {self.longest}   tracker {self.tracker.label}"]
        else:
            lines = [f"score {self.score}   best {self.best}", f"longest streak {self.longest}"]
        for i, line in enumerate(lines):
            render.text(img, line, (w // 2, int(0.33 * h) + i * 34), 0.7, render.WHITE, 2, center=True)
        render.text(img, "Fist = play again", (w // 2, int(0.5 * h)), 0.55, render.GREY, 1, center=True)
        for b in self.over_menu.buttons:
            render.draw_button(img, b)
        self._gesture_hint(img)
