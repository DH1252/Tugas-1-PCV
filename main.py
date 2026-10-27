"""Chroma Slash demo: slice orbs of your marker's color, avoid bombs. Options: python main.py -h.

Keys: Space start, C recalibrate, Esc quit, D debug overlay, V view (camera/mask/corrected),
P practice mode, S save snapshot to debug/. Holding a saber in START also starts.
"""
import os

# Set before cv2 is imported, or Media Foundation can take tens of seconds to open a webcam.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pygame

import tracking
from camera import Camera, list_cameras
from game import V_MIN, Game

W, H = 1280, 720                                  # window size
WHITE, GRAY, YELLOW, RED, GREEN = (240, 240, 240), (160, 160, 160), (250, 210, 60), (235, 70, 60), (80, 220, 110)
START, START_R, DWELL = (W // 2, H // 2 + 80), 80, 1.5   # hold 1.5 s in the circle to start
VIEWS = ["camera", "mask", "corrected"]          # background views (V cycles)


def to_px(u, v):
    """Screen position 0..1 -> window px."""
    return int(u * W), int(v * H)


def proc_to_px(x, y):
    """Processing px -> window px (unclamped)."""
    return int((x - tracking.AREA_X) / tracking.AREA_W * W), int((y - tracking.AREA_Y) / tracking.AREA_H * H)


def camera_picture(frame, dim):
    """Play-area crop of the frame, scaled to the window and darkened by dim."""
    s = frame.shape[1] / tracking.W               # frame px per processing px
    x, y = round(tracking.AREA_X * s), round(tracking.AREA_Y * s)
    w, h = round(tracking.AREA_W * s), round(tracking.AREA_H * s)
    img = cv2.resize(frame[y:y + h, x:x + w], (W, H))
    img = cv2.convertScaleAbs(img, alpha=dim)
    return pygame.image.frombuffer(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes(), (W, H), "RGB")


class SimCamera:
    """Fake webcam: a green ball follows the mouse, a blue one mirrors it."""

    def __init__(self):
        self.demo = None
        room = np.full((480, 640, 3), (150, 170, 190), np.uint8)
        room[360:] = (40, 70, 110)
        self.room = room
        self.next_t = 0.0

    def read(self):
        now = time.perf_counter()
        if now < self.next_t:                     # 30 fps
            return None
        self.next_t = now + 1 / 30
        img = self.room.copy()
        noise = np.random.default_rng().integers(-4, 5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        for color, (u, v) in self.balls():
            # screen position -> px in the un-mirrored 640x480 image
            s = 640 / tracking.W
            x = 639 - s * (tracking.AREA_X + u * tracking.AREA_W)
            y = s * (tracking.AREA_Y + v * tracking.AREA_H)
            cv2.circle(img, (int(x), int(y)), 14, color, -1)
        return img, now

    def balls(self):
        if not pygame.mouse.get_focused():
            return []
        mx, my = pygame.mouse.get_pos()
        u, v = mx / W, my / H
        green, blue = (60, 200, 40), (200, 90, 30)          # BGR
        cal = self.demo.cal
        if self.demo.state == "calibrate":       # only the marker being calibrated
            if cal.background is None:
                return []
            return [(green if cal.marker == "A" else blue, (u, v))]
        return [(green, (u, v)), (blue, (1 - u, v))]

    def close(self):
        pass


class Demo:
    """The window. States: calibrate -> ready -> play -> over (-> play)."""

    def __init__(self, camera, markers, debug=False, practice=False):
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Chroma Slash demo")
        self.fonts = {}
        self.camera = camera
        self.markers = markers
        self.picture = None
        self.debug, self.practice, self.view = debug, practice, 0
        self.fps, self.vision_ms = 0.0, 0.0
        self.calibrate()

    def calibrate(self):
        self.cal = tracking.Calibrator(self.markers)
        self.vision = None
        self.game = None
        self.state = "calibrate"
        self.sabers = {}                          # key -> latest Saber
        self.trails = {}                          # key -> recent window positions
        self.popups = []
        self.dwell = 0.0

    def start(self):
        self.game = Game(W, H, list(self.vision.models))
        self.game.practice = self.practice
        self.popups = []
        self.dwell = 0.0
        self.state = "play"

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.KEYDOWN:
                    self.key(e.key)
            dt = clock.tick(60) / 1000                # cap at 60 fps
            self.fps = clock.get_fps()
            now = time.perf_counter()
            self.tick(now, dt)
            self.draw()
            pygame.display.flip()

    def key(self, key):
        if key == pygame.K_c:
            self.calibrate()
        elif key == pygame.K_SPACE and self.state in ("ready", "over"):
            self.start()
        elif key == pygame.K_d:
            self.debug = not self.debug
        elif key == pygame.K_v:
            self.view = (self.view + 1) % len(VIEWS)
        elif key == pygame.K_p:
            self.practice = not self.practice
            if self.game is not None:
                self.game.practice = self.practice
        elif key == pygame.K_s:
            self.save_snapshot()

    def tick(self, now, dt):
        if self.state == "play":
            self.game.advance(dt)
            for kind, x, y in self.game.events:
                self.popups.append((kind, x, y, now))
            self.game.events.clear()
            if self.game.over:
                self.state = "over"
        got = self.camera.read()
        if got is not None:
            frame, t = got
            self.on_frame(frame, t, now)
        if self.state in ("ready", "over"):
            near = any(s.status != "searching" and math.dist(to_px(*s.pos), START) < START_R
                       for s in self.sabers.values())
            self.dwell = self.dwell + dt if near else 0.0
            if self.dwell > DWELL:
                self.start()

    def on_frame(self, frame, t, now):
        """Handle a camera frame captured at time t."""
        if self.state == "calibrate":
            mirrored, small = tracking.prepare(frame)
            self.cal.feed(small, t)
            if self.cal.done:
                self.vision = tracking.Vision(self.cal)
                self.state = "ready"
            self.picture = camera_picture(mirrored, 0.85)
            return
        started = time.perf_counter()
        mirrored, self.sabers = self.vision(frame, t)
        self.vision_ms = 1000 * (time.perf_counter() - started)
        if VIEWS[self.view] == "mask":
            self.picture = camera_picture(self.mask_image(), 1.0)
        elif VIEWS[self.view] == "corrected":
            self.picture = camera_picture(self.vision.small, 0.6)
        else:
            self.picture = camera_picture(mirrored, 0.4)
        for key, saber in self.sabers.items():
            trail = self.trails.setdefault(key, deque(maxlen=8))
            if saber.status == "searching":
                trail.clear()
            else:
                trail.append((to_px(*saber.pos), math.hypot(*saber.vel) >= V_MIN))
        if self.state == "play":
            capture_time = self.game.time - (now - t)     # game time at capture
            for key, saber in self.sabers.items():
                self.game.swing(key, saber, capture_time)

    # ---------- debug tools ----------

    def mask_image(self):
        """Cleaned masks in marker colors on black (BGR)."""
        img = np.zeros_like(self.vision.small)
        for key, mask in self.vision.masks.items():
            img[mask > 0] = self.vision.models[key].rgb[::-1]     # RGB -> BGR
        return img

    def save_snapshot(self):
        """Save the window, corrected frame and masks as PNGs in debug/."""
        folder = Path("debug")
        folder.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        pygame.image.save(self.screen, str(folder / f"{stamp}-window.png"))
        if self.vision is not None and self.vision.small is not None:
            cv2.imwrite(str(folder / f"{stamp}-corrected.png"), self.vision.small)
            for key, mask in self.vision.masks.items():
                cv2.imwrite(str(folder / f"{stamp}-mask-{key}.png"), mask)
        print(f"Saved debug/{stamp}-*.png")

    def draw_debug(self):
        """Stats, blob centers (+), Kalman prediction and gate (yellow), mask thumbnails."""
        v = self.vision
        lines = [f"FPS {self.fps:.0f}   vision {self.vision_ms:.1f} ms   view: {VIEWS[self.view]} (V)"
                 f"   practice: {'on' if self.practice else 'off'} (P)",
                 "light gains B G R: " + "  ".join(f"{g:.2f}" for g in v.light.gains)]
        for key, model in v.models.items():
            tracker, blobs = v.trackers[key], v.blobs.get(key, [])
            lines.append(f"{key} {model.name}: H {model.h_lo}..{model.h_hi}  S >= {model.s_min}  "
                         f"V >= {model.v_min}   {tracker.status}   blobs {len(blobs)}")
            for x, y, _ in blobs:
                px, py = proc_to_px(x, y)
                pygame.draw.line(self.screen, WHITE, (px - 10, py), (px + 10, py), 2)
                pygame.draw.line(self.screen, WHITE, (px, py - 10), (px, py + 10), 2)
            if tracker.pred is not None:
                radius = max(int(tracker.gate * W / tracking.AREA_W), 2)   # x and y scale match
                pygame.draw.circle(self.screen, YELLOW, proc_to_px(*tracker.pred), radius, 2)
        for i, line in enumerate(lines):
            self.text(line, (20, 110 + 30 * i), 30, left=True)
        # 160x120 masks; the rectangle is the play area.
        s = 160 / tracking.W
        for i, (key, mask) in enumerate(v.masks.items()):
            x0, y0 = 20 + 170 * i, H - 140
            thumb = cv2.cvtColor(cv2.resize(mask, (160, 120), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB)
            self.screen.blit(pygame.image.frombuffer(thumb.tobytes(), (160, 120), "RGB"), (x0, y0))
            area = pygame.Rect(x0 + tracking.AREA_X * s, y0 + tracking.AREA_Y * s,
                               tracking.AREA_W * s, tracking.AREA_H * s)
            pygame.draw.rect(self.screen, v.models[key].rgb, area, 1)
            self.text(key, (x0 + 10, y0 + 12), 26, v.models[key].rgb)

    # ---------- drawing ----------

    def text(self, s, pos, size=40, color=WHITE, left=False):
        if size not in self.fonts:
            self.fonts[size] = pygame.font.Font(None, size)
        img = self.fonts[size].render(s, True, color)
        shadow = self.fonts[size].render(s, True, (0, 0, 0))
        rect = img.get_rect(midleft=pos) if left else img.get_rect(center=pos)
        self.screen.blit(shadow, rect.move(2, 2))
        self.screen.blit(img, rect)

    def draw(self):
        if self.picture is None:
            self.screen.fill((20, 20, 28))
        else:
            self.screen.blit(self.picture, (0, 0))
        if self.state == "calibrate":
            self.draw_calibration()
            return
        if self.game is not None:
            self.draw_orbs()
        self.draw_trails()
        if self.game is not None:
            self.draw_popups()
            self.draw_hud()
        if self.debug:
            self.draw_debug()
        if self.state == "ready":
            self.draw_start("Hold a saber in the circle, or press Space")
        elif self.state == "over":
            self.text("GAME OVER", (W // 2, 150), 110)
            self.text(f"Score {self.game.score}", (W // 2, 240), 64, YELLOW)
            self.draw_start("Play again: hold a saber in the circle, or press Space")

    def draw_calibration(self):
        cal = self.cal
        self.text(cal.message, (W // 2, 60), 52)
        if cal.box is not None:
            x, y, w, h = cal.box
            left, top = to_px((x - tracking.AREA_X) / tracking.AREA_W, (y - tracking.AREA_Y) / tracking.AREA_H)
            size = (w / tracking.AREA_W * W, h / tracking.AREA_H * H)
            pygame.draw.rect(self.screen, GREEN if cal.found else WHITE, pygame.Rect((left, top), size), 4)
            if self.debug:
                self.text(f"spot {cal.spot_i + 1}/3   frames with the marker {cal.hits}/15", (W // 2, H - 50), 34)

    def draw_start(self, hint):
        pygame.draw.circle(self.screen, WHITE, START, START_R, 4)
        if self.dwell > 0:                        # progress ring
            rect = pygame.Rect(0, 0, 2 * START_R + 20, 2 * START_R + 20)
            rect.center = START
            pygame.draw.arc(self.screen, GREEN, rect, math.pi / 2, math.pi / 2 + 2 * math.pi * self.dwell / DWELL, 8)
        self.text("START", START, 48)
        self.text(hint, (START[0], START[1] + 130), 36, GRAY)

    def draw_orbs(self):
        for orb in self.game.orbs:
            pos = (int(orb.body.position.x), int(orb.body.position.y))
            r = int(orb.shape.radius)
            if orb.kind == "bomb":
                pygame.draw.circle(self.screen, (55, 55, 62), pos, r)
                pygame.draw.circle(self.screen, RED, pos, r // 2, 4)
            else:
                pygame.draw.circle(self.screen, self.vision.models[orb.kind].rgb, pos, r)
                pygame.draw.circle(self.screen, WHITE, pos, r, 2)

    def draw_trails(self):
        """Last 8 positions per saber; thick and colored = fast enough to cut."""
        for key, trail in self.trails.items():
            if not trail:
                continue
            color = self.vision.models[key].rgb
            points = list(trail)
            for (p0, _), (p1, fast) in zip(points, points[1:]):
                pygame.draw.line(self.screen, color if fast else GRAY, p0, p1, 10 if fast else 3)
            pygame.draw.circle(self.screen, color, points[-1][0], 14, 4)

    def draw_popups(self):
        labels = {"wrong": ("WRONG COLOR", GRAY), "bomb": ("-1 LIFE", RED), "miss": ("MISS", RED),
                  "slice": ("+10", YELLOW)}
        now = time.perf_counter()
        self.popups = [p for p in self.popups if now - p[3] < 0.9]
        for kind, x, y, born in self.popups:
            text, color = labels[kind]
            self.text(text, (x, y - 60 * (now - born)), 40, color)   # float up

    def draw_hud(self):
        self.text(str(self.game.score), (90, 50), 64)
        if self.game.practice:
            self.text("PRACTICE", (W // 2, 44), 40, YELLOW)
            return
        for i in range(3):
            color = RED if i < self.game.lives else (70, 70, 70)
            pygame.draw.circle(self.screen, color, (W // 2 - 40 + 40 * i, 44), 14)


def main():
    p = argparse.ArgumentParser(description="Chroma Slash demo: slice orbs with colored markers.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--one-hand", action="store_true", help="play with a single marker")
    p.add_argument("--list", action="store_true", help="show which cameras work")
    p.add_argument("--backend", choices=["auto", "msmf", "dshow"], default="auto",
                   help="camera backend (Windows): auto tries Media Foundation, then DirectShow")
    p.add_argument("--size", default="640x480", metavar="WxH", help="capture resolution, e.g. 1280x720")
    p.add_argument("--track-size", default="320x240", metavar="WxH",
                   help="color tracking resolution, 4:3 (default 320x240); higher is finer but slower")
    p.add_argument("--margin", type=float, default=0.1,
                   help="camera border left out of the play area, 0-0.4 (default 0.1)")
    p.add_argument("--opencl", action="store_true", help="run the color masks on the GPU (OpenCL)")
    p.add_argument("--sim", action="store_true", help="no camera: the mouse moves the markers")
    p.add_argument("--debug", action="store_true", help="start with the debug overlay on (D toggles)")
    p.add_argument("--practice", action="store_true", help="no lives are lost (P toggles)")
    args = p.parse_args()
    width, height = args.size.lower().split("x")
    size = (int(width), int(height))
    width, height = args.track_size.lower().split("x")
    if int(width) * 3 != int(height) * 4:
        p.error("--track-size must be 4:3, e.g. 240x180, 320x240 or 640x480")
    tracking.set_size(int(width), int(height))
    tracking.set_margin(min(max(args.margin, 0.0), 0.4))
    if args.opencl:
        device = tracking.enable_opencl()
        print(f"OpenCL: {device}" if device else "OpenCL is not available; running on the CPU.")
    if args.list:
        found = list_cameras(args.backend, size)
        for index, w, h in found:
            print(f"--camera {index}: {w}x{h}")
        if not found:
            print("No camera could be opened. Close other apps that use it and check the privacy settings.")
        return
    if args.sim:
        camera = SimCamera()
    else:
        try:
            camera = Camera(args.camera, args.backend, size)
        except RuntimeError as e:
            sys.exit(str(e))
    pygame.init()
    demo = Demo(camera, ["A"] if args.one_hand else ["A", "B"], args.debug, args.practice)
    camera.demo = demo
    try:
        demo.run()
    finally:
        camera.close()
        pygame.quit()


if __name__ == "__main__":
    main()
