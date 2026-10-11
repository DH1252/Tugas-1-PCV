import argparse
import os
import time

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2  # noqa: E402

from game.engine import Game
from vision.capture import Source, list_cameras
from vision.config import add_resolution_args, apply_resolution, load_config

WINDOW = "Keep It Up"
DEBUG_WINDOW = "Debug"
TRACKBARS = (
    ("threshold", "backproject_thresh", 255, False),
    ("low threshold", "backproject_low", 255, False),
    ("S min", "s_min", 255, False),
    ("V min", "v_min", 255, False),
    ("V max", "v_max", 255, False),
    ("open", "open_kernel", 21, True),
    ("close", "close_kernel", 31, True),
)


def make_trackbars(tracker):
    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    for name, key, top, _ in TRACKBARS:
        cv2.createTrackbar(name, DEBUG_WINDOW, int(tracker.param(key)), top, lambda _: None)
    return sync_trackbars(tracker)


def sync_trackbars(tracker):
    shown = {}
    for name, key, _, _ in TRACKBARS:
        shown[key] = int(tracker.param(key))
        cv2.setTrackbarPos(name, DEBUG_WINDOW, shown[key])
    return shown


def read_trackbars(tracker, shown):
    for name, key, _, odd in TRACKBARS:
        value = cv2.getTrackbarPos(name, DEBUG_WINDOW)
        if odd:
            value = max(1, value | 1)
        if value != shown[key]:
            tracker.override(key, value)
            shown[key] = value


def main():
    ap = argparse.ArgumentParser(description="Keep It Up: HSV hand-tracking keep-up game")
    ap.add_argument("--tracker", choices=("enhanced", "baseline"), default="enhanced")
    ap.add_argument("--hands", type=int, choices=(1, 2), default=1)
    ap.add_argument("--video", help="replay a recorded clip instead of the webcam")
    ap.add_argument("--loop", action="store_true", help="loop the replayed clip")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--recalibrate", action="store_true", help="ignore the saved calibration")
    add_resolution_args(ap)
    ap.add_argument("--camera", type=int, help="camera index (see --list-cameras)")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.list_cameras:
        cams = list_cameras(cfg["camera"])
        for i, w, h in cams:
            print(f"camera {i}: {w}x{h}")
        if not cams:
            print("no cameras found")
        return
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    try:
        apply_resolution(cfg, args.resolution, args.frame)
    except ValueError as e:
        ap.error(str(e))
    cameras = None
    src = Source(cfg, args.video)
    game = Game(cfg, hands=args.hands, baseline=args.tracker == "baseline", debug=args.debug,
                on_calibrated=src.lock_auto if cfg["camera"]["lock_auto"] else None,
                load_calibration=not args.recalibrate)

    scale = cfg["display"]["scale"]
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, int(src.size[0] * scale), int(src.size[1] * scale))
    cv2.setMouseCallback(WINDOW, lambda event, x, y, flags, param: game.on_mouse(event, x, y))
    delay = max(1, int(1000 / src.fps)) if src.is_file else 1
    key, trackbars = -1, False
    while not game.quit:
        frame = src.read()
        if frame is None:
            if args.video and args.loop:
                src.rewind()
                continue
            break
        t = src.index / src.fps if src.is_file else time.perf_counter()
        view, debug_view = game.step(frame, t, key)
        cv2.imshow(WINDOW, view)
        if debug_view is not None:
            if not trackbars:
                shown = make_trackbars(game.tracker)
                synced = game.tracker.tune_version
                trackbars = True
            elif synced != game.tracker.tune_version:
                shown = sync_trackbars(game.tracker)
                synced = game.tracker.tune_version
            read_trackbars(game.tracker, shown)
            cv2.imshow(DEBUG_WINDOW, debug_view)
        elif trackbars:
            cv2.destroyWindow(DEBUG_WINDOW)
            trackbars = False
        key = cv2.waitKey(delay)
        key = -1 if key < 0 else key & 0xFF
        if key == ord("k") and not src.is_file:
            current = src.cam["index"]
            if cameras is None:
                cameras = sorted({current, *(i for i, _, _ in list_cameras(cfg["camera"]))})
            nxt = cameras[(cameras.index(current) + 1) % len(cameras)]
            if nxt != current and src.switch(nxt):
                game.tracker.reset()
                game._toast(f"camera {nxt} - press c to recalibrate", 3.0)
            else:
                game._toast(f"camera {current} (no other camera found)", 2.0)
            key = -1
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
