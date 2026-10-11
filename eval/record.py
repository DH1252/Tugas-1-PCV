import argparse
import json
import time
from pathlib import Path

import cv2

from game import render
from vision.capture import Source
from vision.config import load_config
from vision.tracker import calib_box

CALIB = (1.5, 3.0)


def main():
    ap = argparse.ArgumentParser(description="Record an evaluation clip")
    ap.add_argument("name", help="e.g. C3_warm")
    ap.add_argument("--seconds", type=float, default=25)
    ap.add_argument("--still", type=float, nargs=2, default=(6, 11), metavar=("START", "END"))
    ap.add_argument("--no-still", action="store_true")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera
    src = Source(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video_path = out / f"{args.name}.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30, src.size)
    box = calib_box(cfg, src.size)
    still = None if args.no_still else tuple(args.still)
    calib_idx, still_idx = [], []
    count = 0
    start = time.perf_counter()
    while True:
        frame = src.read()
        if frame is None:
            break
        elapsed = time.perf_counter() - start
        if elapsed >= args.seconds:
            break
        writer.write(frame)
        if CALIB[0] <= elapsed < CALIB[1]:
            calib_idx.append(count)
        if still and still[0] <= elapsed < still[1]:
            still_idx.append(count)
        count += 1

        view = frame.copy()
        w, h = src.size
        if elapsed < CALIB[1]:
            x, y, bw, bh = box
            color = render.GREEN if elapsed >= CALIB[0] else render.YELLOW
            cv2.rectangle(view, (x, y), (x + bw, y + bh), color, 2)
            render.text(view, "Fill the box with your palm", (w // 2, 40), 0.8, center=True)
        elif still and still[0] <= elapsed < still[1]:
            render.text(view, "HOLD STILL", (w // 2, 40), 1.0, render.RED, 3, center=True)
        else:
            render.text(view, f"Move  ({args.name})", (w // 2, 40), 0.8, center=True)
        render.text(view, f"REC {args.seconds - elapsed:4.1f}s", (10, h - 12), 0.6, render.RED, 2)
        cv2.imshow("record", view)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    elapsed = time.perf_counter() - start
    writer.release()
    src.release()
    cv2.destroyAllWindows()

    meta = {
        "name": args.name,
        "video": video_path.name,
        "fps": round(count / max(elapsed, 1e-6), 2),
        "frames": count,
        "size": list(src.size),
        "calib": [calib_idx[0], calib_idx[-1] + 1] if calib_idx else None,
        "roi": list(box),
        "still": [still_idx[0], still_idx[-1] + 1] if still_idx else None,
    }
    (out / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {video_path} ({count} frames, {meta['fps']} fps)")


if __name__ == "__main__":
    main()
