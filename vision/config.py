from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


PRESETS = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}


def load_config(path=None):
    with open(path or DEFAULT_PATH) as f:
        return yaml.safe_load(f)


def parse_size(text):
    text = text.lower().strip()
    if text in PRESETS:
        return PRESETS[text]
    try:
        w, h = (int(v) for v in text.split("x"))
    except ValueError:
        raise ValueError(f"bad size {text!r}, use WxH or one of {', '.join(PRESETS)}") from None
    if w < 160 or h < 120:
        raise ValueError(f"size {text!r} is too small")
    return w, h


def apply_resolution(cfg, capture=None, frame=None):
    cam = cfg["camera"]
    if capture:
        cam["capture_width"], cam["capture_height"] = parse_size(capture)
    if frame:
        cam["width"], cam["height"] = parse_size(frame)
    elif capture:
        cw, ch = cam["capture_width"], cam["capture_height"]
        h = min(cam["height"], ch)
        cam["width"], cam["height"] = int(round(h * cw / ch)), h
    return cfg


def add_resolution_args(ap):
    ap.add_argument("--resolution", metavar="WxH",
                    help="camera capture size, e.g. 1280x720 or 480p/720p/1080p")
    ap.add_argument("--frame", metavar="WxH",
                    help="tracking/game size (default: height 480 with the capture's aspect ratio)")
