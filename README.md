# Keep It Up

A webcam keep-up game. Balls fall from the top of the screen and you bounce them back up with
your hands. The hands are tracked with HSV colour segmentation only: no MediaPipe and no
learned detectors. An enhanced HSV pipeline (E1–E9) is compared against a plain
fixed-threshold baseline. See [PLAN.md](PLAN.md) for the design.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Play

```bash
python main.py                     # enhanced tracker, one hand
python main.py --hands 2           # two hands
python main.py --tracker baseline  # fixed-threshold baseline
python main.py --debug             # pipeline stages + tuning sliders in a second window
python main.py --video clip.avi    # replay a recorded clip instead of the webcam
```

On first start, the game asks you to fill the yellow box with your palm. Keep your face in
view, because calibration also learns where your face is so it can ignore it. Then check that
your hand shows green and the background stays dark. The calibration is saved to
`calibration.npz` and reused next time. Press `c` to redo it, or start with `--recalibrate`.

The webcam opens through Media Foundation (`camera.backend: msmf` in `config.yaml`) with
hardware transforms off and MJPG frames, which avoids the slow open and frozen feed that MSMF
often has. Frames are read on a background thread. If the camera stops sending frames for 2 s,
it is reopened automatically. Set `backend` to `dshow` or `auto` if your camera misbehaves with
MSMF. On Linux and macOS, `msmf` falls back to the default backend.

Calibration tips:
- Use even, front-facing light. Avoid a bright window behind you.
- Long sleeves help.
- Avoid skin-coloured walls or furniture right behind your hands.

### Controls

| Input | Action |
|---|---|
| Hand touches a ball | Bounce it. Swing upward for a stronger hit. |
| Hover a button for 1.2 s | Press it |
| Hold up 1 / 2 / 3 fingers for 1 s (menu) | Play / Practice / Recalibrate |
| Fist for 1 s (menu, game over) | Play |
| Hover the pause icon for 1 s | Pause |
| Open palm for 1 s (paused) | Resume |
| `Enter` / `Space` | Confirm |
| `p` | Pause / resume |
| `t` (menu) | Practice |
| `c` | Recalibrate |
| `b` | Switch baseline ↔ enhanced |
| `h` | Switch one ↔ two hands |
| `1`–`9` | Toggle enhancement E1–E9 |
| `d` | Debug window |
| `q` / `Esc` | Quit |

Items: **ball** (normal), **balloon** (slow and floaty), **heavy** (needs a firm swat),
**spike** (don't touch it, let it fall), **star** (touch it for a life or points). You lose a
life when a ball reaches the bottom or when you touch a spike.

### Enhancements

| | | |
|---|---|---|
| E1 | white balance | blur + gray-world white balance (only darkens) |
| E2 | hue gate | hue rotation by +20 so skin is one range; drops low-S, dark and glare pixels |
| E3 | calibration | per-user HSV and YCrCb ranges and H–S histogram from the calibration box |
| E4 | back-projection | `P(hand \| colour)` from hand vs background histograms |
| E5 | YCrCb | Cr/Cb gate as a second check |
| E6 | open/close | elliptical opening, then closing, then hole filling |
| E7 | face + scoring | mean-shift face removal; blobs ranked by size, motion and track proximity |
| E8 | palm geometry | distance-transform palm centre, forearm cut, finger count |
| E9 | temporal | Kalman per hand, One Euro smoothing, two-hand assignment, adaptive histogram |

With every toggle off, the tracker is the baseline: fixed `inRange`, erode/dilate, largest
contour(s), centroid.

## Evaluate

```bash
python -m eval.record C3_warm --seconds 25        # fill the box, move, hold still when told
python -m eval.annotate clips/C3_warm.avi          # click palms (A left click, B right click), g = polygon
python -m eval.evaluate --loo                      # all labelled clips → eval/results/
```

`eval/results/` gets `results.csv`, `ablation.md` (baseline → +E1 → … → +E9, plus
leave-one-out with `--loo`) and bar charts. Practice-mode sessions are logged to
`practice_log.csv` and summarised in `eval/results/practice.md`.

## Tests

```bash
pytest
```

The tests run without a camera, on synthetic hand frames and clips.
