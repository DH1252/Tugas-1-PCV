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
python main.py --list-cameras      # show available camera indices
python main.py --camera 1          # use a specific camera (also: camera.index in config.yaml)
python main.py --resolution 1080p  # camera capture size: 480p, 720p, 1080p or WxH (default 1280x720)
python main.py --frame 1280x720    # tracking/game size (default: 480 high, same aspect as the capture)
python main.py --camera-fps 30     # if your camera can't do 60 fps (default asks for 60)
python main.py --fps 60            # game frame rate (default 60)
```

A bigger `--frame` gives a sharper picture but costs more tracking time (it scales with the
pixel count); the capture size mostly affects image quality and what your camera supports. If
the camera delivers a different aspect ratio than the frame, the centre is cropped instead of
stretched. Changing the frame size triggers a new calibration. `eval.record` takes the same
options.

While playing, press `k` to switch to the next camera. Recalibrate (`c`) afterwards.

On first start, put your palm (or a gloved hand) in the yellow box and press **Space** to
capture its colour. To sample a different area, drag a new box with the mouse first; `r` resets
it. Then check that your hand shows green and the background stays dark.

Calibration also auto-tunes the main parameters from what it sees:
- the back-projection thresholds, picked from how hand and background pixels score
- the S/V gate, lowered if your hand has dim or pale areas
- the opening/closing kernels, sized to the sample box

It also grows the colour model from the palm patch out to the whole connected hand, so shaded
fingers and edges are learned too, instead of showing up as separate islands. The mask uses
two thresholds (hysteresis): weaker pixels are kept only if they touch strongly matching ones.
The debug window (`--debug`) shows the tuned values. Moving a slider overrides that value until
you recalibrate. Set `vision.auto_tune: false` to use the `config.yaml` values instead.

**Use a coloured glove.** There is no face detection, so with bare hands your face (also
skin) can be picked up as a hand. A glove in a colour that isn't on you or behind you avoids
that; check the room for things of the same colour (a green glove next to a green toy will
fight it). Calibration learns whatever colour is in the box and centres its
hue, so a green, blue or red glove is tracked like skin, and a glove that differs from your skin
and the room is often easier to track. Grey, black or white gloves can't be tracked, since they
have no hue. The baseline tracker (`b`) uses fixed skin thresholds, so it only works on bare
skin. The calibration is saved to
`calibration.npz` and reused next time. Press `c` to redo it, or start with `--recalibrate`.

The webcam opens through Media Foundation (`camera.backend: msmf` in `config.yaml`) at 1280×720 with
hardware transforms off and MJPG frames, which avoids the slow open and frozen feed that MSMF
often has. Frames are read on a background thread. If the camera stops sending frames for 2 s,
it is reopened automatically. Set `backend` to `dshow` or `auto` if your camera misbehaves with
MSMF. On Linux and macOS, `msmf` falls back to the default backend. Frames are scaled to
854×480 (`camera.width`/`height`) for tracking and the game; a saved calibration from a different
size is redone automatically.

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
| `Space` (calibration) | Capture the colour in the box |
| Mouse drag / `r` (calibration) | Draw a new sample box / reset it |
| `b` | Switch baseline ↔ enhanced |
| `h` | Switch one ↔ two hands |
| `k` | Next camera |
| `1`–`9` | Toggle enhancement E1–E9 |
| `d` | Debug window |
| `q` / `Esc` | Quit |

The game runs at 60 fps whatever the camera does. Camera frames arrive at 60 or 30 fps on a
background thread; the hand tracker runs on each new frame, and between frames the hands are
moved ahead along their measured speed, so the paddles stay smooth. The status line shows both
rates (`60 fps  cam 30`). Physics uses [pymunk](https://www.pymunk.org/).

Items: **ball** (normal), **balloon** (slow and floaty), **heavy** (needs a firm swat),
**spike** (don't touch it, let it fall), **star** (touch it for a life or points). You lose a
life when a ball reaches the bottom or when you touch a spike.

### Keeping track (colour-only robustness)

These all use colour only, no learned detectors:
- **Lighting:** the second colour check uses rg chromaticity, which ignores brightness, so
  the glove isn't dropped when it moves into brighter or darker light or the camera
  re-exposes.
- **Motion blur:** the palm is measured on pixels whose chromaticity matches the calibrated
  glove ("core" colour). Smeared edges, where glove and background mix, still count for
  detection but don't drag the palm point.
- **Look-alike objects:** anything glove-coloured that sits still during calibration is
  masked out by position, and kept out of the background colour model so the glove colour
  isn't penalised. Look-alikes that show up later are learned once they've sat still for
  3 s while your hand is tracked elsewhere.
- **Getting stuck:** motion is measured on the colour-probability map (also
  brightness-invariant). If the tracked blob stops moving while another glove-coloured blob
  is moving, the tracker switches to the moving one after about 0.7 s. When the glove
  touches a look-alike and the blobs merge, the palm is picked near where the track
  expects it.
- **Calibration box:** only the dominant colour in the box is learned. Background showing in
  the box corners is ignored.

Pick a glove colour that isn't elsewhere in the room anyway; these help, but a unique colour
is still the strongest cue.

### Enhancements

| | | |
|---|---|---|
| E1 | white balance | blur + gray-world white balance (only darkens) |
| E2 | hue gate | hue rotation by +20 so skin is one range; drops low-S, dark and glare pixels |
| E3 | calibration | per-user HSV and YCrCb ranges and H–S histogram from the calibration box |
| E4 | back-projection | `P(hand \| colour)` from hand vs background histograms |
| E5 | rg chroma | brightness-invariant chromaticity gate, r = R/(R+G+B), g = G/(R+G+B) |
| E6 | open/close | elliptical opening, then closing, then hole filling |
| E7 | blob scoring | look-alike masking, blobs ranked by size, colour motion and track proximity |
| E8 | palm geometry | palm centre from core-colour pixels, forearm cut, finger count |
| E9 | temporal | Kalman per hand, One Euro smoothing, two-hand assignment, adaptive histogram |

With every toggle off, the tracker is the baseline: fixed `inRange`, erode/dilate, largest
contour(s), centroid.

## Evaluate

```bash
python -m eval.record C3_warm --seconds 25        # fill the box, move, hold still when told (--camera N)
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
