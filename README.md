# Chroma Slash

A webcam game where you slice flying orbs with a colored marker or glove. The marker is found by its color alone, with classic OpenCV operations (HSV thresholds, morphology, connected components and a Kalman filter). No neural networks are used.

## Setup

Python 3.11 or newer:

```bash
pip install -r requirements.txt
python main.py --one-hand        # one marker or glove
python main.py                   # two markers: green and blue
python main.py --sim             # no camera: the mouse moves the markers
```

`python main.py -h` lists the other options: camera number and backend, capture size, play-area margin, OpenCL, debug overlay and practice mode.

## Playing

The game calibrates every time it starts, which takes about 10 seconds:

1. Step out of view with your markers for the 2-second countdown, so it can photograph the empty room.
2. Hold each marker inside the square as it moves to three spots.

Then hold a marker in the START circle (or press Space). Swing fast through orbs of your marker's color. A bomb or an orb that falls past you costs a life, and you have three.

| Key | Action |
|---|---|
| Space | start |
| C | calibrate again |
| D | debug overlay |
| V | show the camera, the color mask or the light-corrected frame |
| P | practice mode (no lives lost) |
| S | save a snapshot to `debug/` |
| Esc | quit |

