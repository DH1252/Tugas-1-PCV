# Hand Slash: Plan for an HSV Hand-Tracking Game

A webcam fruit-slicing game. You slice flying fruit by swiping your hand and steer clear of
bombs. The hand is tracked with **HSV colour segmentation**, and the assignment's technical
contribution is an **enhanced HSV pipeline** that we compare against a plain fixed-threshold
baseline.

---

## 1. Goals and scope

| Goal | How we'll show it |
|---|---|
| Playable game driven only by hand movement and simple gestures | Live demo plus a recorded video |
| Baseline HSV tracker (fixed thresholds) | `--tracker baseline` |
| Enhanced HSV tracker, with each enhancement switchable on and off | `--tracker enhanced` and live hotkeys `1`–`8` |
| Measured improvement of enhanced over baseline | Ablation table from the `eval/` harness on recorded clips |

**Out of scope:** learned hand detectors such as MediaPipe or YOLO for *runtime* tracking. The
point of the assignment is classical HSV. MediaPipe may optionally be used **offline in
`eval/` only**, to make reference labels (see §7).

**Assumptions:** one player, a laptop webcam at 640×480 and 30 fps or better, Python 3.10+,
and a single developer working about 2 weeks. Adjust §8 if the deadline or team size differs.

---

## 2. Game design

### Core loop
1. Fruit is launched from the bottom edge on parabolic paths (gravity, random speed and angle).
2. The player's **hand cursor** (smoothed palm centre) leaves a blade trail.
3. A fruit is **sliced** when the cursor's movement segment for this frame passes through the
   fruit's circle **and** the cursor speed is at least `v_min` (about 800 px/s at 640 px width).
   The speed check stops a still hand from "hovering" fruit to death.
4. A sliced fruit splits into two halves and juice particles. Score +10.
5. **Combo:** 3 or more fruit in one swipe (within 250 ms) gives a ×N bonus.
6. **Bombs:** slicing one costs 1 life. **Missed fruit** (falls off-screen) also costs 1 life.
   You start with 3 lives.
7. **Difficulty ramp:** over about 2 minutes, the spawn interval drops from 1.2 s to 0.5 s and
   the bomb chance rises from 5% to 20%.

### States
```
CALIBRATE ──► MENU ──► PLAY ⇄ PAUSE ──► GAME_OVER ──► MENU
                 └──► PRACTICE (fixed targets; doubles as a usability test, see §7)
```

### Hand controls (no keyboard needed while playing)
| Input | Action |
|---|---|
| Swipe | Slice |
| Hover cursor on a button for 1.2 s (radial progress ring) | Select a menu item. This is the main UI method and doesn't depend on finger counting. |
| Open palm (4 or more fingers) held 1 s | Pause / resume |
| Fist (0 fingers) | Shortcut for confirm / start |
| Hand lost for more than 1.5 s | Auto-pause with a "Show your hand" prompt |

### Keyboard (developer and demo controls)
`c` recalibrate · `d` debug overlay · `b` switch baseline/enhanced · `1`–`8` toggle enhancements
E1–E8 · `p` pause · `q`/`Esc` quit

### HUD
Score, lives, combo, and a **tracking-confidence bar**. The bar shows the audience when the
HSV tracker is struggling, which helps when presenting the enhancements.

### Calibration screen (part of the game flow)
"Place your hand inside the box". A 2 s countdown, then about 60 frames are sampled. The
resulting mask is shown so the player can accept it or retry.

---

## 3. Baseline HSV tracker (the comparison point)

```
BGR → HSV → cv2.inRange((0,48,80), (20,255,255))
    → erode×2 / dilate×2 (3×3) → largest contour → centroid (moments)
```
It has no smoothing, no calibration and no temporal logic. Known failure modes:
- Fixed thresholds suit some skin tones and lighting and fail on others.
- Warm or cool light shifts the hue.
- Reddish skin hue wraps around H = 179/0.
- Face, arm and wooden or beige backgrounds are picked up as hand.
- The centroid drifts toward the forearm.
- Frame-to-frame jitter.

Each enhancement below targets one of these failure modes.

---

## 4. Enhanced HSV pipeline

```
Webcam frame (BGR 640×480, mirrored) ──downscale ×0.5──►
 [E1] Preprocess: Gaussian blur 5×5 · gray-world white balance · (lock camera AE/AWB if backend allows)
   ▼
 BGR→HSV  [E2] hue rotation h' = (h+20) mod 180 · reliability gate (S ≥ 30, 40 ≤ V ≤ 250)
   ▼
 [E3] per-user calibration ──► [E4] back-project H–S *ratio* histogram ──► probability map
   ▼                                                                         ▲
 [E5] ∧ calibrated YCrCb gate                                                │
   ▼                                                                         │ [E8c] adaptive
 [E6] threshold · open/close · fill holes · contour filter + scoring         │  histogram
   ▼                                                                         │  update (EMA)
 [E7] palm centre (distance transform) · arm cut · hull/defects → fingers ───┘
   ▼
 [E8] Kalman predict/correct (gating, search ROI, bridge ≤5 lost frames) → One Euro smoothing
   ▼
 HandState{found, palm, tip, velocity, fingers, gesture, confidence, mask, contour}
```

### Enhancements

| ID | Enhancement | Problem it fixes | Implementation notes | Priority |
|---|---|---|---|---|
| **E1** | Preprocessing: blur and **gray-world white balance** | Sensor noise; colour casts from warm or cool lamps shift hue | Scale B, G, R so their means are equal. Try `CAP_PROP_AUTO_WB`/`CAP_PROP_AUTO_EXPOSURE` (backend-dependent) so the camera stops re-tuning mid-game. | Must |
| **E2** | **Hue rotation and reliability gating** | (a) Skin hue straddles 179↔0, so it splits into two ranges. (b) Hue is undefined or noisy when S or V is low (greys, shadows) and in specular highlights. | (a) Use a LUT to add +20 to H mod 180, so skin becomes one range (about 5–45). (b) Drop pixels with S < 30, V < 40 or V > 250 before any hue test. | Must |
| **E3** | **Per-user calibration** | Hard-coded ranges fail across skin tones and rooms | Sample the ROI over about 60 frames and build a 2-D H–S histogram (30×32 bins). Also store 2nd–98th percentile H, S, V and Cr, Cb as fallback box thresholds. Save to `calibration.json`. | Must |
| **E4** | **Discriminative histogram back-projection** (Swain & Ballard ratio histogram) | A box threshold is binary and also takes in background colours that look like skin | `R = min(H_hand / (H_bg + ε), 1)` where `H_bg` is sampled outside the ROI at calibration. `cv2.calcBackProject` gives a per-pixel probability, then blur and threshold it (τ ≈ 50/255). Colours common in the background are down-weighted automatically. | Must |
| **E5** | **Hybrid colour-space fusion**: AND with a YCrCb gate | Some false positives are separable in Cr/Cb but not in H–S | The Cr/Cb range comes from calibration percentiles ± a margin. Fall back to Chai & Ngan (Cr 133–173, Cb 77–127). It is loose on purpose, a veto rather than the main detector. | Should |
| **E6** | **Mask cleanup and contour scoring** | Speckle, holes, face and other skin blobs | Morphological open (ellipse 5×5) then close (9×9), and fill holes. Keep contours with area 1.5–40% of the frame and solidity ≥ 0.45. Pick the best contour by score = size + overlap with a MOG2 motion mask + closeness to the Kalman prediction − overlap with the face box (Haar cascade run every 5 frames). Motion is only a *scoring term*, not a pixel gate, so a hand held still isn't absorbed into the background. | Must (motion and face: Should) |
| **E7** | **Hand geometry** | The centroid drifts toward the forearm; gestures are needed | Palm centre = arg-max of `cv2.distanceTransform`, with palm radius r = its value. Cut the mask outside a 2.5r circle (removes the forearm). Use `convexHull` + `convexityDefects` with depth > 0.4r and angle < 90°, so fingers = defects + 1. Treat it as a fist if the farthest hull point is < 1.6r away. Debounce with a majority vote over 8 frames. | Should |
| **E8** | **Temporal filtering** | Jitter, short dropouts, slow full-frame search | (a) A `cv2.KalmanFilter` with constant velocity on the palm centre. It predicts the next position, rejects jumps more than 80 px from the prediction, carries the cursor through up to 5 lost frames, and limits the search to an ROI around the prediction (CamShift-style). (b) A **One Euro filter** (Casiez et al., 2012) on the output: little smoothing at slow speeds and low lag in fast swipes, which suits slicing. (c) **Adaptive model update**: every 10 frames, if confidence > 0.7, `H_hand ← (1−α)H_hand + α·H_current` with α ≈ 0.03, using pixels from the *eroded* contour only. Reset on `c` or after a long loss so drift can't build up. | Must (a, b), Could (c) |

**Confidence** combines the mean back-projection inside the contour, solidity, whether the
area is plausible, and the Kalman innovation. It feeds the HUD bar, the adaptive update gate
and the auto-pause.

### Starting parameters (tune live with debug-mode trackbars, save to `config.yaml`)
Processing size 320×240 · blur 5×5 · S ≥ 30, 40 ≤ V ≤ 250 · hue shift +20 · H–S bins 30×32 ·
back-projection τ = 50 · open 5×5 / close 9×9 · area 1.5–40% · solidity ≥ 0.45 ·
Kalman gate 80 px, max coast 5 frames · One Euro `min_cutoff=1.0, beta=0.02, d_cutoff=1.0` ·
α = 0.03 every 10 frames · face detection every 5 frames · slice `v_min = 800 px/s`.

---

## 5. Architecture

Vision and game are separate. The game only sees `HandState`, so swapping baseline and
enhanced, or live camera and recorded video, needs no game changes.

```
tugas-1-pcv/
├── main.py                 # python main.py [--tracker baseline|enhanced] [--video clip.mp4] [--debug]
├── config.yaml             # all tunables from §4
├── requirements.txt        # opencv-python, numpy, pyyaml, pytest  (optional: pygame for sound)
├── vision/
│   ├── capture.py          # webcam or video-file source, mirroring, FPS counter
│   ├── baseline.py         # §3
│   ├── preprocess.py       # E1, E2
│   ├── calibration.py      # E3 (+ background hist for E4), load/save calibration.json
│   ├── segmentation.py     # E4, E5, E6 (mask, contour scoring, face, motion)
│   ├── hand.py             # E7 geometry and gestures
│   ├── filters.py          # E8 Kalman, One Euro, adaptive update
│   ├── tracker.py          # HandTracker: runs the stages, honours E1–E8 toggles → HandState
│   └── state.py            # HandState dataclass, Gesture enum
├── game/
│   ├── engine.py           # main loop, state machine, dt-based timing
│   ├── entities.py         # Fruit, Bomb, Half, Particle
│   ├── spawner.py          # launch physics and difficulty curve
│   ├── collision.py        # segment–circle intersection and speed check
│   ├── ui.py               # dwell buttons, HUD, calibration screen
│   ├── render.py           # alpha-blended sprites on the camera frame, blade trail, debug picture-in-picture
│   └── assets/             # PNG sprites with alpha, optional sounds
├── eval/
│   ├── record.py           # record test clips per condition
│   ├── annotate.py         # click palm centre / draw hand polygon → labels.json
│   └── evaluate.py         # run tracker variants on clips → metrics CSV, ablation table and plots
└── tests/                  # pytest: collision, filters, hue rotation, segmentation on synthetic images
```

**Rendering:** draw directly on the mirrored OpenCV frame (one window, one loop, no extra
dependency). Use `pygame.mixer` for sound only if time allows.

**Performance budget (30 fps = 33 ms):** vision ≤ 15 ms at 320×240, game and render ≤ 10 ms.
Haar face detection is the most expensive step, so it runs every 5th frame and its box is
reused in between. Kalman ROI search makes the steady-state cost smaller still.

**Debug overlay (`d`):** a 2×3 picture-in-picture grid showing the white-balanced frame,
reliability gate, back-projection map, final mask, and contour with hull, defects and palm
circle. The Kalman prediction (hollow) and measurement (solid) are drawn too. This doubles as
report figures.

---

## 6. Testing

- **Unit tests (pytest, no camera):**
  - segment–circle collision edge cases
  - the One Euro filter reduces variance on a synthetic noisy track
  - Kalman coasting
  - hue rotation maps 175 and 5 into one interval
  - ratio histogram down-weights a colour present in both hand and background samples
  - finger count on synthetic hand silhouettes
- **Replay mode:** `--video clip.mp4` runs the full game or tracker on recorded footage, so
  debugging is deterministic and testing works without a webcam.
- **Evaluation harness** (§7) acts as the integration test: a change must not lower metrics.

---

## 7. Evaluation (baseline vs enhanced)

**Dataset.** Record 20–30 s clips with `eval/record.py`. Include one 5 s segment with the hand
held still, for the jitter metric. Conditions:

| # | Condition | Stresses |
|---|---|---|
| C1 | Bright, neutral light, plain background | Sanity check |
| C2 | Dim light | E2 gating, E4 |
| C3 | Warm (yellow) lamp | E1 white balance, E3 calibration |
| C4 | Cluttered background with skin-like objects (wood, cardboard) | E4 ratio histogram, E5, E6 |
| C5 | Face, both arms and short sleeves in view | E6 face and scoring, E7 arm cut |
| C6 | Fast swipes | E8 Kalman and One Euro |

If possible, record with 2–3 people of different skin tones. Per-user calibration (E3) is
meant to fix the fairness problem that fixed thresholds have.

**Labels.** Every 10th frame, click the palm centre. For about 20 frames per condition, also
draw a hand polygon to get a mask. Optionally, MediaPipe landmarks can produce palm-centre
references at scale, for evaluation only.

**Metrics:**
- Segmentation: precision, recall, IoU against the polygon masks.
- Tracking: palm-centre error (mean, median, % within 20 px), detection rate, and track-loss
  events per minute.
- Smoothness: positional std during the still segment (jitter), and lag (cross-correlation
  delay against the raw signal in C6).
- Speed: FPS and ms per stage.
- Gameplay: in **PRACTICE** mode, 30 targets appear at fixed positions. Record hit rate and
  mean time-to-hit for baseline vs enhanced.

**Ablation.** Add enhancements cumulatively: Baseline → +E1 → +E2 → … → +E8, one row per
step and columns for metrics × conditions. Also run leave-one-out on the full pipeline to find
the enhancements that matter most. `eval/evaluate.py` produces the CSV, a markdown table and
bar charts for the report.

---

## 8. Milestones

| Phase | Work | Exit criterion | Estimate |
|---|---|---|---|
| **0. Setup** | Skeleton, `requirements.txt`, capture with mirroring and FPS counter, `--video` replay, `HandState` | Webcam window shows FPS; replay works | ½ day |
| **1. Vertical slice** | Baseline tracker, spawner and physics, slicing collision, HUD, lives, game over | **Playable end-to-end with the baseline** | 1½ days |
| **2. Eval harness first** | `record.py`, `annotate.py`, `evaluate.py`; record C1–C6; measure the baseline | Baseline row of the ablation table filled in | 1 day |
| **3. Enhancements** | E1 → E8, each behind a toggle, unit-tested, and measured as soon as it lands | Every "Must" in §4 done and measured | 3–4 days |
| **4. Game polish** | Calibration screen, dwell menus, gestures, auto-pause, difficulty ramp, PRACTICE mode, sprites and particles, debug overlay, optional sound | Gesture-only play through the menus | 1½ days |
| **5. Report and demo** | Final ablation, pipeline-stage figures, README, 2–3 min demo video showing live toggles | Deliverables submitted | 1 day |

Phase 1 gives a working game early. Phase 2 makes sure every later change is measured rather
than guessed at.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Face overlaps or out-scores the hand | Face-box penalty and Kalman proximity in contour scoring (E6). The calibration screen asks the player to keep their hand to the side of their face. |
| Forearm in the mask (short sleeves) | Distance-transform palm centre and 2.5r cut (E7). For the demo, long sleeves are a valid fallback. |
| Camera auto-exposure or white balance drifts mid-game | Lock AE/AWB where the backend allows (E1), gray-world WB, and the adaptive histogram (E8c) |
| Adaptive histogram drifts onto the background | Update only from the eroded contour, only when confidence is high, small α, and reset on `c` or a long loss |
| Background subtraction absorbs a still hand | Motion is only a scoring term, never a hard pixel gate (E6) |
| Finger counting is unreliable | Menus use dwell-to-select. Gestures are shortcuts only, debounced over 8 frames. |
| Below 30 fps on a weak laptop | 320×240 processing, face detection every 5 frames, ROI search, per-stage timing in debug mode |
| Skin-coloured walls or furniture at home | Discriminative ratio histogram (E4) and recalibration; record condition C4 deliberately |

---

## 10. Deliverables checklist

- [ ] Source code (structure in §5) with `README.md` covering how to run it, the controls and calibration tips
- [ ] `config.yaml` with tuned parameters and an example `calibration.json`
- [ ] Unit tests passing (`pytest`)
- [ ] Recorded evaluation clips (or a link) and `labels.json`
- [ ] Ablation table and charts (baseline vs each enhancement, per condition)
- [ ] Report: problem, baseline, each enhancement (why, how, effect), results, limitations
- [ ] Demo video: gameplay plus a live baseline↔enhanced switch and toggles under difficult lighting

## References
- Swain & Ballard (1991), *Color Indexing*: histogram back-projection and the ratio histogram.
- Bradski (1998), *Computer Vision Face Tracking for Use in a Perceptual User Interface*: CamShift.
- Chai & Ngan (1999), *Face Segmentation Using Skin-Color Map in Videophone Applications*: YCrCb skin range.
- Casiez, Roussel & Vogel (2012), *1€ Filter*: speed-adaptive low-pass filter.
- Viola & Jones (2001), *Rapid Object Detection using a Boosted Cascade of Simple Features*: Haar face detection.
