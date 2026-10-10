# Keep It Up: Plan for an HSV Hand-Tracking Game

> **Status:** phases 0–5 are implemented and tested on synthetic frames (see §11). Still to
> do: tuning on a real webcam, recording and labelling C1–C7, and the report.

A webcam keep-up game. Balls and other items fall from the top of the screen, and you use
your hands to bounce them back up so they never reach the bottom. The hands are tracked with
**HSV colour segmentation**, and the assignment's technical contribution is an **enhanced HSV
pipeline** that we compare against a plain fixed-threshold baseline.

---

## 1. Goals and scope

| Goal | How we'll show it |
|---|---|
| Playable game driven only by hand movement (one or two hands) | Live demo plus a recorded video |
| Baseline HSV tracker (fixed thresholds) | `--tracker baseline` |
| Enhanced HSV tracker, with each enhancement switchable on and off | `--tracker enhanced` and live hotkeys `1`–`9` |
| Measured improvement of enhanced over baseline | Ablation table from the `eval/` harness on recorded clips |

**Constraint: colour-based tracking only.** No MediaPipe, and no learned detectors of any
kind (no Haar cascades, DNNs or YOLO), not even for evaluation. The tracker uses only:
- HSV and closely related colour techniques: YCrCb, histogram back-projection, CamShift
- classical image processing: blur, white balance, morphology, contours, convex hull,
  distance transform, frame differencing
- filtering: Kalman and One Euro

**Assumptions:** a laptop webcam at 640×480 and 30 fps or better, Python 3.10+, and a single
developer working about 2 weeks. Adjust §8 if the deadline or team size differs.

---

## 2. Game design

### Core loop
1. Balls drop from random positions along the top edge, with a small random sideways speed.
2. Each tracked hand is a round **paddle** centred on the smoothed palm, with a radius of about
   2× the palm radius. It is drawn as a translucent ring so the player can see what the game
   thinks their hand is.
3. When a ball touches a paddle, it bounces off. The hand's velocity is added to the bounce,
   so swatting upward sends the ball higher than holding the hand still. A gentle touch from
   below still lifts the ball by at least a minimum upward speed.
4. Balls bounce off the side and top walls. **A ball that reaches the bottom edge costs 1
   life** and respawns from the top after 1 s. You start with 3 lives.
5. **Score:** each hit scores +1 × the number of balls in play, so juggling more balls pays
   more. Hits in a row without a drop build a streak bonus.
6. **Item types:**

   | Item | Behaviour | What it asks of the player |
   |---|---|---|
   | Ball | Normal gravity and bounce | The basic task |
   | Balloon | Low gravity, high air drag | Easy, but drifts sideways |
   | Heavy ball | High gravity, low bounce | Needs a firm upward swat |
   | Spike ball | Touching it costs a life | Let it fall; get out of the way |
   | Star | Collect by touching it | +5 points, or +1 life if you have fewer than 3 |

7. **Difficulty ramp:** play starts with 1 ball. Another is added every 20 s, up to 4 at once.
   Gravity rises 5% every 30 s, and heavy and spike items become more common.

### Physics (in a 640×480 game space, scaled up for display)
- Integration is time-based (`dt`), so the game speed doesn't depend on frame rate.
- **Hand–ball collision.** Let n be the unit vector from the hand centre to the ball centre,
  u the hand velocity and v the ball velocity. When the ball overlaps the paddle and is moving
  toward it relative to the hand ((v − u)·n < 0):
  1. Reflect the relative velocity: `v ← u + (v−u) − (1+e)((v−u)·n)n`, with restitution
     e ≈ 0.85.
  2. If the ball is above the hand centre, raise its upward speed to at least 600 px/s if it
     is lower.
  3. Push the ball out so it no longer overlaps the paddle.
  4. Cap the ball's speed at 1500 px/s.

  The hand velocity u comes from the Kalman state and is limited to 2500 px/s, so tracking
  jitter can't fling balls away.
- **Anti-tunnelling.** A fast swat can move the hand 50 px or more between frames and pass
  straight through a ball. To prevent that, each frame is split into 4 substeps, with the hand
  position interpolated between the previous and current frame.
- Starting values: gravity 600 px/s² (a ball falls the full screen height in about 1.3 s),
  ball radius 22 px, wall restitution 0.9.

### States
```
CALIBRATE ──► MENU ──► PLAY ⇄ PAUSE ──► GAME_OVER ──► MENU
                 └──► PRACTICE (one slow ball; used for the gameplay metric in §7)
```

### Hand controls (no keyboard needed)
| Input | When | Action |
|---|---|---|
| Hand touches a ball | PLAY | Bounce. A faster upward swing gives a stronger hit. |
| Hold up 1, 2 or 3 fingers for 1 s | MENU | Choose option 1, 2 or 3 (Play, Practice, Recalibrate). Uses finger counting (E8). |
| Hover over a button for 1.2 s (radial progress ring) | MENU, GAME_OVER | Same as above. This works even when finger counting is unreliable. |
| Fist held 1 s | MENU, GAME_OVER | Confirm or start |
| Hover over the pause icon (top corner) for 1 s | PLAY | Pause |
| Open palm held 1 s | PAUSE | Resume |
| No hand seen for 1.5 s | PLAY | Auto-pause with a "Show your hands" prompt |

Gestures only do something outside PLAY, so a fist or open hand made while hitting balls
never triggers a menu action by accident.

### Keyboard (developer and demo controls)
`c` recalibrate · `d` debug overlay · `b` switch baseline/enhanced · `h` switch one/two hands ·
`1`–`9` toggle enhancements E1–E9 · `p` pause · `q`/`Esc` quit

### HUD
Score, lives, balls in play, streak, and a **tracking-confidence bar** for each hand. The bars
show the audience when the HSV tracker is struggling, which helps when presenting the
enhancements.

### Calibration screen (part of the game flow)
"Place your hand inside the box, and keep your face in view". A 2 s countdown, then about 60
frames are sampled. The resulting mask is shown so the player can accept it or retry. Both
hands share the same colour model, so one hand is enough for calibration.

---

## 3. Baseline HSV tracker (the comparison point)

```
BGR → HSV → cv2.inRange((0,48,80), (20,255,255))
    → erode×2 / dilate×2 (3×3) → largest contour(s) → centroid (moments)
```
It takes the largest contour in one-hand mode, or the two largest in two-hand mode. The
paddle radius is fixed. It has no smoothing, no calibration and no temporal logic. Known
failure modes:
- Fixed thresholds suit some skin tones and lighting and fail on others.
- Warm or cool light shifts the hue.
- Reddish skin hue wraps around H = 179/0.
- Face, arm and wooden or beige backgrounds are picked up as a hand. In two-hand mode the
  face often wins the second slot.
- The centroid drifts toward the forearm.
- Frame-to-frame jitter makes the paddle shake and gives balls random kicks.
- When the two hands cross, their identities swap.

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
   ▼                                                                         │ [E9c] adaptive
 [E6] denoise: opening (remove specks) → closing (fill gaps) · fill holes    │  histogram
   ▼                                                                         │  update (EMA)
 [E7] face removal · contour filter and scoring                              │
   ▼                                                                         │
 [E8] palm centre (distance transform) · arm cut · hull/defects → fingers ───┘
   ▼
 [E9] per-hand Kalman (gating, search ROI, coasting) · One Euro smoothing · two-hand assignment
   ▼
 list[HandState{id, found, palm, radius, velocity, fingers, gesture, confidence, contour}]
```

### Enhancements

| ID | Enhancement | Problem it fixes | Implementation notes | Priority |
|---|---|---|---|---|
| **E1** | Preprocessing: blur and **gray-world white balance** | Sensor noise; colour casts from warm or cool lamps shift hue | Scale B, G, R so their means are equal, then divide the gains by the largest one so the step only darkens channels. Boosting a channel would clip bright skin to V = 255, and E2 would then throw it away as glare. Gains are smoothed over time. Try `CAP_PROP_AUTO_WB`/`CAP_PROP_AUTO_EXPOSURE` (backend-dependent) so the camera stops re-tuning mid-game. | Must |
| **E2** | **Hue rotation and reliability gating** | (a) Skin hue straddles 179↔0, so it splits into two ranges. (b) Hue is undefined or noisy when S or V is low (greys, shadows) and in specular highlights. | (a) Use a LUT to add +20 to H mod 180, so skin becomes one range (about 5–45). (b) Drop pixels with S < 30, V < 40 or V > 250 before any hue test. | Must |
| **E3** | **Per-user calibration** | Hard-coded ranges fail across skin tones and rooms | Sample the ROI over 40 frames and build a 2-D H–S histogram (30×32 bins), keeping only the bins that hold 90% of the mass so stray background pixels in the box are dropped. Also store 2nd–98th percentile H, S, V and Cr, Cb as fallback box thresholds. Calibration is refused if the box doesn't look like skin. The raw calibration frames are saved to `calibration.npz`, so the model can be rebuilt when E1 or E2 is toggled. | Must |
| **E4** | **Discriminative histogram back-projection** (Swain & Ballard ratio histogram) | A box threshold is binary and also takes in background colours that look like skin | `P(hand | colour) = H_hand / (H_hand + H_bg)`, a bounded form of the ratio histogram, where `H_bg` is sampled outside the ROI at calibration. The face and the arm are left out of `H_bg`, or skin would cancel itself out. `cv2.calcBackProject` gives a per-pixel probability, then blur and threshold it (τ = 90/255). Colours common in the background are down-weighted automatically. | Must |
| **E5** | **Hybrid colour-space fusion**: AND with a YCrCb gate | Some false positives are separable in Cr/Cb but not in H–S | The Cr/Cb range comes from calibration percentiles ± a margin. Fall back to Chai & Ngan (Cr 133–173, Cb 77–127). It is loose on purpose, a veto rather than the main detector. | Should |
| **E6** | **Morphological denoising: opening then closing** | The raw mask has isolated false pixels (sensor noise, small skin-coloured patches in the background) and holes inside the hand (shadows, highlights, knuckle creases). Fingers can also break away from the palm. | **Opening** (erode then dilate, elliptical 5×5 kernel) removes specks smaller than the kernel without shrinking the hand. **Closing** (dilate then erode, elliptical 9×9) fills holes and reconnects fingers. Opening comes first so closing doesn't grow the noise. Holes still left are filled by redrawing each outer contour filled. Elliptical kernels keep the hand outline rounder than square ones. The baseline only has a square 3×3 erode/dilate (an opening with no closing), so this step is measured on its own. If closing merges the fingers, E8 counts fingers on a mask closed with a smaller kernel. | Must |
| **E7** | **Face removal and contour scoring** | The face or other skin blobs being picked as a hand | **Face removal without a detector:** at calibration, the largest skin blob outside the hand box is taken as the face. It is then tracked with **mean shift** on the same back-projection map, which is CamShift (Bradski 1998) with the window size fixed at the calibrated face size. Letting the size adapt made the window creep onto a nearby hand. The window freezes when the probability mass inside it jumps by more than 35%, which means a hand has entered it. Its ellipse plus a margin is removed from the hand mask. Keep contours with area 1.5–40% of the frame and solidity ≥ 0.45. Rank them by score = size + motion (overlap with a frame-difference mask) + closeness to a Kalman prediction − a penalty for being still, elliptical and without finger gaps. Take the top 1 or 2. | Must (scoring), Should (face, motion) |
| **E8** | **Hand geometry** | The centroid drifts toward the forearm; paddle size and gestures are needed | Palm centre = arg-max of `cv2.distanceTransform`, with palm radius r = its value. The paddle radius is 2r. Cut the mask outside a 3r circle and below a wrist line 1.3r under the palm centre, which removes the forearm (this assumes the arm comes from below). The cut mask is also what the IoU metric scores. Use `convexHull` + `convexityDefects` with depth > 0.4r and angle < 90°, so fingers = defects + 1. Treat it as a fist if the farthest hull point is < 1.6r away. Debounce with a majority vote over 8 frames. | Must (palm centre), Should (fingers) |
| **E9** | **Temporal filtering and two-hand tracking** | Jitter, short dropouts, slow full-frame search, hands swapping identity | (a) A `cv2.KalmanFilter` per hand with constant velocity. It predicts the next position, rejects jumps more than 100 px from the prediction (the gate widens while coasting), carries the paddle through up to 5 lost frames, and boosts candidates near the prediction in E7's scoring. At 320×240 a full-frame search already takes about 3 ms, so no ROI cropping is done. Its velocity estimate drives the ball physics. (b) A **One Euro filter** (Casiez et al., 2012) on the paddle position: little smoothing when the hand is slow, so the paddle stays steady, and little lag when it is fast, so swats land. (c) **Adaptive model update**: every 10 frames, if confidence > 0.7, `H_hand ← (1−α)H_hand + α·H_current` with α ≈ 0.03, using pixels from the *eroded* contour only. Reset on `c` or after a long loss so drift can't build up. (d) **Two hands:** match blobs to tracks by trying both pairings and keeping the one with the smaller total distance to the predictions. A track starts after 3 frames in a row with a detection and ends after 5 frames of coasting. When the hands touch and merge into one blob, the track left without a match holds on, with its velocity damped, for up to 2 s while its prediction lies inside the merged blob. | Must (a, b), Should (d), Could (c) |

**Confidence** combines the mean back-projection inside the contour, solidity, whether the
area is plausible, and how far the measurement is from the Kalman prediction. It feeds the HUD
bars, the adaptive update gate and the auto-pause.

### Starting parameters (tune live with debug-mode trackbars, save to `config.yaml`)
- **Vision:** processing size 320×240 · blur 5×5 · S ≥ 30, 40 ≤ V ≤ 250 · hue shift +20 ·
  H–S bins 30×32 · back-projection τ = 90 · open 5×5 / close 9×9 (elliptical) · area 0.8–40% ·
  solidity ≥ 0.45 · face margin 15%
- **Tracking:** Kalman gate 100 px, max coast 5 frames, max merge 60 frames, track birth 3 frames · One Euro
  `min_cutoff=1.0, beta=0.02, d_cutoff=1.0` · α = 0.03 every 10 frames
- **Physics:** gravity 600 px/s² · e = 0.85 (hand), 0.9 (walls) · min upward speed 600 px/s ·
  max ball speed 1500 px/s · max hand speed 2500 px/s · 4 substeps per frame

---

## 5. Architecture

Vision and game are separate. The game only sees a list of `HandState`, so swapping baseline
and enhanced, one or two hands, or live camera and recorded video, needs no game changes.

```
tugas-1-pcv/
├── main.py                 # python main.py [--tracker baseline|enhanced] [--hands 1|2] [--video clip.avi] [--debug]
├── config.yaml             # all tunables from §4
├── requirements.txt        # opencv-python, numpy, pyyaml, pytest, matplotlib
├── vision/
│   ├── capture.py          # webcam or video-file source, mirroring, AE/AWB lock
│   ├── preprocess.py       # E1, E2
│   ├── calibration.py      # E3 (+ background hist for E4, face seed for E7)
│   ├── segmentation.py     # E4, E5, E6, E7 (mask, opening/closing, frame differencing, contour scoring)
│   ├── face.py             # E7 mean-shift face track
│   ├── hand.py             # E8 geometry and finger count
│   ├── filters.py          # E9 Kalman, One Euro, tracks, assignment
│   ├── tracker.py          # HandTracker: runs the stages, honours E1–E9 toggles → list[HandState].
│   │                       #   The baseline (§3) is this tracker with every toggle off.
│   └── state.py            # HandState dataclass
├── game/
│   ├── engine.py           # state machine, dt-based timing, scoring, practice log
│   ├── entities.py         # Item (ball, balloon, heavy, spike, star), Paddle
│   ├── physics.py          # integration, walls, substeps, hand–ball collision
│   ├── spawner.py          # spawning and difficulty curve
│   ├── ui.py               # dwell buttons, gesture holds
│   └── render.py           # items drawn with OpenCV shapes, paddle rings, HUD, debug grid
├── eval/
│   ├── record.py           # record test clips per condition (+ calibration and still-segment metadata)
│   ├── annotate.py         # click palm centres (A/B) / draw hand polygons → <clip>.labels.json
│   └── evaluate.py         # run tracker variants on clips → results.csv, ablation.md, charts
└── tests/                  # pytest on synthetic frames and clips (tests/synth.py, tests/clips.py)
```

**Rendering:** draw directly on the mirrored OpenCV frame (one window, one loop, no extra
dependency). Items are drawn with OpenCV shapes, so there are no sprite files. Sound is not
implemented yet.

**Performance budget (30 fps = 33 ms):** vision ≤ 15 ms at 320×240, game and render ≤ 10 ms.
On synthetic frames the full enhanced tracker takes about 3 ms per frame.

**Debug overlay (`d`):** a 2×3 picture-in-picture grid showing the white-balanced frame,
reliability gate, back-projection map, final mask, and contours with hull, defects, palm
circle and face ellipse. The Kalman prediction (hollow) and measurement (solid) are drawn too.
This doubles as report figures.

---

## 6. Testing

- **Unit tests (pytest, no camera):**
  - hand–ball bounce: only when approaching, restitution applied, minimum upward speed,
    speed cap
  - anti-tunnelling: a hand moving 80 px in one frame still hits a ball in its path
  - the One Euro filter reduces variance on a synthetic noisy track
  - Kalman coasting
  - two-hand assignment keeps identities when two synthetic tracks cross
  - hue rotation maps 175 and 5 into one interval
  - opening removes specks smaller than its kernel, closing fills holes, and the hand's area changes by less than 5%
  - ratio histogram down-weights a colour present in both hand and background samples
  - palm centre and finger count on synthetic hand silhouettes
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
| C2 | Dim light (noisy sensor) | E2 gating, E4, E6 denoising |
| C3 | Warm (yellow) lamp | E1 white balance, E3 calibration |
| C4 | Cluttered background with skin-like objects (wood, cardboard) | E4 ratio histogram, E5, E7 |
| C5 | Face in view, hands passing in front of the face, short sleeves | E7 face removal, E8 arm cut |
| C6 | Fast upward swats | E9 Kalman and One Euro |
| C7 | Two hands crossing and touching | E9 two-hand assignment |

If possible, record with 2–3 people of different skin tones. Per-user calibration (E3) is
meant to fix the fairness problem that fixed thresholds have.

**Labels.** All labels are made by hand with `eval/annotate.py`. Every 10th frame, click each
palm centre, marked left or right. For about 20 frames per condition, also draw hand polygons
to get masks.

**Metrics:**
- Segmentation: precision, recall, IoU against the polygon masks.
- Tracking: palm-centre error (mean, median, % within 20 px), detection rate, track-loss events
  per minute, and identity swaps per minute (C7).
- Smoothness: positional std during the still segment (jitter), and lag (cross-correlation
  delay against the raw signal in C6).
- Speed: FPS and ms per stage.
- Gameplay: in **PRACTICE** mode, the same player plays 2 minutes with each tracker. Record
  hits per minute, drops per minute and the longest streak.

**Ablation.** Add enhancements cumulatively: Baseline → +E1 → +E2 → … → +E9, one row per
step and columns for metrics × conditions. Also run leave-one-out on the full pipeline to find
the enhancements that matter most. `eval/evaluate.py` produces the CSV, a markdown table and
bar charts for the report.

---

## 8. Milestones

| Phase | Work | Exit criterion | Estimate |
|---|---|---|---|
| **0. Setup** | Skeleton, `requirements.txt`, capture with mirroring and FPS counter, `--video` replay, `HandState` | Webcam window shows FPS; replay works | ½ day |
| **1. Vertical slice** | Baseline tracker (one hand), one ball type, physics with substeps, paddle ring, HUD, lives, game over | **Playable end-to-end with the baseline** | 1½ days |
| **2. Eval harness first** | `record.py`, `annotate.py`, `evaluate.py`; record C1–C7; measure the baseline | Baseline row of the ablation table filled in | 1 day |
| **3. Enhancements** | E1 → E9, each behind a toggle, unit-tested, and measured as soon as it lands | Every "Must" in §4 done and measured | 3–4 days |
| **4. Two hands** | Top-2 contour selection, two-hand assignment, face removal hardened for two-hand play | C7 identity swaps measured; two-hand play is stable | 1 day |
| **5. Game polish** | Calibration screen, finger-count and dwell menus, pause and resume, auto-pause, item types, difficulty ramp, PRACTICE mode, sprites, debug overlay, optional sound | Gesture-only play through the menus | 1½ days |
| **6. Report and demo** | Final ablation, pipeline-stage figures, README, 2–3 min demo video showing live toggles | Deliverables submitted | 1 day |

Phase 1 gives a working game early. Phase 2 makes sure every later change is measured rather
than guessed at.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Face picked as a hand, especially as the second hand | Mean-shift face track removed from the mask, plus a scoring penalty for still, elliptical blobs without finger gaps (E7) |
| Hand in front of the face is partly cut out with the face | Kalman coasting carries the paddle for up to 5 frames. The calibration screen suggests sitting so the face is near the top centre of the frame and most play happens beside it. |
| Forearm in the mask (short sleeves) | Distance-transform palm centre and 2.5r cut (E8). For the demo, long sleeves are a valid fallback. |
| Fast swats pass through balls | 4 physics substeps with interpolated hand positions, and Kalman velocity for the hit |
| Tracking lag makes hits feel late | One Euro filter instead of heavy smoothing. Could also predict the hand about 1 frame ahead from the Kalman velocity. |
| Jitter kicks balls randomly | One Euro smoothing, the hand-speed cap, and bouncing only when the ball is moving toward the hand |
| Two hands merge into one blob or swap identities | Both tracks coast while merged; the pairing with the smaller total distance decides identities when they split (E9d) |
| Camera auto-exposure or white balance drifts mid-game | Lock AE/AWB where the backend allows (E1), gray-world WB, and the adaptive histogram (E9c) |
| Adaptive histogram drifts onto the background | Update only from the eroded contour, only when confidence is high, small α, and reset on `c` or a long loss |
| Closing merges fingers, so finger counts drop | Keep the closing kernel smaller than the gap between fingers, and count fingers on a lightly closed mask (E6) |
| Finger counting is unreliable | Every menu choice also works by hovering. Gestures are debounced over 8 frames and do nothing during PLAY. |
| Below 30 fps on a weak laptop | 320×240 processing, ROI search, per-stage timing in debug mode |
| Skin-coloured walls or furniture at home | Discriminative ratio histogram (E4) and recalibration; record condition C4 deliberately |

---

## 10. Deliverables checklist

- [ ] Source code (structure in §5) with `README.md` covering how to run it, the controls and calibration tips
- [ ] `config.yaml` with tuned parameters
- [ ] Unit tests passing (`pytest`)
- [ ] Recorded evaluation clips (or a link) and `labels.json`
- [ ] Ablation table and charts (baseline vs each enhancement, per condition)
- [ ] Report: problem, baseline, each enhancement (why, how, effect), results, limitations
- [ ] Demo video: one- and two-hand play plus a live baseline↔enhanced switch and toggles under difficult lighting

## References
- Swain & Ballard (1991), *Color Indexing*: histogram back-projection and the ratio histogram.
- Bradski (1998), *Computer Vision Face Tracking for Use in a Perceptual User Interface*: CamShift.
- Chai & Ngan (1999), *Face Segmentation Using Skin-Color Map in Videophone Applications*: YCrCb skin range.
- Casiez, Roussel & Vogel (2012), *1€ Filter*: speed-adaptive low-pass filter.

---

## 11. Implementation status

| Phase | State |
|---|---|
| 0. Setup | Done |
| 1. Vertical slice | Done |
| 2. Eval harness | Scripts done and tested on a synthetic clip. Recording and labelling real clips C1–C7 is still to do. |
| 3. Enhancements E1–E9 | Done, each toggleable. Default parameters need tuning on a real webcam (debug trackbars). |
| 4. Two hands | Done. Identity through long merges is only partly kept; measure with C7. |
| 5. Game polish | Done except sound |
| 6. Report and demo | Not started |

Where the implementation differs from the plan above, the plan text has been updated: the
white balance only darkens, E4 uses the bounded `P(hand | colour)` form, the face is tracked
with fixed-size mean shift, the arm cut uses a wrist line, and there is no ROI cropping.
