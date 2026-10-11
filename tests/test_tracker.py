import numpy as np
import pytest

from vision.calibration import CalibrationError
from vision.tracker import ENHANCEMENTS, HandTracker, cumulative, preset

from .synth import GLOVE, calibration_frames, make_frame

FACE = (320, 110)


def calibrated(cfg, kind="enhanced", hands=1, toggles=None, face=FACE):
    tr = HandTracker(cfg, hands=hands, toggles=toggles or preset(kind))
    frames, _ = calibration_frames(cfg, face=face)
    for f in frames:
        tr.add_calibration_frame(f)
    tr.finish_calibration((640, 480))
    return tr


def path(i):
    return 200 + 4 * i, 300 - 2 * i


def test_calibration_centres_hue(cfg):
    m = calibrated(cfg).model
    assert m.lo[0] <= 90 <= m.hi[0]


def test_calibration_rejects_grey_box(cfg):
    tr = HandTracker(cfg)
    for i in range(10):
        tr.add_calibration_frame(make_frame([], background=(110, 110, 110), seed=i))
    with pytest.raises(CalibrationError):
        tr.finish_calibration((640, 480))


def test_enhanced_tracks_bare_palm(cfg):
    tr = calibrated(cfg, face=None)
    errors = []
    for i in range(60):
        x, y = path(i)
        states = tr.process(make_frame([(x, y)], face=None, seed=100 + i), i / 30)
        if states:
            errors.append(np.hypot(states[0].palm[0] - x, states[0].palm[1] - y))
    assert len(errors) >= 55
    assert np.median(errors) < 8


def test_enhanced_beats_baseline(cfg):
    medians = {}
    for kind in ("baseline", "enhanced"):
        tr = calibrated(cfg, kind, face=None)
        errs = []
        for i in range(40):
            x, y = path(i)
            states = tr.process(make_frame([(x, y)], face=None, seed=200 + i), i / 30)
            if states:
                errs.append(np.hypot(states[0].palm[0] - x, states[0].palm[1] - y))
        medians[kind] = np.median(errs)
    assert medians["enhanced"] < 0.5 * medians["baseline"]


def test_glove_avoids_face_baseline_does_not(cfg):
    tr = calibrated(cfg, "baseline", hands=2)
    states = tr.process(make_frame([(150, 330)], face=FACE), 0.0)
    assert any(np.hypot(s.palm[0] - FACE[0], s.palm[1] - FACE[1]) < 40 for s in states)

    tr = HandTracker(cfg, hands=2)
    frames, _ = calibration_frames(cfg, face=FACE, color=GLOVE)
    for f in frames:
        tr.add_calibration_frame(f)
    tr.finish_calibration((640, 480))
    for i in range(5):
        states = tr.process(make_frame([{"center": (150, 330), "color": GLOVE}], face=FACE, seed=i), i / 30)
    assert len(states) == 1


def test_two_hands_found(cfg):
    tr = calibrated(cfg, hands=2)
    for i in range(6):
        states = tr.process(make_frame([(140, 330), (500, 330)], face=FACE, seed=i), i / 30)
    xs = sorted(s.palm[0] for s in states)
    assert len(xs) == 2 and abs(xs[0] - 140) < 15 and abs(xs[1] - 500) < 15


@pytest.mark.parametrize("k", range(len(ENHANCEMENTS) + 1))
def test_every_cumulative_variant_runs(cfg, k):
    tr = calibrated(cfg, toggles=cumulative(k))
    tr.debug_enabled = True
    for i in range(5):
        tr.process(make_frame([path(i)], face=FACE, seed=i), i / 30)
    assert set(tr.debug) >= {"input", "mask", "result"}


def test_toggling_white_balance_rebuilds_model(cfg):
    tr = calibrated(cfg)
    before = tr.model
    t = preset("enhanced")
    t["E1"] = False
    tr.set_toggles(t)
    assert tr.model is not None and tr.model is not before


def test_save_and_load(cfg, tmp_path):
    tr = calibrated(cfg)
    tr.save(tmp_path / "calib.npz")
    other = HandTracker(cfg)
    other.load(tmp_path / "calib.npz")
    assert np.allclose(other.model.ratio, tr.model.ratio)


@pytest.mark.parametrize("glove", [(60, 170, 60), (200, 60, 200), (40, 40, 200)])
def test_tracks_coloured_glove(cfg, glove):
    bx, by, bw, bh = cfg["vision"]["calib_box"]
    center = ((bx + bw / 2) * 640, (by + bh / 2) * 480)
    tr = HandTracker(cfg)
    for i in range(cfg["vision"]["calib_frames"]):
        tr.add_calibration_frame(make_frame([{"center": center, "color": glove}], face=FACE, seed=i))
    tr.finish_calibration((640, 480))
    errors = []
    for i in range(40):
        x, y = path(i)
        states = tr.process(make_frame([{"center": (x, y), "color": glove}], face=FACE, seed=50 + i), i / 30)
        for s in states:
            assert np.hypot(s.palm[0] - FACE[0], s.palm[1] - FACE[1]) > 80
        if states:
            errors.append(np.hypot(states[0].palm[0] - x, states[0].palm[1] - y))
    assert len(errors) >= 35 and np.median(errors) < 8
