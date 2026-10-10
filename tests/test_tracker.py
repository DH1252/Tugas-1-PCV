import numpy as np
import pytest

from vision.calibration import CalibrationError
from vision.tracker import ENHANCEMENTS, HandTracker, cumulative, preset

from .synth import calibration_frames, make_frame

FACE = (320, 110)


def calibrated(cfg, kind="enhanced", hands=1, toggles=None):
    tr = HandTracker(cfg, hands=hands, toggles=toggles or preset(kind))
    frames, _ = calibration_frames(cfg, face=FACE)
    for f in frames:
        tr.add_calibration_frame(f)
    tr.finish_calibration((640, 480))
    return tr


def path(i):
    return 200 + 4 * i, 300 - 2 * i


def test_calibration_finds_skin_box_and_face(cfg):
    tr = calibrated(cfg)
    m = tr.model
    assert m.lo[0] <= 33 <= m.hi[0]
    x, y, w, h = m.face_box
    assert abs((x + w / 2) / cfg["vision"]["scale"] - FACE[0]) < 10


def test_calibration_rejects_empty_box(cfg):
    tr = HandTracker(cfg)
    for i in range(10):
        tr.add_calibration_frame(make_frame([], seed=i))
    with pytest.raises(CalibrationError):
        tr.finish_calibration((640, 480))


def test_enhanced_tracks_palm_not_face(cfg):
    tr = calibrated(cfg)
    errors = []
    for i in range(60):
        x, y = path(i)
        states = tr.process(make_frame([(x, y)], face=FACE, seed=100 + i), i / 30)
        for s in states:
            assert np.hypot(s.palm[0] - FACE[0], s.palm[1] - FACE[1]) > 80
        if states:
            errors.append(np.hypot(states[0].palm[0] - x, states[0].palm[1] - y))
    assert len(errors) >= 55
    assert np.median(errors) < 8


def test_enhanced_beats_baseline(cfg):
    medians = {}
    for kind in ("baseline", "enhanced"):
        tr = calibrated(cfg, kind)
        errs = []
        for i in range(40):
            x, y = path(i)
            states = tr.process(make_frame([(x, y)], face=FACE, seed=200 + i), i / 30)
            if states:
                errs.append(np.hypot(states[0].palm[0] - x, states[0].palm[1] - y))
        medians[kind] = np.median(errs)
    assert medians["enhanced"] < 0.5 * medians["baseline"]


def test_baseline_mistakes_face_for_second_hand(cfg):
    tr = calibrated(cfg, "baseline", hands=2)
    states = tr.process(make_frame([(150, 330)], face=FACE), 0.0)
    assert any(np.hypot(s.palm[0] - FACE[0], s.palm[1] - FACE[1]) < 40 for s in states)

    tr = calibrated(cfg, "enhanced", hands=2)
    for i in range(5):
        states = tr.process(make_frame([(150, 330)], face=FACE, seed=i), i / 30)
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
