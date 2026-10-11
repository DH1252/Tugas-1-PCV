import cv2
import numpy as np

from vision import segmentation
from vision.calibration import tune_kernels
from vision.tracker import HandTracker

from .shaded import shaded_frame


def test_hysteresis_keeps_weak_pixels_touching_strong():
    prob = np.zeros((40, 80), np.uint8)
    prob[10:30, 5:35] = 60
    prob[18:22, 15:20] = 200
    prob[10:30, 50:70] = 60
    mask = segmentation.hysteresis(prob, 40, 150)
    assert mask[12, 7] == 255 and mask[12, 60] == 0


def test_kernels_scale_with_box():
    small, big = tune_kernels((0, 0, 30, 30)), tune_kernels((0, 0, 60, 60))
    assert big["close_kernel"] > small["close_kernel"]
    assert all(k % 2 == 1 for k in (*small.values(), *big.values()))


def calibrate(cfg):
    bx, by, bw, bh = cfg["vision"]["calib_box"]
    tr = HandTracker(cfg)
    for i in range(cfg["vision"]["calib_frames"]):
        tr.add_calibration_frame(shaded_frame(((bx + bw / 2) * 640, (by + bh / 2) * 480), seed=i)[0])
    tr.finish_calibration((640, 480))
    return tr


def coverage(tr):
    recall = []
    for i in range(15):
        frame, hand = shaded_frame((220 + 10 * i, 320), seed=60 + i)
        tr.process(frame, i / 30)
        mask = cv2.resize(tr.mask, (640, 480), interpolation=cv2.INTER_NEAREST) > 0
        recall.append((mask & (hand > 0)).sum() / (hand > 0).sum())
    return float(np.mean(recall))


def test_auto_tune_fills_shaded_hand(cfg):
    auto = calibrate(cfg)
    assert {"backproject_thresh", "backproject_low", "s_min", "close_kernel"} <= set(auto.tuned)
    assert auto.param("backproject_low") <= auto.param("backproject_thresh")
    cfg["vision"]["auto_tune"] = False
    fixed = calibrate(cfg)
    assert fixed.tuned == {}
    a, f = coverage(auto), coverage(fixed)
    assert a > 0.85 and a > f + 0.1


def test_slider_override_beats_auto(cfg):
    tr = calibrate(cfg)
    tr.override("backproject_thresh", 123)
    assert tr.param("backproject_thresh") == 123
    tr.finish_calibration((640, 480))
    assert tr.param("backproject_thresh") != 123 or tr.tuned["backproject_thresh"] == 123
