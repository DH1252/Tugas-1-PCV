import cv2
import numpy as np
import pytest

from vision.preprocess import rg_chroma
from vision.tracker import HandTracker

from . import stress
from .synth import GLOVE, draw_hand

DISTRACTOR = ((60, 60), (140, 160))


def test_rg_chroma_ignores_brightness():
    px = np.array([[[60, 170, 60]]], np.uint8)
    dim = (px * 0.5).astype(np.uint8)
    assert np.abs(rg_chroma(px).astype(int) - rg_chroma(dim).astype(int)).max() <= 2


def tracker(cfg, distractor=None):
    tr = HandTracker(cfg)
    bx, by, bw, bh = cfg["vision"]["calib_box"]
    center = ((bx + bw / 2) * stress.SIZE[0], (by + bh / 2) * stress.SIZE[1])
    for i in range(cfg["vision"]["calib_frames"]):
        img = np.empty((stress.SIZE[1], stress.SIZE[0], 3), np.uint8)
        img[:] = (105, 95, 85)
        if distractor:
            cv2.rectangle(img, distractor[0], distractor[1], GLOVE, -1)
        draw_hand(img, center, color=GLOVE)
        tr.add_calibration_frame(img)
    tr.finish_calibration(stress.SIZE)
    return tr


@pytest.mark.parametrize("light,speed,exposure", [(0.5, 0.3, 1 / 60), (0.0, 1.6, 1 / 30), (0.5, 1.6, 1 / 60)])
def test_keeps_glove_through_light_and_blur(cfg_top, light, speed, exposure):
    lost, ids = stress.run(tracker(cfg_top), frames=150, anchor="top", light=light, speed=speed, exposure=exposure)
    assert lost < 0.08 and ids == 1


def test_ignores_distractor_seen_at_calibration(cfg_top):
    tr = tracker(cfg_top, DISTRACTOR)
    assert tr.model.distractors is not None
    lost, ids = stress.run(tr, frames=150, anchor="top", light=0.3, distractor=DISTRACTOR)
    assert lost < 0.05 and ids == 1


def test_recovers_from_distractor_added_later(cfg_top):
    tr = tracker(cfg_top)
    lost, ids = stress.run(tr, frames=150, anchor="top", light=0.3, distractor=DISTRACTOR)
    assert lost < 0.08


def test_palm_anchor_still_robust(cfg):
    lost, ids = stress.run(tracker(cfg), frames=150, light=0.5, speed=1.6, exposure=1 / 60)
    assert lost < 0.08 and ids == 1


@pytest.mark.parametrize("fade", [0.55, 0.8])
def test_glossy_glove_is_one_piece(cfg, fade):
    from . import glossy
    tr = HandTracker(cfg)
    glossy.calibrate(tr, cfg, fade=fade)
    covered, false = glossy.coverage(tr, frames=8, fade=fade)
    assert covered > 0.9 and false < 0.05
