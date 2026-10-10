import cv2
import numpy as np

from vision import hand, segmentation
from vision.calibration import posterior
from vision.preprocess import WhiteBalance, hue_lut, reliability_gate, rotate_hue

from .synth import SKIN, draw_hand


def hsv_pixels(values):
    return np.array([values], np.uint8)


def test_hue_rotation_joins_wraparound():
    hsv = hsv_pixels([(175, 100, 100), (5, 100, 100)])
    rotated = rotate_hue(hsv, hue_lut(20))
    h = rotated[0, :, 0]
    assert list(h) == [15, 25]
    mask = segmentation.hsv_box(rotated, (10, 0, 0), (30, 255, 255))
    assert mask.all()


def test_hsv_box_handles_wrapped_range():
    hsv = hsv_pixels([(175, 100, 100), (5, 100, 100), (90, 100, 100)])
    mask = segmentation.hsv_box(hsv, (170, 0, 0), (10, 255, 255))
    assert list(mask[0] > 0) == [True, True, False]


def test_reliability_gate_drops_grey_dark_and_glare():
    hsv = hsv_pixels([(10, 120, 150), (10, 10, 150), (10, 120, 20), (10, 120, 254)])
    gate = reliability_gate(hsv, 30, 40, 250)
    assert list(gate[0] > 0) == [True, False, False, False]


def test_gray_world_equalises_channel_means():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:] = (60, 100, 160)
    out = WhiteBalance((0.3, 3.0))(img)
    means = out.reshape(-1, 3).mean(axis=0)
    assert means.max() - means.min() < 3
    assert out.max() <= 160


def test_ratio_histogram_downweights_shared_colours():
    hand_p = np.zeros((4, 4), np.float32)
    bg_p = np.zeros((4, 4), np.float32)
    hand_p[1, 1] = 0.5
    hand_p[2, 2] = 0.5
    bg_p[2, 2] = 0.9
    bg_p[0, 0] = 0.1
    r = posterior(hand_p, bg_p, 0.02)
    assert r[1, 1] > 250
    assert r[2, 2] < 0.5 * r[1, 1]
    assert r[0, 0] == 0


def test_denoise_removes_specks_and_fills_holes():
    mask = np.zeros((120, 160), np.uint8)
    cv2.circle(mask, (80, 60), 30, 255, -1)
    area = np.count_nonzero(mask)
    noisy = mask.copy()
    rng = np.random.default_rng(1)
    for x, y in rng.integers(0, [160, 120], (40, 2)):
        if np.hypot(x - 80, y - 60) > 40:
            noisy[y, x] = 255
    cv2.circle(noisy, (80, 60), 4, 0, -1)
    _, clean = segmentation.denoise(noisy, 5, 9)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1
    assert clean[60, 80] == 255
    assert abs(np.count_nonzero(clean) - area) / area < 0.05


def hand_mask(fingers, center=(160, 160), scale=1.0):
    img = np.zeros((360, 320, 3), np.uint8)
    r = draw_hand(img, center, scale=scale, fingers=fingers, color=SKIN)
    return cv2.inRange(img, SKIN, SKIN), r


def measure(mask, v):
    cands = segmentation.find_candidates(mask, 10, 0, 1e9, 0, False)
    c = cands[0]
    segmentation.blob_mask(c, mask.shape)
    hand.measure(c, cv2.distanceTransform(mask, cv2.DIST_L2, 5), mask, v)
    return c


def test_palm_centre_ignores_forearm(cfg):
    mask, r = hand_mask(5)
    c = measure(mask, cfg["vision"])
    assert np.hypot(c.center[0] - 160, c.center[1] - 160) < 0.3 * r
    assert abs(c.radius - r) < 0.2 * r
    centroid = cv2.moments(mask, binaryImage=True)
    assert centroid["m01"] / centroid["m00"] > 160 + 0.5 * r


def test_finger_count(cfg):
    for fingers in (0, 1, 2, 3, 5):
        mask, _ = hand_mask(fingers)
        assert measure(mask, cfg["vision"]).fingers == fingers
