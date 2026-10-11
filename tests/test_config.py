import pytest

from vision.config import apply_resolution, parse_size


def test_parse_size():
    assert parse_size("720p") == (1280, 720)
    assert parse_size("1024X576") == (1024, 576)
    for bad in ("big", "10x10", "1280"):
        with pytest.raises(ValueError):
            parse_size(bad)


def test_apply_resolution(cfg):
    cam = apply_resolution(cfg, "1080p")["camera"]
    assert (cam["capture_width"], cam["capture_height"]) == (1920, 1080)
    assert (cam["width"], cam["height"]) == (853, 480)
    cam = apply_resolution(cfg, "480p")["camera"]
    assert (cam["width"], cam["height"]) == (640, 480)
    cam = apply_resolution(cfg, None, "1280x720")["camera"]
    assert (cam["width"], cam["height"]) == (1280, 720)
