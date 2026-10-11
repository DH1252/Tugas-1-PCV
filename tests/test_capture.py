import threading
import time

import numpy as np
import pytest

from vision import capture


class FakeCapture:
    opened = []

    def __init__(self, index, api=None):
        self.api = api
        self.n = 0
        self.hang = FakeCapture.hang_after
        self.released = False
        FakeCapture.opened.append(self)

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 30.0

    def read(self):
        if self.hang is not None and self.n >= self.hang:
            threading.Event().wait(3600)
        self.n += 1
        time.sleep(0.005)
        return True, np.full((480, 640, 3), self.n % 255, np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def fake(monkeypatch, cfg):
    FakeCapture.opened = []
    FakeCapture.hang_after = None
    monkeypatch.setattr(capture.cv2, "VideoCapture", FakeCapture)
    cfg["camera"].update(stall_timeout=0.3, read_timeout=0.2)
    return cfg


def test_reads_latest_frames(fake):
    src = capture.Source(fake)
    frames = [src.read() for _ in range(5)]
    assert all(f is not None and f.shape == (480, 640, 3) for f in frames)
    assert src.index == 5
    src.release()


def test_reopens_after_freeze(fake):
    FakeCapture.hang_after = 3
    src = capture.Source(fake)
    start = time.perf_counter()
    for _ in range(10):
        assert src.read() is not None
    assert src.restarts >= 1
    assert len(FakeCapture.opened) >= 2
    assert time.perf_counter() - start < 5
    src.release()


def test_read_never_blocks_long(fake):
    FakeCapture.hang_after = 2
    src = capture.Source(fake)
    t0 = time.perf_counter()
    src.read()
    src.read()
    src.read()
    assert time.perf_counter() - t0 < 1.0
    src.release()
