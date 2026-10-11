import os
import sys
import threading
import time

# MSMF hardware transforms make the camera take seconds to open and can stall reads
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402

BACKENDS = {
    "auto": cv2.CAP_ANY,
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
}


def open_camera(cam):
    name = cam.get("backend", "msmf")
    api = BACKENDS.get(name, cv2.CAP_ANY)
    if name in ("msmf", "dshow") and sys.platform != "win32":
        api = cv2.CAP_ANY
    cap = cv2.VideoCapture(cam["index"], api)
    if not cap.isOpened() and api != cv2.CAP_ANY:
        cap = cv2.VideoCapture(cam["index"])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {cam['index']}")
    fourcc = cam.get("fourcc")
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def lock_auto(cap):
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    # manual exposure is 0.25 on DirectShow/MSMF and 1 on V4L2
    for value in (0.25, 1):
        if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value):
            break


def list_cameras(cam, max_index=8):
    found = []
    for i in range(max_index):
        try:
            cap = open_camera({**cam, "index": i})
        except RuntimeError:
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            found.append((i, frame.shape[1], frame.shape[0]))
    return found


class Reader(threading.Thread):
    def __init__(self, cam, on_frame):
        super().__init__(daemon=True)
        self.cam = cam
        self.on_frame = on_frame
        self.running = True
        self.lock_requested = False
        self.error = None
        self.fps = 0.0

    def run(self):
        try:
            cap = open_camera(self.cam)
        except Exception as e:
            self.error = e
            return
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while self.running:
            if self.lock_requested:
                lock_auto(cap)
                self.lock_requested = False
            ok, frame = cap.read()
            if not self.running:
                break
            if ok and frame is not None:
                self.on_frame(self, frame)
            else:
                time.sleep(0.01)
        cap.release()

    def stop(self):
        self.running = False


class Source:
    def __init__(self, cfg, video=None):
        cam = cfg["camera"]
        self.cam = cam
        self.size = (cam["width"], cam["height"])
        self.is_file = video is not None
        self.mirror = cam["mirror"] and not self.is_file
        self.index = 0
        self.restarts = 0
        self.locked = False
        if self.is_file:
            self.cap = cv2.VideoCapture(video)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open {video}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            return
        self.stall = cam.get("stall_timeout", 2.0)
        self.wait = cam.get("read_timeout", 1.0)
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.seen = 0
        self.stamp = time.perf_counter()
        self.reader = None
        self._start_reader()
        self._wait_first(cam.get("open_timeout", 10.0))
        self.fps = self.reader.fps or 30.0

    def _start_reader(self):
        if self.reader is not None:
            self.reader.stop()
        self.reader = Reader(self.cam, self._on_frame)
        self.reader.lock_requested = self.locked
        self.stamp = time.perf_counter()
        self.reader.start()

    def _on_frame(self, reader, frame):
        with self.cond:
            if reader is not self.reader:
                return
            self.frame = frame
            self.seq += 1
            self.stamp = time.perf_counter()
            self.cond.notify_all()

    def _wait_first(self, timeout):
        deadline = time.perf_counter() + timeout
        with self.cond:
            while self.frame is None:
                if self.reader.error is not None:
                    raise RuntimeError(str(self.reader.error))
                left = deadline - time.perf_counter()
                if left <= 0:
                    self.reader.stop()
                    raise RuntimeError(f"camera {self.cam['index']} gave no frames")
                self.cond.wait(min(left, 0.1))

    def switch(self, index):
        if self.is_file or index == self.cam["index"]:
            return False
        old = self.cam
        for cam in ({**old, "index": index}, old):
            self.cam = cam
            with self.cond:
                self.frame = None
                self.seen = self.seq
                self._start_reader()
            try:
                self._wait_first(self.cam.get("open_timeout", 10.0))
            except RuntimeError:
                continue
            self.fps = self.reader.fps or 30.0
            return cam is not old
        raise RuntimeError("no camera available")

    @property
    def stalled(self):
        return not self.is_file and time.perf_counter() - self.stamp > self.stall

    def _latest(self):
        deadline = time.perf_counter() + self.wait
        with self.cond:
            while self.seq == self.seen:
                if self.stalled:
                    self.restarts += 1
                    print(f"camera stalled, reopening (#{self.restarts})", file=sys.stderr)
                    self._start_reader()
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                self.cond.wait(min(left, 0.05))
            self.seen = self.seq
            return None if self.frame is None else self.frame.copy()

    def read(self):
        if self.is_file:
            ok, frame = self.cap.read()
            if not ok:
                return None
        else:
            frame = self._latest()
            if frame is None:
                return None
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        self.index += 1
        return frame

    def rewind(self):
        if self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.index = 0

    def lock_auto(self):
        if not self.is_file:
            self.locked = True
            self.reader.lock_requested = True

    def release(self):
        if self.is_file:
            self.cap.release()
        else:
            self.reader.stop()
