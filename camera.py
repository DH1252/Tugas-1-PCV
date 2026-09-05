"""Webcam input, read on a background thread."""
import sys
import threading
import time

import cv2

# Windows backends: Media Foundation (msmf) and DirectShow (dshow).
BACKENDS = {"msmf": [cv2.CAP_MSMF], "dshow": [cv2.CAP_DSHOW],
            "auto": [cv2.CAP_MSMF, cv2.CAP_DSHOW] if sys.platform == "win32" else [cv2.CAP_ANY]}
# CAP_PROP_AUTO_EXPOSURE value for "auto" per backend (3 = Linux V4L2).
AUTO_EXPOSURE_ON = {cv2.CAP_MSMF: 1, cv2.CAP_DSHOW: 0.75}


def open_camera(index, backend="auto", size=(640, 480)):
    """Open a camera or return None; the camera picks its nearest supported size."""
    for api in BACKENDS[backend]:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # full fps at higher sizes
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            # Undo a manual exposure left by another app (e.g. the full game).
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_ON.get(api, 3))
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            return cap
        cap.release()
    return None


def list_cameras(backend="auto", size=(640, 480)):
    """[(index, width, height)] of cameras 0-4 that deliver a frame."""
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    found = []
    for i in range(5):
        cap = open_camera(i, backend, size)
        if cap is None:
            continue
        ok, frame = cap.read()
        if ok:
            found.append((i, frame.shape[1], frame.shape[0]))
        cap.release()
    return found


class Camera:
    """Keeps the newest frame; only the grab thread touches the camera (MSMF can deadlock)."""

    def __init__(self, index=0, backend="auto", size=(640, 480)):
        self.cap = open_camera(index, backend, size)
        if self.cap is None:
            raise RuntimeError(f"Can't open camera {index}. Close other apps that use it, check the "
                               "Windows camera privacy settings, or try --list.")
        self.latest = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self):
        while self.running:
            ok, frame = self.cap.read()            # blocks until the next frame
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest = (frame, time.perf_counter())   # frame, capture time
        self.cap.release()

    def read(self):
        """Newest (frame, capture time), or None if nothing new."""
        with self.lock:
            latest, self.latest = self.latest, None
        return latest

    def close(self):
        self.running = False
