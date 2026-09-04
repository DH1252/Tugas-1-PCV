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


