import cv2


class Source:
    def __init__(self, cfg, video=None):
        cam = cfg["camera"]
        self.size = (cam["width"], cam["height"])
        self.is_file = video is not None
        self.mirror = cam["mirror"] and not self.is_file
        self.cap = cv2.VideoCapture(video if self.is_file else cam["index"])
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {video if self.is_file else 'camera %d' % cam['index']}")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.index = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
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
        if self.is_file:
            return
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        # manual exposure is 0.25 on DirectShow/MSMF and 1 on V4L2
        for value in (0.25, 1):
            if self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value):
                break

    def release(self):
        self.cap.release()
