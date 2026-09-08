"""Color tracking: calibration, light correction, color masks, blobs and a Kalman tracker.

Per frame (Vision.__call__):
  1. prepare()        mirror, cut to 4:3, shrink to 320x240, blur
  2. LightCorrection  scale each channel so the walls match the calibration picture
  3. HSV + inRange    marker hue, saturation and brightness -> mask (0 or 255)
  4. exclusion        drop places that had the marker's color in the empty room
  5. clean()          opening removes specks, closing fills holes
  6. find_blobs()     connected components with area and palm center
  7. Tracker          Kalman filter, predicts through short dropouts

OpenCV HSV: H is 0..179 (degrees / 2) and wraps (red is near 0 and 179); S, V are 0..255.
OpenCV images are BGR.
"""
import math
from collections import namedtuple

import cv2
import numpy as np

W, H = 320, 240                                  # processing size
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
MIN_AREA = 40                                    # px; smaller blobs are noise
MAX_AREA = 0.6 * W * H                           # a close glove is big
USE_OPENCL = False                               # set by enable_opencl()

# Play area in processing px: the 16:9 crop minus a margin (see set_margin).
AREA_X, AREA_Y, AREA_W, AREA_H = 32.0, 48.0, 256.0, 144.0

# status: searching/tracking/coasting; pos: 0..1 on screen; vel: screen heights/s; t: capture time.
Saber = namedtuple("Saber", "status pos vel t")


