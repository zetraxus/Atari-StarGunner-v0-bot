import cv2
import numpy as np


def process_frame(frame, target):
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[36:36 + 150, 10:10 + 150]
    frame = cv2.resize(frame, target, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*target, 1))
    frame = frame.reshape((-1, target[0], target[1], 1))
    return frame
