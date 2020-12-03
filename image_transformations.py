import cv2
import numpy as np


def process_frame(frame, target=(84, 84)):
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34 + 160, :160]  # crop image - TODO do it better
    frame = cv2.resize(frame, target, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*target, 1))
    return frame
