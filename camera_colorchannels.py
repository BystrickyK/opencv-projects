import numpy as np
import cv2
from tools.filters import *
from tools.processing import resize, grab_frame
import matplotlib.pyplot as plt
import sys

channel = 0
cam = cv2.VideoCapture(0)  # Define camera
colors = {0: "Blue", 1: "Green", 2: "Red"}

img = resize(grab_frame(cam, channel), 0.9)
image = np.stack([img, img, img], axis=2)

threshold = 0

# %% Camera loop
while (True):
    # with timer_manager():

    # with timer_manager("FrameGrab"):
    img = resize(grab_frame(cam, channel), 0.9)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(']'):
        break
    elif key == ord('0'):
        channel = np.mod(channel+1, 3)
        print("ColorChannel " + str(channel))
    elif key == ord('1'):
        threshold = np.clip(threshold+8, 0, 255)
        print("Threshold " + str(threshold))
    elif key == ord('2'):
        threshold = np.clip(threshold-8, 0, 255)
        print("Threshold " + str(threshold))

    if threshold > 0:
        mask = img > threshold
        image[:, :, channel] = img * mask
    else:
        image[:, :, channel] = img
    zero_channels = [*range(3)]
    zero_channels.pop(channel)
    image[:, :, zero_channels] = 0



    cv2.rectangle(image, (0, 0), (20, 120), (0, 0, 0), -1)
    cv2.putText(image, colors[channel], (10, 30),
                fontFace=cv2.QT_FONT_NORMAL, fontScale=0.75, color=(255,255,255))

    cv2.imshow('ColorChannels', image)


cam.release()
cv2.destroyAllWindows()

