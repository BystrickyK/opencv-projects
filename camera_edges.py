import numpy as np
import cv2
from time import time
from tools.filters import *
from tools.control import timer_manager
from tools.processing import resize, grab_frame
import matplotlib.pyplot as plt
import sys

def boost_image(image, boost):
    image = image.astype('uint16') * boost
    image = np.clip(image, 0, 255).astype('uint8')
    return image

mode = 0  # Visualization mode
invert = False

channel = 0  # Color channel
colors = {0: "Blue", 1: "Green", 2: "Red"}

cam = cv2.VideoCapture(0)  # Choose camera device

blur_factor = 5  # side length of the smoothing kernel
threshold = 0 # Larger number -> fewer edges

erode_str = 1  # Size of the erosion kernel
dilate_str = 1  # Size of the dilation kernel
erode_first = True  # Controls the order of morph. operations

pause = False
edge_boost = 1  # Multiplies pixel values to boost weak edges

# %% Camera loop
while (True):
    start_time = time()

    # Capture an image
    img = resize(grab_frame(cam, channel), 0.9)

    # Blur the image to suppress noise and weak edges
    img = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)

    # Edge detection - horizontal
    img_edge_h, img_edge_h_mask = detect_edges(img, threshold, 'h')
    img_edge_h = np.clip(img_edge_h, 0, 255)
    img_edge_h = img_edge_h * img_edge_h_mask

    # Edge detection - vertical
    img_edge_v, img_edge_v_mask = detect_edges(img, threshold, 'v')
    img_edge_v = np.clip(img_edge_v, 0, 255)
    img_edge_v = img_edge_v * img_edge_v_mask

    # Morphological operations
    kernel_erode = np.ones((erode_str, erode_str))
    kernel_dilate = np.ones((dilate_str, dilate_str))
    if erode_first:
        img_edge_h = cv2.erode(img_edge_h, kernel_erode)
        img_edge_v = cv2.erode(img_edge_v, kernel_erode)
    img_edge_h = cv2.dilate(img_edge_h, kernel_dilate)
    img_edge_v = cv2.dilate(img_edge_v, kernel_dilate)
    if not erode_first:
        img_edge_h = cv2.erode(img_edge_h, kernel_erode)
        img_edge_v = cv2.erode(img_edge_v, kernel_erode)

    # Edge boosting
    img_edge_h = boost_image(img_edge_h, edge_boost)
    img_edge_v = boost_image(img_edge_v, edge_boost)

    # Select visualization mode
    if mode == 0:
        image = np.stack([img_edge_h, np.zeros_like(img), img_edge_v], axis=2)
    elif mode == 1:
        image = np.stack([img_edge_h, img_edge_v, np.zeros_like(img)], axis=2)
    elif mode == 2:
        image = np.stack([img_edge_h, img_edge_h, img_edge_v], axis=2)
    elif mode == 3:
        edges = np.max([img_edge_h, img_edge_v], axis=0)
        image = np.stack([edges, np.zeros_like(img), np.zeros_like(img)], axis=2)
    elif mode == 4:
        edges = np.max([img_edge_h, img_edge_v], axis=0)
        image = np.stack([np.zeros_like(img), edges, np.zeros_like(img)], axis=2)
    elif mode == 5:
        edges = np.max([img_edge_h, img_edge_v], axis=0)
        image = np.stack([np.zeros_like(img), np.zeros_like(img), edges], axis=2)
    elif mode == 6:
        edges = np.max([img_edge_h, img_edge_v], axis=0)
        image = np.stack([edges, img>>2, img>>2], axis=2)
    elif mode == 7:
        edges = np.max([img_edge_h, img_edge_v], axis=0)
        image = np.stack([img>>2, img>>2, edges], axis=2)
    elif mode == 8:
        edges = np.max([img_edge_h, img_edge_v], axis=0)
        image = np.stack([edges, edges, edges], axis=2)

    # Keyboard user interface
    key = cv2.waitKey(1) & 0xFF
    if key == ord(']'):
        break
    elif key == ord('['):
        mode = np.mod(mode+1, 9)
        print("ColorMode" + str(mode))
    elif key == ord('0'):
        channel = np.mod(channel+1, 3)
        print("ColorChannel" + str(channel))
    elif key == ord('1'):
        threshold = np.clip(threshold+1, 0, 20)
        print("Threshold " + str(threshold))
    elif key == ord('2'):
        threshold = np.clip(threshold-1, 0, 20)
        print("Threshold " + str(threshold))
    elif key == ord('3'):
        blur_factor = np.clip(blur_factor+2, 1, 31)
        print("BlurFactor " + str(blur_factor))
    elif key == ord('4'):
        blur_factor = np.clip(blur_factor-2, 1, 31)
        print("BlurFactor " + str(blur_factor))
    elif key == ord('5'):
        dilate_str = np.clip(dilate_str+1, 1, 15)
        print("DilateStr " + str(dilate_str))
    elif key == ord('6'):
        dilate_str = np.clip(dilate_str-1, 1, 15)
        print("DilateStr " + str(dilate_str))
    elif key == ord('7'):
        erode_str = np.clip(erode_str+1, 1, 15)
        print("ErodeStr " + str(erode_str))
    elif key == ord('8'):
        erode_str = np.clip(erode_str-1, 1, 15)
        print("ErodeStr " + str(erode_str))
    elif key == ord('9'):
        erode_first = not erode_first
        print("ErodeFirst " + str(erode_first))
    elif key == ord('i'):
        invert = not invert
        print("Invert " + str(invert))
    elif key == ord('/'):
        edge_boost = np.clip(edge_boost+1, 1, 16)
        print("EdgeBoost " + str(edge_boost))
    elif key == ord('='):
        edge_boost = np.clip(edge_boost-1, 1, 16)
        print("EdgeBoost " + str(edge_boost))
    elif key == ord('p'):
        pause = not pause
        print("Pause " + str(pause))

    # FPS counter
    end_time = time()
    fps = 1/(end_time-start_time)
    annotation_str = "FPS: {:.2f} | Mode: {} | Channel: {} |" \
                     " Threshold: {} | BlurFactor: {} | Erode/Dilate: {}/{} | ErodeFirst: {} | EdgeBoost: {}".format(
        fps, mode, colors[channel], threshold, blur_factor, erode_str, dilate_str, erode_first, edge_boost)
    cv2.putText(image, annotation_str, (10, 30),
                fontFace=cv2.QT_FONT_NORMAL, fontScale=0.6, color=(255,255,255))


    # Show the image
    if not pause:
        if invert:
            cv2.imshow('EdgeDetector', np.clip(255 - image, 0, 255))
        else:
            cv2.imshow('EdgeDetector', image)

cam.release()
cv2.destroyAllWindows()


