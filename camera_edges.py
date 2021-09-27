import numpy as np
import cv2
from time import time
from tools.filters import *
from tools.control import timer_manager
from tools.processing import resize, grab_frame
import matplotlib.pyplot as plt
import sys

def boost_edges(edges, boost):
    edges = edges.astype('uint16') * boost
    edges = np.clip(edges, 0, 255).astype('uint8')
    return edges


mode = 4
channel = 0

cam = cv2.VideoCapture(0)  # Define camera

blur_factor = 5
threshold = 0 # Larger number -> fewer edges

erode_str = 1
dilate_str = 1
erode_first = True

invert = False
pause = False
edge_boost = 1

# %% Camera loop
while (True):
    start_time = time()
    # with timer_manager():

    # with timer_manager("FrameGrab"):
    img = resize(grab_frame(cam, channel), 0.9)
    img = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)

    # with timer_manager("EdgeDetect"):
    img_edge_h, img_edge_h_mask = detect_edges(img, threshold, 'h')
    img_edge_h = np.clip(img_edge_h, 0, 255)
    img_edge_h = img_edge_h * img_edge_h_mask
    img_edge_v, img_edge_v_mask = detect_edges(img, threshold, 'v')
    img_edge_v = np.clip(img_edge_v, 0, 255)
    img_edge_v = img_edge_v * img_edge_v_mask

    # with timer_manager("Dilating"):
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

    img_edge_h = boost_edges(img_edge_h, edge_boost)
    img_edge_v = boost_edges(img_edge_v, edge_boost)

    # with timer_manager("Draw"):
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

    if not pause:
        if invert:
            cv2.imshow('EdgeDetector', np.clip(255-image, 0, 255))
        else:
            cv2.imshow('EdgeDetector', image)

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

    end_time = time()
    fps = 1/(end_time-start_time)
    cv2.rectangle(img, (0, 0), (150, 40), (0, 0, 0), -1)
    annotation_str = "FPS: {:.2f}".format(fps)
    cv2.putText(img, annotation_str, (10, 30),
                fontFace=cv2.QT_FONT_NORMAL, fontScale=0.75, color=(255,255,255))

    cv2.rectangle(img, (0, 0), (20, 20), (0, 0, 0), -1)
    cv2.putText(img, str(mode), (10, 30),
                fontFace=cv2.QT_FONT_NORMAL, fontScale=0.75, color=(255,255,255))

cam.release()
cv2.destroyAllWindows()


