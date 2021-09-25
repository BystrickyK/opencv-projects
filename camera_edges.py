import numpy as np
import cv2
from scipy import ndimage as sig
from scipy import signal as s
from tools.filters import *
from tools.control import timer_manager
import matplotlib.pyplot as plt
import matplotlib as mpl
import cupy as cp

def grab_frame(cam):
    ret, frame = cam.read()
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame[:, :, 2]
    return img


def resize(img, scale):
    w, h = img.shape[0], img.shape[1]
    w_after = int(w*scale)
    h_after = int(h*scale)
    img_after = cv2.resize(img, (h_after, w_after), interpolation=cv2.INTER_AREA)
    return img_after

cam = cv2.VideoCapture(0)  # Define camera

# %% Kernel definitions
# kernel_size = 3
blur_factor = 5
# kernel_horizontal = np.vstack([[np.ones(kernel_size) * i for i in np.linspace(1, -1, kernel_size)]])
# kernel_vertical = kernel_horizontal.T

# kernel_blur = np.zeros((blur_factor, blur_factor))
# X = np.abs((np.abs(np.linspace(-0.5, 0.5, blur_factor)) - 1))  # peak function (shape inverted V)
# for ix, x in enumerate(X):  # create parabolic kernel
#     for iy, y in enumerate(X):
#         kernel_blur[ix, iy] = x*x*y*y
# kernel_blur = kernel_blur / kernel_blur.sum() # normalize to sum==1

# kernel_horizontal = s.fftconvolve(kernel_horizontal, kernel_blur)
# kernel_vertical = s.fftconvolve(kernel_vertical, kernel_blur)

threshold = 5 # Larger number -> fewer edges

# %% Camera loop
while (True):
    with timer_manager():

        with timer_manager("FrameGrab"):
            img = resize(grab_frame(cam), 0.9)
            img = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)

        confun = 'space'
        with timer_manager("EdgeDetect"):
            img_edge_h, img_edge_h_mask = detect_edges(img, threshold, 'h')
            img_edge_h = img_edge_h * img_edge_h_mask
            img_edge_v, img_edge_v_mask = detect_edges(img, threshold, 'v')
            img_edge_v = img_edge_v * img_edge_v_mask

        dilate_str = 3
        kernel_dilate = np.ones((dilate_str, dilate_str))
        with timer_manager("Dilating"):
            img_edge_h = cv2.dilate(img_edge_h, kernel_dilate)
            img_edge_v = cv2.dilate(img_edge_v, kernel_dilate)


        with timer_manager("Draw"):
            image = np.stack([img_edge_h, np.zeros_like(img), img_edge_v], axis=2)
            # image = np.stack([edges, np.zeros_like(img), img], axis=2)
            # image = np.stack([edges, np.zeros_like(img), np.zeros_like(img)], axis=2)
            # im.set_data(img)
            # im_edges.set_data(edges)
            # im_edges.set_alpha(masks * 0.75)

            cv2.imshow('Edges', image)
            # cv2.imshow('Greyscale', img)

            # im_hor.set_data(img_edge_h)
            # im_hor.set_alpha(img_edge_h_mask * 0.75)
            #
            # im_vec.set_data(img_edge_v)
            # im_vec.set_alpha(img_edge_v_mask * 0.75)

    # cv2.imshow('frame', edges_horizontal)q;q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
plt.close()
