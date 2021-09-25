import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimg
# import cupyx.scipy.ndimage as cim
# import cupy as cp
from definitions import BITSHIFT
import cv2

shift_left_float = lambda arr: (arr*(2**BITSHIFT)).astype('int64')
shift_left_int = lambda arr: (arr.astype('int64') << BITSHIFT)
shift_right_float = lambda arr: (arr/(2**BITSHIFT)).astype('int64')
shift_right_int = lambda arr: (arr.astype('int64') >> BITSHIFT)

# def kernel_filter(img_, kernel_, confun='space'):
#     img = shift_left_int(img_)
#     kernel = shift_left_float(kernel_)
#     if confun=="fft":
#         result = signal.fftconvolve(img,
#                                        kernel,
#                                        mode='same')
#     else:
#         result = signal.fftconvolve(img, kernel, mode='same')
#     result = shift_right_int(result)
#     return result

def kernel_filter(img, kernel, confun='space'):
    # img = shift_left_int(img_)
    # kernel = shift_left_float(kernel)
    result = cv2.filter2D(img, -1, kernel)
    # result = shift_right_int(result)
    return result

# def detect_edges(img, kernel, threshold, confun='space'):
#     edges = kernel_filter(img, kernel, confun)
#     # edges = np.abs(edges)
#     mask = edges > (threshold * edges.mean())
#     return edges, mask

def detect_edges(img, threshold, dir='h'):
    if dir=='h':
        edges = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    elif dir=='v':
        edges = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    edges = cv2.convertScaleAbs(edges)
    mask = edges > (threshold * edges.mean())
    return edges, mask

# def detect_edges(img, min, max):
#     edges = cv2.Canny(img, min, max)
#     return edges
