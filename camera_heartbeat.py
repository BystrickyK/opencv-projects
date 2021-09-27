import numpy as np
import cv2
from time import time
from tools.filters import *
from tools.processing import resize, grab_frame
from collections import deque
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy
import matplotlib.animation as animation
import sys
import numpy as np

def resample(x, y):
   Interpolator = interp1d(x, y)
   x_new = np.linspace(x[0], x[-1], len(x))
   y_new = Interpolator(x_new)
   dt = x_new[1] - x_new[0]
   return x_new, y_new, dt


channel = 2 # Red channel
cam = cv2.VideoCapture(0)  # Define camera

LENGTH = 256
signal = deque(np.zeros(LENGTH), maxlen=LENGTH)
timestamp = deque(np.zeros(LENGTH), maxlen=LENGTH)
heartbeat = deque(np.ones(LENGTH)*128, maxlen=LENGTH//4)
heartbeat_timestamp = deque(np.zeros(LENGTH), maxlen=LENGTH//4)

psd_previous = np.zeros(LENGTH)

# %% Camera loop
fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
ax = np.reshape(ax, -1)
line_time, = ax[0].plot([], [], '-r')
plt.grid()
line_freq, = ax[1].semilogy([], [], '.b-')
plt.grid()
point_freq_max, = ax[1].semilogy([0, 0], [0, 1e12], 'r')
line_time_filtered, = ax[2].plot([], [], '-b')
plt.grid()
heartbeat_plot, = ax[3].plot([], [], 'k-', linewidth=2)
plt.grid()
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Mean pixel value [-]")
ax[1].set_xlabel("Frequency [$\mathrm{min}^{-1}$]")
ax[1].set_ylabel("Power [-]")
ax[2].set_xlabel("Time [s]")
ax[2].set_ylabel("Filtered signal [-]")
ax[3].set_xlabel("Time [s]")
ax[3].set_ylabel("Heartbeat [$\mathrm{min}^{-1}$]")

def update_plot(line, x, y, update_axis='xy'):
    line.set_data(list(x), list(y))
    ystd = np.std(y)
    if update_axis in ('xy', 'x'):
        line.axes.set_xlim([np.min(x), np.max(x)])
    if update_axis in ('xy', 'y'):
        line.axes.set_ylim([np.mean(y)-3*ystd, np.mean(y)+3*ystd])
    return line,

def update_vline(line, x):
    line.set_xdata([x, x])
    return line,

plt.ion()

pause = False
program_start = time()
k = 0
while (True):
    start_time = time()
    k = k + 1

    image = resize(grab_frame(cam, channel), 0.25)
    image = image << 4


    # Resample to constant time step & run FFT
    if (np.mod(k, 2) == 0):
        time_, signal_, dt = resample(timestamp, signal)
        print(dt)
        signal_freq = np.fft.fft(signal_)
        freqs = np.fft.fftfreq(len(signal_), dt)
        freqs = 60 * freqs  # min^{-1}

        # Band-pass filter in frequency domain and transfer back
        idx_filter = np.any([np.abs(freqs)>160, np.abs(freqs)<40], axis=0)
        signal_freq_filtered = signal_freq
        signal_freq_filtered[idx_filter] = 1e-12
        signal_time_filtered = np.fft.ifft(signal_freq_filtered)

        signal_freq_psd = np.abs(signal_freq_filtered)**2 # Power spectrum density
        signal_freq_psd = (signal_freq_psd + psd_previous) / 2
        psd_previous = copy(signal_freq_psd)


        idx = [np.argmin(np.abs(freqs-40)), np.argmin(np.abs(freqs-180))]  # indices for range 60bpm to 150 bpm
        heartbeat_est_idx = np.argmax(signal_freq_psd[idx[0]:idx[1]])  # where PSD is highest in the 60-150 bpm band
        heartbeat_est = freqs[heartbeat_est_idx+idx[0]]
        heartbeat.append(heartbeat_est)
        heartbeat_timestamp.append(time_[-1])
        ax[1].set_title("{:.0f}".format(heartbeat_est))

        update_plot(line_freq, freqs[idx[0]-1:idx[1]+1], signal_freq_psd[idx[0]-1:idx[1]+1], update_axis=False)
        line_freq.axes.set_xlim(39, 141)
        line_freq.axes.set_ylim(1e1, 1e6)
        update_vline(point_freq_max, heartbeat_est)
        update_plot(heartbeat_plot, heartbeat_timestamp, heartbeat)
        update_plot(line_time_filtered, time_, signal_time_filtered)

    # Plot raw signal
    update_plot(line_time, timestamp, signal)
    plt.pause(0.001)

    image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
    cv2.imshow('ColorChannels', image)

    # Capture signal
    end_time = time()
    timestamp.append(end_time - program_start)
    signal.append(image.mean())


cam.release()
cv2.destroyAllWindows()

