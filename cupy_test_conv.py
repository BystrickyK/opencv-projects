import numpy as np
import scipy.signal as sps
import cupyx.scipy.signal as cps
import cupy as cp
from tools.control import timer_manager

npmat1 = np.random.randint(0, 10, (500, 500))
cpmat1 = cp.array(npmat1)

npker = np.random.randint(0, 10, (7, 7))
cpker = cp.array(npker)


with timer_manager("Numpy"):
    for i in range(100):
        npres = npmat1 @ npmat2

with timer_manager("Cupy"):
    for i in range(100):
        cpres = cpmat1 @ cpmat2
