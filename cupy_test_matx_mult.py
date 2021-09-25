import numpy as np
import cupy as cp
from tools.control import timer_manager


npmat1 = np.random.randint(0, 10, (300, 300))
npmat2 = np.random.randint(0, 10, (300, 300))
cpmat1 = cp.array(npmat1)
cpmat2 = cp.array(npmat2)



with timer_manager("Numpy"):
    for i in range(100):
        npres = npmat1 @ npmat2
    
with timer_manager("Cupy"):
    for i in range(100):
        cpres = cpmat1 @ cpmat2
