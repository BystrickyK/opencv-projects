import signal
from contextlib import contextmanager
from time import time
import contextlib

@contextlib.contextmanager
def timer_manager(signature=None):
    start = time()
    yield
    end = time()
    if signature is None:
        print("Frame runtime: {:.2f}s".format(end-start))
    else:
        print("\t{} runtime: {:.2f}s".format(signature, end-start))

def timeout_handler(signum, frame):
    raise TimeoutError()

@contextmanager
def timeout(seconds):
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Timer decorator for simple profiling
def timer(fun):
    def timed_fun(*args, **kwargs):
        start_time = time()
        output = fun(*args, **kwargs)
        end_time = time()
        print("{}\t\t\t\tRuntime: {:0.1f} ms".format(fun.__name__, (end_time - start_time) * 10 ** 3))
        return output

    return timed_fun

