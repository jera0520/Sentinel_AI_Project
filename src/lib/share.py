import numpy as np
from multiprocessing import shared_memory


def create_shared_frame(shape, dtype):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    return shm, arr

def read_from_shared(shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    out = arr.copy()  # 안전하게 복사
    shm.close()
    return out