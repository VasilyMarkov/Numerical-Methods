import numpy as np
import time
from numba import jit

# Результаты замеров на 1е8: numpy array- 30 сек, python - 10 сек; с jit: numpy - 1 сек, python - 2.45 сек

@jit(nopython=True)
def hofstadter_conway1(n):
    seq = np.zeros(n, dtype=np.int_)
    seq[1] = seq[2] = 1
    for i in range(3, n):
        seq[i] = (seq[seq[i-1]]+seq[i-seq[i-1]])
    seq = np.asarray(seq)
    return seq

@jit(nopython=True)
def hofstadter_conway2(N):
    seq = []
    seq.append(0)
    seq.append(1)
    seq.append(1)
    for i in range(3, N):
        seq.append(seq[seq[i-1]]+seq[i-seq[i-1]])
    return seq

N = int(1e8)
start = time.time()
out = hofstadter_conway1(N)
end = time.time()
print(end-start)

start = time.time()
out = hofstadter_conway2(N)
end = time.time()
print(end-start)
