import numpy as np


def pitch_to_chroma(y):
    chroma = y.reshape(-1, 12, y.shape[1]).sum(0)
    norms = np.linalg.norm(chroma, axis=0)
    norms[norms == 0] = 1
    chroma /= norms
    return chroma


def chroma_to_circular(c):
    a, b = np.ogrid[0:12, 0:-12:-1]
    circular = a + b
    return c[circular]
