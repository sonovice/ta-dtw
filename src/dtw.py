from numba import jit
import numpy as np
import librosa
from scipy.spatial.distance import cdist, cosine

import librosa.display
import matplotlib.pyplot as plt

from . import util


@jit
def calc_accu_matrix(n, m, trans_penalty=7):
    # Create multidimensional circular matrix of input matrix
    n_circular = util.chroma_to_circular(n)

    N = n.shape[1]
    M = m.shape[1]

    ####################################################################################################################
    # Cost Matrix                                                                                                      #
    ####################################################################################################################

    # Calculate distances using cosine similarity
    # C = 1 - np.matmul(n_circular.T, m)
    # print('Distances calculated')

    # Correct axes: (n, t, m) -> (n, m ,t)
    # C = np.swapaxes(C, 1, 2)

    C = np.empty((N, M, 12))
    for t in range(12):
        C[:, :, t] = 1 - np.matmul(n_circular[:, t, :].T, m)

    # Initialize multidimensional accumulated cost matrix
    D = np.ones(C.shape) * np.inf

    allowed_steps = np.array([
        [0, +1, 0],
        [+1, 0, 0],
        [+1, +1, 0],
        [0, +1, +1],
        [0, +1, -1],
        [+1, 0, +1],
        [+1, 0, -1],
        [+1, +1, +1],
        [+1, +1, -1]
    ])

    # Specify additional costs for the individual steps
    weights_add = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    weights_mul = np.array([
        2,
        2,
        1.3,
        2 * trans_penalty,
        2 * trans_penalty,
        2 * trans_penalty,
        2 * trans_penalty,
        2 * trans_penalty,
        2 * trans_penalty,
        1.3 * trans_penalty,
        1.3 * trans_penalty
    ])

    ####################################################################################################################
    # Accumulated Cost Matrix                                                                                          #
    ####################################################################################################################

    D[:, 0, :] = np.cumsum(C[:, 0, :], axis=0)
    D[0, :, :] = np.cumsum(C[0, :, :], axis=1)

    # Calculate D
    for n in range(1, N):
        for m in range(1, M):
            for t in range(0, 12):
                for cur_step_idx, cur_w_add, cur_w_mul in zip(range(allowed_steps.shape[0]), weights_add, weights_mul):
                    cur_D = D[
                        n - allowed_steps[cur_step_idx, 0],
                        m - allowed_steps[cur_step_idx, 1],
                        (t - allowed_steps[cur_step_idx, 2]) % 12
                    ]
                    cur_c = C[n, m, t]
                    cur_cost = cur_D + (cur_w_mul * cur_c + cur_w_add)

                    if cur_cost < D[n, m, t]:
                        D[n, m, t] = cur_cost

    return D


@jit
def backtracking(D):
    ####################################################################################################################
    # Backtracking                                                                                                     #
    ####################################################################################################################

    N = D.shape[0]
    M = D.shape[1]

    allowed_steps = np.array([
        [0, +1, 0],
        [+1, 0, 0],
        [+1, +1, 0],
        [0, +1, +1],
        [0, +1, -1],
        [+1, 0, +1],
        [+1, 0, -1],
        [+1, +1, +1],
        [+1, +1, -1]
    ])

    # Start warping
    path = []
    n = N - 1
    m = M - 1
    t = np.argmin(D[N - 1, M - 1, :])
    path.append((n, m, t))

    while n > 0 or m > 0:
        if n == 0:
            possible_t = np.array((
                t,
                (t - 1) % 12,
                (t + 1) % 12
            ))
            n, m, t = 0, m - 1, possible_t[np.argmin(D[0, m - 1, possible_t])]

        elif m == 0:
            possible_t = np.array((
                t,
                (t - 1) % 12,
                (t + 1) % 12
            ))
            n, m, t = n - 1, 0, possible_t[np.argmin(D[0, m - 1, possible_t])]

        else:
            possible_steps = np.array((
                n - allowed_steps[:, 0],
                m - allowed_steps[:, 1],
                np.mod(t - allowed_steps[:, 2], 12)
            ))
            n, m, t = possible_steps.T[np.argmin(D[tuple(possible_steps.tolist())])]

        path.append((n, m, t))

    return path


@jit
def dtw(n, m):
    D, wp = librosa.dtw(n, m)
    return wp


@jit
def ta_dtw(n, m, trans_penalty=7):
    D = calc_accu_matrix(n, m, trans_penalty=trans_penalty)
    return backtracking(D)
