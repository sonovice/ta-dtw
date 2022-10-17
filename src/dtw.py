from typing import Tuple, List

import librosa
import numpy as np
from numba import jit
from scipy.spatial.distance import cdist


@jit
def compute_cost_matrix(n: np.ndarray, m: np.ndarray, metric: str) -> np.ndarray:
    N = n.shape[1]
    M = m.shape[1]

    cost_matrix = np.empty((N, M, 12))
    for t in range(12):
        if metric == 'cosine':
            cost_matrix[:, :, t] = 1 - (np.roll(n.T, shift=-t, axis=1) @ m)
        else:
            cost_matrix[:, :, t] = cdist(np.roll(n.T, shift=-t, axis=1), m.T, metric=metric)

    return cost_matrix


@jit
def compute_accumulated_cost_matrix(cost_matrix: np.ndarray, transposition_penalty: float = 1.0) -> np.ndarray:
    # Initialize multidimensional accumulated cost matrix
    accumulated_cost_matrix = np.ones(cost_matrix.shape) * np.inf

    N, M, _ = cost_matrix.shape

    allowed_steps = np.array([
        [+0, +1, +0],
        [+1, +0, +0],
        [+1, +1, +0],
        [+0, +1, +1],
        [+0, +1, -1],
        [+1, +0, +1],
        [+1, +0, -1],
        [+1, +1, +1],
        [+1, +1, -1]
    ])

    weights_add = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    weights_mul = np.array([
        1,
        1,
        1,
        transposition_penalty,
        transposition_penalty,
        transposition_penalty,
        transposition_penalty,
        transposition_penalty,
        transposition_penalty,
        transposition_penalty,
        transposition_penalty
    ])

    accumulated_cost_matrix[:, 0, :] = np.cumsum(cost_matrix[:, 0, :], axis=0)
    accumulated_cost_matrix[0, :, :] = np.cumsum(cost_matrix[0, :, :], axis=1)

    for n in range(1, N):
        for m in range(1, M):
            for t in range(0, 12):
                for cur_step_idx, cur_w_add, cur_w_mul in zip(range(allowed_steps.shape[0]), weights_add, weights_mul):
                    cur_D = accumulated_cost_matrix[
                        n - allowed_steps[cur_step_idx, 0],
                        m - allowed_steps[cur_step_idx, 1],
                        (t - allowed_steps[cur_step_idx, 2]) % 12
                    ]
                    cur_cost_matrix_entry = cost_matrix[n, m, t]
                    cur_cost = cur_D + (cur_w_mul * cur_cost_matrix_entry + cur_w_add)

                    if cur_cost < accumulated_cost_matrix[n, m, t]:
                        accumulated_cost_matrix[n, m, t] = cur_cost

    return accumulated_cost_matrix


@jit
def backtracking(accumulated_cost_matrix: np.ndarray) -> List[Tuple[int, int, int]]:
    N, M, _ = accumulated_cost_matrix.shape

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
    t = np.argmin(accumulated_cost_matrix[N - 1, M - 1, :])
    path.append((n, m, t))

    while n > 0 or m > 0:
        if n == 0:
            possible_t = np.array((
                t,
                (t - 1) % 12,
                (t + 1) % 12
            ))
            n, m, t = 0, m - 1, possible_t[np.argmin(accumulated_cost_matrix[0, m - 1, possible_t])]

        elif m == 0:
            possible_t = np.array((
                t,
                (t - 1) % 12,
                (t + 1) % 12
            ))
            n, m, t = n - 1, 0, possible_t[np.argmin(accumulated_cost_matrix[n - 1, 0, possible_t])]

        else:
            possible_steps = np.array((
                n - allowed_steps[:, 0],
                m - allowed_steps[:, 1],
                np.mod(t - allowed_steps[:, 2], 12)
            ))
            n, m, t = possible_steps.T[np.argmin(accumulated_cost_matrix[tuple(possible_steps.tolist())])]

        path.append((n, m, t))

    return path


def dtw(n, m, metric):
    if metric == 'cosine':
        cost_matrix = 1 - (n.T @ m)
        _, wp = librosa.dtw(C=cost_matrix)
    else:
        _, wp = librosa.dtw(n, m, metric=metric)
    return wp


def ta_dtw(n, m, metric, transposition_penalty=1.0):
    cost_matrix = compute_cost_matrix(n, m, metric=metric)
    accumulated_cost_matrix = compute_accumulated_cost_matrix(cost_matrix, transposition_penalty=transposition_penalty)
    return backtracking(accumulated_cost_matrix)
