from typing import Tuple, List

import librosa
import numba as nb
import numpy as np
from scipy.spatial.distance import cdist

allowed_steps = np.array([
    # n,  m,  t
    [+0, +1, +0],
    [+1, +0, +0],
    [+1, +1, +0],

    [+0, +1, +1],
    [+1, +0, +1],
    [+1, +1, +1],

    [+0, +1, -1],
    [+1, +0, -1],
    [+1, +1, -1]
])


@nb.jit
def compute_cost_matrix(n: np.ndarray, m: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    N = n.shape[1]
    M = m.shape[1]

    cost_matrix = np.empty((N, M, 12))

    if metric == 'cosine':
        for t in range(12):
            cost_matrix[:, :, t] = 1 - (np.roll(n, shift=-t, axis=0).T @ m)
    else:
        for t in range(12):
            cost_matrix[:, :, t] = cdist(np.roll(n, shift=-t, axis=0).T, m.T, metric=metric)

    return cost_matrix


@nb.jit
def compute_accumulated_cost_matrix(cost_matrix: np.ndarray, transposition_penalty: float = 6.5) -> np.ndarray:
    # Initialize multidimensional accumulated cost matrix
    accumulated_cost_matrix = np.ones_like(cost_matrix) * np.inf

    N, M, _ = accumulated_cost_matrix.shape

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

                    accumulated_cost_matrix[n, m, t] = min(cur_cost, accumulated_cost_matrix[n, m, t])

    return accumulated_cost_matrix


@nb.jit
def backtracking(accumulated_cost_matrix: np.ndarray) -> List[Tuple[int, int, int]]:
    N, M, _ = accumulated_cost_matrix.shape

    # Start warping
    path = []
    n = N - 1
    m = M - 1
    t = np.argmin(accumulated_cost_matrix[n, m, :])
    path.append((n, m, t))  # starting point

    while n > 0 or m > 0:
        if n == 0:
            m -= 1
            possible_t = np.array((t, (t - 1) % 12, (t + 1) % 12))
            idx_t = np.argmin(accumulated_cost_matrix[n, m, possible_t])
            t = possible_t[idx_t]

        elif m == 0:
            n -= 1
            possible_t = np.array((t, (t - 1) % 12, (t + 1) % 12))
            idx_t = np.argmin(accumulated_cost_matrix[n, m, possible_t])
            t = possible_t[idx_t]

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
        cost_matrix = 1 - (n.T @ m)  # cosine distance
        _, wp = librosa.dtw(C=cost_matrix)
    else:
        cost_matrix = cdist(n.T, m.T, metric=metric)
        _, wp = librosa.dtw(C=cost_matrix)
    return wp


def ta_dtw(n, m, transposition_penalty, metric='euclidean'):
    cost_matrix = compute_cost_matrix(n, m, metric=metric)
    accumulated_cost_matrix = compute_accumulated_cost_matrix(cost_matrix, transposition_penalty=transposition_penalty)
    return backtracking(accumulated_cost_matrix)
