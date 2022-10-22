import numba as nb
import numpy as np
from scipy.spatial.distance import cdist


def align(n, m, metric='euclidean', is_transposition_aware=False, transposition_penalty=1, force_t0=False):
    if is_transposition_aware:
        cost_matrix = compute_cost_matrix_ta(n, m, metric=metric)
        accumulated_cost_matrix = compute_accumulated_cost_matrix_ta(cost_matrix, transposition_penalty=transposition_penalty)
        alignment_path = backtracking_ta(accumulated_cost_matrix, force_t0=force_t0)
    else:
        cost_matrix = compute_cost_matrix(n, m, metric=metric)
        accumulated_cost_matrix = compute_accumulated_cost_matrix(cost_matrix)
        alignment_path = backtracking(accumulated_cost_matrix)

    return alignment_path


###############
# Canonical DTW
###############

allowed_steps = np.array([
    # n,  m
    [+0, +1],
    [+1, +0],
    [+1, +1],
])


def compute_cost_matrix(n: np.ndarray, m: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    if metric == 'cosine':
        cost_matrix = 1 - n.T @ m
    else:
        cost_matrix = cdist(n.T, m.T, metric=metric)

    return cost_matrix


@nb.jit(nopython=True)
def compute_accumulated_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    # Initialize multidimensional accumulated cost matrix
    accumulated_cost_matrix = np.ones_like(cost_matrix) * np.inf

    N, M = accumulated_cost_matrix.shape

    weights_add = np.array([0, 0, 0])
    weights_mul = np.array([1, 1, 1])

    accumulated_cost_matrix[:, 0] = np.cumsum(cost_matrix[:, 0])
    accumulated_cost_matrix[0, :] = np.cumsum(cost_matrix[0, :])

    for n in range(1, N):
        for m in range(1, M):
            for cur_step_idx, cur_w_add, cur_w_mul in zip(range(allowed_steps.shape[0]), weights_add, weights_mul):
                cur_D = accumulated_cost_matrix[
                    n - allowed_steps[cur_step_idx, 0],
                    m - allowed_steps[cur_step_idx, 1]
                ]
                cur_cost_matrix_entry = cost_matrix[n, m]
                cur_cost = cur_D + (cur_w_mul * cur_cost_matrix_entry + cur_w_add)

                accumulated_cost_matrix[n, m] = min(cur_cost, accumulated_cost_matrix[n, m])

    return accumulated_cost_matrix


@nb.jit(forceobj=True)
def backtracking(accumulated_cost_matrix: np.ndarray) -> np.ndarray:
    N, M = accumulated_cost_matrix.shape

    # Start warping
    path = []
    n = N - 1
    m = M - 1
    path.append((n, m))  # starting point

    while n > 0 and m > 0:
        possible_steps = np.array((
            n - allowed_steps[:, 0],
            m - allowed_steps[:, 1]
        ))
        n, m = possible_steps.T[np.argmin(accumulated_cost_matrix[tuple(possible_steps.tolist())])]

        path.append((n, m))

    return np.flip(np.asarray(path), axis=0)


#########################
# Transposition-Aware DTW
#########################

allowed_steps_ta = np.array([
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


def compute_cost_matrix_ta(n: np.ndarray, m: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
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


@nb.jit(nopython=True, cache=True)
def compute_accumulated_cost_matrix_ta(cost_matrix: np.ndarray, transposition_penalty: float = 6.5) -> np.ndarray:
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

    for t in range(0, 12):
        accumulated_cost_matrix[:, 0, t] = np.cumsum(cost_matrix[:, 0, t])
        accumulated_cost_matrix[0, :, t] = np.cumsum(cost_matrix[0, :, t])

    for n in range(1, N):
        for m in range(1, M):
            for t in range(0, 12):
                for cur_step_idx, cur_w_add, cur_w_mul in zip(range(allowed_steps_ta.shape[0]), weights_add, weights_mul):
                    cur_D = accumulated_cost_matrix[
                        n - allowed_steps_ta[cur_step_idx, 0],
                        m - allowed_steps_ta[cur_step_idx, 1],
                        (t - allowed_steps_ta[cur_step_idx, 2]) % 12
                    ]
                    cur_cost_matrix_entry = cost_matrix[n, m, t]
                    cur_cost = cur_D + (cur_w_mul * cur_cost_matrix_entry + cur_w_add)

                    accumulated_cost_matrix[n, m, t] = min(cur_cost, accumulated_cost_matrix[n, m, t])

    return accumulated_cost_matrix


@nb.jit(forceobj=True)
def backtracking_ta(accumulated_cost_matrix: np.ndarray, force_t0: bool = False) -> np.ndarray:
    N, M, _ = accumulated_cost_matrix.shape

    # Start warping
    path = []
    n = N - 1
    m = M - 1
    if force_t0:
        t = 0
    else:
        t = np.argmin(accumulated_cost_matrix[n, m, :])
    path.append((n, m, t))  # starting point

    while n > 0 or m > 0:
        if n == 0:
            m -= 1
            possible_t = (t, (t - 1) % 12, (t + 1) % 12)
            idx_t = int(np.argmin(accumulated_cost_matrix[n, m, possible_t]))
            t = possible_t[idx_t]

        elif m == 0:
            n -= 1
            possible_t = (t, (t - 1) % 12, (t + 1) % 12)
            idx_t = int(np.argmin(accumulated_cost_matrix[n, m, possible_t]))
            t = possible_t[idx_t]

        else:
            possible_steps = np.array((
                n - allowed_steps_ta[:, 0],
                m - allowed_steps_ta[:, 1],
                np.mod(t - allowed_steps_ta[:, 2], 12)
            ))
            n, m, t = possible_steps.T[np.argmin(accumulated_cost_matrix[tuple(possible_steps.tolist())])]

        path.append((n, m, t))

    return np.flip(np.asarray(path), axis=0)
